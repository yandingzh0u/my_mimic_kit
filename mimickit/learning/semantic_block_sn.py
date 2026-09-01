import math

import torch
import torch.nn.functional as functional


class SemanticBlockEqualizedLinear(torch.nn.Module):
    """Dense linear layer with equalized input-block spectral gains.

    For input blocks ``W_g``, the effective weight is

        W_bar = concat(W_g / sigma(W_g)) / sigma(concat(...)).

    Thus the complete layer has spectral norm one and every effective input
    block has the same spectral norm.  The matrix remains fully dense: blocks
    share output neurons and are mixed before the activation.
    """

    def __init__(self, in_features, out_features, groups,
                 power_iterations=1, eps=1e-12, bias=True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.power_iterations = int(power_iterations)
        self.eps = float(eps)
        if self.power_iterations < 1:
            raise ValueError("power_iterations must be positive")

        group_indices = [tuple(int(i) for i in indices)
                         for _, indices in groups]
        flat_indices = [index for indices in group_indices
                        for index in indices]
        if sorted(flat_indices) != list(range(self.in_features)):
            raise ValueError(
                "Semantic groups must partition every input coordinate")
        self.group_names = tuple(name for name, _ in groups)
        self.num_groups = len(group_indices)
        if self.num_groups < 1:
            raise ValueError("At least one semantic group is required")

        self.weight = torch.nn.Parameter(torch.empty(
            self.out_features, self.in_features))
        self.bias = (torch.nn.Parameter(torch.empty(self.out_features))
                     if bias else None)
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            torch.nn.init.zeros_(self.bias)

        for group_id, indices in enumerate(group_indices):
            self.register_buffer(
                "group_indices_{}".format(group_id),
                torch.tensor(indices, dtype=torch.long))
            self.register_buffer(
                "block_left_{}".format(group_id),
                self._normalize(torch.randn(self.out_features)))
            self.register_buffer(
                "block_right_{}".format(group_id),
                self._normalize(torch.randn(len(indices))))

        self.register_buffer(
            "composite_left",
            self._normalize(torch.randn(self.out_features)))
        self.register_buffer(
            "composite_right",
            self._normalize(torch.randn(self.in_features)))
        self.register_buffer(
            "singular_vectors_initialized", torch.tensor(False))

    def _normalize(self, value):
        return value / torch.linalg.vector_norm(value).clamp_min(self.eps)

    def _block(self, group_id):
        indices = getattr(self, "group_indices_{}".format(group_id))
        return torch.index_select(self.weight, 1, indices)

    def _block_spectral_value(self, group_id):
        block = self._block(group_id)
        left = getattr(self, "block_left_{}".format(group_id)).detach()
        right = getattr(self, "block_right_{}".format(group_id)).detach()
        return torch.dot(left, block @ right).abs().clamp_min(self.eps)

    def block_spectral_values(self):
        return torch.stack([
            self._block_spectral_value(group_id)
            for group_id in range(self.num_groups)
        ])

    def block_normalized_weight(self):
        output = torch.empty_like(self.weight)
        for group_id in range(self.num_groups):
            indices = getattr(self, "group_indices_{}".format(group_id))
            block = self._block(group_id)
            block = block / self._block_spectral_value(group_id)
            output.index_copy_(1, indices, block)
        return output

    def composite_spectral_value(self):
        weight = self.block_normalized_weight()
        left = self.composite_left.detach()
        right = self.composite_right.detach()
        return torch.dot(left, weight @ right).abs().clamp_min(self.eps)

    def normalized_weight(self):
        return (self.block_normalized_weight()
                / self.composite_spectral_value())

    @torch.no_grad()
    def update_singular_vectors(self, iterations=None):
        iterations = (self.power_iterations if iterations is None
                      else int(iterations))
        if iterations < 1:
            raise ValueError("iterations must be positive")

        for group_id in range(self.num_groups):
            block = self._block(group_id)
            left_name = "block_left_{}".format(group_id)
            right_name = "block_right_{}".format(group_id)
            left = getattr(self, left_name)
            right = getattr(self, right_name)
            for _ in range(iterations):
                right = self._normalize(block.transpose(0, 1) @ left)
                left = self._normalize(block @ right)
            getattr(self, left_name).copy_(left)
            getattr(self, right_name).copy_(right)

        weight = self.block_normalized_weight()
        left = self.composite_left
        right = self.composite_right
        for _ in range(iterations):
            right = self._normalize(weight.transpose(0, 1) @ left)
            left = self._normalize(weight @ right)
        self.composite_left.copy_(left)
        self.composite_right.copy_(right)

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                "Expected {} input features, got {}".format(
                    self.in_features, inputs.shape[-1]))
        if not bool(self.singular_vectors_initialized.item()):
            # Seven independent estimates plus the composite estimate all
            # start from random vectors.  Warm them once; SGD updates are then
            # tracked by the ordinary one-step updates below.
            self.update_singular_vectors(iterations=200)
            self.singular_vectors_initialized.fill_(True)
        elif self.training:
            self.update_singular_vectors()
        return functional.linear(inputs, self.normalized_weight(), self.bias)

    def exact_effective_spectral_values(self):
        weight = self.normalized_weight()
        values = []
        for group_id in range(self.num_groups):
            indices = getattr(self, "group_indices_{}".format(group_id))
            block = torch.index_select(weight, 1, indices)
            values.append(torch.linalg.matrix_norm(block, ord=2))
        return torch.stack(values)

    def exact_composite_spectral_value(self):
        return torch.linalg.matrix_norm(self.normalized_weight(), ord=2)
