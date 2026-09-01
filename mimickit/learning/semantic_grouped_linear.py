import math

import torch


class SemanticGroupedLinear(torch.nn.Module):
    """One ragged grouped Linear over a semantic direct-sum input.

    The layer owns one packed weight parameter and performs one batched
    operation.  Group blocks have independent weights and equal output width,
    while invalid padded coordinates are structurally masked.  Spectral
    normalization is applied independently to every block, so the induced
    direct-sum operator is 1-Lipschitz.
    """

    def __init__(self, in_features, groups, out_features,
                 n_power_iterations=1, eps=1e-12):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.n_power_iterations = int(n_power_iterations)
        self.eps = float(eps)
        if self.out_features < 1:
            raise ValueError("out_features must be positive")
        if self.n_power_iterations < 1:
            raise ValueError("n_power_iterations must be positive")

        names = []
        indices = []
        for name, group_indices in groups:
            names.append(str(name))
            indices.append(tuple(int(index) for index in group_indices))
        if not indices or any(len(group) == 0 for group in indices):
            raise ValueError("Semantic groups must be non-empty")
        flat = [index for group in indices for index in group]
        if sorted(flat) != list(range(self.in_features)):
            raise ValueError(
                "Semantic groups must partition every input coordinate")

        self.group_names = tuple(names)
        self.num_groups = len(indices)
        self.max_group_dim = max(len(group) for group in indices)

        padded_indices = torch.zeros(
            self.num_groups, self.max_group_dim, dtype=torch.long)
        valid_mask = torch.zeros(
            self.num_groups, self.max_group_dim, dtype=torch.bool)
        group_dims = torch.empty(self.num_groups, dtype=torch.long)
        for group_id, group in enumerate(indices):
            size = len(group)
            padded_indices[group_id, :size] = torch.tensor(group)
            valid_mask[group_id, :size] = True
            group_dims[group_id] = size
        self.register_buffer("group_indices", padded_indices)
        self.register_buffer("valid_mask", valid_mask)
        self.register_buffer("group_dims", group_dims)

        # Columns retain the original 172-D coordinate layout. Packing into
        # ragged group blocks is a view-by-index operation, not an input lift.
        self.weight = torch.nn.Parameter(torch.empty(
            self.out_features, self.in_features))
        self.bias = torch.nn.Parameter(torch.empty(
            self.num_groups, self.out_features))
        self.register_buffer("weight_u", torch.empty(
            self.num_groups, self.out_features))
        self.register_buffer("weight_v", torch.empty(
            self.num_groups, self.max_group_dim))
        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            self.weight.zero_()
            self.bias.zero_()
            for group_id in range(self.num_groups):
                size = int(self.group_dims[group_id])
                index = self.group_indices[group_id, :size]
                bound = 1.0 / math.sqrt(size)
                values = torch.empty(
                    self.out_features, size,
                    device=self.weight.device, dtype=self.weight.dtype)
                values.uniform_(-bound, bound)
                self.weight[:, index] = values
                self.bias[group_id].uniform_(-bound, bound)

            self.weight_u.normal_()
            self.weight_u.copy_(torch.nn.functional.normalize(
                self.weight_u, dim=-1, eps=self.eps))
            self.weight_v.normal_()
            self.weight_v.mul_(self.valid_mask)
            self.weight_v.copy_(torch.nn.functional.normalize(
                self.weight_v, dim=-1, eps=self.eps))
            # Start from an accurate estimate; subsequent training forwards
            # use the standard single power iteration.
            self._power_iteration(self._packed_weight(), 50)

    def _packed_weight(self):
        packed = self.weight[:, self.group_indices].permute(1, 0, 2)
        return packed * self.valid_mask.unsqueeze(1)

    @torch.no_grad()
    def _power_iteration(self, packed_weight, iterations):
        for _ in range(iterations):
            v = torch.bmm(
                packed_weight.transpose(1, 2),
                self.weight_u.unsqueeze(-1)).squeeze(-1)
            v = v * self.valid_mask
            self.weight_v.copy_(torch.nn.functional.normalize(
                v, dim=-1, eps=self.eps))
            u = torch.bmm(
                packed_weight,
                self.weight_v.unsqueeze(-1)).squeeze(-1)
            self.weight_u.copy_(torch.nn.functional.normalize(
                u, dim=-1, eps=self.eps))

    def normalized_weight(self):
        packed = self._packed_weight()
        if self.training:
            self._power_iteration(packed, self.n_power_iterations)
        # Cloning matches torch's spectral_norm parametrization: the buffers
        # can be updated by a later forward before this graph is consumed by
        # backward (ADD evaluates negative and zero inputs separately).
        weight_u = self.weight_u.clone(memory_format=torch.contiguous_format)
        weight_v = self.weight_v.clone(memory_format=torch.contiguous_format)
        sigma = torch.einsum(
            "gm,gmd,gd->g", weight_u, packed, weight_v)
        sigma = sigma.clamp_min(self.eps)
        return packed / sigma[:, None, None]

    def forward(self, inputs):
        if inputs.shape[-1] != self.in_features:
            raise ValueError(
                "Expected {} input features, got {}".format(
                    self.in_features, inputs.shape[-1]))
        grouped_inputs = inputs[..., self.group_indices]
        grouped_inputs = grouped_inputs * self.valid_mask
        output = torch.einsum(
            "...gd,gmd->...gm", grouped_inputs,
            self.normalized_weight())
        return output + self.bias

    def extra_repr(self):
        return ("in_features={}, groups={}, out_features={}, max_group_dim={}"
                .format(self.in_features, self.num_groups,
                        self.out_features, self.max_group_dim))
