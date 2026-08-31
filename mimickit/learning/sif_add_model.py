import math

import torch
import torch.nn.functional as functional
from torch.nn.utils.parametrizations import orthogonal

import learning.add_model as add_model
import learning.nets.net_builder as net_builder


def _orthogonal_linear(in_features, out_features):
    layer = torch.nn.Linear(in_features, out_features, bias=False)
    return orthogonal(
        layer, orthogonal_map="cayley", use_trivialization=False)


class IsometricFold(torch.nn.Module):
    """A trainable reflection fold with an orthogonal Jacobian a.e."""

    def __init__(self, width, rank=None):
        super().__init__()
        rank = width if rank is None else rank
        if rank > width:
            raise ValueError("Fold rank cannot exceed its input width")

        self.linear = _orthogonal_linear(width, rank)
        self.bias = torch.nn.Parameter(torch.zeros(rank))

    def forward(self, x, bias=None):
        weight = self.linear.weight
        bias = self.bias if bias is None else bias
        reflected = functional.linear(x, weight, bias)
        return x - 2.0 * functional.linear(
            torch.relu(reflected), weight.transpose(0, 1))

    def orthogonality_error(self):
        weight = self.linear.weight
        identity = torch.eye(
            weight.shape[0], device=weight.device, dtype=weight.dtype)
        error = weight @ weight.transpose(0, 1) - identity
        return torch.linalg.matrix_norm(error) / math.sqrt(weight.shape[0])


class UnitNormLinear(torch.nn.Module):
    """Scalar readout with direction and bias, but no trainable gain."""

    def __init__(self, in_features):
        super().__init__()
        self.direction = torch.nn.Parameter(torch.empty(in_features))
        self.bias = torch.nn.Parameter(torch.zeros(()))
        torch.nn.init.uniform_(self.direction, -1.0, 1.0)

    @property
    def weight(self):
        direction = functional.normalize(self.direction, dim=0)
        return direction.unsqueeze(0)

    def forward(self, x):
        return functional.linear(x, self.weight, self.bias.unsqueeze(0))


class SIFDiscLayers(torch.nn.Module):
    """Semantic slots followed by local and global isometric folds."""

    def __init__(self, groups, slot_width, global_rank):
        super().__init__()
        self.slot_width = slot_width
        self.total_width = slot_width * len(groups)
        self.out_features = self.total_width

        self._group_buffer_names = []
        for group_id, (_, indices) in enumerate(groups):
            if len(indices) > slot_width:
                raise ValueError(
                    "Semantic group dimension exceeds the slot width")
            name = "group_indices_{}".format(group_id)
            self.register_buffer(name, torch.tensor(indices, dtype=torch.long))
            self._group_buffer_names.append(name)

        self.local_fold = IsometricFold(slot_width)
        self.local_bias = torch.nn.Parameter(
            torch.zeros(len(groups), slot_width))
        self.global_fold = IsometricFold(self.total_width, global_rank)

    def encode_semantics(self, diff):
        slots = []
        for group_id, name in enumerate(self._group_buffer_names):
            indices = getattr(self, name)
            group = torch.index_select(diff, -1, indices)
            group = functional.pad(
                group, (0, self.slot_width - group.shape[-1]))
            group = self.local_fold(group, self.local_bias[group_id])
            slots.append(group)
        return torch.cat(slots, dim=-1)

    def forward(self, diff):
        return self.global_fold(self.encode_semantics(diff))

    def orthogonality_errors(self):
        return (
            self.local_fold.orthogonality_error(),
            self.global_fold.orthogonality_error(),
        )


class SIFADDModel(add_model.ADDModel):
    """ADD with semantic isometric folding as its discriminator."""

    def _build_disc(self, config, env):
        groups = env.get_disc_error_groups()
        base_layers, _ = net_builder.build_net(
            config["disc_net"], self._build_disc_input_dict(env),
            activation=self._activation)
        widths = [
            layer.out_features for layer in base_layers
            if isinstance(layer, torch.nn.Linear)
        ]
        if len(widths) != 2:
            raise ValueError("SIF-ADD requires a two-layer discriminator")

        first_width, global_rank = widths
        self._disc_layers = SIFDiscLayers(
            groups=groups,
            slot_width=first_width // len(groups),
            global_rank=global_rank)
        self._disc_logits = UnitNormLinear(self._disc_layers.out_features)

    def get_disc_logit_weights(self):
        return torch.flatten(self._disc_logits.weight)

    def get_disc_geometry_info(self):
        with torch.no_grad():
            local_error, global_error = (
                self._disc_layers.orthogonality_errors())
            return {
                "disc_local_orth_error": local_error,
                "disc_global_orth_error": global_error,
                "disc_head_raw_norm": self._disc_logits.direction.norm(),
                "disc_group_width": self._disc_logits.bias.new_tensor(
                    float(self._disc_layers.slot_width)),
                "disc_group_total_width": self._disc_logits.bias.new_tensor(
                    float(self._disc_layers.total_width)),
            }
