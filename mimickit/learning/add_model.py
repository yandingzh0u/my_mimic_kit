import math

import torch

import learning.amp_model as amp_model
import learning.nets.net_builder as net_builder


class GroupSeparableDiscLayers(torch.nn.Module):
    """Equal-width semantic encoders over dimension-free group coordinates."""

    def __init__(self, groups, first_width, trunk_widths, activation):
        super().__init__()
        num_groups = len(groups)
        self.group_width = first_width // num_groups
        if self.group_width < 1:
            raise ValueError("Discriminator width must cover every error group")

        self.encoders = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer(
                "group_indices_{}".format(group_id),
                torch.tensor(indices, dtype=torch.long))
            self.encoders.append(torch.nn.Sequential(
                self._build_linear(len(indices), self.group_width),
                activation()))

        self.total_width = self.group_width * num_groups
        trunk = []
        in_size = self.total_width
        for out_size in trunk_widths:
            trunk.extend((self._build_linear(in_size, out_size), activation()))
            in_size = out_size
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_size

    @staticmethod
    def _build_linear(in_features, out_features):
        layer = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(layer.bias)
        torch.nn.utils.parametrizations.spectral_norm(layer)
        return layer

    def forward(self, diff):
        encoded = []
        group_radii = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(self, "group_indices_{}".format(group_id))
            group_diff = torch.index_select(diff, -1, indices)
            group_diff = group_diff / math.sqrt(len(indices))
            encoded.append(encoder(group_diff))
            group_radii.append(torch.linalg.vector_norm(
                group_diff, ord=2, dim=-1))
        features = self.trunk(torch.cat(encoded, dim=-1))
        return features, torch.stack(group_radii, dim=-1)


class ADDModel(amp_model.AMPModel):
    """PDR-ADD: a Pareto core plus a bounded signed interaction residual."""

    def _build_disc(self, config, env):
        # Build and discard the configured dense net to preserve the RNG order
        # of the validated group-separable a30 architecture.
        disc_obs_space = env.get_disc_obs_space()
        base_layers, _ = net_builder.build_net(
            config["disc_net"], {"disc_obs": disc_obs_space},
            activation=self._activation)
        linears = [
            layer for layer in base_layers
            if isinstance(layer, torch.nn.Linear)
        ]
        if len(linears) < 2:
            raise ValueError("PDR-ADD requires a shared discriminator trunk")

        self._disc_layers = GroupSeparableDiscLayers(
            groups=env.get_disc_error_groups(),
            first_width=linears[0].out_features,
            trunk_widths=[layer.out_features for layer in linears[1:]],
            activation=self._activation)

        self._disc_logits = torch.nn.Linear(
            self._disc_layers.out_features, 1, bias=True)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        torch.nn.utils.parametrizations.spectral_norm(
            self._disc_logits)

        self._num_disc_groups = len(env.get_disc_error_groups())
        self._disc_core_scale = 2.0
        self._disc_cert_scale = (
            1.0 + self._disc_core_scale * math.sqrt(self._num_disc_groups))

    def eval_disc_components(self, diff):
        shape = diff.shape[:-1]
        flat_diff = diff.reshape(-1, diff.shape[-1])
        features, group_radii = self._disc_layers(flat_diff)
        residual_logit = self._disc_logits(features).squeeze(-1)
        semantic_core = self._disc_core_scale * torch.sum(
            group_radii, dim=-1)
        logits = (residual_logit - semantic_core) / self._disc_cert_scale
        return (
            logits.reshape(*shape, 1),
            residual_logit.reshape(*shape),
            semantic_core.reshape(*shape),
            group_radii.reshape(*shape, self._num_disc_groups),
        )

    def eval_disc(self, diff):
        logits, _, _, _ = self.eval_disc_components(diff)
        return logits

    def get_disc_params(self):
        return (list(self._disc_layers.parameters())
                + list(self._disc_logits.parameters()))

    def get_disc_logit_weights(self):
        return torch.flatten(self._disc_logits.weight)

    def get_disc_group_width(self):
        return self._disc_logits.bias.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_logits.bias.new_tensor(
            float(self._disc_layers.total_width))

    def get_disc_theoretical_floor(self):
        return self._disc_logits.bias.new_tensor(
            1.0 / self._disc_cert_scale)
