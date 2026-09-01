import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
from learning.semantic_grouped_linear import SemanticGroupedLinear


class DifferentialDirectSumLayers(torch.nn.Module):
    """Semantic direct-sum first layer followed by one shared SN trunk."""

    def __init__(self, groups, first_width, trunk_widths, activation):
        super().__init__()
        num_groups = len(groups)
        self.group_width = first_width // num_groups
        if self.group_width < 1:
            raise ValueError("Discriminator width must cover every group")
        self.total_width = self.group_width * num_groups

        in_features = sum(len(indices) for _, indices in groups)
        self.semantic = SemanticGroupedLinear(
            in_features=in_features,
            groups=groups,
            out_features=self.group_width)
        self.semantic_activation = activation()

        trunk = []
        in_features = self.total_width
        for out_features in trunk_widths:
            layer = torch.nn.Linear(in_features, out_features, bias=True)
            torch.nn.init.zeros_(layer.bias)
            trunk.extend((
                torch.nn.utils.parametrizations.spectral_norm(layer),
                activation()))
            in_features = out_features
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_features

    def forward(self, inputs):
        grouped = self.semantic_activation(self.semantic(inputs))
        features = grouped.flatten(start_dim=-2)
        return self.trunk(features)


class FDADDModel(add_model.ADDModel):
    """Factorized Differential ADD with one grouped Full-SN front-end."""

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [layer for layer in base_layers.modules()
                   if isinstance(layer, torch.nn.Linear)]
        if len(linears) < 2:
            raise ValueError("FD-ADD requires a shared discriminator trunk")

        self._disc_layers = DifferentialDirectSumLayers(
            groups=env.get_disc_error_groups(),
            first_width=linears[0].out_features,
            trunk_widths=[layer.out_features for layer in linears[1:]],
            activation=self._activation)

        output = torch.nn.Linear(
            self._disc_layers.out_features, 1, bias=True)
        torch.nn.init.uniform_(output.weight, -1.0, 1.0)
        torch.nn.init.zeros_(output.bias)
        self._disc_logits = torch.nn.utils.parametrizations.spectral_norm(
            output)

    def get_disc_group_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.total_width))

    def get_disc_lipschitz_bound(self):
        return self._disc_logits.weight.new_tensor(1.0)
