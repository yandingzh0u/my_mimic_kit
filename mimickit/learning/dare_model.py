import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder


class GroupSeparableDiscLayers(torch.nn.Module):
    """Exact a30 group-separable Full-SN discriminator backbone."""

    def __init__(self, groups, first_width, trunk_widths, activation):
        super().__init__()
        num_groups = len(groups)
        self.group_width = first_width // num_groups
        if self.group_width < 1:
            raise ValueError(
                "Discriminator width must cover every error group")

        self.encoders = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer(
                "group_indices_{}".format(group_id),
                torch.tensor(indices, dtype=torch.long))
            layer = self._build_linear(len(indices), self.group_width)
            self.encoders.append(torch.nn.Sequential(layer, activation()))

        self.total_width = self.group_width * num_groups
        trunk = []
        in_size = self.total_width
        for out_size in trunk_widths:
            trunk.append(self._build_linear(in_size, out_size))
            trunk.append(activation())
            in_size = out_size
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_size

    @staticmethod
    def _build_linear(in_features, out_features):
        layer = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(layer.bias)
        torch.nn.utils.parametrizations.spectral_norm(layer)
        return layer

    def forward(self, inputs):
        encoded = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(
                self, "group_indices_{}".format(group_id))
            group_input = torch.index_select(inputs, -1, indices)
            encoded.append(encoder(group_input))
        return self.trunk(torch.cat(encoded, dim=-1))


class DAREModel(add_model.ADDModel):
    """Exact a30 model path used as the DARE restoration baseline."""

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [layer for layer in base_layers
                   if isinstance(layer, torch.nn.Linear)]
        if len(linears) < 2:
            raise ValueError(
                "Group-separable front-end requires a shared trunk")

        self._disc_layers = GroupSeparableDiscLayers(
            groups=env.get_disc_error_groups(),
            first_width=linears[0].out_features,
            trunk_widths=[layer.out_features for layer in linears[1:]],
            activation=self._activation)

        self._disc_logits = torch.nn.Linear(
            self._disc_layers.out_features, 1, bias=True)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)

    def get_disc_group_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.total_width))
