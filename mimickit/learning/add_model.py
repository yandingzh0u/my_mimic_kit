import gymnasium.spaces as spaces
import numpy as np
import torch

import learning.amp_model as amp_model
import learning.nets.net_builder as net_builder
import util.torch_util as torch_util


class GroupSeparableDiscLayers(torch.nn.Module):
    def __init__(self, groups, first_width, trunk_widths, activation,
                 spectral_norm):
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
            layer = self._build_linear(
                len(indices), self.group_width, spectral_norm)
            self.encoders.append(torch.nn.Sequential(layer, activation()))

        self.total_width = self.group_width * num_groups
        trunk = []
        in_size = self.total_width
        for out_size in trunk_widths:
            trunk.append(self._build_linear(
                in_size, out_size, spectral_norm))
            trunk.append(activation())
            in_size = out_size
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_size
        return

    @staticmethod
    def _build_linear(in_features, out_features, spectral_norm):
        layer = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(layer.bias)
        if spectral_norm:
            torch.nn.utils.parametrizations.spectral_norm(layer)
        return layer

    def forward(self, x):
        encoded = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(self, "group_indices_{}".format(group_id))
            encoded.append(encoder(torch.index_select(x, -1, indices)))
        return self.trunk(torch.cat(encoded, dim=-1))


class ADDModel(amp_model.AMPModel):
    def __init__(self, config, env):
        self._disc_geometry = config.get("disc_geometry", "add")
        self._disc_spectral_norm = bool(
            config.get("disc_spectral_norm", False))
        self._disc_group_separable_frontend = bool(
            config.get("disc_group_separable_frontend", False))
        if self._disc_geometry not in {"add", "ref_concat"}:
            raise ValueError(
                "Unsupported ADD discriminator geometry: {}".format(
                    self._disc_geometry))
        super().__init__(config, env)
        return

    def eval_disc(self, diff, context=None):
        disc_input = self.build_disc_input(diff, context)
        return self.eval_disc_input(disc_input)

    def eval_disc_input(self, disc_input):
        h = self._disc_layers(disc_input)
        return self._disc_logits(h)

    def build_disc_input(self, diff, context=None):
        if self._disc_geometry == "add":
            return diff
        if self._disc_geometry == "ref_concat":
            if context is None:
                raise ValueError("RefConcat requires reference context")
            return torch.cat([diff, context], dim=-1)
        raise ValueError("Unsupported ADD discriminator geometry")

    def _build_disc(self, config, env):
        disc_obs_space = env.get_disc_obs_space()
        self._disc_obs_dim = int(np.prod(disc_obs_space.shape))

        input_dim = self._disc_obs_dim
        if self._disc_geometry == "ref_concat":
            input_dim *= 2
        disc_input_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(input_dim,),
            dtype=disc_obs_space.dtype)
        input_dict = {"disc_obs": disc_input_space}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        if self._disc_group_separable_frontend:
            if self._disc_geometry != "add":
                raise ValueError(
                    "Group-separable front-end requires direct ADD geometry")
            linears = [
                layer for layer in base_layers
                if isinstance(layer, torch.nn.Linear)
            ]
            if len(linears) < 2:
                raise ValueError(
                    "Group-separable front-end requires a shared trunk")
            self._disc_layers = GroupSeparableDiscLayers(
                groups=env.get_disc_error_groups(),
                first_width=linears[0].out_features,
                trunk_widths=[layer.out_features for layer in linears[1:]],
                activation=self._activation,
                spectral_norm=self._disc_spectral_norm)
            layers_out_size = self._disc_layers.out_features
        else:
            self._disc_layers = base_layers
            if self._disc_spectral_norm:
                for layer in self._disc_layers.modules():
                    if isinstance(layer, torch.nn.Linear):
                        torch.nn.utils.parametrizations.spectral_norm(layer)
            layers_out_size = torch_util.calc_layers_out_size(
                self._disc_layers)

        self._disc_logits = torch.nn.Linear(layers_out_size, 1)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        if self._disc_spectral_norm:
            torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)
        return

    def get_disc_group_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.total_width))
