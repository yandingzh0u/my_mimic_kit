import contextlib
import math

import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder
from learning.sandwich_layers import SandwichFc, SandwichLin


class DirectSumSandwich(torch.nn.Module):
    """Semantic delayed fusion with one certified local map per error group."""
    def __init__(self, groups, first_width, trunk_widths):
        super().__init__()
        self.group_width = first_width // len(groups)
        if self.group_width < 1:
            raise ValueError("Discriminator width must cover every error group")

        self.encoders = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer(
                "group_indices_{}".format(group_id),
                torch.tensor(indices, dtype=torch.long))
            self.encoders.append(SandwichFc(len(indices), self.group_width))

        self.total_width = self.group_width * len(groups)
        trunk = []
        in_features = self.total_width
        for out_features in trunk_widths:
            trunk.append(SandwichFc(in_features, out_features))
            in_features = out_features
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_features

    def forward(self, inputs):
        encoded = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(self, "group_indices_{}".format(group_id))
            encoded.append(encoder(torch.index_select(inputs, -1, indices)))
        return self.trunk(torch.cat(encoded, dim=-1))


class GainSandwichADDModel(add_model.ADDModel):
    """Zero-anchored Sandwich shape with a regularized learned output gain."""
    def _build_disc(self, config, env):
        obs_dim = int(env.get_disc_obs_space().shape[0])
        self._disc_input_dim = obs_dim
        self._disc_input_scale = 1.0 / math.sqrt(obs_dim)

        # Consume the configured dense discriminator initialization exactly as
        # a30 did, then reuse its widths for the semantic direct-sum network.
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [layer for layer in base_layers.modules()
                   if isinstance(layer, torch.nn.Linear)]
        if len(linears) < 2:
            raise ValueError("Gain-Sandwich ADD requires a shared trunk")

        self._disc_layers = DirectSumSandwich(
            groups=env.get_disc_error_groups(),
            first_width=linears[0].out_features,
            trunk_widths=[layer.out_features for layer in linears[1:]])
        # The certified network learns only a scalar shape. Its constant
        # component is removed at zero differential, while the original ADD
        # output-head magnitude is represented explicitly by ``gain``.
        self._disc_shape = SandwichLin(
            self._disc_layers.out_features, 1, bias=False)
        self._disc_bias = torch.nn.Parameter(torch.zeros(1))
        self._disc_log_gain = torch.nn.Parameter(torch.zeros(()))

    def eval_disc(self, diff):
        scaled = diff * self._disc_input_scale
        shape = self._eval_disc_shape(scaled)
        zero = torch.zeros(
            (1, scaled.shape[-1]), dtype=scaled.dtype, device=scaled.device)
        zero_shape = self._eval_disc_shape(zero)
        centered_shape = shape - zero_shape
        return self._disc_bias + self.get_disc_gain() * centered_shape

    def _eval_disc_shape(self, scaled_diff):
        return self._disc_shape(self._disc_layers(scaled_diff))

    def get_disc_params(self):
        return (list(self._disc_layers.parameters())
                + list(self._disc_shape.parameters())
                + [self._disc_bias, self._disc_log_gain])

    def get_disc_logit_weights(self):
        direction = torch.flatten(self._disc_shape.effective_input_weight())
        return self.get_disc_gain() * direction

    def get_disc_gain(self):
        return torch.exp(self._disc_log_gain)

    def get_disc_bias(self):
        return self._disc_bias

    def get_disc_group_width(self):
        return self._disc_bias.new_tensor(float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_bias.new_tensor(float(self._disc_layers.total_width))

    def get_disc_input_dim(self):
        return self._disc_input_dim

    def get_disc_lipschitz_bound(self):
        return self.get_disc_gain() * self._disc_input_scale

    @contextlib.contextmanager
    def cached_disc_transforms(self):
        layers = [module for module in self.modules()
                  if isinstance(module, (SandwichFc, SandwichLin))]
        for layer in layers:
            layer.set_transform_cache(True)
        try:
            yield
        finally:
            for layer in layers:
                layer.set_transform_cache(False)
