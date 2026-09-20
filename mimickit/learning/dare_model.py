import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder


class GroupSeparableDiscLayers(torch.nn.Module):
    """Semantic group frontend for DARE's Full-SN critic.
    """

    def __init__(self, groups, first_width, trunk_widths, activation):
        super().__init__()
        num_groups = len(groups)
        self.group_width = first_width // num_groups
        if self.group_width < 1:
            raise ValueError("Discriminator width must cover every error group")
        self.total_width = self.group_width * num_groups

        self.encoders = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer(
                "group_indices_{}".format(group_id),
                torch.tensor(indices, dtype=torch.long))
            layer = self._build_linear(len(indices), self.group_width)
            self.encoders.append(torch.nn.Sequential(
                layer, activation()))

        trunk = []
        in_size = self.total_width
        for out_size in trunk_widths:
            trunk.append(self._build_linear(in_size, out_size))
            trunk.append(activation())
            in_size = out_size
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_size

    def _build_linear(self, in_features, out_features):
        layer = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(layer.bias)
        return torch.nn.utils.parametrizations.spectral_norm(layer)

    def forward(self, inputs):
        encoded = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(
                self, "group_indices_{}".format(group_id))
            group_input = torch.index_select(inputs, -1, indices)
            encoded.append(encoder(group_input))
        return self.trunk(torch.cat(encoded, dim=-1))


class DAREModel(add_model.ADDModel):
    """DARE critic with the original calibrated classifier output."""

    def __init__(self, config, env):
        super().__init__(config, env)
        self.register_buffer("_disc_logit_scale", torch.ones(()))
        self.register_buffer("_disc_logit_center", torch.zeros(()))
        self.register_buffer(
            "_disc_logit_calibrated", torch.zeros((), dtype=torch.bool))

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [layer for layer in base_layers.modules()
                   if isinstance(layer, torch.nn.Linear)]
        if len(linears) < 2:
            raise ValueError(
                "DARE requires a shared discriminator trunk")

        self._disc_group_embedding = bool(
            config.get("disc_group_embedding", True))
        if self._disc_group_embedding:
            self._disc_layers = GroupSeparableDiscLayers(
                groups=env.get_disc_error_groups(),
                first_width=linears[0].out_features,
                trunk_widths=[layer.out_features for layer in linears[1:]],
                activation=self._activation)
            disc_out = self._disc_layers.out_features
        else:
            # The ablation retains DARE's Full-SN critic and all subsequent
            # training machinery, but replaces the semantic direct sum by the
            # ordinary dense first layer used by a flat discriminator.
            self._disc_layers = base_layers
            for layer in linears:
                torch.nn.utils.parametrizations.spectral_norm(layer)
            disc_out = linears[-1].out_features

        self._disc_logits = torch.nn.Linear(disc_out, 1, bias=True)
        torch.nn.init.uniform_(self._disc_logits.weight, -1.0, 1.0)
        torch.nn.init.zeros_(self._disc_logits.bias)
        torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)

    def eval_disc_hidden(self, disc_obs):
        """Hidden discriminator representation h(x), used for diagnostics."""
        return self._disc_layers(disc_obs)

    def eval_disc_raw(self, disc_obs):
        return self._disc_logits(self._disc_layers(disc_obs))

    def get_disc_hidden_geometry(self):
        return "full_sn"

    def eval_disc(self, disc_obs):
        # One-shot affine logit standardization: the calibrated logit is
        # centered on the softplus transition region and unit-scaled by the
        # balanced calibration spread, i.e. z = (f - c_f) / s_f.
        return self._disc_logit_scale * (
            self.eval_disc_raw(disc_obs) - self._disc_logit_center)

    @torch.no_grad()
    def set_disc_logit_calibration(self, center, scale):
        """Set the affine calibration (center c_f, kappa = 1 / s_f)."""
        self._disc_logit_center.fill_(float(center))
        self._disc_logit_scale.fill_(float(scale))
        self._disc_logit_calibrated.fill_(True)

    @torch.no_grad()
    def set_disc_logit_scale(self, scale):
        # Legacy scale-only entry point (zero center), kept for compatibility.
        self._disc_logit_center.fill_(0.0)
        self._disc_logit_scale.fill_(float(scale))
        self._disc_logit_calibrated.fill_(True)

    def is_disc_logit_calibrated(self):
        return bool(self._disc_logit_calibrated.item())

    def get_disc_logit_scale(self):
        return self._disc_logit_scale.clone()

    def get_disc_logit_center(self):
        return self._disc_logit_center.clone()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Legacy v6 model/checkpoint state dicts have no calibration buffers.
        state_dict.setdefault(prefix + "_disc_logit_scale", torch.ones(()))
        state_dict.setdefault(prefix + "_disc_logit_center", torch.zeros(()))
        state_dict.setdefault(
            prefix + "_disc_logit_calibrated",
            torch.zeros((), dtype=torch.bool))
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def get_disc_group_width(self):
        if not self._disc_group_embedding:
            return self._disc_logits.weight.new_zeros(())
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        if not self._disc_group_embedding:
            return self._disc_logits.weight.new_zeros(())
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.total_width))

    def uses_disc_group_embedding(self):
        return self._disc_group_embedding
