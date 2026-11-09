import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder


DARE_MODEL_SCHEMA_VERSION = 2


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
    """Ordinary-SN DARE discriminator with unscaled network logits."""

    def __init__(self, config, env):
        super().__init__(config, env)
        self.register_buffer("_dare_schema_version", torch.tensor(
            DARE_MODEL_SCHEMA_VERSION, dtype=torch.int64))

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

    def eval_disc_raw(self, disc_obs):
        return self._disc_logits(self._disc_layers(disc_obs))

    def eval_disc(self, disc_obs):
        return self.eval_disc_raw(disc_obs)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Never silently reinterpret a historical scaled checkpoint, including
        # callers that request strict=False for otherwise partial model loads.
        legacy_keys = (prefix + "_disc_logit_scale",
                       prefix + "_disc_logit_calibrated")
        if any(key in state_dict for key in legacy_keys):
            error_msgs.append("Historical scaled DARE checkpoints cannot be "
                              "loaded into the unscaled discriminator")
        saved_version = state_dict.get(prefix + "_dare_schema_version")
        if (saved_version is None or saved_version.numel() != 1
                or saved_version.dtype != torch.int64
                or int(saved_version.item()) != DARE_MODEL_SCHEMA_VERSION):
            error_msgs.append("Cannot load a different DARE reward schema: "
                              + prefix + "_dare_schema_version")
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)

    def get_disc_group_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_logits.weight.new_tensor(
            float(self._disc_layers.total_width))
