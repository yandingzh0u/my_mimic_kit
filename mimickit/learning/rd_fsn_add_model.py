import torch

import learning.add_model as add_model
import learning.nets.net_builder as net_builder


class GroupSeparableFullSNLayers(torch.nn.Module):
    """The successful a30 delayed-fusion discriminator backbone."""

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
                self._sn_linear(len(indices), self.group_width),
                activation()))

        self.total_width = self.group_width * num_groups
        trunk = []
        in_features = self.total_width
        for out_features in trunk_widths:
            trunk.extend((
                self._sn_linear(in_features, out_features),
                activation()))
            in_features = out_features
        self.trunk = torch.nn.Sequential(*trunk)
        self.out_features = in_features

    @staticmethod
    def _sn_linear(in_features, out_features):
        layer = torch.nn.Linear(in_features, out_features)
        torch.nn.init.zeros_(layer.bias)
        return torch.nn.utils.parametrizations.spectral_norm(layer)

    def forward(self, inputs):
        encoded = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(self, "group_indices_{}".format(group_id))
            encoded.append(encoder(torch.index_select(inputs, -1, indices)))
        return self.trunk(torch.cat(encoded, dim=-1))


class RDFSNADDModel(add_model.ADDModel):
    """a30 Full-SN score with a classification-only confidence scale."""

    def _build_disc(self, config, env):
        input_dict = {"disc_obs": env.get_disc_obs_space()}
        base_layers, _ = net_builder.build_net(
            config["disc_net"], input_dict, activation=self._activation)
        linears = [
            layer for layer in base_layers.modules()
            if isinstance(layer, torch.nn.Linear)
        ]
        if len(linears) < 2:
            raise ValueError("RD-FSN ADD requires a shared discriminator trunk")

        self._disc_layers = GroupSeparableFullSNLayers(
            groups=env.get_disc_error_groups(),
            first_width=linears[0].out_features,
            trunk_widths=[layer.out_features for layer in linears[1:]],
            activation=self._activation)

        output = torch.nn.Linear(self._disc_layers.out_features, 1)
        torch.nn.init.uniform_(output.weight, -1.0, 1.0)
        torch.nn.init.zeros_(output.bias)
        self._disc_logits = torch.nn.utils.parametrizations.spectral_norm(output)

        # exp(0) = 1 exactly recovers a30's classification scale initially.
        # This scalar is the magnitude of the effective classification head;
        # the inherited ADD logit regularizer therefore acts on scale^2.
        self._disc_log_class_scale = torch.nn.Parameter(torch.zeros(()))

    def eval_disc_score(self, diff):
        """Full-SN base score used by the policy reward."""
        return super().eval_disc(diff)

    def eval_disc_classification(self, diff):
        """Confidence-scaled logit used only by discriminator BCE."""
        return self.get_disc_class_scale() * self.eval_disc_score(diff)

    def get_disc_params(self):
        return super().get_disc_params() + [self._disc_log_class_scale]

    def get_disc_logit_weights(self):
        # For a scalar SN output, the effective direction has unit L2 norm.
        # Returning the scaled direction makes the official ADD logit penalty
        # exactly lambda_logit * class_scale^2.
        direction = super().get_disc_logit_weights()
        return self.get_disc_class_scale() * direction

    def get_disc_class_scale(self):
        return torch.exp(self._disc_log_class_scale)

    def get_disc_reward_lipschitz_bound(self):
        return self._disc_log_class_scale.new_tensor(1.0)

    def get_disc_class_lipschitz_bound(self):
        return self.get_disc_class_scale()

    def get_disc_group_width(self):
        return self._disc_log_class_scale.new_tensor(
            float(self._disc_layers.group_width))

    def get_disc_group_total_width(self):
        return self._disc_log_class_scale.new_tensor(
            float(self._disc_layers.total_width))
