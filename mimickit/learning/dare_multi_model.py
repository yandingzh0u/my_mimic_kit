"""Independent, ordinary-SN discriminators for three tracking objectives."""

import torch

import learning.dare_model as dare_model
import learning.nets.net_builder as net_builder


MULTI_DISC_SCHEMA_VERSION = 1
MULTI_DISC_GROUPS = (
    ("root_pose", ("root_pos", "root_rot")),
    ("body_pose", ("body_pos", "body_rot")),
    ("velocity", ("root_vel", "root_ang_vel", "dof_vel")),
)


def build_multi_disc_groups(error_groups, input_dim):
    """Merge the environment's actual indices, including multi-frame layouts."""
    source = {}
    for name, indices in error_groups:
        if name in source:
            raise ValueError("Duplicate discriminator error group: " + name)
        source[name] = tuple(indices)
    expected = {name for _, names in MULTI_DISC_GROUPS for name in names}
    if set(source) != expected:
        raise ValueError("Three discriminators require the seven ADD error groups")
    flat = [index for indices in source.values() for index in indices]
    if (any(not indices for indices in source.values())
            or sorted(flat) != list(range(input_dim))):
        raise ValueError("Error groups must partition all discriminator coordinates")
    return tuple((name, tuple(index for part in parts for index in source[part]))
                 for name, parts in MULTI_DISC_GROUPS)


class IndependentDiscriminators(torch.nn.Module):
    def __init__(self, groups, first_width, trunk_widths, activation):
        super().__init__()
        self.group_width = first_width // len(groups)
        if self.group_width < 1:
            raise ValueError("Discriminator width must cover all three branches")
        self.total_width = self.group_width * len(groups)
        self.branches = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer("group_indices_{}".format(group_id),
                                 torch.tensor(indices, dtype=torch.long))
            layers = []
            in_size = len(indices)
            for out_size in [self.group_width, *trunk_widths]:
                layers.extend((dare_model.GroupSeparableDiscLayers._build_linear(
                    in_size, out_size), activation()))
                in_size = out_size
            head = torch.nn.Linear(in_size, 1)
            torch.nn.init.uniform_(head.weight, -1., 1.)
            torch.nn.init.zeros_(head.bias)
            torch.nn.utils.parametrizations.spectral_norm(head)
            layers.append(head)
            self.branches.append(torch.nn.Sequential(*layers))

    def forward(self, inputs):
        return torch.cat([
            branch(torch.index_select(
                inputs, -1, getattr(self, "group_indices_{}".format(group_id))))
            for group_id, branch in enumerate(self.branches)], dim=-1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        for group_id in range(len(self.branches)):
            name = "group_indices_{}".format(group_id)
            saved = state_dict.get(prefix + name)
            if saved is not None and not torch.equal(
                    saved.cpu(), getattr(self, name).cpu()):
                error_msgs.append("Discriminator group schema differs: " + prefix + name)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)


class DAREMultiModel(dare_model.DAREModel):
    """Three disjoint SN MLPs with the baseline's single shared output scale."""

    def _build_disc(self, config, env):
        space = env.get_disc_obs_space()
        self._disc_error_groups = build_multi_disc_groups(
            env.get_disc_error_groups(), int(space.shape[0]))
        base_layers, _ = net_builder.build_net(
            config["disc_net"], {"disc_obs": space}, activation=self._activation)
        widths = [layer.out_features for layer in base_layers
                  if isinstance(layer, torch.nn.Linear)]
        if not widths:
            raise ValueError("Independent discriminators require a dense disc_net")
        self._disc_branches = IndependentDiscriminators(
            self._disc_error_groups, widths[0], widths[1:], self._activation)

    def get_disc_error_groups(self):
        return self._disc_error_groups

    def eval_disc_raw(self, disc_obs):
        return self._disc_branches(disc_obs)

    def get_disc_params(self):
        return list(self._disc_branches.parameters())

    def get_disc_logit_weights(self):
        return torch.cat([branch[-1].weight.flatten()
                          for branch in self._disc_branches.branches])

    def get_disc_group_width(self):
        return self._disc_logit_scale.new_tensor(float(self._disc_branches.group_width))

    def get_disc_group_total_width(self):
        return self._disc_logit_scale.new_tensor(float(self._disc_branches.total_width))
