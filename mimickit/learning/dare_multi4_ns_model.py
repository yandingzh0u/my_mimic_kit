"""Four tracking classifiers with fixed single-step NS hidden maintenance.

Only the scalar heads use spectral normalization.  Each tall hidden matrix
starts column-orthogonal and is maintained after discriminator optimizer steps;
the maintenance is not a loss term or a differentiable weight parametrization.
"""

import torch

from learning.dare_multi_model import DAREMultiModel


MULTI4_NS_SCHEMA_VERSION = 1
MULTI4_DISC_GROUPS = (
    ("root", ("root_pos", "root_rot", "root_vel", "root_ang_vel")),
    ("body_pos", ("body_pos",)),
    ("body_rot", ("body_rot",)),
    ("dof_vel", ("dof_vel",)),
)
HIDDEN_WIDTHS = (256, 512)


def build_multi4_disc_groups(error_groups, input_dim):
    """Collect actual ADD coordinates, including non-contiguous frame layouts."""
    source = {}
    for name, indices in error_groups:
        if name in source:
            raise ValueError("Duplicate discriminator error group: " + name)
        source[name] = tuple(indices)
    expected = {name for _, names in MULTI4_DISC_GROUPS for name in names}
    if set(source) != expected:
        raise ValueError("Four discriminators require the seven ADD error groups")
    flat = [index for indices in source.values() for index in indices]
    if (any(not indices for indices in source.values())
            or sorted(flat) != list(range(input_dim))):
        raise ValueError("Error groups must partition all discriminator coordinates")
    groups = tuple((name, tuple(index for part in parts for index in source[part]))
                   for name, parts in MULTI4_DISC_GROUPS)
    if any(len(indices) > HIDDEN_WIDTHS[0] for _, indices in groups):
        raise ValueError("Column-orthogonal hidden layers require each group input "
                         "dimension to be at most 256")
    return groups


@torch.no_grad()
def newton_schulz_column_step_(weight):
    """Apply exactly one fixed FP32 W <- 1.5W - 0.5W(W^T W) step in place."""
    if weight.ndim != 2 or weight.shape[0] < weight.shape[1]:
        raise ValueError("Column-orthogonal maintenance requires a tall matrix")
    # Keep the products in FP32 even if called inside an outer autocast region.
    with torch.autocast(device_type=weight.device.type, enabled=False):
        matrix = weight.float()
        gram = matrix.transpose(0, 1) @ matrix
        updated = 1.5 * matrix - 0.5 * (matrix @ gram)
        weight.copy_(updated)
    return weight


class IndependentNSDiscriminators(torch.nn.Module):
    def __init__(self, groups, activation):
        super().__init__()
        self.group_width = HIDDEN_WIDTHS[0]
        self.total_width = self.group_width * len(groups)
        self.register_buffer("ns_update_count", torch.zeros((), dtype=torch.int64))
        self.register_buffer("ns_schema_version", torch.tensor(
            MULTI4_NS_SCHEMA_VERSION, dtype=torch.int64))
        self.branches = torch.nn.ModuleList()
        for group_id, (_, indices) in enumerate(groups):
            self.register_buffer("group_indices_{}".format(group_id),
                                 torch.tensor(indices, dtype=torch.long))
            layers = []
            in_size = len(indices)
            for out_size in HIDDEN_WIDTHS:
                if out_size < in_size:
                    raise ValueError("Hidden weights must have at least as many rows "
                                     "as input columns")
                hidden = torch.nn.Linear(in_size, out_size)
                # QR is used only by this one-time dense random initialization.
                torch.nn.init.orthogonal_(hidden.weight)
                torch.nn.init.zeros_(hidden.bias)
                layers.extend((hidden, activation()))
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

    def hidden_weights(self):
        return tuple(layer.weight for branch in self.branches
                     for layer in (branch[0], branch[2]))

    @torch.no_grad()
    def maintain_hidden_orthogonality_(self):
        for weight in self.hidden_weights():
            newton_schulz_column_step_(weight)
        self.ns_update_count.add_(1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        saved_version = state_dict.get(prefix + "ns_schema_version")
        if (saved_version is None or saved_version.numel() != 1
                or int(saved_version.item()) != MULTI4_NS_SCHEMA_VERSION):
            error_msgs.append("Cannot load a different discriminator constraint schema: "
                              + prefix + "ns_schema_version")
        saved_count = state_dict.get(prefix + "ns_update_count")
        if (saved_count is None or saved_count.numel() != 1
                or saved_count.dtype != torch.int64 or int(saved_count.item()) < 0):
            error_msgs.append("Invalid or missing discriminator NS maintenance count: "
                              + prefix + "ns_update_count")
        for group_id in range(len(self.branches)):
            name = "group_indices_{}".format(group_id)
            saved = state_dict.get(prefix + name)
            if saved is None or not torch.equal(saved.cpu(), getattr(self, name).cpu()):
                error_msgs.append("Discriminator group schema differs: " + prefix + name)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys,
            unexpected_keys, error_msgs)


class DAREMulti4NSModel(DAREMultiModel):
    """Root15/BodyPos45/BodyRot84/DofVel28, each d->256->512->1."""

    def _build_disc(self, config, env):
        space = env.get_disc_obs_space()
        self._disc_error_groups = build_multi4_disc_groups(
            env.get_disc_error_groups(), int(space.shape[0]))
        # Architecture is a method constant, independent of the legacy disc_net
        # width setting.  Actor and critic construction remain unchanged.
        self._disc_branches = IndependentNSDiscriminators(
            self._disc_error_groups, self._activation)

    @torch.no_grad()
    def maintain_hidden_orthogonality_(self):
        self._disc_branches.maintain_hidden_orthogonality_()

    def get_ns_update_count(self):
        return self._disc_branches.ns_update_count.clone()
