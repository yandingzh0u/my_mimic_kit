"""Seven physical ERRs -> one anchored, componentwise monotone score.

CMNN activations follow Runje & Shankaranarayana (ICML 2023), equations 4-6.
This is an ADD adaptation, retaining the existing approximate SN constraint.
It adds a physical-error ordering prior, not a prior-free imitation reward.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm

from learning.dare_model import DAREModel
import util.torch_util as torch_util


MONOTONE_SCHEMA_VERSION = 1
HIDDEN_WIDTHS = (341, 512)
METRIC_NAMES = ("root_pos", "root_rot", "body_pos", "body_rot",
                "dof_vel", "root_vel", "root_ang_vel")
EXPECTED_GROUPS = (
    ("root_pos", tuple(range(0, 3))),
    ("root_rot", tuple(range(3, 9))),
    ("body_pos", tuple(range(93, 138))),
    ("body_rot", tuple(range(9, 93))),
    ("root_vel", tuple(range(138, 141))),
    ("root_ang_vel", tuple(range(141, 144))),
    ("dof_vel", tuple(range(144, 172))),
)


class PositiveWeight(nn.Module):
    def forward(self, weight):
        return weight.abs()


def positive_sn_linear(in_features, out_features, bias=True):
    layer = nn.Linear(in_features, out_features, bias=bias)
    # Applying abs AFTER signed-matrix SN would invalidate its spectral bound.
    parametrize.register_parametrization(layer, "weight", PositiveWeight())
    return spectral_norm(layer, n_power_iterations=1)


class MixedELU(nn.Module):
    """Fixed equal split: convex, concave and saturating monotone activations."""
    def forward(self, values):
        convex, concave, saturated = torch.tensor_split(values, 3, dim=-1)
        saturated = torch.where(
            saturated < 0, F.elu(saturated + 1) - 1,
            -F.elu(1 - saturated) + 1)
        return torch.cat((F.elu(convex), -F.elu(-concave), saturated), -1)


class ADDPairErrorMetrics(nn.Module):
    """Recover the evaluator's seven ERRs from the ORIGINAL paired ADD obs.

    Signed 6D rotation differences cannot recover geodesic angular errors.
    Replay therefore continues to retain both policy and reference states.
    World rotations use the same local-rotation/parent chain as the character's
    forward kinematics; body positions already exist in each observation.
    """
    def __init__(self, env):
        super().__init__()
        groups = tuple((name, tuple(indices))
                       for name, indices in env.get_disc_error_groups())
        if (tuple(env.get_disc_obs_space().shape) != (172,)
                or groups != EXPECTED_GROUPS
                or not getattr(env, "_global_obs", False)
                or env.get_num_disc_obs_steps() != 1):
            raise ValueError("DAREMonotone requires global, one-frame, 172D humanoid ADD observations")
        char = env._kin_char_model
        if char.get_num_joints() != 15 or char.get_dof_size() != 28:
            raise ValueError("DAREMonotone requires the audited 15-body/28-DOF humanoid")
        parents = tuple(int(char.get_parent_id(j)) for j in range(1, 15))
        if any(parent < 0 or parent >= j for j, parent in enumerate(parents, 1)):
            raise ValueError("Character rotations require a topologically ordered parent tree")
        self._parents = parents
        self.register_buffer("parent_ids", torch.tensor(parents, dtype=torch.long))
        self.register_buffer("local_rotations", char._local_rotation.detach().clone())

    def _body_rotations(self, obs):
        root = torch_util.tan_norm_to_quat(obs[..., 3:9])
        joints = torch_util.tan_norm_to_quat(obs[..., 9:93].reshape(-1, 14, 6))
        rotations = [root]
        for joint, parent in enumerate(self._parents, 1):
            local = self.local_rotations[joint].expand_as(joints[:, joint - 1])
            child = torch_util.quat_mul(local, joints[:, joint - 1])
            rotations.append(torch_util.quat_mul(rotations[parent], child))
        return torch.stack(rotations, dim=-2)

    def forward(self, policy, reference):
        if policy.ndim != 2 or policy.shape[-1] != 172 or reference.shape != policy.shape:
            raise ValueError("Physical ERR conversion requires matching [N,172] policy/reference observations")
        if policy.shape[0] == 0:
            return policy.new_zeros((0, 7))
        delta = reference - policy
        # Stack both states before the orientation chain to halve Python/kernel
        # dispatch overhead. No full FK translation pass is needed here.
        rotations = self._body_rotations(torch.cat((policy, reference), 0))
        policy_rot, reference_rot = rotations.split(policy.shape[0], 0)
        angles = torch_util.quat_diff_angle(policy_rot, reference_rot).abs()
        return torch.stack((
            delta[:, :3].norm(dim=-1), angles[:, 0],
            delta[:, 93:138].reshape(-1, 15, 3).norm(dim=-1).mean(-1),
            angles.mean(-1), delta[:, 144:172].abs().mean(-1),
            delta[:, 138:141].abs().mean(-1),
            delta[:, 141:144].abs().mean(-1)), -1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        for name in ("parent_ids", "local_rotations"):
            saved = state_dict.get(prefix + name)
            if saved is not None and not torch.equal(saved.cpu(), getattr(self, name).cpu()):
                error_msgs.append("Monotone physical-error character schema differs: " + prefix + name)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs)


class DAREMonotoneModel(DAREModel):
    """Same baseline actor/critic; one learned monotone seven-ERR discriminator."""
    def _build_disc(self, config, env):
        self._pair_metrics = ADDPairErrorMetrics(env)
        self._disc_energy = nn.Sequential(
            positive_sn_linear(7, HIDDEN_WIDTHS[0]), MixedELU(),
            positive_sn_linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]), MixedELU(),
            positive_sn_linear(HIDDEN_WIDTHS[1], 1, bias=False))
        self._anchor_logit = nn.Parameter(torch.zeros(()))
        self.register_buffer("_monotone_schema_version", torch.tensor(
            MONOTONE_SCHEMA_VERSION, dtype=torch.int64))

    def pair_to_error_metrics(self, policy, reference):
        return self._pair_metrics(policy, reference)

    def eval_disc_raw(self, disc_obs):
        if disc_obs.ndim != 2 or disc_obs.shape[-1] != 7:
            raise ValueError("DAREMonotone discriminator expects normalized seven nonnegative ERRs, not signed172 differences")
        anchor = disc_obs.new_zeros((1, 7))
        # One forward uses identical effective SN weights for sample and zero
        # even in train mode, where power-iteration buffers change per call.
        energy = self._disc_energy(torch.cat((disc_obs, anchor), dim=0))
        return self._anchor_logit - (energy[:-1] - energy[-1:])

    def get_disc_params(self):
        return list(self._disc_energy.parameters()) + [self._anchor_logit]

    def get_disc_logit_weights(self):
        return self._disc_energy[-1].weight.flatten()

    def get_disc_group_width(self):
        return self._anchor_logit.new_tensor(float(HIDDEN_WIDTHS[0]))

    def get_disc_group_total_width(self):
        return self.get_disc_group_width()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + "_monotone_schema_version")
        if (version is None or version.dtype != torch.int64 or version.numel() != 1
                or int(version.item()) != MONOTONE_SCHEMA_VERSION):
            error_msgs.append("Cannot load a different monotone physical-error schema")
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs)
