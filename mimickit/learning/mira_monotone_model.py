"""Unpooled physical errors -> one anchored, componentwise monotone score.

CMNN activations follow Runje & Shankaranarayana (ICML 2023), equations 4-6.
All three linear layers use the non-residual diagonal construction from
Araujo et al. (ICLR 2023), Theorem 1 and Appendix C.3, instead of SN.
It adds a physical-error ordering prior, not a prior-free imitation reward.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import parametrize

from learning.add_model import ADDModel
from learning.diagonal_normalization import PositiveDiagonalRescaling
import util.torch_util as torch_util
from util.carry_geometry import CarryGeometry


MONOTONE_SCHEMA_VERSION = 4
SIGNED_WEIGHT_SCHEMA_VERSION = 5
FIXED_ANCHOR_SCHEMA_VERSION = 6
SIGNED_FIXED_ANCHOR_SCHEMA_VERSION = 7
HIDDEN_WIDTHS = (341, 512)
METRIC_NAMES = ("root_pos", "root_rot", "body_pos", "body_rot",
                "dof_vel", "root_vel", "root_ang_vel")
FEATURE_DIM = 64
FEATURE_SLICES = {
    "root_pos": slice(0, 1), "root_rot": slice(1, 2),
    "body_pos": slice(2, 16), "body_rot": slice(16, 30),
    "dof_vel": slice(30, 58), "root_vel": slice(58, 61),
    "root_ang_vel": slice(61, 64),
}
EXPECTED_GROUPS = (
    ("root_pos", tuple(range(0, 3))),
    ("root_rot", tuple(range(3, 9))),
    ("body_pos", tuple(range(93, 138))),
    ("body_rot", tuple(range(9, 93))),
    ("root_vel", tuple(range(138, 141))),
    ("root_ang_vel", tuple(range(141, 144))),
    ("dof_vel", tuple(range(144, 172))),
)
CARRY_FEATURE_SLICES = dict(
    FEATURE_SLICES, box_pos=slice(64, 65), box_rot=slice(65, 66),
    left_grip=slice(66, 67), right_grip=slice(67, 68))
CARRY_GROUPS = EXPECTED_GROUPS + (("box_pos", (172, 173, 174)), ("box_rot", (175, 176, 177, 178)))
CARRY_VELOCITY_FEATURE_SLICES = dict(CARRY_FEATURE_SLICES, box_vel=slice(68, 69))
CARRY_VELOCITY_GROUPS = CARRY_GROUPS + (("box_vel", (179, 180, 181)),)


def physical_feature_slices(env):
    if not getattr(env, "tracks_box", False):
        return FEATURE_SLICES
    if getattr(env, "tracks_box_velocity", False):
        return CARRY_VELOCITY_FEATURE_SLICES
    return CARRY_FEATURE_SLICES


def aggregate_error_metrics(features):
    """Reconstruct the original seven evaluator metrics, including Root.

    The Root's relative body position is identically zero. Body rotation in
    the evaluator includes the Root, whose error has its own input feature.
    Aggregation is for logging and scale statistics, never the network input.
    """
    if features.shape[-1] != FEATURE_DIM:
        raise ValueError("Physical error aggregation expects 64 features")
    return torch.stack((
        features[..., 0], features[..., 1],
        features[..., 2:16].sum(-1) / 15,
        (features[..., 1] + features[..., 16:30].sum(-1)) / 15,
        features[..., 30:58].mean(-1), features[..., 58:61].mean(-1),
        features[..., 61:64].mean(-1)), -1)


def _physical_rotation_quat(tan_norm):
    """Decode a valid 6D rotation without the axis singularity near pi.

    Select the best-conditioned of four matrix-to-quaternion candidates.
    This local conversion leaves ADD's observation/evaluator utilities intact.
    All candidate denominators are positive common factors and disappear
    when the selected quaternion is normalized.
    """
    matrix = torch_util.tan_norm_to_matrix(tan_norm)
    m00, m01, m02 = matrix[..., 0, :].unbind(-1)
    m10, m11, m12 = matrix[..., 1, :].unbind(-1)
    m20, m21, m22 = matrix[..., 2, :].unbind(-1)
    diagonals = torch.stack((1 + m00 + m11 + m22,
                             1 + m00 - m11 - m22,
                             1 - m00 + m11 - m22,
                             1 - m00 - m11 + m22), -1)
    candidates = torch.stack((
        torch.stack((diagonals[..., 0], m21 - m12, m02 - m20, m10 - m01), -1),
        torch.stack((m21 - m12, diagonals[..., 1], m10 + m01, m02 + m20), -1),
        torch.stack((m02 - m20, m10 + m01, diagonals[..., 2], m12 + m21), -1),
        torch.stack((m10 - m01, m20 + m02, m21 + m12, diagonals[..., 3]), -1)), -2)
    selected = diagonals.argmax(-1)
    indices = selected[..., None, None].expand(selected.shape + (1, 4))
    wxyz = F.normalize(candidates.gather(-2, indices).squeeze(-2), dim=-1)
    return torch.cat((wxyz[..., 1:], wxyz[..., :1]), -1)


class PositiveWeight(nn.Module):
    def forward(self, weight):
        return weight.abs()


def positive_diagonal_linear(in_features, out_features, bias=True):
    layer = nn.Linear(in_features, out_features, bias=bias)
    parametrize.register_parametrization(layer, "weight", PositiveWeight())
    normalization = PositiveDiagonalRescaling(
        in_features, device=layer.weight.device, dtype=layer.weight.dtype)
    normalization.initialize_from_weight(layer.weight)
    parametrize.register_parametrization(layer, "weight", normalization)
    return layer


class SignedMagnitudeDiagonalRescaling(PositiveDiagonalRescaling):
    """Apply the same diagonal scales to signed weights as to their magnitudes."""
    def forward(self, weight):
        # If B bounds |W| entrywise, then |W x| <= B |x|. In this case
        # B is exactly the production magnitude-normalized weight matrix.
        return weight.sign() * super().forward(weight.abs())


def signed_diagonal_linear(in_features, out_features, bias=True):
    layer = nn.Linear(in_features, out_features, bias=bias)
    with torch.no_grad():
        layer.weight.abs_()  # Match MIRA at initialization, then allow both signs.
    normalization = SignedMagnitudeDiagonalRescaling(
        in_features, device=layer.weight.device, dtype=layer.weight.dtype)
    normalization.initialize_from_weight(layer.weight)
    parametrize.register_parametrization(layer, "weight", normalization)
    return layer


class MixedELU(nn.Module):
    """Fixed equal split: convex, concave and saturating monotone activations."""
    def forward(self, values):
        convex, concave, saturated = torch.tensor_split(values, 3, dim=-1)
        saturated = torch.where(
            saturated < 0, F.elu(saturated + 1) - 1,
            -F.elu(1 - saturated) + 1)
        return torch.cat((F.elu(convex), -F.elu(-concave), saturated), -1)


class ADDPairErrorMetrics(nn.Module):
    """Recover 64 physical components from the ORIGINAL paired ADD obs.

    Signed 6D rotation differences cannot recover geodesic angular errors.
    Replay therefore continues to retain both policy and reference states.
    World rotations use the same local-rotation/parent chain as the character's
    forward kinematics; body positions already exist in each observation.
    """
    def __init__(self, env):
        super().__init__()
        self.tracks_box = getattr(env, "tracks_box", False)
        self.feature_slices = physical_feature_slices(env)
        self.tracks_box_velocity = "box_vel" in self.feature_slices
        self.grip_release_aware = self.tracks_box and getattr(env, "grip_release_aware", False)
        if self.tracks_box_velocity:
            expected_groups = CARRY_VELOCITY_GROUPS
        else:
            expected_groups = CARRY_GROUPS if self.tracks_box else EXPECTED_GROUPS
        if self.grip_release_aware:
            first = expected_groups[-1][1][-1] + 1
            expected_groups += (("active_grips", (first, first + 1)),)
        self.obs_dim = sum(len(indices) for _, indices in expected_groups)
        self.feature_dim = max(part.stop for part in self.feature_slices.values())
        self.metric_names = tuple(self.feature_slices)
        groups = tuple((name, tuple(indices))
                       for name, indices in env.get_disc_error_groups())
        if (tuple(env.get_disc_obs_space().shape) != (self.obs_dim,)
                or groups != expected_groups
                or not getattr(env, "_global_obs", False)
                or env.get_num_disc_obs_steps() != 1):
            raise ValueError(f"MIRAMonotone requires global, one-frame, {self.obs_dim}D paired ADD observations")
        char = env._kin_char_model
        if char.get_num_joints() != 15 or char.get_dof_size() != 28:
            raise ValueError("MIRAMonotone requires the audited 15-body/28-DOF humanoid")
        parents = tuple(int(char.get_parent_id(j)) for j in range(1, 15))
        if any(parent < 0 or parent >= j for j, parent in enumerate(parents, 1)):
            raise ValueError("Character rotations require a topologically ordered parent tree")
        self._parents = parents
        self.register_buffer("parent_ids", torch.tensor(parents, dtype=torch.long))
        self.register_buffer("local_rotations", char._local_rotation.detach().clone())
        if self.tracks_box:
            self._carry_geometry = CarryGeometry(env._box_size)
            names = char.get_body_names()
            self._hand_indices = [names.index(name) for name in ("left_hand", "right_hand")]

    def _body_rotations(self, obs):
        root = _physical_rotation_quat(obs[..., 3:9])
        joints = _physical_rotation_quat(obs[..., 9:93].reshape(-1, 14, 6))
        rotations = [root]
        for joint, parent in enumerate(self._parents, 1):
            local = self.local_rotations[joint].expand_as(joints[:, joint - 1])
            child = torch_util.quat_mul(local, joints[:, joint - 1])
            rotations.append(torch_util.quat_mul(rotations[parent], child))
        return torch.stack(rotations, dim=-2)

    def forward(self, policy, reference):
        if policy.ndim != 2 or policy.shape[-1] != self.obs_dim or reference.shape != policy.shape:
            raise ValueError(f"Expected matching [N,{self.obs_dim}] policy and reference observations")
        if policy.shape[0] == 0:
            return policy.new_zeros((0, self.feature_dim))
        delta = reference - policy
        # Stack both states before the orientation chain to halve Python/kernel
        # dispatch overhead. No full FK translation pass is needed here.
        rotations = self._body_rotations(torch.cat((policy, reference), 0))
        policy_rot, reference_rot = rotations.split(policy.shape[0], 0)
        angles = torch_util.quat_diff_angle(policy_rot, reference_rot).abs()
        human = torch.cat((
            delta[:, :3].norm(dim=-1, keepdim=True), angles[:, :1],
            delta[:, 93:138].reshape(-1, 15, 3)[:, 1:].norm(dim=-1),
            angles[:, 1:], delta[:, 144:172].abs(),
            delta[:, 138:141].abs(), delta[:, 141:144].abs()), -1)
        if not self.tracks_box:
            return human
        policy_hands = policy[:, 93:138].reshape(-1, 15, 3)[:, self._hand_indices] + policy[:, None, :3]
        reference_hands = reference[:, 93:138].reshape(-1, 15, 3)[:, self._hand_indices] + reference[:, None, :3]
        carry = self._carry_geometry(policy[:, 172:179], reference[:, 172:179], policy_hands, reference_hands)
        if self.grip_release_aware:
            carry = torch.cat((carry[:, :2], carry[:, 2:].masked_fill(reference[:, -2:] == 0, 0)), -1)
        if self.tracks_box_velocity:
            box_velocity_error = delta[:, 179:182].norm(dim=-1, keepdim=True)
            return torch.cat((human, carry, box_velocity_error), -1)
        return torch.cat((human, carry), -1)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        for name in ("parent_ids", "local_rotations"):
            saved = state_dict.get(prefix + name)
            if saved is not None and not torch.equal(saved.cpu(), getattr(self, name).cpu()):
                error_msgs.append("Monotone physical-error character schema differs: " + prefix + name)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs)


class MIRAMonotoneModel(ADDModel):
    """Same baseline actor/critic; one monotone physical-error discriminator."""
    def __init__(self, config, env):
        super().__init__(config, env)
        # Historical format marker; the separate monotone schema below now
        # rejects old SN weights instead of silently loading another model.
        self.register_buffer("_mira_schema_version", torch.tensor(2, dtype=torch.int64))

    def _build_disc(self, config, env):
        self.disc_signed_weights = config.get("disc_signed_weights", False)
        if not isinstance(self.disc_signed_weights, bool):
            raise ValueError("disc_signed_weights must be a boolean")
        self.disc_learn_anchor_logit = config.get("disc_learn_anchor_logit", True)
        if not isinstance(self.disc_learn_anchor_logit, bool):
            raise ValueError("disc_learn_anchor_logit must be a boolean")
        linear = signed_diagonal_linear if self.disc_signed_weights else positive_diagonal_linear
        self._pair_metrics = ADDPairErrorMetrics(env)
        self._disc_energy = nn.Sequential(
            linear(self._pair_metrics.feature_dim, HIDDEN_WIDTHS[0]), MixedELU(),
            linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]), MixedELU(),
            linear(HIDDEN_WIDTHS[1], 1, bias=False))
        if self.disc_learn_anchor_logit:
            self._anchor_logit = nn.Parameter(torch.zeros(()))
            self.disc_schema_version = (SIGNED_WEIGHT_SCHEMA_VERSION if self.disc_signed_weights
                                        else MONOTONE_SCHEMA_VERSION)
        else:
            self.register_parameter("_anchor_logit", None)
            self.disc_schema_version = (SIGNED_FIXED_ANCHOR_SCHEMA_VERSION if self.disc_signed_weights
                                        else FIXED_ANCHOR_SCHEMA_VERSION)
        self.register_buffer("_monotone_schema_version", torch.tensor(self.disc_schema_version, dtype=torch.int64))

    def pair_to_error_metrics(self, policy, reference):
        return self._pair_metrics(policy, reference)

    @staticmethod
    def aggregate_error_metrics(features):
        return aggregate_error_metrics(features)

    def eval_disc_raw(self, disc_obs):
        feature_dim = self._pair_metrics.feature_dim
        if disc_obs.ndim != 2 or disc_obs.shape[-1] != feature_dim:
            raise ValueError(f"MIRAMonotone expects {feature_dim} normalized physical errors")
        anchor = disc_obs.new_zeros((1, feature_dim))
        # Sample and anchor share the same deterministic effective weights.
        energy = self._disc_energy(torch.cat((disc_obs, anchor), dim=0))
        relative_energy = energy[:-1] - energy[-1:]
        if self._anchor_logit is None:
            return -relative_energy
        return self._anchor_logit - relative_energy

    def eval_disc(self, disc_obs):
        return self.eval_disc_raw(disc_obs)

    def get_disc_params(self):
        params = list(self._disc_energy.parameters())
        if self._anchor_logit is not None:
            params.append(self._anchor_logit)
        return params

    def get_disc_logit_weights(self):
        return self._disc_energy[-1].weight.flatten()

    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + "_monotone_schema_version")
        if (version is None or version.dtype != torch.int64 or version.numel() != 1
                or int(version.item()) != self.disc_schema_version):
            error_msgs.append("Cannot load a different monotone physical-error schema")
        old_marker = prefix + "_dare_schema_version"
        new_marker = prefix + "_mira_schema_version"
        if new_marker not in state_dict and old_marker in state_dict:
            state_dict[new_marker] = state_dict.pop(old_marker)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                     strict, missing_keys, unexpected_keys, error_msgs)
