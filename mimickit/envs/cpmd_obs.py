"""Causal motion summary and context interactions (CPMD).

CPMD adds two blocks to the ADD state differential. Both are computed once per
control step, for the simulated and the reference character separately, and
both enter the same discriminator at the same time. There is no staging and no
second network.

Developed increments. The character path is turned into a stream of
fixed-anchor increments

    xi_t = [ C_0 (p_t - p_{t-1}),
             C_0 Log(R_t R_{t-1}^-1),
             DofLog(Q_{t-1}^-1 Q_t) ]                       in R^(6 + dof)

where C_0 is the per-episode fixed heading anchor taken from the reference
motion at phase 0 (never the current root heading), Log is the quaternion log
of the spatial (left) relative rotation, and DofLog is the joint-type aware log
of the relative joint rotations (hinge joints project onto their axis,
spherical joints use the full exp map). These are path increments, not
velocities: nothing is divided by the physics timestep.

Motion summary. The increments are summarized causally by

    m_t = rho m_{t-1} + xi_t                                in R^D, D = 6 + dof

with rho = exp(-dt / tau_mem) fixed by a physical memory time, so what the
summary remembers does not change with the control rate.

Context interactions. The summary is expanded pairwise on the strict upper
triangle i < j,

    c_t = { 1/2 m_{t,i} m_{t,j} }_{i<j}                     in R^(D(D-1)/2)

which is a pointwise function of m_t and carries no recursive state.

Why the pair goes through the discriminator rather than the difference of the
raw errors. Each side is summarized independently and the ADD difference is
taken after the expansion, so the interaction block reaching the discriminator
is

    c^ref - c^sim = 1/4 (dm sm^T + sm dm^T)|_{i<j},
    dm = m^ref - m^sim,        sm = m^ref + m^sim

i.e. products of the tracking error with the common-mode motion of the two
sides. A differential of raw errors is invariant to anything applied to both
sides at once, so no function of it can depend on how much absolute motion the
error occurred in; the same error looks identical mid-roll and while lying
still. Differencing after the per-side expansion exposes selected error-context
interactions -- specifically the pairwise products above -- and that is enough
to separate a rolling policy from one that reaches the same poses without
rolling. It is not a recovery of the full absolute motion context; the
discriminator still never sees m^ref or m^sim on their own.

Reference frame. Increments are anchored by C_0 and are differences of
consecutive frames, so a global yaw or translation of the whole scene changes
nothing. For WRAP motions the motion library unwraps the reference root
translation across the loop seam (it adds phase * wrap_delta in x and y), so
root translation increments stay continuous; root rotation and joint rotations
are not unwrapped, so their continuity at the seam is only as good as the
clip's first and last frames matching.

Cold start only: on reset both sides are zeroed and their previous root/joint
states are set to the reference reset state, so the first increment is produced
by the first physics step of the new episode and no state can leak across
episodes.
"""

import math
import torch

import util.torch_util as torch_util

@torch.jit.script
def calc_motion_anchor_quat_inv(ref_root_rot0):
    # type: (Tensor) -> Tensor
    """Inverse heading anchor C_0 from the reference root rotation at phase 0."""
    return torch_util.calc_heading_quat_inv(ref_root_rot0)

def calc_memory_decay(memory_seconds, timestep):
    """rho = exp(-dt / tau_mem): the discount is pinned to a physical memory
    time, so changing the control rate does not change what the summary
    remembers."""
    assert memory_seconds > 0.0
    assert timestep > 0.0
    return float(math.exp(-timestep / memory_seconds))

@torch.jit.script
def calc_root_increment(root_pos, prev_root_pos, root_rot, prev_root_rot, anchor_inv):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    """Anchored root position and rotation increments.

    quat_to_exp_map forces w >= 0, so the rotation increment is identical for
    q and -q, and consecutive control steps rotate far less than pi, so the
    per-step log is the minimal one.
    """
    dp = torch_util.quat_rotate(anchor_inv, root_pos - prev_root_pos)
    dq = torch_util.quat_mul(root_rot, torch_util.quat_conjugate(prev_root_rot))
    dw = torch_util.quat_rotate(anchor_inv, torch_util.quat_to_exp_map(dq))
    return dp, dw

def calc_interaction_dim(tangent_dim):
    """Number of strict upper triangle pairs i < j."""
    return tangent_dim * (tangent_dim - 1) // 2

class CPMDHistory():
    """Causal motion summary of one side (sim or ref) of the path.

    Holds only the recursive state (m and the previous pose), never a window of
    past frames: the whole history is summarized by the recursion above. The
    context interactions are derived from m on demand and carry no state.
    """

    def __init__(self, num_envs, kin_char_model, rho, device):
        assert 0.0 < rho <= 1.0

        self._kin_char_model = kin_char_model
        self._rho = float(rho)
        self._device = device

        self._dof_dim = kin_char_model.get_dof_size()
        self._tangent_dim = 6 + self._dof_dim
        self._interaction_dim = calc_interaction_dim(self._tangent_dim)

        dtype = torch.float32
        self._m = torch.zeros([num_envs, self._tangent_dim], dtype=dtype, device=device)

        idx = torch.triu_indices(self._tangent_dim, self._tangent_dim, offset=1, device=device)
        self._pair_i = idx[0]
        self._pair_j = idx[1]

        num_joints = kin_char_model.get_num_joints()
        self._prev_root_pos = torch.zeros([num_envs, 3], dtype=dtype, device=device)
        self._prev_root_rot = torch.zeros([num_envs, 4], dtype=dtype, device=device)
        self._prev_root_rot[..., 3] = 1.0
        self._prev_joint_rot = torch.zeros([num_envs, num_joints - 1, 4], dtype=dtype, device=device)
        self._prev_joint_rot[..., 3] = 1.0

        # diagnostic only: verifies that each side is pushed exactly once per
        # control step (checked at runtime, never used by the operator)
        self._push_count = torch.zeros([num_envs], dtype=torch.long, device=device)
        return

    def get_memory_decay(self):
        return self._rho

    def get_summary_dim(self):
        return self._tangent_dim

    def get_interaction_dim(self):
        return self._interaction_dim

    def get_obs_dim(self):
        return self._tangent_dim + self._interaction_dim

    def get_push_count(self):
        return self._push_count

    def reset(self, env_ids, root_pos, root_rot, joint_rot):
        """Cold start: zero the summary and anchor the previous pose to the
        reset state, so the first increment comes from the first step of the
        new episode."""
        self._m[env_ids] = 0.0
        self._prev_root_pos[env_ids] = root_pos
        self._prev_root_rot[env_ids] = root_rot
        self._prev_joint_rot[env_ids] = joint_rot
        self._push_count[env_ids] = 0
        return

    def calc_increment(self, root_pos, root_rot, joint_rot, anchor_inv):
        dp, dw = calc_root_increment(root_pos, self._prev_root_pos,
                                     root_rot, self._prev_root_rot, anchor_inv)
        # path increment, not a velocity: dt = 1
        dq = self._kin_char_model.compute_dof_vel(self._prev_joint_rot, joint_rot, 1.0)
        return torch.cat([dp, dw, dq], dim=-1)

    def push(self, root_pos, root_rot, joint_rot, anchor_inv):
        xi = self.calc_increment(root_pos, root_rot, joint_rot, anchor_inv)
        self._m = self._rho * self._m + xi

        self._prev_root_pos[:] = root_pos
        self._prev_root_rot[:] = root_rot
        self._prev_joint_rot[:] = joint_rot
        self._push_count += 1
        return

    def calc_motion_summary(self):
        return self._m

    def calc_context_interactions(self):
        """1/2 m_i m_j on the strict upper triangle."""
        m = self._m
        return 0.5 * m[:, self._pair_i] * m[:, self._pair_j]

    def extract(self):
        """[m, c]: the two blocks the discriminator receives, concatenated."""
        return torch.cat([self._m, self.calc_context_interactions()], dim=-1)
