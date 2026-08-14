"""Fixed-anchor developed increments and discounted signatures (LieSig-STADD).

One representation, one operator. The character path is turned into a stream
of fixed-anchor developed increments

    xi_t = [ C_0 (p_t - p_{t-1}),
             C_0 Log(R_t R_{t-1}^-1),
             DofLog(Q_{t-1}^-1 Q_t) ]                       in R^(6 + dof)

where C_0 is the per-episode fixed heading anchor taken from the reference
motion at phase 0 (never the current root heading), Log is the quaternion log
of the spatial (left) relative rotation, and DofLog is the joint-type aware
log of the relative joint rotations (hinge joints project onto their axis,
spherical joints use the full exp map). Increments are path increments, not
velocities: nothing is divided by the physics timestep.

The increments are summarized causally by a discounted level-2 signature

    m_t = rho m_{t-1} + xi_t
    A_t = rho^2 A_{t-1} + (rho/2) (m_{t-1} xi_t^T - xi_t m_{t-1}^T)

with rho = exp(-dt / tau_mem) fixed by a physical memory time, so the meaning
of the operator does not change with control frequency. A_t is exactly
antisymmetric by construction, so only its strict upper triangle is stored:

    Phi^(1)_t = m_t                                  in R^D,       D = 6 + dof
    Phi^(2)_t = [m_t, vech_upper(A_t)]               in R^(D + D(D-1)/2)

Why this catches shortcuts. m alone is a discounted endpoint: a path that
returns to the same place has small m regardless of how it got there. The
antisymmetric part A is a discounted Levy area, i.e. the signed area swept by
pairs of coordinates, which is exactly the leading order-independent
descriptor of *how* the path moved. A vanishes for any path confined to a
single direction in tangent space and flips sign when the path is traversed
in the opposite order, so "lie flat and translate" and "roll" are separated
even when they share endpoints.

Two independent signatures are streamed, one for the simulated character and
one for the reference. The differential is taken *after* the operator,
Phi^ref - Phi^sim, never before: the signature is nonlinear, so
A(ref) - A(sim) is not A(ref - sim).

Cold start only: on reset both sides are zeroed and their previous root/joint
states are set to the reference reset state, so the first increment is
produced by the first physics step of the new episode and no state can leak
across episodes.
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
    time, so changing the control rate does not change what the operator
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

def calc_area_dim(tangent_dim):
    return tangent_dim * (tangent_dim - 1) // 2

def unpack_area(packed, tangent_dim):
    """Rebuild the full antisymmetric A from its strict upper triangle."""
    n = packed.shape[0]
    idx = torch.triu_indices(tangent_dim, tangent_dim, offset=1, device=packed.device)
    area = torch.zeros([n, tangent_dim, tangent_dim], dtype=packed.dtype, device=packed.device)
    area[:, idx[0], idx[1]] = packed
    area = area - area.transpose(-1, -2)
    return area

class LieSigHistory():
    """Causal discounted signature of one side (sim or ref) of the path.

    Holds only the recursive state (m, A, previous pose), never a window of
    past frames: the whole history is summarized by the recursion above.
    """

    def __init__(self, num_envs, kin_char_model, order, rho, device):
        assert order in [1, 2]
        assert 0.0 < rho <= 1.0

        self._kin_char_model = kin_char_model
        self._order = int(order)
        self._rho = float(rho)
        self._device = device

        self._dof_dim = kin_char_model.get_dof_size()
        self._tangent_dim = 6 + self._dof_dim
        self._area_dim = calc_area_dim(self._tangent_dim) if (self._order == 2) else 0

        dtype = torch.float32
        self._m = torch.zeros([num_envs, self._tangent_dim], dtype=dtype, device=device)
        self._area = torch.zeros([num_envs, self._area_dim], dtype=dtype, device=device)

        idx = torch.triu_indices(self._tangent_dim, self._tangent_dim, offset=1, device=device)
        self._wedge_i = idx[0]
        self._wedge_j = idx[1]

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

    def get_order(self):
        return self._order

    def get_memory_decay(self):
        return self._rho

    def get_tangent_dim(self):
        return self._tangent_dim

    def get_area_dim(self):
        return self._area_dim

    def get_obs_dim(self):
        return self._tangent_dim + self._area_dim

    def get_push_count(self):
        return self._push_count

    def reset(self, env_ids, root_pos, root_rot, joint_rot):
        """Cold start: zero the signature and anchor the previous pose to the
        reset state, so the first increment comes from the first step of the
        new episode."""
        self._m[env_ids] = 0.0
        if (self._area_dim > 0):
            self._area[env_ids] = 0.0
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

        # A_t uses the OLD m_{t-1}; update the area before advancing m
        if (self._area_dim > 0):
            prev_m = self._m
            wedge = (prev_m[:, self._wedge_i] * xi[:, self._wedge_j]
                     - xi[:, self._wedge_i] * prev_m[:, self._wedge_j])
            self._area = self._rho * self._rho * self._area + 0.5 * self._rho * wedge

        self._m = self._rho * self._m + xi

        self._prev_root_pos[:] = root_pos
        self._prev_root_rot[:] = root_rot
        self._prev_joint_rot[:] = joint_rot
        self._push_count += 1
        return

    def extract(self):
        """Phi_t: [m] at order 1, [m, vech_upper(A)] at order 2."""
        if (self._order == 1):
            return self._m
        return torch.cat([self._m, self._area], dim=-1)
