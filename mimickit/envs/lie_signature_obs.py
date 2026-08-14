"""Fixed-anchor developed increments and discounted trajectory lifting (LieSig-STADD).

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

The increments are summarized causally by a discounted level-1 sum

    m_t = rho m_{t-1} + xi_t

with rho = exp(-dt / tau_mem) fixed by a physical memory time, so the meaning
of the operator does not change with control frequency, and then lifted to
level 2 by one of two blocks read off the strict upper triangle i < j:

    S_t = 1/2 m_t m_t^T                                    second_order "sym"
    A_t = rho^2 A_{t-1} + (rho/2)(m_{t-1} xi_t^T - xi_t m_{t-1}^T)     "area"

    Phi^(1)_t = m_t                               in R^D,       D = 6 + dof
    Phi^(2)_t = [m_t, vech_upper(S_t or A_t)]     in R^(D + D(D-1)/2)

Both blocks have the same width and cost the discriminator the same
parameters. S carries no recursive state of its own, being a pointwise
function of m; A is exactly antisymmetric by construction.

Why level 2 works, and why it is not about path order. The two sides are
lifted independently and the ADD differential is taken *after* the lift,
Phi^ref - Phi^sim, never before. For the symmetric block that differential is
exactly

    Delta_S = 1/4 (Delta_m Sigma_m^T + Sigma_m Delta_m^T),
    Delta_m = m^ref - m^sim,        Sigma_m = m^ref + m^sim

i.e. the level-1 error modulated by the common-mode motion of the two sides.
A level-1 discriminator only ever sees Delta_m, so the same tracking error
looks the same whether it happens in the middle of a vigorous roll or while
lying almost still; after the per-side quadratic lift it does not. Delta_S is
not a function of Delta_m, so this is information regained rather than level 1
re-encoded: plain ADD subtracts first and discards the absolute motion
context, lifting per side and subtracting afterwards brings it back.

A is a discounted Levy area and is the only block that carries traversal
order: it flips sign when the path is walked backwards, S does not. That was
the original hypothesis for why level 2 fixes Roll, and the experiments
falsified it. With everything else identical (same 767-wide differential,
same 1.31M discriminator parameters, seed 0, 1000 iterations) the order-blind
S rolls at least as well as A -- winding 0.930 vs 0.889, shortcut 5.9% vs
10.5%, and the same ~70% share of the discriminator's gradient energy --
while level 1 alone never rolls at all (100% shortcut). Path-order
information is therefore not necessary; the per-side quadratic lift is what
matters. A is kept behind second_order = "area" so that ablation stays
reproducible.

Cold start only: on reset both sides are zeroed and their previous root/joint
states are set to the reference reset state, so the first increment is
produced by the first physics step of the new episode and no state can leak
across episodes.

Two independent operators are streamed, one for the simulated character and
one for the reference.
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

# level-2 block variants on the same strict upper triangle: the symmetric
# quadratic lift (the method) and the antisymmetric Levy area (completed
# ablation, kept only so it stays reproducible)
SECOND_ORDER_SYM = "sym"
SECOND_ORDER_AREA = "area"
SECOND_ORDER_MODES = [SECOND_ORDER_SYM, SECOND_ORDER_AREA]

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
    """Causal discounted lift of one side (sim or ref) of the path.

    Holds only the recursive state (m, the previous pose, and A for the area
    ablation), never a window of past frames: the whole history is summarized
    by the recursion above.
    """

    def __init__(self, num_envs, kin_char_model, order, rho, device,
                 second_order=SECOND_ORDER_SYM):
        assert order in [1, 2]
        assert 0.0 < rho <= 1.0
        assert second_order in SECOND_ORDER_MODES

        self._kin_char_model = kin_char_model
        self._order = int(order)
        self._rho = float(rho)
        self._second_order = second_order
        self._device = device

        self._dof_dim = kin_char_model.get_dof_size()
        self._tangent_dim = 6 + self._dof_dim
        self._area_dim = calc_area_dim(self._tangent_dim) if (self._order == 2) else 0

        # the symmetric block is a pointwise function of m, so only the area
        # ablation needs recursive level-2 state
        self._stream_area = (self._area_dim > 0) and (self._second_order == SECOND_ORDER_AREA)

        dtype = torch.float32
        self._m = torch.zeros([num_envs, self._tangent_dim], dtype=dtype, device=device)
        area_state = self._area_dim if self._stream_area else 0
        self._area = torch.zeros([num_envs, area_state], dtype=dtype, device=device)

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

    def get_second_order(self):
        return self._second_order

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
        """Cold start: zero the lift and anchor the previous pose to the
        reset state, so the first increment comes from the first step of the
        new episode."""
        self._m[env_ids] = 0.0
        if (self._stream_area):
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
        if (self._stream_area):
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

    def calc_sym_block(self):
        """Level-2 symmetric lift: the strict upper triangle of 1/2 m m^T.
        Stateless per side; what the discriminator receives is its difference,
        which is the level-1 error modulated by the two sides' common-mode
        motion."""
        m = self._m
        return 0.5 * m[:, self._wedge_i] * m[:, self._wedge_j]

    def extract(self):
        """Phi_t: [m] at order 1, and at order 2 [m, vech_upper(1/2 m m^T)]
        or, for the area ablation, [m, vech_upper(A)]."""
        if (self._order == 1):
            return self._m
        if (self._stream_area):
            return torch.cat([self._m, self._area], dim=-1)
        return torch.cat([self._m, self.calc_sym_block()], dim=-1)
