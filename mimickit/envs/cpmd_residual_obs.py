"""Paired causal motion memories for bilinear CPMD.

The operator keeps the difference and sum of the reference and simulated
motion summaries directly::

    h_t = rho * h_{t-1} + (xi_t^ref - xi_t^sim)
    s_t = rho * s_{t-1} + (xi_t^ref + xi_t^sim)

With identical decay and reset semantics these are exactly ``m_ref - m_sim``
and ``m_ref + m_sim`` from the established 767-D CPMD representation. Both
buffers are updated in place so environment ``info`` references remain valid.
"""

import math

import torch

import util.torch_util as torch_util


@torch.jit.script
def calc_motion_anchor_quat_inv(ref_root_rot0):
    # type: (Tensor) -> Tensor
    """Return the inverse phase-zero heading used as the fixed frame."""
    return torch_util.calc_heading_quat_inv(ref_root_rot0)


def calc_memory_decay(memory_seconds, timestep):
    """Compute ``rho = exp(-dt / tau)`` from a physical memory time."""
    assert memory_seconds > 0.0
    assert timestep > 0.0
    return float(math.exp(-timestep / memory_seconds))


@torch.jit.script
def calc_root_increment(root_pos, prev_root_pos, root_rot, prev_root_rot,
                        anchor_inv):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    """Compute fixed-anchor root translation and spatial rotation increments."""
    delta_pos = torch_util.quat_rotate(anchor_inv, root_pos - prev_root_pos)
    delta_rot = torch_util.quat_mul(
        root_rot, torch_util.quat_conjugate(prev_root_rot))
    delta_ang = torch_util.quat_rotate(
        anchor_inv, torch_util.quat_to_exp_map(delta_rot))
    return delta_pos, delta_ang


class CPMDErrorMemory:
    """Causal difference/common-motion summaries for one synchronized pair."""

    def __init__(self, num_envs, kin_char_model, rho, device):
        assert 0.0 < rho <= 1.0

        self._kin_char_model = kin_char_model
        self._rho = float(rho)
        self._device = device

        self._dof_dim = kin_char_model.get_dof_size()
        self._motion_dim = 6 + self._dof_dim

        dtype = torch.float32
        shape = [num_envs, self._motion_dim]
        self._delta_motion = torch.zeros(shape, dtype=dtype, device=device)
        self._sum_motion = torch.zeros(shape, dtype=dtype, device=device)

        num_joints = kin_char_model.get_num_joints()
        joint_shape = [num_envs, num_joints - 1, 4]

        self._prev_sim_root_pos = torch.zeros(
            [num_envs, 3], dtype=dtype, device=device)
        self._prev_sim_root_rot = torch.zeros(
            [num_envs, 4], dtype=dtype, device=device)
        self._prev_sim_root_rot[..., 3] = 1.0
        self._prev_sim_joint_rot = torch.zeros(
            joint_shape, dtype=dtype, device=device)
        self._prev_sim_joint_rot[..., 3] = 1.0

        self._prev_ref_root_pos = torch.zeros(
            [num_envs, 3], dtype=dtype, device=device)
        self._prev_ref_root_rot = torch.zeros(
            [num_envs, 4], dtype=dtype, device=device)
        self._prev_ref_root_rot[..., 3] = 1.0
        self._prev_ref_joint_rot = torch.zeros(
            joint_shape, dtype=dtype, device=device)
        self._prev_ref_joint_rot[..., 3] = 1.0

        self._push_count = torch.zeros(
            [num_envs], dtype=torch.long, device=device)
        return

    def get_memory_decay(self):
        return self._rho

    def get_history_dim(self):
        return self._motion_dim

    def get_motion_dim(self):
        return self._motion_dim

    def get_push_count(self):
        return self._push_count

    def get_history(self):
        """Compatibility alias for the difference memory."""
        return self._delta_motion

    def get_delta_motion(self):
        return self._delta_motion

    def get_sum_motion(self):
        return self._sum_motion

    def reset(self, env_ids, root_pos, root_rot, joint_rot):
        """Cold-start selected environments at their current reference pose."""
        self._delta_motion[env_ids] = 0.0
        self._sum_motion[env_ids] = 0.0

        self._prev_sim_root_pos[env_ids] = root_pos
        self._prev_sim_root_rot[env_ids] = root_rot
        self._prev_sim_joint_rot[env_ids] = joint_rot

        self._prev_ref_root_pos[env_ids] = root_pos
        self._prev_ref_root_rot[env_ids] = root_rot
        self._prev_ref_joint_rot[env_ids] = joint_rot

        self._push_count[env_ids] = 0
        return

    def _calc_increment(self, root_pos, root_rot, joint_rot, prev_root_pos,
                        prev_root_rot, prev_joint_rot, anchor_inv):
        delta_pos, delta_ang = calc_root_increment(
            root_pos, prev_root_pos, root_rot, prev_root_rot, anchor_inv)

        # Path increment, not velocity: dt=1 returns the joint tangent step.
        delta_dof = self._kin_char_model.compute_dof_vel(
            prev_joint_rot, joint_rot, 1.0)
        return torch.cat([delta_pos, delta_ang, delta_dof], dim=-1)

    def push(self, sim_root_pos, sim_root_rot, sim_joint_rot, ref_root_pos,
             ref_root_rot, ref_joint_rot, anchor_inv):
        """Append one synchronized simulated/reference control-step pair."""
        sim_increment = self._calc_increment(
            sim_root_pos, sim_root_rot, sim_joint_rot,
            self._prev_sim_root_pos, self._prev_sim_root_rot,
            self._prev_sim_joint_rot, anchor_inv)
        ref_increment = self._calc_increment(
            ref_root_pos, ref_root_rot, ref_joint_rot,
            self._prev_ref_root_pos, self._prev_ref_root_rot,
            self._prev_ref_joint_rot, anchor_inv)

        # Keep tensor identities stable for info-buffer consumers.
        self._delta_motion.mul_(self._rho).add_(
            ref_increment - sim_increment)
        self._sum_motion.mul_(self._rho).add_(
            ref_increment + sim_increment)

        self._prev_sim_root_pos.copy_(sim_root_pos)
        self._prev_sim_root_rot.copy_(sim_root_rot)
        self._prev_sim_joint_rot.copy_(sim_joint_rot)

        self._prev_ref_root_pos.copy_(ref_root_pos)
        self._prev_ref_root_rot.copy_(ref_root_rot)
        self._prev_ref_joint_rot.copy_(ref_joint_rot)

        self._push_count.add_(1)
        return


CPMDResidualHistory = CPMDErrorMemory
