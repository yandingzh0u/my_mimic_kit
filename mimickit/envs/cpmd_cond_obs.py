"""Motion-error memory and phase-consistent reference context for CPMD."""

import math

import torch

import anim.motion as motion
import util.torch_util as torch_util


def calc_memory_decay(memory_seconds, timestep):
    assert memory_seconds > 0.0
    assert timestep > 0.0
    return float(math.exp(-timestep / memory_seconds))


@torch.jit.script
def calc_motion_anchor_quat_inv(ref_root_rot0):
    # type: (Tensor) -> Tensor
    return torch_util.calc_heading_quat_inv(ref_root_rot0)


@torch.jit.script
def calc_root_increment(root_pos, prev_root_pos, root_rot, prev_root_rot,
                        anchor_inv):
    # type: (Tensor, Tensor, Tensor, Tensor, Tensor) -> Tuple[Tensor, Tensor]
    delta_pos = torch_util.quat_rotate(anchor_inv, root_pos - prev_root_pos)
    delta_rot = torch_util.quat_mul(
        root_rot, torch_util.quat_conjugate(prev_root_rot))
    delta_ang = torch_util.quat_rotate(
        anchor_inv, torch_util.quat_to_exp_map(delta_rot))
    return delta_pos, delta_ang


def calc_motion_increment(kin_char_model, root_pos, root_rot, joint_rot,
                          prev_root_pos, prev_root_rot, prev_joint_rot,
                          anchor_inv):
    delta_pos, delta_ang = calc_root_increment(
        root_pos, prev_root_pos, root_rot, prev_root_rot, anchor_inv)
    # This is a path increment, not a velocity.  dt=1 returns the joint
    # tangent displacement without introducing a 30x scale factor.
    delta_dof = kin_char_model.compute_dof_vel(
        prev_joint_rot, joint_rot, 1.0)
    return torch.cat([delta_pos, delta_ang, delta_dof], dim=-1)


class PhaseReferenceContext:
    """Lookup table for deterministic discounted reference motion context.

    Each grid value approximates the infinite discounted back-sum

        c(t) = sum_{k>=0} rho^k xi_ref(t - k * dt).

    This definition works for arbitrary clip length/control-rate ratios.  WRAP
    clips use periodic interpolation; non-WRAP clips naturally truncate at the
    beginning because MotionLib clamps negative time to its first frame.
    """

    def __init__(self, motion_lib, kin_char_model, rho, timestep, grid_size,
                 tail_tolerance, device):
        assert 0.0 < rho < 1.0
        assert timestep > 0.0
        assert grid_size >= 2
        assert 0.0 < tail_tolerance < 1.0

        self._motion_lib = motion_lib
        self._kin_char_model = kin_char_model
        self._rho = float(rho)
        self._timestep = float(timestep)
        self._grid_size = int(grid_size)
        self._device = device
        self._motion_dim = 6 + kin_char_model.get_dof_size()
        self._tail_steps = int(math.ceil(
            math.log(tail_tolerance) / math.log(rho)))
        self._tail_steps = max(1, self._tail_steps)
        self._table = self._build_table()
        return

    def get_grid_size(self):
        return self._grid_size

    def get_tail_steps(self):
        return self._tail_steps

    def get_table(self):
        return self._table

    def _build_table(self):
        num_motions = self._motion_lib.get_num_motions()
        table = torch.zeros(
            [num_motions, self._grid_size, self._motion_dim],
            dtype=torch.float32, device=self._device)

        with torch.no_grad():
            for motion_id in range(num_motions):
                ids = torch.full(
                    [self._grid_size], motion_id, dtype=torch.long,
                    device=self._device)
                length = self._motion_lib.get_motion_length(ids[:1])[0]
                loop_mode = int(
                    self._motion_lib.get_motion_loop_mode(ids[:1])[0].item())

                if loop_mode == motion.LoopMode.WRAP.value:
                    phase = torch.arange(
                        self._grid_size, dtype=torch.float32,
                        device=self._device) / float(self._grid_size)
                else:
                    phase = torch.linspace(
                        0.0, 1.0, self._grid_size, dtype=torch.float32,
                        device=self._device)
                times = phase * length

                zero_time = torch.zeros(1, device=self._device)
                _, root_rot0, _, _, _, _ = (
                    self._motion_lib.calc_motion_frame(ids[:1], zero_time))
                anchor = calc_motion_anchor_quat_inv(root_rot0).expand(
                    self._grid_size, -1)

                curr = self._fetch_pose(ids, times)
                context = torch.zeros_like(table[motion_id])
                weight = 1.0
                for lag in range(self._tail_steps):
                    prev_times = times - float(lag + 1) * self._timestep
                    prev = self._fetch_pose(ids, prev_times)
                    increment = calc_motion_increment(
                        self._kin_char_model,
                        curr[0], curr[1], curr[2],
                        prev[0], prev[1], prev[2], anchor)
                    context.add_(increment, alpha=weight)
                    curr = prev
                    weight *= self._rho

                table[motion_id].copy_(context)
        return table

    def _fetch_pose(self, motion_ids, motion_times):
        root_pos, root_rot, _, _, joint_rot, _ = (
            self._motion_lib.calc_motion_frame(motion_ids, motion_times))
        return root_pos, root_rot, joint_rot

    def lookup(self, motion_ids, motion_times):
        phase = self._motion_lib.calc_motion_phase(motion_ids, motion_times)
        loop_mode = self._motion_lib.get_motion_loop_mode(motion_ids)
        wrap = loop_mode == motion.LoopMode.WRAP.value

        wrap_pos = phase * float(self._grid_size)
        wrap_floor = torch.floor(wrap_pos)
        wrap_idx0 = wrap_floor.long() % self._grid_size
        # A WRAP clip can contain a real pose jump between its last and first
        # frame.  The phase is periodic, but the motion context is then
        # piecewise continuous with a jump at phase zero.  Interpolating the
        # last table cell toward cell zero would invent values that the online
        # recurrence never visits, so keep the two sides of the seam separate.
        wrap_idx1 = torch.minimum(
            wrap_idx0 + 1,
            torch.full_like(wrap_idx0, self._grid_size - 1))
        wrap_blend = wrap_pos - wrap_floor

        clamp_pos = phase * float(self._grid_size - 1)
        clamp_idx0 = torch.floor(clamp_pos).long().clamp(
            min=0, max=self._grid_size - 1)
        clamp_idx1 = torch.minimum(
            clamp_idx0 + 1,
            torch.full_like(clamp_idx0, self._grid_size - 1))
        clamp_blend = clamp_pos - torch.floor(clamp_pos)

        idx0 = torch.where(wrap, wrap_idx0, clamp_idx0)
        idx1 = torch.where(wrap, wrap_idx1, clamp_idx1)
        blend = torch.where(wrap, wrap_blend, clamp_blend).unsqueeze(-1)

        value0 = self._table[motion_ids, idx0]
        value1 = self._table[motion_ids, idx1]
        return (1.0 - blend) * value0 + blend * value1


class CPMDConditionalMemory:
    """Episode-local motion error plus deterministic reference context."""

    def __init__(self, num_envs, kin_char_model, rho, device):
        assert 0.0 < rho <= 1.0
        self._kin_char_model = kin_char_model
        self._rho = float(rho)
        self._motion_dim = 6 + kin_char_model.get_dof_size()

        dtype = torch.float32
        shape = [num_envs, self._motion_dim]
        self._error_memory = torch.zeros(shape, dtype=dtype, device=device)
        self._ref_context = torch.zeros(shape, dtype=dtype, device=device)

        num_joints = kin_char_model.get_num_joints() - 1
        joint_shape = [num_envs, num_joints, 4]
        self._prev_sim_root_pos = torch.zeros(
            [num_envs, 3], dtype=dtype, device=device)
        self._prev_sim_root_rot = torch.zeros(
            [num_envs, 4], dtype=dtype, device=device)
        self._prev_sim_root_rot[..., 3] = 1.0
        self._prev_sim_joint_rot = torch.zeros(
            joint_shape, dtype=dtype, device=device)
        self._prev_sim_joint_rot[..., 3] = 1.0

        self._prev_ref_root_pos = torch.zeros_like(self._prev_sim_root_pos)
        self._prev_ref_root_rot = torch.zeros_like(self._prev_sim_root_rot)
        self._prev_ref_root_rot[..., 3] = 1.0
        self._prev_ref_joint_rot = torch.zeros_like(self._prev_sim_joint_rot)
        self._prev_ref_joint_rot[..., 3] = 1.0
        self._push_count = torch.zeros(
            [num_envs], dtype=torch.long, device=device)
        return

    def get_memory_decay(self):
        return self._rho

    def get_motion_dim(self):
        return self._motion_dim

    def get_error_memory(self):
        return self._error_memory

    def get_ref_context(self):
        return self._ref_context

    def get_push_count(self):
        return self._push_count

    def reset(self, env_ids, root_pos, root_rot, joint_rot, ref_context):
        self._error_memory[env_ids] = 0.0
        self._ref_context[env_ids] = ref_context

        self._prev_sim_root_pos[env_ids] = root_pos
        self._prev_sim_root_rot[env_ids] = root_rot
        self._prev_sim_joint_rot[env_ids] = joint_rot
        self._prev_ref_root_pos[env_ids] = root_pos
        self._prev_ref_root_rot[env_ids] = root_rot
        self._prev_ref_joint_rot[env_ids] = joint_rot
        self._push_count[env_ids] = 0
        return

    def push(self, sim_root_pos, sim_root_rot, sim_joint_rot, ref_root_pos,
             ref_root_rot, ref_joint_rot, anchor_inv):
        sim_increment = calc_motion_increment(
            self._kin_char_model,
            sim_root_pos, sim_root_rot, sim_joint_rot,
            self._prev_sim_root_pos, self._prev_sim_root_rot,
            self._prev_sim_joint_rot, anchor_inv)
        ref_increment = calc_motion_increment(
            self._kin_char_model,
            ref_root_pos, ref_root_rot, ref_joint_rot,
            self._prev_ref_root_pos, self._prev_ref_root_rot,
            self._prev_ref_joint_rot, anchor_inv)

        # Preserve tensor identities because environment info stores references.
        self._error_memory.mul_(self._rho).add_(
            ref_increment - sim_increment)
        self._ref_context.mul_(self._rho).add_(ref_increment)

        self._prev_sim_root_pos.copy_(sim_root_pos)
        self._prev_sim_root_rot.copy_(sim_root_rot)
        self._prev_sim_joint_rot.copy_(sim_joint_rot)
        self._prev_ref_root_pos.copy_(ref_root_pos)
        self._prev_ref_root_rot.copy_(ref_root_rot)
        self._prev_ref_joint_rot.copy_(ref_joint_rot)
        self._push_count.add_(1)
        return
