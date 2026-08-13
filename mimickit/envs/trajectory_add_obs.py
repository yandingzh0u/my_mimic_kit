"""Trajectory differentials for Structured Trajectory ADD (ST-ADD).

ST-ADD extends the ADD state differential with causal, multi-window
trajectory features computed identically for the simulated character and the
reference motion. For each history window h (in control steps):

  accumulated root displacement   H_0^-1 (p_t - p_{t-h})
  accumulated dof displacement    q_t - q_{t-h}
  accumulated body path           H_0^-1 [(b_t - p_t) - (b_{t-h} - p_{t-h})]
  accumulated directed rotation   Theta_{t,h} = sum_k H_0^-1 log(R_k R_{k-1}^-1)

The winding Theta is accumulated from per-step relative-rotation log
increments, never from endpoint quaternions, so a full 2*pi rotation gives
2*pi instead of 0. H_0 is the per-episode fixed motion yaw anchor (reference
root heading at motion time 0), never the current root heading.

The same TrajHistory ring is streamed for both sides:
  sim ring <- engine state each post-physics step
  ref ring <- the env's _ref_* buffers, which are sampled at unwrapped motion
              times (the motion lib accumulates per-loop root offsets), so
              WRAP motions produce seam-free reference trajectories.

On reset the rings are filled with the reset state and zero increments, so
every trajectory feature is exactly zero at the start of an episode and no
window ever mixes states across episodes. Partial windows shortly after a
reset accumulate from the reset state on both sides identically.
"""

import torch

import util.circular_buffer as circular_buffer
import util.torch_util as torch_util

@torch.jit.script
def calc_motion_anchor_quat_inv(ref_root_rot0):
    # type: (Tensor) -> Tensor
    """Inverse yaw anchor H_0^-1 from the reference root rotation at phase 0."""
    return torch_util.calc_heading_quat_inv(ref_root_rot0)

@torch.jit.script
def calc_rot_increment(root_rot, prev_root_rot, anchor_inv):
    # type: (Tensor, Tensor, Tensor) -> Tensor
    """Per-step anchored rotation increment H_0^-1 log(R_t R_{t-1}^-1).

    quat_to_exp_map forces w >= 0 internally, so the increment is identical
    for q and -q and the per-step angle is minimal (consecutive control steps
    rotate far less than pi).
    """
    delta = torch_util.quat_mul(root_rot, torch_util.quat_conjugate(prev_root_rot))
    incr = torch_util.quat_to_exp_map(delta)
    incr = torch_util.quat_rotate(anchor_inv, incr)
    return incr

@torch.jit.script
def rotate_body_points(anchor_inv, points):
    # type: (Tensor, Tensor) -> Tensor
    """Rotate [n, B, 3] points by per-env quaternions [n, 4]."""
    n = points.shape[0]
    num_points = points.shape[1]
    anchor_exp = anchor_inv.unsqueeze(-2).repeat(1, num_points, 1)
    anchor_flat = anchor_exp.reshape(n * num_points, 4)
    points_flat = points.reshape(n * num_points, 3)
    rotated = torch_util.quat_rotate(anchor_flat, points_flat)
    return rotated.reshape(n, num_points * 3)

class TrajHistory():
    """Causal history ring for one side (sim or ref) of the ST-ADD
    trajectory differential. Never reads the future; never crosses resets."""

    def __init__(self, num_envs, windows, dof_dim, num_bodies, device):
        assert len(windows) > 0
        assert all(int(h) > 0 for h in windows)
        self._windows = [int(h) for h in windows]
        self._max_window = max(self._windows)
        self._dof_dim = dof_dim
        self._body_dim = 3 * num_bodies
        self._device = device

        hist_len = self._max_window + 1
        dtype = torch.float32
        self._pos_buf = circular_buffer.CircularBuffer(batch_size=num_envs, buffer_len=hist_len,
                                                       shape=[3], dtype=dtype, device=device)
        self._dof_buf = circular_buffer.CircularBuffer(batch_size=num_envs, buffer_len=hist_len,
                                                       shape=[dof_dim], dtype=dtype, device=device)
        self._body_buf = circular_buffer.CircularBuffer(batch_size=num_envs, buffer_len=hist_len,
                                                        shape=[self._body_dim], dtype=dtype, device=device)
        self._incr_buf = circular_buffer.CircularBuffer(batch_size=num_envs, buffer_len=hist_len,
                                                        shape=[3], dtype=dtype, device=device)
        self._prev_rot = torch.zeros([num_envs, 4], dtype=dtype, device=device)
        self._prev_rot[..., 3] = 1.0
        return

    def get_windows(self):
        return list(self._windows)

    def get_motion_obs_dim(self):
        # per window: root displacement (3) + dof displacement + body path
        return len(self._windows) * (3 + self._dof_dim + self._body_dim)

    def get_rot_obs_dim(self):
        return len(self._windows) * 3

    def reset_fill(self, env_ids, root_pos, root_rot, dof_pos, body_rel, anchor_inv):
        """Fill the ring with the reset state: all trajectory features become
        exactly zero and nothing from the previous episode can leak."""
        hist_len = self._pos_buf.get_buffer_len()

        pos = torch_util.quat_rotate(anchor_inv, root_pos)
        body = rotate_body_points(anchor_inv, body_rel)

        self._pos_buf.fill(env_ids, pos.unsqueeze(1).repeat(1, hist_len, 1))
        self._dof_buf.fill(env_ids, dof_pos.unsqueeze(1).repeat(1, hist_len, 1))
        self._body_buf.fill(env_ids, body.unsqueeze(1).repeat(1, hist_len, 1))
        self._incr_buf.fill(env_ids, torch.zeros([len(env_ids), hist_len, 3],
                                                 dtype=torch.float32, device=self._device))
        self._prev_rot[env_ids] = root_rot
        return

    def push(self, root_pos, root_rot, dof_pos, body_rel, anchor_inv):
        incr = calc_rot_increment(root_rot, self._prev_rot, anchor_inv)
        self._prev_rot[:] = root_rot

        self._pos_buf.push(torch_util.quat_rotate(anchor_inv, root_pos))
        self._dof_buf.push(dof_pos)
        self._body_buf.push(rotate_body_points(anchor_inv, body_rel))
        self._incr_buf.push(incr)
        return

    def extract(self):
        """Multi-window trajectory features at the current time.

        Returns:
          motion_obs [n, num_windows * (3 + dof + body)]:
              per window [root displacement, dof displacement, body path]
          rot_obs    [n, num_windows * 3]:
              per window accumulated directed rotation (winding)
        """
        pos_all = self._pos_buf.get_all()
        dof_all = self._dof_buf.get_all()
        body_all = self._body_buf.get_all()
        incr_all = self._incr_buf.get_all()

        hist_len = pos_all.shape[1]
        motion_obs = []
        rot_obs = []
        for h in self._windows:
            past = hist_len - 1 - h
            delta_pos = pos_all[:, -1] - pos_all[:, past]
            delta_dof = dof_all[:, -1] - dof_all[:, past]
            delta_body = body_all[:, -1] - body_all[:, past]
            winding = torch.sum(incr_all[:, hist_len - h:], dim=1)

            motion_obs.append(torch.cat([delta_pos, delta_dof, delta_body], dim=-1))
            rot_obs.append(winding)

        motion_obs = torch.cat(motion_obs, dim=-1)
        rot_obs = torch.cat(rot_obs, dim=-1)
        return motion_obs, rot_obs
