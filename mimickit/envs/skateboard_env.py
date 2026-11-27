"""Track a skater with a freely simulated board and a synchronized reference."""
import math

import numpy as np
import torch

from engines.engine import ObjType
from envs.add_env import ADDEnv
from util import torch_util


class SkateboardEnv(ADDEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        with np.load(env_config["board_reference"], allow_pickle=False) as ref:
            self._fps = int(ref["fps"])
            self._board_joint_names = tuple(ref["dof_names"].tolist())
            self._board = torch.as_tensor(ref["root"], device=device)
            self._board_dof_pos = torch.as_tensor(ref["dof_pos"], device=device)
            self._board_dof_vel = torch.as_tensor(ref["dof_vel"], device=device)
            self._robot_velocity = tuple(torch.as_tensor(ref[key], device=device) for key in
                                         ("robot_root_vel", "robot_root_ang_vel", "robot_dof_vel"))
        self._last_frame = len(self._board) - 1
        expected_shapes = ((len(self._board), 13), (len(self._board), len(self._board_joint_names)),
                           (len(self._board), len(self._board_joint_names)),
                           (len(self._board), 3), (len(self._board), 3), (len(self._board), 28))
        states = (self._board, self._board_dof_pos, self._board_dof_vel, *self._robot_velocity)
        if (self._fps != engine_config["control_freq"] or
                not math.isclose(self._last_frame / self._fps, env_config["episode_length"]) or
                any(s.shape != shape or not torch.isfinite(s).all() for s, shape in zip(states, expected_shapes))):
            raise ValueError("Skater and board reference must share one complete control timeline")
        self._preview_steps = torch.tensor(env_config["tar_obs_steps"], device=device)
        super().__init__(env_config, engine_config, num_envs, device, visualize, record_video)

    def _load_motions(self, motion_file):
        super()._load_motions(motion_file)
        ids = torch.zeros(1, device=self._device, dtype=torch.long)
        length = self._motion_lib.get_motion_length(ids).item()
        if not math.isclose(length, self._last_frame / self._fps, abs_tol=1e-6):
            raise ValueError("Humanoid motion length differs from the board reference")

    def _build_static_objects(self, env_id):
        super()._build_static_objects(env_id)
        start = self._board[0].cpu().numpy()
        self._board_id = self._engine.create_obj(
            env_id, ObjType.articulated, self._board_asset, "skateboard",
            start_pos=start[:3], start_rot=start[3:7], enable_self_collisions=False, fix_root=False)

    def _build_envs(self, config, num_envs):
        self._board_asset = config["board_asset"]
        super()._build_envs(config, num_envs)

    def _build_sim_tensors(self, config):
        super()._build_sim_tensors(config)
        names = self._engine.get_obj_dof_names(self._board_id)
        if set(names) != set(self._board_joint_names):
            raise ValueError(f"Unexpected board articulation joint names: {names}")
        order = [self._board_joint_names.index(name) for name in names]
        self._board_dof_pos = self._board_dof_pos[:, order].contiguous()
        self._board_dof_vel = self._board_dof_vel[:, order].contiguous()
        self._tilt = names.index("board_tilt_joint")

    def _sample_motion_times(self, n):
        ids, time = super()._sample_motion_times(n)
        frame = (time * self._fps).long().clamp_max(self._last_frame - 1)
        return ids, frame / self._fps

    def _frame(self, env_ids=None):
        offsets, elapsed = self._motion_time_offsets, self._timestep_buf
        if env_ids is not None:
            offsets, elapsed = offsets[env_ids], elapsed[env_ids]
        return ((offsets * self._fps).round().long() + elapsed).clamp_max(self._last_frame)

    def _get_motion_times(self, env_ids=None):
        return self._frame(env_ids) / self._fps

    def _ref_state_init(self, env_ids):
        super()._ref_state_init(env_ids)
        frame = self._frame(env_ids)
        engine, char, board = self._engine, self._get_char_id(), self._board_id
        root_vel, root_ang_vel, dof_vel = self._robot_velocity
        engine.set_root_vel(env_ids, char, root_vel[frame])
        engine.set_root_ang_vel(env_ids, char, root_ang_vel[frame])
        engine.set_dof_vel(env_ids, char, dof_vel[frame])
        state = self._board[frame]
        engine.set_root_pos(env_ids, board, state[:, :3])
        engine.set_root_rot(env_ids, board, state[:, 3:7])
        engine.set_root_vel(env_ids, board, state[:, 7:10])
        engine.set_root_ang_vel(env_ids, board, state[:, 10:13])
        engine.set_dof_pos(env_ids, board, self._board_dof_pos[frame])
        engine.set_dof_vel(env_ids, board, self._board_dof_vel[frame])

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        engine, char, board = self._engine, self._get_char_id(), self._board_id
        state = (engine.get_root_pos(char), engine.get_root_rot(char), engine.get_root_pos(board),
                 engine.get_root_rot(board), engine.get_root_vel(board), engine.get_root_ang_vel(board),
                 engine.get_dof_pos(board)[:, self._tilt:self._tilt + 1],
                 engine.get_dof_vel(board)[:, self._tilt:self._tilt + 1])
        root, robot_rot, pos, rot, vel, ang_vel, tilt, tilt_vel = (
            state if env_ids is None else tuple(x[env_ids] for x in state))
        heading = torch_util.calc_heading_quat_inv(robot_rot)
        local_rot = torch_util.quat_to_matrix(torch_util.quat_mul(heading, rot))
        frame = self._frame(env_ids)
        frames = (frame[:, None] + self._preview_steps).clamp_max(self._last_frame)
        target = self._board[frames]
        target_heading = heading[:, None, :].expand_as(target[..., 3:7])
        relative_rot = torch_util.quat_mul(torch_util.quat_conjugate(rot)[:, None, :].expand_as(target_heading),
                                          target[..., 3:7])
        relative_rot = torch_util.quat_to_matrix(relative_rot)
        preview = torch.cat((torch_util.quat_rotate(target_heading, target[..., :3] - pos[:, None]),
                             relative_rot[..., :, 0], relative_rot[..., :, 2],
                             torch_util.quat_rotate(target_heading, target[..., 7:10] - vel[:, None]),
                             self._board_dof_pos[frames, self._tilt, None] - tilt[:, None]), dim=-1)
        return torch.cat((obs, torch_util.quat_rotate(heading, pos - root), local_rot[:, :, 0], local_rot[:, :, 2],
                          torch_util.quat_rotate(heading, vel), torch_util.quat_rotate(heading, ang_vel),
                          tilt, tilt_vel, preview.flatten(1)), dim=-1)
