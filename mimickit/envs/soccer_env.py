"""Single reference shot with a freely simulated ball."""

import math

import torch

from engines.engine import ObjType
from envs.add_env import ADDEnv
from envs.base_env import EnvMode
from util import torch_util


class SoccerEnv(ADDEnv):
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._ball_asset = env_config["ball_asset"]
        self._ball_pos = torch.as_tensor(env_config["ball_pos"], device=device, dtype=torch.float32)
        self._impact_time = float(env_config["impact_time"])
        self._guide_steps = int(env_config["guide_preview_steps"])
        if "physical_reference" in env_config or "residual_limit_rad" in env_config:
            raise ValueError("Soccer expects a pose reference, not a recorded control sequence")
        self._reset_speed_radius = float(env_config.get("reset_speed_radius_mps", 0.0))
        self._reset_ball_radius = float(env_config.get("reset_ball_radius_m", 0.0))
        if not all(math.isfinite(x) and x >= 0 for x in (self._reset_speed_radius, self._reset_ball_radius)):
            raise ValueError("Soccer reset perturbation radii must be finite and nonnegative")
        if not 0.25 < self._impact_time < env_config["episode_length"]:
            raise ValueError("Expected impact must follow the pre-contact reset window")
        if engine_config.get("control_mode") != "pos" or self._guide_steps < 1:
            raise ValueError("Soccer guide requires position control and a positive preview")
        super().__init__(env_config, engine_config, num_envs, device, visualize, record_video)

    def _build_static_objects(self, env_id):
        super()._build_static_objects(env_id)
        self._ball_id = self._engine.create_obj(
            env_id, ObjType.rigid, self._ball_asset, "football",
            start_pos=self._ball_pos.cpu().numpy(), fix_root=False)

    def _sample_motion_times(self, n):
        ids, times = super()._sample_motion_times(n)
        if self._rand_reset and (self._mode == EnvMode.TRAIN or self._test_random_start):
            # The ball is placed at rest. Never reset after the expected hit.
            times = times * (self._impact_time - 0.12) / self._motion_lib.get_motion_length(ids)
            if self._mode == EnvMode.TRAIN:
                draw = torch.rand_like(times)
                pre_contact = self._impact_time - 0.25 + 0.13 * torch.rand_like(times)
                times = torch.where(draw >= 0.75, pre_contact, times)
                times = times.masked_fill(draw < 0.5, 0)
        return ids, times

    def _ref_state_init(self, env_ids):
        super()._ref_state_init(env_ids)
        n = len(env_ids)
        ball_pos = self._ball_pos.expand(n, -1)
        if self._mode == EnvMode.TRAIN and self._reset_speed_radius:
            char = self._get_char_id()
            velocity = self._engine.get_root_vel(char)[env_ids].clone()
            velocity[:, :2] += self._disk_noise(n, self._reset_speed_radius)
            self._engine.set_root_vel(env_ids, char, velocity)
        if self._mode == EnvMode.TRAIN and self._reset_ball_radius:
            ball_pos = ball_pos.clone()
            ball_pos[:, :2] += self._disk_noise(n, self._reset_ball_radius)
        self._engine.set_root_pos(env_ids, self._ball_id, ball_pos)
        identity = torch.zeros((n, 4), device=self._device)
        identity[:, 3] = 1
        self._engine.set_root_rot(env_ids, self._ball_id, identity)
        self._engine.set_root_vel(env_ids, self._ball_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, self._ball_id, 0.0)

    def _disk_noise(self, count, radius):
        u = torch.rand((count, 2), device=self._device)
        angle = (2 * math.pi) * u[:, 0]
        direction = torch.stack((angle.cos(), angle.sin()), dim=-1)
        return radius * u[:, 1:].sqrt() * direction

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        char = self._get_char_id()
        root = self._engine.get_root_pos(char)
        rot = self._engine.get_root_rot(char)
        ball = self._engine.get_root_pos(self._ball_id)
        velocity = self._engine.get_root_vel(self._ball_id)
        if env_ids is not None:
            root, rot, ball, velocity = (x[env_ids] for x in (root, rot, ball, velocity))
        heading = torch_util.calc_heading_quat_inv(rot)
        return torch.cat((obs, torch_util.quat_rotate(heading, ball - root),
                          torch_util.quat_rotate(heading, velocity)), dim=-1)

    def _apply_action(self, actions):
        # Center the existing PD action range on the reference pose. Either
        # residual endpoint still reaches the same original joint bound.
        low, high = self._action_bound_low, self._action_bound_high
        time = self._get_motion_times() + self._guide_steps * self._engine.get_timestep()
        joint_rot = self._motion_lib.calc_motion_frame(self._motion_ids, time)[4]
        guide = self._motion_lib.joint_rot_to_dof(joint_rot).clamp(low, high)
        residual = 2 * (actions - low) / (high - low) - 1
        available = torch.where(residual >= 0, high - guide, guide - low)
        super()._apply_action(guide + residual.clamp(-1, 1) * available)
