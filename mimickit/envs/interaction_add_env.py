"""DARE/ADD tracking environment with a physical interaction object.

The character discriminator remains differential.  For dynamic-object tasks,
the object's pose is appended to both policy and demonstration discriminator
observations, creating one additional semantic error group.  Static furniture
is exposed only to the policy and collision system.
"""

import pickle

import numpy as np
import torch

import envs.add_env as add_env
import envs.base_env as base_env
import engines.engine as engine
import util.torch_util as torch_util


class InteractionADDEnv(add_env.ADDEnv):
    def __init__(self, env_config, engine_config, num_envs, device,
                 visualize, record_video=False):
        cfg = env_config.get("interaction_object")
        if cfg is not None and not isinstance(cfg, dict):
            raise ValueError("interaction_object must be a mapping")
        self._has_interaction_object = cfg is not None
        cfg = {} if cfg is None else cfg
        self._interaction_cfg = cfg
        self._interaction_obj_ids = []
        self._object_dynamic = bool(cfg.get("dynamic", False))
        self._track_object = bool(cfg.get("track_reference", False))
        if self._track_object and not self._object_dynamic:
            raise ValueError("Only a dynamic interaction object can track a reference")
        if self._track_object and int(env_config["num_disc_obs_steps"]) != 1:
            raise ValueError("Object-differential DARE currently requires one discriminator frame")

        self._object_start_pos = torch.tensor(
            cfg.get("pos", [0.0, 0.0, 0.0]), device=device,
            dtype=torch.float32)
        self._object_start_rot = torch.tensor(
            cfg.get("rot", [0.0, 0.0, 0.0, 1.0]),
            device=device, dtype=torch.float32)
        self._object_start_rot /= torch.linalg.norm(self._object_start_rot)
        self._object_ref_pos = None
        self._object_ref_rot = None
        self._object_ref_fps = None
        if self._track_object:
            with open(cfg["reference_file"], "rb") as stream:
                ref = pickle.load(stream)
            self._object_ref_pos = torch.tensor(
                np.asarray(ref["pos"]), device=device, dtype=torch.float32)
            self._object_ref_rot = torch.tensor(
                np.asarray(ref["rot"]), device=device, dtype=torch.float32)
            self._object_ref_fps = float(ref["fps"])

        sim2real = env_config.get("sim2real", {})
        self._obs_noise_std = float(sim2real.get("observation_noise_std", 0.0))
        self._sim2real_warmup_samples = int(
            sim2real.get("warmup_samples", 0))
        self._sim2real_ramp_samples = int(
            sim2real.get("ramp_samples", 0))
        self._transition_samples = 0
        delay_range = sim2real.get("action_delay_steps", [0, 0])
        self._delay_min = int(delay_range[0])
        self._delay_max = int(delay_range[1])
        if self._delay_min < 0 or self._delay_max < self._delay_min:
            raise ValueError("action_delay_steps must be an ordered nonnegative range")
        self._char_mass_scale = sim2real.get("char_mass_scale", [1.0, 1.0])
        self._object_mass_scale = sim2real.get("object_mass_scale", [1.0, 1.0])
        self._friction_range = sim2real.get("friction", [1.0, 1.0])
        self._stiffness_scale = sim2real.get("stiffness_scale", [1.0, 1.0])
        self._damping_scale = sim2real.get("damping_scale", [1.0, 1.0])
        self._object_pos_jitter = float(sim2real.get("object_position_jitter", 0.0))

        super().__init__(env_config, engine_config, num_envs, device,
                         visualize, record_video=record_video)

    def _build_env(self, env_id, config):
        super()._build_env(env_id, config)
        if not self._has_interaction_object:
            return
        obj_id = self._engine.create_obj(
            env_id=env_id,
            obj_type=engine.ObjType.rigid,
            asset_file=self._interaction_cfg["file"],
            name="interaction_object",
            start_pos=self._object_start_pos.cpu().numpy(),
            start_rot=self._object_start_rot.cpu().numpy(),
            fix_root=not self._object_dynamic,
            color=np.asarray(self._interaction_cfg.get("color", [0.45, 0.3, 0.15])))
        if env_id == 0:
            self._interaction_obj_ids.append(obj_id)
        else:
            assert obj_id == self._interaction_obj_ids[0]

    def _get_interaction_obj_id(self):
        return self._interaction_obj_ids[0]

    def _build_data_buffers(self):
        super()._build_data_buffers()
        num_envs = self.get_num_envs()
        action_size = self.get_action_space().shape[0]
        self._action_history = torch.zeros(
            self._delay_max + 1, num_envs, action_size,
            device=self._device, dtype=torch.float32)
        self._action_delay = torch.zeros(
            num_envs, device=self._device, dtype=torch.long)

    def _uniform(self, limits, n):
        lo, hi = float(limits[0]), float(limits[1])
        strength = self._sim2real_strength()
        lo = 1.0 + strength * (lo - 1.0)
        hi = 1.0 + strength * (hi - 1.0)
        return lo + (hi - lo) * torch.rand(n, device=self._device)

    def _sim2real_strength(self):
        if self._mode != base_env.EnvMode.TRAIN:
            return 0.0
        elapsed = self._transition_samples - self._sim2real_warmup_samples
        if elapsed <= 0:
            return 0.0
        if self._sim2real_ramp_samples <= 0:
            return 1.0
        return min(1.0, elapsed / self._sim2real_ramp_samples)

    def _post_physics_step(self):
        super()._post_physics_step()
        if self._mode == base_env.EnvMode.TRAIN:
            self._transition_samples += self.get_num_envs()

    def _apply_action(self, actions):
        self._action_history[1:] = self._action_history[:-1].clone()
        self._action_history[0] = actions
        env_ids = torch.arange(actions.shape[0], device=self._device)
        delayed = self._action_history[self._action_delay, env_ids]
        super()._apply_action(delayed)

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) == 0:
            return

        self._reset_interaction_object(env_ids)
        current_targets = self._engine.get_dof_pos(self._get_char_id())[env_ids]
        self._action_history[:, env_ids] = current_targets.unsqueeze(0)

        if self._mode == base_env.EnvMode.TRAIN:
            strength = self._sim2real_strength()
            delay_min = int(np.floor(strength * self._delay_min))
            delay_max = int(np.floor(strength * self._delay_max))
            self._action_delay[env_ids] = torch.randint(
                delay_min, delay_max + 1, (len(env_ids),),
                device=self._device)
            self._randomize_physics(env_ids)
        else:
            self._action_delay[env_ids] = 0

    def _reset_interaction_object(self, env_ids):
        if not self._has_interaction_object or not self._object_dynamic:
            return
        times = self._get_motion_times(env_ids)
        pos, rot = self._calc_object_reference(times)
        jitter = self._sim2real_strength() * self._object_pos_jitter
        if self._mode == base_env.EnvMode.TRAIN and jitter > 0:
            pos = pos + (2.0 * torch.rand_like(pos) - 1.0) * jitter
            pos[:, 2] = torch.clamp(pos[:, 2], min=0.05)
        obj_id = self._get_interaction_obj_id()
        self._engine.set_root_pos(env_ids, obj_id, pos)
        self._engine.set_root_rot(env_ids, obj_id, rot)
        self._engine.set_root_vel(env_ids, obj_id, 0.0)
        self._engine.set_root_ang_vel(env_ids, obj_id, 0.0)

    def _randomize_physics(self, env_ids):
        n = len(env_ids)
        friction = self._uniform(self._friction_range, n)
        self._engine.randomize_obj_physics(
            env_ids, self._get_char_id(),
            mass_scale=self._uniform(self._char_mass_scale, n),
            friction=friction,
            stiffness_scale=self._uniform(self._stiffness_scale, n),
            damping_scale=self._uniform(self._damping_scale, n))
        if self._has_interaction_object and self._object_dynamic:
            self._engine.randomize_obj_physics(
                env_ids, self._get_interaction_obj_id(),
                mass_scale=self._uniform(self._object_mass_scale, n),
                friction=friction)

    def _calc_object_reference(self, times):
        frame = torch.clamp(
            times * self._object_ref_fps, 0.0,
            float(self._object_ref_pos.shape[0] - 1))
        idx0 = torch.floor(frame).long()
        idx1 = torch.clamp(idx0 + 1, max=self._object_ref_pos.shape[0] - 1)
        blend = frame - idx0.float()
        pos = (1.0 - blend.unsqueeze(-1)) * self._object_ref_pos[idx0]
        pos += blend.unsqueeze(-1) * self._object_ref_pos[idx1]
        rot = torch_util.slerp(
            self._object_ref_rot[idx0], self._object_ref_rot[idx1], blend)
        return pos, rot

    def _object_policy_obs(self, env_ids):
        obj_id = self._get_interaction_obj_id()
        char_id = self._get_char_id()
        obj_pos = self._engine.get_root_pos(obj_id)
        obj_rot = self._engine.get_root_rot(obj_id)
        obj_vel = self._engine.get_root_vel(obj_id)
        obj_ang_vel = self._engine.get_root_ang_vel(obj_id)
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        if env_ids is not None:
            obj_pos, obj_rot = obj_pos[env_ids], obj_rot[env_ids]
            obj_vel, obj_ang_vel = obj_vel[env_ids], obj_ang_vel[env_ids]
            root_pos, root_rot = root_pos[env_ids], root_rot[env_ids]

        heading_inv = torch_util.calc_heading_quat_inv(root_rot)
        rel_pos = torch_util.quat_rotate(heading_inv, obj_pos - root_pos)
        rel_rot = torch_util.quat_mul(heading_inv, obj_rot)
        local_vel = torch_util.quat_rotate(heading_inv, obj_vel)
        local_ang_vel = torch_util.quat_rotate(heading_inv, obj_ang_vel)
        obs = [rel_pos, torch_util.quat_to_tan_norm(rel_rot),
               local_vel, local_ang_vel]
        if self._track_object:
            times = self._get_motion_times(env_ids)
            target_pos, target_rot = self._calc_object_reference(times)
            target_delta = torch_util.quat_rotate(
                heading_inv, target_pos - obj_pos)
            target_rot_delta = torch_util.quat_mul(
                torch_util.quat_conjugate(obj_rot), target_rot)
            obs += [target_delta, torch_util.quat_to_tan_norm(target_rot_delta)]
        return torch.cat(obs, dim=-1)

    def _compute_obs(self, env_ids=None):
        obs = super()._compute_obs(env_ids)
        if self._has_interaction_object:
            obs = torch.cat([obs, self._object_policy_obs(env_ids)], dim=-1)
        noise_std = self._sim2real_strength() * self._obs_noise_std
        if self._mode == base_env.EnvMode.TRAIN and noise_std > 0:
            obs = obs + noise_std * torch.randn_like(obs)
        return obs

    def _object_disc_current(self, env_ids=None):
        obj_id = self._get_interaction_obj_id()
        pos = self._engine.get_root_pos(obj_id)
        rot = self._engine.get_root_rot(obj_id)
        if env_ids is not None:
            pos, rot = pos[env_ids], rot[env_ids]
        return torch.cat([pos, torch_util.quat_to_tan_norm(rot)], dim=-1)

    def _object_disc_demo(self, motion_times):
        pos, rot = self._calc_object_reference(motion_times)
        return torch.cat([pos, torch_util.quat_to_tan_norm(rot)], dim=-1)

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        disc = super()._compute_disc_obs_demo(motion_ids, motion_times0)
        if self._track_object:
            disc = torch.cat([disc, self._object_disc_demo(motion_times0)], dim=-1)
        return disc

    def _update_disc_obs(self, env_ids=None):
        if not self._track_object:
            return super()._update_disc_obs(env_ids)
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        body_pos = self._disc_hist_body_pos.get_all()
        if env_ids is not None:
            root_pos, root_rot = root_pos[env_ids], root_rot[env_ids]
            root_vel, root_ang_vel = root_vel[env_ids], root_ang_vel[env_ids]
            joint_rot, dof_vel = joint_rot[env_ids], dof_vel[env_ids]
            body_pos = body_pos[env_ids]
        disc = add_env.compute_disc_obs(
            root_pos, root_rot, root_vel, root_ang_vel,
            joint_rot, dof_vel, body_pos, self._global_obs)
        disc = torch.cat([disc, self._object_disc_current(env_ids)], dim=-1)
        if env_ids is None:
            self._disc_obs_buf[:] = disc
        else:
            self._disc_obs_buf[env_ids] = disc

    def get_disc_error_groups(self):
        if not self._track_object:
            return super().get_disc_error_groups()
        char_id = self._get_char_id()
        total_dim = int(self.get_disc_obs_space().shape[0])
        object_dim = 9
        groups = list(add_env.build_disc_error_groups(
            num_steps=self._num_disc_obs_steps,
            num_joints=self._kin_char_model.get_num_joints(),
            num_bodies=self._engine.get_obj_num_bodies(char_id),
            num_dofs=self._engine.get_obj_num_dofs(char_id),
            total_dim=total_dim - object_dim))
        groups.append(("object_pose", tuple(range(total_dim - object_dim, total_dim))))
        return tuple(groups)

    def record_diagnostics(self):
        self._diagnostics["sim2real_strength"] = self._sim2real_strength()
        if self._track_object:
            pos = self._engine.get_root_pos(self._get_interaction_obj_id())
            target, _ = self._calc_object_reference(self._get_motion_times())
            self._diagnostics["object_pos_err"] = torch.linalg.norm(
                pos - target, dim=-1).mean()
        return super().record_diagnostics()
