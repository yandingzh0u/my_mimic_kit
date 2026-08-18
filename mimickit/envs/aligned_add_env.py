import torch

import envs.add_env as add_env


def validate_aligned_add_config(env_config):
    if env_config.get("enable_tar_obs", False):
        raise ValueError("aligned ADD replaces the legacy target observation; set enable_tar_obs=false")
    if env_config.get("enable_phase_obs", False):
        raise ValueError("aligned ADD does not use a separate phase observation")
    if int(env_config.get("num_disc_obs_steps", 1)) != 1:
        raise ValueError("aligned ADD preserves the stock one-frame 172D ADD discriminator")
    if int(env_config.get("aligned_command_step", 1)) != 1:
        raise ValueError("the first implementation uses exactly one control-step reference motion")
    if not env_config.get("global_obs", False):
        raise ValueError("aligned ADD requires global_obs=true so error and reference motion share one frame")


class AlignedADDEnv(add_env.ADDEnv):
    """ADD with a factorized differential command for policy conditioning.

    The discriminator path is inherited without modification.  The actor sees
    its ordinary character observation followed by

      e_t   = phi(ref_t)   - phi(sim_t)
      m_t^r = phi(ref_t+1) - phi(ref_t).

    Here phi is exactly ``add_env.compute_disc_obs``. If the action produces
    delta_t = phi(sim_t+1) - phi(sim_t), the untouched ADD reward observes

      e_t+1 = e_t + m_t^r - delta_t.

    The two command factors preserve their distinct feedback and feedforward
    semantics and statistics while sharing ADD's differential feature axes.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        validate_aligned_add_config(env_config)
        self._aligned_command_step = int(env_config.get("aligned_command_step", 1))
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)

    def get_aligned_self_obs_dim(self):
        obs_dim = self.get_obs_space().shape[0]
        command_dim = self.get_aligned_command_dim()
        return int(obs_dim - 2 * command_dim)

    def get_aligned_command_dim(self):
        return int(self.get_disc_obs_space().shape[0])

    def _track_global_root(self):
        # Preserve the stock target-conditioned ADD termination/tracking
        # semantics even though the legacy target-observation branch is off.
        return self._global_obs

    def _compute_obs(self, env_ids=None):
        self_obs = super()._compute_obs(env_ids)
        curr_error, ref_motion = self._compute_aligned_commands(env_ids)
        return torch.cat([self_obs, curr_error, ref_motion], dim=-1)

    def _compute_aligned_commands(self, env_ids=None):
        if env_ids is None:
            motion_ids = self._motion_ids
        else:
            motion_ids = self._motion_ids[env_ids]

        motion_times = self._get_motion_times(env_ids)
        ref_obs = self._compute_disc_obs_demo(motion_ids, motion_times)
        dt = self._engine.get_timestep() * self._aligned_command_step
        next_ref_obs = self._compute_disc_obs_demo(motion_ids, motion_times + dt)
        sim_obs = self._compute_current_sim_disc_obs(env_ids)

        return compute_factorized_command(ref_obs, next_ref_obs, sim_obs)

    def _compute_current_sim_disc_obs(self, env_ids=None):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)
        body_pos = self._engine.get_body_pos(char_id)

        if env_ids is not None:
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_pos = dof_pos[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        # ADD's one-frame implementation expects an explicit history axis.
        return add_env.compute_disc_obs(
            root_pos=root_pos.unsqueeze(1),
            root_rot=root_rot.unsqueeze(1),
            root_vel=root_vel.unsqueeze(1),
            root_ang_vel=root_ang_vel.unsqueeze(1),
            joint_rot=joint_rot.unsqueeze(1),
            dof_vel=dof_vel.unsqueeze(1),
            body_pos=body_pos.unsqueeze(1),
            global_obs=self._global_obs)


def compute_factorized_command(ref_obs, next_ref_obs, sim_obs):
    """Feedback residual and feedforward tangent in ADD feature coordinates."""
    curr_error = ref_obs - sim_obs
    ref_motion = next_ref_obs - ref_obs
    return curr_error, ref_motion
