import math

import torch

import envs.add_env as add_env


_ABSOLUTE = "absolute"
_RESIDUAL = "residual"


def validate_rcci_add_config(env_config):
    if env_config.get("enable_tar_obs", False):
        raise ValueError("RCCI replaces the legacy target observation")
    if env_config.get("enable_phase_obs", False):
        raise ValueError("RCCI does not use a separate phase observation")
    if int(env_config.get("num_disc_obs_steps", 1)) != 1:
        raise ValueError("RCCI preserves the stock one-frame ADD discriminator")
    if int(env_config.get("rcci_command_step", 1)) != 1:
        raise ValueError("RCCI uses exactly one control-step lookahead")
    if not env_config.get("global_obs", False):
        raise ValueError("RCCI requires global_obs=true for a shared phi frame")

    representation = env_config.get("rcci_representation")
    if representation not in (_ABSOLUTE, _RESIDUAL):
        raise ValueError("rcci_representation must be absolute or residual")

    if int(env_config.get("rcci_stats_samples", 4096)) <= 0:
        raise ValueError("rcci_stats_samples must be positive")


class RCCIADDEnv(add_env.ADDEnv):
    """Strict information-equivalent ADD policy interfaces.

    Both variants expose the same self observation and current ADD feature
    state x_t.  The remaining two blocks are either absolute reference states

      [x_t, ref_t, ref_t+1]

    or the residual-closed coordinates

      [x_t, ref_t - x_t, ref_t+1 - ref_t].

    The two triples are related by a fixed affine bijection.  The ADD
    discriminator, reward, and training objective remain untouched.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        validate_rcci_add_config(env_config)
        self._rcci_representation = env_config["rcci_representation"]
        self._rcci_command_step = int(env_config.get("rcci_command_step", 1))
        self._rcci_stats_samples = int(env_config.get("rcci_stats_samples", 4096))
        self._rcci_phi_mean = None
        self._rcci_phi_std = None
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)

    def get_rcci_self_obs_dim(self):
        obs_dim = self.get_obs_space().shape[0]
        return int(obs_dim - 3 * self.get_rcci_phi_dim())

    def get_rcci_phi_dim(self):
        return int(self.get_disc_obs_space().shape[0])

    def get_rcci_representation(self):
        return self._rcci_representation

    def get_rcci_phi_stats(self):
        if self._rcci_phi_mean is None:
            self._build_rcci_phi_stats()
        return self._rcci_phi_mean, self._rcci_phi_std

    def _track_global_root(self):
        return self._global_obs

    def _compute_obs(self, env_ids=None):
        self_obs = super()._compute_obs(env_ids)
        sim_obs, ref_obs, next_ref_obs = self._compute_rcci_states(env_ids)
        block1, block2, block3 = compute_rcci_command(
            sim_obs, ref_obs, next_ref_obs, self._rcci_representation)
        return torch.cat([self_obs, block1, block2, block3], dim=-1)

    def _compute_rcci_states(self, env_ids=None):
        if env_ids is None:
            motion_ids = self._motion_ids
        else:
            motion_ids = self._motion_ids[env_ids]

        motion_times = self._get_motion_times(env_ids)
        ref_obs = self._compute_disc_obs_demo(motion_ids, motion_times)
        dt = self._engine.get_timestep() * self._rcci_command_step
        next_ref_obs = self._compute_disc_obs_demo(motion_ids, motion_times + dt)
        sim_obs = self._compute_current_sim_disc_obs(env_ids)
        return sim_obs, ref_obs, next_ref_obs

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
        return add_env.compute_disc_obs(
            root_pos=root_pos.unsqueeze(1),
            root_rot=root_rot.unsqueeze(1),
            root_vel=root_vel.unsqueeze(1),
            root_ang_vel=root_ang_vel.unsqueeze(1),
            joint_rot=joint_rot.unsqueeze(1),
            dof_vel=dof_vel.unsqueeze(1),
            body_pos=body_pos.unsqueeze(1),
            global_obs=self._global_obs)

    def _build_rcci_phi_stats(self):
        """Compute deterministic, demonstration-only fixed phi statistics."""
        num_motions = self._motion_lib.get_num_motions()
        samples_per_motion = int(math.ceil(self._rcci_stats_samples / num_motions))
        phase = (torch.arange(samples_per_motion, device=self._device,
                              dtype=torch.float32) + 0.5) / samples_per_motion
        motion_ids = torch.arange(num_motions, device=self._device,
                                  dtype=torch.long).repeat_interleave(samples_per_motion)
        phase = phase.repeat(num_motions)
        motion_lengths = self._motion_lib.get_motion_length(motion_ids)
        motion_times = phase * motion_lengths

        with torch.no_grad():
            phi = self._compute_disc_obs_demo(motion_ids, motion_times)
            mean = torch.mean(phi, dim=0)
            std = torch.std(phi, dim=0, correction=0)
            # Structurally constant demonstration coordinates use unit scale;
            # this avoids injecting arbitrary large weights while remaining a
            # fixed, invertible coordinate transform.
            std = torch.where(std > 1e-4, std, torch.ones_like(std))

        self._rcci_phi_mean = mean.detach().clone()
        self._rcci_phi_std = std.detach().clone()


def compute_rcci_command(sim_obs, ref_obs, next_ref_obs, representation):
    if representation == _ABSOLUTE:
        return sim_obs, ref_obs, next_ref_obs
    if representation == _RESIDUAL:
        return sim_obs, ref_obs - sim_obs, next_ref_obs - ref_obs
    raise ValueError("unsupported RCCI representation: {}".format(representation))
