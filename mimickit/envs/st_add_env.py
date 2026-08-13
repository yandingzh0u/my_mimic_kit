import torch

import envs.add_env as add_env
import envs.trajectory_add_obs as trajectory_add_obs

class STADDEnv(add_env.ADDEnv):
    """Structured Trajectory ADD environment.

    Extends the ADD discriminator observation with causal multi-window
    trajectory features so the discriminator differential becomes

        [ state differential | motion trajectory residual | winding residual ]

    Both the simulated character and the reference motion stream through
    identical TrajHistory rings (same windows, same per-episode motion
    anchor). The reference side is fed from the env's _ref_* buffers, which
    are sampled at unwrapped motion times, so WRAP motions are seam-free.
    Rings are refilled with the reset state on every reset: trajectory
    features start at exactly zero and never cross episodes.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._traj_obs_steps = [int(h) for h in env_config.get("traj_obs_steps", [8, 16, 32])]
        assert len(self._traj_obs_steps) > 0
        assert all(h > 0 for h in self._traj_obs_steps)

        # per-episode fixed motion yaw anchor H_0^-1 (identity until first reset)
        self._motion_anchor_inv = torch.zeros([num_envs, 4], dtype=torch.float32, device=device)
        self._motion_anchor_inv[..., 3] = 1.0

        self._disc_state_obs_dim = None

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def get_traj_obs_steps(self):
        return list(self._traj_obs_steps)

    def get_disc_state_obs_dim(self):
        assert self._disc_state_obs_dim is not None
        return self._disc_state_obs_dim

    def get_disc_traj_motion_obs_dim(self):
        return self._sim_traj_hist.get_motion_obs_dim()

    def get_disc_traj_rot_obs_dim(self):
        return self._sim_traj_hist.get_rot_obs_dim()

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        char_id = self._get_char_id()
        dof_dim = self._engine.get_dof_pos(char_id).shape[-1]
        num_bodies = self._engine.get_body_pos(char_id).shape[-2]

        self._sim_traj_hist = trajectory_add_obs.TrajHistory(num_envs=num_envs,
                                                             windows=self._traj_obs_steps,
                                                             dof_dim=dof_dim,
                                                             num_bodies=num_bodies,
                                                             device=self._device)
        self._ref_traj_hist = trajectory_add_obs.TrajHistory(num_envs=num_envs,
                                                             windows=self._traj_obs_steps,
                                                             dof_dim=dof_dim,
                                                             num_bodies=num_bodies,
                                                             device=self._device)

        # sizes _disc_obs_buf via get_disc_obs_space -> _compute_disc_obs_demo
        super()._build_disc_obs_buffers()
        return

    def _reset_char(self, env_ids):
        super()._reset_char(env_ids)
        self._update_motion_anchors(env_ids)
        return

    def _update_motion_anchors(self, env_ids):
        motion_ids = self._motion_ids[env_ids]
        motion_times0 = torch.zeros_like(self._get_motion_times(env_ids))
        _, root_rot0, _, _, _, _ = self._motion_lib.calc_motion_frame(motion_ids, motion_times0)
        self._motion_anchor_inv[env_ids] = trajectory_add_obs.calc_motion_anchor_quat_inv(root_rot0)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if (len(env_ids) > 0):
            self._reset_traj_hist(env_ids)
        return

    def _reset_traj_hist(self, env_ids):
        # At reset the character is initialized exactly to the reference
        # frame, so both rings are filled from the reference state: all
        # trajectory features are exactly zero and nothing leaks across
        # episodes.
        root_pos, root_rot, dof_pos, body_rel = self._get_ref_traj_state()
        anchor_inv = self._motion_anchor_inv[env_ids]

        args = (env_ids, root_pos[env_ids], root_rot[env_ids], dof_pos[env_ids],
                body_rel[env_ids], anchor_inv)
        self._sim_traj_hist.reset_fill(*args)
        self._ref_traj_hist.reset_fill(*args)
        return

    def _get_sim_traj_state(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)
        body_pos = self._engine.get_body_pos(char_id)
        body_rel = body_pos - root_pos.unsqueeze(-2)
        return root_pos, root_rot, dof_pos, body_rel

    def _get_ref_traj_state(self):
        root_pos = self._ref_root_pos
        root_rot = self._ref_root_rot
        dof_pos = self._motion_lib.joint_rot_to_dof(self._ref_joint_rot)
        body_rel = self._ref_body_pos - root_pos.unsqueeze(-2)
        return root_pos, root_rot, dof_pos, body_rel

    def _update_misc(self):
        # super() advances the reference to t+1 and pushes the AMP disc
        # history; runs post-physics only, so each side is pushed exactly
        # once per control step.
        super()._update_misc()
        anchor_inv = self._motion_anchor_inv
        self._sim_traj_hist.push(*self._get_sim_traj_state(), anchor_inv)
        self._ref_traj_hist.push(*self._get_ref_traj_state(), anchor_inv)
        return

    def _update_disc_obs(self, env_ids=None):
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        body_pos = self._disc_hist_body_pos.get_all()

        motion_obs, rot_obs = self._sim_traj_hist.extract()

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            joint_rot = joint_rot[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]
            motion_obs = motion_obs[env_ids]
            rot_obs = rot_obs[env_ids]

        state_obs = add_env.compute_disc_obs(root_pos=root_pos,
                                             root_rot=root_rot,
                                             root_vel=root_vel,
                                             root_ang_vel=root_ang_vel,
                                             joint_rot=joint_rot,
                                             dof_vel=dof_vel,
                                             body_pos=body_pos,
                                             global_obs=self._global_obs)

        disc_obs = torch.cat([state_obs, motion_obs, rot_obs], dim=-1)

        if (env_ids is None):
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs
        return

    def _update_disc_obs_demo(self, env_ids=None):
        if (env_ids is None):
            motion_ids = self._motion_ids
        else:
            motion_ids = self._motion_ids[env_ids]
        motion_times0 = self._get_motion_times(env_ids)

        # state part only; the trajectory part comes from the streamed
        # reference ring (same windows/anchor as the sim side)
        state_obs = add_env.ADDEnv._compute_disc_obs_demo(self, motion_ids, motion_times0)

        motion_obs, rot_obs = self._ref_traj_hist.extract()
        if (env_ids is not None):
            motion_obs = motion_obs[env_ids]
            rot_obs = rot_obs[env_ids]

        disc_obs = torch.cat([state_obs, motion_obs, rot_obs], dim=-1)

        if (env_ids is None):
            self._disc_obs_demo_buf[:] = disc_obs
        else:
            self._disc_obs_demo_buf[env_ids] = disc_obs
        return

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        # Probe/sampling path (disc obs space, demo fetches): trajectory
        # features are zero placeholders. Rollout data always flows through
        # _update_disc_obs / _update_disc_obs_demo, which use the rings.
        state_obs = super()._compute_disc_obs_demo(motion_ids, motion_times0)
        self._disc_state_obs_dim = state_obs.shape[-1]

        n = state_obs.shape[0]
        traj_dim = self.get_disc_traj_motion_obs_dim() + self.get_disc_traj_rot_obs_dim()
        traj_obs = torch.zeros([n, traj_dim], dtype=state_obs.dtype, device=state_obs.device)

        disc_obs = torch.cat([state_obs, traj_obs], dim=-1)
        return disc_obs
