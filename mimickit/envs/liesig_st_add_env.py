import torch

import envs.add_env as add_env
import envs.lie_signature_obs as lie_signature_obs

class LieSigSTADDEnv(add_env.ADDEnv):
    """ADD with a lifted trajectory differential.

    The discriminator differential becomes

        Delta_t = [ o^ref_t - o^sim_t ,  Phi^ref_t - Phi^sim_t ]

    where o is the unchanged ADD state observation and Phi is the causal
    discounted lift of the fixed-anchor developed increments: order 1 is the
    discounted increment sum m, order 2 adds the symmetric quadratic block
    1/2 m m^T (or, under liesig_second_order: "area", the discounted Levy
    area, kept only as a completed ablation).

    Both sides stream through identical operators: the simulated character
    from the engine, the reference from the env's _ref_* buffers, which are
    sampled at unwrapped motion times, so WRAP motions are seam-free. The
    difference is taken after the operator, never before.

    Everything else is stock ADD: one discriminator, one BCE, one
    softplus reward. The env only changes what the differential contains.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._liesig_order = int(env_config.get("liesig_order", 2))
        assert self._liesig_order in [1, 2]

        self._liesig_memory_seconds = float(env_config.get("liesig_memory_seconds", 32.0 / 30.0))
        assert self._liesig_memory_seconds > 0.0

        # "sym" is the method; "area" is the same-width Levy-area ablation,
        # which the experiments showed to be unnecessary
        self._liesig_second_order = env_config.get("liesig_second_order",
                                                   lie_signature_obs.SECOND_ORDER_SYM)
        assert self._liesig_second_order in lie_signature_obs.SECOND_ORDER_MODES

        # cold start only: warm starting would change what a level-2 feature
        # numerically means, so it is not allowed to silently mix in
        self._liesig_reset_mode = env_config.get("liesig_reset_mode", "zero")
        assert self._liesig_reset_mode == "zero"

        self._liesig_schema_version = int(env_config.get("liesig_schema_version", 1))
        assert self._liesig_schema_version == 1

        # per-episode fixed heading anchor C_0 (identity until the first reset)
        self._motion_anchor_inv = torch.zeros([num_envs, 4], dtype=torch.float32, device=device)
        self._motion_anchor_inv[..., 3] = 1.0

        self._disc_state_obs_dim = None

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def get_liesig_order(self):
        return self._liesig_order

    def get_liesig_second_order(self):
        return self._sim_liesig.get_second_order()

    def get_liesig_memory_decay(self):
        return self._sim_liesig.get_memory_decay()

    def get_disc_state_obs_dim(self):
        assert self._disc_state_obs_dim is not None
        return self._disc_state_obs_dim

    def get_liesig_tangent_dim(self):
        return self._sim_liesig.get_tangent_dim()

    def get_liesig_area_dim(self):
        return self._sim_liesig.get_area_dim()

    def get_liesig_obs_dim(self):
        return self._sim_liesig.get_obs_dim()

    def get_liesig_push_counts(self):
        return self._sim_liesig.get_push_count(), self._ref_liesig.get_push_count()

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        rho = lie_signature_obs.calc_memory_decay(self._liesig_memory_seconds,
                                                  self._engine.get_timestep())

        self._sim_liesig = lie_signature_obs.LieSigHistory(num_envs=num_envs,
                                                           kin_char_model=self._kin_char_model,
                                                           order=self._liesig_order,
                                                           rho=rho,
                                                           device=self._device,
                                                           second_order=self._liesig_second_order)
        self._ref_liesig = lie_signature_obs.LieSigHistory(num_envs=num_envs,
                                                           kin_char_model=self._kin_char_model,
                                                           order=self._liesig_order,
                                                           rho=rho,
                                                           device=self._device,
                                                           second_order=self._liesig_second_order)

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
        self._motion_anchor_inv[env_ids] = lie_signature_obs.calc_motion_anchor_quat_inv(root_rot0)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if (len(env_ids) > 0):
            self._reset_liesig(env_ids)
        return

    def _reset_liesig(self, env_ids):
        # The character is initialized exactly to the reference frame, so both
        # sides start from the same pose with zero signature: the differential
        # is exactly zero at reset and nothing crosses episodes. Partial
        # resets only touch their own envs.
        root_pos, root_rot, joint_rot = self._get_ref_liesig_state()

        args = (env_ids, root_pos[env_ids], root_rot[env_ids], joint_rot[env_ids])
        self._sim_liesig.reset(*args)
        self._ref_liesig.reset(*args)
        return

    def _get_sim_liesig_state(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        joint_rot = self._kin_char_model.dof_to_rot(self._engine.get_dof_pos(char_id))
        return root_pos, root_rot, joint_rot

    def _get_ref_liesig_state(self):
        return self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot

    def _update_misc(self):
        # super() advances physics bookkeeping, the reference motion and the
        # ADD history to t+1; it runs post-physics only, so pushing here gives
        # each side exactly one increment per control step
        super()._update_misc()
        anchor_inv = self._motion_anchor_inv
        self._sim_liesig.push(*self._get_sim_liesig_state(), anchor_inv)
        self._ref_liesig.push(*self._get_ref_liesig_state(), anchor_inv)
        return

    def _update_disc_obs(self, env_ids=None):
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        body_pos = self._disc_hist_body_pos.get_all()

        sig_obs = self._sim_liesig.extract()

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            joint_rot = joint_rot[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]
            sig_obs = sig_obs[env_ids]

        state_obs = add_env.compute_disc_obs(root_pos=root_pos,
                                             root_rot=root_rot,
                                             root_vel=root_vel,
                                             root_ang_vel=root_ang_vel,
                                             joint_rot=joint_rot,
                                             dof_vel=dof_vel,
                                             body_pos=body_pos,
                                             global_obs=self._global_obs)

        disc_obs = torch.cat([state_obs, sig_obs], dim=-1)

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

        # state part from the motion lib, signature part from the streamed
        # reference operator (same anchor and same recursion as the sim side)
        state_obs = add_env.ADDEnv._compute_disc_obs_demo(self, motion_ids, motion_times0)

        sig_obs = self._ref_liesig.extract()
        if (env_ids is not None):
            sig_obs = sig_obs[env_ids]

        disc_obs = torch.cat([state_obs, sig_obs], dim=-1)

        if (env_ids is None):
            self._disc_obs_demo_buf[:] = disc_obs
        else:
            self._disc_obs_demo_buf[env_ids] = disc_obs
        return

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        # Probe/sampling path (disc obs space): the signature block is a zero
        # placeholder. Rollout data always flows through _update_disc_obs /
        # _update_disc_obs_demo, which read the streamed operators.
        state_obs = super()._compute_disc_obs_demo(motion_ids, motion_times0)
        self._disc_state_obs_dim = state_obs.shape[-1]

        n = state_obs.shape[0]
        sig_obs = torch.zeros([n, self._sim_liesig.get_obs_dim()],
                              dtype=state_obs.dtype, device=state_obs.device)

        disc_obs = torch.cat([state_obs, sig_obs], dim=-1)
        return disc_obs
