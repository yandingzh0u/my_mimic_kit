import torch

import envs.add_env as add_env
import envs.temporal_add_obs as temporal_add_obs

class TangentADDEnv(add_env.ADDEnv):
    """ADD env that additionally exposes the data for the TangentADD tangent
    branch through info (observations, rewards, dones, and all discriminator
    data are inherited from ADDEnv unchanged, and no extra randomness is
    consumed, so rollouts are numerically identical to ADDEnv):

      tangent_cfg_err   [n, 3 + 3 + 3*(J-1)]  pre-step configuration error xi_t
      tangent_vel_resid [n, 3 + 3 + D]        post-step generalized velocity
                                              residual e^u_{t+1}

    Timing per action a_t:
      1. _pre_physics_step records xi_t (sim state and _ref_* both at time t).
      2. physics runs.
      3. _post_physics_step advances time and _update_misc moves _ref_* to t+1.
      4. _update_info records e^u_{t+1} (sim and ref both at t+1), before any
         env is reset; done envs are reset externally after the agent has
         recorded the step, so a transition never mixes states across a reset.

    The yaw anchor H_m is fixed per episode: the heading of the episode's
    motion at time 0 (looked up per motion at reset). It is never derived from
    the current root rotation, which is ill posed while inverted.
    """
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._motion_anchor_inv = None
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def get_tangent_cfg_group_dims(self):
        num_joints = self._kin_char_model.get_num_joints()
        return [3, 3, 3 * (num_joints - 1)]

    def get_tangent_vel_group_dims(self):
        dof_size = self._kin_char_model.get_dof_size()
        return [3, 3, dof_size]

    def _build_sim_tensors(self, env_config):
        super()._build_sim_tensors(env_config)

        num_envs = self.get_num_envs()
        cfg_dim = sum(self.get_tangent_cfg_group_dims())
        vel_dim = sum(self.get_tangent_vel_group_dims())
        self._tangent_cfg_err_buf = torch.zeros([num_envs, cfg_dim], device=self._device, dtype=torch.float32)
        self._tangent_vel_resid_buf = torch.zeros([num_envs, vel_dim], device=self._device, dtype=torch.float32)

        self._tangent_anchor_inv = torch.zeros([num_envs, 4], device=self._device, dtype=torch.float32)
        self._tangent_anchor_inv[..., 3] = 1.0
        return

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._info["tangent_cfg_err"] = self._tangent_cfg_err_buf
        self._info["tangent_vel_resid"] = self._tangent_vel_resid_buf
        return

    def _reset_char(self, env_ids):
        super()._reset_char(env_ids)
        self._update_motion_anchors(env_ids)
        return

    def _build_motion_anchors(self):
        num_motions = self._motion_lib.get_num_motions()
        motion_ids = torch.arange(num_motions, device=self._device, dtype=torch.long)
        motion_times = torch.zeros(num_motions, device=self._device, dtype=torch.float32)
        _, root_rot0, _, _, _, _ = self._motion_lib.calc_motion_frame(motion_ids, motion_times)
        self._motion_anchor_inv = temporal_add_obs.calc_motion_anchor_quat_inv(root_rot0)
        return

    def _update_motion_anchors(self, env_ids):
        if (self._motion_anchor_inv is None):
            # lazily built so it works regardless of init ordering; the motion
            # lib is guaranteed to exist by the first reset
            self._build_motion_anchors()
        self._tangent_anchor_inv[env_ids] = self._motion_anchor_inv[self._motion_ids[env_ids]]
        return

    def _pre_physics_step(self, actions):
        # _ref_* buffers still hold the reference state at the current time t
        self._update_tangent_cfg_err()
        super()._pre_physics_step(actions)
        return

    def _update_info(self, env_ids=None):
        super()._update_info(env_ids)
        if (env_ids is None):
            # post-physics path (and full resets): sim state and _ref_* are in
            # sync, so the residual is valid; the pre-step config error buffer
            # must NOT be touched here, it still holds xi_t for this transition
            self._update_tangent_vel_resid()
        elif (len(env_ids) > 0):
            # partial reset path: runs only after the agent has recorded the
            # transition; the char was just synced to the reference state, so
            # both quantities are recomputed to their (near zero) reset values
            self._update_tangent_cfg_err(env_ids)
            self._update_tangent_vel_resid(env_ids)
        return

    def _update_tangent_cfg_err(self, env_ids=None):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        dof_pos = self._engine.get_dof_pos(char_id)

        ref_root_pos = self._ref_root_pos
        ref_root_rot = self._ref_root_rot
        ref_joint_rot = self._ref_joint_rot
        anchor_inv = self._tangent_anchor_inv

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            dof_pos = dof_pos[env_ids]
            ref_root_pos = ref_root_pos[env_ids]
            ref_root_rot = ref_root_rot[env_ids]
            ref_joint_rot = ref_joint_rot[env_ids]
            anchor_inv = anchor_inv[env_ids]

        joint_rot = self._kin_char_model.dof_to_rot(dof_pos)

        cfg_err = temporal_add_obs.calc_config_error(anchor_inv=anchor_inv,
                                                     root_pos=root_pos,
                                                     root_rot=root_rot,
                                                     joint_rot=joint_rot,
                                                     ref_root_pos=ref_root_pos,
                                                     ref_root_rot=ref_root_rot,
                                                     ref_joint_rot=ref_joint_rot)
        if (env_ids is None):
            self._tangent_cfg_err_buf[:] = cfg_err
        else:
            self._tangent_cfg_err_buf[env_ids] = cfg_err
        return

    def _update_tangent_vel_resid(self, env_ids=None):
        char_id = self._get_char_id()
        root_vel = self._engine.get_root_vel(char_id)
        root_ang_vel = self._engine.get_root_ang_vel(char_id)
        dof_vel = self._engine.get_dof_vel(char_id)

        ref_root_vel = self._ref_root_vel
        ref_root_ang_vel = self._ref_root_ang_vel
        ref_dof_vel = self._ref_dof_vel
        anchor_inv = self._tangent_anchor_inv

        if (env_ids is not None):
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            dof_vel = dof_vel[env_ids]
            ref_root_vel = ref_root_vel[env_ids]
            ref_root_ang_vel = ref_root_ang_vel[env_ids]
            ref_dof_vel = ref_dof_vel[env_ids]
            anchor_inv = anchor_inv[env_ids]

        vel_resid = temporal_add_obs.calc_vel_residual(anchor_inv=anchor_inv,
                                                       root_vel=root_vel,
                                                       root_ang_vel=root_ang_vel,
                                                       dof_vel=dof_vel,
                                                       ref_root_vel=ref_root_vel,
                                                       ref_root_ang_vel=ref_root_ang_vel,
                                                       ref_dof_vel=ref_dof_vel)
        if (env_ids is None):
            self._tangent_vel_resid_buf[:] = vel_resid
        else:
            self._tangent_vel_resid_buf[env_ids] = vel_resid
        return
