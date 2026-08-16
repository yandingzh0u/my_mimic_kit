import gymnasium.spaces as spaces
import numpy as np
import torch

import envs.add_env as add_env
import envs.cpmd_obs as cpmd_obs
import envs.static_objects_env as static_objects_env
import util.torch_util as torch_util

class CPMDEnv(add_env.ADDEnv):
    """ADD with a context-preserving motion differential.

    The discriminator differential is

        Delta_t = [ o^ref_t - o^sim_t,
                    m^ref_t - m^sim_t,
                    c^ref_t - c^sim_t ]

    where o is the unchanged ADD state observation, m is the causal motion
    summary and c its pairwise context interactions (see cpmd_obs). Both extra
    blocks are produced every control step for both sides and concatenated into
    a single differential; they are not separate stages and not separate
    networks.

    Everything on the training side is stock ADD: one discriminator, one
    zero-vs-policy BCE, one softplus reward. The env only changes what the
    differential contains.

    The reference side is streamed from the env's _ref_* buffers, at the same
    rate and through the same operator as the simulated side, so the two are
    paired step for step. That pairing is the reason there is no way to
    synthesize a reference differential from a randomly sampled motion frame:
    the summary depends on how long the current episode has been running.
    fetch_disc_obs_demo is therefore disabled rather than left to return a
    misleading placeholder.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._cpmd_memory_seconds = float(env_config["cpmd_memory_seconds"])
        assert self._cpmd_memory_seconds > 0.0

        self._cpmd_schema_version = int(env_config.get("cpmd_schema_version", 1))
        assert self._cpmd_schema_version == 1

        # per-episode fixed heading anchor C_0 (identity until the first reset)
        self._motion_anchor_inv = torch.zeros([num_envs, 4], dtype=torch.float32, device=device)
        self._motion_anchor_inv[..., 3] = 1.0

        self._disc_state_obs_dim = None

        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def get_cpmd_memory_decay(self):
        return self._sim_cpmd.get_memory_decay()

    def get_cpmd_memory_seconds(self):
        return self._cpmd_memory_seconds

    def get_cpmd_mean_motion_length(self):
        return float(torch.mean(self._motion_lib.get_motion_lengths()).item())

    def get_cpmd_dof_dim(self):
        return self._kin_char_model.get_dof_size()

    def get_disc_state_obs_dim(self):
        assert self._disc_state_obs_dim is not None
        return self._disc_state_obs_dim

    def get_cpmd_summary_dim(self):
        return self._sim_cpmd.get_summary_dim()

    def get_cpmd_interaction_dim(self):
        return self._sim_cpmd.get_interaction_dim()

    def get_cpmd_push_counts(self):
        return self._sim_cpmd.get_push_count(), self._ref_cpmd.get_push_count()

    def _build_env(self, env_id, env_config):
        """Build the character and optional fixed task geometry.

        Most motion clips only need the ground plane.  The ADD benchmark's
        Double Kong and Climb clips, however, are defined relative to fixed
        boxes.  Reusing StaticObjectsEnv's object builder keeps those two
        tasks on the same CPMD observation and learning path as every other
        skill; the objects only change the simulated scene.
        """
        super()._build_env(env_id, env_config)
        if len(env_config.get("objects", [])) > 0:
            static_objects_env.StaticObjectsEnv._build_static_object(
                self, env_id, env_config)
        return

    def get_disc_obs_space(self):
        """Shape is derived from the block widths, not by sampling a demo.

        The base class builds this space by calling fetch_disc_obs_demo, which
        CPMD cannot answer; the state width is therefore taken from the ADD
        state observation directly and the two CPMD blocks are added to it.
        """
        if (self._disc_state_obs_dim is None):
            motion_ids, motion_times0 = self._sample_motion_times(1)
            state_obs = add_env.ADDEnv._compute_disc_obs_demo(self, motion_ids, motion_times0)
            self._disc_state_obs_dim = state_obs.shape[-1]
            self._disc_obs_dtype = state_obs.dtype

        total = self._disc_state_obs_dim + self._sim_cpmd.get_obs_dim()
        return spaces.Box(low=-np.inf, high=np.inf, shape=[total],
                          dtype=torch_util.torch_dtype_to_numpy(self._disc_obs_dtype))

    def fetch_disc_obs_demo(self, num_samples):
        raise RuntimeError(
            "CPMD has no standalone reference differential: the motion summary "
            "m and its interactions c depend on how long the current episode "
            "has run, so they cannot be synthesized from a randomly sampled "
            "motion frame. Use the paired differential produced during the "
            "rollout (info['disc_obs_demo'] - info['disc_obs']), which streams "
            "both sides through the same operator at the same rate.")

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        raise RuntimeError(
            "CPMD cannot compute a reference differential from sampled motion "
            "times; see fetch_disc_obs_demo. The ADD state block alone is "
            "available through add_env.ADDEnv._compute_disc_obs_demo.")

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        rho = cpmd_obs.calc_memory_decay(self._cpmd_memory_seconds,
                                         self._engine.get_timestep())

        self._sim_cpmd = cpmd_obs.CPMDHistory(num_envs=num_envs,
                                              kin_char_model=self._kin_char_model,
                                              rho=rho, device=self._device)
        self._ref_cpmd = cpmd_obs.CPMDHistory(num_envs=num_envs,
                                              kin_char_model=self._kin_char_model,
                                              rho=rho, device=self._device)

        # sizes _disc_obs_buf and _disc_obs_demo_buf via get_disc_obs_space
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
        self._motion_anchor_inv[env_ids] = cpmd_obs.calc_motion_anchor_quat_inv(root_rot0)
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if (len(env_ids) > 0):
            self._reset_cpmd(env_ids)
        return

    def _reset_cpmd(self, env_ids):
        # The character is initialized exactly to the reference frame, so both
        # sides start from the same pose with a zero summary: the differential
        # is exactly zero at reset and nothing crosses episodes. Partial resets
        # only touch their own envs.
        root_pos, root_rot, joint_rot = self._get_ref_cpmd_state()

        args = (env_ids, root_pos[env_ids], root_rot[env_ids], joint_rot[env_ids])
        self._sim_cpmd.reset(*args)
        self._ref_cpmd.reset(*args)
        return

    def _get_sim_cpmd_state(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        joint_rot = self._kin_char_model.dof_to_rot(self._engine.get_dof_pos(char_id))
        return root_pos, root_rot, joint_rot

    def _get_ref_cpmd_state(self):
        return self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot

    def _update_misc(self):
        # super() advances physics bookkeeping, the reference motion and the
        # ADD history to t+1; it runs post-physics only, so pushing here gives
        # each side exactly one increment per control step
        super()._update_misc()
        anchor_inv = self._motion_anchor_inv
        self._sim_cpmd.push(*self._get_sim_cpmd_state(), anchor_inv)
        self._ref_cpmd.push(*self._get_ref_cpmd_state(), anchor_inv)
        return

    def _update_disc_obs(self, env_ids=None):
        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        body_pos = self._disc_hist_body_pos.get_all()

        cpmd_block = self._sim_cpmd.extract()

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            joint_rot = joint_rot[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]
            cpmd_block = cpmd_block[env_ids]

        state_obs = add_env.compute_disc_obs(root_pos=root_pos,
                                             root_rot=root_rot,
                                             root_vel=root_vel,
                                             root_ang_vel=root_ang_vel,
                                             joint_rot=joint_rot,
                                             dof_vel=dof_vel,
                                             body_pos=body_pos,
                                             global_obs=self._global_obs)

        disc_obs = torch.cat([state_obs, cpmd_block], dim=-1)

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

        # state block from the motion lib, CPMD blocks from the streamed
        # reference operator (same anchor and same recursion as the sim side)
        state_obs = add_env.ADDEnv._compute_disc_obs_demo(self, motion_ids, motion_times0)

        cpmd_block = self._ref_cpmd.extract()
        if (env_ids is not None):
            cpmd_block = cpmd_block[env_ids]

        disc_obs = torch.cat([state_obs, cpmd_block], dim=-1)

        if (env_ids is None):
            self._disc_obs_demo_buf[:] = disc_obs
        else:
            self._disc_obs_demo_buf[env_ids] = disc_obs
        return
