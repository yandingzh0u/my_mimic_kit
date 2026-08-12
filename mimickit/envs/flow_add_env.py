import torch

import envs.add_env as add_env
import envs.flow_add_disc_obs as flow_add_disc_obs

class FlowADDEnv(add_env.ADDEnv):
    """ADD env that also exposes the previous step's discriminator observations,
    so the agent can form the differential flow v_t = delta_t - delta_t-1.

    The prev buffers are snapshotted right before each physics step and synced
    to the current observations on reset, so v is always a within-episode
    difference (the first step of an episode uses the reset-state differential
    as its previous frame).

    With disc_ref_heading_frame = True, the disc features of both the agent
    and the demo are expressed in a common frame anchored at the reference
    motion's current root (heading + translation). The differential is then
    invariant to a global yaw/translation of the scene, which removes the
    world-rotation confound from the circulation term while keeping the full
    relative tracking error (including heading error). Intended as a variant
    of global_obs = True; the feature dimensions are unchanged.
    """
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._disc_ref_heading_frame = env_config.get("disc_ref_heading_frame", False)
        super().__init__(env_config=env_config, engine_config=engine_config,
                         num_envs=num_envs, device=device, visualize=visualize,
                         record_video=record_video)
        return

    def reset(self, env_ids=None):
        obs, info = super().reset(env_ids)

        if (env_ids is None):
            self._disc_obs_prev_buf[:] = self._disc_obs_buf
            self._disc_obs_demo_prev_buf[:] = self._disc_obs_demo_buf
        elif (len(env_ids) > 0):
            self._disc_obs_prev_buf[env_ids] = self._disc_obs_buf[env_ids]
            self._disc_obs_demo_prev_buf[env_ids] = self._disc_obs_demo_buf[env_ids]

        return obs, info

    def _build_disc_obs_buffers(self):
        super()._build_disc_obs_buffers()
        self._disc_obs_prev_buf = torch.zeros_like(self._disc_obs_buf)
        self._disc_obs_demo_prev_buf = torch.zeros_like(self._disc_obs_demo_buf)
        return

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._info["disc_obs_prev"] = self._disc_obs_prev_buf
        self._info["disc_obs_demo_prev"] = self._disc_obs_demo_prev_buf
        return

    def _pre_physics_step(self, actions):
        self._disc_obs_prev_buf[:] = self._disc_obs_buf
        self._disc_obs_demo_prev_buf[:] = self._disc_obs_demo_buf
        super()._pre_physics_step(actions)
        return

    def _update_disc_obs(self, env_ids=None):
        if (not self._disc_ref_heading_frame):
            super()._update_disc_obs(env_ids)
            return

        root_pos = self._disc_hist_root_pos.get_all()
        root_rot = self._disc_hist_root_rot.get_all()
        root_vel = self._disc_hist_root_vel.get_all()
        root_ang_vel = self._disc_hist_root_ang_vel.get_all()
        joint_rot = self._disc_hist_joint_rot.get_all()
        dof_vel = self._disc_hist_dof_vel.get_all()
        body_pos = self._disc_hist_body_pos.get_all()

        # reference frame: the demo's root at the current motion time,
        # updated before observations in both the step and reset paths
        ref_root_pos = self._ref_root_pos
        ref_root_rot = self._ref_root_rot

        if (env_ids is not None):
            root_pos = root_pos[env_ids]
            root_rot = root_rot[env_ids]
            root_vel = root_vel[env_ids]
            root_ang_vel = root_ang_vel[env_ids]
            joint_rot = joint_rot[env_ids]
            dof_vel = dof_vel[env_ids]
            body_pos = body_pos[env_ids]
            ref_root_pos = ref_root_pos[env_ids]
            ref_root_rot = ref_root_rot[env_ids]

        disc_obs = flow_add_disc_obs.compute_ref_frame_disc_obs(
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            body_pos=body_pos)

        if (env_ids is None):
            self._disc_obs_buf[:] = disc_obs
        else:
            self._disc_obs_buf[env_ids] = disc_obs

        return

    def _compute_disc_obs_demo(self, motion_ids, motion_times0):
        if (not self._disc_ref_heading_frame):
            return super()._compute_disc_obs_demo(motion_ids, motion_times0)

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = \
            self._fetch_disc_demo_data(motion_ids, motion_times0)

        # same frame as the agent side: the demo's own root at the current
        # time, which is the last (most recent) history step
        ref_root_pos = root_pos[..., -1, :]
        ref_root_rot = root_rot[..., -1, :]

        disc_obs = flow_add_disc_obs.compute_ref_frame_disc_obs(
            ref_root_pos=ref_root_pos,
            ref_root_rot=ref_root_rot,
            root_pos=root_pos,
            root_rot=root_rot,
            root_vel=root_vel,
            root_ang_vel=root_ang_vel,
            joint_rot=joint_rot,
            dof_vel=dof_vel,
            body_pos=body_pos)
        return disc_obs
