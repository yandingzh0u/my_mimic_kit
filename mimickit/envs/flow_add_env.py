import torch

import envs.add_env as add_env

class FlowADDEnv(add_env.ADDEnv):
    """ADD env that also exposes the previous step's discriminator observations,
    so the agent can form the differential flow v_t = delta_t - delta_t-1.

    The prev buffers are snapshotted right before each physics step and synced
    to the current observations on reset, so v is always a within-episode
    difference (the first step of an episode uses the reset-state differential
    as its previous frame).
    """
    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
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
