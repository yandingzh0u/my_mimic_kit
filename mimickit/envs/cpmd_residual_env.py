"""Bilinear CPMD environment built on the stock ADD differential.

The discriminator observation remains the 172-D ADD state differential. Two
synchronized post-step motion summaries are exposed separately through
``info``:

``cpmd_delta_motion``
    ``h = m_ref - m_sim``, the discounted motion mismatch.

``cpmd_sum_motion``
    ``s = m_ref + m_sim``, the corresponding common motion.

Both tensors are persistent in-place buffers and are recorded from the same
post-physics instant as the ADD differential.
"""

import torch

import envs.add_env as add_env
import envs.cpmd_residual_obs as cpmd_residual_obs
import envs.static_objects_env as static_objects_env


class CPMDResidualEnv(add_env.ADDEnv):
    """ADD plus paired 34-D difference and common-motion memories."""

    SCHEMA_VERSION = 3

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._cpmd_memory_seconds = float(env_config["cpmd_memory_seconds"])
        assert self._cpmd_memory_seconds > 0.0

        self._cpmd_schema_version = int(env_config["cpmd_schema_version"])
        assert self._cpmd_schema_version == self.SCHEMA_VERSION

        self._motion_anchor_inv = torch.zeros(
            [num_envs, 4], dtype=torch.float32, device=device)
        self._motion_anchor_inv[..., 3] = 1.0

        super().__init__(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )
        return

    def get_cpmd_schema_version(self):
        return self._cpmd_schema_version

    def get_cpmd_memory_decay(self):
        return self._cpmd_motion_memory.get_memory_decay()

    def get_cpmd_memory_seconds(self):
        return self._cpmd_memory_seconds

    def get_cpmd_mean_motion_length(self):
        return float(torch.mean(self._motion_lib.get_motion_lengths()).item())

    def get_cpmd_dof_dim(self):
        return self._kin_char_model.get_dof_size()

    def get_cpmd_motion_dim(self):
        return self._cpmd_motion_memory.get_motion_dim()

    def get_cpmd_history_dim(self):
        """Compatibility alias used by older tooling."""
        return self.get_cpmd_motion_dim()

    def get_cpmd_sum_motion_dim(self):
        return self.get_cpmd_motion_dim()

    def get_cpmd_push_count(self):
        return self._cpmd_motion_memory.get_push_count()

    def get_disc_state_obs_dim(self):
        return self.get_disc_obs_space().shape[0]

    def _build_env(self, env_id, env_config):
        super()._build_env(env_id, env_config)
        if len(env_config.get("objects", [])) > 0:
            static_objects_env.StaticObjectsEnv._build_static_object(
                self, env_id, env_config)
        return

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        rho = cpmd_residual_obs.calc_memory_decay(
            self._cpmd_memory_seconds, self._engine.get_timestep())

        self._cpmd_motion_memory = cpmd_residual_obs.CPMDErrorMemory(
            num_envs=num_envs,
            kin_char_model=self._kin_char_model,
            rho=rho,
            device=self._device,
        )

        # ADD owns the discriminator buffers and keeps their pure state width.
        super()._build_disc_obs_buffers()
        return

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._info["cpmd_delta_motion"] = (
            self._cpmd_motion_memory.get_delta_motion())
        self._info["cpmd_sum_motion"] = (
            self._cpmd_motion_memory.get_sum_motion())
        return

    def _reset_char(self, env_ids):
        super()._reset_char(env_ids)
        self._update_motion_anchors(env_ids)
        return

    def _update_motion_anchors(self, env_ids):
        motion_ids = self._motion_ids[env_ids]
        motion_times0 = torch.zeros_like(self._get_motion_times(env_ids))
        _, root_rot0, _, _, _, _ = self._motion_lib.calc_motion_frame(
            motion_ids, motion_times0)
        self._motion_anchor_inv[env_ids] = (
            cpmd_residual_obs.calc_motion_anchor_quat_inv(root_rot0))
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_cpmd(env_ids)
        return

    def _reset_cpmd(self, env_ids):
        # Sim and reference have just been initialized to the same random phase.
        self._cpmd_motion_memory.reset(
            env_ids,
            self._ref_root_pos[env_ids],
            self._ref_root_rot[env_ids],
            self._ref_joint_rot[env_ids],
        )
        return

    def _get_sim_cpmd_state(self):
        char_id = self._get_char_id()
        root_pos = self._engine.get_root_pos(char_id)
        root_rot = self._engine.get_root_rot(char_id)
        joint_rot = self._kin_char_model.dof_to_rot(
            self._engine.get_dof_pos(char_id))
        return root_pos, root_rot, joint_rot

    def _get_ref_cpmd_state(self):
        return self._ref_root_pos, self._ref_root_rot, self._ref_joint_rot

    def _update_misc(self):
        # Parent first advances the reference to t+dt and pushes ADD state.
        super()._update_misc()
        self._cpmd_motion_memory.push(
            *self._get_sim_cpmd_state(),
            *self._get_ref_cpmd_state(),
            self._motion_anchor_inv,
        )
        return
