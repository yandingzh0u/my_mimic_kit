"""Context-paired ADD environment.

The ADD discriminator observation remains unchanged.  The environment exposes
two synchronized motion tensors through ``info``: an episode-local tracking
error memory and a deterministic reference context indexed by motion phase.
"""

import torch

import envs.add_env as add_env
import envs.cpmd_cond_obs as cpmd_cond_obs
import envs.static_objects_env as static_objects_env


class CPMDConditionalEnv(add_env.ADDEnv):
    """ADD plus paired error memory and phase-consistent reference context."""

    SCHEMA_VERSION = 1

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._cpmd_memory_seconds = float(env_config["cpmd_memory_seconds"])
        assert self._cpmd_memory_seconds > 0.0
        self._cpmd_schema_version = int(env_config["cpmd_schema_version"])
        assert self._cpmd_schema_version == self.SCHEMA_VERSION
        self._cpmd_context_grid_size = int(
            env_config.get("cpmd_context_grid_size", 1024))
        self._cpmd_context_tail_tolerance = float(
            env_config.get("cpmd_context_tail_tolerance", 1e-6))

        self._motion_anchor_inv = torch.zeros(
            [num_envs, 4], dtype=torch.float32, device=device)
        self._motion_anchor_inv[..., 3] = 1.0
        self._context_lookup_rmse = torch.zeros(
            [], dtype=torch.float32, device=device)
        self._context_lookup_max = torch.zeros(
            [], dtype=torch.float32, device=device)

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
        return self._cpmd_memory.get_memory_decay()

    def get_cpmd_memory_seconds(self):
        return self._cpmd_memory_seconds

    def get_cpmd_motion_dim(self):
        return self._cpmd_memory.get_motion_dim()

    def get_cpmd_error_dim(self):
        return self.get_cpmd_motion_dim()

    def get_cpmd_context_dim(self):
        return self.get_cpmd_motion_dim()

    def get_disc_state_obs_dim(self):
        return self.get_disc_obs_space().shape[0]

    def get_cpmd_context_grid_size(self):
        return self._phase_context.get_grid_size()

    def get_cpmd_context_tail_steps(self):
        return self._phase_context.get_tail_steps()

    def get_cpmd_push_count(self):
        return self._cpmd_memory.get_push_count()

    def get_cpmd_phase_context(self, motion_ids, motion_times):
        return self._phase_context.lookup(motion_ids, motion_times)

    def _build_env(self, env_id, env_config):
        super()._build_env(env_id, env_config)
        if len(env_config.get("objects", [])) > 0:
            static_objects_env.StaticObjectsEnv._build_static_object(
                self, env_id, env_config)
        return

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        timestep = self._engine.get_timestep()
        rho = cpmd_cond_obs.calc_memory_decay(
            self._cpmd_memory_seconds, timestep)

        self._phase_context = cpmd_cond_obs.PhaseReferenceContext(
            motion_lib=self._motion_lib,
            kin_char_model=self._kin_char_model,
            rho=rho,
            timestep=timestep,
            grid_size=self._cpmd_context_grid_size,
            tail_tolerance=self._cpmd_context_tail_tolerance,
            device=self._device,
        )
        self._cpmd_memory = cpmd_cond_obs.CPMDConditionalMemory(
            num_envs=num_envs,
            kin_char_model=self._kin_char_model,
            rho=rho,
            device=self._device,
        )

        # The discriminator observation itself is the unmodified ADD state.
        super()._build_disc_obs_buffers()
        return

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._info["cpmd_error_memory"] = (
            self._cpmd_memory.get_error_memory())
        self._info["cpmd_ref_context"] = self._cpmd_memory.get_ref_context()
        return

    def record_diagnostics(self):
        self._diagnostics["cpmd_context_lookup_rmse"] = (
            self._context_lookup_rmse)
        self._diagnostics["cpmd_context_lookup_max"] = (
            self._context_lookup_max)
        return super().record_diagnostics()

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
            cpmd_cond_obs.calc_motion_anchor_quat_inv(root_rot0))
        return

    def _reset_envs(self, env_ids):
        super()._reset_envs(env_ids)
        if len(env_ids) > 0:
            self._reset_cpmd(env_ids)
        return

    def _reset_cpmd(self, env_ids):
        motion_ids = self._motion_ids[env_ids]
        motion_times = self._get_motion_times(env_ids)
        context = self._phase_context.lookup(motion_ids, motion_times)
        self._cpmd_memory.reset(
            env_ids,
            self._ref_root_pos[env_ids],
            self._ref_root_rot[env_ids],
            self._ref_joint_rot[env_ids],
            context,
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
        # The parent first advances the reference and ADD history to t + dt.
        super()._update_misc()
        self._cpmd_memory.push(
            *self._get_sim_cpmd_state(),
            *self._get_ref_cpmd_state(),
            self._motion_anchor_inv,
        )

        # The live recurrence and phase lookup must describe the same reference
        # context.  The interpolation residual is logged, not used for reward.
        with torch.no_grad():
            expected = self._phase_context.lookup(
                self._motion_ids, self._get_motion_times())
            error = self._cpmd_memory.get_ref_context() - expected
            self._context_lookup_rmse.copy_(
                torch.sqrt(torch.mean(torch.square(error))))
            self._context_lookup_max.copy_(torch.max(torch.abs(error)))
        return
