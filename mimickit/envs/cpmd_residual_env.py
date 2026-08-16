"""Residual-context CPMD environment built on an unchanged ADD differential.

The discriminator observation remains the stock ADD state observation.  Two
paired post-step tensors are exposed separately through ``info``:

``cpmd_hist_err``
    The causal 6 + dof reference-minus-simulation increment history.

``cpmd_ref_motion``
    Current reference root linear/angular velocity in the fixed phase-zero
    heading frame, followed by the reference dof velocity.

They are persistent, in-place-updated buffers and therefore stay synchronized
with ``disc_obs``/``disc_obs_demo`` when the agent records ``next_info``.
"""

import torch

import envs.add_env as add_env
import envs.cpmd_residual_obs as cpmd_residual_obs
import envs.static_objects_env as static_objects_env


class CPMDResidualEnv(add_env.ADDEnv):
    """ADD plus a separate 34-D error memory and 34-D reference context."""

    SCHEMA_VERSION = 2

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._cpmd_memory_seconds = float(env_config["cpmd_memory_seconds"])
        assert self._cpmd_memory_seconds > 0.0

        self._cpmd_schema_version = int(env_config["cpmd_schema_version"])
        assert self._cpmd_schema_version == self.SCHEMA_VERSION

        # Per-motion, per-episode fixed heading frame.  It is refreshed after
        # each new motion id is sampled and is never changed within an episode.
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
        return self._cpmd_error_memory.get_memory_decay()

    def get_cpmd_memory_seconds(self):
        return self._cpmd_memory_seconds

    def get_cpmd_mean_motion_length(self):
        return float(torch.mean(self._motion_lib.get_motion_lengths()).item())

    def get_cpmd_dof_dim(self):
        return self._kin_char_model.get_dof_size()

    def get_cpmd_history_dim(self):
        return self._cpmd_error_memory.get_history_dim()

    def get_cpmd_ref_motion_dim(self):
        return self._cpmd_ref_motion_buf.shape[-1]

    def get_cpmd_push_count(self):
        return self._cpmd_error_memory.get_push_count()

    def get_disc_state_obs_dim(self):
        """Width of the unchanged ADD discriminator input."""
        return self.get_disc_obs_space().shape[0]

    def _build_env(self, env_id, env_config):
        """Build the character and any optional fixed benchmark objects."""
        super()._build_env(env_id, env_config)
        if len(env_config.get("objects", [])) > 0:
            static_objects_env.StaticObjectsEnv._build_static_object(
                self, env_id, env_config)
        return

    def _build_disc_obs_buffers(self):
        num_envs = self.get_num_envs()
        rho = cpmd_residual_obs.calc_memory_decay(
            self._cpmd_memory_seconds, self._engine.get_timestep())

        self._cpmd_error_memory = cpmd_residual_obs.CPMDErrorMemory(
            num_envs=num_envs,
            kin_char_model=self._kin_char_model,
            rho=rho,
            device=self._device,
        )

        context_dim = 6 + self._kin_char_model.get_dof_size()
        self._cpmd_ref_motion_buf = torch.zeros(
            [num_envs, context_dim], dtype=torch.float32, device=self._device)

        # ADD owns the discriminator buffers and derives their pure state width.
        # Neither residual tensor is concatenated into this path.
        super()._build_disc_obs_buffers()
        return

    def _build_data_buffers(self):
        super()._build_data_buffers()

        # Both tensors keep their identity for the environment lifetime.  Agent
        # rollout recording copies each post-step row before any done-env reset.
        self._info["cpmd_hist_err"] = self._cpmd_error_memory.get_history()
        self._info["cpmd_ref_motion"] = self._cpmd_ref_motion_buf
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
        # DeepMimic has just initialized sim exactly to this random-phase
        # reference pose.  h starts at zero and both previous streams start here.
        self._cpmd_error_memory.reset(
            env_ids,
            self._ref_root_pos[env_ids],
            self._ref_root_rot[env_ids],
            self._ref_joint_rot[env_ids],
        )
        self._update_ref_motion_context(env_ids)
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
        # The parent first advances reference buffers to t+dt and pushes the ADD
        # simulated-state history.  Reading both streams afterwards pairs the
        # same post-physics control instant used by the ADD differential.
        super()._update_misc()

        self._cpmd_error_memory.push(
            *self._get_sim_cpmd_state(),
            *self._get_ref_cpmd_state(),
            self._motion_anchor_inv,
        )
        self._update_ref_motion_context()
        return

    def _update_ref_motion_context(self, env_ids=None):
        if env_ids is None:
            anchor_inv = self._motion_anchor_inv
            root_vel = self._ref_root_vel
            root_ang_vel = self._ref_root_ang_vel
            dof_vel = self._ref_dof_vel
        else:
            anchor_inv = self._motion_anchor_inv[env_ids]
            root_vel = self._ref_root_vel[env_ids]
            root_ang_vel = self._ref_root_ang_vel[env_ids]
            dof_vel = self._ref_dof_vel[env_ids]

        context = cpmd_residual_obs.compute_ref_motion_context(
            root_vel, root_ang_vel, dof_vel, anchor_inv)
        if env_ids is None:
            self._cpmd_ref_motion_buf.copy_(context)
        else:
            # Long-tensor indexing produces a temporary, so assign through the
            # original buffer to update only the reset environments in place.
            self._cpmd_ref_motion_buf[env_ids] = context
        return
