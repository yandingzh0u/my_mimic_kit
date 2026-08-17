"""Reference-conditioned metric environment for CPMD schema 2.

The discriminator observation is exactly the original ADD state observation.
CPMD additionally exposes an intrinsic representation of the synchronized
reference through ``info["cpmd_context"]``. The context is consumed only by
the discriminator-side metric and never enters the actor observation.
"""

import gymnasium.spaces as spaces
import numpy as np
import torch

import envs.add_env as add_env
import envs.cpmd_obs as cpmd_obs
import envs.static_objects_env as static_objects_env
import util.torch_util as torch_util


class CPMDEnv(add_env.ADDEnv):
    """ADD observations plus synchronized intrinsic reference context."""

    SCHEMA_VERSION = 2

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._cpmd_schema_version = int(env_config["cpmd_schema_version"])
        assert self._cpmd_schema_version == self.SCHEMA_VERSION
        if env_config.get("enable_tar_obs", False):
            raise ValueError(
                "CPMD schema 2 requires a reference-blind actor: "
                "enable_tar_obs must be false")
        if env_config.get("enable_phase_obs", False):
            raise ValueError(
                "CPMD schema 2 requires a reference-blind actor: "
                "enable_phase_obs must be false")

        self._disc_state_obs_dim = None
        self._cpmd_context_dim = None
        self._cpmd_context_buf = None

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

    def get_disc_state_obs_dim(self):
        assert self._disc_state_obs_dim is not None
        return self._disc_state_obs_dim

    def get_cpmd_context_dim(self):
        assert self._cpmd_context_dim is not None
        return self._cpmd_context_dim

    def get_cpmd_context_space(self):
        assert self._cpmd_context_buf is not None
        return spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=[self.get_cpmd_context_dim()],
            dtype=torch_util.torch_dtype_to_numpy(self._cpmd_context_buf.dtype),
        )

    def _build_env(self, env_id, env_config):
        """Build the character and optional fixed benchmark geometry."""
        super()._build_env(env_id, env_config)
        if len(env_config.get("objects", [])) > 0:
            static_objects_env.StaticObjectsEnv._build_static_object(
                self, env_id, env_config)
        return

    def _build_disc_obs_buffers(self):
        # This is deliberately the unmodified ADD discriminator observation.
        super()._build_disc_obs_buffers()
        self._disc_state_obs_dim = int(self._disc_obs_buf.shape[-1])

        # Use a deterministic valid motion frame to establish the intrinsic
        # context width without consuming the environment's reset RNG stream.
        motion_ids = torch.zeros([1], dtype=torch.long, device=self._device)
        motion_times = torch.zeros(
            [1], dtype=torch.float32, device=self._device)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = (
            self._motion_lib.calc_motion_frame(motion_ids, motion_times))
        body_pos, _ = self._kin_char_model.forward_kinematics(
            root_pos, root_rot, joint_rot)
        context = cpmd_obs.compute_intrinsic_context(
            root_pos, root_rot, root_vel, root_ang_vel,
            joint_rot, dof_vel, body_pos)

        self._cpmd_context_dim = int(context.shape[-1])
        self._cpmd_context_buf = torch.zeros(
            [self.get_num_envs(), self._cpmd_context_dim],
            dtype=context.dtype,
            device=self._device,
        )
        return

    def _build_data_buffers(self):
        super()._build_data_buffers()
        # Store the tensor itself in info. All later updates are in-place so
        # agents retain a stable reference to the live context buffer.
        self._info["cpmd_context"] = self._cpmd_context_buf
        return

    def _update_observations(self, env_ids=None):
        super()._update_observations(env_ids)
        if env_ids is None or len(env_ids) > 0:
            self._update_cpmd_context(env_ids)
        return

    def _update_cpmd_context(self, env_ids=None):
        if env_ids is None:
            root_pos = self._ref_root_pos
            root_rot = self._ref_root_rot
            root_vel = self._ref_root_vel
            root_ang_vel = self._ref_root_ang_vel
            joint_rot = self._ref_joint_rot
            dof_vel = self._ref_dof_vel
            body_pos = self._ref_body_pos
        else:
            root_pos = self._ref_root_pos[env_ids]
            root_rot = self._ref_root_rot[env_ids]
            root_vel = self._ref_root_vel[env_ids]
            root_ang_vel = self._ref_root_ang_vel[env_ids]
            joint_rot = self._ref_joint_rot[env_ids]
            dof_vel = self._ref_dof_vel[env_ids]
            body_pos = self._ref_body_pos[env_ids]

        context = cpmd_obs.compute_intrinsic_context(
            root_pos, root_rot, root_vel, root_ang_vel,
            joint_rot, dof_vel, body_pos)

        if env_ids is None:
            self._cpmd_context_buf.copy_(context)
        else:
            self._cpmd_context_buf[env_ids] = context
        return
