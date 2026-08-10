from __future__ import annotations

from copy import deepcopy

import torch

import envs.task_location_env as task_location_env
from learning.skill_conditioned_runtime import (
    build_dataset_manifest,
    context_times,
    rsi_context_starts,
    resolve_manifest_clip,
)
from learning.skill_encoder.motion_features import build_motion_dynamic_features


class SkillConditionedLocationEnv(task_location_env.TaskLocationEnv):
    """Location task exposing a read-only, mocap-backed R2 skill context API."""

    NAME = "skill_conditioned_location"

    def __init__(self, env_config, engine_config, num_envs, device, visualize, record_video=False):
        self._skill_motion_file = env_config["motion_file"]
        self._skill_command = None
        self._skill_dataset_manifest_cache = None
        super().__init__(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )

    def get_skill_dataset_manifest(self):
        if self._skill_dataset_manifest_cache is None:
            files = [
                self._motion_lib.get_motion_file(i)
                for i in range(self._motion_lib.get_num_motions())
            ]
            lengths = self._motion_lib.get_motion_lengths().detach().cpu().tolist()
            self._skill_dataset_manifest_cache = build_dataset_manifest(
                self._skill_motion_file, files, lengths
            )
        return deepcopy(self._skill_dataset_manifest_cache)

    @torch.no_grad()
    def get_expert_skill_context(
        self,
        *,
        motion_path=None,
        clip_sha256=None,
        context_start_sec: float,
        steps=20,
        feature_schema=None,
    ):
        """Encode-ready features for one manifest-backed expert context.

        This is intentionally a read-only evaluation API: callers identify an
        existing expert clip and a valid A0 start, rather than supplying an
        arbitrary latent command.
        """
        if feature_schema is None:
            raise ValueError("feature_schema is required for exact runtime feature parity")
        steps = int(steps)
        if steps != 20:
            raise ValueError("R2 expert contexts require exactly 20 feature frames")
        clip, start = self._resolve_skill_context(
            motion_path=motion_path,
            clip_sha256=clip_sha256,
            context_start_sec=context_start_sec,
            steps=steps,
        )
        motion_ids = torch.tensor(
            [int(clip["motion_id"])], device=self._device, dtype=torch.long
        )
        starts = torch.tensor([start], device=self._device, dtype=torch.float32)
        features = self._build_skill_features(
            motion_ids, starts, steps, feature_schema
        )
        return {
            "features": features,
            "motion_id": int(clip["motion_id"]),
            "motion_path": clip["file"],
            "clip_sha256": clip["sha256"],
            "context_start_sec": start,
        }

    @torch.no_grad()
    def get_skill_evaluation_state(self, env_ids=None):
        """Return a detached snapshot of policy-relevant physical/task state."""
        if env_ids is None:
            env_ids = torch.arange(self.get_num_envs(), device=self._device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long).flatten()
        char_id = self._get_char_id()

        def select(value):
            return value[env_ids].detach().clone()

        return {
            "root_pos": select(self._engine.get_root_pos(char_id)),
            "root_rot": select(self._engine.get_root_rot(char_id)),
            "root_vel": select(self._engine.get_root_vel(char_id)),
            "root_ang_vel": select(self._engine.get_root_ang_vel(char_id)),
            "dof_pos": select(self._engine.get_dof_pos(char_id)),
            "dof_vel": select(self._engine.get_dof_vel(char_id)),
            "body_pos": select(self._engine.get_body_pos(char_id)),
            "body_rot": select(self._engine.get_body_rot(char_id)),
            "body_vel": select(self._engine.get_body_vel(char_id)),
            "body_ang_vel": select(self._engine.get_body_ang_vel(char_id)),
            "target_pos": select(self._tar_pos),
            "target_change_time": select(self._tar_change_times),
            "motion_id": select(self._motion_ids),
            "motion_time_offset": select(self._motion_time_offsets),
            "observation": select(self._obs_buf),
            "disc_observation": select(self._disc_obs_buf),
            "timestep": select(self._timestep_buf),
            "time": select(self._time_buf),
        }

    @torch.no_grad()
    def get_skill_reset_context(self, env_ids=None, steps=20, feature_schema=None):
        if feature_schema is None:
            raise ValueError("feature_schema is required for exact runtime feature parity")
        if env_ids is None:
            env_ids = torch.arange(self.get_num_envs(), device=self._device, dtype=torch.long)
        else:
            env_ids = torch.as_tensor(env_ids, device=self._device, dtype=torch.long).flatten()
        motion_ids = self._motion_ids[env_ids]
        reset_times = self._get_motion_times(env_ids)
        lengths = self._motion_lib.get_motion_length(motion_ids)
        starts = rsi_context_starts(
            reset_times, lengths, steps=int(steps), control_freq=self._skill_control_freq()
        )
        features = self._build_skill_features(motion_ids, starts, int(steps), feature_schema)
        return {
            "features": features,
            "motion_ids": motion_ids.detach().clone(),
            "reset_times": reset_times.detach().clone(),
            "context_start_sec": starts.detach().clone(),
        }

    def set_skill_command(
        self, *, motion_path=None, clip_sha256=None, context_start_sec: float
    ):
        manifest = self.get_skill_dataset_manifest()
        clip = resolve_manifest_clip(
            manifest, motion_path=motion_path, clip_sha256=clip_sha256
        )
        start = float(context_start_sec)
        max_start = float(clip["length_seconds"]) - 20.0 / 30.0
        if not (0.0 <= start <= max_start + 1e-8):
            raise ValueError("context_start_sec is outside the valid 20-frame interval")
        self._skill_command = {
            "motion_id": int(clip["motion_id"]),
            "context_start_sec": min(start, max_start),
            "clip_sha256": clip["sha256"],
        }
        return dict(self._skill_command)

    def clear_skill_command(self):
        self._skill_command = None

    def _resolve_skill_context(
        self, *, motion_path=None, clip_sha256=None, context_start_sec: float, steps: int
    ):
        manifest = self.get_skill_dataset_manifest()
        clip = resolve_manifest_clip(
            manifest, motion_path=motion_path, clip_sha256=clip_sha256
        )
        control_freq = self._skill_control_freq()
        start = float(context_start_sec)
        max_start = float(clip["length_seconds"]) - int(steps) / float(control_freq)
        if not (0.0 <= start <= max_start + 1e-8):
            raise ValueError(
                "context_start_sec is outside the valid {}-frame interval".format(steps)
            )
        return clip, min(start, max_start)

    def _sample_motion_times(self, n):
        if self._skill_command is not None:
            motion_ids = torch.full(
                (n,), self._skill_command["motion_id"], device=self._device, dtype=torch.long
            )
            reset_time = self._skill_command["context_start_sec"] + 19.0 / 30.0
            return motion_ids, torch.full(
                (n,), reset_time, device=self._device, dtype=torch.float32
            )

        lengths = self._motion_lib.get_motion_lengths()
        eligible = lengths >= 20.0 / 30.0
        weights = self._motion_lib.get_motion_weights() * eligible.to(torch.float32)
        if weights.sum() <= 0:
            raise RuntimeError("dataset has no clips eligible for an A0..A20 context")
        motion_ids = torch.multinomial(weights, num_samples=n, replacement=True)
        max_starts = lengths[motion_ids] - 20.0 / 30.0
        if self._rand_reset:
            starts = torch.rand(n, device=self._device) * max_starts
        else:
            starts = torch.zeros(n, device=self._device)
        return motion_ids, starts + 19.0 / 30.0

    def _skill_control_freq(self):
        frequency = int(round(1.0 / self._engine.get_timestep()))
        if frequency != 30:
            raise ValueError("R2 skill contexts require a 30 Hz control frequency")
        return frequency

    def _validate_feature_schema(self, schema):
        if int(schema.get("feature_dim", -1)) != 44:
            raise ValueError("R2 encoder feature schema must have feature_dim=44")
        names = list(schema.get("foot_body_names", []))
        ids = list(schema.get("foot_body_ids", []))
        if len(names) != 2 or len(ids) != 2:
            raise ValueError("feature schema must identify exactly two feet")
        runtime_ids = [self._kin_char_model.get_body_id(name) for name in names]
        if runtime_ids != ids:
            raise ValueError(
                "feature schema foot ids do not match the runtime character: {} != {}".format(
                    ids, runtime_ids
                )
            )
        proxy = schema.get("contact_proxy")
        if not isinstance(proxy, dict):
            raise ValueError("feature schema is missing frozen contact thresholds")
        for key in ("ground_height", "height_threshold", "speed_threshold"):
            if key not in proxy:
                raise ValueError("feature schema contact_proxy is missing {}".format(key))
        return ids, proxy

    def _build_skill_features(self, motion_ids, starts, steps, schema):
        foot_ids, proxy = self._validate_feature_schema(schema)
        times = context_times(starts, steps=steps + 1, control_freq=30)
        tiled_ids = motion_ids.unsqueeze(1).expand(-1, steps + 1).reshape(-1)
        states = self._motion_lib.calc_motion_frame(tiled_ids, times.reshape(-1))
        batch = motion_ids.shape[0]
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = states
        root_pos = root_pos.reshape(batch, steps + 1, -1)
        root_rot = root_rot.reshape(batch, steps + 1, -1)
        root_vel = root_vel.reshape(batch, steps + 1, -1)
        root_ang_vel = root_ang_vel.reshape(batch, steps + 1, -1)
        joint_rot = joint_rot.reshape(batch, steps + 1, *joint_rot.shape[-2:])
        dof_vel = dof_vel.reshape(batch, steps + 1, -1)
        body_pos, _ = self._kin_char_model.forward_kinematics(
            root_pos.reshape(-1, 3),
            root_rot.reshape(-1, 4),
            joint_rot.reshape(-1, *joint_rot.shape[-2:]),
        )
        body_pos = body_pos.reshape(batch, steps + 1, body_pos.shape[-2], 3)
        return build_motion_dynamic_features(
            root_rot=root_rot[:, :steps],
            root_vel=root_vel[:, :steps],
            root_ang_vel=root_ang_vel[:, :steps],
            dof_vel=dof_vel[:, :steps],
            foot_pos=body_pos[:, :, foot_ids, :],
            timestep=1.0 / 30.0,
            ground_height=float(proxy["ground_height"]),
            contact_height_threshold=float(proxy["height_threshold"]),
            contact_speed_threshold=float(proxy["speed_threshold"]),
        )
