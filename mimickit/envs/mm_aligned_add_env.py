import torch

import envs.aligned_add_env as aligned_add_env


def phase_to_control_bin(phase, num_phases):
    """Map normalized reference phase to a control-rate phase index."""
    if num_phases <= 0:
        raise ValueError("num_phases must be positive")

    phase_idx = torch.floor(phase * num_phases).to(dtype=torch.long)
    return torch.clamp(phase_idx, min=0, max=num_phases - 1)


class MMAlignedADDEnv(aligned_add_env.AlignedADDEnv):
    """Aligned ADD with a synchronized control-phase label for each step.

    The label is metadata only: it is neither appended to the policy input nor
    passed to the ADD discriminator.  It is computed from the same post-step
    reference clock used to build ``disc_obs_demo``.
    """

    def _load_motions(self, motion_file):
        super()._load_motions(motion_file)

        motion_lengths = self._motion_lib.get_motion_lengths()
        if motion_lengths.numel() == 0:
            raise ValueError("MM-ADD requires at least one reference motion")

        # A single shared phase partition is well-defined only when all clips
        # have the same duration.  The first experiment uses the single Roll
        # clip, so fail loudly instead of silently mixing unlike time axes.
        max_length = torch.max(motion_lengths)
        min_length = torch.min(motion_lengths)
        if not torch.isclose(max_length, min_length, atol=1e-6, rtol=1e-5):
            raise ValueError(
                "MM-ADD currently requires equal-duration reference clips")

        control_dt = float(self._engine.get_timestep())
        if control_dt <= 0.0:
            raise ValueError("engine timestep must be positive")
        self._mm_num_phases = max(
            1, int(round(float(max_length.item()) / control_dt)))
        return

    def get_mm_num_phases(self):
        return self._mm_num_phases

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._mm_phase_idx_buf = torch.zeros(
            self.get_num_envs(), device=self._device, dtype=torch.long)
        self._info["mm_phase_idx"] = self._mm_phase_idx_buf
        return

    def _update_observations(self, env_ids=None):
        # ADD updates disc_obs_demo here from the current reference clock.
        super()._update_observations(env_ids)
        if env_ids is None or len(env_ids) > 0:
            self._update_mm_phase_idx(env_ids)
        return

    def _update_mm_phase_idx(self, env_ids=None):
        if env_ids is None:
            motion_ids = self._motion_ids
        else:
            motion_ids = self._motion_ids[env_ids]

        motion_times = self._get_motion_times(env_ids)
        phase = self._motion_lib.calc_motion_phase(motion_ids, motion_times)
        phase_idx = phase_to_control_bin(phase, self._mm_num_phases)

        if env_ids is None:
            self._mm_phase_idx_buf.copy_(phase_idx)
        else:
            self._mm_phase_idx_buf[env_ids] = phase_idx
        return
