import math
import numbers

import torch

import envs.add_env as add_env


def phase_to_control_bin(phase, num_phases):
    """Map normalized reference phase to a control-rate phase index."""
    if num_phases <= 0:
        raise ValueError("num_phases must be positive")

    phase_idx = torch.floor(phase * num_phases).to(dtype=torch.long)
    return torch.clamp(phase_idx, min=0, max=num_phases - 1)


def resolve_plot_num_phases(config_value, motion_length, control_dt):
    """Resolve an explicit phase count or a deterministic automatic count.

    Python's :func:`round` uses ties-to-even and a CUDA-produced float can land
    on the opposite side of a half integer from the same CPU computation.  The
    automatic mode therefore snaps numerical half ties and uses round-half-up.
    Locked experiments should still specify the integer explicitly so the
    discretization is visible in the saved environment configuration.
    """
    if isinstance(config_value, numbers.Integral) and not isinstance(
            config_value, bool):
        num_phases = int(config_value)
        if num_phases <= 0:
            raise ValueError("plot_num_phases must be positive")
        return num_phases, False

    if config_value not in (None, "auto"):
        raise ValueError(
            "plot_num_phases must be a positive integer or 'auto'")
    if motion_length <= 0.0:
        raise ValueError("motion_length must be positive")
    if control_dt <= 0.0:
        raise ValueError("control_dt must be positive")

    control_intervals = float(motion_length) / float(control_dt)
    lower = math.floor(control_intervals)
    half_tie = lower + 0.5
    if math.isclose(control_intervals, half_tie, rel_tol=0.0, abs_tol=1e-5):
        control_intervals = half_tie
    num_phases = max(1, math.floor(control_intervals + 0.5))
    return num_phases, True


class PLOTEnv(add_env.ADDEnv):
    """Expose synchronized phase metadata without conditioning the networks.

    The phase label is computed from the same post-step reference clock used
    for the learned differential reward.  It is available only to PLOT's
    rollout scalarization and is never appended to policy or discriminator
    observations.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        self._plot_num_phases_config = env_config.get(
            "plot_num_phases", "auto")
        super().__init__(
            env_config=env_config, engine_config=engine_config,
            num_envs=num_envs, device=device, visualize=visualize,
            record_video=record_video)

    def _load_motions(self, motion_file):
        super()._load_motions(motion_file)

        motion_lengths = self._motion_lib.get_motion_lengths()
        if motion_lengths.numel() == 0:
            raise ValueError("PLOT requires at least one reference motion")

        max_length = torch.max(motion_lengths)
        min_length = torch.min(motion_lengths)
        if not torch.isclose(max_length, min_length, atol=1e-6, rtol=1e-5):
            raise ValueError(
                "PLOT currently requires equal-duration reference clips")

        control_dt = float(self._engine.get_timestep())
        if control_dt <= 0.0:
            raise ValueError("engine timestep must be positive")
        (self._plot_num_phases,
         self._plot_num_phases_is_auto) = resolve_plot_num_phases(
             self._plot_num_phases_config,
             float(max_length.item()), control_dt)
        return

    def get_plot_num_phases(self):
        return self._plot_num_phases

    def get_plot_num_phases_is_auto(self):
        return self._plot_num_phases_is_auto

    def _build_data_buffers(self):
        super()._build_data_buffers()
        self._plot_phase_idx_buf = torch.zeros(
            self.get_num_envs(), device=self._device, dtype=torch.long)
        self._info["plot_phase_idx"] = self._plot_phase_idx_buf
        return

    def _update_observations(self, env_ids=None):
        super()._update_observations(env_ids)
        if env_ids is None or len(env_ids) > 0:
            self._update_plot_phase_idx(env_ids)
        return

    def _update_plot_phase_idx(self, env_ids=None):
        if env_ids is None:
            motion_ids = self._motion_ids
        else:
            motion_ids = self._motion_ids[env_ids]

        motion_times = self._get_motion_times(env_ids)
        phase = self._motion_lib.calc_motion_phase(motion_ids, motion_times)
        phase_idx = phase_to_control_bin(phase, self._plot_num_phases)

        if env_ids is None:
            self._plot_phase_idx_buf.copy_(phase_idx)
        else:
            self._plot_phase_idx_buf[env_ids] = phase_idx
        return
