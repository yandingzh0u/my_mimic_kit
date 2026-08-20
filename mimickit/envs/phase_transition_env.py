import math

import torch

import anim.motion as motion
import envs.aligned_add_env as aligned_add_env


def validate_phase_transition_config(env_config):
    """Validate the policy and reference protocol used by the critic."""
    aligned_add_env.validate_aligned_add_config(env_config)
    if int(env_config.get("phase_transition_stats_samples", 4096)) <= 0:
        raise ValueError("phase_transition_stats_samples must be positive")


class PhaseTransitionCriticEnv(aligned_add_env.AlignedADDEnv):
    """Aligned policy interface with deterministic reference statistics.

    The environment deliberately leaves the actor observation untouched.  It
    only exposes fixed statistics of one-step reference transitions to the
    phase-matched critic.  Statistics are computed from deterministic,
    uniformly spaced phases and never depend on policy rollouts.
    """

    def __init__(self, env_config, engine_config, num_envs, device, visualize,
                 record_video=False):
        validate_phase_transition_config(env_config)
        self._phase_transition_stats_samples = int(
            env_config.get("phase_transition_stats_samples", 4096))
        self._phase_transition_stats = None
        super().__init__(
            env_config=env_config,
            engine_config=engine_config,
            num_envs=num_envs,
            device=device,
            visualize=visualize,
            record_video=record_video,
        )

    def get_phase_transition_reference_stats(self):
        """Return ``(mu_x, scale_x, mu_m, scale_m)`` in ADD coordinates."""
        if self._phase_transition_stats is None:
            self._phase_transition_stats = self._build_reference_stats()
        return self._phase_transition_stats

    def get_phase_transition_metadata(self, env_ids=None):
        """Return pre-step motion identity, normalized phase, and loop flag."""
        if env_ids is None:
            motion_ids = self._motion_ids
        else:
            motion_ids = self._motion_ids[env_ids]
        motion_times = self._get_motion_times(env_ids)
        phase = self._motion_lib.calc_motion_phase(motion_ids, motion_times)
        loop_mode = self._motion_lib.get_motion_loop_mode(motion_ids)
        is_wrap = loop_mode == motion.LoopMode.WRAP.value
        # Return detached values so callers cannot mutate environment state.
        return {
            "motion_id": motion_ids.detach().clone(),
            "motion_phase": phase.detach().clone(),
            "motion_is_wrap": is_wrap.detach().clone(),
        }

    def _build_reference_stats(self):
        num_motions = int(self._motion_lib.get_num_motions())
        samples_per_motion = int(math.ceil(
            self._phase_transition_stats_samples / num_motions))

        phase = (
            torch.arange(
                samples_per_motion,
                device=self._device,
                dtype=torch.float32,
            )
            + 0.5
        ) / samples_per_motion
        motion_ids = torch.arange(
            num_motions, device=self._device, dtype=torch.long
        ).repeat_interleave(samples_per_motion)
        phase = phase.repeat(num_motions)
        motion_lengths = self._motion_lib.get_motion_length(motion_ids)
        motion_times = phase * motion_lengths
        dt = self._engine.get_timestep() * self._aligned_command_step

        with torch.no_grad():
            ref_state = self._compute_disc_obs_demo(motion_ids, motion_times)
            next_ref_state = self._compute_disc_obs_demo(
                motion_ids, motion_times + dt)
            ref_motion = next_ref_state - ref_state

            state_mean, state_scale = _mean_and_safe_scale(ref_state)
            motion_mean, motion_scale = _mean_and_safe_scale(ref_motion)

        return tuple(
            value.detach().clone()
            for value in (
                state_mean,
                state_scale,
                motion_mean,
                motion_scale,
            )
        )


def _mean_and_safe_scale(samples, min_scale=1e-4):
    mean = torch.mean(samples, dim=0)
    scale = torch.std(samples, dim=0, correction=0)
    # Constant reference coordinates contain no data-derived scale.  A unit
    # fallback leaves them unamplified instead of inventing a huge weight.
    scale = torch.where(scale > min_scale, scale, torch.ones_like(scale))
    return mean, scale
