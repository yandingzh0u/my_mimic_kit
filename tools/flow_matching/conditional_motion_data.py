from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

from tools.skill_encoder.motion_view_data import PairedMotionViewSampler


CONTEXT_STEPS = 20
MOTION_WINDOW_STEPS = 10
CONTROL_FREQUENCY = 30


def build_conditional_pair_times(
    starts: torch.Tensor,
    *,
    context_steps: int = CONTEXT_STEPS,
    window_steps: int = MOTION_WINDOW_STEPS,
    control_freq: int = CONTROL_FREQUENCY,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact, non-overlapping R2 context and motion-window schedule."""
    if starts.ndim != 1 or not torch.is_floating_point(starts):
        raise ValueError("starts must be a floating-point vector")
    if context_steps <= 0 or window_steps <= 0 or control_freq <= 0:
        raise ValueError("context_steps, window_steps, and control_freq must be positive")
    dt = 1.0 / float(control_freq)
    context_offsets = torch.arange(
        context_steps, device=starts.device, dtype=starts.dtype
    ) * dt
    window_offsets = (
        context_steps
        + torch.arange(window_steps, device=starts.device, dtype=starts.dtype)
    ) * dt
    return starts.unsqueeze(1) + context_offsets, starts.unsqueeze(1) + window_offsets


class ConditionalMotionWindowSampler(PairedMotionViewSampler):
    """Pair a 20-step encoder context with the following 10-step SMP window.

    The inherited motion/feature setup guarantees that context features use
    exactly the Stage-1A 44-D schema. ``_features_at_times`` fetches one extra
    kinematic state solely for the final foot-velocity forward difference.
    The encoder input itself ends at ``tau + 19/30`` and the motion window
    begins at ``tau + 20/30``.
    """

    def __init__(self, config: dict, device: str | torch.device):
        super().__init__(config, device)
        self.window_steps = int(self.dataset._num_disc_obs_steps)
        expected_clips = int(config.get("expected_num_clips", 54))
        if self.view_steps != CONTEXT_STEPS:
            raise ValueError(f"R2 Stage 1B requires context_steps={CONTEXT_STEPS}")
        if self.window_steps != MOTION_WINDOW_STEPS:
            raise ValueError(
                f"R2 Stage 1B requires motion_window_steps={MOTION_WINDOW_STEPS}"
            )
        if self.control_freq != CONTROL_FREQUENCY:
            raise ValueError(f"R2 Stage 1B requires control_freq={CONTROL_FREQUENCY}")
        if self.feature_dim != 44:
            raise ValueError("R2 Stage 1B requires the exact 44-D encoder schema")
        if len(self.manifest_files) != expected_clips:
            raise ValueError(
                f"R2 manifest must contain {expected_clips} clips, got "
                f"{len(self.manifest_files)}"
            )
        self.pair_end_offset = float(
            self.view_steps + self.window_steps - 1
        ) / float(self.control_freq)
        if torch.any(self.lengths < self.pair_end_offset):
            raise RuntimeError(
                "the R2 manifest contains a clip shorter than the paired A/W schedule"
            )

    def sample_pairs(
        self,
        batch_size: int,
        split: str,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        candidate_ids = self._split_ids(split)
        candidate_weights = self.weights[candidate_ids]
        sampled_index = torch.multinomial(
            candidate_weights / candidate_weights.sum(),
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        )
        return self._sample_conditional_pairs_for_ids(
            candidate_ids[sampled_index], generator, include_audit_labels=False
        )

    def sample_equal_clip_panel(
        self,
        samples_per_clip: int,
        split: str,
        generator: torch.Generator,
    ) -> dict[str, Any]:
        if samples_per_clip <= 0:
            raise ValueError("samples_per_clip must be positive")
        motion_ids = self._split_ids(split).repeat_interleave(samples_per_clip)
        order = torch.randperm(motion_ids.numel(), generator=generator)
        return self._sample_conditional_pairs_for_ids(
            motion_ids[order],
            generator,
            include_audit_labels=(split == "heldout"),
        )

    @torch.no_grad()
    def _sample_conditional_pairs_for_ids(
        self,
        motion_ids_cpu: torch.Tensor,
        generator: torch.Generator,
        *,
        include_audit_labels: bool = False,
    ) -> dict[str, Any]:
        motion_ids_cpu = torch.as_tensor(motion_ids_cpu, dtype=torch.long, device="cpu")
        available = self.lengths[motion_ids_cpu] - self.pair_end_offset
        if torch.any(available < 0):
            raise RuntimeError("sampled clip cannot contain the required paired schedule")
        starts = torch.rand(motion_ids_cpu.numel(), generator=generator) * available
        context_times, window_times = build_conditional_pair_times(
            starts,
            context_steps=self.view_steps,
            window_steps=self.window_steps,
            control_freq=self.control_freq,
        )
        if not torch.all(context_times[:, -1] < window_times[:, 0]):
            raise RuntimeError("encoder context and SMP motion window overlap")

        motion_ids = motion_ids_cpu.to(self.device)
        context_features = self._features_at_times(
            motion_ids, context_times.to(self.device)
        )
        # MotionPriorData treats the provided time as the final frame and
        # constructs the preceding H-1 frames. Passing tau+29/30 therefore
        # yields W_j=tau+(20+j)/30 for j=0,...,9 exactly.
        motion_window = self.dataset._compute_smp_obs_demo(
            motion_ids, window_times[:, -1].to(self.device)
        )
        motion_window = motion_window.reshape(motion_ids.numel(), self.window_steps, -1)
        if context_features.shape != (motion_ids.numel(), self.view_steps, 44):
            raise RuntimeError(
                f"unexpected encoder-context shape {tuple(context_features.shape)}"
            )
        if not torch.isfinite(motion_window).all():
            raise FloatingPointError("SMP motion window contains non-finite values")

        result = {
            "context_features": context_features,
            "motion_window": motion_window,
            "motion_ids": motion_ids_cpu,
            "motion_lengths": self.lengths[motion_ids_cpu],
            "mirror_families": [
                self.mirror_family(int(motion_id)) for motion_id in motion_ids_cpu
            ],
            "starts": starts,
            "context_times": context_times,
            "window_times": window_times,
        }
        if include_audit_labels:
            result["audit_tags"] = [
                self.audit_tag(int(motion_id)) for motion_id in motion_ids_cpu
            ]
            result["gate_audit_tags"] = [
                self.gate_audit_tag(int(motion_id)) for motion_id in motion_ids_cpu
            ]
        return result

    def data_audit(self) -> dict:
        audit = deepcopy(super().data_audit())
        audit["stage"] = "R2 Stage 1B conditional flow"
        audit["paired_sampling"] = {
            "context": "A_i=tau+i/30, i=0,...,19",
            "context_kinematic_states": 21,
            "extra_state_use": "final foot-velocity forward difference only",
            "motion_window": "W_j=tau+(20+j)/30, j=0,...,9",
            "tau_interval": "[0, L-29/30]",
            "pair_end_offset_seconds": self.pair_end_offset,
            "encoder_motion_window_overlap": False,
            "semantic_labels_used_for_training": False,
        }
        return audit
