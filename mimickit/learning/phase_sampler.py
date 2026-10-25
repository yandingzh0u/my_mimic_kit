"""Two-stage, without-replacement sampling of a phase-balanced PPO target."""
import torch
from util.phase_coverage import control_cells


class PhaseSampler:
    def __init__(self, phase, control_intervals, batch_budget, eligible=None):
        phase = phase.detach().flatten().double()
        self.size = phase.numel()
        self.control_intervals = int(control_intervals)
        if self.size == 0 or self.control_intervals < 1 or batch_budget < 1:
            raise ValueError("Phase population, resolution and budget must be positive")
        if self.control_intervals > 2**31 - 1:
            raise ValueError("Motion exceeds supported control resolution")
        if not bool((torch.isfinite(phase) & (phase >= 0) & (phase <= 1)).all()):
            raise ValueError("Motion phases must be finite and in [0, 1]")
        self.num_bins = min(self.control_intervals, int(batch_budget), self.size)
        native = (phase * self.control_intervals).long().clamp_max(self.control_intervals - 1)
        self.bin_ids = native * self.num_bins // self.control_intervals
        if eligible is None:
            eligible = torch.ones_like(phase, dtype=torch.bool)
        if eligible.dtype != torch.bool or eligible.numel() != self.size:
            raise ValueError("Eligibility must be a Boolean mask of the population")
        self.indices = eligible.flatten().nonzero().flatten()
        self.present_bins, self.labels, self.counts = torch.unique(
            self.bin_ids[self.indices], sorted=True, return_inverse=True, return_counts=True)
        self.num_present = self.counts.numel()
        # Integer boundaries preserve time mass when F is not divisible by K.
        left = (self.present_bins * self.control_intervals + self.num_bins - 1) // self.num_bins
        right = ((self.present_bins + 1) * self.control_intervals + self.num_bins - 1) // self.num_bins
        widths = (right - left).double()
        self.target_mass = widths / widths.sum()
        self._sorted_labels = torch.repeat_interleave(
            torch.arange(self.num_present, device=phase.device), self.counts)
        offsets = self.counts.cumsum(0) - self.counts
        self._ranks = torch.arange(len(self.indices), device=phase.device) - offsets[self._sorted_labels]
        self.coverage = {
            "control_cells": self.control_intervals,
            "partition_bins": self.num_bins,
            "occupied_bins": self.num_present,
            "empty_bins": self.num_bins - self.num_present,
            "covered_time_fraction": float(widths.sum()) / self.control_intervals,
            "missing_control_cells": self.control_intervals - native[self.indices].unique().numel(),
            "eligible_transitions": len(self.indices),
        }

    def full_weights(self):
        weights = torch.zeros(self.size, device=self.indices.device, dtype=torch.float64)
        weights[self.indices] = (self.target_mass / self.counts)[self.labels]
        return weights

    def sample(self, batch_size, generator=None):
        """Return population indices, normalized weights and final phase counts.

        Every occupied phase has at least one slot. Quotas are capped by the
        pool sizes; unused slots are filled uniformly from remaining samples.
        The batch is fresh each call, rather than constrained across an epoch.
        """
        n = len(self.indices)
        b = min(int(batch_size), n)
        h = self.num_present
        if batch_size < 1 or b < h:
            raise ValueError("Batch budget cannot cover all occupied phases")
        if b == n:
            return self.indices, self.full_weights()[self.indices], self.counts.clone()
        device = self.indices.device
        quota = torch.full((h,), b // h, device=device, dtype=torch.long)
        priority = torch.randperm(h, device=device, generator=generator)
        quota[priority[:b % h]] += 1
        capped = torch.minimum(quota, self.counts)
        # A random permutation followed by stable grouping gives an exact
        # uniform permutation within every phase, without floating-point keys.
        order = torch.randperm(n, device=device, generator=generator)
        grouped = order[torch.argsort(self.labels[order], stable=True)]
        chosen = grouped[self._ranks < capped[self._sorted_labels]]
        missing = b - len(chosen)
        if missing:
            remaining_mask = torch.ones(n, device=device, dtype=torch.bool)
            remaining_mask[chosen] = False
            remaining = remaining_mask.nonzero().flatten()
            fill = torch.randperm(len(remaining), device=device, generator=generator)[:missing]
            chosen = torch.cat((chosen, remaining[fill]))
        labels = self.labels[chosen]
        final_counts = torch.bincount(labels, minlength=h)
        weights = (self.target_mass / final_counts)[labels]
        shuffle = torch.randperm(b, device=device, generator=generator)
        return self.indices[chosen[shuffle]], weights[shuffle], final_counts
