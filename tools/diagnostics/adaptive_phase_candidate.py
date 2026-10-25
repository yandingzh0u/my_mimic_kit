"""Isolated candidate: control-resolution phase strata and balanced reuse.

This module is deliberately not enabled in PPO.  It changes the empirical
phase target when its resolution differs from the historical eight bins.
No training-performance claim follows from the sampling guarantees below.
"""
from __future__ import annotations

import math
import random

import torch


def control_cells(duration: float, control_dt: float) -> int:
    """Count control intervals; pass duration calculated from frame count/fps.

    Moving by one double ULP avoids a spurious extra cell at an exactly
    integral duration/control_dt.  This is a numerical guard, not a cutoff
    for changing the temporal resolution.
    """
    if not math.isfinite(duration) or duration <= 0:
        raise ValueError("duration must be finite and positive")
    if not math.isfinite(control_dt) or control_dt <= 0:
        raise ValueError("control_dt must be finite and positive")
    ratio = duration / control_dt
    if not math.isfinite(ratio):
        raise ValueError("duration/control_dt must be finite")
    return max(1, math.ceil(math.nextafter(ratio, -math.inf)))


class AdaptivePhaseSampler:
    """Time strata, with at least one draw per occupied stratum.

    K = min(control_cells, smallest_PPO_batch, rollout_size). Native control
    cells are merged at integer cell boundaries when K < control_cells.
    Bin masses follow their temporal widths, not the number of transitions.
    Actor and
    critic should use the same smallest_PPO_batch and full rollout phases,
    with a separate actor eligibility mask.  Pool construction happens once
    per rollout.  Weights replace phase_weight; they must not multiply it.

    Every stratum gets floor(B / K_present) samples, plus rotating remainder
    slots.  Within each stratum every transition appears q or q+1 times.
    A fresh permutation is drawn for EVERY batch, independently of previous
    calls.  This is not the historical stratified_epoch sampler.

    Empty strata are explicitly reported. The estimand is the time-width
    weighted mean of OCCUPIED strata, not the unobserved parts of a motion.
    It becomes an equal-stratum mean when K == control_cells.
    """

    def __init__(self, phase: torch.Tensor, *, frame_count: int,
                 batch_budget: int, eligible: torch.Tensor | None = None):
        if not isinstance(frame_count, int) or isinstance(frame_count, bool) or frame_count < 1:
            raise ValueError("frame_count must be a positive integer")
        if frame_count > 2**31 - 1:
            raise ValueError("frame_count exceeds the supported integer range")
        if not isinstance(batch_budget, int) or isinstance(batch_budget, bool) or batch_budget < 1:
            raise ValueError("batch_budget must be a positive integer")
        if phase.numel() == 0 or not phase.is_floating_point():
            raise ValueError("phase must be a nonempty floating-point tensor")
        flat = phase.detach().flatten().to(torch.float64)
        if not bool(torch.isfinite(flat).all()) or bool(((flat < 0) | (flat > 1)).any()):
            raise ValueError("phases must be finite and in [0, 1]")
        if eligible is None:
            mask = torch.ones_like(flat, dtype=torch.bool)
        else:
            if eligible.dtype != torch.bool or eligible.numel() != flat.numel():
                raise ValueError("eligible must be a bool mask matching phase size")
            mask = eligible.detach().flatten().to(flat.device)
        self.indices = mask.nonzero(as_tuple=False).flatten()
        if not self.indices.numel():
            raise ValueError("no eligible transitions: skip this update; do not invent a sample")
        self.frame_count = frame_count
        self.num_bins = min(frame_count, batch_budget, flat.numel())
        native = (flat * frame_count).floor().long().clamp_max(frame_count - 1)
        self.bin_ids = (native * self.num_bins) // frame_count
        present, self.labels = torch.unique(self.bin_ids[self.indices], sorted=True,
                                            return_inverse=True)
        self.present_bins = present
        self.num_present = present.numel()
        left = (present * frame_count + self.num_bins - 1) // self.num_bins
        right = ((present + 1) * frame_count + self.num_bins - 1) // self.num_bins
        widths = (right - left).to(torch.float64)
        self.target_mass = widths / widths.sum()
        self.counts = torch.bincount(self.labels, minlength=self.num_present)
        self.offsets = self.counts.cumsum(0) - self.counts
        self._remainder_cursor = 0
        native_present = torch.unique(native[self.indices]).numel()
        self.coverage = dict(control_cells=frame_count, partition_bins=self.num_bins,
                             occupied_bins=self.num_present,
                             empty_bins=self.num_bins - self.num_present,
                             covered_partition_time_fraction=float(widths.sum()/frame_count),
                             occupied_control_cells=native_present,
                             missing_control_cells=frame_count - native_present,
                             eligible_transitions=self.indices.numel())

    @torch.no_grad()
    def sample(self, batch_size: int, *, generator: torch.Generator | None = None):
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < self.num_present:
            raise ValueError("batch_size must cover every occupied bin; rebuild with a smaller batch_budget")
        device = self.indices.device
        base, remainder = divmod(batch_size, self.num_present)
        quota = torch.full((self.num_present,), base, dtype=torch.long, device=device)
        extra = (torch.arange(remainder, device=device) + self._remainder_cursor) % self.num_present
        quota[extra] += 1
        self._remainder_cursor = (self._remainder_cursor + remainder) % self.num_present

        return self._draw_quotas(quota, batch_size, generator=generator)

    def _draw_quotas(self, quota, batch_size, *, generator=None, exact_population=False):
        """Draw independent within-bin permutations for a given quota plan."""
        device = self.indices.device

        # An exact uniform permutation within every bin, using two batched
        # operations instead of launching one randperm per bin on the GPU.
        perm = torch.randperm(self.indices.numel(), device=device, generator=generator)
        grouped = self.indices[perm[torch.argsort(self.labels[perm], stable=True)]]
        bins = torch.repeat_interleave(torch.arange(self.num_present, device=device),
                                      quota, output_size=batch_size)
        start = quota.cumsum(0) - quota
        ordinal = torch.arange(batch_size, device=device) - start[bins]
        picked = grouped[self.offsets[bins] + ordinal % self.counts[bins]]
        if exact_population:
            # All transitions are present. Compensate each transition's
            # multiplicity, so extra copies do not alter its aggregate mass.
            q = quota[bins] // self.counts[bins]
            r = quota[bins] % self.counts[bins]
            multiplicity = q + ((ordinal % self.counts[bins]) < r).long()
            weights = self.target_mass[bins] / (self.counts[bins] * multiplicity).double()
        else:
            weights = self.target_mass[bins] / quota[bins].to(torch.float64)
        shuffle = torch.randperm(batch_size, device=device, generator=generator)
        return picked[shuffle], weights[shuffle], quota

    def full_weights(self):
        """Exact full-rollout weights, e.g. for matching advantage statistics."""
        weights = torch.zeros(self.bin_ids.numel(), device=self.indices.device,
                              dtype=torch.float64)
        weights[self.indices] = self.target_mass[self.labels] / self.counts[self.labels].to(torch.float64)
        return weights


def capped_quota_plan(counts, batch_size, cursor=0, refund_cursor=0):
    """Capacity-capped equal quotas that preserve the original useful draws.

    For K <= B <= sum(counts), every m_k is in [1, N_k], sums to B,
    and m_k >= min(original_equal_quota_k, N_k). This last invariant gives
    a covariance dominance guarantee over the balanced-reuse sampler for
    the SAME fixed target, parameters and rollout. It is not Neyman-optimal
    allocation: no gradient-variance estimate is used here.

    For B > sum(counts), every transition is included; callers must correct
    multiplicities to keep an exact full-population estimate.
    """
    if (not counts or any(type(n) is not int or n < 1 for n in counts)
            or type(batch_size) is not int or batch_size < len(counts)):
        raise ValueError("positive populations and a batch covering every bin are required")
    k, total = len(counts), sum(counts)
    base, rem = divmod(batch_size, k)
    old = [base] * k
    for j in range(rem):
        old[(cursor+j) % k] += 1
    next_cursor = (cursor+rem) % k
    if batch_size >= total:
        extra, remainder = divmod(batch_size-total, k)
        quota = [n+extra for n in counts]
        for j in range(remainder):
            quota[(refund_cursor+j) % k] += 1
        return quota, next_cursor, (refund_cursor+remainder) % k, old
    if all(m <= n for m, n in zip(old, counts)):
        return old, next_cursor, refund_cursor, old

    # Largest water level with sum min(N_k, level) <= B. This is based only
    # on exact counts, with no density threshold or estimated variance.
    low, high = base, batch_size
    while low < high:
        middle = (low+high+1)//2
        if sum(min(n, middle) for n in counts) <= batch_size:
            low = middle
        else:
            high = middle-1
    quota = [min(n, low) for n in counts]
    # Retain any original remainder slot that still contains a useful draw.
    protected = [False] * k
    for j in range(k):
        useful = min(old[j], counts[j])
        if useful > quota[j]:
            quota[j] = useful
            protected[j] = True
    left = batch_size-sum(quota)
    next_refund = refund_cursor
    for distance in range(k):
        if not left:
            break
        j = (refund_cursor+distance) % k
        if counts[j] > low and not protected[j]:
            quota[j] += 1
            left -= 1
            # Save the next position, rather than repeatedly rewarding bin 0.
            next_refund = (j+1) % k
    if left:
        raise RuntimeError("capacity redistribution did not exhaust its budget")
    return quota, next_cursor, next_refund, old


class CapacityAwarePhaseSampler(AdaptivePhaseSampler):
    """Same temporal target; move redundant copies into distinct observations."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._counts_cpu = self.counts.cpu().tolist()
        self._refund_cursor = 0

    @torch.no_grad()
    def sample(self, batch_size: int, *, generator: torch.Generator | None = None):
        quota, self._remainder_cursor, self._refund_cursor, old = capped_quota_plan(
            self._counts_cpu, batch_size, self._remainder_cursor, self._refund_cursor)
        quota = torch.tensor(quota, dtype=torch.long, device=self.indices.device)
        return self._draw_quotas(quota, batch_size, generator=generator,
                                 exact_population=batch_size >= self.indices.numel())


class PhaseCoverageRotor:
    """Reference-reset candidate, single motion, O(1) state even for long clips.

    An affine permutation of control-cell IDs repeats every F reset requests.
    Every consecutive F requests visit every cell exactly once.  Uniform
    random offset makes each FIXED request index marginally uniform across
    cells in isolation; independent jitter gives a uniform-time marginal.
    This does not assert IID resets or conditional uniformity given history.
    These are dependent resets.  The guarantee concerns requested reference
    starts, NOT successful trajectories or final tracking error.

    A production multi-motion integration needs one rotor per motion, with
    the motion-selection distribution explicitly preserved.  This prototype
    does not silently replace motion weights by clip lengths.
    """

    def __init__(self, frame_count: int, *, seed: int = 0, device="cpu"):
        if not isinstance(frame_count, int) or isinstance(frame_count, bool) or not 1 <= frame_count <= 2**31 - 1:
            raise ValueError("frame_count must be an integer in [1, 2**31-1]")
        self.frame_count = frame_count
        rng = random.Random(seed)
        self.stride = rng.randrange(1, max(2, frame_count))
        while math.gcd(self.stride, frame_count) != 1:
            self.stride = rng.randrange(1, max(2, frame_count))
        self.offset = rng.randrange(frame_count)
        self.cursor = 0
        self.device = torch.device(device)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)

    @torch.no_grad()
    def draw(self, count: int):
        if not isinstance(count, int) or isinstance(count, bool) or count < 0:
            raise ValueError("count must be a nonnegative integer")
        slots = (torch.arange(count, device=self.device) + self.cursor) % self.frame_count
        cells = (slots * self.stride + self.offset) % self.frame_count
        self.cursor = (self.cursor + count) % self.frame_count
        jitter = torch.rand(count, dtype=torch.float64, device=self.device,
                            generator=self.generator)
        phase = (cells.to(torch.float64) + jitter) / self.frame_count
        # Keep rounding off an adjacent cell's boundary, even for long clips.
        low = cells.to(torch.float64) / self.frame_count
        high = (cells.to(torch.float64) + 1) / self.frame_count
        phase = torch.maximum(phase, torch.nextafter(low, torch.full_like(low, math.inf)))
        phase = torch.minimum(phase, torch.nextafter(high, torch.full_like(high, -math.inf)))
        return phase, cells

    def state_dict(self):
        return dict(frame_count=self.frame_count, stride=self.stride,
                    offset=self.offset, cursor=self.cursor,
                    generator_state=self.generator.get_state().clone())

    @torch.no_grad()
    def draw_times(self, durations: torch.Tensor):
        """Return reference times whose *actual dtype* retains the scheduled cell.

        Float32 multiplication/division can move a jittered boundary into the
        next cell. Repair such cases with the cell midpoint and verify again.
        If a clip is too long for its timestamp dtype, fail explicitly: no
        sampler can represent all its control cells in that dtype.
        """
        if (durations.ndim != 1 or not durations.is_floating_point()
                or durations.device != self.device
                or not bool(torch.isfinite(durations).all())
                or not bool((durations > 0).all())):
            raise ValueError("durations must be positive finite floating-point values on the rotor device")
        phase, cells = self.draw(durations.numel())
        times = (phase * durations.double()).to(durations.dtype)

        def actual_cells(value):
            return ((value/durations).double().clamp(0, 1)*self.frame_count).floor().long().clamp_max(self.frame_count-1)

        bad = actual_cells(times) != cells
        if bool(bad.any()):
            midpoint = (cells[bad].double()+.5)/self.frame_count
            times[bad] = (midpoint*durations[bad].double()).to(durations.dtype)
        if not bool((actual_cells(times) == cells).all()):
            raise ValueError("timestamp dtype cannot represent the requested control cells; increase precision")
        return times, cells

    def load_state_dict(self, state):
        if state["frame_count"] != self.frame_count:
            raise ValueError("rotor state has a different motion resolution")
        self.stride, self.offset, self.cursor = state["stride"], state["offset"], state["cursor"]
        self.generator.set_state(state["generator_state"])
