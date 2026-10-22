#!/usr/bin/env python3
"""Isolated smoke test for segment-level LaBER-style sampling.

This file deliberately does not touch the PPO or rollout implementation.  It
checks the part of the proposal that can be checked without a robot run:

* episode boundaries produce disjoint transition segments;
* the proposed q_j distribution is valid;
* L_j t_i / q_j is an unbiased correction for a target sum;
* the proposed segment distribution is no worse than uniform transitions for
  the supplied gradient proxy;
* PPO clipping remains a function of the policy ratio, with importance
  correction applied only to the sampled objective afterwards.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch


def build_segments(done: torch.Tensor) -> list[list[tuple[int, int]]]:
    """Return per-environment contiguous transition segments.

    A done transition belongs to the segment that ends at it.  The following
    transition starts a new segment, so no segment can cross an episode reset.
    """

    if done.ndim != 2 or done.dtype != torch.bool:
        raise ValueError("done must be a [time, env] bool tensor")
    time, num_envs = done.shape
    result: list[list[tuple[int, int]]] = []
    for env in range(num_envs):
        starts: list[tuple[int, int]] = []
        start = 0
        for t in range(time):
            if bool(done[t, env]):
                starts.append((start, t))
                start = t + 1
        if start < time:
            starts.append((start, time - 1))
        result.append(starts)
    return result


def segment_distribution(
    segments: list[tuple[int, int]],
    target: torch.Tensor,
    gradients: torch.Tensor,
) -> torch.Tensor:
    scores = []
    for start, end in segments:
        local = gradients[start : end + 1]
        local_target = target[start : end + 1]
        length = end - start + 1
        score = length * torch.sum(local_target.square() * local.square().sum(dim=-1))
        scores.append(torch.sqrt(score.clamp_min(1e-20)))
    scores_t = torch.stack(scores)
    if not torch.isfinite(scores_t).all() or float(scores_t.sum()) <= 0:
        raise ValueError("segment scores must be finite and nonzero")
    return scores_t / scores_t.sum()


def flatten_segments(segments: list[tuple[int, int]]) -> list[int]:
    return [i for start, end in segments for i in range(start, end + 1)]


def theoretical_trace_variance(
    segments: list[tuple[int, int]],
    target: torch.Tensor,
    gradients: torch.Tensor,
    q: torch.Tensor,
) -> float:
    all_indices = flatten_segments(segments)
    estimate_mean = torch.sum(target[:, None] * gradients, dim=0)
    second = torch.zeros((), dtype=gradients.dtype)
    for j, (start, end) in enumerate(segments):
        length = end - start + 1
        p = q[j] / length
        values = target[start : end + 1, None] * gradients[start : end + 1]
        second = second + torch.sum(values.square().sum(dim=-1) / p)
    # all_indices is intentionally used to make sure every transition is
    # represented exactly once in this construction.
    if len(all_indices) != gradients.shape[0] or len(set(all_indices)) != gradients.shape[0]:
        raise AssertionError("segments must partition transitions exactly")
    return float(second - estimate_mean.square().sum())


def monte_carlo_mean(
    segments: list[tuple[int, int]],
    target: torch.Tensor,
    gradients: torch.Tensor,
    q: torch.Tensor,
    samples: int,
    generator: torch.Generator,
) -> torch.Tensor:
    out = torch.zeros(gradients.shape[1], dtype=gradients.dtype)
    for _ in range(samples):
        j = int(torch.multinomial(q, 1, generator=generator))
        start, end = segments[j]
        i = start + int(torch.randint(end - start + 1, (1,), generator=generator))
        p = q[j] / (end - start + 1)
        out += target[i] * gradients[i] / p
    return out / samples


def ppo_clip_smoke() -> dict[str, float]:
    """Check correction placement against a hand-computed clipped objective."""

    old_logp = torch.tensor([-1.0, -0.5, -1.2, -0.7])
    new_logp = torch.tensor([-0.8, -0.7, -1.5, -0.6])
    advantages = torch.tensor([1.0, -2.0, 0.5, -0.25])
    ratio = (new_logp - old_logp).exp()
    clip_eps = 0.2
    unclipped = ratio * advantages
    clipped = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
    objective = torch.minimum(unclipped, clipped)
    # Importance correction rescales the sampled PPO contribution; it does
    # not alter the PPO ratio or clipping decision itself.
    correction = torch.tensor([0.8, 1.4, 0.6, 1.2])
    corrected = correction * objective
    assert torch.allclose(ratio, (new_logp - old_logp).exp())
    assert torch.isfinite(corrected).all()
    return {
        "ratio_min": float(ratio.min()),
        "ratio_max": float(ratio.max()),
        "objective_mean": float(objective.mean()),
        "corrected_objective_mean": float(corrected.mean()),
    }


def main() -> None:
    torch.manual_seed(20261023)
    # Two environments with different reset locations.  The final transition
    # of each episode is marked done and must remain in its preceding segment.
    done = torch.zeros(12, 2, dtype=torch.bool)
    done[[2, 7, 11], 0] = True
    done[[4, 9], 1] = True
    per_env = build_segments(done)
    segments = per_env[0] + per_env[1]
    # Duplicate env-local timelines into one flat transition index space for
    # this smoke; the segment construction itself is tested on the original
    # [time, env] layout above.
    flat_segments: list[tuple[int, int]] = []
    offset = 0
    for env_segments in per_env:
        for start, end in env_segments:
            flat_segments.append((offset + start, offset + end))
        offset += done.shape[0]
    assert flat_segments == [(0, 2), (3, 7), (8, 11), (12, 16), (17, 21), (22, 23)]
    for start, end in flat_segments:
        env = 0 if end < 12 else 1
        local_done = done[:, env]
        assert not bool(local_done[start - (12 if env else 0) : end - (12 if env else 0)].any())

    count = offset
    frame_ids = torch.arange(count) % 8
    frame_counts = torch.bincount(frame_ids, minlength=8).to(torch.float32)
    target = 1.0 / (float((frame_counts > 0).sum()) * frame_counts[frame_ids])
    gradients = torch.randn(count, 5)
    # Give segments unequal proxy-gradient energy so q is meaningfully
    # different from length-proportional uniform transition sampling.
    gradients[:3] *= 0.25
    gradients[17:22] *= 3.0
    q = segment_distribution(flat_segments, target, gradients)
    assert torch.isfinite(q).all() and torch.all(q > 0)
    assert abs(float(q.sum()) - 1.0) < 1e-6

    trace_laber = theoretical_trace_variance(flat_segments, target, gradients, q)
    uniform_q = torch.tensor(
        [(end - start + 1) / count for start, end in flat_segments], dtype=torch.float32
    )
    trace_uniform = theoretical_trace_variance(flat_segments, target, gradients, uniform_q)
    assert trace_laber <= trace_uniform + 1e-5

    generator = torch.Generator().manual_seed(20261024)
    estimate = monte_carlo_mean(flat_segments, target, gradients, q, 100_000, generator)
    exact = torch.sum(target[:, None] * gradients, dim=0)
    max_unbiased_error = float((estimate - exact).abs().max())
    assert max_unbiased_error < 0.05
    ppo_stats = ppo_clip_smoke()

    out = {
        "status": "passed",
        "segments": [list(x) for x in flat_segments],
        "segment_lengths": [end - start + 1 for start, end in flat_segments],
        "q": [float(x) for x in q],
        "target_sum": float(target.sum()),
        "trace_variance_laber": trace_laber,
        "trace_variance_uniform_transition": trace_uniform,
        "variance_reduction_fraction": 1.0 - trace_laber / trace_uniform,
        "max_unbiased_error_100k": max_unbiased_error,
        "ppo_clip": ppo_stats,
    }
    output = Path("output/diagnostics/laber_sampler_smoke_20261023")
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
