#!/usr/bin/env python3
"""Strict held-out ranking gate for a phase-matched transition critic.

The score convention is "larger is better".  For a centered critic the
reference score should be zero.  All comparisons are reduced to episode means
before probabilities are computed, preventing thousands of autocorrelated
frames from masquerading as independent evidence.

Expected ``.npz`` fields:

* ``reference_score``, ``reference_episode_id``, ``reference_alive``
* ``success_score``, ``success_episode_id``, ``success_alive``
* ``shortcut_score``, ``shortcut_episode_id``, ``shortcut_alive``
* ``reference_phase_shuffled_score`` and
  ``reference_phase_shuffled_alive`` (wrong-phase reference transitions,
  matching the critic's hard-negative construction)
* ``pose_sensitivity``, ``velocity_sensitivity`` (nonnegative per-row gradient
  RMS for the two feature blocks)
* ``next_error_sensitivity``, ``motion_error_sensitivity`` (gradient RMS for
  the two transition-error blocks; this rejects a critic that ignores motion)
* ``gp_norm`` (critic input-gradient norm on held-out interpolants)

Success/shortcut membership is evaluation metadata only.  This tool never
fits a model and the raw transition bundles reject these labels.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_VERSION = 1


@dataclass(frozen=True)
class GateThresholds:
    min_episodes: int = 16
    min_pair_probability: float = 0.75
    max_reference_abs_score: float = 1e-5
    min_score_std: float = 1e-4
    min_sensitivity: float = 1e-6
    max_block_sensitivity_share: float = 0.90
    gp_mean_min: float = 0.75
    gp_mean_max: float = 1.25
    gp_q05_min: float = 0.50
    gp_q95_max: float = 1.50
    bootstrap_samples: int = 5000
    bootstrap_confidence: float = 0.95
    seed: int = 0


def _require(bundle: Mapping[str, object], key: str) -> np.ndarray:
    if key not in bundle:
        raise ValueError(f"score bundle is missing {key}")
    return np.asarray(bundle[key])


def _valid_rows(
    bundle: Mapping[str, object], prefix: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = _require(bundle, f"{prefix}_score")
    episode = _require(bundle, f"{prefix}_episode_id")
    alive = _require(bundle, f"{prefix}_alive")
    if score.ndim == 2 and score.shape[-1] == 1:
        score = score[:, 0]
    if score.ndim != 1:
        raise ValueError(f"{prefix}_score must be one-dimensional")
    if episode.shape != score.shape or alive.shape != score.shape:
        raise ValueError(f"{prefix} score/id/alive shapes must match")
    if not np.issubdtype(episode.dtype, np.integer):
        raise ValueError(f"{prefix}_episode_id must be integer")
    if alive.dtype != np.bool_:
        raise ValueError(f"{prefix}_alive must be boolean")
    if not np.all(np.isfinite(score[alive])):
        raise ValueError(f"{prefix}_score contains NaN/Inf in valid rows")
    return score, episode, alive


def _episode_mean(
    values: np.ndarray, episode: np.ndarray, alive: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.unique(episode[alive])
    means = np.asarray(
        [np.mean(values[np.logical_and(alive, episode == item)]) for item in ids],
        dtype=np.float64,
    )
    return ids, means


def _pair_probability(higher: np.ndarray, lower: np.ndarray) -> float:
    comparison = higher[:, None] - lower[None, :]
    return float(
        np.mean((comparison > 0).astype(np.float64))
        + 0.5 * np.mean(comparison == 0)
    )


def _bootstrap_margin(
    higher: np.ndarray,
    lower: np.ndarray,
    thresholds: GateThresholds,
    *,
    paired: bool,
) -> dict[str, float]:
    rng = np.random.default_rng(thresholds.seed)
    if paired:
        if higher.shape != lower.shape:
            raise ValueError("paired bootstrap arrays must have matching shapes")
        delta = higher - lower
        indices = rng.integers(
            0, delta.size, size=(thresholds.bootstrap_samples, delta.size)
        )
        samples = np.mean(delta[indices], axis=1)
    else:
        high_indices = rng.integers(
            0, higher.size, size=(thresholds.bootstrap_samples, higher.size)
        )
        low_indices = rng.integers(
            0, lower.size, size=(thresholds.bootstrap_samples, lower.size)
        )
        samples = np.mean(higher[high_indices], axis=1) - np.mean(
            lower[low_indices], axis=1
        )
    alpha = 1.0 - thresholds.bootstrap_confidence
    return {
        "mean": float(np.mean(higher) - np.mean(lower)),
        "ci_low": float(np.quantile(samples, alpha / 2.0)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha / 2.0)),
    }


def _paired_episode_values(
    values: np.ndarray,
    episode: np.ndarray,
    alive: np.ndarray,
    expected_ids: np.ndarray,
) -> np.ndarray:
    ids, means = _episode_mean(values, episode, alive)
    if not np.array_equal(ids, expected_ids):
        raise ValueError("paired diagnostic does not cover the same episodes")
    return means


def evaluate_gate(
    bundle: Mapping[str, object],
    thresholds: GateThresholds = GateThresholds(),
) -> dict[str, object]:
    if thresholds.min_episodes < 2:
        raise ValueError("min_episodes must be at least two")
    if not 0.5 < thresholds.min_pair_probability <= 1.0:
        raise ValueError("min_pair_probability must lie in (0.5, 1]")

    raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    episode_scores: dict[str, np.ndarray] = {}
    episode_ids: dict[str, np.ndarray] = {}
    for prefix in ("reference", "success", "shortcut"):
        raw[prefix] = _valid_rows(bundle, prefix)
        episode_ids[prefix], episode_scores[prefix] = _episode_mean(*raw[prefix])

    checks: dict[str, dict[str, object]] = {}
    for prefix in ("reference", "success", "shortcut"):
        count = int(episode_scores[prefix].size)
        checks[f"{prefix}_episode_count"] = {
            "pass": count >= thresholds.min_episodes,
            "value": count,
            "minimum": thresholds.min_episodes,
        }

    ref_success_probability = _pair_probability(
        episode_scores["reference"], episode_scores["success"]
    )
    success_shortcut_probability = _pair_probability(
        episode_scores["success"], episode_scores["shortcut"]
    )
    checks["reference_over_success"] = {
        "pass": ref_success_probability >= thresholds.min_pair_probability,
        "pair_probability": ref_success_probability,
        "minimum": thresholds.min_pair_probability,
        "margin": _bootstrap_margin(
            episode_scores["reference"],
            episode_scores["success"],
            thresholds,
            paired=False,
        ),
    }
    checks["success_over_shortcut"] = {
        "pass": success_shortcut_probability >= thresholds.min_pair_probability,
        "pair_probability": success_shortcut_probability,
        "minimum": thresholds.min_pair_probability,
        "margin": _bootstrap_margin(
            episode_scores["success"],
            episode_scores["shortcut"],
            thresholds,
            paired=False,
        ),
    }

    reference_abs = float(np.max(np.abs(raw["reference"][0][raw["reference"][2]])))
    checks["centered_reference_anchor"] = {
        "pass": reference_abs <= thresholds.max_reference_abs_score,
        "max_abs_score": reference_abs,
        "maximum": thresholds.max_reference_abs_score,
    }
    pooled = np.concatenate(
        [episode_scores["success"], episode_scores["shortcut"]]
    )
    pooled_std = float(np.std(pooled))
    checks["noncollapsed_scores"] = {
        "pass": pooled_std >= thresholds.min_score_std,
        "std": pooled_std,
        "minimum": thresholds.min_score_std,
    }

    reference_score, reference_episode, reference_alive = raw["reference"]
    shuffled = _require(bundle, "reference_phase_shuffled_score")
    if shuffled.ndim == 2 and shuffled.shape[-1] == 1:
        shuffled = shuffled[:, 0]
    shuffled_alive = _require(bundle, "reference_phase_shuffled_alive")
    if shuffled.shape != reference_score.shape or shuffled_alive.shape != reference_score.shape:
        raise ValueError(
            "reference phase-shuffle score/alive shapes must match reference_score"
        )
    if shuffled_alive.dtype != np.bool_:
        raise ValueError("reference_phase_shuffled_alive must be boolean")
    paired_alive = np.logical_and(reference_alive, shuffled_alive)
    if not np.all(np.isfinite(shuffled[paired_alive])):
        raise ValueError("reference_phase_shuffled_score contains NaN/Inf")
    shuffled_ids, shuffled_episode = _episode_mean(
        shuffled, reference_episode, paired_alive
    )
    correct_episode = _paired_episode_values(
        reference_score,
        reference_episode,
        paired_alive,
        shuffled_ids,
    )
    phase_probability = float(
        np.mean(correct_episode > shuffled_episode)
        + 0.5 * np.mean(correct_episode == shuffled_episode)
    )
    checks["correct_phase_over_shuffle"] = {
        "pass": phase_probability >= thresholds.min_pair_probability,
        "paired_win_probability": phase_probability,
        "minimum": thresholds.min_pair_probability,
        "margin": _bootstrap_margin(
            correct_episode,
            shuffled_episode,
            thresholds,
            paired=True,
        ),
    }

    success_score, success_episode, success_alive = raw["success"]
    sensitivity = {}
    for key in (
        "pose_sensitivity",
        "velocity_sensitivity",
        "next_error_sensitivity",
        "motion_error_sensitivity",
    ):
        value = _require(bundle, key)
        if value.ndim == 2 and value.shape[-1] == 1:
            value = value[:, 0]
        if value.shape != success_score.shape:
            raise ValueError(f"{key} shape must match success_score")
        if not np.all(np.isfinite(value[success_alive])) or np.any(
            value[success_alive] < 0
        ):
            raise ValueError(f"{key} must be finite and nonnegative")
        sensitivity[key] = float(np.mean(value[success_alive]))
    sensitivity_total = sensitivity["pose_sensitivity"] + sensitivity[
        "velocity_sensitivity"
    ]
    if sensitivity_total > 0:
        pose_share = sensitivity["pose_sensitivity"] / sensitivity_total
        velocity_share = sensitivity["velocity_sensitivity"] / sensitivity_total
    else:
        pose_share = velocity_share = 0.0
    sensitivity_pass = (
        sensitivity["pose_sensitivity"] >= thresholds.min_sensitivity
        and sensitivity["velocity_sensitivity"] >= thresholds.min_sensitivity
        and max(pose_share, velocity_share)
        <= thresholds.max_block_sensitivity_share
    )
    checks["pose_velocity_sensitivity"] = {
        "pass": sensitivity_pass,
        "pose_mean": sensitivity["pose_sensitivity"],
        "velocity_mean": sensitivity["velocity_sensitivity"],
        "pose_share": pose_share,
        "velocity_share": velocity_share,
        "minimum_each": thresholds.min_sensitivity,
        "maximum_single_block_share": thresholds.max_block_sensitivity_share,
    }

    transition_total = sensitivity["next_error_sensitivity"] + sensitivity[
        "motion_error_sensitivity"
    ]
    if transition_total > 0:
        next_share = sensitivity["next_error_sensitivity"] / transition_total
        motion_share = sensitivity["motion_error_sensitivity"] / transition_total
    else:
        next_share = motion_share = 0.0
    transition_sensitivity_pass = (
        sensitivity["next_error_sensitivity"] >= thresholds.min_sensitivity
        and sensitivity["motion_error_sensitivity"] >= thresholds.min_sensitivity
        and max(next_share, motion_share)
        <= thresholds.max_block_sensitivity_share
    )
    checks["next_motion_error_sensitivity"] = {
        "pass": transition_sensitivity_pass,
        "next_error_mean": sensitivity["next_error_sensitivity"],
        "motion_error_mean": sensitivity["motion_error_sensitivity"],
        "next_error_share": next_share,
        "motion_error_share": motion_share,
        "minimum_each": thresholds.min_sensitivity,
        "maximum_single_block_share": thresholds.max_block_sensitivity_share,
    }

    gp_norm = _require(bundle, "gp_norm").astype(np.float64, copy=False).reshape(-1)
    if gp_norm.size == 0 or not np.all(np.isfinite(gp_norm)):
        raise ValueError("gp_norm must be nonempty and finite")
    gp_mean = float(np.mean(gp_norm))
    gp_q05 = float(np.quantile(gp_norm, 0.05))
    gp_q95 = float(np.quantile(gp_norm, 0.95))
    gp_pass = (
        thresholds.gp_mean_min <= gp_mean <= thresholds.gp_mean_max
        and gp_q05 >= thresholds.gp_q05_min
        and gp_q95 <= thresholds.gp_q95_max
    )
    checks["gradient_norm"] = {
        "pass": gp_pass,
        "mean": gp_mean,
        "q05": gp_q05,
        "q95": gp_q95,
        "mean_interval": [thresholds.gp_mean_min, thresholds.gp_mean_max],
        "quantile_interval": [thresholds.gp_q05_min, thresholds.gp_q95_max],
    }

    passed = all(bool(item["pass"]) for item in checks.values())
    return {
        "schema_version": SCHEMA_VERSION,
        "pass": passed,
        "decision": "GO" if passed else "NO_GO",
        "thresholds": asdict(thresholds),
        "episode_score_means": {
            key: float(np.mean(value)) for key, value in episode_scores.items()
        },
        "checks": checks,
        "scientific_scope": (
            "Held-out diagnostic only; passing is necessary for a full RL run, "
            "not evidence that the policy will learn the motion."
        ),
    }


def load_score_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", required=True)
    parser.add_argument("--out-json", default="")
    parser.add_argument("--min-episodes", type=int, default=16)
    parser.add_argument("--min-pair-probability", type=float, default=0.75)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    thresholds = GateThresholds(
        min_episodes=args.min_episodes,
        min_pair_probability=args.min_pair_probability,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    report = evaluate_gate(load_score_bundle(args.scores), thresholds)
    rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
    print(rendered)
    if args.out_json:
        _atomic_json(Path(args.out_json), report)
    return 0 if report["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
