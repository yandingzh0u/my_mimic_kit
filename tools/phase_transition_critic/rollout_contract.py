#!/usr/bin/env python3
"""Schema and integrity checks for held-out transition rollouts.

The raw bundle intentionally contains no success, shortcut, completion, reward,
or winding labels.  Those quantities may select held-out files for evaluation,
but must never become critic inputs or critic-training targets.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Mapping

import numpy as np


SCHEMA_VERSION = 1
FEATURE_KEYS = ("x_t", "x_t1", "r_t", "r_t1")
INDEX_KEYS = ("episode_id", "step_index", "phase", "alive")
REQUIRED_KEYS = FEATURE_KEYS + INDEX_KEYS
FORBIDDEN_LABEL_TOKENS = (
    "completion",
    "success",
    "shortcut",
    "winding",
    "reward",
    "label",
)


def _as_array_dict(bundle: Mapping[str, object]) -> dict[str, np.ndarray]:
    return {key: np.asarray(value) for key, value in bundle.items()}


def validate_transition_bundle(
    bundle: Mapping[str, object],
    *,
    continuity_atol: float = 2e-5,
    reject_behavior_labels: bool = True,
) -> dict[str, object]:
    """Validate shapes, finite values, indices, and one-step continuity.

    Rows may be time-major or episode-major.  Continuity is checked after
    sorting valid rows by ``(episode_id, step_index)``.  A transition ending an
    episode is valid; continuity is only required when its next step exists.
    """

    arrays = _as_array_dict(bundle)
    missing = [key for key in REQUIRED_KEYS if key not in arrays]
    if missing:
        raise ValueError(f"transition bundle is missing: {', '.join(missing)}")

    if reject_behavior_labels:
        forbidden = sorted(
            key
            for key in arrays
            if any(token in key.lower() for token in FORBIDDEN_LABEL_TOKENS)
        )
        if forbidden:
            raise ValueError(
                "raw critic bundle contains behavior labels: "
                + ", ".join(forbidden)
            )

    feature_shape = arrays[FEATURE_KEYS[0]].shape
    if len(feature_shape) != 2 or feature_shape[0] == 0 or feature_shape[1] == 0:
        raise ValueError("x_t must have nonempty shape [num_rows, phi_dim]")
    num_rows, phi_dim = feature_shape
    for key in FEATURE_KEYS:
        value = arrays[key]
        if value.shape != feature_shape:
            raise ValueError(
                f"{key} has shape {value.shape}, expected {feature_shape}"
            )

    for key in INDEX_KEYS:
        if arrays[key].shape != (num_rows,):
            raise ValueError(f"{key} must have shape ({num_rows},)")

    episode_id = arrays["episode_id"]
    step_index = arrays["step_index"]
    phase = arrays["phase"]
    alive = arrays["alive"]
    if not np.issubdtype(episode_id.dtype, np.integer):
        raise ValueError("episode_id must be an integer array")
    if not np.issubdtype(step_index.dtype, np.integer):
        raise ValueError("step_index must be an integer array")
    if alive.dtype != np.bool_:
        raise ValueError("alive must be a boolean array")
    if np.any(step_index < 0):
        raise ValueError("step_index must be nonnegative")
    if not np.all(np.isfinite(phase[alive])):
        raise ValueError("valid phases must be finite")
    if np.any((phase[alive] < -1e-6) | (phase[alive] > 1.0 + 1e-6)):
        raise ValueError("valid phases must lie in [0, 1]")
    for key in FEATURE_KEYS:
        if not np.all(np.isfinite(arrays[key][alive])):
            raise ValueError(f"{key} contains NaN/Inf in valid transitions")

    valid_rows = np.flatnonzero(alive)
    if valid_rows.size == 0:
        raise ValueError("transition bundle contains no valid transitions")
    ordered = valid_rows[
        np.lexsort((step_index[valid_rows], episode_id[valid_rows]))
    ]
    ordered_episode = episode_id[ordered]
    ordered_step = step_index[ordered]
    same_episode = ordered_episode[:-1] == ordered_episode[1:]
    consecutive = ordered_step[1:] == ordered_step[:-1] + 1
    links = np.logical_and(same_episode, consecutive)
    if np.any(links):
        left = ordered[:-1][links]
        right = ordered[1:][links]
        x_error = float(
            np.max(np.abs(arrays["x_t1"][left] - arrays["x_t"][right]))
        )
        r_error = float(
            np.max(np.abs(arrays["r_t1"][left] - arrays["r_t"][right]))
        )
        if x_error > continuity_atol:
            raise ValueError(
                f"policy transition continuity error {x_error:.3g} exceeds "
                f"{continuity_atol:.3g}"
            )
        if r_error > continuity_atol:
            raise ValueError(
                f"reference transition continuity error {r_error:.3g} exceeds "
                f"{continuity_atol:.3g}"
            )
    else:
        x_error = 0.0
        r_error = 0.0

    unique_episode_ids = np.unique(episode_id[alive])
    return {
        "schema_version": SCHEMA_VERSION,
        "num_rows": int(num_rows),
        "valid_rows": int(valid_rows.size),
        "num_episodes": int(unique_episode_ids.size),
        "phi_dim": int(phi_dim),
        "continuity_links": int(np.count_nonzero(links)),
        "policy_continuity_max_abs": x_error,
        "reference_continuity_max_abs": r_error,
        "contains_behavior_labels": False,
    }


def load_transition_bundle(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def atomic_savez_compressed(path: str | Path, **arrays: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_name(output.name + ".tmp.npz")
    np.savez_compressed(temp, **arrays)
    os.replace(temp, output)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle")
    parser.add_argument("--continuity-atol", type=float, default=2e-5)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = validate_transition_bundle(
        load_transition_bundle(args.bundle),
        continuity_atol=args.continuity_atol,
    )
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
