#!/usr/bin/env python3
"""Offline identifiability audit for physics-result contrastive rewards.

This does not update DARE.  It asks whether frozen rollout errors contain
information about eventual completion beyond phase, error magnitude, and the
frozen CPL logit.  Episodes, not individual frames, define the folds.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


ERROR_COUNT = 7


def collect(data_dir: Path, phase_bins: int = 32):
    episodes = np.load(data_dir / "episodes.npz")
    series = np.load(data_dir / "timeseries.npz")
    errors = series["tracking_error"][:, :, :ERROR_COUNT].astype(np.float64)
    phase = series["phase"].astype(np.float64)
    alive = series["alive"].astype(bool)
    labels = episodes["completion"].astype(np.int64)

    # One state per episode and phase bin avoids giving long-lived episodes
    # hundreds of times more weight than early failures.
    rows = []
    for bin_id in range(phase_bins):
        target = (bin_id + 0.5) / phase_bins
        distance = np.abs(phase - target)
        distance[~alive] = np.inf
        steps, envs = np.where(distance == np.min(distance, axis=0, keepdims=True))
        # np.where above can return duplicate ties; retain the closest state
        # for each environment and discard starts too far from this bin.
        best = {}
        for step, env in zip(steps.tolist(), envs.tolist()):
            value = distance[step, env]
            if value <= 0.02 and (env not in best or value < best[env][0]):
                best[env] = (value, step)
        for env, (_, step) in best.items():
            rows.append((env, target, errors[step, env], series["disc_logit"][step, env], labels[env]))

    if not rows:
        raise RuntimeError(f"no valid phase samples in {data_dir}")
    env_id = np.asarray([r[0] for r in rows], dtype=np.int64)
    phase_x = np.asarray([r[1] for r in rows], dtype=np.float64)
    group_errors = np.asarray([r[2] for r in rows], dtype=np.float64)
    logit = np.asarray([r[3] for r in rows], dtype=np.float64)
    y = np.asarray([r[4] for r in rows], dtype=np.int64)
    finite = np.isfinite(group_errors).all(1) & np.isfinite(logit)
    return {
        "episode_id": env_id[finite],
        "phase": phase_x[finite],
        "errors": group_errors[finite],
        "logit": logit[finite],
        "label": y[finite],
    }


def evaluate_probe(data, name, feature):
    x = np.asarray(feature, dtype=np.float64)
    y = data["label"]
    groups = data["episode_id"]
    scores = []
    balanced = []
    losses = []
    for train, test in GroupKFold(n_splits=5).split(x, y, groups):
        if np.unique(y[train]).size < 2 or np.unique(y[test]).size < 2:
            continue
        model = make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced"))
        model.fit(x[train], y[train])
        prob = model.predict_proba(x[test])[:, 1]
        scores.append(float(roc_auc_score(y[test], prob)))
        balanced.append(float(balanced_accuracy_score(y[test], prob >= 0.5)))
        losses.append(float(log_loss(y[test], prob, labels=[0, 1])))
    return {"name": name, "folds": len(scores), "auc": float(np.mean(scores)),
            "balanced_accuracy": float(np.mean(balanced)), "log_loss": float(np.mean(losses))}


def pair_audit(data, tolerance=0.05):
    phase = data["phase"]
    errors = data["errors"]
    y = data["label"]
    logit = data["logit"]
    scale = np.nanstd(errors, axis=0) + 1e-8
    magnitude = np.linalg.norm(errors / scale, axis=1)
    pairs = 0
    logit_correct = 0
    magnitude_tied = 0
    # Pair only inside the same phase bin.  A pair is useful when outcomes
    # differ but the normalized error magnitude is nearly the same.
    for bin_id in range(32):
        idx = np.flatnonzero((phase >= bin_id / 32) & (phase < (bin_id + 1) / 32))
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        for i in pos:
            close = neg[np.abs(magnitude[neg] - magnitude[i]) <= tolerance]
            pairs += int(close.size)
            magnitude_tied += int(close.size)
            if close.size:
                logit_correct += int(np.sum(logit[i] > logit[close]))
    return {"near_magnitude_pairs": pairs,
            "cpl_logit_correct_rate": (logit_correct / pairs if pairs else None),
            "magnitude_tolerance": tolerance,
            "scale": scale.tolist()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", required=True)
    parser.add_argument("--out-file", required=True)
    args = parser.parse_args()
    report = {}
    for directory in args.data_dir:
        data_dir = Path(directory)
        data = collect(data_dir)
        phase = data["phase"][:, None]
        magnitude = np.linalg.norm(data["errors"] / (np.nanstd(data["errors"], axis=0) + 1e-8), axis=1)[:, None]
        logit = data["logit"][:, None]
        probes = [
            evaluate_probe(data, "phase_only", phase),
            evaluate_probe(data, "phase_plus_error_magnitude", np.c_[phase, magnitude]),
            evaluate_probe(data, "phase_plus_cpl_logit", np.c_[phase, logit]),
            evaluate_probe(data, "phase_plus_seven_group_errors", np.c_[phase, data["errors"]]),
        ]
        report[str(data_dir)] = {
            "samples": int(data["label"].size),
            "episodes": int(np.unique(data["episode_id"]).size),
            "success_rate": float(data["label"].mean()),
            "probes": probes,
            "pair_audit": pair_audit(data),
        }
    Path(args.out_file).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
