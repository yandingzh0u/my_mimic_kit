#!/usr/bin/env python3
"""Aggregate paper-evaluation summaries with the seed as sampling unit."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate summary.json files without pooling episodes across seeds."
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="summary.json files or directories searched recursively.",
    )
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def discover_summaries(inputs: Iterable[str | Path]) -> list[Path]:
    paths: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            if path.name != "summary.json":
                raise ValueError(f"expected summary.json, got {path}")
            paths.add(path)
        elif path.is_dir():
            paths.update(path.rglob("summary.json"))
        else:
            raise FileNotFoundError(path)
    return sorted(paths)


def load_summary(path: str | Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or "metadata" not in value:
        raise ValueError(f"invalid paper-eval summary: {path}")
    value["_source_file"] = str(Path(path).resolve())
    return value


def _numeric_leaf(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    return None


def extract_paper_metrics(summary: Mapping[str, Any]) -> dict[str, float]:
    """Extract only scientific outcomes, excluding protocol/metadata numbers."""

    output: dict[str, float] = {}

    completion = summary.get("completion", {})
    rate = _numeric_leaf(completion.get("rate"))
    completion_available = bool(completion.get("available", True))
    if rate is not None and completion_available:
        output["completion.rate"] = rate
    if completion_available:
        for name, value in completion.get("components", {}).items():
            number = _numeric_leaf(value)
            if number is not None:
                output[f"completion.components.{name}"] = number

    metrics = summary.get("metrics", {})
    for family in ("tracking", "behavior"):
        for name, stats in metrics.get(family, {}).items():
            number = _numeric_leaf(stats.get("mean"))
            if number is not None:
                output[f"metrics.{family}.{name}"] = number
    for name, stats in metrics.get("reward", {}).items():
        number = _numeric_leaf(stats.get("mean"))
        if number is not None:
            output[f"metrics.reward.{name}"] = number
    for name, stats in metrics.get("intervention", {}).items():
        number = _numeric_leaf(stats.get("mean"))
        if number is not None:
            output[f"metrics.intervention.{name}"] = number

    efficiency = summary.get("efficiency", {})
    for name in (
        "simulator_env_steps_per_second",
        "policy_latency_us_per_env_step",
        "batch1_policy_latency_us",
        "peak_gpu_memory_mb",
    ):
        number = _numeric_leaf(efficiency.get(name))
        if number is not None:
            output[f"efficiency.{name}"] = number
    return output


def _group_key(summary: Mapping[str, Any]) -> tuple[Any, ...]:
    metadata = summary["metadata"]
    checkpoint = metadata.get("checkpoint", {})
    return (
        metadata.get("method"),
        metadata.get("motion"),
        metadata.get("representation"),
        metadata.get("condition", "nominal"),
        checkpoint.get("iteration"),
        checkpoint.get("samples"),
    )


def _sample_stats(values: Iterable[float]) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    n = int(array.size)
    if n == 0:
        raise ValueError("cannot aggregate an empty metric")
    std = float(np.std(array, ddof=1)) if n > 1 else 0.0
    sem = std / math.sqrt(n) if n > 1 else 0.0
    return {
        "n_seeds": n,
        "mean": float(np.mean(array)),
        "std": std,
        "sem": sem,
        "ci95_low": float(np.mean(array) - 1.96 * sem),
        "ci95_high": float(np.mean(array) + 1.96 * sem),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
    }


def aggregate_summaries(
    summaries: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Aggregate episode means across distinct seeds for each matched group."""

    grouped: dict[tuple[Any, ...], dict[Any, Mapping[str, Any]]] = {}
    for summary in summaries:
        key = _group_key(summary)
        seed = summary["metadata"].get("seed")
        seed_map = grouped.setdefault(key, {})
        if seed in seed_map:
            raise ValueError(
                "duplicate seed in aggregate group {}: seed {} from {} and {}".format(
                    key,
                    seed,
                    seed_map[seed].get("_source_file", "unknown"),
                    summary.get("_source_file", "unknown"),
                )
            )
        seed_map[seed] = summary

    results: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda item: tuple(str(x) for x in item)):
        method, motion, representation, condition, iteration, samples = key
        seed_map = grouped[key]
        seed_metrics = {
            seed: extract_paper_metrics(summary)
            for seed, summary in seed_map.items()
        }
        metric_names: set[str] = set()
        for metrics in seed_metrics.values():
            metric_names.update(metrics)

        aggregate_metrics: dict[str, dict[str, float | int]] = {}
        for metric_name in sorted(metric_names):
            values = [
                metrics[metric_name]
                for metrics in seed_metrics.values()
                if metric_name in metrics
            ]
            aggregate_metrics[metric_name] = _sample_stats(values)

        results.append(
            {
                "method": method,
                "motion": motion,
                "representation": representation,
                "condition": condition,
                "checkpoint_iteration": iteration,
                "checkpoint_samples": samples,
                "seeds": sorted(seed_map, key=str),
                "metrics": aggregate_metrics,
                "sources": sorted(
                    summary.get("_source_file", "unknown")
                    for summary in seed_map.values()
                ),
            }
        )
    return results


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def write_aggregate_tables(groups: list[dict[str, Any]], out_dir: str | Path) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    _atomic_json(
        out_path / "aggregate.json",
        {"schema_version": 1, "groups": groups},
    )

    identity = [
        "method",
        "motion",
        "representation",
        "condition",
        "checkpoint_iteration",
        "checkpoint_samples",
    ]
    with (out_path / "aggregate_long.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = identity + [
            "metric",
            "n_seeds",
            "mean",
            "std",
            "sem",
            "ci95_low",
            "ci95_high",
            "min",
            "max",
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            common = {name: group[name] for name in identity}
            for metric_name, stats in group["metrics"].items():
                writer.writerow({**common, "metric": metric_name, **stats})

    metric_names = sorted(
        {name for group in groups for name in group["metrics"].keys()}
    )
    with (out_path / "aggregate.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        fieldnames = identity + [
            f"{name}.{stat}"
            for name in metric_names
            for stat in ("mean", "std", "n_seeds")
        ]
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for group in groups:
            row = {name: group[name] for name in identity}
            for metric_name, stats in group["metrics"].items():
                for stat in ("mean", "std", "n_seeds"):
                    row[f"{metric_name}.{stat}"] = stats[stat]
            writer.writerow(row)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    paths = discover_summaries(args.inputs)
    if not paths:
        raise ValueError("no summary.json artifacts found")
    summaries = [load_summary(path) for path in paths]
    groups = aggregate_summaries(summaries)
    write_aggregate_tables(groups, args.out_dir)
    print(
        json.dumps(
            {
                "summaries": len(paths),
                "groups": len(groups),
                "output": str(Path(args.out_dir).resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
