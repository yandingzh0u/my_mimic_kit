from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import yaml


def rsi_context_starts(
    reset_times: torch.Tensor,
    motion_lengths: torch.Tensor,
    *,
    steps: int = 20,
    control_freq: int = 30,
) -> torch.Tensor:
    """Recover A0 from an RSI state fixed at A19 (A20 is velocity lookahead)."""
    if steps != 20 or control_freq != 30:
        raise ValueError("R2 requires H_A=20 and control_freq=30")
    if reset_times.shape != motion_lengths.shape:
        raise ValueError("reset_times and motion_lengths must have matching shapes")
    required_span = steps / float(control_freq)
    if torch.any(motion_lengths < required_span):
        raise ValueError("motion is too short for A0..A20 without repeated states")
    starts = reset_times - (steps - 1) / float(control_freq)
    max_starts = motion_lengths - required_span
    tolerance = 2e-6
    if torch.any(starts < -tolerance) or torch.any(starts > max_starts + tolerance):
        raise ValueError("reset time is not an A19 state from a valid R2 context")
    return torch.minimum(torch.clamp_min(starts, 0.0), max_starts)


def context_times(starts: torch.Tensor, *, steps: int = 20, control_freq: int = 30):
    offsets = torch.arange(steps, device=starts.device, dtype=starts.dtype)
    return starts.unsqueeze(-1) + offsets.unsqueeze(0) / float(control_freq)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_manifest_sha256(clips: Sequence[Mapping[str, Any]]) -> str:
    payload = json.dumps(
        [dict(clip) for clip in clips],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_dataset_manifest(
    motion_file: str | Path,
    motion_files: Sequence[str | Path],
    lengths_seconds: Sequence[float],
) -> dict[str, Any]:
    """Build the canonical ordered dataset identity used by encoder/flow/runtime."""
    motion_path = Path(motion_file).resolve()
    with motion_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    entries = document.get("motions") if isinstance(document, dict) else None
    if not isinstance(entries, list) or len(entries) != len(motion_files):
        raise ValueError("motion YAML order does not match the loaded MotionLib")
    if len(lengths_seconds) != len(entries):
        raise ValueError("motion lengths do not match the motion YAML")

    clips = []
    for motion_id, (entry, loaded_file, length) in enumerate(
        zip(entries, motion_files, lengths_seconds)
    ):
        declared = Path(entry["file"])
        loaded = Path(loaded_file)
        if declared.resolve() != loaded.resolve():
            raise ValueError(
                "MotionLib order mismatch at motion_id {}: YAML={}, loaded={}".format(
                    motion_id, declared, loaded
                )
            )
        clips.append(
            {
                "motion_id": motion_id,
                "file": declared.as_posix(),
                "weight": float(entry["weight"]),
                "length_seconds": float(length),
                "sha256": sha256_file(declared),
            }
        )
    return {
        "clips": clips,
        "dataset_yaml_sha256": sha256_file(motion_path),
        "canonical_manifest_sha256": canonical_manifest_sha256(clips),
    }


def assert_dataset_manifest_equal(
    artifact: Mapping[str, Any], runtime: Mapping[str, Any], *, length_tol: float = 1e-5
) -> None:
    for key in ("dataset_yaml_sha256", "canonical_manifest_sha256", "clips"):
        if key not in artifact or key not in runtime:
            raise ValueError("dataset_manifest is missing {!r}".format(key))
    a_clips, r_clips = artifact["clips"], runtime["clips"]
    if len(a_clips) != len(r_clips):
        raise ValueError("dataset manifest clip count mismatch")
    for index, (expected, actual) in enumerate(zip(a_clips, r_clips)):
        for key in ("motion_id", "file", "sha256"):
            if expected.get(key) != actual.get(key):
                raise ValueError(
                    "dataset manifest mismatch at clip {} field {}".format(index, key)
                )
        if abs(float(expected["weight"]) - float(actual["weight"])) > 1e-8:
            raise ValueError("dataset manifest weight mismatch at clip {}".format(index))
        if abs(float(expected["length_seconds"]) - float(actual["length_seconds"])) > length_tol:
            raise ValueError("dataset manifest length mismatch at clip {}".format(index))
    for key in ("dataset_yaml_sha256", "canonical_manifest_sha256"):
        if artifact[key] != runtime[key]:
            raise ValueError("dataset manifest {} mismatch".format(key))


def resolve_manifest_clip(
    manifest: Mapping[str, Any], *, motion_path: str | None = None, clip_sha256: str | None = None
) -> dict[str, Any]:
    if (motion_path is None) == (clip_sha256 is None):
        raise ValueError("specify exactly one of motion_path or clip_sha256")
    candidates = []
    for clip in manifest["clips"]:
        if motion_path is not None and Path(clip["file"]).resolve() == Path(motion_path).resolve():
            candidates.append(clip)
        elif clip_sha256 is not None and clip["sha256"] == clip_sha256:
            candidates.append(clip)
    if len(candidates) != 1:
        raise ValueError("skill command must resolve to exactly one manifest clip")
    return dict(candidates[0])
