#!/usr/bin/env python3
"""Prepare, smoke-test, and serially train the CPMD Table-I skill suite.

The suite intentionally contains only the 13 single-clip humanoid skills that
are available locally.  AMASS DanceDB and LaFAN1 are dataset experiments and
are not silently substituted by individual clips.

Examples:
  # Generate frozen per-skill configs and metadata only.
  python tools/cpmd/run_humanoid_table1_suite.py --phase prepare

  # Run every small smoke, then every 8192-env smoke, then serial training.
  python tools/cpmd/run_humanoid_table1_suite.py --phase all

  # Re-run a selected phase for a subset (completed outputs are skipped).
  python tools/cpmd/run_humanoid_table1_suite.py \
      --phase smoke-small --skills run roll climb
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import pickle
import re
import subprocess
import sys
import time

import yaml


REPO_DIR = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = REPO_DIR / "data/experiments/cpmd_humanoid_table1_seed0.yaml"
NONFINITE_METRIC_RE = re.compile(
    r"\|\s*[^|]+\|\s*[+-]?(?:nan|inf)\s*\|", re.IGNORECASE)


def has_nonfinite_metric(line: str) -> bool:
    """Match logger metric values, without false positives such as 'Info'."""
    return NONFINITE_METRIC_RE.search(line) is not None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--phase",
        choices=["prepare", "smoke-small", "smoke-full", "train", "all"],
        default="prepare",
    )
    parser.add_argument("--skills", nargs="*", default=[])
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def repo_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_DIR / path


def load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {path}")
    return value


def local_motion_metadata(path: Path) -> dict:
    # Motion files are trusted repository data and use MimicKit's native pickle
    # format.  This is metadata validation only; simulation loads the same file.
    with path.open("rb") as stream:
        motion = pickle.load(stream)
    fps = float(motion["fps"])
    frames = motion["frames"]
    return {
        "frames": len(frames),
        "fps": fps,
        "length_s": (len(frames) - 1) / fps,
        "loop_mode": int(motion["loop_mode"]),
        "frame_dim": len(frames[0]),
    }


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_manifest(manifest: dict) -> None:
    required = [
        "base_env_config",
        "engine_config",
        "agent_config",
        "rand_seed",
        "num_envs",
        "iterations",
        "steps_per_iter",
        "skills",
    ]
    for key in required:
        if key not in manifest:
            raise KeyError(f"Missing manifest key: {key}")
    if int(manifest["rand_seed"]) != 0:
        raise ValueError("This frozen suite is seed 0 only")
    if int(manifest["num_envs"]) != 8192:
        raise ValueError("This frozen suite requires 8192 environments")
    if int(manifest["iterations"]) != 2000:
        raise ValueError("This frozen suite requires 2000 iterations")

    names = [item["name"] for item in manifest["skills"]]
    if len(names) != 13 or len(names) != len(set(names)):
        raise ValueError("Expected exactly 13 unique single-clip skills")

    for path_key in ["base_env_config", "engine_config", "agent_config"]:
        path = repo_path(manifest[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)


def select_skills(manifest: dict, requested: list[str]) -> list[dict]:
    skills = manifest["skills"]
    if not requested:
        return skills
    known = {item["name"]: item for item in skills}
    unknown = sorted(set(requested) - set(known))
    if unknown:
        raise ValueError(f"Unknown skills: {unknown}")
    requested_set = set(requested)
    return [item for item in skills if item["name"] in requested_set]


def prepare(manifest: dict, skills: list[dict]) -> tuple[Path, dict[str, Path]]:
    suite_root = REPO_DIR / "output" / manifest["suite_name"]
    config_dir = suite_root / "configs"
    config_dir.mkdir(parents=True, exist_ok=True)

    base = load_yaml(repo_path(manifest["base_env_config"]))
    configs: dict[str, Path] = {}
    metadata = {
        "suite_name": manifest["suite_name"],
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "git_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_DIR, text=True
        ).strip(),
        "rand_seed": int(manifest["rand_seed"]),
        "num_envs": int(manifest["num_envs"]),
        "iterations": int(manifest["iterations"]),
        "steps_per_iter": int(manifest["steps_per_iter"]),
        "max_samples_per_skill": (
            int(manifest["num_envs"])
            * int(manifest["steps_per_iter"])
            * int(manifest["iterations"])
        ),
        "excluded_datasets": ["AMASS DanceDB", "LaFAN1 subset"],
        "git_status": subprocess.check_output(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=REPO_DIR,
            text=True,
        ).splitlines(),
        "source_sha256": {},
        "skills": [],
    }
    source_paths = [
        REPO_DIR / "mimickit/envs/cpmd_env.py",
        REPO_DIR / "mimickit/envs/cpmd_obs.py",
        REPO_DIR / "mimickit/learning/add_agent.py",
        REPO_DIR / "mimickit/learning/cpmd_agent.py",
        REPO_DIR / "mimickit/learning/cpmd_model.py",
        REPO_DIR / "data/agents/cpmd_humanoid_agent.yaml",
        repo_path(manifest["base_env_config"]),
        Path(__file__).resolve(),
    ]
    metadata["source_sha256"] = {
        str(path.relative_to(REPO_DIR)): file_sha256(path) for path in source_paths
    }

    for skill in skills:
        motion_path = repo_path(skill["motion_file"])
        if not motion_path.is_file():
            raise FileNotFoundError(motion_path)
        motion_meta = local_motion_metadata(motion_path)
        if motion_meta["frame_dim"] != 34:
            raise ValueError(
                f"{skill['name']}: expected humanoid frame dim 34, "
                f"got {motion_meta['frame_dim']}"
            )

        env = copy.deepcopy(base)
        env["motion_file"] = skill["motion_file"]
        env["contact_bodies"] = skill["contact_bodies"]
        if skill.get("objects"):
            env["objects"] = skill["objects"]
        else:
            env.pop("objects", None)

        config_path = config_dir / f"cpmd_humanoid_{skill['name']}_env.yaml"
        with config_path.open("w", encoding="utf-8") as stream:
            yaml.safe_dump(env, stream, sort_keys=False)
        configs[skill["name"]] = config_path

        metadata["skills"].append(
            {
                "name": skill["name"],
                "display_name": skill["display_name"],
                "motion_file": skill["motion_file"],
                "reported_length_s": float(skill["reported_length_s"]),
                "local_motion": motion_meta,
                "contact_bodies": skill["contact_bodies"],
                "objects": skill.get("objects", []),
                "generated_env_config": str(config_path.relative_to(REPO_DIR)),
            }
        )

    metadata_path = suite_root / "suite_metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2)
        stream.write("\n")
    return suite_root, configs


def phase_spec(manifest: dict, phase: str, skill: dict) -> tuple[int, int, Path]:
    steps = int(manifest["steps_per_iter"])
    if phase == "smoke-small":
        num_envs, iterations = 64, 5
        out_dir = REPO_DIR / "output" / manifest["suite_name"] / "smoke_small" / skill["name"]
    elif phase == "smoke-full":
        num_envs, iterations = int(manifest["num_envs"]), 2
        out_dir = REPO_DIR / "output" / manifest["suite_name"] / "smoke_full" / skill["name"]
    elif phase == "train":
        num_envs, iterations = int(manifest["num_envs"]), int(manifest["iterations"])
        out_dir = REPO_DIR / "output" / f"cpmd_{skill['name']}_2k_seed0"
    else:
        raise ValueError(phase)
    return num_envs, num_envs * steps * iterations, out_dir


def run_one(manifest: dict, skill: dict, config: Path, phase: str, dry_run: bool) -> None:
    num_envs, max_samples, out_dir = phase_spec(manifest, phase, skill)
    marker = out_dir / "suite_complete.json"
    if marker.is_file():
        print(f"[{phase}] {skill['name']}: already complete; skipping", flush=True)
        return
    if out_dir.exists() and any(out_dir.iterdir()):
        raise RuntimeError(
            f"Refusing to overwrite incomplete output {out_dir}. "
            "Inspect or move it before resuming."
        )
    command = [
        sys.executable,
        "mimickit/run.py",
        "--mode", "train",
        "--num_envs", str(num_envs),
        "--engine_config", manifest["engine_config"],
        "--env_config", str(config.relative_to(REPO_DIR)),
        "--agent_config", manifest["agent_config"],
        "--visualize", "false",
        "--out_dir", str(out_dir.relative_to(REPO_DIR)),
        "--rand_seed", str(manifest["rand_seed"]),
        "--max_samples", str(max_samples),
        "--save_int_models", "true" if phase == "train" else "false",
        "--logger", "txt",
    ]
    print(f"[{phase}] {skill['name']}: {' '.join(command)}", flush=True)
    if dry_run:
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")

    start = time.time()
    console_path = out_dir / "console.log"
    with console_path.open("w", encoding="utf-8", buffering=1) as console:
        process = subprocess.Popen(
            command,
            cwd=REPO_DIR,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        nonfinite_line = None
        for line in process.stdout:
            console.write(line)
            # Keep suite-level progress visible without duplicating the very
            # large per-environment build stream on the controlling terminal.
            if ("Iteration" in line or "Wall_Time" in line or "CPMD differential" in line
                    or "Traceback" in line or "Error" in line or "CUDA out of memory" in line):
                print(f"[{skill['name']}] {line.rstrip()}", flush=True)
            if nonfinite_line is None and has_nonfinite_metric(line):
                nonfinite_line = line.strip()
                process.terminate()
        return_code = process.wait()

    elapsed = time.time() - start
    if nonfinite_line is not None:
        raise RuntimeError(
            f"Non-finite training metric for {skill['name']}: {nonfinite_line}; "
            f"see {console_path}"
        )
    if return_code != 0:
        raise RuntimeError(
            f"{phase} failed for {skill['name']} with code {return_code}; "
            f"see {console_path}"
        )
    for required in [out_dir / "model.pt", out_dir / "log.txt", out_dir / "env_config.yaml"]:
        if not required.is_file():
            raise RuntimeError(f"Missing expected output: {required}")
    console_text = console_path.read_text(encoding="utf-8", errors="replace")
    bad_tokens = ["Traceback (most recent call last)", "CUDA out of memory", "RuntimeError:"]
    found = [token for token in bad_tokens if token in console_text]
    if found:
        raise RuntimeError(f"Unhealthy console for {skill['name']}: {found}")

    marker.write_text(
        json.dumps(
            {
                "phase": phase,
                "skill": skill["name"],
                "rand_seed": int(manifest["rand_seed"]),
                "num_envs": num_envs,
                "max_samples": max_samples,
                "elapsed_seconds": elapsed,
                "completed_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            },
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(f"[{phase}] {skill['name']}: complete in {elapsed:.1f}s", flush=True)


def run_phase(manifest: dict, skills: list[dict], configs: dict[str, Path], phase: str,
              dry_run: bool) -> None:
    for skill in skills:
        run_one(manifest, skill, configs[skill["name"]], phase, dry_run)


def main() -> None:
    args = parse_args()
    manifest = load_yaml(args.manifest.resolve())
    validate_manifest(manifest)
    skills = select_skills(manifest, args.skills)
    suite_root, configs = prepare(manifest, skills)
    print(f"Prepared {len(skills)} skills under {suite_root}", flush=True)

    if args.phase == "prepare":
        return
    if args.phase == "all":
        for phase in ["smoke-small", "smoke-full", "train"]:
            run_phase(manifest, skills, configs, phase, args.dry_run)
    else:
        run_phase(manifest, skills, configs, args.phase, args.dry_run)


if __name__ == "__main__":
    main()
