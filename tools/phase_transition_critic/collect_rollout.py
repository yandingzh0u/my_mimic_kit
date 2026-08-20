#!/usr/bin/env python3
"""Collect raw, phase-matched policy/reference transitions for offline audit.

Run this with the Isaac Lab Python environment.  The output deliberately has
no behavioral class labels; collect successful and shortcut checkpoints into
separate files and keep both held out from critic fitting.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICKIT_ROOT = REPO_ROOT / "mimickit"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.phase_transition_critic.rollout_contract import (  # noqa: E402
    SCHEMA_VERSION,
    atomic_savez_compressed,
    validate_transition_bundle,
)
from tools.paper_eval.evaluate_checkpoint import (  # noqa: E402
    _reset_with_protocol,
    resolve_repo_path,
    sha256_file,
)


START_MODES = ("phase0", "random", "grid")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument(
        "--engine-config", default="data/engines/isaac_lab_engine.yaml"
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--start-mode", choices=START_MODES, default="grid")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--master-port", type=int, default=6391)
    parser.add_argument("--out", required=True)
    return parser.parse_args(argv)


def _git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _metadata_json(args: argparse.Namespace, paths: dict[str, Path]) -> str:
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "held_out_critic_evaluation",
        "contains_behavior_labels": False,
        "model_file": str(paths["model"]),
        "model_sha256": sha256_file(paths["model"]),
        "env_config": str(paths["env"]),
        "env_config_sha256": sha256_file(paths["env"]),
        "agent_config": str(paths["agent"]),
        "agent_config_sha256": sha256_file(paths["agent"]),
        "engine_config": str(paths["engine"]),
        "engine_config_sha256": sha256_file(paths["engine"]),
        "git_commit": _git_revision(),
        "num_envs": args.num_envs,
        "requested_steps": args.steps,
        "start_mode": args.start_mode,
        "seed": args.seed,
    }
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"))


def _info_phi(info: dict[str, Any], key: str, num_envs: int):
    import torch

    if key not in info:
        raise ValueError(
            f"environment info does not expose {key!r}; this collector requires "
            "ADD-compatible raw discriminator observations"
        )
    value = info[key]
    if not torch.is_tensor(value) or value.ndim != 2 or value.shape[0] != num_envs:
        raise ValueError(f"{key} must be a [num_envs, phi_dim] tensor")
    return value.detach().clone()


def collect(args: argparse.Namespace) -> dict[str, object]:
    if args.num_envs <= 0 or args.steps <= 0:
        raise ValueError("num-envs and steps must be positive")

    paths = {
        "model": resolve_repo_path(args.model_file),
        "env": resolve_repo_path(args.env_config),
        "agent": resolve_repo_path(args.agent_config),
        "engine": resolve_repo_path(args.engine_config),
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)

    # Simulator imports remain lazy so schema/tests run without Isaac Lab.
    import torch

    if str(MIMICKIT_ROOT) not in sys.path:
        sys.path.insert(0, str(MIMICKIT_ROOT))
    import envs.base_env as base_env
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import learning.base_agent as base_agent
    import util.mp_util as mp_util
    import util.util as util

    original_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        util.set_rand_seed(args.seed)
        mp_util.init(
            rank=0,
            num_procs=1,
            device=args.device,
            master_port=args.master_port,
        )
        env = env_builder.build_env(
            str(paths["env"]),
            str(paths["engine"]),
            args.num_envs,
            args.device,
            visualize=False,
            record_video=False,
        )
        agent = agent_builder.build_agent(str(paths["agent"]), env, args.device)
        agent.load(str(paths["model"]))
        agent.eval()
        agent.set_mode(base_agent.AgentMode.TEST)

        obs, info = _reset_with_protocol(env, args.start_mode)
        num_envs = env.get_num_envs()
        dt = float(env._engine.get_timestep())
        motion_lengths = env._motion_lib.get_motion_length(env._motion_ids)
        loop_modes = env._motion_lib.get_motion_loop_mode(env._motion_ids)
        wrap_mask = loop_modes == 1
        if bool(torch.all(wrap_mask).item()):
            env._episode_length = max(float(env._episode_length), args.steps * dt)

        alive = torch.ones(num_envs, device=args.device, dtype=torch.bool)
        episode_ids = torch.arange(
            num_envs, device=args.device, dtype=torch.int64
        )
        records: dict[str, list[np.ndarray]] = {
            key: []
            for key in (
                "x_t",
                "x_t1",
                "r_t",
                "r_t1",
                "episode_id",
                "step_index",
                "phase",
                "alive",
            )
        }

        with torch.no_grad():
            for step in range(args.steps):
                active_before = alive.clone()
                x_t = _info_phi(info, "disc_obs", num_envs)
                r_t = _info_phi(info, "disc_obs_demo", num_envs)
                phase = env._motion_lib.calc_motion_phase(
                    env._motion_ids, env._get_motion_times()
                )
                action, _ = agent._decide_action(obs, info)
                action = torch.where(
                    active_before.unsqueeze(-1), action, torch.zeros_like(action)
                )
                next_obs, _, done, next_info = env.step(action)
                x_t1 = _info_phi(next_info, "disc_obs", num_envs)
                r_t1 = _info_phi(next_info, "disc_obs_demo", num_envs)

                tensor_rows = {
                    "x_t": x_t,
                    "x_t1": x_t1,
                    "r_t": r_t,
                    "r_t1": r_t1,
                    "episode_id": episode_ids,
                    "step_index": torch.full_like(episode_ids, step),
                    "phase": phase,
                    "alive": active_before,
                }
                for key, value in tensor_rows.items():
                    records[key].append(value.detach().cpu().numpy())

                alive = torch.logical_and(
                    alive, done == base_env.DoneFlags.NULL.value
                )
                obs, info = next_obs, next_info
                if not torch.any(alive):
                    break

        arrays = {
            key: np.concatenate(value, axis=0) for key, value in records.items()
        }
        arrays["schema_version"] = np.asarray(SCHEMA_VERSION, dtype=np.int64)
        arrays["metadata_json"] = np.asarray(_metadata_json(args, paths))
        report = validate_transition_bundle(arrays)
        atomic_savez_compressed(resolve_repo_path(args.out), **arrays)
        return report
    finally:
        os.chdir(original_cwd)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = collect(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
