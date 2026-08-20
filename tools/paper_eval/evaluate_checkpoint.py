#!/usr/bin/env python3
"""Method-independent checkpoint evaluation for the paper benchmark.

The evaluator deliberately measures physical simulator/reference states rather
than reusing a method's training reward or its online diagnostic accumulator.
This gives DeepMimic, AMP, ADD, and RCCI exactly the same episode protocol and
metric implementation.  A run writes three auditable artifacts:

``summary.json``
    Human-readable metadata, episode-distribution summaries, completion rates,
    and efficiency measurements.
``episodes.npz``
    One raw value per evaluation environment/episode.
``timeseries.npz``
    Per-step, per-episode physical errors and progress traces for figures.

Simulator imports are intentionally lazy.  The metadata/parser helpers can be
unit-tested on machines without Isaac Lab.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Any, Mapping

import numpy as np
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICKIT_ROOT = REPO_ROOT / "mimickit"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.paper_eval.metrics import (  # noqa: E402
    CompletionThresholds,
    TRACKING_ERROR_NAMES,
    canonical_motion_name,
    compute_completion,
    compute_tracking_errors,
    projected_motion_metrics,
    quaternion_angle,
    quaternion_up_dot,
    signed_winding_ratio,
)
from tools.paper_eval.input_geometry import InputGeometryAccumulator  # noqa: E402


SCHEMA_VERSION = 1
CONDITIONS = ("nominal", "zero_e", "zero_m", "shuffle_m", "reverse_e")
START_MODES = ("phase0", "random", "grid")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate one DeepMimic/AMP/ADD/RCCI checkpoint under a "
        "shared physical protocol."
    )
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument(
        "--engine-config", default="data/engines/isaac_lab_engine.yaml"
    )
    parser.add_argument(
        "--method",
        default="",
        help="Paper label; inferred from the agent/environment when omitted.",
    )
    parser.add_argument(
        "--motion",
        default="",
        help="Canonical motion label; inferred from motion_name/motion_file.",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument(
        "--steps",
        type=int,
        default=300,
        help=(
            "Maximum control steps (300 = 10 s at the benchmark's 30 Hz); "
            "0 uses ceil(environment episode_length / dt). Non-looping motions "
            "still end at their natural reference endpoint."
        ),
    )
    parser.add_argument("--start-mode", choices=START_MODES, default="phase0")
    parser.add_argument("--condition", choices=CONDITIONS, default="nominal")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--master-port", type=int, default=6381)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args(argv)


def resolve_repo_path(path: str | Path) -> Path:
    value = Path(path).expanduser()
    if not value.is_absolute():
        value = REPO_ROOT / value
    return value.resolve()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while True:
            block = stream.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"configuration is not a mapping: {path}")
    return value


def infer_method(
    agent_config: Mapping[str, Any], env_config: Mapping[str, Any]
) -> str:
    agent_name = str(agent_config.get("agent_name", "")).upper()
    env_name = str(env_config.get("env_name", "")).lower()
    if agent_name == "PPO" and env_name == "deepmimic":
        return "DeepMimic"
    if agent_name == "AMP":
        return "AMP"
    if agent_name == "ADD":
        return "ADD"
    if agent_name in ("RCCI_ADD", "ALIGNED_ADD"):
        return "RCCI" if agent_name == "RCCI_ADD" else "AlignedADD"
    return agent_name or env_name or "unknown"


def infer_motion(env_config: Mapping[str, Any]) -> str:
    if env_config.get("motion_name"):
        return canonical_motion_name(str(env_config["motion_name"]))
    motion_file = env_config.get("motion_file")
    if not motion_file:
        raise ValueError("motion requires env motion_name or motion_file")
    return canonical_motion_name(str(motion_file))


def parse_training_log(
    log_file: str | Path, requested_iteration: int | None = None
) -> dict[str, Any]:
    """Read the matching/latest training row, including appended log segments.

    MimicKit's text logger uses carriage returns and repeats the header when a
    resumed run appends to the file.  ``splitlines`` handles both, while this
    parser always binds data rows to the most recent header.
    """

    path = Path(log_file)
    if not path.is_file():
        return {}

    rows: list[dict[str, str]] = []
    header: list[str] | None = None
    text = path.read_text(encoding="utf-8", errors="replace")
    for raw_line in text.splitlines():
        tokens = raw_line.strip().split()
        if not tokens:
            continue
        if tokens[0] == "Iteration":
            header = tokens
            continue
        if header is None:
            continue
        try:
            int(float(tokens[0]))
        except ValueError:
            continue
        if len(tokens) < len(header):
            continue
        rows.append(dict(zip(header, tokens[: len(header)])))

    if not rows:
        return {}
    if requested_iteration is None:
        row = max(rows, key=lambda item: int(float(item["Iteration"])))
    else:
        matches = [
            item
            for item in rows
            if int(float(item["Iteration"])) == requested_iteration
        ]
        if not matches:
            return {"iteration": requested_iteration, "source": "filename"}
        row = matches[-1]

    result: dict[str, Any] = {
        "iteration": int(float(row["Iteration"])),
        "source": "log",
        "log_file": str(path.resolve()),
    }
    if "Samples" in row:
        result["samples"] = int(float(row["Samples"]))
    if "Wall_Time" in row:
        result["wall_time_hours"] = float(row["Wall_Time"])
    if "Samples_Per_Second" in row:
        result["samples_per_second"] = float(row["Samples_Per_Second"])
    return result


def infer_checkpoint_metadata(model_file: str | Path) -> dict[str, Any]:
    path = Path(model_file)
    match = re.search(r"model[_-](\d+)\.pt$", path.name)
    requested_iteration = int(match.group(1)) if match else None

    candidates = [path.parent / "log.txt"]
    if path.parent.name == "int_models":
        candidates.append(path.parent.parent / "log.txt")
    for candidate in candidates:
        metadata = parse_training_log(candidate, requested_iteration)
        if metadata:
            return metadata

    if requested_iteration is not None:
        return {"iteration": requested_iteration, "source": "filename"}
    return {"iteration": None, "samples": None, "source": "unavailable"}


def git_metadata(repo_root: str | Path = REPO_ROOT) -> dict[str, Any]:
    root = Path(repo_root)
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        return {"commit": revision, "dirty": bool(status.strip())}
    except (OSError, subprocess.CalledProcessError):
        return {"commit": None, "dirty": None}


def distribution_stats(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = array[np.isfinite(array)]
    result: dict[str, Any] = {
        "n": int(array.size),
        "finite_n": int(finite.size),
        "nan_inf_count": int(array.size - finite.size),
    }
    if finite.size == 0:
        result.update(
            mean=None,
            std=None,
            median=None,
            q10=None,
            q90=None,
            min=None,
            max=None,
        )
        return result
    std = float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
    result.update(
        mean=float(np.mean(finite)),
        std=std,
        median=float(np.median(finite)),
        q10=float(np.quantile(finite, 0.10)),
        q90=float(np.quantile(finite, 0.90)),
        min=float(np.min(finite)),
        max=float(np.max(finite)),
    )
    return result


def _query_reference(env) -> dict[str, torch.Tensor]:
    """Query the reference at the actual clock without trusting cached arrays.

    AMP intentionally leaves ``_ref_*`` unchanged in non-visual rollouts.  A
    direct motion-library query is therefore essential for a fair physical
    evaluation shared by AMP and tracking methods.
    """

    motion_times = env._get_motion_times()
    (
        root_pos,
        root_rot,
        root_vel,
        root_ang_vel,
        joint_rot,
        dof_vel,
    ) = env._motion_lib.calc_motion_frame(env._motion_ids, motion_times)
    body_pos, body_rot = env._kin_char_model.forward_kinematics(
        root_pos, root_rot, joint_rot
    )
    return {
        "root_pos": root_pos,
        "root_rot": root_rot,
        "root_vel": root_vel,
        "root_ang_vel": root_ang_vel,
        "joint_rot": joint_rot,
        "dof_vel": dof_vel,
        "body_pos": body_pos,
        "body_rot": body_rot,
    }


def _query_sim(env) -> dict[str, torch.Tensor]:
    char_id = env._get_char_id()
    return {
        "root_pos": env._engine.get_root_pos(char_id),
        "root_rot": env._engine.get_root_rot(char_id),
        "root_vel": env._engine.get_root_vel(char_id),
        "root_ang_vel": env._engine.get_root_ang_vel(char_id),
        "dof_vel": env._engine.get_dof_vel(char_id),
        "body_pos": env._engine.get_body_pos(char_id),
        "body_rot": env._engine.get_body_rot(char_id),
    }


def _copy_reference_cache(env, ref: Mapping[str, torch.Tensor]) -> None:
    env._ref_root_pos[:] = ref["root_pos"]
    env._ref_root_rot[:] = ref["root_rot"]
    env._ref_root_vel[:] = ref["root_vel"]
    env._ref_root_ang_vel[:] = ref["root_ang_vel"]
    env._ref_joint_rot[:] = ref["joint_rot"]
    env._ref_dof_vel[:] = ref["dof_vel"]
    env._ref_body_pos[:] = ref["body_pos"]
    env._ref_body_rot[:] = ref["body_rot"]
    env._ref_dof_pos[:] = env._motion_lib.joint_rot_to_dof(ref["joint_rot"])


def _reset_with_protocol(env, start_mode: str):
    if start_mode == "random":
        env._rand_reset = True
        return env.reset()

    env._rand_reset = False
    obs, info = env.reset()
    if start_mode == "phase0":
        return obs, info

    # Evenly cover the orbit while preserving deterministic simulator/reference
    # initialization.  Configurations in the paper matrix contain one motion;
    # this also works for sampled multi-motion IDs because each gets its length.
    num_envs = env.get_num_envs()
    phase = torch.arange(
        num_envs, device=env._device, dtype=torch.float32
    ) / float(num_envs)
    lengths = env._motion_lib.get_motion_length(env._motion_ids)
    env._motion_time_offsets[:] = phase * lengths
    env._time_buf.zero_()
    env._timestep_buf.zero_()
    ref = _query_reference(env)
    _copy_reference_cache(env, ref)

    env_ids = env._env_ids
    env._ref_state_init(env_ids)
    env._reset_char_rigid_body_state(env_ids)
    if hasattr(env, "_reset_disc_hist"):
        env._reset_disc_hist(env_ids)
    env._update_observations()
    env._update_info()
    return env._obs_buf, env._info


def _command_layout(env) -> tuple[str, int, int]:
    if hasattr(env, "get_rcci_phi_dim"):
        representation = env.get_rcci_representation()
        if representation != "residual":
            return "rcci_absolute", env.get_rcci_self_obs_dim(), env.get_rcci_phi_dim()
        return "rcci_residual", env.get_rcci_self_obs_dim(), env.get_rcci_phi_dim()
    if hasattr(env, "get_aligned_command_dim"):
        return "aligned", env.get_aligned_self_obs_dim(), env.get_aligned_command_dim()
    return "none", 0, 0


def _input_geometry_blocks(
    layout: str, self_dim: int, phi_dim: int, obs_dim: int
) -> tuple[dict[str, slice], tuple[tuple[str, str], ...]]:
    if layout == "aligned":
        return (
            {
                "self": slice(0, self_dim),
                "feedback_error": slice(self_dim, self_dim + phi_dim),
                "feedforward_motion": slice(
                    self_dim + phi_dim, self_dim + 2 * phi_dim
                ),
            },
            (("feedback_error", "feedforward_motion"),),
        )
    if layout in ("rcci_residual", "rcci_absolute"):
        names = (
            ("state_feature", "feedback_error", "feedforward_motion")
            if layout == "rcci_residual"
            else ("state_feature", "reference_current", "reference_next")
        )
        blocks = {"self": slice(0, self_dim)}
        for index, name in enumerate(names):
            start = self_dim + index * phi_dim
            blocks[name] = slice(start, start + phi_dim)
        paired = (
            ((names[1], names[2]),)
            if names[1] != "feedback_error"
            else (("feedback_error", "feedforward_motion"),)
        )
        return blocks, paired
    return {"actor_input": slice(0, obs_dim)}, ()


def _slice_info_batch(info: Mapping[str, Any], size: int) -> dict[str, Any]:
    sliced: dict[str, Any] = {}
    for key, value in info.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] >= size:
            sliced[key] = value[:size]
        else:
            sliced[key] = value
    return sliced


def _measure_batch1_policy_latency(
    agent, obs: torch.Tensor, info: Mapping[str, Any], device: str,
    warmup: int = 10, repeats: int = 100,
) -> float:
    obs1 = obs[:1]
    info1 = _slice_info_batch(info, 1)
    with torch.no_grad():
        for _ in range(warmup):
            agent._decide_action(obs1, info1)
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
        start = time.perf_counter()
        for _ in range(repeats):
            agent._decide_action(obs1, info1)
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.synchronize(device)
    return 1e6 * (time.perf_counter() - start) / repeats


def intervene_observation(
    obs: torch.Tensor,
    condition: str,
    layout: str,
    self_dim: int,
    phi_dim: int,
    shuffle_index: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply an inference-only command intervention to a raw observation."""

    if condition == "nominal":
        return obs
    if layout not in ("aligned", "rcci_residual"):
        raise ValueError(
            f"{condition} requires a residual [e,m] policy interface, got {layout}"
        )

    if layout == "aligned":
        e_start = self_dim
    else:
        # RCCI raw observation is [self, x_t, e_t, m_t].
        e_start = self_dim + phi_dim
    m_start = e_start + phi_dim
    if obs.shape[-1] < m_start + phi_dim:
        raise ValueError("observation is too short for the declared command layout")

    result = obs.clone()
    e_slice = slice(e_start, e_start + phi_dim)
    m_slice = slice(m_start, m_start + phi_dim)
    if condition == "zero_e":
        result[..., e_slice] = 0.0
    elif condition == "zero_m":
        result[..., m_slice] = 0.0
    elif condition == "reverse_e":
        result[..., e_slice] = -result[..., e_slice]
    elif condition == "shuffle_m":
        if shuffle_index is None:
            raise ValueError("shuffle_m requires a shuffle index")
        motion = result[..., m_slice].clone()
        result[..., m_slice] = motion[shuffle_index]
    else:
        raise ValueError(f"unsupported intervention: {condition}")
    return result


def _policy_reward(
    agent,
    obs: torch.Tensor,
    info,
    env_reward: torch.Tensor,
) -> torch.Tensor:
    """Reconstruct the reward optimized by each method for diagnostics only."""

    transition_reward = getattr(
        agent, "calc_policy_reward_from_transition", None
    )
    if callable(transition_reward):
        # Transition-conditioned rewards require the raw pre-step observation
        # together with post-step info. Normalizing or replacing ``obs`` here
        # would break exact reconstruction of (x_t, x_{t+1}).
        return transition_reward(obs, info, env_reward)

    if not hasattr(agent, "_calc_disc_rewards") or "disc_obs" not in info:
        return env_reward

    if "disc_obs_demo" in info:
        disc_input = info["disc_obs_demo"] - info["disc_obs"]
    else:
        disc_input = info["disc_obs"]
    norm_disc_input = agent._disc_obs_norm.normalize(disc_input)
    disc_reward = agent._calc_disc_rewards(norm_disc_input)
    task_weight = float(getattr(agent, "_task_reward_weight", 0.0))
    disc_weight = float(getattr(agent, "_disc_reward_weight", 1.0))
    return task_weight * env_reward + disc_weight * disc_reward


def _body_ids(env, names: tuple[str, ...]) -> list[int]:
    available = set(env._kin_char_model.get_body_names())
    return [env._kin_char_model.get_body_id(name) for name in names if name in available]


def _to_numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy()


def _mean_per_episode(
    total: torch.Tensor, count: torch.Tensor
) -> torch.Tensor:
    return total / torch.clamp_min(count, 1.0)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temp, path)


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")
    if args.steps < 0:
        raise ValueError("steps must be nonnegative")
    if args.condition == "shuffle_m" and args.num_envs < 2:
        raise ValueError("shuffle_m requires at least two environments")
    if args.condition == "shuffle_m" and args.start_mode == "phase0":
        raise ValueError(
            "shuffle_m is identical at a shared phase0; use random or grid starts"
        )

    model_file = resolve_repo_path(args.model_file)
    env_file = resolve_repo_path(args.env_config)
    agent_file = resolve_repo_path(args.agent_config)
    engine_file = resolve_repo_path(args.engine_config)
    out_dir = resolve_repo_path(args.out_dir)
    for path in (model_file, env_file, agent_file, engine_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    env_config = load_yaml(env_file)
    agent_config = load_yaml(agent_file)
    method = args.method or infer_method(agent_config, env_config)
    motion_name = canonical_motion_name(args.motion) if args.motion else infer_motion(env_config)

    # MimicKit uses top-level imports such as ``envs.*``.
    if str(MIMICKIT_ROOT) not in sys.path:
        sys.path.insert(0, str(MIMICKIT_ROOT))
    import envs.base_env as base_env  # noqa: WPS433
    import envs.env_builder as env_builder  # noqa: WPS433
    import learning.agent_builder as agent_builder  # noqa: WPS433
    import learning.base_agent as base_agent  # noqa: WPS433
    import util.mp_util as mp_util  # noqa: WPS433
    import util.util as util  # noqa: WPS433

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
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            # torch 2.7 + Blackwell accepts only the current-device form here,
            # while synchronize/max_memory_allocated still accept cuda:0.
            torch.cuda.reset_peak_memory_stats()

        env = env_builder.build_env(
            str(env_file),
            str(engine_file),
            args.num_envs,
            args.device,
            visualize=False,
            record_video=False,
        )
        agent = agent_builder.build_agent(str(agent_file), env, args.device)
        agent.load(str(model_file))
        agent.eval()
        agent.set_mode(base_agent.AgentMode.TEST)

        obs, info = _reset_with_protocol(env, args.start_mode)
        layout, self_dim, phi_dim = _command_layout(env)
        if args.condition != "nominal" and layout not in (
            "aligned",
            "rcci_residual",
        ):
            raise ValueError(
                f"condition {args.condition} is unsupported for {layout} observations"
            )

        geometry_blocks, geometry_pairs = _input_geometry_blocks(
            layout, self_dim, phi_dim, int(obs.shape[-1])
        )
        input_geometry = InputGeometryAccumulator(
            geometry_blocks, args.device, geometry_pairs
        )

        num_envs = env.get_num_envs()
        dt = float(env._engine.get_timestep())
        steps = args.steps or int(math.ceil(float(env._episode_length) / dt))
        if steps <= 0:
            raise ValueError("evaluation horizon resolved to zero steps")

        motion_lengths = env._motion_lib.get_motion_length(env._motion_ids)
        motion_loop_modes = env._motion_lib.get_motion_loop_mode(env._motion_ids)
        wrap_mask = motion_loop_modes == 1
        remaining_motion_steps = torch.ceil(
            torch.clamp_min(motion_lengths - env._motion_time_offsets, 0.0) / dt
        ).long()
        expected_steps = torch.where(
            wrap_mask,
            torch.full_like(remaining_motion_steps, steps),
            torch.minimum(
                remaining_motion_steps,
                torch.full_like(remaining_motion_steps, steps),
            ),
        )
        expected_steps = torch.clamp_min(expected_steps, 1)
        original_episode_length = float(env._episode_length)
        if bool(torch.all(wrap_mask).item()):
            # Wrapped skills are evaluated continuously for the full 10-second
            # protocol even if their training YAML used a shorter episode.
            env._episode_length = max(original_episode_length, steps * dt)

        shuffle_index = torch.roll(
            torch.arange(num_envs, device=args.device, dtype=torch.long), shifts=1
        )
        alive = torch.ones(num_envs, device=args.device, dtype=torch.bool)
        first_done_step = torch.full(
            (num_envs,), steps, device=args.device, dtype=torch.long
        )
        done_reason = torch.full(
            (num_envs,), base_env.DoneFlags.NULL.value, device=args.device, dtype=torch.int
        )
        step_count = torch.zeros(num_envs, device=args.device)
        error_sum = {
            name: torch.zeros(num_envs, device=args.device)
            for name in TRACKING_ERROR_NAMES
        }
        env_reward_sum = torch.zeros(num_envs, device=args.device)
        policy_reward_sum = torch.zeros(num_envs, device=args.device)
        action_delta_sum = torch.zeros(num_envs, device=args.device)

        sim0 = _query_sim(env)
        ref0 = _query_reference(env)
        sim_root_pos0 = sim0["root_pos"].clone()
        ref_root_pos0 = ref0["root_pos"].clone()
        sim_root_pos1 = sim_root_pos0.clone()
        ref_root_pos1 = ref_root_pos0.clone()
        sim_root_rot1 = sim0["root_rot"].clone()
        ref_root_rot1 = ref0["root_rot"].clone()
        sim_ang_integral = torch.zeros_like(sim_root_pos0)
        ref_ang_integral = torch.zeros_like(ref_root_pos0)
        prev_sim_ang_vel = sim0["root_ang_vel"].clone()
        prev_ref_ang_vel = ref0["root_ang_vel"].clone()
        sim_max_height = sim_root_pos0[:, 2].clone()
        ref_max_height = ref_root_pos0[:, 2].clone()

        foot_ids = _body_ids(env, ("right_foot", "left_foot"))
        if foot_ids:
            sim_feet0 = sim0["body_pos"][:, foot_ids, :].clone()
            ref_feet0 = ref0["body_pos"][:, foot_ids, :].clone()
            sim_feet1 = sim_feet0.clone()
            ref_feet1 = ref_feet0.clone()
        else:
            sim_feet0 = torch.empty(num_envs, 0, 3, device=args.device)
            ref_feet0 = torch.empty_like(sim_feet0)
            sim_feet1 = torch.empty_like(sim_feet0)
            ref_feet1 = torch.empty_like(sim_feet0)

        ts_alive: list[np.ndarray] = []
        ts_errors: list[np.ndarray] = []
        ts_sim_root: list[np.ndarray] = []
        ts_ref_root: list[np.ndarray] = []
        ts_winding: list[np.ndarray] = []
        ts_displacement: list[np.ndarray] = []
        decision_seconds = 0.0
        eval_start = time.perf_counter()

        with torch.no_grad():
            for step in range(steps):
                active_before = alive.clone()
                nominal_obs = obs
                normalized_nominal_obs = agent._obs_norm.normalize(nominal_obs)
                input_geometry.update(normalized_nominal_obs, active_before)
                actor_obs = intervene_observation(
                    nominal_obs,
                    args.condition,
                    layout,
                    self_dim,
                    phi_dim,
                    shuffle_index,
                )

                if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
                    torch.cuda.synchronize(args.device)
                decision_start = time.perf_counter()
                nominal_action, _ = agent._decide_action(nominal_obs, info)
                if args.condition == "nominal":
                    action = nominal_action
                else:
                    action, _ = agent._decide_action(actor_obs, info)
                if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
                    torch.cuda.synchronize(args.device)
                decision_seconds += time.perf_counter() - decision_start

                action_delta = torch.linalg.vector_norm(
                    action - nominal_action, dim=-1
                )
                action_delta_sum += torch.where(
                    active_before, action_delta, torch.zeros_like(action_delta)
                )
                action = torch.where(active_before.unsqueeze(-1), action, torch.zeros_like(action))
                obs, env_reward, done, info = env.step(action)
                policy_reward = _policy_reward(
                    agent, nominal_obs, info, env_reward
                )

                sim = _query_sim(env)
                ref = _query_reference(env)
                errors = compute_tracking_errors(
                    root_pos=sim["root_pos"],
                    root_rot=sim["root_rot"],
                    body_pos=sim["body_pos"],
                    body_rot=sim["body_rot"],
                    dof_vel=sim["dof_vel"],
                    root_vel=sim["root_vel"],
                    root_ang_vel=sim["root_ang_vel"],
                    ref_root_pos=ref["root_pos"],
                    ref_root_rot=ref["root_rot"],
                    ref_body_pos=ref["body_pos"],
                    ref_body_rot=ref["body_rot"],
                    ref_dof_vel=ref["dof_vel"],
                    ref_root_vel=ref["root_vel"],
                    ref_root_ang_vel=ref["root_ang_vel"],
                )
                active_float = active_before.float()
                step_count += active_float
                env_reward_sum += env_reward * active_float
                policy_reward_sum += policy_reward * active_float
                for name, value in errors.items():
                    error_sum[name] += value * active_float

                sim_ang_integral += (
                    0.5 * (prev_sim_ang_vel + sim["root_ang_vel"]) * dt
                    * active_float.unsqueeze(-1)
                )
                ref_ang_integral += (
                    0.5 * (prev_ref_ang_vel + ref["root_ang_vel"]) * dt
                    * active_float.unsqueeze(-1)
                )
                prev_sim_ang_vel = sim["root_ang_vel"].clone()
                prev_ref_ang_vel = ref["root_ang_vel"].clone()
                sim_root_pos1 = torch.where(
                    active_before.unsqueeze(-1), sim["root_pos"], sim_root_pos1
                )
                ref_root_pos1 = torch.where(
                    active_before.unsqueeze(-1), ref["root_pos"], ref_root_pos1
                )
                sim_root_rot1 = torch.where(
                    active_before.unsqueeze(-1), sim["root_rot"], sim_root_rot1
                )
                ref_root_rot1 = torch.where(
                    active_before.unsqueeze(-1), ref["root_rot"], ref_root_rot1
                )
                sim_max_height = torch.where(
                    active_before,
                    torch.maximum(sim_max_height, sim["root_pos"][:, 2]),
                    sim_max_height,
                )
                ref_max_height = torch.where(
                    active_before,
                    torch.maximum(ref_max_height, ref["root_pos"][:, 2]),
                    ref_max_height,
                )
                if foot_ids:
                    sim_feet1 = torch.where(
                        active_before[:, None, None],
                        sim["body_pos"][:, foot_ids, :],
                        sim_feet1,
                    )
                    ref_feet1 = torch.where(
                        active_before[:, None, None],
                        ref["body_pos"][:, foot_ids, :],
                        ref_feet1,
                    )

                newly_done = torch.logical_and(
                    active_before, done != base_env.DoneFlags.NULL.value
                )
                first_done_step[newly_done] = step + 1
                done_reason[newly_done] = done[newly_done]
                alive = torch.logical_and(
                    alive, done == base_env.DoneFlags.NULL.value
                )

                sim_disp_now = sim_root_pos1 - sim_root_pos0
                ref_disp_now = ref_root_pos1 - ref_root_pos0
                disp_ratio_now, _, _ = projected_motion_metrics(
                    sim_disp_now, ref_disp_now
                )
                winding_now = signed_winding_ratio(
                    sim_ang_integral, ref_ang_integral
                )
                stacked_errors = torch.stack(
                    [errors[name] for name in TRACKING_ERROR_NAMES], dim=-1
                )
                stacked_errors = torch.where(
                    active_before.unsqueeze(-1),
                    stacked_errors,
                    torch.full_like(stacked_errors, torch.nan),
                )
                ts_alive.append(_to_numpy(active_before))
                ts_errors.append(_to_numpy(stacked_errors))
                ts_sim_root.append(_to_numpy(sim_root_pos1))
                ts_ref_root.append(_to_numpy(ref_root_pos1))
                ts_winding.append(_to_numpy(winding_now))
                ts_displacement.append(_to_numpy(disp_ratio_now))

                if not torch.any(alive):
                    break

        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            torch.cuda.synchronize(args.device)
        eval_seconds = time.perf_counter() - eval_start
        batch1_policy_latency_us = _measure_batch1_policy_latency(
            agent, obs, info, args.device
        )

        sim_delta = sim_root_pos1 - sim_root_pos0
        ref_delta = ref_root_pos1 - ref_root_pos0
        displacement_ratio, lateral_displacement, lateral_ratio = projected_motion_metrics(
            sim_delta, ref_delta
        )
        winding_ratio = signed_winding_ratio(
            sim_ang_integral, ref_ang_integral
        )
        survival_ratio = torch.clamp_max(
            first_done_step.float() / expected_steps.float(), 1.0
        )
        final_up_dot = quaternion_up_dot(sim_root_rot1)
        ref_final_up_dot = quaternion_up_dot(ref_root_rot1)
        final_root_rot_error = quaternion_angle(
            sim_root_rot1, ref_root_rot1
        )
        final_height_ratio = sim_root_pos1[:, 2] / torch.clamp_min(
            ref_root_pos1[:, 2], 1e-6
        )
        sim_height_gain = sim_max_height - sim_root_pos0[:, 2]
        ref_height_gain = ref_max_height - ref_root_pos0[:, 2]
        max_height_gain_ratio = sim_height_gain / torch.clamp_min(
            ref_height_gain, 1e-6
        )
        final_height_error = torch.abs(
            sim_root_pos1[:, 2] - ref_root_pos1[:, 2]
        )

        if foot_ids:
            sim_foot_delta = sim_feet1 - sim_feet0
            ref_foot_delta = ref_feet1 - ref_feet0
            foot_ref_sq = torch.sum(ref_foot_delta * ref_foot_delta, dim=-1)
            foot_ratio = torch.sum(
                sim_foot_delta * ref_foot_delta, dim=-1
            ) / torch.clamp_min(foot_ref_sq, 1e-8)
            feet_progress_ratio = torch.min(foot_ratio, dim=-1).values
        else:
            feet_progress_ratio = torch.full(
                (num_envs,), torch.nan, device=args.device
            )

        episode_tracking = {
            name: _mean_per_episode(error_sum[name], step_count)
            for name in TRACKING_ERROR_NAMES
        }
        behavior = {
            "survival_ratio": survival_ratio,
            "displacement_ratio": displacement_ratio,
            "lateral_displacement": lateral_displacement,
            "lateral_displacement_ratio": lateral_ratio,
            "winding_ratio": winding_ratio,
            "final_up_dot": final_up_dot,
            "reference_final_up_dot": ref_final_up_dot,
            "final_root_rot_error": final_root_rot_error,
            "final_height_ratio": final_height_ratio,
            "max_height_gain_ratio": max_height_gain_ratio,
            "final_height_error": final_height_error,
            "feet_progress_ratio": feet_progress_ratio,
        }
        completion, completion_components = compute_completion(
            motion_name, behavior, CompletionThresholds()
        )
        completion_protocol_valid = args.start_mode == "phase0"

        episode_arrays: dict[str, np.ndarray] = {
            "episode_id": np.arange(num_envs, dtype=np.int64),
            "motion_id": _to_numpy(env._motion_ids),
            "start_phase": _to_numpy(
                env._motion_lib.calc_motion_phase(
                    env._motion_ids, env._motion_time_offsets
                )
            ),
            "step_count": _to_numpy(step_count),
            "expected_step_count": _to_numpy(expected_steps),
            "first_done_step": _to_numpy(first_done_step),
            "done_reason": _to_numpy(done_reason),
            "env_reward_mean": _to_numpy(
                _mean_per_episode(env_reward_sum, step_count)
            ),
            "policy_reward_mean": _to_numpy(
                _mean_per_episode(policy_reward_sum, step_count)
            ),
            "action_intervention_l2_mean": _to_numpy(
                _mean_per_episode(action_delta_sum, step_count)
            ),
            "completion": _to_numpy(completion),
        }
        for name, value in episode_tracking.items():
            episode_arrays[name] = _to_numpy(value)
        for name, value in behavior.items():
            episode_arrays[name] = _to_numpy(value)
        for name, value in completion_components.items():
            episode_arrays[f"completion_{name}"] = _to_numpy(value)

        done_counts = {
            flag.name.lower(): int(
                torch.sum(done_reason == flag.value).detach().cpu().item()
            )
            for flag in base_env.DoneFlags
        }
        checkpoint_metadata = infer_checkpoint_metadata(model_file)
        config_hashes = {
            "environment_sha256": sha256_file(env_file),
            "agent_sha256": sha256_file(agent_file),
            "engine_sha256": sha256_file(engine_file),
        }
        representation = (
            env.get_rcci_representation()
            if hasattr(env, "get_rcci_representation")
            else None
        )
        executed_steps = len(ts_alive)
        evaluated_samples = int(torch.sum(step_count).detach().cpu().item())
        peak_gpu_memory_mb = 0.0
        gpu_name = None
        if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
            peak_gpu_memory_mb = float(
                torch.cuda.max_memory_allocated(args.device) / (1024 * 1024)
            )
            gpu_name = torch.cuda.get_device_name(args.device)

        summary: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "metadata": {
                "method": method,
                "motion": motion_name,
                "representation": representation,
                "condition": args.condition,
                "seed": args.seed,
                "model_file": str(model_file),
                "model_sha256": sha256_file(model_file),
                "environment_config": str(env_file),
                "agent_config": str(agent_file),
                "engine_config": str(engine_file),
                "config_hashes": config_hashes,
                "checkpoint": checkpoint_metadata,
                "git": git_metadata(),
                "torch_version": torch.__version__,
                "device": args.device,
                "gpu_name": gpu_name,
            },
            "protocol": {
                "num_episodes": num_envs,
                "requested_steps": steps,
                "executed_steps": executed_steps,
                "timestep_seconds": dt,
                "horizon_seconds": steps * dt,
                "original_environment_episode_seconds": original_episode_length,
                "wrapped_episode_extended_to_horizon": bool(
                    torch.all(wrap_mask).item()
                    and original_episode_length < steps * dt
                ),
                "start_mode": args.start_mode,
                "deterministic_mean_action": True,
                "no_automatic_reset": True,
                "completion_protocol_valid": completion_protocol_valid,
            },
            "metrics": {
                "tracking": {
                    name: distribution_stats(episode_arrays[name])
                    for name in TRACKING_ERROR_NAMES
                },
                "behavior": {
                    name: distribution_stats(episode_arrays[name])
                    for name in behavior
                },
                "reward": {
                    "environment": distribution_stats(
                        episode_arrays["env_reward_mean"]
                    ),
                    "optimized_policy": distribution_stats(
                        episode_arrays["policy_reward_mean"]
                    ),
                },
                "intervention": {
                    "action_l2": distribution_stats(
                        episode_arrays["action_intervention_l2_mean"]
                    )
                },
            },
            "completion": {
                "available": completion_protocol_valid,
                "rate": float(np.mean(episode_arrays["completion"])),
                "components": {
                    name: float(np.mean(episode_arrays[f"completion_{name}"]))
                    for name in completion_components
                },
                "thresholds": asdict(CompletionThresholds()),
            },
            "termination": {
                "counts": done_counts,
                "first_done_step": distribution_stats(
                    episode_arrays["first_done_step"]
                ),
            },
            "efficiency": {
                "evaluation_wall_seconds": eval_seconds,
                "evaluated_env_steps": evaluated_samples,
                "simulator_env_steps_per_second": evaluated_samples
                / max(eval_seconds, 1e-12),
                "policy_decision_seconds": decision_seconds,
                "policy_latency_us_per_env_step": 1e6
                * decision_seconds
                / max(evaluated_samples, 1),
                "batch1_policy_latency_us": batch1_policy_latency_us,
                "peak_gpu_memory_mb": peak_gpu_memory_mb,
            },
            "input_geometry": input_geometry.finalize(),
            "artifacts": {
                "episodes": "episodes.npz",
                "timeseries": "timeseries.npz",
            },
        }

        out_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_dir / "episodes.npz", **episode_arrays)
        np.savez_compressed(
            out_dir / "timeseries.npz",
            time_seconds=np.arange(1, executed_steps + 1, dtype=np.float64) * dt,
            alive=np.stack(ts_alive, axis=0),
            tracking_error=np.stack(ts_errors, axis=0),
            tracking_error_names=np.asarray(TRACKING_ERROR_NAMES),
            sim_root_pos=np.stack(ts_sim_root, axis=0),
            ref_root_pos=np.stack(ts_ref_root, axis=0),
            winding_ratio=np.stack(ts_winding, axis=0),
            displacement_ratio=np.stack(ts_displacement, axis=0),
        )
        _write_json(out_dir / "summary.json", summary)
        return summary
    finally:
        os.chdir(original_cwd)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = evaluate(args)
    print(
        json.dumps(
            {
                "summary": str(resolve_repo_path(args.out_dir) / "summary.json"),
                "method": summary["metadata"]["method"],
                "motion": summary["metadata"]["motion"],
                "completion_rate": summary["completion"]["rate"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
