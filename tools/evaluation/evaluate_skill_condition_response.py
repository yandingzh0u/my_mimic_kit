from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import util.mp_util as mp_util
import util.util as util
from learning.base_agent import AgentMode
from util.arg_parser import ArgParser


FLOW_WINDOW_STEPS = 10
MISMATCH_WARMUP_STEPS = FLOW_WINDOW_STEPS - 1


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fix_seed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    util.set_rand_seed(int(seed))


def _context_kwargs(context: Mapping[str, Any]) -> dict[str, Any]:
    motion_path = context.get("motion_path")
    clip_sha256 = context.get("clip_sha256")
    if (motion_path is None) == (clip_sha256 is None):
        raise ValueError("each context requires exactly one motion_path or clip_sha256")
    if "context_start_sec" not in context:
        raise ValueError("each context requires context_start_sec")
    start = float(context["context_start_sec"])
    if not math.isfinite(start):
        raise ValueError("context_start_sec must be finite")
    return {
        "motion_path": str(motion_path) if motion_path is not None else None,
        "clip_sha256": str(clip_sha256) if clip_sha256 is not None else None,
        "context_start_sec": start,
    }


def _cpu_row(value: torch.Tensor) -> torch.Tensor:
    tensor = torch.as_tensor(value).detach().cpu()
    if tensor.shape[0] != 1:
        raise ValueError("paired condition-response evaluation requires exactly one environment")
    return tensor[0].clone()


@torch.no_grad()
def rollout_branch(
    agent,
    *,
    latent_context: Mapping[str, Any],
    rollout_steps: int,
    seed: int,
) -> dict[str, Any]:
    """Deterministically roll out one legal expert-derived condition."""
    if agent.get_num_envs() != 1:
        raise ValueError("paired condition-response evaluation requires num_envs=1")
    if rollout_steps <= 0:
        raise ValueError("rollout_steps must be positive")
    fix_seed(seed)
    obs, info = agent._reset_envs()
    agent._curr_obs, agent._curr_info = obs, info
    applied = agent.set_evaluation_skill_context(**_context_kwargs(latent_context))
    env = agent._env
    state_getter = getattr(env, "get_skill_evaluation_state", None)
    if not callable(state_getter):
        raise TypeError("R2 environment lacks get_skill_evaluation_state()")

    initial_state = {
        key: _cpu_row(value) for key, value in state_getter(env_ids=[0]).items()
    }
    states = [initial_state]
    actions = []
    disc_windows = []
    done_flag = 0
    for _ in range(int(rollout_steps)):
        action, _ = agent._decide_action(obs, info)
        next_obs, _, done, next_info = env.step(action)
        actions.append(_cpu_row(action))
        disc_windows.append(_cpu_row(next_info["disc_obs"]))
        states.append(
            {
                key: _cpu_row(value)
                for key, value in state_getter(env_ids=[0]).items()
            }
        )
        obs, info = next_obs, next_info
        done_flag = int(done[0].item())
        if done_flag != 0:
            break
    agent._curr_obs, agent._curr_info = obs, info
    return {
        "applied_context": applied,
        "initial_state": initial_state,
        "states": states,
        "actions": torch.stack(actions),
        "disc_windows": torch.stack(disc_windows),
        "done_flag": done_flag,
        "steps": len(actions),
    }


def compare_initial_states(
    state_a: Mapping[str, torch.Tensor],
    state_b: Mapping[str, torch.Tensor],
    *,
    tolerance: float,
) -> dict[str, Any]:
    if tolerance < 0.0 or not math.isfinite(tolerance):
        raise ValueError("initial-state tolerance must be finite and non-negative")
    if set(state_a) != set(state_b):
        raise ValueError("initial-state snapshots have different fields")
    per_field = {}
    for key in sorted(state_a):
        left, right = state_a[key].float(), state_b[key].float()
        if left.shape != right.shape:
            raise ValueError("initial-state field {} has different shapes".format(key))
        per_field[key] = float(torch.max(torch.abs(left - right))) if left.numel() else 0.0
    maximum = max(per_field.values(), default=0.0)
    return {
        "equal_within_tolerance": maximum <= tolerance,
        "tolerance": float(tolerance),
        "max_abs_difference": maximum,
        "per_field_max_abs_difference": per_field,
    }


def _stack_state(branch: Mapping[str, Any], key: str, count: int) -> torch.Tensor:
    return torch.stack([state[key].float() for state in branch["states"][:count]])


def compare_behavior(branch_a: Mapping[str, Any], branch_b: Mapping[str, Any]) -> dict[str, Any]:
    common_steps = min(int(branch_a["steps"]), int(branch_b["steps"]))
    common_states = common_steps + 1
    root_a = _stack_state(branch_a, "root_pos", common_states)
    root_b = _stack_state(branch_b, "root_pos", common_states)
    root_distance = torch.linalg.vector_norm(root_a - root_b, dim=-1)
    dof_delta = _stack_state(branch_a, "dof_pos", common_states) - _stack_state(
        branch_b, "dof_pos", common_states
    )
    body_delta = _stack_state(branch_a, "body_pos", common_states) - _stack_state(
        branch_b, "body_pos", common_states
    )
    action_delta = branch_a["actions"][:common_steps].float() - branch_b["actions"][
        :common_steps
    ].float()
    action_l2 = torch.linalg.vector_norm(action_delta, dim=-1)

    disp_a = root_a[-1] - root_a[0]
    disp_b = root_b[-1] - root_b[0]
    return {
        "steps_a": int(branch_a["steps"]),
        "steps_b": int(branch_b["steps"]),
        "common_steps": common_steps,
        "done_flag_a": int(branch_a["done_flag"]),
        "done_flag_b": int(branch_b["done_flag"]),
        "first_action_l2_difference": float(action_l2[0]),
        "action_l2_difference_mean": float(action_l2.mean()),
        "action_l2_difference_max": float(action_l2.max()),
        "root_trajectory_l2_mean": float(root_distance.mean()),
        "root_trajectory_l2_max": float(root_distance.max()),
        "root_endpoint_separation": float(root_distance[-1]),
        "root_xy_displacement_a": [float(value) for value in disp_a[:2]],
        "root_xy_displacement_b": [float(value) for value in disp_b[:2]],
        "dof_position_rmse": float(torch.sqrt(torch.mean(torch.square(dof_delta)))),
        "body_position_rmse": float(torch.sqrt(torch.mean(torch.square(body_delta)))),
        "behavior_changed": bool(
            float(action_l2.max()) > 1e-6 or float(root_distance.max()) > 1e-5
        ),
    }


def summarize(values: torch.Tensor) -> dict[str, float]:
    values = torch.as_tensor(values).detach().float().cpu().flatten()
    if values.numel() == 0 or not torch.isfinite(values).all():
        raise ValueError("summary requires a non-empty finite tensor")
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "median": float(torch.quantile(values, 0.5)),
        "p05": float(torch.quantile(values, 0.05)),
        "p95": float(torch.quantile(values, 0.95)),
    }


@torch.no_grad()
def compare_window_conditions(
    agent,
    windows: torch.Tensor,
    *,
    matched_context: Mapping[str, Any],
    counterfactual_context: Mapping[str, Any],
) -> dict[str, Any]:
    matched = agent.score_evaluation_windows(
        windows, **_context_kwargs(matched_context)
    )
    counterfactual = agent.score_evaluation_windows(
        windows, **_context_kwargs(counterfactual_context)
    )
    matched_scaled = matched["scaled_mismatch"].float()
    counter_scaled = counterfactual["scaled_mismatch"].float()
    eps = torch.finfo(torch.float32).eps
    return {
        "matched_context": {
            key: matched[key]
            for key in ("motion_id", "motion_path", "clip_sha256", "context_start_sec")
        },
        "counterfactual_context": {
            key: counterfactual[key]
            for key in ("motion_id", "motion_path", "clip_sha256", "context_start_sec")
        },
        "matched_raw": summarize(matched["raw_mismatch"]),
        "counterfactual_raw": summarize(counterfactual["raw_mismatch"]),
        "matched_scaled": summarize(matched_scaled),
        "counterfactual_scaled": summarize(counter_scaled),
        "paired_matched_lower_rate": float((matched_scaled < counter_scaled).float().mean()),
        "counterfactual_over_matched_mean_ratio": float(
            counter_scaled.mean() / torch.clamp_min(matched_scaled.mean(), eps)
        ),
        "matched_minus_counterfactual_mean": float(
            matched_scaled.mean() - counter_scaled.mean()
        ),
    }


@torch.no_grad()
def evaluate_condition_response(
    agent,
    *,
    reset_context: Mapping[str, Any],
    context_a: Mapping[str, Any],
    context_b: Mapping[str, Any],
    rollout_steps: int,
    seed: int,
    initial_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Run the same reset twice, changing only the legal expert-derived z."""
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    reset_kwargs = _context_kwargs(reset_context)
    agent.set_skill_command(**reset_kwargs)

    branch_a = rollout_branch(
        agent,
        latent_context=context_a,
        rollout_steps=rollout_steps,
        seed=seed,
    )
    branch_b = rollout_branch(
        agent,
        latent_context=context_b,
        rollout_steps=rollout_steps,
        seed=seed,
    )
    identity_a = (
        branch_a["applied_context"]["clip_sha256"],
        float(branch_a["applied_context"]["context_start_sec"]),
    )
    identity_b = (
        branch_b["applied_context"]["clip_sha256"],
        float(branch_b["applied_context"]["context_start_sec"]),
    )
    if identity_a == identity_b:
        raise ValueError("context_a and context_b must identify different expert contexts")
    for name, branch in (("a", branch_a), ("b", branch_b)):
        if int(branch["steps"]) < FLOW_WINDOW_STEPS:
            raise RuntimeError(
                "branch {} ended after {} steps; at least {} are required before "
                "a fully-policy H={} window exists".format(
                    name, branch["steps"], FLOW_WINDOW_STEPS, FLOW_WINDOW_STEPS
                )
            )
    initial = compare_initial_states(
        branch_a["initial_state"], branch_b["initial_state"], tolerance=initial_tolerance
    )
    if not initial["equal_within_tolerance"]:
        raise RuntimeError(
            "paired rollouts did not start from the same state (max abs diff={})".format(
                initial["max_abs_difference"]
            )
        )

    latent_a = branch_a["applied_context"]["latent"].float()
    latent_b = branch_b["applied_context"]["latent"].float()
    latent_cosine = torch.nn.functional.cosine_similarity(
        latent_a.unsqueeze(0), latent_b.unsqueeze(0)
    )[0]
    w_a_test = compare_window_conditions(
        agent,
        branch_a["disc_windows"][MISMATCH_WARMUP_STEPS:],
        matched_context=context_a,
        counterfactual_context=context_b,
    )
    w_b_test = compare_window_conditions(
        agent,
        branch_b["disc_windows"][MISMATCH_WARMUP_STEPS:],
        matched_context=context_b,
        counterfactual_context=context_a,
    )
    for audit, branch in ((w_a_test, branch_a), (w_b_test, branch_b)):
        audit["flow_window_steps"] = FLOW_WINDOW_STEPS
        audit["warmup_steps"] = MISMATCH_WARMUP_STEPS
        audit["first_scored_rollout_step"] = FLOW_WINDOW_STEPS
        audit["rollout_window_count"] = int(branch["disc_windows"].shape[0])
        audit["scored_window_count"] = int(
            branch["disc_windows"].shape[0] - MISMATCH_WARMUP_STEPS
        )

    return {
        "protocol": {
            "name": "r2_paired_expert_context_condition_response",
            "version": 1,
            "same_reset_context": True,
            "deterministic_policy_mode": True,
            "arbitrary_latent_injection": False,
            "rollout_steps_requested": int(rollout_steps),
            "flow_window_steps": FLOW_WINDOW_STEPS,
            "mismatch_warmup_steps": MISMATCH_WARMUP_STEPS,
            "seed": int(seed),
        },
        "reset_context": reset_kwargs,
        "context_a": {
            key: branch_a["applied_context"][key]
            for key in ("motion_id", "motion_path", "clip_sha256", "context_start_sec")
        },
        "context_b": {
            key: branch_b["applied_context"][key]
            for key in ("motion_id", "motion_path", "clip_sha256", "context_start_sec")
        },
        "latent_comparison": {
            "l2_distance": float(torch.linalg.vector_norm(latent_a - latent_b)),
            "cosine_similarity": float(latent_cosine),
            "norm_a": float(torch.linalg.vector_norm(latent_a)),
            "norm_b": float(torch.linalg.vector_norm(latent_b)),
        },
        "initial_state_check": initial,
        "behavior_difference": compare_behavior(branch_a, branch_b),
        "w_a_context_test": w_a_test,
        "w_b_context_test": w_b_test,
    }


def _add_context_arguments(parser: argparse.ArgumentParser, prefix: str, required: bool) -> None:
    group = parser.add_mutually_exclusive_group(required=required)
    group.add_argument("--{}-motion-path".format(prefix))
    group.add_argument("--{}-clip-sha256".format(prefix))
    parser.add_argument("--{}-start-sec".format(prefix), type=float, required=required)


def _context_from_args(args, prefix: str, fallback=None) -> dict[str, Any]:
    motion_path = getattr(args, prefix.replace("-", "_") + "_motion_path")
    clip_sha256 = getattr(args, prefix.replace("-", "_") + "_clip_sha256")
    start = getattr(args, prefix.replace("-", "_") + "_start_sec")
    if motion_path is None and clip_sha256 is None and start is None:
        if fallback is None:
            raise ValueError("missing {} context".format(prefix))
        return dict(fallback)
    if (motion_path is None) == (clip_sha256 is None) or start is None:
        raise ValueError(
            "{} context needs one clip identity and --{}-start-sec".format(
                prefix, prefix
            )
        )
    return {
        "motion_path": motion_path,
        "clip_sha256": clip_sha256,
        "context_start_sec": start,
    }


def build_runtime(arg_file: str, model_file: str, device: str, seed: int):
    if mp_util.get_num_procs() == 0:
        mp_util.init(0, 1, device, 0)
    fix_seed(seed)
    runtime_args = ArgParser()
    if not runtime_args.load_file(arg_file):
        raise OSError("failed to load arg file {}".format(arg_file))
    env = env_builder.build_env(
        runtime_args.parse_string("env_config"),
        runtime_args.parse_string("engine_config"),
        num_envs=1,
        device=device,
        visualize=False,
        record_video=False,
    )
    agent = agent_builder.build_agent(
        runtime_args.parse_string("agent_config"), env, device
    )
    agent.load(model_file)
    return agent, runtime_args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate R2 response to two legal expert contexts from the exact same "
            "physical/task reset state."
        )
    )
    parser.add_argument(
        "--arg-file", default="args/skill_conditioned_flow_humanoid_args.txt"
    )
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--rollout-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--initial-tolerance", type=float, default=1e-6)
    parser.add_argument("--output")
    _add_context_arguments(parser, "context-a", required=True)
    _add_context_arguments(parser, "context-b", required=True)
    _add_context_arguments(parser, "reset", required=False)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    context_a = _context_from_args(args, "context-a")
    context_b = _context_from_args(args, "context-b")
    reset_context = _context_from_args(args, "reset", fallback=context_a)
    agent, runtime_args = build_runtime(
        args.arg_file, args.model_file, args.device, args.seed
    )
    report = evaluate_condition_response(
        agent,
        reset_context=reset_context,
        context_a=context_a,
        context_b=context_b,
        rollout_steps=args.rollout_steps,
        seed=args.seed,
        initial_tolerance=args.initial_tolerance,
    )
    prior_path = Path(agent._prior_path)
    arg_path = Path(args.arg_file)
    agent_config_path = Path(runtime_args.parse_string("agent_config"))
    env_config_path = Path(runtime_args.parse_string("env_config"))
    engine_config_path = Path(runtime_args.parse_string("engine_config"))
    manifest = agent._env.get_skill_dataset_manifest()
    report["artifacts"] = {
        "arg_file": str(arg_path),
        "arg_file_sha256": file_sha256(arg_path),
        "model_file": str(Path(args.model_file)),
        "model_sha256": file_sha256(args.model_file),
        "agent_config": str(agent_config_path),
        "agent_config_sha256": file_sha256(agent_config_path),
        "conditional_prior_model": str(prior_path),
        "conditional_prior_sha256": file_sha256(prior_path),
        "env_config": str(env_config_path),
        "env_config_sha256": file_sha256(env_config_path),
        "engine_config": str(engine_config_path),
        "engine_config_sha256": file_sha256(engine_config_path),
        "dataset_yaml_sha256": manifest["dataset_yaml_sha256"],
        "canonical_manifest_sha256": manifest["canonical_manifest_sha256"],
        "device": args.device,
    }
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return report


if __name__ == "__main__":
    main()
