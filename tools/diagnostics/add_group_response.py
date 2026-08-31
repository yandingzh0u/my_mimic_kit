#!/usr/bin/env python3
"""Measure an ADD critic's per-group local and finite-difference response.

This is an inference-only diagnostic.  It never updates the policy,
discriminator, normalizers, replay buffer, or optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICKIT_ROOT = REPO_ROOT / "mimickit"
for path in (REPO_ROOT, MIMICKIT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from tools.paper_eval.evaluate_checkpoint import _reset_with_protocol  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-file", required=True)
    parser.add_argument("--env-config", required=True)
    parser.add_argument("--agent-config", required=True)
    parser.add_argument(
        "--engine-config", default="data/engines/isaac_lab_engine.yaml")
    parser.add_argument("--out-file", required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--sample-stride", type=int, default=3)
    parser.add_argument(
        "--ig-steps", type=int, default=8,
        help="Midpoint integration steps for real-transition attribution.")
    parser.add_argument(
        "--start-mode", choices=("phase0", "random", "grid"),
        default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--master-port", type=int, default=6382)
    parser.add_argument(
        "--rollout-mode", choices=("test", "train"), default="test",
        help="Use deterministic test actions or stochastic training actions.")
    parser.add_argument(
        "--disc-input-geometry", choices=("a30", "group_rms"),
        default="a30",
        help="Critic input geometry used when the checkpoint was trained.")
    parser.add_argument(
        "--compare-model-file",
        help=("Optional second checkpoint. Its discriminator and saved "
              "normalizer are evaluated on exactly the same raw rollout "
              "residuals as --model-file."))
    parser.add_argument(
        "--compare-agent-config",
        help=("Agent config for --compare-model-file. Defaults to "
              "--agent-config."))
    parser.add_argument(
        "--compare-disc-input-geometry", choices=("a30", "group_rms"),
        default="a30",
        help="Critic input geometry used by --compare-model-file.")
    parser.add_argument("--primary-label", default="primary")
    parser.add_argument("--compare-label", default="compare")
    parser.add_argument(
        "--rollout-source", choices=("primary", "compare"),
        default="primary",
        help=("Actor that generates the one shared on-policy residual "
              "stream. 'compare' requires --compare-model-file."))
    return parser.parse_args()


def resolve(path: str) -> Path:
    value = Path(path).expanduser()
    return (value if value.is_absolute() else REPO_ROOT / value).resolve()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(module) -> str:
    """Hash tensor values so reports identify the saved normalizer used."""
    digest = hashlib.sha256()
    for name, value in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        if torch.is_tensor(value):
            tensor = value.detach().cpu().contiguous()
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def _eval_disc(agent, diff, input_scale):
    return agent._model.eval_disc(diff * input_scale).squeeze(-1)


def _disc_state(agent, info, input_scale, requires_grad=False):
    raw_diff = info["disc_obs_demo"] - info["disc_obs"]
    norm_diff = agent._disc_obs_norm.normalize(raw_diff).detach()
    norm_diff.requires_grad_(requires_grad)

    logits = _eval_disc(agent, norm_diff, input_scale)
    reward_scale = float(agent._disc_reward_scale)
    rewards = reward_scale * torch.nn.functional.softplus(logits)
    if requires_grad:
        gradient = torch.autograd.grad(logits.sum(), norm_diff)[0]
        reward_gradient = (
            reward_scale * torch.sigmoid(logits).unsqueeze(-1) * gradient)
    else:
        gradient = None
        reward_gradient = None
    return norm_diff, logits, rewards, gradient, reward_gradient


def measure_batch(agent, groups, info, input_scale):
    norm_diff, logits, base_rewards, gradient, reward_gradient = _disc_state(
        agent, info, input_scale, requires_grad=True)

    result = {}
    half_logits = {}
    with torch.no_grad():
        base_logits = logits.detach()
        reward_scale = float(agent._disc_reward_scale)
        zero_logits = _eval_disc(
            agent, torch.zeros_like(norm_diff), input_scale)
        zero_rewards = reward_scale * torch.nn.functional.softplus(zero_logits)
        reward_gap = zero_rewards - base_rewards
        for name, indices in groups:
            index = torch.as_tensor(indices, device=norm_diff.device)
            group_gradient = torch.index_select(gradient, -1, index)
            group_reward_gradient = torch.index_select(
                reward_gradient, -1, index)
            group_diff = torch.index_select(norm_diff.detach(), -1, index)
            half_diff = norm_diff.detach().clone()
            half_diff[:, index] *= 0.5
            half_logits[name] = _eval_disc(agent, half_diff, input_scale)
            response = half_logits[name] - base_logits
            half_rewards = reward_scale * torch.nn.functional.softplus(
                half_logits[name])
            reward_response = half_rewards - base_rewards
            group_only_diff = torch.zeros_like(norm_diff)
            group_only_diff[:, index] = group_diff
            group_only_logits = _eval_disc(
                agent, group_only_diff, input_scale)
            result[name] = {
                "sensitivity": torch.linalg.vector_norm(
                    group_gradient, dim=-1),
                "reward_sensitivity": torch.linalg.vector_norm(
                    group_reward_gradient, dim=-1),
                "local_zero_direction_logit_slope": -torch.sum(
                    group_gradient * group_diff, dim=-1),
                "local_zero_direction_reward_slope": -torch.sum(
                    group_reward_gradient * group_diff, dim=-1),
                "half_error_logit_change": response,
                "half_error_reward_change": reward_response,
                "zero_to_full_reward_gap": reward_gap,
                "normalized_error_rms": torch.sqrt(
                    torch.mean(torch.square(group_diff), dim=-1)),
                "group_only_logit": group_only_logits,
                "zero_to_group_only_logit_drop": (
                    zero_logits - group_only_logits),
                "group_only_negative": group_only_logits < 0,
            }
        interactions = {}
        for group_id, (name_g, indices_g) in enumerate(groups):
            for name_h, indices_h in groups[group_id + 1:]:
                both_diff = norm_diff.detach().clone()
                both_diff[:, torch.as_tensor(
                    indices_g, device=norm_diff.device)] *= 0.5
                both_diff[:, torch.as_tensor(
                    indices_h, device=norm_diff.device)] *= 0.5
                both_logits = _eval_disc(agent, both_diff, input_scale)
                interactions[(name_g, name_h)] = (
                    both_logits - half_logits[name_g]
                    - half_logits[name_h] + base_logits)
    return result, interactions


def measure_transition(agent, groups, info, next_info, input_scale, ig_steps):
    """Attribute a real one-step reward change with integrated gradients."""
    current_diff, _, current_reward, _, _ = _disc_state(
        agent, info, input_scale, requires_grad=False)
    next_diff, _, next_reward, _, _ = _disc_state(
        agent, next_info, input_scale, requires_grad=False)
    delta = next_diff.detach() - current_diff.detach()
    integrated_gradient = torch.zeros_like(delta)
    reward_scale = float(agent._disc_reward_scale)
    for integration_step in range(ig_steps):
        alpha = (integration_step + 0.5) / ig_steps
        point = (current_diff.detach() + alpha * delta).requires_grad_(True)
        point_logits = _eval_disc(agent, point, input_scale)
        point_rewards = reward_scale * torch.nn.functional.softplus(
            point_logits)
        integrated_gradient += torch.autograd.grad(
            point_rewards.sum(), point)[0] / ig_steps

    group_values = {}
    for name, indices in groups:
        index = torch.as_tensor(indices, device=next_diff.device)
        group_delta = torch.index_select(delta, -1, index)
        group_gradient = torch.index_select(
            integrated_gradient, -1, index)
        group_values[name] = torch.sum(group_gradient * group_delta, dim=-1)

    return {
        "groups": group_values,
        "integrated_reward_change": torch.sum(
            integrated_gradient * delta, dim=-1),
        "actual_reward_change": next_reward.detach() - current_reward.detach(),
    }


def _build_input_scale(env, groups, geometry, device):
    scale = torch.ones(env.get_disc_obs_space().shape[0], device=device)
    if geometry == "group_rms":
        for _, indices in groups:
            scale[torch.as_tensor(indices, device=device)] = (
                1.0 / math.sqrt(len(indices)))
    return scale


def _new_static_totals(groups):
    return {
        name: {
            "sensitivity_sum": 0.0,
            "reward_sensitivity_sum": 0.0,
            "local_zero_direction_logit_slope_sum": 0.0,
            "local_zero_direction_reward_slope_sum": 0.0,
            "half_error_logit_change_sum": 0.0,
            "half_error_logit_change_abs_sum": 0.0,
            "half_error_positive_count": 0.0,
            "half_error_reward_change_sum": 0.0,
            "half_error_reward_change_abs_sum": 0.0,
            "zero_to_full_reward_gap_sum": 0.0,
            "normalized_error_rms_sum": 0.0,
            "group_only_logit_sum": 0.0,
            "zero_to_group_only_logit_drop_sum": 0.0,
            "group_only_negative_count": 0.0,
        }
        for name, _ in groups
    }


def _accumulate_static(totals, batch):
    for name, values in batch.items():
        response = values["half_error_logit_change"]
        reward_response = values["half_error_reward_change"]
        totals[name]["sensitivity_sum"] += float(
            values["sensitivity"].sum().item())
        totals[name]["reward_sensitivity_sum"] += float(
            values["reward_sensitivity"].sum().item())
        totals[name]["local_zero_direction_logit_slope_sum"] += float(
            values["local_zero_direction_logit_slope"].sum().item())
        totals[name]["local_zero_direction_reward_slope_sum"] += float(
            values["local_zero_direction_reward_slope"].sum().item())
        totals[name]["half_error_logit_change_sum"] += float(
            response.sum().item())
        totals[name]["half_error_logit_change_abs_sum"] += float(
            response.abs().sum().item())
        totals[name]["half_error_positive_count"] += float(
            (response > 0).sum().item())
        totals[name]["half_error_reward_change_sum"] += float(
            reward_response.sum().item())
        totals[name]["half_error_reward_change_abs_sum"] += float(
            reward_response.abs().sum().item())
        totals[name]["zero_to_full_reward_gap_sum"] += float(
            values["zero_to_full_reward_gap"].sum().item())
        totals[name]["normalized_error_rms_sum"] += float(
            values["normalized_error_rms"].sum().item())
        totals[name]["group_only_logit_sum"] += float(
            values["group_only_logit"].sum().item())
        totals[name]["zero_to_group_only_logit_drop_sum"] += float(
            values["zero_to_group_only_logit_drop"].sum().item())
        totals[name]["group_only_negative_count"] += float(
            values["group_only_negative"].sum().item())


def _summarize_static(groups, totals, sample_count):
    results = {}
    for name, indices in groups:
        values = totals[name]
        gap_sum = values["zero_to_full_reward_gap_sum"]
        results[name] = {
            "dimension": len(indices),
            "S_gradient_norm_mean": (
                values["sensitivity_sum"] / sample_count),
            "S_reward_gradient_norm_mean": (
                values["reward_sensitivity_sum"] / sample_count),
            "local_zero_direction_logit_slope_mean": (
                values["local_zero_direction_logit_slope_sum"]
                / sample_count),
            "local_zero_direction_reward_slope_mean": (
                values["local_zero_direction_reward_slope_sum"]
                / sample_count),
            "M_half_error_logit_change_mean": (
                values["half_error_logit_change_sum"] / sample_count),
            "M_half_error_logit_change_abs_mean": (
                values["half_error_logit_change_abs_sum"] / sample_count),
            "M_positive_fraction": (
                values["half_error_positive_count"] / sample_count),
            "M_half_error_reward_change_mean": (
                values["half_error_reward_change_sum"] / sample_count),
            "M_half_error_reward_change_abs_mean": (
                values["half_error_reward_change_abs_sum"] / sample_count),
            "M_half_error_reward_gap_coverage": (
                values["half_error_reward_change_sum"]
                / max(gap_sum, 1e-8)),
            "zero_to_full_reward_gap_mean": gap_sum / sample_count,
            "normalized_error_rms_mean": (
                values["normalized_error_rms_sum"] / sample_count),
            "group_only_logit_mean": (
                values["group_only_logit_sum"] / sample_count),
            "zero_to_group_only_logit_drop_mean": (
                values["zero_to_group_only_logit_drop_sum"] / sample_count),
            "group_only_negative_fraction": (
                values["group_only_negative_count"] / sample_count),
        }
    return results


def _root_cross_summary(evaluators):
    metrics = (
        "M_half_error_reward_change_mean",
        "S_reward_gradient_norm_mean",
        "M_half_error_reward_gap_coverage",
        "zero_to_full_reward_gap_mean",
    )
    return {
        label: {metric: data["groups"]["root_pos"][metric]
                for metric in metrics}
        for label, data in evaluators.items()
    }


def main() -> None:
    args = parse_args()
    if (args.num_envs <= 0 or args.steps <= 0 or args.sample_stride <= 0
            or args.ig_steps <= 0):
        raise ValueError(
            "num-envs, steps, sample-stride, and ig-steps must be positive")

    model_file = resolve(args.model_file)
    env_file = resolve(args.env_config)
    agent_file = resolve(args.agent_config)
    engine_file = resolve(args.engine_config)
    out_file = resolve(args.out_file)
    for path in (model_file, env_file, agent_file, engine_file):
        if not path.is_file():
            raise FileNotFoundError(path)
    compare_model_file = None
    compare_agent_file = None
    if args.compare_model_file is not None:
        compare_model_file = resolve(args.compare_model_file)
        compare_agent_file = resolve(
            args.compare_agent_config or args.agent_config)
        for path in (compare_model_file, compare_agent_file):
            if not path.is_file():
                raise FileNotFoundError(path)
        if args.primary_label == args.compare_label:
            raise ValueError("primary-label and compare-label must differ")
    elif args.rollout_source == "compare":
        raise ValueError(
            "--rollout-source compare requires --compare-model-file")

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
            rank=0, num_procs=1, device=args.device,
            master_port=args.master_port)
        env = env_builder.build_env(
            str(env_file), str(engine_file), args.num_envs, args.device,
            visualize=False, record_video=False)
        agent = agent_builder.build_agent(str(agent_file), env, args.device)
        agent.load(str(model_file))
        agent.eval()
        compare_agent = None
        if compare_model_file is not None:
            compare_agent = agent_builder.build_agent(
                str(compare_agent_file), env, args.device)
            compare_agent.load(str(compare_model_file))
            compare_agent.eval()
        rollout_agent_mode = (
            base_agent.AgentMode.TRAIN
            if args.rollout_mode == "train"
            else base_agent.AgentMode.TEST)
        rollout_agent = (
            compare_agent if args.rollout_source == "compare" else agent)
        rollout_agent.set_mode(rollout_agent_mode)
        obs, info = _reset_with_protocol(env, args.start_mode)
        groups = env.get_disc_error_groups()

        totals: dict[str, dict[str, float]] = {
            name: {
                "sensitivity_sum": 0.0,
                "reward_sensitivity_sum": 0.0,
                "local_zero_direction_logit_slope_sum": 0.0,
                "local_zero_direction_reward_slope_sum": 0.0,
                "half_error_logit_change_sum": 0.0,
                "half_error_logit_change_abs_sum": 0.0,
                "half_error_positive_count": 0.0,
                "half_error_reward_change_sum": 0.0,
                "half_error_reward_change_abs_sum": 0.0,
                "zero_to_full_reward_gap_sum": 0.0,
                "trajectory_projection_sum": 0.0,
                "trajectory_projection_abs_sum": 0.0,
                "normalized_error_rms_sum": 0.0,
                "group_only_logit_sum": 0.0,
                "zero_to_group_only_logit_drop_sum": 0.0,
                "group_only_negative_count": 0.0,
            }
            for name, _ in groups
        }
        interaction_totals = {
            (name_g, name_h): {"sum": 0.0, "abs_sum": 0.0}
            for group_id, (name_g, _) in enumerate(groups)
            for name_h, _ in groups[group_id + 1:]
        }
        sample_count = 0
        transition_count = 0
        transition_integrated_sum = 0.0
        transition_actual_sum = 0.0
        transition_residual_abs_sum = 0.0

        input_scale = _build_input_scale(
            env, groups, args.disc_input_geometry, args.device)
        compare_input_scale = None
        compare_totals = None
        raw_stream_digest = None
        if compare_agent is not None:
            compare_input_scale = _build_input_scale(
                env, groups, args.compare_disc_input_geometry, args.device)
            compare_totals = _new_static_totals(groups)
            raw_stream_digest = hashlib.sha256()

        for step in range(args.steps):
            current_info = {
                "disc_obs": info["disc_obs"].clone(),
                "disc_obs_demo": info["disc_obs_demo"].clone(),
            }
            with torch.no_grad():
                action, _ = rollout_agent._decide_action(obs, info)
                next_obs, _, done, next_info = env.step(action)

            if step % args.sample_stride == 0:
                if raw_stream_digest is not None:
                    raw_diff = (
                        next_info["disc_obs_demo"] - next_info["disc_obs"])
                    raw_stream_digest.update(
                        raw_diff.detach().to(
                            device="cpu", dtype=torch.float32
                        ).contiguous().numpy().tobytes())
                batch, interactions = measure_batch(
                    agent, groups, next_info, input_scale)
                if compare_agent is not None:
                    compare_batch, _ = measure_batch(
                        compare_agent, groups, next_info,
                        compare_input_scale)
                    _accumulate_static(compare_totals, compare_batch)
                transition = measure_transition(
                    agent, groups, current_info, next_info, input_scale,
                    args.ig_steps)
                count = int(next(iter(batch.values()))["sensitivity"].numel())
                sample_count += count
                transition_count += count
                integrated = transition["integrated_reward_change"]
                actual = transition["actual_reward_change"]
                transition_integrated_sum += float(integrated.sum().item())
                transition_actual_sum += float(actual.sum().item())
                transition_residual_abs_sum += float(
                    (actual - integrated).abs().sum().item())
                _accumulate_static(totals, batch)
                for name, values in batch.items():
                    projection = transition["groups"][name]
                    totals[name]["trajectory_projection_sum"] += float(
                        projection.sum().item())
                    totals[name]["trajectory_projection_abs_sum"] += float(
                        projection.abs().sum().item())
                for pair, interaction in interactions.items():
                    interaction_totals[pair]["sum"] += float(
                        interaction.sum().item())
                    interaction_totals[pair]["abs_sum"] += float(
                        interaction.abs().sum().item())

            done_ids = torch.flatten(
                (done != 0).nonzero(as_tuple=False))
            obs, info = env.reset(done_ids)

        group_results: dict[str, Any] = _summarize_static(
            groups, totals, sample_count)
        for name, _ in groups:
            values = totals[name]
            group_results[name].update({
                "trajectory_reward_projection_mean": (
                    values["trajectory_projection_sum"] / transition_count),
                "trajectory_reward_projection_abs_mean": (
                    values["trajectory_projection_abs_sum"]
                    / transition_count),
            })

        interaction_results = {
            "{}__{}".format(*pair): {
                "I_mean": values["sum"] / sample_count,
                "I_abs_mean": values["abs_sum"] / sample_count,
            }
            for pair, values in interaction_totals.items()
        }

        report = {
            "schema": "add_group_response_v2",
            "diagnostic_only": True,
            "differential_space": "saved DiffNormalizer output",
            "disc_input_geometry": args.disc_input_geometry,
            "checkpoint": str(model_file),
            "checkpoint_sha256": sha256(model_file),
            "start_mode": args.start_mode,
            "rollout_mode": args.rollout_mode,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "sample_stride": args.sample_stride,
            "ig_steps": args.ig_steps,
            "sample_count": sample_count,
            "groups": group_results,
            "trajectory": {
                "transition_count": transition_count,
                "actual_reward_change_mean": (
                    transition_actual_sum / transition_count),
                "integrated_reward_change_mean": (
                    transition_integrated_sum / transition_count),
                "integration_abs_residual_mean": (
                    transition_residual_abs_sum / transition_count),
            },
            "pairwise_half_error_interactions": interaction_results,
        }
        if compare_agent is not None:
            compare_group_results = _summarize_static(
                groups, compare_totals, sample_count)
            evaluator_results = {
                args.primary_label: {
                    "checkpoint": str(model_file),
                    "checkpoint_sha256": sha256(model_file),
                    "normalizer_state_sha256": state_dict_sha256(
                        agent._disc_obs_norm),
                    "disc_input_geometry": args.disc_input_geometry,
                    "groups": group_results,
                },
                args.compare_label: {
                    "checkpoint": str(compare_model_file),
                    "checkpoint_sha256": sha256(compare_model_file),
                    "normalizer_state_sha256": state_dict_sha256(
                        compare_agent._disc_obs_norm),
                    "disc_input_geometry": args.compare_disc_input_geometry,
                    "groups": compare_group_results,
                },
            }
            report["cross_evaluation"] = {
                "strict_shared_raw_stream": True,
                "rollout_source": args.rollout_source,
                "rollout_actor_label": (
                    args.compare_label
                    if args.rollout_source == "compare"
                    else args.primary_label),
                "raw_stream_sha256": raw_stream_digest.hexdigest(),
                "raw_stream_sample_count": sample_count,
                "evaluators": evaluator_results,
                "root_pos": _root_cross_summary(evaluator_results),
            }
        out_file.parent.mkdir(parents=True, exist_ok=True)
        temp = out_file.with_suffix(out_file.suffix + ".tmp")
        temp.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8")
        os.replace(temp, out_file)
        print(json.dumps(report, indent=2, sort_keys=True))
    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
