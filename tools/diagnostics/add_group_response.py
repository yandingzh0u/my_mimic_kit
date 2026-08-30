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
        "--start-mode", choices=("phase0", "random", "grid"),
        default="random")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--master-port", type=int, default=6382)
    parser.add_argument(
        "--disc-input-geometry", choices=("a30", "group_rms"),
        default="a30",
        help="Critic input geometry used when the checkpoint was trained.")
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


def _eval_disc(agent, diff, input_scale):
    return agent._model.eval_disc(diff * input_scale).squeeze(-1)


def measure_batch(agent, groups, info, input_scale):
    raw_diff = info["disc_obs_demo"] - info["disc_obs"]
    norm_diff = agent._disc_obs_norm.normalize(raw_diff).detach()
    norm_diff.requires_grad_(True)

    logits = _eval_disc(agent, norm_diff, input_scale)
    gradient = torch.autograd.grad(logits.sum(), norm_diff)[0]

    result = {}
    half_logits = {}
    with torch.no_grad():
        base_logits = logits.detach()
        zero_logits = _eval_disc(
            agent, torch.zeros_like(norm_diff), input_scale)
        for name, indices in groups:
            index = torch.as_tensor(indices, device=norm_diff.device)
            group_gradient = torch.index_select(gradient, -1, index)
            group_diff = torch.index_select(norm_diff.detach(), -1, index)
            half_diff = norm_diff.detach().clone()
            half_diff[:, index] *= 0.5
            half_logits[name] = _eval_disc(agent, half_diff, input_scale)
            response = half_logits[name] - base_logits
            group_only_diff = torch.zeros_like(norm_diff)
            group_only_diff[:, index] = group_diff
            group_only_logits = _eval_disc(
                agent, group_only_diff, input_scale)
            result[name] = {
                "sensitivity": torch.linalg.vector_norm(
                    group_gradient, dim=-1),
                "half_error_logit_change": response,
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


def main() -> None:
    args = parse_args()
    if args.num_envs <= 0 or args.steps <= 0 or args.sample_stride <= 0:
        raise ValueError("num-envs, steps, and sample-stride must be positive")

    model_file = resolve(args.model_file)
    env_file = resolve(args.env_config)
    agent_file = resolve(args.agent_config)
    engine_file = resolve(args.engine_config)
    out_file = resolve(args.out_file)
    for path in (model_file, env_file, agent_file, engine_file):
        if not path.is_file():
            raise FileNotFoundError(path)

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
        agent.set_mode(base_agent.AgentMode.TEST)
        obs, info = _reset_with_protocol(env, args.start_mode)
        groups = env.get_disc_error_groups()

        totals: dict[str, dict[str, float]] = {
            name: {
                "sensitivity_sum": 0.0,
                "half_error_logit_change_sum": 0.0,
                "half_error_logit_change_abs_sum": 0.0,
                "half_error_positive_count": 0.0,
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

        input_scale = torch.ones(
            env.get_disc_obs_space().shape[0], device=args.device)
        if args.disc_input_geometry == "group_rms":
            for _, indices in groups:
                input_scale[torch.as_tensor(
                    indices, device=args.device)] = 1.0 / math.sqrt(len(indices))

        for step in range(args.steps):
            if step % args.sample_stride == 0:
                batch, interactions = measure_batch(
                    agent, groups, info, input_scale)
                count = int(next(iter(batch.values()))["sensitivity"].numel())
                sample_count += count
                for name, values in batch.items():
                    response = values["half_error_logit_change"]
                    totals[name]["sensitivity_sum"] += float(
                        values["sensitivity"].sum().item())
                    totals[name]["half_error_logit_change_sum"] += float(
                        response.sum().item())
                    totals[name]["half_error_logit_change_abs_sum"] += float(
                        response.abs().sum().item())
                    totals[name]["half_error_positive_count"] += float(
                        (response > 0).sum().item())
                    totals[name]["normalized_error_rms_sum"] += float(
                        values["normalized_error_rms"].sum().item())
                    totals[name]["group_only_logit_sum"] += float(
                        values["group_only_logit"].sum().item())
                    totals[name]["zero_to_group_only_logit_drop_sum"] += float(
                        values["zero_to_group_only_logit_drop"].sum().item())
                    totals[name]["group_only_negative_count"] += float(
                        values["group_only_negative"].sum().item())
                for pair, interaction in interactions.items():
                    interaction_totals[pair]["sum"] += float(
                        interaction.sum().item())
                    interaction_totals[pair]["abs_sum"] += float(
                        interaction.abs().sum().item())

            with torch.no_grad():
                action, _ = agent._decide_action(obs, info)
                obs, _, _, info = env.step(action)

        group_results: dict[str, Any] = {}
        for name, indices in groups:
            values = totals[name]
            group_results[name] = {
                "dimension": len(indices),
                "S_gradient_norm_mean": (
                    values["sensitivity_sum"] / sample_count),
                "M_half_error_logit_change_mean": (
                    values["half_error_logit_change_sum"] / sample_count),
                "M_half_error_logit_change_abs_mean": (
                    values["half_error_logit_change_abs_sum"] / sample_count),
                "M_positive_fraction": (
                    values["half_error_positive_count"] / sample_count),
                "normalized_error_rms_mean": (
                    values["normalized_error_rms_sum"] / sample_count),
                "group_only_logit_mean": (
                    values["group_only_logit_sum"] / sample_count),
                "zero_to_group_only_logit_drop_mean": (
                    values["zero_to_group_only_logit_drop_sum"] / sample_count),
                "group_only_negative_fraction": (
                    values["group_only_negative_count"] / sample_count),
            }

        interaction_results = {
            "{}__{}".format(*pair): {
                "I_mean": values["sum"] / sample_count,
                "I_abs_mean": values["abs_sum"] / sample_count,
            }
            for pair, values in interaction_totals.items()
        }

        report = {
            "schema": "add_group_response_v1",
            "diagnostic_only": True,
            "differential_space": "saved DiffNormalizer output",
            "disc_input_geometry": args.disc_input_geometry,
            "checkpoint": str(model_file),
            "checkpoint_sha256": sha256(model_file),
            "start_mode": args.start_mode,
            "num_envs": args.num_envs,
            "steps": args.steps,
            "sample_stride": args.sample_stride,
            "sample_count": sample_count,
            "groups": group_results,
            "pairwise_half_error_interactions": interaction_results,
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
