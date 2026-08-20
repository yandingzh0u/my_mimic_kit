#!/usr/bin/env python3
"""Fit the transition critic on one unlabeled policy-negative rollout.

The CLI deliberately accepts no successful or shortcut evaluation bundle.
Model selection uses only an episode-disjoint validation split of the supplied
policy-negative rollout.  This makes behavior-label leakage structurally hard.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import sys
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.phase_transition_critic.rollout_contract import (  # noqa: E402
    load_transition_bundle,
    validate_transition_bundle,
)
from tools.paper_eval.evaluate_checkpoint import resolve_repo_path, sha256_file  # noqa: E402


OFFLINE_CHECKPOINT_VERSION = 1


@dataclass(frozen=True)
class FitConfig:
    epochs: int = 12
    batch_size: int = 512
    learning_rate: float = 1e-4
    gp_weight: float = 10.0
    gp_target: float = 1.0
    input_clip: float = 10.0
    min_phase_distance: float = 0.1
    min_shuffle_rms: float = 1e-3
    validation_fraction: float = 0.25
    grad_clip: float = 10.0
    seed: int = 918273


def build_transition_critic(phi_dim: int, init_output_scale: float = 0.01):
    import torch

    if phi_dim <= 0:
        raise ValueError("phi_dim must be positive")
    input_dim = 4 * phi_dim
    model = torch.nn.Sequential(
        torch.nn.Linear(input_dim, 1024),
        torch.nn.ReLU(),
        torch.nn.Linear(1024, 512),
        torch.nn.ReLU(),
        torch.nn.Linear(512, 1),
    )
    torch.nn.init.zeros_(model[0].bias)
    torch.nn.init.zeros_(model[2].bias)
    torch.nn.init.zeros_(model[4].bias)
    torch.nn.init.uniform_(
        model[4].weight, -float(init_output_scale), float(init_output_scale)
    )
    return model


def reference_statistics(
    ref_state: np.ndarray, ref_motion: np.ndarray, min_scale: float = 1e-4
) -> dict[str, np.ndarray]:
    state_mean = np.mean(ref_state, axis=0, dtype=np.float64).astype(np.float32)
    state_scale = np.std(ref_state, axis=0, dtype=np.float64).astype(np.float32)
    motion_mean = np.mean(ref_motion, axis=0, dtype=np.float64).astype(np.float32)
    motion_scale = np.std(ref_motion, axis=0, dtype=np.float64).astype(np.float32)
    state_scale = np.where(state_scale > min_scale, state_scale, 1.0).astype(np.float32)
    motion_scale = np.where(motion_scale > min_scale, motion_scale, 1.0).astype(np.float32)
    return {
        "state_mean": state_mean,
        "state_scale": state_scale,
        "motion_mean": motion_mean,
        "motion_scale": motion_scale,
    }


def normalize_transition_torch(
    *,
    sim_state,
    sim_motion,
    ref_state,
    ref_motion,
    stats: dict[str, Any],
    clip: float,
):
    import torch

    curr_error = ref_state - sim_state
    motion_error = ref_motion - sim_motion
    next_error = curr_error + motion_error
    error = torch.cat(
        [
            next_error / stats["state_scale"],
            motion_error / stats["motion_scale"],
        ],
        dim=-1,
    )
    context = torch.cat(
        [
            (ref_state - stats["state_mean"]) / stats["state_scale"],
            (ref_motion - stats["motion_mean"]) / stats["motion_scale"],
        ],
        dim=-1,
    )
    return (
        torch.clamp(error, -float(clip), float(clip)),
        torch.clamp(context, -float(clip), float(clip)),
    )


def raw_score(model, error, context):
    import torch

    return model(torch.cat([error, context], dim=-1)).squeeze(-1)


def anchored_score(model, error, context):
    return raw_score(model, error, context) - raw_score(
        model, error.new_zeros(error.shape), context
    )


def phase_derangement_numpy(
    phase: np.ndarray,
    min_phase_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Match the online same-motion, half-sorted-batch derangement."""

    phase = np.asarray(phase, dtype=np.float32).reshape(-1)
    count = phase.size
    partner = np.arange(count, dtype=np.int64)
    valid = np.zeros(count, dtype=np.bool_)
    distance = np.zeros(count, dtype=np.float32)
    if count < 2:
        return partner, valid, distance
    ordered = np.argsort(phase, kind="stable")
    shift = max(1, count // 2)
    ordered_partner = np.roll(ordered, -shift)
    partner[ordered] = ordered_partner
    delta = np.abs(phase[ordered] - phase[ordered_partner])
    delta = np.minimum(delta, 1.0 - delta)
    distance[ordered] = delta
    valid[ordered] = np.logical_and(
        ordered_partner != ordered, delta >= float(min_phase_distance)
    )
    return partner, valid, distance


def _prepare_data(
    bundle: dict[str, np.ndarray], config: FitConfig
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    alive = np.asarray(bundle["alive"], dtype=np.bool_)
    rows = np.flatnonzero(alive)
    x_t = np.asarray(bundle["x_t"][rows], dtype=np.float32)
    x_t1 = np.asarray(bundle["x_t1"][rows], dtype=np.float32)
    r_t = np.asarray(bundle["r_t"][rows], dtype=np.float32)
    r_t1 = np.asarray(bundle["r_t1"][rows], dtype=np.float32)
    phase = np.asarray(bundle["phase"][rows], dtype=np.float32)
    episode = np.asarray(bundle["episode_id"][rows], dtype=np.int64)
    ref_motion = r_t1 - r_t
    transition = {
        "sim_state": x_t,
        "sim_motion": x_t1 - x_t,
        "ref_state": r_t,
        "ref_motion": ref_motion,
        "phase": phase,
        "episode": episode,
    }
    stats = reference_statistics(r_t, ref_motion)
    unique_episodes = np.unique(episode)
    if unique_episodes.size < 4:
        raise ValueError("offline fitting requires at least four episodes")
    rng = np.random.default_rng(config.seed)
    shuffled_episodes = rng.permutation(unique_episodes)
    validation_count = max(
        1, int(round(config.validation_fraction * unique_episodes.size))
    )
    validation_ids = shuffled_episodes[:validation_count]
    train_ids = shuffled_episodes[validation_count:]
    split = {
        "train_rows": np.flatnonzero(np.isin(episode, train_ids)),
        "validation_rows": np.flatnonzero(np.isin(episode, validation_ids)),
        "train_episode_ids": train_ids,
        "validation_episode_ids": validation_ids,
    }
    if split["train_rows"].size == 0 or split["validation_rows"].size == 0:
        raise ValueError("episode-disjoint split produced an empty partition")
    return transition, stats, split


def _torch_stats(stats: dict[str, np.ndarray], device: str):
    import torch

    return {
        key: torch.as_tensor(value, device=device, dtype=torch.float32)
        for key, value in stats.items()
    }


def _batch_tensors(
    transition: dict[str, np.ndarray], rows: np.ndarray, device: str
):
    import torch

    return {
        key: torch.as_tensor(transition[key][rows], device=device, dtype=torch.float32)
        for key in ("sim_state", "sim_motion", "ref_state", "ref_motion")
    }


def _loss_batch(
    model,
    transition: dict[str, np.ndarray],
    rows: np.ndarray,
    stats_torch: dict[str, Any],
    config: FitConfig,
    *,
    generator,
    create_graph: bool,
):
    import torch

    data = _batch_tensors(transition, rows, next(model.parameters()).device)
    policy_error, context = normalize_transition_torch(
        **data, stats=stats_torch, clip=config.input_clip
    )

    partner_local, valid_shuffle, _ = phase_derangement_numpy(
        transition["phase"][rows], config.min_phase_distance
    )
    partner_local = partner_local[valid_shuffle]
    context_rows = np.flatnonzero(valid_shuffle)
    if context_rows.size == 0:
        raise ValueError("batch has no valid wrong-phase reference derangement")
    shuffle_data = {
        "sim_state": data["ref_state"][partner_local],
        "sim_motion": data["ref_motion"][partner_local],
        "ref_state": data["ref_state"][context_rows],
        "ref_motion": data["ref_motion"][context_rows],
    }
    shuffle_error, shuffle_context = normalize_transition_torch(
        **shuffle_data, stats=stats_torch, clip=config.input_clip
    )
    shuffle_rms = torch.sqrt(torch.mean(torch.square(shuffle_error), dim=-1))
    shuffle_keep = shuffle_rms > config.min_shuffle_rms
    shuffle_error = shuffle_error[shuffle_keep]
    shuffle_context = shuffle_context[shuffle_keep]
    if shuffle_error.shape[0] == 0:
        raise ValueError("batch wrong-phase errors are numerically empty")

    policy_advantage = anchored_score(model, policy_error, context)
    shuffle_advantage = anchored_score(model, shuffle_error, shuffle_context)
    wasserstein = 0.5 * (
        torch.mean(policy_advantage) + torch.mean(shuffle_advantage)
    )
    wasserstein_scale = math.sqrt(float(policy_error.shape[-1]))
    wasserstein_objective = wasserstein / wasserstein_scale

    def interp_norm(error, curr_context):
        alpha = torch.rand(
            [error.shape[0], 1],
            device=error.device,
            dtype=error.dtype,
            generator=generator,
        )
        interp = (alpha * error).detach().requires_grad_(True)
        score = raw_score(model, interp, curr_context)
        grad = torch.autograd.grad(
            score,
            interp,
            grad_outputs=torch.ones_like(score),
            create_graph=create_graph,
            retain_graph=create_graph,
            only_inputs=True,
        )[0]
        return torch.linalg.vector_norm(grad, dim=-1)

    policy_norm = interp_norm(policy_error, context)
    shuffle_norm = interp_norm(shuffle_error, shuffle_context)
    gp_norm = torch.cat([policy_norm, shuffle_norm], dim=0)
    gp = torch.mean(torch.square(gp_norm - config.gp_target))
    loss = wasserstein_objective + config.gp_weight * gp
    return loss, {
        "loss": float(loss.detach().cpu()),
        "wasserstein": float(wasserstein.detach().cpu()),
        "wasserstein_objective": float(
            wasserstein_objective.detach().cpu()
        ),
        "wasserstein_scale": wasserstein_scale,
        "gp": float(gp.detach().cpu()),
        "gp_norm": float(torch.mean(gp_norm).detach().cpu()),
        "policy_advantage": float(torch.mean(policy_advantage).detach().cpu()),
        "shuffle_advantage": float(torch.mean(shuffle_advantage).detach().cpu()),
        "policy_negative_fraction": float(
            torch.mean((policy_advantage < 0).float()).detach().cpu()
        ),
        "shuffle_negative_fraction": float(
            torch.mean((shuffle_advantage < 0).float()).detach().cpu()
        ),
    }


def _aggregate(metrics: list[dict[str, float]]) -> dict[str, float]:
    return {
        key: float(np.mean([item[key] for item in metrics]))
        for key in metrics[0]
    }


def fit(args: argparse.Namespace) -> dict[str, object]:
    import torch

    config = FitConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gp_weight=args.gp_weight,
        validation_fraction=args.validation_fraction,
        seed=args.seed,
    )
    if config.epochs <= 0 or config.batch_size < 4:
        raise ValueError("epochs must be positive and batch-size at least four")
    if not 0 < config.validation_fraction < 0.5:
        raise ValueError("validation-fraction must lie in (0, 0.5)")
    source = resolve_repo_path(args.train_transitions)
    output = resolve_repo_path(args.out)
    if not source.is_file():
        raise FileNotFoundError(source)
    bundle = load_transition_bundle(source)
    contract = validate_transition_bundle(bundle)
    if not 0 < args.velocity_dim < int(contract["phi_dim"]):
        raise ValueError("velocity-dim must lie in (0, phi_dim)")
    transition, stats, split = _prepare_data(bundle, config)

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available() and torch.device(args.device).type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
    model = build_transition_critic(int(contract["phi_dim"])).to(args.device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.0, 0.9),
        weight_decay=0.0,
    )
    stats_torch = _torch_stats(stats, args.device)
    generator = torch.Generator(device=args.device)
    generator.manual_seed(config.seed + 1)

    best_validation = math.inf
    best_state = None
    history: list[dict[str, object]] = []
    train_rng = np.random.default_rng(config.seed + 2)
    for epoch in range(config.epochs):
        model.train()
        order = train_rng.permutation(split["train_rows"])
        train_metrics = []
        for start in range(0, order.size, config.batch_size):
            rows = order[start : start + config.batch_size]
            if rows.size < 4:
                continue
            optimizer.zero_grad(set_to_none=True)
            loss, metrics = _loss_batch(
                model,
                transition,
                rows,
                stats_torch,
                config,
                generator=generator,
                create_graph=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            train_metrics.append(metrics)

        # Fixed generator state per epoch keeps validation independent of the
        # number/order of training batches and makes model selection auditable.
        model.eval()
        validation_generator = torch.Generator(device=args.device)
        validation_generator.manual_seed(config.seed + 1000 + epoch)
        validation_metrics = []
        for start in range(0, split["validation_rows"].size, config.batch_size):
            rows = split["validation_rows"][start : start + config.batch_size]
            if rows.size < 4:
                continue
            _, metrics = _loss_batch(
                model,
                transition,
                rows,
                stats_torch,
                config,
                generator=validation_generator,
                create_graph=False,
            )
            validation_metrics.append(metrics)
        train_summary = _aggregate(train_metrics)
        validation_summary = _aggregate(validation_metrics)
        row = {
            "epoch": epoch,
            "train": train_summary,
            "validation": validation_summary,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        if validation_summary["loss"] < best_validation:
            best_validation = validation_summary["loss"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("offline fit produced no checkpoint")
    checkpoint = {
        "offline_transition_critic_version": OFFLINE_CHECKPOINT_VERSION,
        "model_state_dict": best_state,
        "phi_dim": int(contract["phi_dim"]),
        "velocity_dim": int(args.velocity_dim),
        "normalization": {
            key: torch.from_numpy(value.copy()) for key, value in stats.items()
        },
        "fit_config": asdict(config),
        "provenance": {
            "training_transition_file": str(source),
            "training_transition_sha256": sha256_file(source),
            "behavior_labels_read": False,
            "train_episode_ids": split["train_episode_ids"].tolist(),
            "validation_episode_ids": split["validation_episode_ids"].tolist(),
        },
        "best_validation_loss": best_validation,
        "history": history,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix(output.suffix + ".tmp")
    torch.save(checkpoint, temp)
    os.replace(temp, output)
    return {
        "output": str(output),
        "output_sha256": sha256_file(output),
        "contract": contract,
        "train_episodes": int(split["train_episode_ids"].size),
        "validation_episodes": int(split["validation_episode_ids"].size),
        "best_validation_loss": best_validation,
        "last_epoch": history[-1],
        "behavior_labels_read": False,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-transitions", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gp-weight", type=float, default=10.0)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=918273)
    parser.add_argument(
        "--velocity-dim",
        type=int,
        default=34,
        help="root linear/angular plus DoF velocity features in phi",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = fit(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
