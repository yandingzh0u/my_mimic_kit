#!/usr/bin/env python3
"""Score held-out Roll transitions with a fitted phase transition critic.

This is evaluation only.  Successful/shortcut file membership is used to
report a ranking and is never passed into the critic or an optimizer.
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
    atomic_savez_compressed,
    load_transition_bundle,
    validate_transition_bundle,
)
from tools.paper_eval.evaluate_checkpoint import (  # noqa: E402
    resolve_repo_path,
    sha256_file,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--critic-model-file", required=True)
    parser.add_argument("--critic-env-config", default="")
    parser.add_argument("--critic-agent-config", default="")
    parser.add_argument(
        "--engine-config", default="data/engines/isaac_lab_engine.yaml"
    )
    parser.add_argument("--success-transitions", required=True)
    parser.add_argument("--shortcut-transitions", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--master-port", type=int, default=6392)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument(
        "--gp-alphas",
        default="0.1,0.3,0.5,0.7,0.9",
        help="comma-separated interpolation points for the held-out GP audit",
    )
    parser.add_argument("--velocity-dim", type=int, default=34)
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


def _parse_gp_alphas(value: str) -> tuple[float, ...]:
    try:
        result = tuple(float(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("gp-alphas must be comma-separated numbers") from exc
    if not result or any(not 0 < item < 1 for item in result):
        raise ValueError("every gp alpha must lie strictly between zero and one")
    if len(set(result)) != len(result):
        raise ValueError("gp-alphas must not contain duplicates")
    return result


def _as_tensor(array: np.ndarray, device: str):
    import torch

    return torch.as_tensor(array, device=device, dtype=torch.float32)


class _OfflineModelAdapter:
    def __init__(self, network):
        self._network = network

    def eval_transition_score(self, transition_error, reference_context):
        import torch

        return self._network(
            torch.cat([transition_error, reference_context], dim=-1)
        )

    def eval_anchored_score(self, transition_error, reference_context):
        score = self.eval_transition_score(
            transition_error, reference_context
        )
        reference = self.eval_transition_score(
            transition_error.new_zeros(transition_error.shape),
            reference_context,
        )
        return score - reference


class _OfflineAgentAdapter:
    def __init__(self, network, stats, clip: float, phase_distance: float):
        self._model = _OfflineModelAdapter(network)
        self._stats = stats
        self._clip = float(clip)
        self._phase_shuffle_min_distance = float(phase_distance)

    def _normalize_transition(
        self, sim_state, sim_motion, ref_state, ref_motion, **_metadata
    ):
        from tools.phase_transition_critic.offline_fit import (
            normalize_transition_torch,
        )

        return normalize_transition_torch(
            sim_state=sim_state,
            sim_motion=sim_motion,
            ref_state=ref_state,
            ref_motion=ref_motion,
            stats=self._stats,
            clip=self._clip,
        )


def _try_load_offline_agent(path: Path, device: str):
    import torch
    from tools.phase_transition_critic.offline_fit import (
        OFFLINE_CHECKPOINT_VERSION,
        build_transition_critic,
    )

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict) or "offline_transition_critic_version" not in checkpoint:
        return None
    if int(checkpoint["offline_transition_critic_version"]) != OFFLINE_CHECKPOINT_VERSION:
        raise ValueError("unsupported offline transition critic checkpoint")
    phi_dim = int(checkpoint["phi_dim"])
    network = build_transition_critic(phi_dim)
    network.load_state_dict(checkpoint["model_state_dict"], strict=True)
    network.to(device).eval()
    stats = {
        key: value.to(device=device, dtype=torch.float32)
        for key, value in checkpoint["normalization"].items()
    }
    config = checkpoint["fit_config"]
    agent = _OfflineAgentAdapter(
        network,
        stats,
        clip=float(config["input_clip"]),
        phase_distance=float(config["min_phase_distance"]),
    )
    return agent, phi_dim, int(checkpoint.get("velocity_dim", 34)), checkpoint


def _transition_arrays(bundle: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    return {
        "sim_state": np.asarray(bundle["x_t"], dtype=np.float32),
        "sim_motion": np.asarray(bundle["x_t1"] - bundle["x_t"], dtype=np.float32),
        "ref_state": np.asarray(bundle["r_t"], dtype=np.float32),
        "ref_motion": np.asarray(bundle["r_t1"] - bundle["r_t"], dtype=np.float32),
    }


def _score_chunks(
    agent,
    transition: dict[str, np.ndarray],
    rows: np.ndarray,
    device: str,
    batch_size: int,
) -> np.ndarray:
    import torch

    result = np.full(transition["sim_state"].shape[0], np.nan, dtype=np.float32)
    with torch.no_grad():
        for start in range(0, rows.size, batch_size):
            index = rows[start : start + batch_size]
            tensors = {
                key: _as_tensor(value[index], device)
                for key, value in transition.items()
            }
            error, context = agent._normalize_transition(**tensors)
            score = agent._model.eval_anchored_score(error, context).squeeze(-1)
            result[index] = score.detach().cpu().numpy()
    return result


def _gradient_audit_chunks(
    agent,
    transition: dict[str, np.ndarray],
    rows: np.ndarray,
    device: str,
    batch_size: int,
    gp_alpha: float,
    pose_dim: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    import torch

    num_rows = transition["sim_state"].shape[0]
    phi_dim = transition["sim_state"].shape[1]
    pose_indices = torch.cat(
        [
            torch.arange(0, pose_dim, device=device),
            torch.arange(phi_dim, phi_dim + pose_dim, device=device),
        ]
    )
    velocity_indices = torch.cat(
        [
            torch.arange(pose_dim, phi_dim, device=device),
            torch.arange(phi_dim + pose_dim, 2 * phi_dim, device=device),
        ]
    )
    gp_norm = np.full(num_rows, np.nan, dtype=np.float32)
    pose_rms = np.full(num_rows, np.nan, dtype=np.float32)
    velocity_rms = np.full(num_rows, np.nan, dtype=np.float32)
    next_error_rms = np.full(num_rows, np.nan, dtype=np.float32)
    motion_error_rms = np.full(num_rows, np.nan, dtype=np.float32)
    for start in range(0, rows.size, batch_size):
        index = rows[start : start + batch_size]
        tensors = {
            key: _as_tensor(value[index], device)
            for key, value in transition.items()
        }
        error, context = agent._normalize_transition(**tensors)
        interp = (float(gp_alpha) * error).detach().requires_grad_(True)
        raw_score = agent._model.eval_transition_score(interp, context).squeeze(-1)
        grad = torch.autograd.grad(
            raw_score,
            interp,
            grad_outputs=torch.ones_like(raw_score),
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]
        gp_norm[index] = torch.linalg.vector_norm(grad, dim=-1).detach().cpu().numpy()
        pose_rms[index] = torch.sqrt(
            torch.mean(torch.square(grad[:, pose_indices]), dim=-1)
        ).detach().cpu().numpy()
        velocity_rms[index] = torch.sqrt(
            torch.mean(torch.square(grad[:, velocity_indices]), dim=-1)
        ).detach().cpu().numpy()
        next_error_rms[index] = torch.sqrt(
            torch.mean(torch.square(grad[:, :phi_dim]), dim=-1)
        ).detach().cpu().numpy()
        motion_error_rms[index] = torch.sqrt(
            torch.mean(torch.square(grad[:, phi_dim:]), dim=-1)
        ).detach().cpu().numpy()
    return (
        gp_norm,
        pose_rms,
        velocity_rms,
        next_error_rms,
        motion_error_rms,
    )


def _build_reference_derangement(
    bundle: dict[str, np.ndarray],
    transition: dict[str, np.ndarray],
    device: str,
    min_phase_distance: float,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    from tools.phase_transition_critic.offline_fit import (
        phase_derangement_numpy,
    )

    alive = np.asarray(bundle["alive"], dtype=np.bool_)
    valid_rows = np.flatnonzero(alive)
    partner_local, valid_local, distance = phase_derangement_numpy(
        np.asarray(bundle["phase"])[valid_rows],
        min_phase_distance=min_phase_distance,
    )
    partner = valid_rows[partner_local]
    shuffle_alive = np.zeros(alive.shape, dtype=np.bool_)
    shuffle_alive[valid_rows] = valid_local
    phase_distance = np.full(alive.shape, np.nan, dtype=np.float32)
    phase_distance[valid_rows] = distance

    # Training's hard negative places reference transition j in context i.
    shuffled = {
        "sim_state": transition["ref_state"].copy(),
        "sim_motion": transition["ref_motion"].copy(),
        "ref_state": transition["ref_state"],
        "ref_motion": transition["ref_motion"],
    }
    shuffled["sim_state"][valid_rows] = transition["ref_state"][partner]
    shuffled["sim_motion"][valid_rows] = transition["ref_motion"][partner]
    return shuffled, shuffle_alive, phase_distance


def score(args: argparse.Namespace) -> dict[str, object]:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    gp_alphas = _parse_gp_alphas(args.gp_alphas)

    paths = {
        "model": resolve_repo_path(args.critic_model_file),
        "success": resolve_repo_path(args.success_transitions),
        "shortcut": resolve_repo_path(args.shortcut_transitions),
    }
    if args.critic_env_config:
        paths["env"] = resolve_repo_path(args.critic_env_config)
    if args.critic_agent_config:
        paths["agent"] = resolve_repo_path(args.critic_agent_config)
    if args.engine_config:
        paths["engine"] = resolve_repo_path(args.engine_config)
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(path)
    success_bundle = load_transition_bundle(paths["success"])
    shortcut_bundle = load_transition_bundle(paths["shortcut"])
    success_report = validate_transition_bundle(success_bundle)
    shortcut_report = validate_transition_bundle(shortcut_bundle)
    if success_report["phi_dim"] != shortcut_report["phi_dim"]:
        raise ValueError("success and shortcut phi dimensions differ")

    import torch

    original_cwd = Path.cwd()
    os.chdir(REPO_ROOT)
    try:
        phi_dim = int(success_report["phi_dim"])
        offline = _try_load_offline_agent(paths["model"], args.device)
        if offline is not None:
            agent, expected_phi_dim, velocity_dim, offline_checkpoint = offline
            if phi_dim != expected_phi_dim:
                raise ValueError(
                    f"rollout phi_dim={phi_dim}, offline critic expects "
                    f"{expected_phi_dim}"
                )
        else:
            required_online = [key for key in ("env", "agent", "engine") if key not in paths]
            if required_online:
                raise ValueError(
                    "online agent checkpoint requires --critic-env-config, "
                    "--critic-agent-config, and --engine-config"
                )
            if str(MIMICKIT_ROOT) not in sys.path:
                sys.path.insert(0, str(MIMICKIT_ROOT))
            import envs.env_builder as env_builder
            import learning.agent_builder as agent_builder
            import learning.base_agent as base_agent
            import util.mp_util as mp_util
            import util.util as util

            util.set_rand_seed(0)
            mp_util.init(
                rank=0,
                num_procs=1,
                device=args.device,
                master_port=args.master_port,
            )
            env = env_builder.build_env(
                str(paths["env"]),
                str(paths["engine"]),
                1,
                args.device,
                visualize=False,
                record_video=False,
            )
            agent = agent_builder.build_agent(
                str(paths["agent"]), env, args.device
            )
            agent.load(str(paths["model"]))
            agent.eval()
            agent.set_mode(base_agent.AgentMode.TEST)
            if phi_dim != int(env.get_aligned_command_dim()):
                raise ValueError(
                    f"rollout phi_dim={phi_dim}, critic expects "
                    f"{env.get_aligned_command_dim()}"
                )
            velocity_dim = 6 + int(env.get_action_space().shape[-1])
        pose_dim = phi_dim - velocity_dim
        if pose_dim <= 0:
            raise ValueError("invalid pose/velocity feature split")

        success_transition = _transition_arrays(success_bundle)
        shortcut_transition = _transition_arrays(shortcut_bundle)
        success_rows = np.flatnonzero(success_bundle["alive"])
        shortcut_rows = np.flatnonzero(shortcut_bundle["alive"])
        success_score = _score_chunks(
            agent, success_transition, success_rows, args.device, args.batch_size
        )
        shortcut_score = _score_chunks(
            agent, shortcut_transition, shortcut_rows, args.device, args.batch_size
        )

        reference_transition = {
            "sim_state": success_transition["ref_state"],
            "sim_motion": success_transition["ref_motion"],
            "ref_state": success_transition["ref_state"],
            "ref_motion": success_transition["ref_motion"],
        }
        reference_score = _score_chunks(
            agent, reference_transition, success_rows, args.device, args.batch_size
        )
        shuffled_transition, shuffle_alive, phase_distance = (
            _build_reference_derangement(
                success_bundle,
                success_transition,
                args.device,
                float(agent._phase_shuffle_min_distance),
            )
        )
        shuffled_rows = np.flatnonzero(shuffle_alive)
        shuffled_score = _score_chunks(
            agent,
            shuffled_transition,
            shuffled_rows,
            args.device,
            args.batch_size,
        )

        success_audits = []
        shortcut_audits = []
        shuffle_audits = []
        for gp_alpha in gp_alphas:
            success_audits.append(_gradient_audit_chunks(
                agent,
                success_transition,
                success_rows,
                args.device,
                args.batch_size,
                gp_alpha,
                pose_dim,
            ))
            shortcut_audits.append(_gradient_audit_chunks(
                agent,
                shortcut_transition,
                shortcut_rows,
                args.device,
                args.batch_size,
                gp_alpha,
                pose_dim,
            ))
            shuffle_audits.append(_gradient_audit_chunks(
                agent,
                shuffled_transition,
                shuffled_rows,
                args.device,
                args.batch_size,
                gp_alpha,
                pose_dim,
            ))

        def mean_audit(audits, index):
            return np.nanmean(
                np.stack([audit[index] for audit in audits], axis=0),
                axis=0,
            )

        pose_sensitivity = mean_audit(success_audits, 1)
        velocity_sensitivity = mean_audit(success_audits, 2)
        next_error_sensitivity = mean_audit(success_audits, 3)
        motion_error_sensitivity = mean_audit(success_audits, 4)
        gp_norm = np.concatenate(
            [
                audit[0][np.isfinite(audit[0])]
                for audits in (success_audits, shortcut_audits, shuffle_audits)
                for audit in audits
            ]
        )

        metadata = {
            "schema_version": 1,
            "evaluation_only": True,
            "behavior_labels_used_for_critic_fit": False,
            "critic_model_sha256": sha256_file(paths["model"]),
            "success_transition_sha256": sha256_file(paths["success"]),
            "shortcut_transition_sha256": sha256_file(paths["shortcut"]),
            "git_commit": _git_revision(),
            "pose_dim": pose_dim,
            "velocity_dim": velocity_dim,
            "gp_alphas": list(gp_alphas),
        }
        arrays: dict[str, Any] = {
            "schema_version": np.asarray(1, dtype=np.int64),
            "metadata_json": np.asarray(
                json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            ),
            "reference_score": reference_score,
            "reference_episode_id": success_bundle["episode_id"],
            "reference_alive": success_bundle["alive"],
            "success_score": success_score,
            "success_episode_id": success_bundle["episode_id"],
            "success_alive": success_bundle["alive"],
            "shortcut_score": shortcut_score,
            "shortcut_episode_id": shortcut_bundle["episode_id"],
            "shortcut_alive": shortcut_bundle["alive"],
            "reference_phase_shuffled_score": shuffled_score,
            "reference_phase_shuffled_alive": shuffle_alive,
            "reference_phase_distance": phase_distance,
            "pose_sensitivity": pose_sensitivity,
            "velocity_sensitivity": velocity_sensitivity,
            "next_error_sensitivity": next_error_sensitivity,
            "motion_error_sensitivity": motion_error_sensitivity,
            "gp_norm": gp_norm,
        }
        atomic_savez_compressed(resolve_repo_path(args.out), **arrays)
        return {
            "output": str(resolve_repo_path(args.out)),
            "success": success_report,
            "shortcut": shortcut_report,
            "reference_anchor_max_abs": float(
                np.nanmax(np.abs(reference_score))
            ),
            "phase_shuffle_valid_fraction": float(np.mean(shuffle_alive)),
            "gp_samples": int(gp_norm.size),
            "pose_dim": pose_dim,
            "velocity_dim": velocity_dim,
        }
    finally:
        os.chdir(original_cwd)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = score(args)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
