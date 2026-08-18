#!/usr/bin/env python3
"""Evaluate causal interventions on an aligned-ADD policy command.

This is an inference-only diagnostic.  It keeps the trained model, ADD reward,
environment, and deterministic mean action fixed while altering only the raw
actor observation command:

  baseline: [self, e, m]
  zero_e:   [self, 0, m]
  zero_m:   [self, e, 0]
  shuffle_m:[self, e, m from another environment/initial phase]

Run each condition in a fresh process with the same seed so simulator initial
conditions and sampled reference phases are matched.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MIMICKIT_ROOT = REPO_ROOT / "mimickit"
if str(MIMICKIT_ROOT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT_ROOT))

import envs.base_env as base_env  # noqa: E402
import envs.env_builder as env_builder  # noqa: E402
import learning.agent_builder as agent_builder  # noqa: E402
import learning.base_agent as base_agent  # noqa: E402
import util.mp_util as mp_util  # noqa: E402
import util.util as util  # noqa: E402


CONDITIONS = ("baseline", "zero_e", "zero_m", "shuffle_m")
TRACKING_NAMES = (
    "root_pos_err",
    "root_rot_err",
    "body_pos_err",
    "body_rot_err",
    "dof_vel_err",
    "root_vel_err",
    "root_ang_vel_err",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument(
        "--model-file",
        default="output/aligned_add_roll_2k_8192_seed0/model.pt",
    )
    parser.add_argument(
        "--env-config",
        default="data/envs/aligned_add_humanoid_roll_eval_env.yaml",
    )
    parser.add_argument(
        "--agent-config",
        default="data/agents/aligned_add_humanoid_agent.yaml",
    )
    parser.add_argument(
        "--engine-config",
        default="data/engines/isaac_lab_engine.yaml",
    )
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default="output/aligned_add_roll_interventions_seed0")
    return parser.parse_args()


def stats(x: np.ndarray) -> dict[str, float]:
    x = np.asarray(x, dtype=np.float64)
    return {
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "median": float(np.median(x)),
        "q10": float(np.quantile(x, 0.10)),
        "q90": float(np.quantile(x, 0.90)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
    }


def intervene(
    obs: torch.Tensor,
    condition: str,
    self_dim: int,
    command_dim: int,
    shuffle_index: torch.Tensor,
) -> torch.Tensor:
    if condition == "baseline":
        return obs

    obs = obs.clone()
    error_slice = slice(self_dim, self_dim + command_dim)
    motion_slice = slice(self_dim + command_dim, self_dim + 2 * command_dim)

    if condition == "zero_e":
        obs[:, error_slice] = 0.0
    elif condition == "zero_m":
        obs[:, motion_slice] = 0.0
    elif condition == "shuffle_m":
        motion = obs[:, motion_slice].clone()
        obs[:, motion_slice] = motion[shuffle_index]
    else:
        raise ValueError(f"unsupported intervention: {condition}")
    return obs


def get_root_state(env) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    char_id = env._get_char_id()
    root_pos = env._engine.get_root_pos(char_id).detach().clone()
    root_ang_vel = env._engine.get_root_ang_vel(char_id).detach().clone()
    ref_root_pos = env._ref_root_pos.detach().clone()
    ref_root_ang_vel = env._ref_root_ang_vel.detach().clone()
    return root_pos, root_ang_vel, ref_root_pos, ref_root_ang_vel


def main() -> None:
    args = parse_args()
    if args.num_envs < 2 and args.condition == "shuffle_m":
        raise ValueError("shuffle_m requires at least two environments")

    util.set_rand_seed(args.seed)
    mp_util.init(rank=0, num_procs=1, device=args.device, master_port=6381)

    env = env_builder.build_env(
        args.env_config,
        args.engine_config,
        args.num_envs,
        args.device,
        visualize=False,
        record_video=False,
    )
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)

    obs, info = env.reset()
    self_dim = env.get_aligned_self_obs_dim()
    command_dim = env.get_aligned_command_dim()
    expected_dim = self_dim + 2 * command_dim
    if obs.shape[-1] != expected_dim:
        raise RuntimeError(f"aligned observation size {obs.shape[-1]} != {expected_dim}")

    # A deterministic derangement.  Because reference phase offsets remain
    # fixed through the episode, this supplies each environment with another
    # trajectory's phase-conditioned tangent at every control step.
    shuffle_index = torch.roll(
        torch.arange(args.num_envs, device=args.device, dtype=torch.long), shifts=1
    )

    root_pos0, _, ref_root_pos0, _ = get_root_state(env)
    sim_ang_integral = torch.zeros_like(root_pos0)
    ref_ang_integral = torch.zeros_like(root_pos0)
    env_reward_sum = torch.zeros(args.num_envs, device=args.device)
    add_reward_sum = torch.zeros(args.num_envs, device=args.device)
    first_done_step = torch.full(
        (args.num_envs,), args.steps, device=args.device, dtype=torch.long
    )
    done_counts = {flag.name.lower(): 0 for flag in base_env.DoneFlags}
    dt = float(env._engine.get_timestep())

    with torch.no_grad():
        for step in range(args.steps):
            actor_obs = intervene(
                obs, args.condition, self_dim, command_dim, shuffle_index
            )
            action, _ = agent._decide_action(actor_obs, info)
            obs, reward, done, info = env.step(action)

            _, root_ang_vel, _, ref_root_ang_vel = get_root_state(env)
            sim_ang_integral += root_ang_vel * dt
            ref_ang_integral += ref_root_ang_vel * dt
            env_reward_sum += reward
            disc_diff = info["disc_obs_demo"] - info["disc_obs"]
            norm_disc_diff = agent._disc_obs_norm.normalize(disc_diff)
            add_reward_sum += agent._calc_disc_rewards(norm_disc_diff)

            newly_done = (done != base_env.DoneFlags.NULL.value) & (
                first_done_step == args.steps
            )
            first_done_step[newly_done] = step + 1
            for flag in base_env.DoneFlags:
                done_counts[flag.name.lower()] += int(
                    torch.sum(done == flag.value).item()
                )

    root_pos1, _, ref_root_pos1, _ = get_root_state(env)
    sim_disp = root_pos1 - root_pos0
    ref_disp = ref_root_pos1 - ref_root_pos0

    ref_ang_sq = torch.sum(ref_ang_integral * ref_ang_integral, dim=-1)
    winding_ratio = torch.sum(sim_ang_integral * ref_ang_integral, dim=-1) / torch.clamp_min(
        ref_ang_sq, 1e-8
    )
    ref_turns = torch.linalg.vector_norm(ref_ang_integral, dim=-1) / (2.0 * math.pi)
    sim_turns = winding_ratio * ref_turns

    ref_disp_sq = torch.sum(ref_disp * ref_disp, dim=-1)
    displacement_ratio = torch.sum(sim_disp * ref_disp, dim=-1) / torch.clamp_min(
        ref_disp_sq, 1e-8
    )
    lateral_disp = torch.linalg.vector_norm(
        sim_disp
        - displacement_ratio.unsqueeze(-1) * ref_disp,
        dim=-1,
    )

    winding_np = winding_ratio.cpu().numpy()
    displacement_np = displacement_ratio.cpu().numpy()
    shortcut = winding_np < 0.5
    diagnostics = env.record_diagnostics()
    tracking = {
        name: float(diagnostics[name].item())
        for name in TRACKING_NAMES
        if name in diagnostics
    }

    summary = {
        "condition": args.condition,
        "model_file": str(Path(args.model_file).resolve()),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": args.steps,
        "horizon_seconds": args.steps * dt,
        "mean_step_add_reward": stats((add_reward_sum / args.steps).cpu().numpy()),
        "mean_step_env_reward": stats((env_reward_sum / args.steps).cpu().numpy()),
        "winding_ratio": stats(winding_np),
        "reference_turns": stats(ref_turns.cpu().numpy()),
        "sim_turns": stats(sim_turns.cpu().numpy()),
        "displacement_ratio": stats(displacement_np),
        "lateral_displacement": stats(lateral_disp.cpu().numpy()),
        "shortcut_rate_winding_lt_0_5": float(np.mean(shortcut)),
        "first_done_step": stats(first_done_step.cpu().numpy()),
        "done_counts": done_counts,
        "tracking_error": tracking,
    }

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.condition}_seed{args.seed}"
    json_file = out_dir / f"{stem}.json"
    npz_file = out_dir / f"{stem}.npz"
    json_file.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    np.savez_compressed(
        npz_file,
        add_reward_sum=add_reward_sum.cpu().numpy(),
        env_reward_sum=env_reward_sum.cpu().numpy(),
        winding_ratio=winding_np,
        reference_turns=ref_turns.cpu().numpy(),
        sim_turns=sim_turns.cpu().numpy(),
        displacement_ratio=displacement_np,
        lateral_displacement=lateral_disp.cpu().numpy(),
        first_done_step=first_done_step.cpu().numpy(),
        shuffle_index=shuffle_index.cpu().numpy(),
    )

    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {json_file}")
    print(f"Wrote {npz_file}")


if __name__ == "__main__":
    main()
