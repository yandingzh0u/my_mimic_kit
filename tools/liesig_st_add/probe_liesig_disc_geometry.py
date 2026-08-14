"""Offline probe of the LieSig discriminator's geometry along the ray to zero.

The shared-head ST-ADD v2 failure was invisible in the loss and obvious in the
geometry: the discriminator won so hard that the policy region sat on a flat,
saturated shell and softplus(z) delivered almost no reward gradient. This tool
makes that failure mode directly observable for any ADD-style checkpoint,
without training anything.

It rolls out the policy, collects real differentials Delta, and scans

    D(alpha * Delta)   and   r(alpha) = scale * softplus(D(alpha * Delta))

for alpha in [0, 1]: alpha = 0 is the exact ADD ideal point, alpha = 1 is the
policy's own state. A healthy discriminator interpolates smoothly between the
two; a saturated one is flat at alpha = 1 (no reward signal to climb) or
collapses immediately away from alpha = 0.

Reported per alpha:
  logit mean/std          where the policy sits on the logit scale
  reward mean             scale * softplus(logit), the actual PPO reward
  d logit / d alpha       finite-difference slope along the ray
  |grad_Delta D|          gradient magnitude at that point (reward signal)

Also reports the per-block share of |grad_Delta D| (state / level 1 / level 2)
at alpha = 1, i.e. which part of the differential the discriminator actually
uses on policy data.

Example:
    python tools/liesig_st_add/probe_liesig_disc_geometry.py \
        --env_config output/liesig_l2_roll/env_config.yaml \
        --engine_config output/liesig_l2_roll/engine_config.yaml \
        --agent_config output/liesig_l2_roll/agent_config.yaml \
        --model_file output/liesig_l2_roll/model.pt \
        --num_envs 64 --steps 120
"""

import argparse
import os
import sys

import numpy as np
import torch

REPO_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_DIR, "mimickit"))

import envs.env_builder as env_builder
import learning.agent_builder as agent_builder
import learning.base_agent as base_agent
import util.mp_util as mp_util
import util.util as util

def parse_args():
    parser = argparse.ArgumentParser(description="LieSig discriminator geometry probe")
    parser.add_argument("--env_config", type=str, required=True)
    parser.add_argument("--engine_config", type=str, required=True)
    parser.add_argument("--agent_config", type=str, required=True)
    parser.add_argument("--model_file", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--num_alphas", type=int, default=11)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--rand_seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="")
    return parser.parse_args()

def build(args):
    mp_util.init(0, 1, args.device, 6000)
    util.set_rand_seed(args.rand_seed)

    env = env_builder.build_env(args.env_config, args.engine_config, args.num_envs,
                                args.device, visualize=False)
    agent = agent_builder.build_agent(args.agent_config, env, args.device)
    agent.load(args.model_file)
    agent.eval()
    agent.set_mode(base_agent.AgentMode.TEST)
    return env, agent

def collect_diffs(env, agent, steps):
    """Normalized differentials Delta from a policy rollout."""
    obs, info = env.reset()
    diffs = []
    with torch.no_grad():
        for _ in range(steps):
            action, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(action)
            diff = info["disc_obs_demo"] - info["disc_obs"]
            diffs.append(agent._disc_obs_norm.normalize(diff).clone())
    return torch.cat(diffs, dim=0)

def scan(agent, diffs, num_alphas):
    model = agent._model
    scale = agent._disc_reward_scale
    alphas = np.linspace(0.0, 1.0, num_alphas)

    rows = []
    for alpha in alphas:
        x = (float(alpha) * diffs).detach().requires_grad_(True)
        logit = model.eval_disc(x).squeeze(-1)
        grad = torch.autograd.grad(logit.sum(), x)[0]

        reward = scale * torch.nn.functional.softplus(logit.detach())
        rows.append({
            "alpha": float(alpha),
            "logit_mean": float(logit.mean().item()),
            "logit_std": float(logit.std().item()),
            "reward_mean": float(reward.mean().item()),
            "grad_norm": float(torch.norm(grad, dim=-1).mean().item()),
        })

    for i, row in enumerate(rows):
        if (i == 0):
            row["dlogit_dalpha"] = float("nan")
        else:
            d_alpha = rows[i]["alpha"] - rows[i - 1]["alpha"]
            row["dlogit_dalpha"] = (rows[i]["logit_mean"] - rows[i - 1]["logit_mean"]) / d_alpha
    return rows

def block_shares(agent, env, diffs):
    """Share of |grad_Delta D| carried by each differential block at alpha=1."""
    x = diffs.detach().requires_grad_(True)
    logit = agent._model.eval_disc(x).squeeze(-1)
    grad = torch.autograd.grad(logit.sum(), x)[0]
    energy = torch.mean(torch.square(grad), dim=0)

    s = env.get_disc_state_obs_dim()
    d = env.get_liesig_tangent_dim()
    a = env.get_liesig_area_dim()

    blocks = {"state": energy[:s].sum(), "level1": energy[s:s + d].sum()}
    if (a > 0):
        blocks["level2"] = energy[s + d:].sum()

    total = sum(float(v.item()) for v in blocks.values())
    total = max(total, 1e-12)
    return {k: float(v.item()) / total for k, v in blocks.items()}

def main():
    args = parse_args()
    env, agent = build(args)

    diffs = collect_diffs(env, agent, args.steps)
    rows = scan(agent, diffs, args.num_alphas)
    shares = block_shares(agent, env, diffs)

    print("=" * 78)
    print("samples: {}   differential dim: {}".format(diffs.shape[0], diffs.shape[1]))
    print("{:>6} {:>12} {:>10} {:>12} {:>14} {:>12}".format(
        "alpha", "logit_mean", "logit_std", "reward_mean", "dlogit/dalpha", "|grad|"))
    for row in rows:
        print("{:>6.2f} {:>12.4f} {:>10.4f} {:>12.5f} {:>14.4f} {:>12.5f}".format(
            row["alpha"], row["logit_mean"], row["logit_std"], row["reward_mean"],
            row["dlogit_dalpha"], row["grad_norm"]))
    print("-" * 78)
    print("gradient energy share at alpha=1: " +
          "  ".join("{} {:.3f}".format(k, v) for k, v in shares.items()))
    print("=" * 78)

    if (args.out != ""):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(args.out,
                 alpha=np.array([r["alpha"] for r in rows]),
                 logit_mean=np.array([r["logit_mean"] for r in rows]),
                 logit_std=np.array([r["logit_std"] for r in rows]),
                 reward_mean=np.array([r["reward_mean"] for r in rows]),
                 grad_norm=np.array([r["grad_norm"] for r in rows]),
                 **{"share_" + k: np.array(v) for k, v in shares.items()})
        print("wrote", args.out)
    return

if __name__ == "__main__":
    main()
