"""Offline probe of the CPMD discriminator's geometry along the ray to zero.

A discriminator can win so hard that the policy region sits on a flat,
saturated shell where softplus(z) delivers almost no reward gradient. That
failure is invisible in the loss and obvious in the geometry, so this tool
makes it directly observable for any ADD-style checkpoint, without training
anything.

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

Also reports how much the learned metric allocation changes across reference
contexts.  The current CPMD differential is the original 172-D ADD state
differential; there are no history or interaction blocks.

Example:
    python tools/cpmd/probe_cpmd_disc_geometry.py \
        --env_config output/cpmd_roll_cycle_1k_seed0/env_config.yaml \
        --engine_config output/cpmd_roll_cycle_1k_seed0/engine_config.yaml \
        --agent_config output/cpmd_roll_cycle_1k_seed0/agent_config.yaml \
        --model_file output/cpmd_roll_cycle_1k_seed0/model.pt \
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
    parser = argparse.ArgumentParser(description="CPMD discriminator geometry probe")
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

def collect_inputs(env, agent, steps):
    """Normalized differentials and matched contexts from a rollout."""
    obs, info = env.reset()
    diffs = []
    contexts = []
    with torch.no_grad():
        for _ in range(steps):
            action, _ = agent._decide_action(obs, info)
            obs, r, done, info = env.step(action)
            diff = info["disc_obs_demo"] - info["disc_obs"]
            diffs.append(agent._disc_obs_norm.normalize(diff).clone())
            contexts.append(
                agent._context_norm.normalize(info["cpmd_context"]).clone())
    return torch.cat(diffs, dim=0), torch.cat(contexts, dim=0)

def scan(agent, diffs, contexts, num_alphas):
    model = agent._model
    scale = agent._disc_reward_scale
    alphas = np.linspace(0.0, 1.0, num_alphas)

    rows = []
    for alpha in alphas:
        x = (float(alpha) * diffs).detach().requires_grad_(True)
        logit = model.eval_disc(x, contexts).squeeze(-1)
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

def metric_summary(agent, diffs, contexts):
    """Summarize the trace-one allocation at the policy differential."""
    x = diffs.detach().requires_grad_(True)
    terms = agent._model.eval_metric_terms(x, contexts)
    logit = terms["logit"].squeeze(-1)
    grad = torch.autograd.grad(logit.sum(), x)[0]
    diag = terms["metric_diag"].detach()
    return {
        "trace_mean": float(terms["trace"].mean().item()),
        "diag_min": float(diag.mean(dim=0).min().item()),
        "diag_max": float(diag.mean(dim=0).max().item()),
        "diag_context_std": float(
            diag.std(dim=0, unbiased=False).mean().item()),
        "grad_norm": float(torch.linalg.vector_norm(grad, dim=-1).mean().item()),
    }

def main():
    args = parse_args()
    env, agent = build(args)

    diffs, contexts = collect_inputs(env, agent, args.steps)
    rows = scan(agent, diffs, contexts, args.num_alphas)
    summary = metric_summary(agent, diffs, contexts)

    print("=" * 78)
    print("samples: {}   differential dim: {}".format(diffs.shape[0], diffs.shape[1]))
    print("{:>6} {:>12} {:>10} {:>12} {:>14} {:>12}".format(
        "alpha", "logit_mean", "logit_std", "reward_mean", "dlogit/dalpha", "|grad|"))
    for row in rows:
        print("{:>6.2f} {:>12.4f} {:>10.4f} {:>12.5f} {:>14.4f} {:>12.5f}".format(
            row["alpha"], row["logit_mean"], row["logit_std"], row["reward_mean"],
            row["dlogit_dalpha"], row["grad_norm"]))
    print("-" * 78)
    print("metric at alpha=1: " +
          "  ".join("{} {:.5f}".format(k, v) for k, v in summary.items()))
    print("=" * 78)

    if (args.out != ""):
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        np.savez(args.out,
                 alpha=np.array([r["alpha"] for r in rows]),
                 logit_mean=np.array([r["logit_mean"] for r in rows]),
                 logit_std=np.array([r["logit_std"] for r in rows]),
                 reward_mean=np.array([r["reward_mean"] for r in rows]),
                 grad_norm=np.array([r["grad_norm"] for r in rows]),
                 **{"metric_" + k: np.array(v) for k, v in summary.items()})
        print("wrote", args.out)
    return

if __name__ == "__main__":
    main()
