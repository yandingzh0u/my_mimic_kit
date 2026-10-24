"""Check whether the DARE discriminator actually rewards moving toward zero.

For a frozen checkpoint the script collects real residuals e = demo - policy
from a rollout and evaluates the discriminator at 0, 0.5 e and e.  A reward
that teaches correction must satisfy f(0) > f(0.5 e) > f(e); any violation is
counted per phase bin.  Only diagnostics, nothing is trained.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (ROOT / "output/paper_benchmark/"
           "dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkpoint", type=str,
                        default=str(ARCHIVE / "int_models/model_0000002999.pt"))
    parser.add_argument("--agent-config", type=str,
                        default=str(ARCHIVE / "agent_config.yaml"))
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--bins", type=int, default=8)
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "mimickit"))
    sys.path.insert(0, str(ROOT))

    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util

    mp_util.init(0, 1, "cuda:0", None)
    util.set_rand_seed(0)
    env = env_builder.build_env(str(ARCHIVE / "env_config.yaml"),
                                str(ARCHIVE / "engine_config.yaml"),
                                args.num_envs, "cuda:0", visualize=False)
    agent = agent_builder.build_agent(args.agent_config, env, "cuda:0")
    agent.load(args.checkpoint)
    agent.eval()
    agent.set_mode(AgentMode.TRAIN)

    residuals, phases, rewards = [], [], []
    obs, _ = env.reset()
    with torch.no_grad():
        for _ in range(args.steps):
            phases.append(env.get_motion_phase().detach())
            dist = agent._model.eval_actor(agent._obs_norm.normalize(obs))
            a_norm = dist.sample()
            action = agent._a_norm.unnormalize(a_norm)
            obs, _, done, info = env.step(action)
            raw = info["disc_obs_demo"] - info["disc_obs"]
            residuals.append(agent._disc_obs_norm.normalize(raw).detach())
            rewards.append(agent._calc_disc_rewards(
                agent._disc_obs_norm.normalize(raw)).detach())
    e = torch.cat([r.reshape(-1, r.shape[-1]) for r in residuals], dim=0)
    phase = torch.cat([p.reshape(-1) for p in phases], dim=0)
    reward = torch.cat([r.reshape(-1) for r in rewards], dim=0)

    with torch.no_grad():
        f0 = agent._model.eval_disc_raw(torch.zeros_like(e)).squeeze(-1)
        fh = agent._model.eval_disc_raw(0.5 * e).squeeze(-1)
        f1 = agent._model.eval_disc_raw(e).squeeze(-1)
        fneg = agent._model.eval_disc_raw(-e).squeeze(-1)

    zero_violation = (f1 > f0).float()
    direction_violation = (f1 > fh).float()
    # The reviewer's counterexample: overshooting past zero must not be better.
    overshoot_violation = (fneg > f0).float()
    bins = torch.clamp((phase * args.bins).long(), 0, args.bins - 1)
    summary = {
        "checkpoint": args.checkpoint,
        "num_samples": int(e.numel() // e.shape[-1]),
        "mean_abs_residual": float(e.abs().mean()),
        "mean_logit_zero": float(f0.mean()),
        "mean_logit_half": float(fh.mean()),
        "mean_logit_full": float(f1.mean()),
        "zero_violation_rate": float(zero_violation.mean()),
        "direction_violation_rate": float(direction_violation.mean()),
        "overshoot_violation_rate": float(overshoot_violation.mean()),
        "mean_logit_neg": float(fneg.mean()),
        "reward_mean": float(reward.mean()),
        "per_phase": {},
    }
    for b in range(args.bins):
        mask = bins == b
        if int(mask.sum()) == 0:
            continue
        summary["per_phase"][str(b)] = {
            "n": int(mask.sum()),
            "zero_violation": float(zero_violation[mask].mean()),
            "direction_violation": float(direction_violation[mask].mean()),
            "overshoot_violation": float(overshoot_violation[mask].mean()),
            "logit_zero": float(f0[mask].mean()),
            "logit_full": float(f1[mask].mean()),
        }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    env._engine._app_launcher.app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
