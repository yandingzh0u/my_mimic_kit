"""Compare phase-weighting schemes on one frozen-policy rollout.

Schemes
  uniform : every transition weight 1
  bins8   : fixed 8-bin inverse-frequency weighting
  frame   : one bin per motion frame
  eb      : reliability-shrunk weighting (phase_balance.compute)

Reports weight statistics, Kish ESS and the empirical actor-gradient variance
across paired bootstrap resamples.  No training is performed.
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
DISCOUNT = 0.99


def bin_weights(phase, bins):
    index = torch.clamp((phase * bins).long(), 0, bins - 1)
    counts = torch.bincount(index, minlength=bins).to(torch.float64)
    present = counts > 0
    total = float(index.numel())
    weight = torch.zeros_like(counts)
    weight[present] = total / (float(present.sum()) * counts[present])
    return weight[index], counts


def weight_stats(weight):
    w = weight.flatten().to(torch.float64)
    ess = float(w.sum() ** 2 / torch.sum(w * w))
    total = float(w.numel())
    return {"mean": float(w.mean()), "max": float(w.max()),
            "p99": float(torch.quantile(w, 0.99)),
            "share_gt_5": float((w > 5).to(torch.float64).mean()),
            "share_gt_10": float((w > 10).to(torch.float64).mean()),
            "ess": ess, "ess_fraction": ess / total,
            "variance_inflation": total / ess}


@torch.no_grad()
def returns_to_go(reward, done):
    steps = reward.shape[0]
    out = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1], device=reward.device)
    for t in range(steps - 1, -1, -1):
        running = reward[t] + DISCOUNT * running * (1.0 - done[t])
        out[t] = running
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--grad-envs", type=int, default=256)
    parser.add_argument("--batch", type=int, default=2048)
    parser.add_argument("--resamples", type=int, default=8)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[400, 1400])
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "mimickit"))
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import learning.phase_balance as phase_balance
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util

    mp_util.init(0, 1, "cuda:0", None)
    util.set_rand_seed(0)
    env = env_builder.build_env(str(ARCHIVE / "env_config.yaml"),
                                str(ARCHIVE / "engine_config.yaml"),
                                args.num_envs, "cuda:0", visualize=False)
    agent = agent_builder.build_agent(str(ARCHIVE / "agent_config.yaml"), env, "cuda:0")
    frames = int(env._motion_lib._motion_num_frames.max().item()) - 1
    results = {"num_envs": args.num_envs, "steps": args.steps,
               "motion_frames": frames, "batch": args.batch,
               "resamples": args.resamples, "checkpoints": {}}
    try:
        for ckpt in args.checkpoints:
            agent.load(str(ARCHIVE / "int_models/model_{:010d}.pt".format(ckpt)))
            agent.eval()
            agent.set_mode(AgentMode.TRAIN)
            phase, mean_a, std_a, norm_a = [], [], [], []
            obs_pool, act_pool, rew, val, done_buf = [], [], [], [], []
            obs, _ = env.reset()
            with torch.no_grad():
                for _ in range(args.steps):
                    phase.append(env.get_motion_phase().detach())
                    dist = agent._model.eval_actor(agent._obs_norm.normalize(obs))
                    mean_a.append(dist.mean)
                    std_a.append(dist.stddev)
                    a_norm = dist.sample()
                    norm_a.append(a_norm)
                    action = agent._a_norm.unnormalize(a_norm)
                    obs_pool.append(obs[:args.grad_envs].cpu())
                    act_pool.append(action[:args.grad_envs].cpu())
                    val.append(agent._model.eval_critic(
                        agent._obs_norm.normalize(obs)).squeeze(-1))
                    obs, _, done, info = env.step(action)
                    raw = info["disc_obs_demo"] - info["disc_obs"]
                    rew.append(agent._calc_disc_rewards(
                        agent._disc_obs_norm.normalize(raw)))
                    done_buf.append((done != 0).to(torch.float32))
            phase = torch.cat([p.unsqueeze(0) for p in phase], 0)
            mean_a = torch.cat([m.unsqueeze(0) for m in mean_a], 0)
            std_a = torch.cat([s.unsqueeze(0) for s in std_a], 0)
            norm_a = torch.cat([a.unsqueeze(0) for a in norm_a], 0)
            rew = torch.cat([r.unsqueeze(0) for r in rew], 0)
            val = torch.cat([v.unsqueeze(0) for v in val], 0)
            done_t = torch.cat([d.unsqueeze(0) for d in done_buf], 0)
            adv = (returns_to_go(rew, done_t) - val).detach()
            score = adv.unsqueeze(-1) * (norm_a - mean_a) / std_a.square().clamp_min(1e-8)
            w_eb, info_eb = phase_balance.compute(phase, score, frames)
            w_uniform = torch.ones_like(phase)
            w8, _ = bin_weights(phase.reshape(-1), 8)
            wf, _ = bin_weights(phase.reshape(-1), frames)
            pool_phase = phase[:args.steps, :args.grad_envs].reshape(-1)
            pool = {"uniform": torch.ones_like(pool_phase),
                    "bins8": bin_weights(pool_phase, 8)[0],
                    "frame": bin_weights(pool_phase, frames)[0],
                    "eb": w_eb[:args.steps, :args.grad_envs].reshape(-1)}
            obs_pool = torch.cat(obs_pool).to("cuda:0")
            act_pool = torch.cat(act_pool).to("cuda:0")
            entry = {"weights": {
                "uniform": weight_stats(w_uniform),
                "bins8": weight_stats(w8),
                "frame": weight_stats(wf),
                "eb": weight_stats(w_eb)},
                "eb_alpha_mean": float(info_eb["phase_alpha_mean"]),
                "eb_tau2": float(info_eb["phase_tau2"]),
                "gradient": {}}
            for scheme, weight in pool.items():
                grads = []
                gen = torch.Generator(device="cpu").manual_seed(1234)
                for _ in range(args.resamples):
                    pick = torch.randint(0, obs_pool.shape[0], (args.batch,), generator=gen)
                    pick = pick.to("cuda:0")
                    dist = agent._model.eval_actor(
                        agent._obs_norm.normalize(obs_pool[pick]))
                    logp = dist.log_prob(agent._a_norm.normalize(act_pool[pick]))
                    w = weight[pick]
                    loss = -(w * logp).sum() / w.sum().clamp_min(1e-12)
                    agent._model.zero_grad(set_to_none=True)
                    loss.backward()
                    grads.append(torch.cat([p.grad.flatten() for p in
                                            agent._model.get_actor_params()
                                            if p.grad is not None]).double().cpu())
                stacked = torch.stack(grads)
                mean = stacked.mean(0)
                entry["gradient"][scheme] = {
                    "grad_variance": float(((stacked - mean) ** 2).sum(1).mean()),
                    "grad_norm_mean": float(mean.norm())}
                agent._model.zero_grad(set_to_none=True)
            base = entry["gradient"]["uniform"]["grad_variance"]
            for scheme in ("bins8", "frame", "eb"):
                entry["gradient"][scheme]["variance_ratio_vs_uniform"] = (
                    entry["gradient"][scheme]["grad_variance"] / max(base, 1e-30))
            entry["gradient"]["variance_ratio_frame_over_bins8"] = (
                entry["gradient"]["frame"]["grad_variance"]
                / max(entry["gradient"]["bins8"]["grad_variance"], 1e-30))
            entry["gradient"]["variance_ratio_eb_over_bins8"] = (
                entry["gradient"]["eb"]["grad_variance"]
                / max(entry["gradient"]["bins8"]["grad_variance"], 1e-30))
            results["checkpoints"][str(ckpt)] = entry
            print(json.dumps({
                "checkpoint": ckpt,
                "weights": {k: {"max": round(v["max"], 2),
                                "ess": round(v["ess_fraction"], 4)}
                            for k, v in entry["weights"].items()},
                "grad_var": {k: round(v["grad_variance"], 3)
                             for k, v in entry["gradient"].items()
                             if isinstance(v, dict)},
                "eb_over_bins8": round(
                    entry["gradient"]["variance_ratio_eb_over_bins8"], 3),
                "frame_over_bins8": round(
                    entry["gradient"]["variance_ratio_frame_over_bins8"], 3)}),
                flush=True)
        (out / "summary.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"complete": True}), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env._engine._app_launcher.app.close()


if __name__ == "__main__":
    main()
