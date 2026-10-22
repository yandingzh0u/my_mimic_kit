"""Compare 8-bin and per-frame phase reweighting on a frozen rollout.

No training. One rollout per checkpoint, then:
  * sample-weight statistics (mean / max / p99 / share above 5, 10, 50)
  * Kish ESS and the implied gradient-variance inflation N / ESS
  * empirical actor-gradient variance across paired bootstrap resamples

Schemes
  uniform : every transition weight 1
  bins8   : the current phasebalanced scheme (`_phase_balance_bins = 8`)
  frame   : one bin per motion frame (climb -> 299 bins), allocated from the
            motion length instead of a fixed constant
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = (ROOT / "output/paper_benchmark/"
           "dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012")


def bin_weights(phase, bins):
    """Inverse-frequency weights so occupied bins contribute equally."""
    index = torch.clamp((phase * bins).long(), 0, bins - 1)
    counts = torch.bincount(index, minlength=bins).to(torch.float64)
    present = counts > 0
    total = float(index.numel())
    weight = torch.zeros_like(counts)
    weight[present] = total / (float(present.sum()) * counts[present])
    return weight[index], counts


def weight_stats(weight):
    w = weight.to(torch.float64)
    ess = float(w.sum() ** 2 / torch.sum(w * w))
    total = float(w.numel())
    return {
        "mean": float(w.mean()),
        "max": float(w.max()),
        "p99": float(torch.quantile(w, 0.99)),
        "cv": float(w.std(unbiased=False) / w.mean()),
        "share_gt_5": float((w > 5).to(torch.float64).mean()),
        "share_gt_10": float((w > 10).to(torch.float64).mean()),
        "share_gt_50": float((w > 50).to(torch.float64).mean()),
        "ess": ess,
        "ess_fraction": ess / total,
        "variance_inflation": total / ess,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--grad-envs", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--resamples", type=int, default=16)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[400, 900, 1400])
    args = parser.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT / "mimickit"))
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util

    mp_util.init(0, 1, "cuda:0", None)
    util.set_rand_seed(0)
    print(json.dumps({"stage": "build_env"}), flush=True)
    env = env_builder.build_env(str(ARCHIVE / "env_config.yaml"),
                                str(ARCHIVE / "engine_config.yaml"),
                                args.num_envs, "cuda:0", visualize=False)
    agent = agent_builder.build_agent(str(ARCHIVE / "agent_config.yaml"), env, "cuda:0")
    print(json.dumps({"stage": "build_agent"}), flush=True)
    agent.eval()
    agent.set_mode(AgentMode.TRAIN)
    frames = int(env._motion_lib._motion_num_frames[0].item()) - 1
    device = "cuda:0"
    results = {"num_envs": args.num_envs, "steps": args.steps,
               "motion_frames": frames, "batch": args.batch,
               "resamples": args.resamples, "grad_envs": args.grad_envs,
               "checkpoints": {}}
    try:
        for ckpt in args.checkpoints:
            print(json.dumps({"stage": "load", "ckpt": ckpt}), flush=True)
            agent.load(str(ARCHIVE / "int_models/model_{:010d}.pt".format(ckpt)))
            agent.eval()
            agent.set_mode(AgentMode.TRAIN)
            phase = []
            phase_pool = []
            obs_pool = []
            act_pool = []
            obs, _ = env.reset()
            with torch.no_grad():
                for _ in range(args.steps):
                    current = env.get_motion_phase().detach().cpu()
                    phase.append(current)
                    phase_pool.append(current[:args.grad_envs])
                    dist = agent._model.eval_actor(agent._obs_norm.normalize(obs))
                    norm_a = dist.sample()
                    obs_pool.append(obs[:args.grad_envs].detach().cpu())
                    act_pool.append(agent._a_norm.unnormalize(norm_a)[:args.grad_envs].cpu())
                    obs, _, done, _ = env.step(agent._a_norm.unnormalize(norm_a))
                    ids = (done != 0).nonzero().flatten()
                    obs, _ = env.reset(ids)
            print(json.dumps({"stage": "rollout_done", "ckpt": ckpt}), flush=True)
            phase = torch.cat(phase).flatten()
            phase_pool = torch.cat(phase_pool).flatten()
            obs_pool = torch.cat(obs_pool).to(device)
            act_pool = torch.cat(act_pool).to(device)
            w_uniform = torch.ones_like(phase, dtype=torch.float64)
            w8, counts8 = bin_weights(phase, 8)
            wf, countsf = bin_weights(phase, frames)
            pu8, _ = bin_weights(phase_pool, 8)
            puf, _ = bin_weights(phase_pool, frames)
            pool_w = {
                "uniform": torch.ones_like(phase_pool, dtype=torch.float64).to(device),
                "bins8": pu8.to(device),
                "frame": puf.to(device),
            }
            entry = {
                "occupied_bins": {"bins8": int((counts8 > 0).sum()),
                                  "frame": int((countsf > 0).sum())},
                "min_bin_count": {"bins8": int(counts8[counts8 > 0].min()),
                                  "frame": int(countsf[countsf > 0].min())},
                "median_bin_count": {"bins8": float(counts8[counts8 > 0].median()),
                                     "frame": float(countsf[countsf > 0].median())},
                "weights": {
                    "uniform": weight_stats(w_uniform),
                    "bins8": weight_stats(w8),
                    "frame": weight_stats(wf)},
                "gradient": {},
            }
            print(json.dumps({"stage": "weights_done", "ckpt": ckpt}), flush=True)
            for scheme, weight in pool_w.items():
                grads = []
                gen = torch.Generator(device="cpu").manual_seed(1234)
                for _ in range(args.resamples):
                    pick = torch.randint(0, obs_pool.shape[0], (args.batch,), generator=gen)
                    pick = pick.to(device)
                    dist = agent._model.eval_actor(agent._obs_norm.normalize(obs_pool[pick]))
                    logp = dist.log_prob(agent._a_norm.normalize(act_pool[pick]))
                    w = weight[pick]
                    loss = -(w * logp).sum() / w.sum().clamp_min(1e-12)
                    agent._model.zero_grad(set_to_none=True)
                    loss.backward()
                    grads.append(torch.cat([p.grad.flatten() for p in
                                            agent._model.get_actor_params()
                                            if p.grad is not None]).double().cpu())
                print(json.dumps({"stage": "grads_done", "scheme": scheme,
                                  "n": len(grads), "p": int(grads[0].numel())}), flush=True)
                stacked = torch.stack(grads)
                print(json.dumps({"stage": "stack_done", "scheme": scheme}), flush=True)
                mean = stacked.mean(0)
                var = ((stacked - mean) ** 2).sum(1).mean()
                entry["gradient"][scheme] = {
                    "grad_norm_mean": float(mean.norm()),
                    "grad_variance": float(var),
                    "relative_std": float(torch.sqrt(var) / mean.norm().clamp_min(1e-12)),
                }
                agent._model.zero_grad(set_to_none=True)
            print(json.dumps({"stage": "schemes_done", "ckpt": ckpt}), flush=True)
            base = entry["gradient"]["uniform"]["grad_variance"]
            for scheme in ("bins8", "frame"):
                entry["gradient"][scheme]["variance_ratio_vs_uniform"] = (
                    entry["gradient"][scheme]["grad_variance"] / max(base, 1e-30))
            entry["gradient"]["variance_ratio_frame_over_bins8"] = (
                entry["gradient"]["frame"]["grad_variance"]
                / max(entry["gradient"]["bins8"]["grad_variance"], 1e-30))
            results["checkpoints"][str(ckpt)] = entry
            print(json.dumps({"checkpoint": ckpt,
                              "weights": {k: {"max": round(v["max"], 2),
                                              "ess": round(v["ess_fraction"], 4),
                                              "infl": round(v["variance_inflation"], 2)}
                                          for k, v in entry["weights"].items()},
                              "grad_var": {k: round(v["grad_variance"], 6)
                                           for k, v in entry["gradient"].items()
                                           if isinstance(v, dict)},
                              "frame_over_bins8": round(
                                  entry["gradient"]["variance_ratio_frame_over_bins8"], 3)}),
                             flush=True)
        (out / "summary.json").write_text(json.dumps(results, indent=2))
        print(json.dumps({"complete": True, "out": str(out / "summary.json")}), flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        raise
    finally:
        env._engine._app_launcher.app.close()


if __name__ == "__main__":
    main()
