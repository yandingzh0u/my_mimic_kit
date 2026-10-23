"""Compare uniform PPO minibatches with LaBER priority sampling.

Frozen policy, one rollout, paired bootstrap resamples.  For every resample we
build the real PPO clipped surrogate gradient of the actor under

  uniform : without-replacement minibatch, phase-weighted mean
  laber   : with-replacement priority sample, importance correction

and report gradient variance, bias against the uniform mean and the tail of
the importance weights.  No training is performed.
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


@torch.no_grad()
def returns_to_go(reward, done):
    out = torch.zeros_like(reward)
    running = torch.zeros(reward.shape[1], device=reward.device)
    for t in range(reward.shape[0] - 1, -1, -1):
        running = reward[t] + DISCOUNT * running * (1.0 - done[t])
        out[t] = running
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=2048)
    parser.add_argument("--steps", type=int, default=32)
    parser.add_argument("--batch", type=int, default=8192)
    parser.add_argument("--resamples", type=int, default=12)
    parser.add_argument("--checkpoint", type=int, default=400)
    parser.add_argument("--agent-config", type=str,
                        default="data/agents/dare_10layer_cpl_climb_laber_agent.yaml")
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
    env = env_builder.build_env(str(ARCHIVE / "env_config.yaml"),
                                str(ARCHIVE / "engine_config.yaml"),
                                args.num_envs, "cuda:0", visualize=False)
    agent = agent_builder.build_agent(args.agent_config, env, "cuda:0")
    agent.load(str(ARCHIVE / "int_models/model_{:010d}.pt".format(args.checkpoint)))
    agent.eval()
    agent.set_mode(AgentMode.TRAIN)

    obs_list, act_list, logp_list, phase_list = [], [], [], []
    rew_list, val_list, done_list = [], [], []
    obs, _ = env.reset()
    with torch.no_grad():
        for _ in range(args.steps):
            phase_list.append(env.get_motion_phase().detach())
            dist = agent._model.eval_actor(agent._obs_norm.normalize(obs))
            a_norm = dist.sample()
            logp_list.append(dist.log_prob(a_norm).detach())
            action = agent._a_norm.unnormalize(a_norm)
            obs_list.append(obs)
            act_list.append(action)
            val_list.append(agent._model.eval_critic(
                agent._obs_norm.normalize(obs)).squeeze(-1))
            obs, _, done, info = env.step(action)
            raw = info["disc_obs_demo"] - info["disc_obs"]
            rew_list.append(agent._calc_disc_rewards(
                agent._disc_obs_norm.normalize(raw)))
            done_list.append(done.clone())
    obs = torch.stack(obs_list)
    action = torch.stack(act_list)
    old_logp = torch.stack(logp_list)
    phase = torch.stack(phase_list)
    reward = torch.stack(rew_list)
    value = torch.stack(val_list)
    done = torch.stack(done_list)

    adv = (returns_to_go(reward, (done != 0).float()) - value).detach()
    ones = torch.ones_like(adv)
    unit_score = agent._build_phase_score(obs, action, ones)
    phase_weight, _ = agent._build_phase_weights(unit_score * adv.unsqueeze(-1))
    mask = torch.ones_like(adv, dtype=torch.bool).flatten()
    adv_flat = adv.flatten()
    adv_mean, adv_std = agent._weighted_mean_std(
        adv_flat[mask], phase_weight.flatten()[mask])
    norm_adv = torch.clamp((adv - adv_mean) / adv_std.clamp_min(1e-5),
                           -agent._norm_adv_clip, agent._norm_adv_clip)
    laber = agent._build_laber_sampling(
        done, torch.ones_like(done).float(), norm_adv, phase_weight, unit_score)

    probs = laber["probabilities"]
    correction = laber["correction"]
    target = phase_weight.flatten().float()
    target = target / target.sum().clamp_min(1e-12)
    total = probs.numel()
    w = target / probs.clamp_min(1e-12)
    ess = float(1.0 / (target * w).sum())
    corr = correction[correction > 0]

    obs_f = obs.reshape(-1, obs.shape[-1])
    act_f = agent._a_norm.normalize(action).reshape(-1, action.shape[-1])
    logp_f = old_logp.reshape(-1)
    adv_f = norm_adv.reshape(-1)
    pw_f = phase_weight.reshape(-1).float()

    def grad_of(idx, kind):
        dist = agent._model.eval_actor(agent._obs_norm.normalize(obs_f[idx]))
        logp = dist.log_prob(act_f[idx])
        ratio = torch.exp(logp - logp_f[idx])
        obj = torch.minimum(adv_f[idx] * ratio,
                            adv_f[idx] * torch.clamp(
                                ratio, 1.0 - agent._ppo_clip_ratio,
                                1.0 + agent._ppo_clip_ratio))
        if kind == "full":
            loss = -(pw_f[idx] * obj).sum() / total
        elif kind == "uniform":
            loss = -(pw_f[idx] * obj).mean()
        elif kind == "uniform_repl":
            loss = -(pw_f[idx] * obj).mean()
        else:
            loss = -(correction[idx] * obj).mean()
        agent._model.zero_grad(set_to_none=True)
        loss.backward()
        g = torch.cat([p.grad.flatten() for p in agent._model.get_actor_params()
                       if p.grad is not None]).double().cpu()
        agent._model.zero_grad(set_to_none=True)
        return g

    all_idx = torch.arange(total, device="cuda:0")
    full_grad = grad_of(all_idx, "full")
    grads = {"uniform": [], "uniform_repl": [], "laber": []}
    for r in range(args.resamples):
        gen = torch.Generator(device="cpu").manual_seed(900 + r)
        perm = torch.randperm(total, generator=gen)[:args.batch].to("cuda:0")
        grads["uniform"].append(grad_of(perm, "uniform"))
        gen_d = torch.Generator(device=probs.device).manual_seed(900 + r)
        draw = torch.multinomial(torch.full_like(probs, 1.0 / total), args.batch,
                                 replacement=True, generator=gen_d)
        grads["uniform_repl"].append(grad_of(draw, "uniform_repl"))
        cand = torch.multinomial(probs, args.batch, replacement=True,
                                 generator=torch.Generator(device=probs.device).manual_seed(900 + r))
        grads["laber"].append(grad_of(cand, "laber"))

    summary = {"checkpoint": args.checkpoint, "batch": args.batch,
               "resamples": args.resamples, "num_envs": args.num_envs,
               "segments": laber["segments"],
               "proposal_ess_fraction": laber["is_ess_fraction"].item(),
               "correction_max": float(corr.max()),
               "correction_p99": float(torch.quantile(corr, 0.99)),
               "correction_mean": float(corr.mean()),
               "weights": {}}
    means = {}
    for kind in ("uniform", "uniform_repl", "laber"):
        g = torch.stack(grads[kind])
        mean = g.mean(0)
        means[kind] = mean
        var = ((g - mean) ** 2).sum(1).mean()
        summary["weights"][kind] = {
            "grad_variance": float(var),
            "grad_norm_mean": float(mean.norm()),
            "relative_std": float(torch.sqrt(var) / mean.norm().clamp_min(1e-12))}
    base = summary["weights"]["uniform_repl"]["grad_variance"]
    summary["variance_ratio_laber_over_uniform"] = (
        summary["weights"]["laber"]["grad_variance"] / max(base, 1e-30))
    for kind in ("uniform", "uniform_repl", "laber"):
        diff = (means[kind] - full_grad).norm()
        summary["weights"][kind]["bias_vs_full"] = float(
            diff / full_grad.norm().clamp_min(1e-12))
        summary["weights"][kind]["cosine_vs_full"] = float(
            torch.nn.functional.cosine_similarity(
                means[kind].unsqueeze(0), full_grad.unsqueeze(0)).item())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    env._engine._app_launcher.app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
