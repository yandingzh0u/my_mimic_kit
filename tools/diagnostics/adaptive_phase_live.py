"""Frozen-parameter PPO gradient audit on fresh Isaac Lab rollouts.

Uses real DARE rewards, terminal-aware TD-lambda returns, actor eligibility,
advantage clipping and the repository's entire actor loss.  No optimizer
updates or training-performance measurements are made.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
import types

import numpy as np
import torch

from adaptive_phase_candidate import AdaptivePhaseSampler, CapacityAwarePhaseSampler, PhaseCoverageRotor, control_cells

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/"mimickit"))
ARCHIVE = ROOT/"output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012"


def gradient(agent, tensors, weights, adv, indices, chunk=2048):
    params = [p for p in agent._model.get_actor_params() if p.requires_grad]
    agent._model.zero_grad(set_to_none=True)
    # Each chunk is normalized by _compute_actor_loss, so multiply by its
    # mass to recover the exact gradient of the full weighted objective.
    total = weights.sum()
    for start in range(0, len(indices), chunk):
        idx = indices[start:start+chunk]
        w = weights[start:start+chunk].to(torch.float32)
        batch = {key: tensor[idx] for key, tensor in tensors.items()}
        batch["adv"] = adv[idx]
        batch["phase_weight"] = w
        info = agent._compute_actor_loss(batch)
        (info["actor_loss"] * (w.sum()/total)).backward()
    result = torch.cat([p.grad.detach().flatten() for p in params])
    agent._model.zero_grad(set_to_none=True)
    return result


def audit_gradients(agent, frames, repeats, seed, include_capacity=False):
    phase = agent._exp_buffer.get_data_flat("diag_phase").detach().clone()
    eligible = agent._exp_buffer.get_data_flat("rand_action_mask") == 1
    tensors = {key: agent._exp_buffer.get_data_flat(key).detach()
               for key in ["obs", "action", "a_logp", "rand_action_mask"]}
    with torch.no_grad():
        obs = agent._obs_norm.normalize(tensors["obs"])
        raw_adv = (agent._exp_buffer.get_data_flat("tar_val")
                   - agent._model.eval_critic(obs).squeeze(-1))
    size = 4*agent.get_num_envs()
    budget = 2*agent.get_num_envs()
    gen = torch.Generator(device=phase.device).manual_seed(seed)
    output = dict(actor_batch=size, critic_batch_budget=budget, methods={})
    exacts = {}
    definitions = {}
    methods = [("historical_8_wr", 8, True), ("balanced_8", 8, False),
               ("adaptive_balanced", frames, False)]
    if include_capacity:
        methods.append(("adaptive_capacity", frames, False))
    for name, resolution, replacement in methods:
        cls = CapacityAwarePhaseSampler if name == "adaptive_capacity" else AdaptivePhaseSampler
        sampler = cls(phase, frame_count=resolution, batch_budget=budget, eligible=eligible)
        if include_capacity and name in ["adaptive_balanced", "adaptive_capacity"]:
            # Same global permutations and output shuffles for the paired
            # same-target comparison. Neither method updates any parameters.
            gen.manual_seed(seed)
        full_weights = sampler.full_weights()
        with torch.no_grad():
            mean = (full_weights * raw_adv).sum()
            std = ((full_weights * raw_adv.square()).sum() - mean.square()).clamp_min(0).sqrt()
            adv = ((raw_adv-mean)/std.clamp_min(1e-5)).clamp(-agent._norm_adv_clip, agent._norm_adv_clip).float()
        idx_all = sampler.indices
        if name == "adaptive_capacity":
            old_w, old_adv = definitions["adaptive_balanced"]
            assert torch.equal(full_weights, old_w)
            assert torch.equal(adv, old_adv)
            exact = exacts["adaptive_balanced"]
        else:
            exact = gradient(agent, tensors, full_weights[idx_all], adv, idx_all).double()
        exacts[name] = exact
        definitions[name] = (full_weights, adv)
        pools = [(sampler.bin_ids == b).logical_and(eligible).nonzero().flatten()
                 for b in sampler.present_bins.tolist()] if replacement else None
        summed = torch.zeros_like(exact)
        sum_sq = torch.zeros((), device=exact.device, dtype=torch.float64)
        min_unique = size
        errors = []
        for _ in range(repeats):
            idx, weight, quota = sampler.sample(size, generator=gen)
            if replacement:
                idx = torch.cat([pool[torch.randint(len(pool), (int(q),),
                                                   device=phase.device, generator=gen)]
                                 for pool, q in zip(pools, quota.tolist())])
                weight = torch.cat([torch.full((int(q),), 1/(sampler.num_present*int(q)),
                                                device=phase.device, dtype=torch.float64)
                                    for q in quota.tolist()])
            min_unique = min(min_unique, len(idx.unique()))
            if name == "adaptive_capacity":
                assert len(idx.unique()) == min(size, sampler.indices.numel())
                assert bool((quota <= sampler.counts).all())
            g = gradient(agent, tensors, weight, adv, idx).double()
            summed += g
            sum_sq += g.square().sum()
            errors.append(float((g-exact).square().sum()))
        mean_g = summed/repeats
        variance = (sum_sq - summed.square().sum()/repeats)/(repeats-1)
        mse = sum_sq/repeats - 2*(mean_g*exact).sum() + exact.square().sum()
        output["methods"][name] = dict(coverage=sampler.coverage,
            smallest_pool=int(sampler.counts.min()), largest_pool=int(sampler.counts.max()),
            minimum_unique_in_batch=min_unique, gradient_variance=float(variance),
            mse_against_own_full_target=float(mse), full_gradient_norm=float(exact.norm()),
            gradient_mse_mean=float(np.mean(errors)),
            gradient_mse_standard_error=float(np.std(errors, ddof=1)/np.sqrt(repeats)),
            gradient_squared_error_samples=errors,
            mean_gradient_error_norm=float((mean_g-exact).norm()),
            advantage_mean=float(mean), advantage_std=float(std))
        print(json.dumps(dict(method=name, variance=float(variance),
                              present=sampler.num_present, min_pool=int(sampler.counts.min()))), flush=True)
    baseline = exacts["historical_8_wr"]
    candidate = exacts["adaptive_balanced"]
    output["adaptive_full_target_cosine_vs_8"] = float(torch.dot(candidate, baseline)/(candidate.norm()*baseline.norm()))
    output["adaptive_full_target_relative_change_vs_8"] = float((candidate-baseline).norm()/baseline.norm())
    basevar = output["methods"]["historical_8_wr"]["gradient_variance"]
    for data in output["methods"].values():
        data["variance_ratio_vs_8_wr"] = data["gradient_variance"]/basevar
    rng = np.random.default_rng(seed)
    base_errors = np.array(output["methods"]["historical_8_wr"]["gradient_squared_error_samples"])
    for data in output["methods"].values():
        errors = np.array(data["gradient_squared_error_samples"])
        ratios = []
        for _ in range(2000):
            ratios.append(float(rng.choice(errors, len(errors), replace=True).mean()
                                / rng.choice(base_errors, len(base_errors), replace=True).mean()))
        data["mse_ratio_bootstrap_95ci_vs_8_wr"] = np.quantile(ratios, [.025, .975]).tolist()
    if include_capacity:
        old = np.array(output["methods"]["adaptive_balanced"]["gradient_squared_error_samples"])
        new = np.array(output["methods"]["adaptive_capacity"]["gradient_squared_error_samples"])
        bootstrap_delta, bootstrap_ratio = [], []
        for _ in range(5000):
            pick = rng.integers(repeats, size=repeats)
            bootstrap_delta.append(float((new[pick]-old[pick]).mean()))
            bootstrap_ratio.append(float(new[pick].mean()/old[pick].mean()))
        output["capacity_same_target"] = dict(
            full_target_difference=0., advantage_normalization_identical=True,
            paired_permutations=True, mse_ratio=float(new.mean()/old.mean()),
            mse_difference=float((new-old).mean()),
            paired_bootstrap_95ci_ratio=np.quantile(bootstrap_ratio, [.025, .975]).tolist(),
            paired_bootstrap_95ci_difference=np.quantile(bootstrap_delta, [.025, .975]).tolist())
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=128)
    parser.add_argument("--checkpoints", type=int, nargs="+", default=[0, 1600, 2999])
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--capacity-audit", action="store_true")
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util

    torch.set_num_threads(4)
    mp_util.init(0, 1, "cuda:0", 6393)
    util.set_rand_seed(args.seed)
    env = env_builder.build_env(str(ARCHIVE/"env_config.yaml"),
                                str(ARCHIVE/"engine_config.yaml"),
                                args.num_envs, "cuda:0", visualize=False)
    agent = agent_builder.build_agent(str(ARCHIVE/"agent_config.yaml"), env, "cuda:0")
    assert env._motion_lib.get_num_motions() == 1, "this audit is explicitly for one motion"
    duration = (int(env._motion_lib._motion_num_frames[0])-1)/float(env._motion_lib._motion_fps[0])
    frames = control_cells(duration, env._engine.get_timestep())
    original_reset = env._sample_motion_times
    report = dict(scope="frozen-policy gradient sampling; no training and no tracking-error improvement claim",
                  seed=args.seed, num_envs=args.num_envs, steps=32, resamples=args.repeats,
                  control_cells=frames, duration=duration, warmup_steps=frames,
                  capacity_audit=args.capacity_audit, checkpoints={})
    try:
        for checkpoint in args.checkpoints:
            util.set_rand_seed(args.seed)
            agent._init_train()
            agent.load(str(ARCHIVE/f"int_models/model_{checkpoint:010d}.pt"))
            agent._sample_count = int(1e12)  # Freeze all checkpoint normalizers.
            agent.eval()
            agent.set_mode(AgentMode.TRAIN)
            env._sample_motion_times = original_reset
            agent._curr_obs, agent._curr_info = env.reset()
            reset_phase = env.get_motion_phase().detach().clone()
            with torch.no_grad():
                # Let every initial episode finish at least once. This is a
                # continuous-rollout probe, not a freshly uniform RSI sample.
                agent._rollout_train(frames)
                agent._init_iter()
                agent._rollout_train(32)
                agent._build_train_data()
            result = audit_gradients(agent, frames, args.repeats, args.seed, args.capacity_audit)
            result["iid_initial_missing_cells"] = frames - len((reset_phase.double()*frames).floor().long().clamp_max(frames-1).unique())
            # Separate rollout: does the coverage reset survive the real env's
            # float32 time conversion and reference initialization?
            rotor = PhaseCoverageRotor(frames, seed=args.seed, device="cuda:0")
            def covered_reset(self, count):
                ids, times = original_reset(count)
                times, _ = rotor.draw_times(self._motion_lib.get_motion_length(ids))
                return ids, times
            env._sample_motion_times = types.MethodType(covered_reset, env)
            agent._init_iter()
            agent._curr_obs, agent._curr_info = env.reset()
            initial_phase = env.get_motion_phase().detach().clone()
            with torch.no_grad():
                agent._rollout_train(32)
            p = agent._exp_buffer.get_data_flat("diag_phase")
            probe = AdaptivePhaseSampler(p, frame_count=frames, batch_budget=2*args.num_envs)
            result["coverage_reset"] = dict(initial_missing_cells=frames-len((initial_phase.double()*frames).floor().long().clamp_max(frames-1).unique()),
                                             rollout_coverage=probe.coverage)
            assert result["coverage_reset"]["initial_missing_cells"] == 0
            assert probe.coverage["missing_control_cells"] == 0
            report["checkpoints"][str(checkpoint)] = result
            (args.out/"live.json").write_text(json.dumps(report, indent=2)+"\n")
            print(json.dumps(dict(checkpoint=checkpoint, complete=True)), flush=True)
    finally:
        env._sample_motion_times = original_reset
        # Kit shutdown hangs on this headless setup after the report has
        # finished. This isolated process owns the simulator; the OS releases
        # its resources. Preserve failures and flush diagnostics before exit.
        failed = sys.exc_info()[0] is not None
        if failed:
            traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
