"""Real Isaac Lab checks for the production two-stage PPO sampler."""
import argparse
import gc
import json
import os
from pathlib import Path
import sys
import time
import traceback
import types

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mimickit"))
ARCHIVE = ROOT / "output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012"
CONFIG = ROOT / "data/agents/dare_10layer_cpl_climb_twostage_agent.yaml"


def audit(agent, repeats, seed):
    from learning.phase_sampler import PhaseSampler
    from adaptive_phase_candidate import AdaptivePhaseSampler
    from adaptive_phase_live import gradient
    phase = agent._exp_buffer.get_data_flat("diag_phase").detach()
    eligible = agent._exp_buffer.get_data_flat("rand_action_mask") == 1
    tensors = {k: agent._exp_buffer.get_data_flat(k).detach()
               for k in ("obs", "action", "a_logp", "rand_action_mask")}
    frames, size, budget = agent._phase_control_cells, 4*agent.get_num_envs(), 2*agent.get_num_envs()
    sampler = PhaseSampler(phase, frames, budget, eligible)
    old = AdaptivePhaseSampler(phase, frame_count=frames, batch_budget=budget, eligible=eligible)
    full = sampler.full_weights()
    torch.testing.assert_close(full, old.full_weights(), atol=1e-15, rtol=0)
    with torch.no_grad():
        raw = agent._exp_buffer.get_data_flat("tar_val") - agent._model.eval_critic(
            agent._obs_norm.normalize(tensors["obs"])).squeeze(-1)
        mean = (full*raw).sum()
        std = ((full*raw.square()).sum()-mean.square()).clamp_min(0).sqrt()
        adv = ((raw-mean)/std.clamp_min(1e-5)).clamp(-agent._norm_adv_clip, agent._norm_adv_clip).float()
    exact = gradient(agent, tensors, full[sampler.indices], adv, sampler.indices).double()
    result = dict(coverage=sampler.coverage, full_target_difference=0, methods={})
    errors = {}
    for name, candidate in (("balanced_reuse", old), ("twostage", sampler)):
        summed = torch.zeros_like(exact)
        squared = 0.
        errors[name] = []
        unique = size
        for repeat in range(repeats):
            # Independent pair seeds: different RNG consumption cannot shift
            # all later pairs. Shared within-bin order where possible.
            gen = torch.Generator(device=phase.device).manual_seed(seed+repeat)
            idx, weight, quota = candidate.sample(size, generator=gen)
            unique = min(unique, len(idx.unique()))
            if name == "twostage":
                assert unique == min(size, len(sampler.indices)) and bool((quota >= 1).all())
            g = gradient(agent, tensors, weight, adv, idx).double()
            summed += g
            squared += float(g.square().sum())
            errors[name].append(float((g-exact).square().sum()))
        variance = (squared-float(summed.square().sum())/repeats)/(repeats-1)
        result["methods"][name] = dict(gradient_variance=variance, mse=float(np.mean(errors[name])),
                                      minimum_unique=unique, samples=errors[name])
    result["mse_ratio"] = result["methods"]["twostage"]["mse"] / result["methods"]["balanced_reuse"]["mse"]
    rng = np.random.default_rng(seed)
    ratios = []
    for _ in range(2000):
        picks = rng.integers(repeats, size=repeats)
        ratios.append(float(np.asarray(errors["twostage"])[picks].mean()
                            / np.asarray(errors["balanced_reuse"])[picks].mean()))
    result["paired_bootstrap_95ci_ratio"] = np.quantile(ratios, [.025, .975]).tolist()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("gradient", "train"), default="train")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--agent-config", type=Path, default=CONFIG)
    args = parser.parse_args()
    os.chdir(ROOT)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    import learning.base_agent as base_agent
    from learning.phase_sampler import PhaseSampler
    import util.mp_util as mp_util
    import util.util as util

    torch.set_num_threads(4)
    mp_util.init(0, 1, "cuda:0", 6403)
    util.set_rand_seed(args.seed)
    env = env_builder.build_env(str(ROOT / "data/envs/paper_benchmark/dare_climb_env.yaml"),
                                str(ROOT / "data/engines/isaac_lab_engine.yaml"),
                                args.num_envs, "cuda:0", visualize=False)
    agent = agent_builder.build_agent(str(args.agent_config), env, "cuda:0")
    report = dict(mode=args.mode, num_envs=args.num_envs, seed=args.seed,
                  phase_sampler=agent._phase_sampler_mode,
                  control_cells=getattr(agent,"_phase_control_cells",None))
    try:
        if args.mode == "gradient":
            report["scope"] = "Frozen-policy sampling, identical finite-rollout target; no tracking-error claim"
            report["checkpoints"] = {}
            for checkpoint in (0, 1600, 2999):
                agent.load(str(ARCHIVE / f"int_models/model_{checkpoint:010d}.pt"))
                agent._init_train()
                agent._sample_count = 10**12  # Freeze checkpoint normalizers.
                agent.eval()
                agent.set_mode(base_agent.AgentMode.TRAIN)
                agent._curr_obs, agent._curr_info = env.reset()
                with torch.no_grad():
                    agent._rollout_train(agent._phase_control_cells)
                    agent._init_iter()
                    agent._rollout_train(32)
                    agent._build_train_data()
                result = audit(agent, args.repeats, 20261025)
                report["checkpoints"][str(checkpoint)] = result
                print(json.dumps(dict(checkpoint=checkpoint, mse_ratio=result["mse_ratio"], ci=result["paired_bootstrap_95ci_ratio"])), flush=True)
                (args.out / "gradient.json").write_text(json.dumps(report, indent=2)+"\n")
        else:
            report.update(iterations=[], batch_checks=0, sampled_indices=0)
            original_sample = PhaseSampler.sample
            def checked_sample(self, b, generator=None):
                idx, weight, quota = original_sample(self, b, generator)
                assert len(idx.unique()) == len(idx) == min(b, len(self.indices))
                assert bool(((quota >= 1) & (quota <= self.counts)).all())
                mass = torch.zeros(self.num_bins, device=idx.device, dtype=torch.float64).scatter_add_(0, self.bin_ids[idx], weight)
                torch.testing.assert_close(mass[self.present_bins], self.target_mass, atol=1e-12, rtol=0)
                report["batch_checks"] += 1
                report["sampled_indices"] += len(idx)
                return idx, weight, quota
            PhaseSampler.sample = checked_sample
            original_iter = agent._train_iter
            def checked_iter(self):
                start = time.perf_counter()
                info = original_iter()
                names = ("actor_loss", "critic_loss", "disc_loss", "actor_grad_norm", "critic_grad_norm", "adv_mean", "adv_std")
                values = {name: float(info[name]) for name in names}
                assert all(np.isfinite(v) for v in values.values())
                for p in self._model.parameters():
                    assert bool(torch.isfinite(p).all())
                for k in ("obs", "action", "adv", "phase_weight", "tar_val"):
                    assert bool(torch.isfinite(self._exp_buffer.get_data(k)).all())
                values.update(iteration=self._iter, seconds=time.perf_counter()-start,
                              missing_cells=(self._critic_phase_sampler.coverage["missing_control_cells"]
                                             if hasattr(self,"_critic_phase_sampler") else None),
                              selected_bins=(self._critic_phase_sampler.num_bins
                                             if hasattr(self,"_critic_phase_sampler") else None))
                report["iterations"].append(values)
                print("SMOKE_ITER " + json.dumps(values), flush=True)
                return info
            agent._train_iter = types.MethodType(checked_iter, agent)
            # Exercise the normal production loop, both evaluation modes,
            # model/checkpoint saving, and then a real optimizer continuation.
            agent.train_model(args.iterations*32*args.num_envs, str(args.out), True, "txt")
            checkpoint = agent._torch_load(args.out / "checkpoint.pt")
            expected_extra = checkpoint["agent_extra_state"]
            steps = {name: opt.get_steps() for name, opt in agent._get_optimizers().items()}
            del checkpoint
            gc.collect()
            agent.resume(args.out / "checkpoint.pt")
            assert agent._get_checkpoint_extra_state() == expected_extra
            agent.train_model((args.iterations+1)*32*args.num_envs, str(args.out), True, "txt")
            assert all(opt.get_steps() > steps[name] for name, opt in agent._get_optimizers().items())
            assert agent._sample_count == (args.iterations+1)*32*args.num_envs
            report.update(resume_passed=True, completed_iterations=agent._iter,
                          samples=agent._sample_count, maximum_torch_reserved_mb=torch.cuda.max_memory_reserved()/2**20,
                          finite_parameters=True, no_duplicate_or_starved_batches=True)
            (args.out / "smoke.json").write_text(json.dumps(report, indent=2)+"\n")
        print("LIVE_CHECK_COMPLETE", flush=True)
    finally:
        # Kit shutdown can hang in this installation. The isolated diagnostic
        # exits after flushing results; failures retain a nonzero exit status.
        failed = sys.exc_info()[0] is not None
        if failed:
            traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1 if failed else 0)


if __name__ == "__main__":
    main()
