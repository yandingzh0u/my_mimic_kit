"""Compare reward geometry on fresh physical rollouts without policy updates."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

from dare_paper_stress import ROOT
from dare_ray_candidate import RayDisc, diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20261024)
    args = parser.parse_args()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util

    torch.set_num_threads(4)
    archive = ROOT / "output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012"
    mp_util.init(0, 1, "cuda:0", 6392)
    util.set_rand_seed(args.seed)
    env = env_builder.build_env(str(archive / "env_config.yaml"),
                                str(archive / "engine_config.yaml"),
                                args.num_envs, "cuda:0", visualize=False)
    agent = agent_builder.build_agent(str(archive / "agent_config.yaml"), env, "cuda:0")
    report = dict(protocol="fresh stochastic policy rollouts with random reference starts and explicit resets after termination; no policy updates",
                  seed=args.seed, num_envs=args.num_envs, steps=args.steps,
                  checkpoints={})
    for iteration in [0, 100, 1600, 2999]:
        util.set_rand_seed(args.seed)
        agent.load(str(archive / f"int_models/model_{iteration:010d}.pt"))
        agent.eval()
        agent.set_mode(AgentMode.TRAIN)
        obs, _ = env.reset()
        data, phases = [], []
        failures = successes = 0
        with torch.no_grad():
            for _ in range(args.steps):
                dist = agent._model.eval_actor(agent._obs_norm.normalize(obs))
                action = agent._a_norm.unnormalize(dist.sample())
                obs, _, done, info = env.step(action)
                diff = info["disc_obs_demo"] - info["disc_obs"]
                data.append(agent._disc_obs_norm.normalize(diff).detach().clone())
                phases.append(env.get_motion_phase().detach().clone())
                failures += int((done == 1).sum())
                successes += int((done == 2).sum())
                ids = (done != 0).nonzero(as_tuple=False).flatten()
                if ids.numel():
                    obs, _ = env.reset(ids)
        z = torch.cat(data)
        phase = torch.cat(phases)
        order = torch.randperm(len(z), device=z.device)
        fitting, heldout = z[order[:8192]], z[order[8192:10240]]
        ray = RayDisc(agent._model).to(z.device).eval()
        # DAREModel accepts eval_disc_raw rather than forward.
        class RawWrapper(torch.nn.Module):
            def forward(self, x):
                return agent._model.eval_disc_raw(x).squeeze(-1)
        original = RawWrapper().eval()
        current = dict(samples=len(z), failures_observed=failures,
                       successes_observed=successes,
                       baseline=diagnostics(original, fitting, heldout),
                       ray=diagnostics(ray, fitting, heldout),
                       per_phase={})
        with torch.no_grad():
            for b in range(8):
                selected = z[(phase >= b/8) & (phase < (b+1)/8)][:1024]
                if not len(selected):
                    continue
                for name, model in [("baseline", original), ("ray", ray)]:
                    scores = torch.stack([model(t * selected) for t in [0., .25, .5, 1., 2., 4.]])
                    current["per_phase"].setdefault(str(b), {})[name] = dict(
                        samples=len(selected),
                        nonmonotone=float(((scores[1:] - scores[:-1]) > 1e-5).any(0).float().mean()))
        report["checkpoints"][str(iteration)] = current
        (args.out / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(dict(iteration=iteration,
                              samples=len(z),
                              baseline_nonmonotone=current["baseline"]["heldout_radial"]["nonmonotone_ray_fraction"],
                              ray_nonmonotone=current["ray"]["heldout_radial"]["nonmonotone_ray_fraction"],
                              baseline_bce=current["baseline"]["heldout_balanced_bce"],
                              ray_bce=current["ray"]["heldout_balanced_bce"])), flush=True)
    env._engine._app_launcher.app.close()
    os._exit(0)


if __name__ == "__main__":
    main()
