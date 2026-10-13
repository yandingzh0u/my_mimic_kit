"""Frozen-checkpoint factorial evaluation; no learning or training-code edits."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'mimickit'))
import envs.env_builder as env_builder
import envs.base_env as base_env
import learning.agent_builder as agent_builder
import learning.add_agent as add_agent
import util.mp_util as mp_util
import util.util as util


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-envs', type=int, default=128)
    parser.add_argument('--seeds', type=int, default=4)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    source = ROOT / 'output/paper_benchmark/dare_10layer_cpl_resetfix_climb_3000_8192_seed0_headless_20261010'
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    util.set_rand_seed(0)
    mp_util.init(0, 1, 'cuda:0', None)
    env = env_builder.build_env(str(source / 'env_config.yaml'),
                                str(source / 'engine_config.yaml'),
                                args.num_envs, 'cuda:0', visualize=False)
    agents = {}
    hashes = {}
    for iteration in (2200, 2300):
        agent = agent_builder.build_agent(str(source / 'agent_config.yaml'), env, 'cuda:0')
        checkpoint = source / f'int_models/model_{iteration:010d}.pt'
        agent.load(str(checkpoint))
        agent.eval()
        agents[iteration] = agent
        hashes[str(iteration)] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    # TRAIN restores the original early-termination semantics without editing
    # the currently modified TEST path. Action sampling is controlled below.
    env.set_mode(base_env.EnvMode.TRAIN)
    assert env._enable_early_termination and env._track_global_root()
    body_names = env._engine.get_obj_body_names(env._get_char_id())
    rows = []
    for start in ('phase0', 'random'):
        env._rand_reset = start == 'random'
        for stochastic in (False, True):
            for seed in range(args.seeds):
                initials = []
                for iteration in (2200, 2300):
                    actor = agents[iteration]
                    util.set_rand_seed(10000 + seed)
                    obs, _ = env.reset()
                    initial_obs = obs.clone()
                    initial_phase = env.get_motion_phase().clone()
                    initials.append(initial_obs)
                    if len(initials) == 2:
                        assert torch.equal(initials[0], initials[1]), 'Paired reset mismatch'
                    generator = torch.Generator(device='cuda:0').manual_seed(20000 + seed)
                    active = torch.ones(args.num_envs, device='cuda:0', dtype=torch.bool)
                    traces = {k: [] for k in ('active', 'done', 'reward', 'discounted_reward',
                        'root_error', 'body_error', 'max_body_error', 'contact_max',
                        'contact_body', 'max_error_body', 'phase', 'mean_action',
                        'shared_action_delta', 'root_pos', 'body_pos', 'contact_force')}
                    with torch.no_grad():
                        for step in range(310):
                            dist = actor._model.eval_actor(actor._obs_norm.normalize(obs))
                            other = agents[2300 if iteration == 2200 else 2200]
                            other_mean = other._model.eval_actor(other._obs_norm.normalize(obs)).mean
                            noise = torch.randn(dist.mean.shape, device='cuda:0', generator=generator)
                            norm_a = dist.mean + dist.stddev * noise if stochastic else dist.mean
                            mean_action = dist.mean.clone()
                            shared_delta = (other_mean - dist.mean).square().mean(-1).sqrt()
                            obs, _, done, info = env.step(actor._a_norm.unnormalize(norm_a))
                            raw_diff = info['disc_obs_demo'] - info['disc_obs']
                            rewards = []
                            for scorer in agents.values():
                                logit = scorer._model.eval_disc_raw(scorer._disc_obs_norm.normalize(raw_diff)).squeeze(-1)
                                rewards.append(scorer._disc_reward_scale * add_agent.calc_unscaled_disc_reward(logit))
                            reward = torch.stack(rewards, -1)
                            char = env._get_char_id()
                            body = env._engine.get_body_pos(char)
                            error = (body - env._ref_body_pos).norm(dim=-1)
                            rel = body - body[:, :1]
                            ref_rel = env._ref_body_pos - env._ref_body_pos[:, :1]
                            forces = env._engine.get_ground_contact_forces(char).clone()
                            masked = forces.clone()
                            masked[:, env._contact_body_ids] = 0
                            force_per_body = masked.abs().amax(-1)
                            values = dict(active=active, done=done, reward=reward,
                                discounted_reward=reward * actor._discount ** step,
                                root_error=error[:, 0], body_error=(rel-ref_rel).norm(dim=-1).mean(-1),
                                max_body_error=error.amax(-1), max_error_body=error.argmax(-1),
                                contact_max=force_per_body.amax(-1), contact_body=force_per_body.argmax(-1),
                                phase=env.get_motion_phase(), mean_action=mean_action,
                                shared_action_delta=shared_delta, root_pos=body[:, 0],
                                body_pos=body, contact_force=forces)
                            for key, value in values.items():
                                traces[key].append(value.detach().cpu().numpy().copy())
                            active = active & (done == base_env.DoneFlags.NULL.value)
                            if not active.any():
                                break
                    assert not active.any(), 'Evaluation censored before termination'
                    arrays = {k: np.stack(v) for k, v in traces.items()}
                    valid = arrays['active']
                    lengths = valid.sum(0)
                    flags = arrays['done'][lengths - 1, np.arange(args.num_envs)]
                    returns = (arrays['discounted_reward'] * valid[..., None]).sum(0)
                    sums = (arrays['reward'] * valid[..., None]).sum(0)
                    terminal_contact = arrays['contact_max'][lengths - 1, np.arange(args.num_envs)]
                    terminal_error = arrays['max_body_error'][lengths - 1, np.arange(args.num_envs)]
                    filename = f'{start}_{"sample" if stochastic else "mean"}_seed{seed}_actor{iteration}.npz'
                    np.savez_compressed(out / filename, **arrays, initial_obs=initial_obs.cpu().numpy(),
                        initial_phase=initial_phase.cpu().numpy(), lengths=lengths, returns=returns, reward_sums=sums)
                    row = dict(start=start, stochastic=stochastic, seed=seed, actor=iteration,
                        length_mean=float(lengths.mean()), return_D2200=float(returns[:,0].mean()),
                        return_D2300=float(returns[:,1].mean()), success=float((flags==2).mean()),
                        fail=float((flags==1).mean()), contact_fail=float(((flags==1)&(terminal_contact>0.1)).mean()),
                        pose_fail=float(((flags==1)&(terminal_error>env._pose_termination_dist)).mean()),
                        reset_phase_mean=float(initial_phase.mean()), file=filename)
                    rows.append(row)
                    print('CASE_RESULT '+json.dumps(row), flush=True)
                    (out / 'summary.json').write_text(json.dumps(dict(checkpoint_hashes=hashes,
                        body_names=body_names, num_envs=args.num_envs, seeds=args.seeds,
                        protocol='One episode per env; original TRAIN termination; paired reset/noise; frozen D2200 and D2300; no optimizer updates',
                        results=rows), indent=2))


if __name__ == '__main__':
    main()
