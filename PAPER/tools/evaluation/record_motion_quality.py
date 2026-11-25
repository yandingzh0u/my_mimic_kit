"""Record fixed-start robot motion and ground-support traces for the paper."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from evaluate_interactions import BUDGET, METHODS, N, ROOT, SEED, TASKS, run_directory
from evaluate_checkpoint import _query_reference, _reset_with_protocol


def record(env, agent):
    from envs.base_env import DoneFlags
    from util.torch_util import quat_to_matrix

    engine = env._engine
    character = env._get_char_id()
    feet = [engine.find_obj_body_id(character, name) for name in ('left_foot', 'right_foot')]
    device, dt = env._device, engine.get_timestep()
    sole = torch.tensor([.045, 0., -.05], device=device)
    # Corners of the feet's collision boxes in the local body frame.
    corners = torch.tensor([[x, y, z] for x in (-.0435, .1335)
                            for y in (-.045, .045) for z in (-.05, .005)], device=device)
    active = torch.ones(env.get_num_envs(), device=device, dtype=torch.bool)
    records = {}

    def sample():
        ref = _query_reference(env)
        body = engine.get_body_pos(character)
        foot_vel = engine.get_body_vel(character)[:, feet]
        foot_ang_vel = engine.get_body_ang_vel(character)[:, feet]
        rot = quat_to_matrix(engine.get_body_rot(character)[:, feet])
        ref_rot = quat_to_matrix(ref['body_rot'][:, feet])
        ref_foot = ref['body_pos'][:, feet]
        sole_vel = foot_vel + torch.cross(foot_ang_vel, rot @ sole, dim=-1)
        lowest = (ref_rot[..., 2, :] @ corners.T).amin(dim=-1) + ref_foot[..., 2]
        fields = dict(
            active=active, body=body, ref_body=ref['body_pos'],
            sole_speed=sole_vel[..., :2].norm(dim=-1), ref_sole=ref_foot + ref_rot @ sole,
            ref_lowest=lowest, foot_force=engine.get_ground_contact_forces(character)[:, feet, 2],
        )
        for key, value in fields.items():
            records.setdefault(key, []).append(value.clone())

    with torch.no_grad():
        obs, info = _reset_with_protocol(env, 'phase0')
        sample()  # The reset pose is needed for the first displacement.
        for _ in range(round(env._episode_length / dt)):
            action = agent._decide_action(obs, info)[0]
            obs, _, done, info = env.step(action)
            sample()
            active &= done == DoneFlags.NULL.value
            if not bool(active.any()):
                break
    return {key: torch.stack(value).cpu().numpy() for key, value in records.items()}, round(1 / dt)


def main():
    from envs import env_builder
    from learning import agent_builder
    from learning.base_agent import AgentMode
    from util import mp_util, util

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('motion', choices=TASKS)
    parser.add_argument('method', choices=METHODS)
    parser.add_argument('--destination', type=Path, required=True, help='Directory for traces')
    args = parser.parse_args()
    destination = args.destination.resolve()
    os.chdir(ROOT)
    run = run_directory(args.motion, args.method)
    model = run / 'model.pt'
    final = json.loads((run / 'train_metrics.jsonl').read_text().splitlines()[-1])
    if (final['Iteration'], final['Samples']) != (1999, BUDGET):
        raise ValueError(f'Expected the final seed-0 model: {run}')
    util.set_rand_seed(np.uint64(SEED))
    mp_util.init(rank=0, num_procs=1, device='cuda:0', master_port=6382)
    env = env_builder.build_env(str(run / 'env_config.yaml'), str(run / 'engine_config.yaml'),
                                N, 'cuda:0', visualize=False)
    agent = agent_builder.build_agent(str(run / 'agent_config.yaml'), env, 'cuda:0')
    agent.load(str(model))
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    traces, hz = record(env, agent)
    destination.mkdir(parents=True, exist_ok=True)
    stem = destination / f'{args.motion}_{args.method}'
    np.savez_compressed(stem.with_suffix('.npz'), hz=hz, **traces)
    metadata = dict(model_sha256=hashlib.sha256(model.read_bytes()).hexdigest(),
                    n=N, evaluation_seed=SEED, training_samples=BUDGET, start='phase0')
    stem.with_suffix('.json').write_text(json.dumps(metadata, indent=2) + '\n')


if __name__ == '__main__':
    main()
