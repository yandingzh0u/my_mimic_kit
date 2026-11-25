#!/usr/bin/env python3
"""Evaluate the first fixed-start episode in each lane, using PhysX contacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / 'mimickit')]

from PAPER.tools.evaluation.evaluate_checkpoint import _query_reference, _query_sim, _reset_with_protocol
from PAPER.tools.evaluation.metrics import TRACKING_ERROR_NAMES, compute_tracking_errors


TASKS = ('carry_box', 'soccer_cmu_10_02', 'sofa003')
METHODS = ('AMP', 'DeepMimic', 'ADD', 'MIRA')
SEED, N, BUDGET = 0, 256, 524_288_000
CRITERIA = {
    'carry_box': dict(clearance_m=.10, lift_s=.20, hand_force_n=5., air_travel_m=.30,
                      total_travel_m=1., goal_error_m=.15, floor_error_m=.04,
                      tilt_deg=15., speed_m_s=.20, spin_rad_s=1., settle_s=.20),
    'soccer_cmu_10_02': dict(impulse_n_s=.05, timing_s=.20, travel_window_s=.20,
                             travel_m=.50, direction_deg=30.),
    'sofa003': dict(settle_s=.50, support_n=30., pelvis_speed_m_s=.20),
}


def run_directory(task, method):
    if method == 'MIRA':
        dims = 68 if task == 'carry_box' else 64
        return ROOT / 'output' / f'mira_cmnn{dims}_group_rms_{task}_2000_8192_seed0_20261122'
    return ROOT / 'output' / f'{method.lower()}_{task}_2k_8192_seed0_20261122'


def steps_at(seconds, dt):
    return math.ceil(seconds / dt - 1e-8)


class PairContacts:
    def __init__(self, env, obj_id, names):
        from engines.isaac_lab_engine import ENV_PATH_TEMPLATE, OBJ_PATH_TEMPLATE
        from isaacsim.core.simulation_manager import SimulationManager

        self.engine = env._engine
        sensor = self.engine._ground_contact_sensors[env._get_char_id()]
        if sensor is None:
            raise RuntimeError('Humanoid has no contact reporting')
        parent = sensor.cfg.prim_path.rsplit('/', 1)[0].replace('.*', '*')
        obj = OBJ_PATH_TEMPLATE.format('*', obj_id)
        num_envs = env.get_num_envs()
        paths = {ENV_PATH_TEMPLATE.format(i): i for i in range(num_envs)}
        sim = SimulationManager.get_physics_sim_view()
        self.views = []
        for name in names:
            view = sim.create_rigid_contact_view(f'{parent}/{name}', [obj])
            ids = [paths[path.split('/obj_', 1)[0]] for path in view.sensor_paths]
            if sorted(ids) != list(range(num_envs)) or view.filter_count != 1:
                raise RuntimeError(f'Invalid {name} contact view for object {obj_id}')
            self.views.append((view, torch.as_tensor(ids, device=env._device).argsort()))
        shape = (num_envs, len(names), 3)
        self.last = torch.zeros(shape, device=env._device)
        self.impulse = torch.zeros_like(self.last)
        self.impulse_mag = torch.zeros(shape[:2], device=env._device)
        self.physics_dt = self.engine.get_sim_timestep()
        self.original = self.engine._post_sim_step

    def __enter__(self):
        self.engine._post_sim_step = self.sample
        return self

    def __exit__(self, *exc):
        self.engine._post_sim_step = self.original

    def clear(self):
        self.impulse.zero_()
        self.impulse_mag.zero_()

    def sample(self):
        self.original()
        for i, (view, order) in enumerate(self.views):
            force = view.get_contact_force_matrix(dt=self.physics_dt)[order, 0, :]
            self.last[:, i] = force
            self.impulse[:, i].add_(force * self.physics_dt)
            self.impulse_mag[:, i].add_(force.norm(dim=-1) * self.physics_dt)


def carry_outcome(d, cfg, dt, complete):
    c = CRITERIA['carry_box']
    live, box = d['active'], d['object_pos']
    touching = d['hand_force_n'] >= c['hand_force_n']
    lifted = live & (d['box_bottom_m'] >= c['clearance_m'])
    bilateral = lifted & touching.all(axis=-1)
    start = np.asarray(cfg['box_start_xy'])
    overall = np.linalg.norm(box[..., :2] - start, axis=-1)
    end_error = np.linalg.norm(box - np.asarray(cfg['box_goal']), axis=-1)
    steady = (live & ~touching.any(axis=-1) & (overall >= c['total_travel_m'])
              & (end_error <= c['goal_error_m']) & (np.abs(d['box_bottom_m']) <= c['floor_error_m'])
              & (d['box_tilt_deg'] <= c['tilt_deg']) & (d['object_speed_m_s'] <= c['speed_m_s'])
              & (d['box_spin_rad_s'] <= c['spin_rad_s']))
    min_lift = steps_at(c['lift_s'], dt)
    if abs(cfg['initial_box_bottom_m']) > c['floor_error_m']:
        raise ValueError('Carry reference does not start with the box on the floor')
    rows = []
    for lane, length in enumerate(live.sum(axis=0)):
        lift = support = settle = 0
        peak_lift = 0
        peak_travel = travel = 0.
        for t in range(length):
            lift = lift + 1 if lifted[t, lane] else 0
            peak_lift = max(lift, peak_lift)
            if bilateral[t, lane]:
                if support == 0:
                    support_start, travel = box[t, lane, :2], 0.
                support += 1
                travel = max(travel, float(np.linalg.norm(box[t, lane, :2] - support_start)))
                if support >= min_lift:
                    peak_travel = max(peak_travel, travel)
            else:
                support = 0
            settle = settle + 1 if steady[t, lane] else 0
        pickup = peak_lift >= min_lift
        transport = pickup and peak_travel >= c['air_travel_m']
        placement = settle >= steps_at(c['settle_s'], dt)
        rows.append(dict(pickup=pickup, transport=transport, placement=placement,
                         success=bool(complete[lane] and transport and placement),
                         airborne_displacement_m=peak_travel, final_goal_distance_m=float(end_error[length-1, lane]),
                         final_box_travel_m=float(overall[length-1, lane]), settled_s=settle*dt))
    return rows


def soccer_outcome(d, cfg, dt, complete):
    c = CRITERIA['soccer_cmu_10_02']
    toward = np.array(cfg['reference_direction_xy'])
    observe = steps_at(c['travel_window_s'], dt)
    rows = []
    for lane, length in enumerate(d['active'].sum(axis=0)):
        impulse = d['foot_impulse_n_s'][:length, lane]
        hits = np.flatnonzero(impulse.max(axis=-1) > c['impulse_n_s'])
        hit = int(hits[0]) if len(hits) else None
        foot = ('right' if impulse[hit, 0] > impulse[hit, 1] else 'left') if hit is not None else None
        hit_time = (hit + 1) * dt if hit is not None else None
        timed = hit_time is not None and foot == 'right' and abs(hit_time - cfg['impact_time_s']) <= c['timing_s'] + 1e-8
        displacement = angle = speed = None
        if hit is not None and hit + observe - 1 < length:
            xy = d['object_pos'][:, lane, :2]
            before = xy[hit-1] if hit else np.array(cfg['ball_start_xy'])
            delta = xy[hit + observe - 1] - before
            displacement = float(np.linalg.norm(delta))
            speed = float(d['ball_xy_speed_m_s'][hit:hit+observe, lane].max())
            if displacement > 1e-8:
                angle = float(np.degrees(np.arccos(np.clip(delta @ toward / displacement, -1, 1))))
        effective = timed and displacement is not None and displacement >= c['travel_m']
        rows.append(dict(first_foot=foot, first_contact_s=hit_time, timed_right_contact=bool(timed),
                         effective_contact=bool(effective), travel_m=displacement, direction_deg=angle,
                         peak_ball_xy_speed_m_s=speed,
                         success=bool(effective and angle is not None and angle <= c['direction_deg'])))
    return rows


def sofa_outcome(d, cfg, dt, complete):
    c = CRITERIA['sofa003']
    window = steps_at(c['settle_s'], dt)
    rows = []
    for i, length in enumerate(d['active'].sum(axis=0)):
        tail = slice(max(0, length-window), length)
        force = float(d['seat_support_n'][tail, i].mean())
        speed = float(d['pelvis_speed_m_s'][tail, i].mean())
        height_span = float(np.ptp(d['root_pos'][tail, i, 2]))
        rows.append(dict(mean_seat_support_n=force, mean_pelvis_speed_m_s=speed,
                         pelvis_height_span_m=height_span,
                         success=bool(complete[i] and force >= c['support_n'] and speed <= c['pelvis_speed_m_s'])))
    return rows


def measure(task, env, agent, n, controller):
    from envs.base_env import DoneFlags
    from util.torch_util import quat_to_matrix

    engine = env._engine
    dt = engine.get_timestep()
    horizon = round(env._episode_length / dt)
    if not math.isclose(horizon * dt, env._episode_length, abs_tol=1e-6):
        raise ValueError('Task duration is not aligned to the control clock')
    cfg = {}
    if task == 'carry_box':
        initial = env._reference.box[0]
        half_size = env._box_size / 2
        cfg.update(box_start_xy=initial[:2].tolist(), box_goal=env._box_goal.tolist(),
                   initial_box_bottom_m=float(initial[2] - (quat_to_matrix(initial[3:7])[2].abs() * half_size).sum()))
        obj, names, outcome = env._box_id, ('left_hand', 'right_hand'), carry_outcome
    elif task == 'soccer_cmu_10_02':
        with np.load('data/motions/humanoid/humanoid_soccer_cmu_10_02_reference.npz') as ref:
            h = round(env._impact_time / dt)
            end = h + steps_at(CRITERIA[task]['travel_window_s'], dt) - 1
            delta = ref['ball_pos'][end, :2] - ref['ball_pos'][h - 1, :2]
            direction = delta / np.linalg.norm(delta)
        cfg.update(reference_direction_xy=direction.tolist(), impact_time_s=env._impact_time,
                   ball_start_xy=env._ball_pos[:2].tolist())
        obj, names, outcome = env._ball_id, ('right_foot', 'left_foot'), soccer_outcome
    else:
        obj, = [i for i, x in enumerate(engine._obj_cfgs[0]) if Path(x.asset_file).name == 'samp_sofa003.usd']
        names, outcome = ('pelvis', 'left_thigh', 'right_thigh'), sofa_outcome

    keys = TRACKING_ERROR_NAMES[:-1]
    data = {}
    alive = torch.ones(n, device=env._device, dtype=torch.bool)
    termination = torch.zeros(n, device=env._device, dtype=torch.int)
    obs, info = _reset_with_protocol(env, 'phase0')
    if env._motion_time_offsets.abs().max().item() > 1e-7:
        raise RuntimeError('Evaluation did not start at phase zero')
    neutral = (env._action_bound_low + env._action_bound_high) / 2

    def record(**frame):
        for key, val in frame.items():
            data.setdefault(key, []).append(val.clone())

    with torch.inference_mode(), PairContacts(env, obj, names) as contacts:
        for _ in range(horizon):
            contacts.clear()
            actions = (agent._decide_action(obs, info)[0] if controller == 'policy'
                       else neutral.expand(n, -1))
            actions = torch.where(alive[:, None], actions, neutral)
            obs, _, done, info = env.step(actions)
            sim, ref = _query_sim(env), _query_reference(env)
            errs = compute_tracking_errors(**sim, **{f'ref_{k}': ref[k] for k in sim})
            record(active=alive, done=done, root_pos=sim['root_pos'], reference_root_pos=ref['root_pos'],
                   tracking_error=torch.stack([errs[k] for k in keys], dim=-1))
            if task == 'carry_box':
                pos = engine.get_root_pos(obj)
                z_axes = quat_to_matrix(engine.get_root_rot(obj))[..., 2, :].abs()
                record(object_pos=pos, hand_force_n=contacts.last.norm(dim=-1),
                       box_bottom_m=pos[:, 2] - (z_axes * half_size).sum(dim=-1),
                       box_tilt_deg=z_axes.amax(dim=-1).clamp(0, 1).acos().rad2deg(),
                       object_speed_m_s=engine.get_root_vel(obj).norm(dim=-1),
                       box_spin_rad_s=engine.get_root_ang_vel(obj).norm(dim=-1))
            elif task == 'soccer_cmu_10_02':
                record(object_pos=engine.get_root_pos(obj), foot_impulse_n_s=contacts.impulse_mag,
                       ball_xy_speed_m_s=engine.get_root_vel(obj)[:, :2].norm(dim=-1))
            else:
                record(seat_support_n=contacts.impulse[:, :, 2].sum(dim=-1) / dt,
                       pelvis_speed_m_s=sim['root_vel'].norm(dim=-1))
            ended = alive & (done != DoneFlags.NULL.value)
            termination = torch.where(ended, done, termination)
            alive = alive & ~ended
            if not bool(alive.any()):
                break

    traces = {key: torch.stack(val).cpu().numpy() for key, val in data.items()}
    active = traces['active']
    for key, val in traces.items():
        if not np.isfinite(val[active]).all():
            raise ValueError(f'Nonfinite active trajectory: {key}')
    lengths = active.sum(axis=0)
    termination = termination.cpu().numpy()
    complete = (lengths == horizon) & (termination != DoneFlags.FAIL.value)
    rows = outcome(traces, cfg, dt, complete)
    err_means = np.where(active[..., None], traces['tracking_error'], 0).sum(0) / lengths[:, None]
    for i, row in enumerate(rows):
        row.update(episode=i, steps=int(lengths[i]), completed=bool(complete[i]),
                   termination=DoneFlags(int(termination[i])).name,
                   tracking={name: float(err_means[i, k]) for k, name in enumerate(keys)})
    return dict(criteria=CRITERIA[task], reference=cfg, contact_bodies=names, horizon=horizon,
                control_hz=round(1/dt), completed=int(complete.sum()),
                success=sum(row['success'] for row in rows), episodes=rows), traces


def main():
    import envs.env_builder as env_builder
    import learning.agent_builder as agent_builder
    from learning.base_agent import AgentMode
    import util.mp_util as mp_util
    import util.util as util

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task', choices=TASKS)
    parser.add_argument('method', choices=METHODS)
    parser.add_argument('--controller', choices=('policy', 'guide'), default='policy')
    parser.add_argument('--num-envs', type=int, default=N)
    parser.add_argument('--evaluation-seed', type=int, default=SEED)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    if args.num_envs < 1 or (args.controller == 'guide' and args.task == 'sofa003'):
        parser.error('The guide exists only for carry and soccer; num-envs must be positive')
    dest = args.destination.resolve()
    os.chdir(ROOT)
    run = run_directory(args.task, args.method)
    model = run / 'model.pt'
    final = json.loads((run / 'train_metrics.jsonl').read_text().splitlines()[-1])
    if (final['Iteration'], final['Samples']) != (1999, BUDGET):
        raise ValueError(f'Model has not reached the shared budget: {run}')
    env_path, engine_path, agent_path = (run / f'{k}_config.yaml' for k in ('env', 'engine', 'agent'))
    util.set_rand_seed(np.uint64(args.evaluation_seed))
    mp_util.init(rank=0, num_procs=1, device='cuda:0', master_port=6381)
    env = env_builder.build_env(str(env_path), str(engine_path), args.num_envs, 'cuda:0', visualize=False)
    agent = agent_builder.build_agent(str(agent_path), env, 'cuda:0')
    if args.controller == 'policy':
        agent.load(str(model))
    agent.eval()
    agent.set_mode(AgentMode.TEST)
    report, traces = measure(args.task, env, agent, args.num_envs, args.controller)
    report.update(task=args.task, method=args.method if args.controller == 'policy' else 'Guide only',
                  controller=args.controller, policy='deterministic mean action',
                  start='phase0', n=args.num_envs, evaluation_seed=args.evaluation_seed,
                  run=str(run.relative_to(ROOT)), model_sha256=hashlib.sha256(model.read_bytes()).hexdigest()
                  if args.controller == 'policy' else None, training_samples=BUDGET if args.controller == 'policy' else 0,
                  tracking_columns=TRACKING_ERROR_NAMES[:-1])
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / 'trajectories.npz', **traces)
    (dest / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
    print(f"{report['method']} / {args.task}: {report['success']}/{args.num_envs} successes; {report['completed']} complete", flush=True)


if __name__ == '__main__':
    main()
