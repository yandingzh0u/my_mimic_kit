#!/usr/bin/env python3
"""Audit physical trial summaries and export the paper's interaction table."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import yaml

from evaluate_interactions import (
    BUDGET, CRITERIA, METHODS, N, ROOT, SEED, TASKS,
    carry_outcome, run_directory, soccer_outcome, sofa_outcome,
)


def read_trial(source, task, method, guide, config, engine):
    directory = source / task / ('guide' if guide else method)
    report = json.loads((directory / 'summary.json').read_text())
    label = 'Guide only' if guide else method
    hz = engine['control_freq']
    horizon = round(config['episode_length'] * hz)
    expected = dict(task=task, method=label, n=N, evaluation_seed=SEED, start='phase0',
                    training_samples=0 if guide else BUDGET, horizon=horizon,
                    control_hz=hz, controller='guide' if guide else 'policy',
                    criteria=CRITERIA[task], run=str(run_directory(task, method).relative_to(ROOT)))
    for key, value in expected.items():
        if report[key] != value:
            raise ValueError(f'{directory}: unexpected {key}: {report[key]}')
    log = (directory / 'eval.log').read_text()
    if '[Error]' in log or 'Traceback (most recent call last)' in log:
        raise ValueError(f'{directory}: simulator reported an error')
    episodes = report['episodes']
    if [e['episode'] for e in episodes] != list(range(N)):
        raise ValueError(f'{directory}: expected one episode per simulator replica')
    digest = None if guide else hashlib.sha256((ROOT / report['run'] / 'model.pt').read_bytes()).hexdigest()
    if digest != report['model_sha256']:
        raise ValueError(f'{directory}: checkpoint hash does not match the evaluated model')
    with np.load(directory / 'trajectories.npz') as archive:
        data = dict(archive)
    active = data['active']
    length = active.sum(axis=0)
    if np.any(length < 1) or not np.array_equal(active, np.arange(len(active))[:, None] < length):
        raise ValueError(f'{directory}: invalid sequence of executed steps')
    if not np.array_equal(length, [e['steps'] for e in episodes]):
        raise ValueError(f'{directory}: trajectory lengths do not match the report')
    terminal = data['done'][length - 1, np.arange(N)]
    complete = (length == horizon) & (terminal != 1)  # DoneFlags.FAIL
    scorer = {'carry_box': carry_outcome, 'soccer_cmu_10_02': soccer_outcome, 'sofa003': sofa_outcome}[task]
    scored = scorer(data, report['reference'], 1 / hz, complete)
    if ([e['success'] for e in scored] != [e['success'] for e in episodes]
            or sum(e['success'] for e in scored) != report['success']
            or complete.tolist() != [e['completed'] for e in episodes]
            or int(complete.sum()) != report['completed']):
        raise ValueError(f'{directory}: success or completion count does not match the saved trace')
    return dict(success=report['success'], completed=report['completed'],
                run=report['run'], model_sha256=report['model_sha256'],
                tracking_episode_mean={
                    key: sum(e['tracking'][key] for e in episodes) / N for key in report['tracking_columns']
                })


def aggregate(source):
    tasks = {}
    for task in TASKS:
        shared = None
        for method in METHODS:
            run = run_directory(task, method)
            config = yaml.safe_load((run / 'env_config.yaml').read_text())
            engine = yaml.safe_load((run / 'engine_config.yaml').read_text())
            # Only the method-specific reward and discriminator inputs differ.
            actor_config = {k: v for k, v in config.items() if k not in (
                'disc_observation', 'num_disc_obs_steps', 'enable_tracking_reward')}
            if shared is None:
                shared = actor_config, engine
            if shared != (actor_config, engine):
                raise ValueError(f'{run}: environment does not match the shared protocol')
            record = tasks.setdefault(task, dict(control_hz=engine['control_freq'], criteria=CRITERIA[task],
                                                  episode_s=config['episode_length'], results={}))
            record['results'][method] = read_trial(source, task, method, False, config, engine)
        if task != 'sofa003':
            record['guide_only'] = read_trial(source, task, 'MIRA', True, config, engine)
    return dict(evaluation=dict(replicas=N, seed=SEED, start='phase0', actor='deterministic mean action',
                                training_samples=BUDGET, training_seed=0), tasks=tasks)


def latex(data):
    tasks = ('carry_box', 'sofa003', 'soccer_cmu_10_02')
    by_task = [data['tasks'][task]['results'] for task in tasks]
    maxima = [max(r['success'] for r in results.values()) for results in by_task]
    rows = []
    for method in METHODS:
        cells = []
        for results, best in zip(by_task, maxima):
            count = results[method]['success']
            cell = f'{100 * count / N:.1f}'
            cells.append(r'\textbf{' + cell + '}' if count == best else cell)
        rows.append(' & '.join([method, *cells]) + r' \\')
    return '\n'.join([
        '% Generated by PAPER/tools/evaluation/build_interaction_table.py.',
        r'\begin{table}[!htbp]',
        r'\caption{Physical interaction success (\%) in ' + str(N) + r' fixed-start trials per method.',
        r'Each checkpoint uses one training seed. Bold marks the highest rate, including ties.}',
        r'\label{tab:interaction}',
        r'\centering\small',
        r'\setlength{\tabcolsep}{5pt}',
        r'\begin{tabular}{lccc}\toprule',
        r'Method & Box Carry & Sofa Sit & Soccer \\',
        r'\midrule',
        *rows,
        r'\bottomrule\end{tabular}',
        r'\end{table}',
        '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT / 'PAPER/build/interaction_eval')
    args = parser.parse_args()
    data = aggregate(args.source)
    dest = ROOT / 'PAPER/generated'
    (dest / 'mira_interaction_results.json').write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    (dest / 'mira_interaction_table.tex').write_text(latex(data))


if __name__ == '__main__':
    main()
