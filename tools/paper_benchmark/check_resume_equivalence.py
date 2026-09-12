"""Compare five uninterrupted updates with two + process restart + three."""
import json
import os
from pathlib import Path
import subprocess
import sys
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'output/paper_benchmark/ablations/resume_equivalence_5iter'
SOURCE = ROOT / 'output/paper_benchmark/dare_climb_2k_8192_seed0'


def run(name, iterations, resume=False):
    dest = OUT / name
    dest.mkdir(exist_ok=True)
    cmd = [sys.executable, 'mimickit/run.py', '--mode', 'train',
           '--num_envs', '8192', '--devices', 'cuda:0', '--rand_seed', '0',
           '--visualize', 'false', '--save_int_models', 'true', '--logger', 'txt',
           '--max_samples', str(iterations * 8192 * 32), '--out_dir', str(dest),
           '--agent_config', str(OUT / 'agent.yaml'),
           '--env_config', str(SOURCE / 'env_config.yaml'),
           '--engine_config', str(SOURCE / 'engine_config.yaml')]
    if resume:
        cmd += ['--resume_file', str(dest / 'checkpoint.pt'), '--resume_initial_reset', 'test']
    print('START', name, iterations, resume, flush=True)
    with (dest / ('resume_console.log' if resume else 'console.log')).open('w') as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, TERM='xterm', PYTHONUNBUFFERED='1'))
    checkpoint = torch.load(dest / 'checkpoint.pt', map_location='cpu', weights_only=False)
    assert checkpoint['trainer_state']['next_iter'] == iterations
    print('DONE', name, iterations, flush=True)


def main():
    os.chdir(ROOT)
    run('continuous', 5)
    run('split', 2)
    run('split', 5, True)
    report = {}
    for i in range(5):
        states = [torch.load(OUT / n / 'int_models' / f'model_{i:010d}.pt',
                             map_location='cpu', weights_only=False)
                  for n in ('continuous', 'split')]
        differences = {}
        for k in states[0]:
            a, b = states[0][k], states[1][k]
            if not torch.equal(a, b):
                differences[k] = float((a.double() - b.double()).abs().max())
        report[i] = dict(differing_tensors=len(differences),
                         max_abs=max(differences.values(), default=0), differences=differences)
    (OUT / 'comparison.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {a:b for a,b in v.items() if a != 'differences'} for k,v in report.items()}), flush=True)


if __name__ == '__main__':
    main()
