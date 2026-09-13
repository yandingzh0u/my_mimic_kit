"""Serial 500-iteration continuation of the four existing Climb runs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
QUEUE = ROOT / 'output/paper_benchmark/ablations/climb_resume500_queue'
PYTHON = '/home/y/miniconda3/envs/env_isaaclab/bin/python'
TARGET = 655360000
SOURCES = [
    ('dare', 'output/paper_benchmark/dare_climb_2k_8192_seed0'),
    ('wogroup', 'output/paper_benchmark/ablations/climb_wogroup_2k_8192_seed0'),
    ('wocalibration', 'output/paper_benchmark/ablations/climb_wocalibration_2k_8192_seed0'),
    ('base', 'output/paper_benchmark/ablations/climb_base_2k_8192_seed0'),
]


def event(variant, state, **kwargs):
    record = dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'), variant=variant,
                  state=state, **kwargs)
    with (QUEUE / 'events.jsonl').open('a') as f:
        f.write(json.dumps(record) + '\n')
    print(record, flush=True)


def main():
    os.chdir(ROOT)
    QUEUE.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (QUEUE / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    jobs = []
    for variant, relative in SOURCES:
        source = ROOT / relative
        checkpoint = torch.load(source / 'checkpoint.pt', map_location='cpu',
                                weights_only=False, mmap=True)
        assert checkpoint['trainer_state']['sample_count'] == 524288000
        assert checkpoint['trainer_state']['next_iter'] == 2000
        assert checkpoint['metadata']['num_envs'] == 8192
        for name in ('agent_config', 'env_config', 'engine_config'):
            digest = hashlib.sha256((source / (name + '.yaml')).read_bytes()).hexdigest()
            assert digest == checkpoint['metadata']['checkpoint_context'][name + '_sha256']
        dest = QUEUE / (variant + '_resume500')
        dest.mkdir(exist_ok=True)
        for name in ('agent_config.yaml', 'env_config.yaml', 'engine_config.yaml'):
            if (dest / name).exists():
                assert (dest / name).read_bytes() == (source / name).read_bytes()
            else:
                shutil.copy2(source / name, dest / name)
        jobs.append((variant, source, dest))
    for variant, source, dest in jobs:
        resume = dest / 'checkpoint.pt'
        if not resume.exists():
            resume = source / 'checkpoint.pt'
        current = torch.load(resume, map_location='cpu', weights_only=False, mmap=True)
        if current['trainer_state']['sample_count'] >= TARGET:
            event(variant, 'SKIPPED_COMPLETE')
            continue
        cmd = [PYTHON, 'mimickit/run.py', '--mode', 'train', '--num_envs', '8192',
               '--devices', 'cuda:0', '--rand_seed', '0', '--visualize', 'false',
               '--save_int_models', 'true', '--logger', 'txt',
               '--max_samples', str(TARGET), '--resume_file', str(resume),
               '--out_dir', str(dest)]
        for name in ('agent_config', 'env_config', 'engine_config'):
            cmd += ['--' + name, str(source / (name + '.yaml'))]
        event(variant, 'STARTED', output=str(dest), command=cmd)
        with (dest / 'console.log').open('a') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, TERM='xterm', PYTHONUNBUFFERED='1'))
        if result.returncode:
            event(variant, 'FAILED', returncode=result.returncode)
            raise SystemExit(result.returncode)
        saved = torch.load(dest / 'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
        assert saved['trainer_state']['sample_count'] == TARGET
        assert saved['trainer_state']['next_iter'] == 2500
        event(variant, 'DONE', samples=TARGET)
    event('all', 'DONE')


if __name__ == '__main__':
    main()
