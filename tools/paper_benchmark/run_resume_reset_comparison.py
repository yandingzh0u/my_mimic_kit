"""Two isolated 200-iteration runs differing only in the first reset mode."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'output/paper_benchmark/dare_climb_2k_8192_seed0'
OUT = ROOT / 'output/paper_benchmark/ablations/dare_resume_reset_comparison'
PYTHON = '/home/y/miniconda3/envs/env_isaaclab/bin/python'
TARGET = 576716800  # 2200 * 8192 * 32


def main():
    import fcntl
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'queue.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    checkpoint = torch.load(SOURCE / 'checkpoint.pt', map_location='cpu',
                            weights_only=False, mmap=True)
    assert checkpoint['trainer_state']['next_iter'] == 2000
    assert checkpoint['trainer_state']['sample_count'] == 524288000
    for name in ('agent_config', 'env_config', 'engine_config'):
        digest = hashlib.sha256((SOURCE / (name + '.yaml')).read_bytes()).hexdigest()
        assert digest == checkpoint['metadata']['checkpoint_context'][name + '_sha256']
    for mode in ('train', 'test'):
        dest = OUT / ('initial_reset_' + mode)
        dest.mkdir(exist_ok=False)
        cmd = [PYTHON, 'mimickit/run.py', '--mode', 'train', '--num_envs', '8192',
               '--devices', 'cuda:0', '--rand_seed', '0', '--visualize', 'false',
               '--save_int_models', 'true', '--logger', 'txt',
               '--resume_initial_reset', mode, '--max_samples', str(TARGET),
               '--resume_file', str(SOURCE / 'checkpoint.pt'), '--out_dir', str(dest)]
        for name in ('agent_config', 'env_config', 'engine_config'):
            cmd += ['--' + name, str(SOURCE / (name + '.yaml'))]
        print(json.dumps(dict(time=time.strftime('%Y-%m-%dT%H:%M:%S%z'),
                              mode=mode, state='STARTED', command=cmd)), flush=True)
        with (dest / 'console.log').open('w') as log:
            result = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, TERM='xterm', PYTHONUNBUFFERED='1'))
        assert result.returncode == 0, result.returncode
        saved = torch.load(dest / 'checkpoint.pt', map_location='cpu', weights_only=False, mmap=True)
        assert saved['trainer_state']['sample_count'] == TARGET
        assert saved['trainer_state']['next_iter'] == 2200
        print(json.dumps(dict(mode=mode, state='DONE', samples=TARGET)), flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
