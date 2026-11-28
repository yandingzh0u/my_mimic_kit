#!/usr/bin/env python3
"""Freeze a Backflip CPL control and launch it after the log-space baseline."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[3]
ENVS, STEPS, SEED = 8192, 32, 0
CONFIGS = {
    "agent": "data/agents/mira_monotone_humanoid_agent.yaml",
    "engine": "data/engines/isaac_lab_engine.yaml",
    "env": "PAPER/configs/envs/mira_backflip_env.yaml",
}


def replace_once(path, original, replacement):
    source = path.read_text()
    if source.count(original) != 1:
        raise ValueError(f"Cannot freeze CPL: live source changed at {path}")
    path.write_text(source.replace(original, replacement))


def freeze(queue):
    frozen = queue / "cpl_source"
    shutil.copytree(ROOT / "mimickit", frozen / "mimickit",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for key, relative in CONFIGS.items():
        shutil.copy2(ROOT / relative, frozen / f"{key}.yaml")
    learning = frozen / "mimickit/learning"
    shutil.copy2(Path(__file__).with_name("mira_cpl_layer.py"), learning)
    model = learning / "mira_monotone_model.py"
    replace_once(model, "from learning.add_model import ADDModel",
                 "from learning.add_model import ADDModel\nfrom learning.mira_cpl_layer import CPLLinear")
    replace_once(model, "layer = nn.Linear(in_features, out_features, bias=True)",
                 "layer = CPLLinear(in_features, out_features, bias=True)")
    replace_once(model, "MONOTONE_SCHEMA_VERSION = 27", "MONOTONE_SCHEMA_VERSION = 28")
    replace_once(learning / "mira_monotone_agent.py",
                 "SLL_nonresidual_log_P_log_q_Perron_init", "SLL_CPL_shared_readout_log_P_log_q_Perron_init")
    manifest = {str(p.relative_to(frozen)): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(frozen.rglob("*")) if p.is_file()}
    (queue / "cpl_source_sha256.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return frozen


def verify(frozen, manifest):
    for relative, digest in manifest.items():
        if hashlib.sha256((frozen / relative).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Frozen source changed: {frozen / relative}")


def status(queue, phase, **details):
    path = queue / "cpl_status.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({"phase": phase, "pid": os.getpid(), **details}, indent=2) + "\n")
    temporary.replace(path)


# The standalone Isaac Lab worker may hang in native teardown. Return only
# after production training has closed the logs and saved its checkpoints.
TRAIN_ENTRY = """
import os, sys, traceback
from run import main
try:
    main(sys.argv)
except BaseException:
    traceback.print_exc()
    code = 1
else:
    code = 0
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
"""


def run(queue, stage, directory, frozen, updates, manifest):
    directory.mkdir(parents=True, exist_ok=False)
    provenance = directory / "source_snapshot"
    if stage == "formal":
        shutil.copytree(frozen, provenance)
        frozen = provenance
        shutil.copy2(queue / "cpl_source_sha256.json", directory)
    verify(frozen, manifest)
    command = [
        sys.executable, "-u", "-c", TRAIN_ENTRY,
        "--mode", "train", "--num_envs", str(ENVS), "--devices", "cuda:0",
        "--master_port", "6497", "--rand_seed", str(SEED), "--visualize", "false",
        "--logger", "txt", "--max_samples", str(updates * ENVS * STEPS),
        "--out_dir", str(directory),
    ]
    for kind in CONFIGS:
        command.extend((f"--{kind}_config", str(frozen / f"{kind}.yaml")))
    if stage == "formal":
        command.extend(("--save_int_models", "true"))
    environment = dict(os.environ, ACCEPT_EULA="Y", OMNI_KIT_DISABLE_TELEMETRY="1",
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4",
                       MKL_NUM_THREADS="4", MIMICKIT_CRITIC_EVAL_TIME_BATCH="2",
                       PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.55",
                       PYTHONPATH=str(frozen / "mimickit"))
    console = queue / f"cpl_{stage}.console.log"
    with console.open("x") as stream:
        process = subprocess.Popen(command, cwd=ROOT, env=environment,
                                   stdout=stream, stderr=subprocess.STDOUT)
        status(queue, stage, trainer_pid=process.pid, run_dir=str(directory),
               updates=updates, envs=ENVS, seed=SEED)
        if process.wait() != 0:
            raise RuntimeError(f"CPL {stage} failed; see {console}")
    verify(frozen, manifest)
    rows = [json.loads(line) for line in (directory / "train_metrics.jsonl").read_text().splitlines()]
    last = rows[-1]
    if (last["Iteration"], last["Samples"]) != (updates - 1, updates * ENVS * STEPS):
        raise ValueError(f"Missing terminal budget in {directory}")
    for row in rows:
        for name in ("Critic_Loss", "Disc_Cls_Loss", "Disc_Reward_Mean",
                     "Disc_Reward_Std", "Disc_Param_Grad_Norm", "Actor_Grad_Norm"):
            if not math.isfinite(row[name]):
                raise ValueError(f"Nonfinite {name} in {directory}")
    for name in ("Disc_Reward_Std", "Disc_Param_Grad_Norm", "Actor_Grad_Norm"):
        if not last[name] > 0:
            raise ValueError(f"Inactive {name} in {directory}")
    if not all((directory / file).stat().st_size for file in ("model.pt", "checkpoint.pt")):
        raise ValueError(f"Empty training artifact in {directory}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    queue, target = args.queue_dir.resolve(), args.out_dir.resolve()
    try:
        if target.exists():
            raise FileExistsError(target)
        manifest_path = queue / "cpl_source_sha256.json"
        frozen = queue / "cpl_source" if manifest_path.is_file() else freeze(queue)
        manifest = json.loads(manifest_path.read_text())
        verify(frozen, manifest)
        status(queue, "queued", after="logspace", updates=2000, envs=ENVS, seed=SEED,
               run_dir=str(target))
        phase = queue / "logspace.phase"
        first_pid = int((queue / "logspace_queue.pid").read_text())
        while True:
            current = phase.read_text().strip()
            if current == "complete":
                break
            if current not in ("pressure", "formal") or not Path(f"/proc/{first_pid}").exists():
                raise RuntimeError(f"Log-space baseline stopped in phase {current}")
            time.sleep(10)
        run(queue, "pressure", queue / "cpl_smoke", frozen, 3, manifest)
        run(queue, "formal", target, frozen, 2000, manifest)
        status(queue, "complete", run_dir=str(target))
    except BaseException as error:
        status(queue, "failed", error=repr(error))
        raise


if __name__ == "__main__":
    main()
