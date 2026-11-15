#!/usr/bin/env python3
"""Run the three frozen, authorized MIRA experiments serially when CUDA is ready."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from datetime import datetime
from zoneinfo import ZoneInfo


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def validate_frozen(job):
    out = Path(job["out_dir"])
    manifest = read_json(out / "source_manifest.json")
    assert manifest["updates"] == 2000
    assert manifest["seed"] == 0 and manifest["num_envs"] == 8192
    assert manifest["max_samples"] == 524288000
    assert manifest["motion"] == job["motion"]
    for name, expected in manifest["files"].items():
        file = out / "source_snapshot" / name
        if hashlib.sha256(file.read_bytes()).hexdigest() != expected:
            raise ValueError(f"Frozen source changed: {file}")
    for asset in manifest["inputs"]["runtime_assets"].values():
        if hashlib.sha256(Path(asset["path"]).read_bytes()).hexdigest() != asset["sha256"]:
            raise ValueError(f"Motion/character asset changed: {asset['path']}")


def validate_completed(job):
    out = Path(job["out_dir"])
    status = read_json(out / "status.json")
    check = read_json(out / "checkpoint_check.json")
    assert status["status"] == "completed"
    assert status["updates"] == 2000 and status["samples"] == 524288000
    assert status["frozen_sources_unchanged"] is True
    assert check["status"] == "passed"
    assert check["saved_next_iter"] == 2000 and check["saved_samples"] == 524288000
    assert check["model_and_optimizer_state_reload"] is True
    assert all((out / name).is_file() for name in ("model.pt", "checkpoint.pt"))
    checks = [json.loads(line) for line in (out / "constraint_rounds.jsonl").read_text().splitlines()]
    assert {1, 2, 3}.issubset({r["completed_updates"] for r in checks if r["finite_checks_passed"]})
    validate_frozen(job)


def gpu_ready(python):
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.free", "--format=csv,noheader,nounits"],
            text=True, capture_output=True, timeout=15)
        if result.returncode:
            return False, (result.stderr.strip() or result.stdout.strip())[:1000]
        cuda = subprocess.run(
            [python, "-c", "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
             "x=torch.ones(1,device='cuda:0'); assert x.item()==1; print(torch.cuda.get_device_name(0))"],
            text=True, capture_output=True, timeout=60)
        if cuda.returncode:
            return False, (cuda.stderr.strip() or cuda.stdout.strip())[-1500:]
        return True, result.stdout.strip()
    except (OSError, subprocess.TimeoutExpired) as exc:
        return False, repr(exc)


def terminate_worker(child):
    if child.poll() is None:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=10)


def run_queue(queue_dir):
    queue_dir = Path(queue_dir).resolve()
    lock = (queue_dir / ".queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan = read_json(queue_dir / "plan.json")
    jobs = plan["jobs"]
    progress = {
        "status": "starting", "pid": os.getpid(), "started_at": now(),
        "updates_per_motion": 2000, "num_envs": 8192, "seed": 0, "GP": 0,
        "order": [job["motion"] for job in jobs], "completed_motions": [],
        "active_motion": None, "child_pid": None, "training_started": False,
    }

    def publish(status, **extra):
        progress.update(status=status, updated_at=now(), **extra)
        write_json(queue_dir / "queue_progress.json", progress)

    child = None
    try:
        for job in jobs:
            validate_frozen(job)
            out = Path(job["out_dir"])
            status = read_json(out / "status.json")
            if status["status"] == "completed":
                validate_completed(job)
                progress["completed_motions"].append(job["motion"])
                continue
            publish("waiting_for_gpu", next_motion=job["motion"], active_motion=None, child_pid=None)
            previous_reason = None
            while True:
                ready, reason = gpu_ready(plan["python"])
                publish("waiting_for_gpu" if not ready else "gpu_ready", gpu_diagnostic=reason)
                if ready:
                    break
                if reason != previous_reason:
                    print(f"{now()} WAITING_FOR_GPU: {reason}", flush=True)
                    previous_reason = reason
                time.sleep(30)

            runner = out / "source_snapshot/tools/paper_benchmark/run_dare_diagonal.py"
            command = [plan["python"], "-u", str(runner), "--motion", job["motion"],
                       "--updates", "2000", "--out-dir", str(out)]
            if (out / "checkpoint.pt").exists():
                command.append("--resume")
            elif status["status"] == "frozen":
                command.append("--frozen-worker")
            else:
                raise RuntimeError(f"Incomplete run without a recovery checkpoint: {out}")
            write_json(out / "launch.json", {"command": command, "started_at": now(),
                                              "queue_dir": str(queue_dir), "motion": job["motion"],
                                              "num_envs": 8192, "seed": 0, "GP": 0,
                                              "from_zero": "--resume" not in command})
            env = os.environ.copy()
            env.setdefault("OMP_NUM_THREADS", "4")
            env.setdefault("MKL_NUM_THREADS", "4")
            with (out / "console.log").open("a", buffering=1) as log:
                child = subprocess.Popen(command, cwd=plan["workspace_root"], env=env,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                publish("running", active_motion=job["motion"], child_pid=child.pid,
                        training_started=True, completed_updates=0)
                print(f"{now()} START {job['motion']} pid={child.pid}", flush=True)
                completed_at = None
                while child.poll() is None:
                    current = read_json(out / "status.json")
                    if current["status"] == "completed":
                        validate_completed(job)
                        completed_at = completed_at or time.monotonic()
                        if time.monotonic() - completed_at >= 45:
                            terminate_worker(child)
                            break
                    latest_file = out / "progress.json"
                    latest = read_json(latest_file) if latest_file.exists() else {}
                    publish("running", completed_updates=latest.get("completed_updates", 0))
                    time.sleep(5)
                if child.returncode != 0 and completed_at is None:
                    raise RuntimeError(f"{job['motion']} worker exited {child.returncode}; see {out / 'console.log'}")
            validate_completed(job)
            print(f"{now()} COMPLETED {job['motion']}", flush=True)
            child = None
            progress["completed_motions"].append(job["motion"])
            publish("between_jobs", active_motion=None, child_pid=None, completed_updates=0)
        publish("completed", active_motion=None, child_pid=None, next_motion=None,
                finished_at=now())
        print(f"{now()} ALL_THREE_MOTIONS_COMPLETED", flush=True)
    except BaseException as exc:
        if child is not None:
            terminate_worker(child)
        publish("stopped" if isinstance(exc, KeyboardInterrupt) else "failed",
                error=repr(exc), traceback=traceback.format_exc(), active_motion=None, child_pid=None)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queue-dir", type=Path, required=True)
    args = parser.parse_args()

    def stop(signum, frame):
        raise KeyboardInterrupt(f"Queue interrupted by signal {signum}")

    signal.signal(signal.SIGTERM, stop)
    run_queue(args.queue_dir)


if __name__ == "__main__":
    main()
