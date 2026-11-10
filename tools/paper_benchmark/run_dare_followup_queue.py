#!/usr/bin/env python3
"""Fail-closed, persistent, from-scratch DARE follow-up experiment queue.

The supervisor uses no CUDA. A job becomes eligible only after the preceding
process has exited, its launcher reports completion, its final checkpoint has
the exact sample budget, and its reviewed source/configuration gate passes.
There is deliberately no automatic resume or retry after a training failure.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Callable


class QueueError(RuntimeError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise QueueError(f"Expected JSON object: {path}")
    return value


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def process_info(pid: int) -> tuple[str, str] | None:
    """Return /proc start ticks and state, immune to spaces in process names."""
    try:
        tail = Path(f"/proc/{int(pid)}/stat").read_text().rsplit(")", 1)[1].split()
        return tail[19], tail[0]  # fields 22 (starttime), 3 (state).
    except (FileNotFoundError, ProcessLookupError):
        return None


def same_process_alive(pid: int, start_ticks: str) -> bool:
    actual = process_info(pid)
    return (actual is not None and actual[0] == str(start_ticks)
            and actual[1] not in ("Z", "X", "x"))


def verify_checkpoint(run_dir: Path, iterations: int, samples: int) -> dict:
    """Read the atomically saved final checkpoint on CPU, once process exits."""
    import torch

    checkpoint = torch.load(run_dir / "checkpoint.pt", map_location="cpu",
                            weights_only=False)
    trainer = checkpoint.get("trainer_state", {})
    expected = {"next_iter": iterations, "sample_count": samples, "resume_count": 0}
    for key, value in expected.items():
        if trainer.get(key) != value:
            raise QueueError(f"{run_dir}: checkpoint {key}={trainer.get(key)!r}; "
                             f"expected {value}")
    return expected


def validate_plan(plan: dict) -> None:
    if plan.get("version") != 1:
        raise QueueError("Unsupported queue plan version")
    predecessor = plan.get("predecessor", {})
    for key in ("run_dir", "pid", "proc_start_ticks", "target_iterations", "target_samples"):
        if key not in predecessor:
            raise QueueError(f"Missing predecessor field: {key}")
    if (not Path(predecessor["run_dir"]).is_absolute()
            or int(predecessor["pid"]) <= 0
            or not str(predecessor["proc_start_ticks"]).isdigit()
            or predecessor["target_iterations"] <= 0
            or predecessor["target_samples"] <= 0):
        raise QueueError("Invalid predecessor identity or training budget")
    jobs = plan.get("jobs", [])
    ids = [job.get("id") for job in jobs]
    if not jobs or len(set(ids)) != len(ids) or any(not name for name in ids):
        raise QueueError("Jobs require nonempty, unique IDs")
    run_dirs = [job.get("run_dir") for job in jobs]
    if len(set(run_dirs)) != len(run_dirs):
        raise QueueError("Jobs must use separate output directories")
    for job in jobs:
        if not Path(job["run_dir"]).is_absolute() or not Path(job["ready_file"]).is_absolute():
            raise QueueError("Job and readiness paths must be absolute")
        command = job.get("command")
        if (not isinstance(command, list) or not command
                or any(not isinstance(value, str) or not value for value in command)):
            raise QueueError("Each job command must be a nonempty argument list")
        if job["run_dir"] == predecessor["run_dir"]:
            raise QueueError("A follow-up job cannot overwrite the predecessor")


class QueueRunner:
    def __init__(self, campaign: Path, *, poll_seconds: float = 15,
                 checkpoint_verifier: Callable = verify_checkpoint):
        self.campaign = Path(campaign).resolve()
        self.poll_seconds = poll_seconds
        self.checkpoint_verifier = checkpoint_verifier
        self.plan_path = self.campaign / "plan.json"
        self.plan = read_json(self.plan_path)
        validate_plan(self.plan)
        self.plan_hash = sha256(self.plan_path)
        self.state_path = self.campaign / "queue_state.json"
        self.events_path = self.campaign / "events.jsonl"
        self.state: dict = {}
        self.children: dict[str, subprocess.Popen] = {}

    def _save(self) -> None:
        self.state["updated_at_unix"] = time.time()
        atomic_json(self.state_path, self.state)

    def _event(self, event: str, **details) -> None:
        row = {"time_unix": time.time(), "event": event, **details}
        with self.events_path.open("a") as stream:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        print(json.dumps(row, sort_keys=True), flush=True)

    def _initialize(self) -> None:
        if self.state_path.exists():
            self.state = read_json(self.state_path)
            if self.state.get("plan_sha256") != self.plan_hash:
                raise QueueError("Queue plan changed after supervisor initialization")
            return
        self.state = {"version": 1, "plan_sha256": self.plan_hash,
                      "status": "waiting_predecessor", "created_at_unix": time.time(),
                      "predecessor_verified_complete": False,
                      "jobs": [{"id": job["id"], "status": "pending"}
                               for job in self.plan["jobs"]]}
        self._save()
        self._event("queue_initialized", jobs=[job["id"] for job in self.plan["jobs"]])

    def _budget(self) -> tuple[int, int]:
        predecessor = self.plan["predecessor"]
        return predecessor["target_iterations"], predecessor["target_samples"]

    def _verify_complete(self, run_dir: Path) -> dict:
        execution = read_json(run_dir / "execution.json")
        iterations, samples = self._budget()
        if (execution.get("status") != "complete"
                or execution.get("next_iter") != iterations
                or execution.get("sample_count") != samples
                or execution.get("resume_count") != 0):
            raise QueueError(f"Missing exact successful completion state: {run_dir}")
        return self.checkpoint_verifier(run_dir, iterations, samples)

    def _wait_predecessor(self) -> bool:
        predecessor = self.plan["predecessor"]
        run_dir = Path(predecessor["run_dir"])
        execution_path = run_dir / "execution.json"
        execution = read_json(execution_path) if execution_path.exists() else {}
        if execution.get("status") == "failed":
            raise QueueError(f"Predecessor reported failure: {execution.get('error', '')}")
        if same_process_alive(predecessor["pid"], predecessor["proc_start_ticks"]):
            return False
        verified = self._verify_complete(run_dir)
        self.state["predecessor_verified_complete"] = True
        self.state["status"] = "waiting_ready"
        self._save()
        self._event("predecessor_complete", run_dir=str(run_dir), checkpoint=verified)
        return True

    def _ready(self, job: dict) -> bool:
        gate_path = Path(job["ready_file"])
        if not gate_path.exists():
            return False
        gate = read_json(gate_path)
        if gate.get("status") == "failed":
            raise QueueError(f"{job['id']}: readiness stress checks failed")
        if gate.get("status") != "passed":
            return False
        if gate.get("job_id") != job["id"]:
            raise QueueError(f"{job['id']}: readiness gate belongs to another job")
        run_dir = Path(job["run_dir"])
        for field, filename in (("source_manifest_sha256", "source_manifest.json"),
                                ("agent_config_sha256", "frozen_agent.yaml")):
            if not gate.get(field) or sha256(run_dir / filename) != gate[field]:
                raise QueueError(f"{job['id']}: readiness hash mismatch for {filename}")
        manifest = read_json(run_dir / "source_manifest.json")
        snapshot = (run_dir / "source_snapshot").resolve()
        if not manifest:
            raise QueueError(f"{job['id']}: empty source manifest")
        for name, expected in manifest.items():
            source = (snapshot / name).resolve()
            if not source.is_relative_to(snapshot) or sha256(source) != expected:
                raise QueueError(f"{job['id']}: frozen source mismatch: {name}")
        return True

    def _start(self, job: dict, record: dict) -> None:
        run_dir = Path(job["run_dir"])
        # A restart between Popen and recording its PID intentionally fails
        # closed. No from-scratch experiment may overwrite training artifacts.
        protected = ("checkpoint.pt", "model.pt", "all_iterations.jsonl",
                     "train_metrics.jsonl", "execution.json", "train.pid", "console.log")
        existing = [name for name in protected if (run_dir / name).exists()]
        if existing:
            raise QueueError(f"{job['id']}: refusing existing training artifacts: {existing}")
        record.update(status="launching", started_at_unix=time.time())
        self.state["status"] = "running"
        self._save()
        self._event("job_launching", job_id=job["id"], command=job["command"])
        with (run_dir / "console.log").open("x") as console:
            child = subprocess.Popen(job["command"], cwd=run_dir,
                                     stdout=console, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
        self.children[job["id"]] = child
        info = process_info(child.pid)
        # A very short child may already be a zombie: retain its true start
        # ticks if /proc is still present; Popen.poll provides the exit code.
        record.update(status="running", pid=child.pid,
                      proc_start_ticks=info[0] if info is not None else "")
        self._save()
        self._event("job_started", job_id=job["id"], pid=child.pid,
                    proc_start_ticks=record["proc_start_ticks"])

    def _monitor(self, job: dict, record: dict) -> bool:
        run_dir = Path(job["run_dir"])
        execution_path = run_dir / "execution.json"
        execution = read_json(execution_path) if execution_path.exists() else {}
        if execution.get("status") == "failed":
            raise QueueError(f"{job['id']}: launcher reported failure: {execution.get('error', '')}")
        child = self.children.get(job["id"])
        if child is not None:
            return_code = child.poll()
            if return_code is None:
                return False
            record["return_code"] = return_code
            if return_code != 0:
                raise QueueError(f"{job['id']}: training process exited {return_code}")
        else:
            if not record.get("pid") or not record.get("proc_start_ticks"):
                raise QueueError(f"{job['id']}: interrupted launch has no durable process identity")
            if same_process_alive(record["pid"], record["proc_start_ticks"]):
                return False
            record["exit_code_unavailable_after_supervisor_restart"] = True
        verified = self._verify_complete(run_dir)
        record.update(status="complete", finished_at_unix=time.time(), checkpoint=verified)
        self._save()
        self._event("job_complete", job_id=job["id"], checkpoint=verified)
        return True

    def advance(self) -> bool:
        """Advance once; return True only for a terminal campaign state."""
        if self.state["status"] in ("complete", "failed"):
            return True
        if not self.state["predecessor_verified_complete"]:
            self._wait_predecessor()
            return False
        for job, record in zip(self.plan["jobs"], self.state["jobs"]):
            if record["status"] == "complete":
                continue
            previous_active_job = self.state.get("active_job")
            self.state["active_job"] = job["id"]
            if record["status"] == "pending":
                if not self._ready(job):
                    if (self.state["status"] != "waiting_ready"
                            or previous_active_job != job["id"]):
                        self.state["status"] = "waiting_ready"
                        self._save()
                        self._event("waiting_ready", job_id=job["id"])
                    return False
                self._start(job, record)
            elif record["status"] == "running":
                self._monitor(job, record)
            else:
                raise QueueError(f"{job['id']}: interrupted or invalid status {record['status']!r}")
            return False
        self.state.update(status="complete", finished_at_unix=time.time(), active_job=None)
        self._save()
        self._event("queue_complete")
        return True

    def run(self) -> int:
        self.campaign.mkdir(parents=True, exist_ok=True)
        with (self.campaign / "queue.lock").open("a+") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print("Another supervisor owns this campaign lock", file=sys.stderr, flush=True)
                return 2
            try:
                self._initialize()
                while not self.advance():
                    time.sleep(self.poll_seconds)
                return 0 if self.state["status"] == "complete" else 1
            except Exception as error:
                if not self.state:
                    self.state = {"version": 1, "plan_sha256": self.plan_hash, "jobs": []}
                self.state.update(status="failed", failed_at_unix=time.time(), error=repr(error))
                self._save()
                self._event("queue_failed", error=repr(error),
                            active_job=self.state.get("active_job"))
                return 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args(argv)
    if args.poll_seconds <= 0:
        parser.error("--poll-seconds must be positive")
    try:
        return QueueRunner(args.campaign, poll_seconds=args.poll_seconds).run()
    except Exception as error:
        print(f"Queue initialization failed: {error!r}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
