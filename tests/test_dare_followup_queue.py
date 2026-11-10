"""Queue safety tests use real tiny subprocesses and CPU-only fake checkpoints."""

import fcntl
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "tools/paper_benchmark/run_dare_followup_queue.py"
SPEC = importlib.util.spec_from_file_location("dare_followup_queue", SCRIPT)
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


def write(path, value):
    path.write_text(json.dumps(value))


def completed(run_dir, *, samples=200, resume=0):
    metadata = {"next_iter": 2, "sample_count": samples, "resume_count": resume}
    write(run_dir / "execution.json", {"status": "complete", **metadata})
    write(run_dir / "checkpoint.pt", {"trainer_state": metadata})


def fake_checkpoint(run_dir, iterations, samples):
    trainer = json.loads((run_dir / "checkpoint.pt").read_text())["trainer_state"]
    expected = {"next_iter": iterations, "sample_count": samples, "resume_count": 0}
    if trainer != expected:
        raise queue.QueueError("checkpoint budget mismatch")
    return expected


def make_job(root, job_id, order_path, *, failure=False, wrong_checkpoint=False):
    run_dir = root / job_id
    run_dir.mkdir()
    snapshot = run_dir / "source_snapshot"
    snapshot.mkdir()
    (snapshot / "model.py").write_text("# frozen test source\n")
    write(run_dir / "source_manifest.json", {"model.py": queue.sha256(snapshot / "model.py")})
    (run_dir / "frozen_agent.yaml").write_text("agent_name: Test\n")
    script = run_dir / "launch_train.py"
    script.write_text(
        "import json, os, pathlib, sys, time\n"
        f"out=pathlib.Path({str(run_dir)!r})\n"
        f"order=pathlib.Path({str(order_path)!r})\n"
        f"with order.open('a') as stream: stream.write({job_id!r}+' start\\n')\n"
        "(out/'train.pid').write_text(str(os.getpid()))\n"
        "(out/'execution.json').write_text(json.dumps({'status':'running'}))\n"
        "time.sleep(.025)\n"
        + ("(out/'execution.json').write_text(json.dumps({'status':'failed','error':'injected'}))\n"
           "sys.exit(7)\n" if failure else
           f"trainer={{'next_iter':2,'sample_count':{199 if wrong_checkpoint else 200},'resume_count':0}}\n"
           "(out/'checkpoint.pt').write_text(json.dumps({'trainer_state':trainer}))\n"
           "trainer['sample_count']=200\n"
           "(out/'execution.json').write_text(json.dumps({'status':'complete',**trainer}))\n"
           f"with order.open('a') as stream: stream.write({job_id!r}+' end\\n')\n")
    )
    job = {"id": job_id, "run_dir": str(run_dir), "ready_file": str(run_dir / "ready.json"),
           "command": [sys.executable, str(script)]}
    gate = {"status": "passed", "job_id": job_id,
            "source_manifest_sha256": queue.sha256(run_dir / "source_manifest.json"),
            "agent_config_sha256": queue.sha256(run_dir / "frozen_agent.yaml")}
    write(Path(job["ready_file"]), gate)
    return job


def campaign(tmp_path, *, bad_first=False, wrong_checkpoint=False):
    predecessor = tmp_path / "original"
    predecessor.mkdir()
    completed(predecessor)
    order = tmp_path / "order.txt"
    jobs = [make_job(tmp_path, name, order,
                     failure=bad_first and name == "cpl1",
                     wrong_checkpoint=wrong_checkpoint and name == "cpl1")
            for name in ("cpl1", "cpl7", "monotone")]
    plan = {"version": 1,
            "predecessor": {"run_dir": str(predecessor), "pid": 2147483647,
                            "proc_start_ticks": "1", "target_iterations": 2,
                            "target_samples": 200}, "jobs": jobs}
    write(tmp_path / "plan.json", plan)
    return queue.QueueRunner(tmp_path, poll_seconds=.005,
                             checkpoint_verifier=fake_checkpoint), plan, order


def test_real_subprocesses_run_in_strict_order_and_terminal_restart_is_noop(tmp_path):
    runner, plan, order = campaign(tmp_path)
    assert runner.run() == 0
    assert order.read_text().splitlines() == [
        "cpl1 start", "cpl1 end", "cpl7 start", "cpl7 end", "monotone start", "monotone end"]
    state = queue.read_json(tmp_path / "queue_state.json")
    assert state["status"] == "complete"
    assert all(job["status"] == "complete" for job in state["jobs"])
    assert all(job["return_code"] == 0 for job in state["jobs"])
    assert all(job["proc_start_ticks"].isdigit() for job in state["jobs"])
    before = order.read_text()
    second = queue.QueueRunner(tmp_path, poll_seconds=.005, checkpoint_verifier=fake_checkpoint)
    assert second.run() == 0
    assert order.read_text() == before


@pytest.mark.parametrize("kind", ["failure", "wrong_checkpoint"])
def test_failure_or_wrong_budget_blocks_every_later_job(tmp_path, kind):
    runner, plan, order = campaign(tmp_path, bad_first=kind == "failure",
                                   wrong_checkpoint=kind == "wrong_checkpoint")
    assert runner.run() == 1
    assert queue.read_json(tmp_path / "queue_state.json")["status"] == "failed"
    assert not (Path(plan["jobs"][1]["run_dir"]) / "console.log").exists()
    assert not (Path(plan["jobs"][2]["run_dir"]) / "console.log").exists()
    for child in runner.children.values():
        child.wait(timeout=5)
    before = order.read_text()
    assert runner.run() == 1
    assert order.read_text() == before


def test_live_predecessor_blocks_even_when_completion_files_already_exist(tmp_path, monkeypatch):
    runner, plan, order = campaign(tmp_path)
    runner._initialize()
    monkeypatch.setattr(queue, "same_process_alive", lambda pid, ticks: True)
    assert not runner.advance()
    assert not runner.state["predecessor_verified_complete"]
    assert not order.exists()
    monkeypatch.setattr(queue, "same_process_alive", lambda pid, ticks: False)
    assert not runner.advance()
    assert runner.state["predecessor_verified_complete"]


def test_predecessor_log_progress_is_not_final_checkpoint_completion(tmp_path):
    runner, plan, order = campaign(tmp_path)
    predecessor = Path(plan["predecessor"]["run_dir"])
    write(predecessor / "execution.json", {"status": "running", "next_iter": 2,
                                           "sample_count": 200})
    assert runner.run() == 1
    assert not order.exists()


def test_absent_or_pending_gate_waits_without_training_artifacts(tmp_path):
    runner, plan, order = campaign(tmp_path)
    ready = Path(plan["jobs"][0]["ready_file"])
    ready.unlink()
    runner._initialize()
    runner.advance()  # Verifies the predecessor.
    runner.advance()  # Waits for cpl1 readiness.
    assert runner.state["status"] == "waiting_ready"
    assert queue.read_json(tmp_path / "queue_state.json")["active_job"] == "cpl1"
    assert not order.exists()
    write(ready, {"status": "pending", "job_id": "cpl1"})
    runner.advance()
    assert runner.state["jobs"][0]["status"] == "pending"


@pytest.mark.parametrize("corruption", ["gate_hash", "source", "config", "job_id"])
def test_stale_or_wrong_readiness_gate_never_launches(tmp_path, corruption):
    runner, plan, order = campaign(tmp_path)
    job = plan["jobs"][0]
    run_dir = Path(job["run_dir"])
    if corruption in ("gate_hash", "job_id"):
        gate = queue.read_json(Path(job["ready_file"]))
        gate["source_manifest_sha256" if corruption == "gate_hash" else "job_id"] = "wrong"
        write(Path(job["ready_file"]), gate)
    elif corruption == "source":
        (run_dir / "source_snapshot/model.py").write_text("changed\n")
    else:
        (run_dir / "frozen_agent.yaml").write_text("agent_name: Changed\n")
    assert runner.run() == 1
    assert not order.exists()


def test_existing_artifacts_are_not_overwritten_or_resumed(tmp_path):
    runner, plan, order = campaign(tmp_path)
    model = Path(plan["jobs"][0]["run_dir"]) / "model.pt"
    model.write_text("irreplaceable old model")
    assert runner.run() == 1
    assert model.read_text() == "irreplaceable old model"
    assert not order.exists()


def test_exclusive_supervisor_lock_rejects_second_runner(tmp_path):
    runner, _, order = campaign(tmp_path)
    with (tmp_path / "queue.lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert runner.run() == 2
    assert not order.exists()
    assert not (tmp_path / "queue_state.json").exists()


def test_restart_reattaches_to_recorded_child_without_duplicate_launch(tmp_path):
    first, plan, order = campaign(tmp_path)
    first._initialize()
    first.advance()
    first.advance()  # Launch the actual first child and save its PID identity.
    first.children["cpl1"].wait(timeout=5)
    restarted = queue.QueueRunner(tmp_path, poll_seconds=.005,
                                  checkpoint_verifier=fake_checkpoint)
    assert restarted.run() == 0
    assert order.read_text().count("cpl1 start") == 1
    state = queue.read_json(tmp_path / "queue_state.json")
    assert state["jobs"][0]["exit_code_unavailable_after_supervisor_restart"]


def test_launch_without_persisted_pid_fails_closed(tmp_path):
    runner, plan, order = campaign(tmp_path)
    runner._initialize()
    runner.advance()
    runner.state["jobs"][0]["status"] = "launching"
    runner._save()
    assert runner.run() == 1
    assert not order.exists()


@pytest.mark.parametrize("failure", ["predecessor", "ready_gate"])
def test_explicit_failed_prerequisite_never_launches(tmp_path, failure):
    runner, plan, order = campaign(tmp_path)
    if failure == "predecessor":
        path = Path(plan["predecessor"]["run_dir"]) / "execution.json"
    else:
        path = Path(plan["jobs"][0]["ready_file"])
    write(path, {"status": "failed", "error": "injected"})
    assert runner.run() == 1
    assert not order.exists()


def test_process_identity_rejects_reused_pid_and_treats_zombie_as_exited(monkeypatch):
    monkeypatch.setattr(queue, "process_info", lambda pid: ("123", "R"))
    assert queue.same_process_alive(7, "123")
    assert not queue.same_process_alive(7, "124")
    monkeypatch.setattr(queue, "process_info", lambda pid: ("123", "Z"))
    assert not queue.same_process_alive(7, "123")


@pytest.mark.parametrize("change", [None, "next_iter", "resume_count"])
def test_real_torch_checkpoint_verification_reads_exact_cpu_budget(tmp_path, change):
    import torch

    trainer = {"next_iter": 2000, "sample_count": 524288000, "resume_count": 0}
    if change is not None:
        trainer[change] += 1
    torch.save({"trainer_state": trainer, "model_state_dict": {"x": torch.ones(2)}},
               tmp_path / "checkpoint.pt")
    if change is None:
        assert queue.verify_checkpoint(tmp_path, 2000, 524288000) == trainer
    else:
        with pytest.raises(queue.QueueError, match=change):
            queue.verify_checkpoint(tmp_path, 2000, 524288000)
