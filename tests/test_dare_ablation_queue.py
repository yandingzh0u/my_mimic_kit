"""Exercise queue safety with tiny subprocess workers; no GPU or Isaac imports."""
import fcntl
import importlib.util
import json
from pathlib import Path
import sys
import time

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "dare_ablation_queue", ROOT / "tools/paper_benchmark/run_dare_ablation_queue.py")
queue = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(queue)


FAKE_RUNNER = r'''
import argparse
import json
import os
from pathlib import Path
import sys
import time

parser = argparse.ArgumentParser()
parser.add_argument('--variant')
parser.add_argument('--updates', type=int)
parser.add_argument('--out-dir', type=Path)
args = parser.parse_args()
out = args.out_dir
out.mkdir(parents=True, exist_ok=True)
with Path(os.environ['FAKE_QUEUE_EVENT_LOG']).open('a') as stream:
    stream.write(json.dumps({'variant': args.variant, 'updates': args.updates,
                             'source': __file__, 'pid': os.getpid()}) + '\n')
if os.environ.get('FAKE_QUEUE_FAIL') == args.variant:
    (out / 'status.json').write_text(json.dumps({'status': 'failed', 'error': 'deliberate failure'}))
    sys.exit(17)
samples = args.updates * 8192 * 32
(out / 'model.pt').write_bytes(b'model')
(out / 'checkpoint.pt').write_bytes(b'checkpoint')
(out / 'checkpoint_check.json').write_text(json.dumps({
    'status': 'passed', 'saved_next_iter': args.updates, 'saved_samples': samples,
    'model_and_optimizer_state_reload': True}))
(out / 'progress.json').write_text(json.dumps({'completed_updates': args.updates}))
(out / 'constraint_rounds.jsonl').write_text(''.join(
    json.dumps({'completed_updates': i, 'finite_checks_passed': True}) + '\n'
    for i in (1, 2, 3)))
status = out / 'status.json'
temporary = out / 'status.tmp'
temporary.write_text(json.dumps({'status': 'completed', 'updates': args.updates,
                                 'samples': samples, 'frozen_sources_unchanged': True}))
temporary.replace(status)
if os.environ.get('FAKE_QUEUE_HANG') == '1':
    time.sleep(30)
'''


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    for relative in ("mimickit", "tools/paper_benchmark", "data/agents", "data/envs", "data/engines"):
        (root / relative).mkdir(parents=True)
    (root / queue.RUNNER_REL).write_text(FAKE_RUNNER)
    (root / queue.QUEUE_REL).write_text("# frozen queue source\n")
    (root / "mimickit/model.py").write_text("SCHEMA = 'original'\n")
    events = tmp_path / "events.jsonl"
    monkeypatch.setenv("FAKE_QUEUE_EVENT_LOG", str(events))
    return root, tmp_path / "experiments", events


def events_at(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def run(root, out, phase="all"):
    return queue.run_queue(out, root, phase, sys.executable,
                           poll_seconds=.01, grace_seconds=.1)


def test_all_smokes_precede_formal_and_completed_jobs_are_reused(workspace):
    root, out, events = workspace
    result = run(root, out)
    jobs = events_at(events)
    assert [(x["updates"], x["variant"]) for x in jobs] == [
        (3, "no_a"), (3, "no_b"), (3, "no_ab"),
        (2000, "no_a"), (2000, "no_b"), (2000, "no_ab")]
    assert result["status"] == "completed"
    assert len(result["jobs"]) == 6
    assert all(Path(x["source"]).is_relative_to(out / "source_snapshot") for x in jobs)
    again = run(root, out)
    assert events_at(events) == jobs
    assert all(x["reused_completed_result"] for x in again["jobs"])


def test_failed_smoke_stops_before_remaining_smoke_and_all_formal(workspace, monkeypatch):
    root, out, events = workspace
    monkeypatch.setenv("FAKE_QUEUE_FAIL", "no_b")
    with pytest.raises(RuntimeError, match="Experiment failed"):
        run(root, out)
    assert [(x["updates"], x["variant"]) for x in events_at(events)] == [(3, "no_a"), (3, "no_b")]
    assert not (out / "formal").exists()
    assert queue.read_json(out / "queue_progress.json")["status"] == "failed"


def test_formal_requires_every_smoke(workspace):
    root, out, events = workspace
    with pytest.raises(RuntimeError, match="Every smoke run must pass"):
        run(root, out, "formal")
    assert not events.exists()


def test_formal_uses_original_snapshot_after_workspace_changes(workspace):
    root, out, events = workspace
    run(root, out, "smoke")
    (root / queue.RUNNER_REL).write_text("raise RuntimeError('mutable workspace must not execute')\n")
    (root / "mimickit/model.py").write_text("SCHEMA = 'changed'\n")
    result = run(root, out, "formal")
    assert result["status"] == "completed"
    assert len(events_at(events)) == 6
    assert (out / "source_snapshot/mimickit/model.py").read_text() == "SCHEMA = 'original'\n"


def test_tampered_snapshot_is_rejected(workspace):
    root, out, events = workspace
    run(root, out, "smoke")
    (out / "source_snapshot/mimickit/model.py").write_text("SCHEMA = 'tampered'\n")
    with pytest.raises(ValueError, match="Frozen queue source changed"):
        run(root, out, "formal")
    assert len(events_at(events)) == 3


def test_hung_completed_workers_are_cleaned_before_next_launch(workspace, monkeypatch):
    root, out, events = workspace
    monkeypatch.setenv("FAKE_QUEUE_HANG", "1")
    started = time.monotonic()
    result = run(root, out, "smoke")
    assert time.monotonic() - started < 10
    assert all(job["cleanup"] == "terminated_after_completed" for job in result["jobs"])
    assert all(queue.process_identity(row["pid"]) is None for row in events_at(events))


def test_completed_marker_without_checkpoint_or_pressure_proof_is_rejected(workspace):
    root, out, events = workspace
    run(root, out, "smoke")
    directory = out / "smoke/no_a"
    check = queue.read_json(directory / "checkpoint_check.json")
    check["saved_next_iter"] = 2
    queue.write_json(directory / "checkpoint_check.json", check)
    with pytest.raises(ValueError, match="matching checkpoint check"):
        queue.validate_completed(directory, 3)
    check["saved_next_iter"] = 3
    queue.write_json(directory / "checkpoint_check.json", check)
    (directory / "constraint_rounds.jsonl").write_text(json.dumps(
        {"completed_updates": 1, "finite_checks_passed": True}) + "\n")
    with pytest.raises(ValueError, match="First three pressure checks"):
        queue.validate_completed(directory, 3)
    assert len(events_at(events)) == 3


def test_incomplete_output_is_preserved(workspace):
    root, out, events = workspace
    directory = out / "smoke/no_a"
    directory.mkdir(parents=True)
    (directory / "stdout.log").write_text("evidence from interrupted run\n")
    with pytest.raises(RuntimeError, match="Incomplete experiment already exists"):
        run(root, out)
    assert (directory / "stdout.log").read_text() == "evidence from interrupted run\n"
    assert not events.exists()


def test_exclusive_queue_lock_prevents_second_launcher(workspace):
    root, out, events = workspace
    out.mkdir()
    with (out / ".queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="already owns"):
            run(root, out)
    assert not events.exists()
