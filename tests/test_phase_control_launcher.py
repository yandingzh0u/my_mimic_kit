"""Guard against accidental warm starts, mismatched budgets and parallel jobs."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location("phase_control_launcher",ROOT/"tools/paper_benchmark/run_phase_supported_control.py")
launcher=importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def test_argument_paths_and_from_scratch_budget(tmp_path):
    snapshot=tmp_path/"source_snapshot"
    output=tmp_path/"output"
    for _,agent in launcher.RUNS:
        args=launcher.launch_arguments(snapshot,output,agent)
        assert args["max_samples"]==3000*8192*32
        assert args["num_envs"]==8192 and args["rand_seed"]==0
        assert "resume_file" not in args and "model_file" not in args
        assert args["agent_config"]==str(snapshot/agent)
        assert args["out_dir"]==str(output)


def test_fingerprint_is_deterministic_covers_values_and_preserves_rng():
    state={"vector":torch.tensor([1.,2.,3.]),"scalar":torch.tensor(0)}
    before=torch.get_rng_state()
    expected=launcher.initial_fingerprint(state)
    assert launcher.initial_fingerprint(dict(reversed(list(state.items()))))==expected
    assert torch.equal(torch.get_rng_state(),before)
    state["scalar"]+=1
    assert launcher.initial_fingerprint(state)!=expected


@pytest.mark.parametrize("iteration,samples",[(2900,760479744),(2999,786431999)])
def test_incomplete_metrics_never_count_as_completed(tmp_path,iteration,samples):
    (tmp_path/"train_metrics.jsonl").write_text(json.dumps(dict(iteration=iteration,samples=samples))+"\n")
    with pytest.raises(RuntimeError,match="Incomplete"):
        launcher.final_metrics(tmp_path)


def test_final_checkpoint_must_match_budget_and_be_from_scratch(tmp_path):
    (tmp_path/"train_metrics.jsonl").write_text(json.dumps(dict(iteration=2999,samples=786432000))+"\n")
    torch.save({},tmp_path/"model.pt")
    for state in [dict(next_iter=2999,sample_count=786432000,resume_count=0),
                  dict(next_iter=3000,sample_count=786432000,resume_count=1)]:
        torch.save(dict(trainer_state=state),tmp_path/"checkpoint.pt")
        with pytest.raises(RuntimeError): launcher.validate_checkpoint(tmp_path)
    torch.save(dict(trainer_state=dict(next_iter=3000,sample_count=786432000,resume_count=0)),tmp_path/"checkpoint.pt")
    assert launcher.validate_checkpoint(tmp_path)["samples"]==786432000


def _campaign(tmp_path):
    campaign=tmp_path/"campaign";campaign.mkdir()
    (tmp_path/"output/paper_benchmark").mkdir(parents=True)
    jobs=[]
    for name in ["first","second"]:
        output=tmp_path/name;output.mkdir()
        jobs.append(dict(name=name,output=str(output),command=[name],state="queued"))
    launcher.write_json(campaign/"manifest.json",dict(repo=str(tmp_path),snapshot=str(tmp_path/"snapshot"),
        source_sha256={},asset_sha256={},jobs=jobs))
    return campaign


def test_second_job_waits_for_validated_first_checkpoint(tmp_path):
    campaign=_campaign(tmp_path);events=[]
    def launch(command,**kwargs):
        events.append("launch:"+command[0])
        return SimpleNamespace(pid=100+len(events),wait=lambda:0)
    def validate(output):
        events.append("validate:"+output.name)
        return dict(iteration=2999,samples=786432000)
    with mock.patch.object(launcher.subprocess,"Popen",side_effect=launch),mock.patch.object(launcher,"validate_checkpoint",side_effect=validate):
        launcher.worker(campaign)
    assert events==["launch:first","validate:first","launch:second","validate:second"]
    assert json.loads((campaign/"status.json").read_text())["state"]=="complete"


def test_failure_stops_queue_instead_of_starting_second_job(tmp_path):
    campaign=_campaign(tmp_path)
    with mock.patch.object(launcher.subprocess,"Popen",return_value=SimpleNamespace(pid=123,wait=lambda:1)) as launch:
        with pytest.raises(RuntimeError,match="exited"):
            launcher.worker(campaign)
    assert launch.call_count==1
    state=json.loads((campaign/"status.json").read_text())
    assert state["state"]==state["jobs"][0]["state"]=="failed"
    assert state["jobs"][1]["state"]=="queued"
