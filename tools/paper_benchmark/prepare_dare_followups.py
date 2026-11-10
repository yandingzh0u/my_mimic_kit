#!/usr/bin/env python3
"""Freeze the explicitly authorized CPL1 -> CPL7 -> monotone campaign."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
BENCHMARK = ROOT / "output/paper_benchmark"
PREDECESSOR = BENCHMARK / "dare_body_split_sn_noscale_backflip_2000_8192_seed0_20261109"
CAMPAIGN = BENCHMARK / "dare_cpl_then_monotone_backflip_queue_20261110"
JOBS = {
    "cpl1": ("dare_body_split_cpl1_backflip_2000_8192_seed0_20261110", "dare_body_split_cpl1_humanoid_agent.yaml"),
    "cpl7": ("dare_body_split_cpl7_backflip_2000_8192_seed0_20261110", "dare_body_split_cpl7_humanoid_agent.yaml"),
    "monotone": ("dare_monotone7_backflip_2000_8192_seed0_20261110", "dare_monotone_humanoid_agent.yaml"),
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare(job_id):
    name, agent_file = JOBS[job_id]
    out = BENCHMARK / name
    out.mkdir(exist_ok=True)
    if any((out / item).exists() for item in ("source_snapshot", "source_manifest.json", "launch_audit.json",
            "frozen_agent.yaml", "runtime_assets", "checkpoint.pt", "execution.json", "console.log")):
        raise RuntimeError("Refusing to overwrite frozen run: " + str(out))
    baseline_agent = yaml.safe_load((PREDECESSOR / "frozen_agent.yaml").read_text())
    agent = yaml.safe_load((ROOT / "data/agents" / agent_file).read_text())
    expected = copy.deepcopy(baseline_agent)
    expected["agent_name"] = "DAREMonotone" if job_id == "monotone" else "DAREBodySplitCPL"
    if job_id != "monotone":
        expected["model"]["disc_cpl_depth"] = 1 if job_id == "cpl1" else 7
    if agent != expected:
        raise RuntimeError("Configuration changed beyond requested discriminator variant")
    if (agent["steps_per_iter"] != 32 or agent["phase_sampler"] != "none"
            or agent["disc_grad_penalty"] != 0 or agent.get("disc_group_negatives", False)
            or "disc_reward_scale" in agent or agent.get("task_reward_weight", 0) != 0):
        raise RuntimeError("Invalid PPO/reward/sampling configuration")
    env = yaml.safe_load((PREDECESSOR / "frozen_env.yaml").read_text())
    if env["motion_name"] != "backflip" or env.get("reference_centered_actions", False):
        raise RuntimeError("Unexpected action or motion configuration")
    runtime_assets = {}
    for name in (env["char_file"], str(Path(env["char_file"]).with_suffix(".usd")), env["motion_file"]):
        source = ROOT / name
        dest = out / "runtime_assets" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        runtime_assets[name] = sha(dest)
    # The archived XML and adjacent USD are authoritative at runtime; future
    # edits to live character/motion assets cannot change a queued experiment.
    env["char_file"] = str(out / "runtime_assets" / env["char_file"])
    env["motion_file"] = str(out / "runtime_assets" / env["motion_file"])
    (out / "frozen_env.yaml").write_text(yaml.safe_dump(env, sort_keys=False))
    shutil.copyfile(PREDECESSOR / "frozen_engine.yaml", out / "frozen_engine.yaml")
    shutil.copyfile(ROOT / "data/agents" / agent_file, out / "frozen_agent.yaml")
    shutil.copyfile(Path(__file__).with_name("train_dare_followup.py"), out / "launch_train.py")
    shutil.copyfile(Path(__file__), out / "prepare_run.py")
    snapshot = out / "source_snapshot"
    shutil.copytree(ROOT / "mimickit", snapshot / "mimickit",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for source in (ROOT / "tools/paper_eval/evaluate_checkpoint.py",
                   ROOT / "tests/test_dare_body_split_cpl.py", ROOT / "tests/test_dare_monotone.py"):
        dest = snapshot / source.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    manifest = {str(p.relative_to(snapshot)): sha(p) for p in sorted(snapshot.rglob("*")) if p.is_file()}
    write_json(out / "source_manifest.json", manifest)
    write_json(out / "runtime_asset_manifest.json", runtime_assets)
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    write_json(out / "launch_audit.json", {
        "job_id": job_id, "variant": "monotone" if job_id == "monotone" else "cpl",
        "cpl_depth": 0 if job_id == "monotone" else int(job_id[-1]),
        "repository_root": str(ROOT), "prepared_at_unix": time.time(), "git_head": revision,
        "runtime_snapshot_is_authoritative": True, "from_scratch": True,
        "baseline_directory": str(PREDECESSOR), "num_envs": 8192, "rounds": 2000,
        "steps_per_round": 32, "samples": 524288000, "seed": 0,
        "projection_head_constraint": "positive_SN" if job_id == "monotone" else "none",
        "cpl_internal_constraint": "none" if job_id == "monotone" else "SN",
        "reward_multiplier": "none", "logit_calibration": "none", "disc_grad_penalty": 0,
        "action_interface": "absolute_pd", "phase_sampler": "none", "new_loss_weights": [],
        "reward": "original_ADD_single_monotone_score" if job_id == "monotone" else
                  "(map(root)+(map(body_pos)+map(body_rot))/2+map(velocity))/3",
        "comparison_caveat": "CPL1/7 remove projection/head SN as explicitly requested; they are not globally Lipschitz-controlled networks.",
        "source_file_count": len(manifest), "launcher_sha256": sha(out / "launch_train.py"),
        "frozen_config_sha256": {kind: sha(out / ("frozen_" + kind + ".yaml"))
                                 for kind in ("agent", "env", "engine")},
        "archived_runtime_assets_sha256": runtime_assets,
    })
    print("PREPARED", job_id, out, flush=True)


def initialize():
    CAMPAIGN.mkdir(exist_ok=True)
    plan_path = CAMPAIGN / "plan.json"
    if plan_path.exists():
        raise RuntimeError("Campaign already initialized")
    state = json.loads((PREDECESSOR / "execution.json").read_text())
    pid = state["pid"]
    stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    plan = {"version": 1, "predecessor": {"run_dir": str(PREDECESSOR), "pid": pid,
            "proc_start_ticks": stat[19], "target_iterations": 2000, "target_samples": 524288000},
            "jobs": [{"id": job_id, "run_dir": str(BENCHMARK / name),
                      "ready_file": str(CAMPAIGN / (job_id + "_ready.json")),
                      "command": [sys.executable, "-u", str(BENCHMARK / name / "launch_train.py")],
                      "target_iterations": 2000, "target_samples": 524288000}
                     for job_id, (name, _) in JOBS.items()]}
    shutil.copyfile(Path(__file__).with_name("run_dare_followup_queue.py"), CAMPAIGN / "run_queue.py")
    write_json(plan_path, plan)
    print("CAMPAIGN_INITIALIZED", CAMPAIGN, flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--job", choices=JOBS)
    parser.add_argument("--initialize", action="store_true")
    args = parser.parse_args()
    if args.initialize:
        initialize()
    if args.job:
        prepare(args.job)
    if not args.initialize and not args.job:
        parser.error("Specify --initialize and/or --job")


if __name__ == "__main__":
    main()
