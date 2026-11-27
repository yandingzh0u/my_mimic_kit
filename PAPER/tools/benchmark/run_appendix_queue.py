#!/usr/bin/env python3
"""Run the single-seed appendix or Run benchmark with a full-size smoke gate."""

import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

import yaml


ROOT = Path(__file__).resolve().parents[3]
OUTPUT = ROOT / "output"
ENVS, STEPS, ROUNDS, SMOKE_ROUNDS = 8192, 32, 2000, 5
NONFINITE_UPDATE = re.compile(
    r"\|\s*(?:Actor_Loss|Critic_Loss|Disc_Loss|Adv_Std|Actor_Grad_Norm)\s*\|\s*[+-]?(?:nan|inf)\s*\|", re.I)
AGENTS = {
    "amp": "amp_humanoid_agent.yaml",
    "add": "add_humanoid_agent.yaml",
    "deepmimic": "deepmimic_humanoid_ppo_agent.yaml",
}
INTERACTIONS = (
    ("soccer_cmu_10_02", "mira_cmnn64_group_rms_soccer_cmu_10_02_2000_8192_seed0_20261122"),
    ("sofa003", "mira_cmnn64_group_rms_sofa003_2000_8192_seed0_20261122"),
    ("carry_box", "mira_cmnn68_group_rms_carry_box_2000_8192_seed0_20261122"),
)


def read_yaml(path):
    return yaml.safe_load(path.read_text())


def write_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)


def matrix(tag):
    amp_frames = read_yaml(ROOT / "PAPER/configs/envs/amp_backflip_env.yaml")["num_disc_obs_steps"]
    for task, name in INTERACTIONS:
        source = OUTPUT / name
        engine = read_yaml(source / "engine_config.yaml")
        mira_agent = read_yaml(source / "agent_config.yaml")
        for method, filename in AGENTS.items():
            env = read_yaml(source / "env_config.yaml")
            agent = read_yaml(ROOT / "data/agents" / filename)
            # Match per-task PPO timing (notably the 60 Hz CMU task).
            for key in ("discount", "td_lambda", "test_episodes", "iters_per_output"):
                agent[key] = mira_agent[key]
            if method == "amp":
                env.update(disc_observation="amp", num_disc_obs_steps=amp_frames)
                agent["disc_eval_batch_size"] = 16384
                # Wide, goal-conditioned observations made the original
                # unclipped SGD critic overflow, before PhysX saw invalid actions.
                agent["critic_optimizer"]["grad_clip"] = 1.0
            if method == "deepmimic":
                env["enable_tracking_reward"] = True
            yield f"{method}_{task}_2k_8192_seed0_{tag}", method, env, engine, agent


def run_matrix(tag):
    configs = ROOT / "PAPER/configs/envs"
    engine = read_yaml(ROOT / "data/engines/isaac_lab_engine.yaml")
    agents = dict(AGENTS, mira="mira_monotone_humanoid_agent.yaml")
    reference = read_yaml(configs / "add_run_env.yaml")
    for method in ("mira", "add", "deepmimic", "amp"):
        env = read_yaml(configs / f"{'add' if method == 'mira' else method}_run_env.yaml")
        # Match the actor's reference lookahead across rewards, including AMP.
        # AMP retains its native transition history and discriminator loss.
        for key in ("enable_tar_obs", "tar_obs_steps"):
            env[key] = reference[key]
        agent = read_yaml(ROOT / "data/agents" / agents[method])
        if method == "amp":
            # A looping style demo covers one cycle; anchor each transition
            # to its final root so absolute forward travel cannot separate classes.
            env["disc_track_global_root"] = False
            agent["disc_eval_batch_size"] = 16384
        name = (f"mira_cmnn64_group_rms_run_2000_8192_seed0_{tag}" if method == "mira"
                else f"{method}_run_2k_8192_seed0_{tag}")
        yield name, method, env, engine, agent


def evaluate_run(queue, name, method):
    destination = queue / "evaluation" / method
    destination.mkdir(parents=True)
    run_dir = OUTPUT / name
    command = [
        sys.executable, "-u", "PAPER/tools/evaluation/evaluate_checkpoint.py",
        "--model-file", str(run_dir / "model.pt"),
        "--env-config", str(run_dir / "env_config.yaml"),
        "--engine-config", str(run_dir / "engine_config.yaml"),
        "--agent-config", str(run_dir / "agent_config.yaml"),
        "--motion", "run", "--method", method, "--steps", "60",
        "--num-envs", "256", "--start-mode", "phase0", "--seed", "0",
        "--out-dir", str(destination),
    ]
    with (destination / "console.log").open("x") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                       env=dict(os.environ, ACCEPT_EULA="Y", OMNI_KIT_DISABLE_TELEMETRY="1"))
    import numpy as np

    with np.load(destination / "timeseries.npz") as trace:
        alive, errors = trace["alive"], trace["tracking_error"]
        names = trace["tracking_error_names"][:7].tolist()
        # An early cutoff should remain visible; never pad an incomplete motion.
        complete = alive.shape == (60, 256) and bool(alive.all())
        return dict(evaluated_steps=int(alive.shape[0]), min_executed_steps=int(alive.sum(0).min()),
                    mean_full_window_errors=dict(zip(names, errors[..., :7].mean((0, 1), dtype=np.float64)))
                    if complete else None)


def run(job, destination, rounds, smoke, resume=False, reserve_for_physx=False):
    console = destination / "console.log"
    if resume:
        old_pid = int((destination / "train.pid").read_text())
        old_cmd = Path(f"/proc/{old_pid}/cmdline")
        if old_cmd.exists() and str(destination).encode() in old_cmd.read_bytes():
            raise RuntimeError(f"Training process {old_pid} is already using {destination}")
        console = destination / f"resume_console_{time.strftime('%Y%m%d_%H%M%S')}.log"
    else:
        destination.mkdir(parents=True, exist_ok=False)
    command = [sys.executable, "-u", "mimickit/run.py", "--mode", "train", "--num_envs", str(ENVS),
               "--devices", "cuda:0", "--master_port", "6498", "--rand_seed", "0", "--visualize", "false",
               "--logger", "txt", "--max_samples", str(rounds * ENVS * STEPS), "--out_dir", str(destination)]
    for field in ("env", "engine", "agent"):
        command.extend((f"--{field}_config", str(job / f"{field}.yaml")))
    if not smoke:
        command.extend(("--save_int_models", "true"))
    if resume:
        command.extend(("--resume_file", str(destination / "checkpoint.pt")))
    environment = dict(os.environ, PYTHONPATH=str(ROOT / "mimickit"), ACCEPT_EULA="Y", OMNI_KIT_DISABLE_TELEMETRY="1",
                       PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4")
    if reserve_for_physx:
        # Each chunk is two time steps for all environments. Only inference
        # workspace and cached allocations change; rollout/training batches don't.
        environment["MIMICKIT_CRITIC_EVAL_TIME_BATCH"] = "2"
        environment["PYTORCH_CUDA_ALLOC_CONF"] = "garbage_collection_threshold:0.55"
    start = time.monotonic()
    peak_mib = 0
    with console.open("x") as log:
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=log, stderr=subprocess.STDOUT)
        (destination / "train.pid").write_text(f"{child.pid}\n")
        with console.open(errors="replace") as reader:
            while True:
                finished = child.poll() is not None
                for line in reader.readlines():
                    if ("the simulation will miss interactions" in line or "CUDA out of memory" in line
                            or "[omni.physx.plugin] PhysX error" in line or NONFINITE_UPDATE.search(line)):
                        child.terminate()
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            child.kill()
                            child.wait()
                        raise RuntimeError(f"{destination}: {line.strip()}")
                if finished:
                    break
                if smoke:
                    smi = subprocess.check_output(["nvidia-smi", "-i", "0", "--query-gpu=memory.used",
                                                   "--format=csv,noheader,nounits"], text=True)
                    peak_mib = max(peak_mib, int(smi.strip()))
                time.sleep(1 if smoke else 3)
        rc = child.wait()
    if rc:
        raise RuntimeError(f"trainer exited with code {rc}; see {console}")
    rows = [json.loads(line) for line in (destination / "train_metrics.jsonl").read_text().splitlines()]
    if rows[-1]["Iteration"] != rounds - 1 or rows[-1]["Samples"] != rounds * ENVS * STEPS:
        raise ValueError(f"{destination}: missing exact final sample budget")
    if not all((destination / name).stat().st_size > 0 for name in ("checkpoint.pt", "model.pt", "log.txt")):
        raise ValueError(f"{destination}: empty training artifact")
    return rows, peak_mib, time.monotonic() - start


def smoke_report(method, rows, peak_mib, elapsed, check_tracking=False):
    required = ("Actor_Grad_Norm", "Adv_Std", "Critic_Loss", "Test_Episode_Length_Start0", "Peak_GPU_Memory_MB")
    if check_tracking:
        required += ("Root_Pos_Err", "Root_Rot_Err", "Body_Pos_Err", "Body_Rot_Err",
                     "Dof_Vel_Err", "Root_Vel_Err", "Root_Ang_Vel_Err")
    if method != "deepmimic":
        required += ("Disc_Loss", "Disc_Reward_Mean", "Disc_Reward_Std", "Disc_Param_Grad_Norm", "Disc_Grad_Penalty")
    else:
        required += ("Test_Return",)
    for row in rows:
        if not all(math.isfinite(row[key]) for key in required):
            raise ValueError(f"Non-finite objective or update statistic: {method}")
        active = ("Actor_Grad_Norm", "Adv_Std", "Test_Episode_Length_Start0")
        if method == "deepmimic":
            active += ("Test_Return",)
        else:
            active += ("Disc_Reward_Mean", "Disc_Reward_Std", "Disc_Param_Grad_Norm")
            if (row["Disc_Grad_Penalty"] == 0) != (method == "mira"):
                raise ValueError(f"Unexpected gradient penalty: {method}")
        if not all(row[key] > 0 for key in active):
            raise ValueError(f"Inactive update or reward: {method}: { {key: row[key] for key in active} }")
    return {"status": "passed", "envs": ENVS, "updates": SMOKE_ROUNDS, "elapsed_seconds": elapsed,
            "peak_device_memory_mib": peak_mib,
            "final": {key: rows[-1][key] for key in ("Iteration", "Samples", *required)}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("appendix", "run"), default="appendix")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--queue-dir", required=True, type=Path)
    parser.add_argument("--smoke-first", default="", help="Exact run name to pressure-test first")
    parser.add_argument("--reuse-smoke", type=Path, help="Reuse passed reports with identical frozen configs")
    parser.add_argument("--continue-from", type=Path, help="Continue a stopped formal queue without repeating finished jobs")
    args = parser.parse_args()
    queue = args.queue_dir.resolve()
    queue.mkdir(parents=True, exist_ok=True)
    plan = list((run_matrix if args.suite == "run" else matrix)(args.tag))
    ordered_names = [name for name, *_ in plan]
    if args.smoke_first and args.smoke_first not in ordered_names:
        parser.error("--smoke-first must be an exact run name in this matrix")
    completed = []
    interrupted = None
    source = args.continue_from.resolve() if args.continue_from else None
    if source:
        prior = json.loads((source / "status.json").read_text())
        if prior["order"] != ordered_names or prior.get("failed_phase", prior["phase"]) != "formal":
            raise ValueError("Continuation requires the same stopped formal queue")
        prior_cmdline = Path(f"/proc/{int(prior['pid'])}/cmdline")
        if prior_cmdline.exists() and b"run_appendix_queue.py" in prior_cmdline.read_bytes():
            raise RuntimeError(f"Prior queue process {prior['pid']} is still running")
        completed = list(prior["completed"])
        interrupted = prior["current"]
        if completed + [interrupted] != ordered_names[:len(completed) + 1]:
            raise ValueError("Prior results are not an ordered prefix of this queue")
    for name in ordered_names:
        existing = OUTPUT / name
        if name in completed:
            final = json.loads((existing / "train_metrics.jsonl").read_text().splitlines()[-1])
            if (final["Iteration"], final["Samples"]) != (ROUNDS - 1, ROUNDS * ENVS * STEPS):
                raise ValueError(f"Prior completed run has no terminal metric: {name}")
        elif name == interrupted:
            if not (existing / "checkpoint.pt").is_file():
                raise FileNotFoundError(existing / "checkpoint.pt")
        elif existing.exists():
            raise FileExistsError(existing)

    jobs = queue / "configs"
    jobs.mkdir(exist_ok=False)
    for name, _, env, engine, agent in plan:
        job = jobs / name
        job.mkdir()
        if agent["steps_per_iter"] != STEPS:
            raise ValueError(f"Unexpected steps per update: {name}")
        for field, config in (("env", env), ("engine", engine), ("agent", agent)):
            filename = f"{field}.yaml"
            if source:
                (job / filename).write_bytes((source / "configs" / name / filename).read_bytes())
            else:
                (job / filename).write_text(yaml.safe_dump(config, sort_keys=False))

    state = {"pid": os.getpid(), "order": ordered_names, "smoke_passed": [], "completed": completed}
    smoke_source = source or args.reuse_smoke
    if smoke_source:
        smoke_source = smoke_source.resolve()
        for name in ordered_names:
            report_file = smoke_source / "smoke" / name / "pressure.json"
            if not report_file.is_file():
                continue
            if not all(p.read_bytes() == (smoke_source / "configs" / name / p.name).read_bytes()
                       for p in (jobs / name).glob("*.yaml")):
                continue
            report = json.loads(report_file.read_text())
            if (report["status"], report["envs"], report["updates"]) == ("passed", ENVS, SMOKE_ROUNDS):
                out = queue / "smoke" / name
                out.mkdir(parents=True)
                report["reused_from"] = str(report_file)
                write_json(out / "pressure.json", report)
                state["smoke_passed"].append(name)
                print(f"reused passed smoke {name}", flush=True)
    try:
        for phase, rounds in (("smoke", SMOKE_ROUNDS), ("formal", ROUNDS)):
            phase_plan = sorted(plan, key=lambda entry: entry[0] != args.smoke_first) if phase == "smoke" else plan
            for name, method, *_ in phase_plan:
                if phase == "smoke" and name in state["smoke_passed"]:
                    continue
                if name in completed:
                    continue
                destination = (queue / "smoke" if phase == "smoke" else OUTPUT) / name
                state.update(phase=phase, current=name, directory=str(destination))
                write_json(queue / "status.json", state)
                print(f"{time.strftime('%F %T')} {phase} {name}", flush=True)
                rows, peak_mib, elapsed = run(
                    jobs / name, destination, rounds, phase == "smoke", resume=phase == "formal" and name == interrupted,
                    reserve_for_physx=(bool(source) or args.suite == "run") and method == "amp")
                if phase == "smoke":
                    report = smoke_report(method, rows, peak_mib, elapsed, check_tracking=args.suite == "run")
                    write_json(destination / "pressure.json", report)
                    state["smoke_passed"].append(name)
                else:
                    state["completed"].append(name)
                write_json(queue / "status.json", state)
                print(f"{time.strftime('%F %T')} passed {phase} {name} ({elapsed:.1f} s)", flush=True)
        if args.suite == "run":
            state["evaluation"] = {}
            for name, method, *_ in plan:
                state.update(phase="evaluation", current=name,
                             directory=str(queue / "evaluation" / method))
                write_json(queue / "status.json", state)
                state["evaluation"][method] = evaluate_run(queue, name, method)
                write_json(queue / "status.json", state)
        state.update(phase="complete", current=None, directory=None)
    except BaseException as exc:
        state.update(error=str(exc), failed_phase=state["phase"], phase="failed")
        raise
    finally:
        write_json(queue / "status.json", state)


if __name__ == "__main__":
    main()
