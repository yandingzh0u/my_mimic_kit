"""Train repaired Climb from scratch; an additional baseline is opt-in."""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import yaml

ROOT = Path(__file__).resolve().parents[2]
ENV_CONFIG = "data/envs/paper_benchmark/dare_climb_env.yaml"
ENGINE_CONFIG = "data/engines/isaac_lab_engine.yaml"
RUNS = (("supported", "data/agents/dare_10layer_cpl_climb_supported_agent.yaml"),
        ("nophase", "data/agents/dare_10layer_cpl_climb_nophase_agent.yaml"))
ITERATIONS, ENVS, STEPS, SEED = 3000, 8192, 32, 0
SAMPLES = ITERATIONS * ENVS * STEPS


def now():
    return datetime.now().astimezone().isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n")
    temporary.replace(path)


def launch_arguments(snapshot, output, agent):
    # Build individual values; replacing path substrings can corrupt an
    # absolute snapshot path containing the output directory name itself.
    return {"mode": "train", "num_envs": ENVS, "devices": "cuda:0",
            "engine_config": str(snapshot/ENGINE_CONFIG),
            "env_config": str(snapshot/ENV_CONFIG),
            "agent_config": str(snapshot/agent),
            "max_samples": SAMPLES, "rand_seed": SEED,
            "visualize": "false", "save_int_models": "true", "logger": "txt",
            "out_dir": str(output)}


def initial_fingerprint(state):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def final_metrics(output):
    rows = [json.loads(line) for line in (output/"train_metrics.jsonl").read_text().splitlines() if line]
    if not rows or int(rows[-1]["iteration"]) != ITERATIONS-1 or int(rows[-1]["samples"]) != SAMPLES:
        raise RuntimeError(f"Incomplete training metrics: {output}")
    return rows


def validate_checkpoint(output):
    import torch
    rows = final_metrics(output)
    checkpoint = torch.load(output/"checkpoint.pt", map_location="cpu", weights_only=False)
    trainer = checkpoint["trainer_state"]
    if int(trainer["next_iter"]) != ITERATIONS or int(trainer["sample_count"]) != SAMPLES:
        raise RuntimeError(f"Final checkpoint does not match the completed budget: {output}")
    if int(trainer.get("resume_count", 0)) != 0:
        raise RuntimeError(f"Expected from-scratch, uninterrupted training: {output}")
    if not (output/"model.pt").is_file():
        raise RuntimeError(f"Final inference model missing: {output}")
    return rows[-1]


def prepare(campaign, include_baseline=False):
    root = ROOT
    runs = RUNS if include_baseline else RUNS[:1]
    evidence = root/"output/research/phase_rootcause_20261025"
    diagnosis = json.loads((evidence/"diagnosis.json").read_text())
    for name, digest in diagnosis["source_sha256"].items():
        if sha256(root/name) != digest:
            raise RuntimeError(f"Source/config differs from the validated candidate: {name}")
    configs = [yaml.safe_load((root/path).read_text()) for _, path in runs]
    differences = ({key for key in configs[0].keys() | configs[1].keys()
                    if configs[0].get(key) != configs[1].get(key)} if include_baseline else set())
    if ((include_baseline and differences != {"phase_sampler"})
            or any(c["phase_coverage_reset"] for c in configs)):
        raise RuntimeError("The controlled pair must differ only in phase_sampler, with original random resets")
    if shutil.disk_usage(root).free < 4*1024**3:
        raise RuntimeError("Insufficient space for both runs and checkpoints")
    outputs = [root/"output/paper_benchmark"/f"dare_10layer_cpl_{name}_climb_3000_8192_seed0_headless_20261025"
               for name, _ in runs]
    if campaign.exists() or any(path.exists() for path in outputs):
        raise FileExistsError("Campaign/output exists; refusing to overwrite an experiment")
    campaign.mkdir(parents=True)
    snapshot = campaign/"source_snapshot"
    shutil.copytree(root/"mimickit", snapshot/"mimickit", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    files = [ENV_CONFIG, ENGINE_CONFIG, *(p for _, p in runs),
             "tools/paper_benchmark/run_phase_supported_control.py"]
    for name in files:
        target = snapshot/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(root/name, target)
    validation = campaign/"validation"
    validation.mkdir()
    for name in ("diagnosis.json", "audit.json", "support_stress.json", "support_gpu_stress.json", "pytest.log"):
        shutil.copy2(evidence/name, validation/name)
    for group in ("supported_smoke8192", "nophase_smoke64"):
        shutil.copy2(evidence/group/"smoke.json", validation/(group+".json"))
    (campaign/"working_tree.patch").write_bytes(subprocess.check_output(["git", "diff", "--binary"], cwd=root))
    environment = yaml.safe_load((root/ENV_CONFIG).read_text())
    assets = {environment["motion_file"], environment["char_file"]}
    char_usd = Path(environment["char_file"]).with_suffix(".usd")
    if (root/char_usd).is_file(): assets.add(str(char_usd))
    for obj in environment.get("objects", []):
        assets.add(obj["file"])
        usd = Path(obj["file"]).with_suffix(".usd")
        if (root/usd).is_file(): assets.add(str(usd))
    manifest = dict(created_at=now(), repo=str(root), snapshot=str(snapshot), python=sys.executable,
                    git_head=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                    initialization="from scratch", iterations=ITERATIONS, environments=ENVS, steps=STEPS,
                    seed=SEED, samples=SAMPLES, reset="original iid reference starts",
                    config_differences=sorted(differences),
                    asset_sha256={name: sha256(root/name) for name in sorted(assets)},
                    source_sha256={str(p.relative_to(snapshot)): sha256(p)
                                   for p in snapshot.rglob("*") if p.is_file()}, jobs=[])
    runner = snapshot/"tools/paper_benchmark/run_phase_supported_control.py"
    for (name, agent), output in zip(runs, outputs):
        output.mkdir()
        arguments = launch_arguments(snapshot, output, agent)
        for key in ("engine_config", "env_config", "agent_config"):
            if not Path(arguments[key]).is_file(): raise FileNotFoundError(arguments[key])
        args_file = output/"launch_args.txt"
        args_file.write_text("".join(f"--{key} {value}\n" for key, value in arguments.items()))
        job = dict(name=name, output=str(output), args_file=str(args_file), state="queued",
                   command=[sys.executable, "-u", str(runner), "--job", str(campaign), "--name", name])
        manifest["jobs"].append(job)
    write_json(campaign/"manifest.json", manifest)
    write_json(campaign/"status.json", dict(state="prepared", updated_at=now(), jobs=manifest["jobs"]))
    return manifest


def job_main(campaign, name):
    manifest = json.loads((campaign/"manifest.json").read_text())
    job = next(row for row in manifest["jobs"] if row["name"] == name)
    output = Path(job["output"])
    os.chdir(manifest["repo"])
    sys.path.insert(0, str(Path(manifest["snapshot"])/"mimickit"))
    import run as training
    build_agent = training.build_agent
    def audited_build_agent(args, env, device):
        agent = build_agent(args, env, device)
        digest = initial_fingerprint(agent.state_dict())
        first = Path(manifest["jobs"][0]["output"])/"initial_state.json"
        if name != manifest["jobs"][0]["name"]:
            if digest != json.loads(first.read_text())["sha256"]:
                raise RuntimeError("The controlled runs have different initial model/normalizer states")
        write_json(output/"initial_state.json", dict(sha256=digest, seed=SEED,
                    phase_sampler=agent._phase_sampler_mode, phase_coverage_reset=agent._phase_coverage_reset))
        print(f"INITIAL_STATE_SHA256 {digest}", flush=True)
        return agent
    training.build_agent = audited_build_agent
    training.main(["run.py", "--arg_file", job["args_file"]])


def worker(campaign):
    manifest = json.loads((campaign/"manifest.json").read_text())
    root, snapshot = Path(manifest["repo"]), Path(manifest["snapshot"])
    status = dict(state="running", controller_pid=os.getpid(), updated_at=now(), jobs=manifest["jobs"])
    try:
        lock = (root/"output/paper_benchmark/.serial_matrix.lock").open("a+")
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for name, digest in manifest["source_sha256"].items():
            if sha256(snapshot/name) != digest: raise RuntimeError(f"Snapshot changed: {name}")
        write_json(campaign/"controller.json", dict(pid=os.getpid(), started_at=now()))
        for job in status["jobs"]:
            for name, digest in manifest["asset_sha256"].items():
                if sha256(root/name) != digest: raise RuntimeError(f"Experiment asset changed: {name}")
            output = Path(job["output"])
            env = os.environ.copy()
            env.update(OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="4", PYTHONPATH=str(snapshot/"mimickit"))
            with (output/"console.log").open("wb") as log:
                process = subprocess.Popen(job["command"], cwd=root, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
            job.update(state="running", pid=process.pid, started_at=now())
            (output/"train.pid").write_text(str(process.pid)+"\n")
            write_json(output/"launch.json", job)
            status["updated_at"] = now()
            write_json(campaign/"status.json", status)
            print(json.dumps(job), flush=True)
            code = process.wait()
            if code != 0: raise RuntimeError(f"{job['name']} exited with status {code}")
            last = validate_checkpoint(output)
            job.update(state="complete", completed_at=now(), final_iteration=int(last["iteration"]),
                       final_samples=int(last["samples"]))
            write_json(output/"DONE.json", job)
            status["updated_at"] = now()
            write_json(campaign/"status.json", status)
        status.update(state="complete", updated_at=now())
        write_json(campaign/"status.json", status)
    except BaseException as exc:
        for job in status["jobs"]:
            if job["state"] == "running":
                job.update(state="failed", error=str(exc))
        status.update(state="failed", updated_at=now(), error=str(exc))
        write_json(campaign/"status.json", status)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path)
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--job", type=Path)
    parser.add_argument("--name")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--include-baseline", action="store_true",
                        help="Explicitly queue an additional no-phase baseline after the repaired run")
    args = parser.parse_args()
    if args.job:
        job_main(args.job.resolve(), args.name)
    elif args.worker:
        worker(args.worker.resolve())
    else:
        campaign_name = ("phase_supported_vs_nophase_climb_20261025" if args.include_baseline
                         else "phase_supported_climb_20261025")
        campaign = (args.campaign or ROOT/"output/paper_benchmark"/campaign_name).resolve()
        manifest = prepare(campaign, include_baseline=args.include_baseline)
        if args.prepare_only:
            print(json.dumps(dict(campaign=str(campaign), state="prepared")))
            return
        runner = Path(manifest["snapshot"])/"tools/paper_benchmark/run_phase_supported_control.py"
        with (campaign/"controller.log").open("wb") as log:
            process = subprocess.Popen([sys.executable, "-u", str(runner), "--worker", str(campaign)],
                cwd=manifest["repo"], stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                start_new_session=True)
        (campaign/"controller.pid").write_text(str(process.pid)+"\n")
        print(json.dumps(dict(campaign=str(campaign), controller_pid=process.pid, jobs=manifest["jobs"]), indent=2))


if __name__ == "__main__":
    main()
