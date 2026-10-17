"""Three-iteration physical parity audit: immutable old source vs isolated PPO.

Both arms use the archived configs unmodified, including output frequency 100.
Read-only instrumentation records EVERY log row and post-update state without
introducing extra evaluation, optimizer steps, random draws or SN forwards.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
REVISION = "58373240451d6ab093c7e3a154164a94f9dabab3"
SOURCE = ROOT / "output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012"
PERFORMANCE_FIELDS = {"Wall_Time", "Samples_Per_Second", "Peak_GPU_Memory_MB"}


def digest(path):
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def child(source_root, out):
    sys.path.insert(0, str(source_root / "mimickit"))
    import torch
    import run as application
    from learning.base_agent import BaseAgent
    from util.logger import Logger

    original_print = Logger.print_log
    original_train = BaseAgent._train_iter

    def print_and_capture(logger):
        original_print(logger)
        row = {key: entry.val for key, entry in logger.log_current_row.items()}
        with (out / "all_iterations.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")

    def train_and_capture(agent):
        result = original_train(agent)
        torch.save({"model": agent.state_dict(), "optimizers": {
            name: getattr(agent, name).state_dict() for name in
            ("_actor_optimizer", "_critic_optimizer", "_disc_optimizer")}},
            out / f"update_{agent._iter}.pt")
        return result

    Logger.print_log = print_and_capture
    BaseAgent._train_iter = train_and_capture
    argv = [str(source_root / "mimickit/run.py"),
            "--mode", "train", "--num_envs", "8192", "--devices", "cuda:0",
            "--visualize", "false", "--rand_seed", "0", "--logger", "txt",
            "--agent_config", str(SOURCE / "agent_config.yaml"),
            "--env_config", str(SOURCE / "env_config.yaml"),
            "--engine_config", str(SOURCE / "engine_config.yaml"),
            "--out_dir", str(out), "--max_samples", str(3 * 8192 * 32),
            "--save_int_models", "true"]
    print(json.dumps({"source_root": str(source_root), "argv": argv}), flush=True)
    application.main(argv)


def compare_rows(left, right):
    if len(left) != len(right):
        raise RuntimeError("Different iteration counts")
    result = []
    for a, b in zip(left, right):
        if list(a) != list(b):
            raise RuntimeError("Training log schema or column order changed")
        differences = {}
        for key in a.keys() - PERFORMANCE_FIELDS:
            if a[key] != b[key] and not (math.isnan(a[key]) and math.isnan(b[key])):
                differences[key] = {"reference": a[key], "isolated_ppo": b[key],
                                    "absolute_difference": abs(a[key] - b[key])}
        result.append({"iteration": a["Iteration"],
                       "compared_fields": len(a.keys() - PERFORMANCE_FIELDS),
                       "exact": not differences, "differences": differences})
    return result


def compare_states(left, right):
    import torch
    differences, count = {}, 0

    def visit(a, b, path):
        nonlocal count
        if torch.is_tensor(a):
            count += 1
            if a.shape != b.shape or a.dtype != b.dtype:
                raise RuntimeError(f"Tensor schema changed: {path}")
            if not torch.equal(a, b):
                differences[path] = float((a.double() - b.double()).abs().max())
        elif isinstance(a, dict):
            if a.keys() != b.keys():
                raise RuntimeError(f"State keys changed: {path}")
            for key in a:
                visit(a[key], b[key], f"{path}.{key}")
        elif isinstance(a, (list, tuple)):
            if len(a) != len(b):
                raise RuntimeError(f"State length changed: {path}")
            for index, (x, y) in enumerate(zip(a, b)):
                visit(x, y, f"{path}.{index}")
        elif a != b:
            raise RuntimeError(f"Non-tensor state changed: {path}: {a} vs {b}")

    visit(left, right, "state")
    return {"bitwise_equal": not differences, "compared_tensors": count,
            "max_abs": max(differences.values(), default=0.),
            "differences": differences}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--child-source", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    out = args.out.resolve()
    if args.child_source:
        child(args.child_source.resolve(), out)
        return
    out.mkdir(parents=True, exist_ok=False)
    archived = out / "archived_source"
    archived.mkdir()
    # git archive does not mutate the index, worktree or existing experiments.
    archive = subprocess.check_output(["git", "archive", REVISION, "mimickit"], cwd=ROOT)
    subprocess.run(["tar", "-x", "-C", str(archived)], input=archive, check=True)
    working = out / "refactored_source"
    shutil.copytree(ROOT / "mimickit", working / "mimickit",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    protected = ("learning/dare_agent.py", "learning/dare_model.py",
                 "learning/add_agent.py", "learning/add_model.py", "learning/rl_util.py",
                 "learning/mp_optimizer.py", "learning/base_agent.py",
                 "learning/experience_buffer.py", "envs/add_env.py", "envs/char_env.py")
    for path in protected:
        if digest(archived / "mimickit" / path) != digest(working / "mimickit" / path):
            raise RuntimeError(f"Protected baseline implementation changed: {path}")
    hashes = {str(p.relative_to(ROOT)): digest(p)
              for p in (ROOT / "mimickit").rglob("*.py")}
    configs = {name: digest(SOURCE / name) for name in
               ("agent_config.yaml", "env_config.yaml", "engine_config.yaml")}
    model_hash = digest(SOURCE / "int_models/model_0000002999.pt")
    manifest = {"revision": REVISION, "iterations": 3, "num_envs": 8192, "seed": 0,
                "comparison_completed": False, "config_hashes": configs,
                "source_hashes": hashes, "source_model_sha256": model_hash,
                "performance_fields_reported_but_not_equality_gated": sorted(PERFORMANCE_FIELDS),
                "protected_source_files": list(protected), "arms": {}}
    manifest_file = out / "comparison.json"

    def save():
        manifest_file.write_text(json.dumps(manifest, indent=2))

    save()
    import torch
    for name, source_root in (("reference", archived), ("isolated_ppo", working)):
        dest = out / name
        dest.mkdir()
        cmd = [sys.executable, "-B", "-u", str(Path(__file__).resolve()),
               "--child-source", str(source_root), "--out", str(dest)]
        with (dest / "console.log").open("x") as stream:
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env=dict(os.environ, OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1"))
            manifest["arms"][name] = {"pid": process.pid, "argv": cmd}
            save()
            print(f"START {name}: PID {process.pid}", flush=True)
            code = process.wait()
        manifest["arms"][name]["exit_code"] = code
        save()
        if code:
            raise RuntimeError(f"{name} failed: see {dest / 'console.log'}")
        checkpoint = torch.load(dest / "checkpoint.pt", map_location="cpu", weights_only=False)
        steps = {k: v["steps"] for k, v in checkpoint["optimizer_state_dicts"].items()}
        assert checkpoint["trainer_state"]["next_iter"] == 3
        assert checkpoint["trainer_state"]["sample_count"] == 3 * 8192 * 32
        assert checkpoint["trainer_state"]["resume_count"] == 0
        assert steps == {"_actor_optimizer": 120, "_critic_optimizer": 96, "_disc_optimizer": 96}
        manifest["arms"][name]["optimizer_steps"] = steps
        save()
        print(f"DONE {name}", flush=True)
        del checkpoint

    rows = [[json.loads(line) for line in (out / name / "all_iterations.jsonl").read_text().splitlines()]
            for name in ("reference", "isolated_ppo")]
    assert [[row["Iteration"] for row in arm] for arm in rows] == [[0, 1, 2], [0, 1, 2]]
    manifest["per_iteration_logs"] = compare_rows(*rows)
    manifest["per_iteration_states"] = []
    for iteration in range(3):
        states = [torch.load(out / name / f"update_{iteration}.pt", map_location="cpu", weights_only=False)
                  for name in ("reference", "isolated_ppo")]
        manifest["per_iteration_states"].append({"iteration": iteration, **compare_states(*states)})
        del states
    lines = (SOURCE / "log.txt").read_text().splitlines()
    original = dict(zip(lines[0].split(), map(float, lines[1].split())))
    manifest["original_archive_iteration0"] = compare_rows([original], [rows[1][0]])[0]
    manifest["historical_iterations1_and2_available"] = False
    manifest["working_sources_unchanged_during_test"] = all(digest(ROOT / p) == h for p, h in hashes.items())
    manifest["archive_configs_unchanged"] = all(digest(SOURCE / p) == h for p, h in configs.items())
    manifest["archive_model_unchanged"] = digest(SOURCE / "int_models/model_0000002999.pt") == model_hash
    manifest["strict_training_log_parity_passed"] = all(row["exact"] for row in manifest["per_iteration_logs"])
    manifest["strict_post_update_state_parity_passed"] = all(row["bitwise_equal"] for row in manifest["per_iteration_states"])
    manifest["comparison_completed"] = True
    save()
    print(json.dumps({key: manifest[key] for key in
                     ("comparison_completed", "strict_training_log_parity_passed",
                      "strict_post_update_state_parity_passed", "working_sources_unchanged_during_test")}), flush=True)


if __name__ == "__main__":
    main()
