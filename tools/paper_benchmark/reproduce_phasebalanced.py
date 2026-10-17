"""Re-run the archived phasebalanced baseline, without experimental agents."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
REVISION = "58373240451d6ab093c7e3a154164a94f9dabab3"
SOURCE = ROOT / "output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=3000)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("Positive iteration count required")
    paths = subprocess.check_output(
        ["git", "ls-tree", "-r", "--name-only", REVISION, "mimickit"],
        cwd=ROOT, text=True).splitlines()
    hashes, newline_only, registry_only = {}, [], []
    for name in paths:
        if not name.endswith(".py"):
            continue
        archived = subprocess.check_output(["git", "show", f"{REVISION}:{name}"], cwd=ROOT)
        current = (ROOT / name).read_bytes()
        if current != archived:
            extension = (b'    elif (agent_name == "DARE_REPPO"):\n'
                         b'        from learning.reppo_agent import REPPOAgent\n'
                         b'        agent = REPPOAgent(config=agent_config, env=env, device=device)\n')
            if (name == "mimickit/learning/agent_builder.py"
                    and current.count(extension) == 1
                    and current.replace(extension, b"", 1).rstrip(b"\n") == archived.rstrip(b"\n")):
                registry_only.append(name)
            elif current.rstrip(b"\n") == archived.rstrip(b"\n"):
                newline_only.append(name)
            else:
                raise RuntimeError(f"Baseline source differs from archive: {name}")
        # Also catch syntax errors before allocating the simulation.
        ast.parse(current, filename=name)
        hashes[name] = digest(ROOT / name)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    argv = [sys.executable, "-u", str(ROOT / "mimickit/run.py"),
        "--mode", "train", "--num_envs", "8192", "--devices", "cuda:0",
        "--visualize", "false", "--rand_seed", "0", "--logger", "txt",
        "--agent_config", str(SOURCE / "agent_config.yaml"),
        "--env_config", str(SOURCE / "env_config.yaml"),
        "--engine_config", str(SOURCE / "engine_config.yaml"),
        "--out_dir", str(out), "--max_samples", str(args.iterations * 8192 * 32),
        "--save_int_models", "true"]
    manifest = dict(revision=REVISION, argv=argv, iterations=args.iterations,
        fresh_initialization=True, num_envs=8192, seed=0,
        source_hashes=hashes, newline_only_differences=newline_only,
        inactive_reppo_registry_differences=registry_only,
        config_hashes={name: digest(SOURCE / name) for name in
                       ("agent_config.yaml", "env_config.yaml", "engine_config.yaml")},
        source_model_sha256=digest(SOURCE / "int_models/model_0000002999.pt"),
        started_unix=time.time(), complete=False)
    with (out / "console.log").open("x") as log:
        process = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS="4", PYTHONUNBUFFERED="1"))
        manifest["pid"] = process.pid
        (out / "reproduction.json").write_text(json.dumps(manifest, indent=2))
        print(json.dumps(dict(pid=process.pid, out=str(out), iterations=args.iterations)), flush=True)
        code = process.wait()
    manifest.update(exit_code=code, finished_unix=time.time())
    if code == 0:
        import torch
        checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        trainer = checkpoint["trainer_state"]
        steps = {k: v["steps"] for k, v in checkpoint["optimizer_state_dicts"].items()}
        assert trainer["next_iter"] == args.iterations
        assert trainer["sample_count"] == args.iterations * 8192 * 32
        assert trainer["resume_count"] == 0
        assert steps == {"_actor_optimizer": 40 * args.iterations,
                         "_critic_optimizer": 32 * args.iterations,
                         "_disc_optimizer": 32 * args.iterations}
        source_zero = torch.load(SOURCE / "int_models/model_0000000000.pt", map_location="cpu", weights_only=False)
        current_zero = torch.load(out / "int_models/model_0000000000.pt", map_location="cpu", weights_only=False)
        differences = {k: float((current_zero[k].double() - v.double()).abs().max())
                       for k, v in source_zero.items() if not torch.equal(current_zero[k], v)}
        manifest.update(complete=True, optimizer_steps=steps,
            initial_output_bitwise_equal=not differences,
            initial_output_tensor_differences=differences,
            source_model_unchanged=digest(SOURCE / "int_models/model_0000002999.pt") == manifest["source_model_sha256"],
            baseline_sources_unchanged=all(digest(ROOT / p) == h for p, h in hashes.items()))
    (out / "reproduction.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2), flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    main()
