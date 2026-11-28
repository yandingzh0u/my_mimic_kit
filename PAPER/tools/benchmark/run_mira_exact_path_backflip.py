#!/usr/bin/env python3
"""Paired fresh Backflip runs: global path cap versus current per-layer cap.

Start this with the same Isaac Lab Python as the benchmark scripts. Freezing
does not import Isaac Lab or touch the production model on disk.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import run_mira_diagonal as trainer


ROOT = trainer.ROOT
ARMS = ("exact_path", "per_layer")
MODEL = Path("mimickit/learning/mira_monotone_model.py")
AGENT = MODEL.with_name("mira_monotone_agent.py")
HELPER = MODEL.with_name("mira_exact_path_weights.py")
EVALUATION = Path("PAPER/tools/evaluation")
EVAL_STEPS, EVAL_ENVS = 60, 256
INITIAL_KEYS = ("actor_initial_sha256", "critic_initial_sha256", "discriminator_initial_sha256",
                "cpu_rng_sha256", "cuda_rng_sha256", "host_rng_sha256")


def replace_once(path, old, new):
    source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"Unable to apply frozen patch at {path}: {old[:75]!r}")
    path.write_text(source.replace(old, new))


def freeze_arm(out, arm, updates):
    trainer.freeze(argparse.Namespace(out_dir=out, motion="backflip", preflight=False,
                                     freeze_only=True, updates=updates))
    snapshot = out / "source_snapshot"
    runner = snapshot / trainer.RUNNER_REL
    manifest_path = out / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if arm == "exact_path":
        model = snapshot / MODEL
        shutil.copyfile(Path(__file__).with_name(HELPER.name), snapshot / HELPER)
        replace_once(model, "from learning.weight_constraints import enforce_nonnegative_contraction_, spectral_norm",
                     "from learning.weight_constraints import enforce_nonnegative_contraction_\n"
                     "from learning.mira_exact_path_weights import (\n"
                     "    exact_positive_path_gain, enforce_nonnegative_path_contraction_)")
        replace_once(model, "MONOTONE_SCHEMA_VERSION = 29", "MONOTONE_SCHEMA_VERSION = 30")
        replace_once(model, "    def enforce_disc_weight_constraints(self):\n"
                     "        for layer in self._disc_energy:\n"
                     "            if isinstance(layer, nn.Linear):\n"
                     "                enforce_nonnegative_contraction_(layer.weight)",
                     "    def enforce_disc_weight_constraints(self):\n"
                     "        weights = [layer.weight for layer in self._disc_energy if isinstance(layer, nn.Linear)]\n"
                     "        enforce_nonnegative_path_contraction_(weights)")
        replace_once(model, "        else:\n            for index, layer in enumerate(self._disc_energy):",
                     "        else:\n            checkpoint_weights = []\n"
                     "            for index, layer in enumerate(self._disc_energy):")
        replace_once(model, "                            or (weight < 0).any()\n"
                     "                            or spectral_norm(weight) > 1 + 8 * torch.finfo(weight.dtype).eps):\n"
                     "                        error_msgs.append(\"Invalid nonnegative contraction in MIRA checkpoint: \" + key)",
                     "                            or (weight < 0).any()):\n"
                     "                        error_msgs.append(\"Invalid nonnegative weight in MIRA checkpoint: \" + key)\n"
                     "                    else:\n"
                     "                        checkpoint_weights.append(weight)\n"
                     "            if len(checkpoint_weights) == sum(isinstance(layer, nn.Linear) for layer in self._disc_energy):\n"
                     "                tolerance = 8 * max(torch.finfo(weight.dtype).eps for weight in checkpoint_weights)\n"
                     "                if exact_positive_path_gain(checkpoint_weights) > 1 + tolerance:\n"
                     "                    error_msgs.append(\"MIRA checkpoint exceeds the nonnegative path contraction bound\")")
        replace_once(snapshot / AGENT, "post_sgd_nonnegative_clip_uniform_spectral_cap",
                     "post_sgd_nonnegative_clip_exact_path_cap_via_final_weight")
    replace_once(runner, "    assert ROOT == resolve_relocated_path(manifest[\"frozen_code_root\"])",
                 "    arm = manifest['constraint_arm']\n"
                 "    assert ROOT == resolve_relocated_path(manifest[\"frozen_code_root\"])")
    replace_once(runner, "    @torch.no_grad()\n    def verify_weights(weights):",
                 "    def path_gain(weights):\n"
                 "        # Independent CPU double audit; no experiment helper is used.\n"
                 "        row = weights[-1].detach().to('cpu', torch.float64).flatten()\n"
                 "        for weight in reversed(weights[:-1]):\n"
                 "            row = row @ weight.detach().to('cpu', torch.float64)\n"
                 "        return float(row.norm())\n\n"
                 "    @torch.no_grad()\n    def verify_weights(weights):")
    replace_once(runner, "        assert all(norm <= 1 + 4 * torch.finfo(weight.dtype).eps for norm, weight in zip(norms, weights)), norms",
                 "        if arm == 'per_layer':\n"
                 "            assert all(norm <= 1 + 4 * torch.finfo(weight.dtype).eps for norm, weight in zip(norms, weights)), norms\n"
                 "        gain = path_gain(weights)\n"
                 "        assert math.isfinite(gain) and gain <= 1 + 8 * max(torch.finfo(w.dtype).eps for w in weights), gain")
    replace_once(runner, "                   \"critic_initial_sha256\": fingerprint(agent._model, (\"_critic_layers.\", \"_critic_out.\")),",
                 "                   \"critic_initial_sha256\": fingerprint(agent._model, (\"_critic_layers.\", \"_critic_out.\")),\n"
                 "                   \"discriminator_initial_sha256\": fingerprint(agent._model, (\"_disc_energy.\",)),\n"
                 "                   \"cpu_rng_sha256\": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),\n"
                 "                   \"cuda_rng_sha256\": hashlib.sha256(torch.cuda.get_rng_state(agent._device).numpy().tobytes()).hexdigest(),\n"
                 "                   \"host_rng_sha256\": host_rng_hash(),\n"
                 "                   \"initial_path_gain_L2\": path_gain(weights),")
    replace_once(runner, "    def capture_env(*values, **kwargs):",
                 "    def host_rng_hash():\n"
                 "        import pickle, random\n"
                 "        import numpy as np\n"
                 "        return hashlib.sha256(pickle.dumps((random.getstate(), np.random.get_state()))).hexdigest()\n\n"
                 "    def capture_env(*values, **kwargs):")
    replace_once(runner, "                    \"sampled_rollout_states\": len(selection),",
                 "                    \"sampled_rollout_states\": len(selection),\n"
                 "                    \"sampled_policy_and_reference_sha256\": hashlib.sha256(\n"
                 "                        torch.cat((policy[selection], reference[selection]), dim=-1)\n"
                 "                        .contiguous().cpu().numpy().tobytes()).hexdigest(),")
    replace_once(runner, "                    \"normalized_error_Lipschitz_upper_bound\": math.prod(norms),",
                 "                    \"normalized_error_Lipschitz_upper_bound\": path_gain(weights),\n"
                 "                    \"layerwise_spectral_product\": math.prod(norms),")
    replace_once(runner, "                  \"saved_exact_spectral_norms\": norms,",
                 "                  \"saved_exact_spectral_norms\": norms,\n"
                 "                  \"saved_exact_path_gain_L2\": path_gain(saved_weights),")
    replace_once(runner, '"experiment": "MIRA64_direct_nonnegative_unit_operator_norm_weights"',
                 '"experiment": "MIRA64_Backflip_constraint_pair", "constraint_arm": arm')
    for name in ("__init__.py", "evaluate_checkpoint.py", "metrics.py", "input_geometry.py"):
        relative = EVALUATION / name
        path = snapshot / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, path)
    manifest["constraint_arm"] = arm
    manifest["files"] = {str(path.relative_to(snapshot)): trainer.sha(path)
                         for path in sorted(snapshot.rglob("*")) if path.is_file()}
    trainer.save(manifest_path, manifest)
    for relative in (MODEL, AGENT, trainer.RUNNER_REL) + ((HELPER,) if arm == "exact_path" else ()):
        compile((snapshot / relative).read_text(), str(relative), "exec")


EVALUATE = """
import os, sys, traceback
from pathlib import Path
sys.path.insert(0, sys.argv.pop(1))
from PAPER.tools.evaluation import evaluate_checkpoint as evaluator
evaluator.REPO_ROOT = Path(sys.argv.pop(1))
evaluator.MIMICKIT_ROOT = Path(sys.argv.pop(1)) / 'mimickit'
try:
    code = evaluator.main()
except BaseException:
    traceback.print_exc()
    code = 1
sys.stdout.flush()
sys.stderr.flush()
os._exit(code)
"""


def run(out, command, log_name):
    env = dict(os.environ, ACCEPT_EULA="Y", OMNI_KIT_DISABLE_TELEMETRY="1",
               PYTHONDONTWRITEBYTECODE="1", OMP_NUM_THREADS="4", MKL_NUM_THREADS="4",
               MIMICKIT_CRITIC_EVAL_TIME_BATCH="2",
               PYTORCH_CUDA_ALLOC_CONF="garbage_collection_threshold:0.55")
    with (out / log_name).open("x") as console:
        subprocess.run([sys.executable, "-u", *map(str, command)], env=env, cwd=ROOT,
                       stdout=console, stderr=subprocess.STDOUT, check=True)


def read_json(path):
    return json.loads(path.read_text())


def verify_pair_sources(directories, updates):
    path_run, baseline = (read_json(directories[arm] / "source_manifest.json") for arm in ARMS)
    for key in ("original_repository", "motion", "from_zero", "resumed", "seed", "updates",
                "num_envs", "steps_per_update", "max_samples", "inputs"):
        if path_run[key] != baseline[key]:
            raise ValueError(f"Unmatched paired-training source metadata: {key}")
    assert baseline["updates"] == updates and baseline["motion"] == "backflip"
    assert (baseline["from_zero"], baseline["resumed"], baseline["seed"], baseline["num_envs"],
            baseline["steps_per_update"]) == (True, False, trainer.SEED, trainer.NUM_ENVS, trainer.STEPS)
    allowed = {str(MODEL), str(AGENT), str(HELPER), str(trainer.RUNNER_REL)}
    left, right = (manifest["files"] for manifest in (path_run, baseline))
    diffs = {name for name in left.keys() | right.keys() if left.get(name) != right.get(name)
             and not name.startswith(str(EVALUATION) + "/")}
    if diffs - allowed:
        raise ValueError(f"Unexpected differences between paired training sources: {sorted(diffs - allowed)}")
    for arm in ARMS:
        manifest = path_run if arm == "exact_path" else baseline
        for name, expected in manifest["files"].items():
            if trainer.sha(directories[arm] / "source_snapshot" / name) != expected:
                raise ValueError(f"{arm} frozen source checksum mismatch: {name}")
    return sorted(diffs)


def compare(directories, evaluation_dirs):
    import numpy as np

    audits = {arm: read_json(directories[arm] / "initial_state.json") for arm in ARMS}
    matched_fingerprints = []
    for key in INITIAL_KEYS:
        if key not in audits["per_layer"] and key not in INITIAL_KEYS[:2]:
            continue  # pre-existing baseline may have only actor/critic tensor hashes
        if audits[ARMS[0]][key] != audits[ARMS[1]][key]:
            raise ValueError(f"Fresh paired initialization does not match: {key}")
        matched_fingerprints.append(key)
    left, right = (audits[arm]["fixed_probe_initial_energy"] for arm in ARMS)
    if not np.allclose(left, right, atol=1e-7, rtol=1e-7):
        raise ValueError("Initial discriminator probe energy differs across paired runs")
    first_rollout = {arm: read_json(directories[arm] / "first_rollout_before_update.json") for arm in ARMS}
    rollout_hashes = {arm: row.get("sampled_policy_and_reference_sha256") for arm, row in first_rollout.items()}
    rollout_stats = {arm: row["sampled_rollout_distributions"] for arm, row in first_rollout.items()}
    traces, evaluations, time_to_first_termination, terminal_training = {}, {}, {}, {}
    for arm in ARMS:
        directory = directories[arm]
        progress = read_json(directory / "progress.json")
        pressure = progress["last_pressure_check"]
        terminal_training[arm] = {
            "completed_updates": progress["completed_updates"], "samples": progress["samples"],
            "phase_zero_online_test": progress["last_phase_zero_test"],
            "disc_BCE": progress["disc_BCE"], "reward_mean": progress["reward_mean"],
            "reward_std": progress["reward_std"],
            "exact_layer_spectral_norms": pressure["exact_spectral_norms"],
            "phase_zero_online_err_averaging_differs_from_fixed_window": True,
        }
        assert progress["completed_updates"] == progress["last_phase_zero_test"]["completed_updates"]
        report = read_json(evaluation_dirs[arm] / "summary.json")
        protocol = report["protocol"]
        assert (protocol["requested_steps"], protocol["num_episodes"], protocol["start_mode"],
                report["metadata"]["seed"], protocol["deterministic_mean_action"],
                protocol["no_automatic_reset"]) == (EVAL_STEPS, EVAL_ENVS, "phase0", 0, True, True)
        assert report["metadata"]["model_sha256"] == trainer.sha(directory / "model.pt")
        assert report["metadata"]["config_hashes"] == {
            metadata_key + "_sha256": trainer.sha(directory / f"frozen_{config_key}.yaml")
            for metadata_key, config_key in (("agent", "agent"), ("environment", "env"), ("engine", "engine"))
        }
        with np.load(evaluation_dirs[arm] / "timeseries.npz") as trace:
            alive = trace["alive"]
            assert alive.ndim == 2 and alive.shape == (protocol["executed_steps"], EVAL_ENVS)
            not_full = np.flatnonzero(~alive.all(axis=1))
            time_to_first_termination[arm] = int(not_full[0]) if not_full.size else len(alive)
            traces[arm] = (trace["tracking_error_names"], trace["tracking_error"])
        evaluations[arm] = report
    prefix = min(time_to_first_termination.values())
    tracking = {}
    for arm, (names, errors) in traces.items():
        assert tuple(names[:7]) == ("root_pos_err", "root_rot_err", "body_pos_err", "body_rot_err",
                                    "dof_vel_err", "root_vel_err", "root_ang_vel_err")
        assert np.isfinite(errors[:prefix, :, :7]).all()
        tracking[arm] = (dict(zip(names[:7], errors[:prefix, :, :7].mean((0, 1), dtype=np.float64)))
                         if prefix else None)
    delta = ({key: value - tracking["per_layer"][key] for key, value in tracking["exact_path"].items()}
             if prefix else None)
    return {"initial_identical_fingerprints": matched_fingerprints,
            "initial_discriminator_probe_max_abs_difference": float(np.max(np.abs(np.asarray(left) - right))),
            "same_initial_model_weights_and_rng": len(matched_fingerprints) == len(INITIAL_KEYS),
            "first_rollout_fingerprints_match": (len(set(rollout_hashes.values())) == 1
                                                 if all(rollout_hashes.values()) else None),
            "first_rollout_fingerprints": rollout_hashes, "first_rollout_distributions": rollout_stats,
            "requested_evaluation_steps": EVAL_STEPS,
            "compared_steps": prefix, "full_window_available": prefix == EVAL_STEPS,
            "all_episode_prefix_by_arm": time_to_first_termination,
            "eval_model_sha256": {arm: evaluations[arm]["metadata"]["model_sha256"] for arm in ARMS},
            "terminal_online_diagnostics_not_matched_window": terminal_training,
            "mean_tracking_errors": tracking, "exact_path_minus_per_layer": delta}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--existing-baseline", type=Path,
                        help="Use an existing same-budget, fresh per-layer experiment without modifying it")
    args = parser.parse_args()
    if not 3 <= args.updates <= 2000:
        parser.error("--updates must be between 3 and 2000")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=bool(args.existing_baseline))
    directories = {arm: out / arm for arm in ARMS}
    if args.existing_baseline:
        directories["per_layer"] = args.existing_baseline.resolve()
        if directories["per_layer"] == directories["exact_path"]:
            parser.error("--existing-baseline cannot overlap the exact-path directory")
        if not args.freeze_only:
            baseline_status = read_json(directories["per_layer"] / "status.json")
            if (baseline_status["status"], baseline_status["updates"]) != ("completed", args.updates):
                parser.error("--existing-baseline must have completed the same training budget")
    evaluation_dirs = {arm: out / "evaluations" / arm for arm in ARMS}
    status_path = out / "pair_status.json"

    def status(phase, **kwargs):
        trainer.save(status_path, dict(status=phase, pid=os.getpid(), updates=args.updates, **kwargs))

    with (out / ".pair.lock").open("x") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            status("freezing")
            train_arms = ARMS[:1] if args.existing_baseline else ARMS
            for arm in train_arms:
                freeze_arm(directories[arm], arm, args.updates)
            diffs = verify_pair_sources(directories, args.updates)
            status("frozen", changed_training_sources=diffs)
            if args.freeze_only:
                return
            for arm in train_arms:
                directory = directories[arm]
                snapshot = directory / "source_snapshot"
                status("training", arm=arm)
                run(directory, (snapshot / trainer.RUNNER_REL, "--frozen-worker", "--updates",
                                args.updates, "--out-dir", directory), "console.log")
                finished = read_json(directory / "status.json")
                assert (finished["status"], finished["updates"]) == ("completed", args.updates)
            for arm in ARMS:
                directory = directories[arm]
                status("evaluating", arm=arm)
                evaluation_dirs[arm].mkdir(parents=True, exist_ok=False)
                run(evaluation_dirs[arm], ("-c", EVALUATE, directories["exact_path"] / "source_snapshot", ROOT,
                                directory / "source_snapshot",
                                "--model-file", directory / "model.pt",
                                "--env-config", directory / "frozen_env.yaml",
                                "--agent-config", directory / "frozen_agent.yaml",
                                "--engine-config", directory / "frozen_engine.yaml",
                                "--method", "MIRA", "--motion", "backflip", "--num-envs", EVAL_ENVS,
                                "--steps", EVAL_STEPS, "--start-mode", "phase0", "--seed", 0,
                                "--out-dir", evaluation_dirs[arm]), "console.log")
            trainer.save(out / "comparison.json",
                         {**compare(directories, evaluation_dirs), "changed_training_sources": diffs})
            status("completed")
        except BaseException as error:
            status("failed", error=repr(error))
            raise


if __name__ == "__main__":
    main()
