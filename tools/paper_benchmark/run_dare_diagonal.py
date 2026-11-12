#!/usr/bin/env python3
"""Freeze and run the production seven-ERR diagonal CMNN from scratch.

Examples (use the Isaac Lab Python environment):
  python tools/paper_benchmark/run_dare_diagonal.py --updates 30 --out-dir output/diagnostics/diagonal_smoke
  python tools/paper_benchmark/run_dare_diagonal.py --updates 2000 --out-dir output/paper_benchmark/diagonal_formal

This runner does not replace model methods, alter YAMLs, load initialization
weights, or resume another experiment. Its hooks only observe training.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
RUNNER_REL = Path("tools/paper_benchmark/run_dare_diagonal.py")
CONFIGS = {
    "agent": Path("data/agents/dare_monotone_humanoid_agent.yaml"),
    "env": Path("data/envs/paper_benchmark/dare_backflip_env.yaml"),
    "engine": Path("data/engines/isaac_lab_engine.yaml"),
}
NUM_ENVS, STEPS, SEED = 8192, 32, 0
METRICS = ("root_pos", "root_rot", "body_pos", "body_rot", "dof_vel", "root_vel", "root_ang_vel")


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def append(path, value):
    with Path(path).open("a") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def preflight():
    """Read configuration and assets without importing torch or Isaac Lab."""
    import yaml
    configs = {kind: yaml.safe_load((ROOT / rel).read_text()) for kind, rel in CONFIGS.items()}
    config = configs["agent"]
    assert config["agent_name"] == "DAREMonotone"
    assert config["steps_per_iter"] == STEPS
    assert config["iters_per_output"] == 100
    assert config["disc_grad_penalty"] == 0
    assert config["task_reward_weight"] == 0 and config["disc_reward_weight"] == 1
    assert "disc_reward_scale" not in config
    assert config["disc_optimizer"] == {"type": "SGD", "learning_rate": 2.5e-4, "weight_decay": .0001}
    for kind in ("actor", "critic"):
        optimizer = config[kind + "_optimizer"]
        assert optimizer["type"] == "SGD"
        assert float(optimizer["learning_rate"]) == 1e-4
    assert configs["engine"]["engine_name"] == "isaac_lab"
    assert configs["env"]["motion_name"] == "backflip"
    assets = {}
    for key in ("char_file", "motion_file"):
        path = Path(configs["env"][key])
        if not path.is_absolute():
            path = ROOT / path
        assert path.is_file(), str(path)
        assets[key] = {"path": str(path.resolve()), "sha256": sha(path)}
    assert (ROOT / "mimickit/learning/diagonal_normalization.py").is_file()
    return {"configs": {kind: {"path": str(ROOT / rel), "sha256": sha(ROOT / rel)}
                        for kind, rel in CONFIGS.items()}, "runtime_assets": assets}


def freeze(args):
    out = args.out_dir.resolve()
    inputs = preflight()
    if args.preflight:
        print(json.dumps({"status": "passed_static_preflight", **inputs}, indent=2))
        return
    forbidden = ("source_snapshot", "model.pt", "checkpoint.pt", "protocol.json", "log.txt",
                 "progress.json", "status.json", "training_rounds.jsonl", "train_metrics.jsonl")
    if any((out / name).exists() for name in forbidden):
        raise ValueError("Refusing to reuse an experiment directory: " + str(out))
    out.mkdir(parents=True, exist_ok=True)
    target = out / "source_snapshot"
    shutil.copytree(ROOT / "mimickit", target / "mimickit",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (target / RUNNER_REL).parent.mkdir(parents=True)
    shutil.copyfile(ROOT / RUNNER_REL, target / RUNNER_REL)
    for kind, relative in CONFIGS.items():
        frozen = target / relative
        frozen.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, frozen)
        shutil.copyfile(frozen, out / ("frozen_" + kind + ".yaml"))
    # Compare only actor/critic initialization, never a historical disc layout.
    expected = None
    candidates = (
        ROOT / "output/diagnostics/cmnn_spectral_constraint_20261112/short_SN7_8192_50/initial_state.json",
        ROOT / "output/paper_benchmark/dare_monotone7_backflip_2000_8192_seed0_20261110/initial_state.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            previous = json.loads(candidate.read_text())
            keys = ("actor_initial_sha256", "critic_initial_sha256")
            if all(key in previous for key in keys):
                expected = {"source": str(candidate), **{key: previous[key] for key in keys}}
                break
    save(out / "expected_initialization.json", expected)
    files = {str(path.relative_to(target)): sha(path)
             for path in sorted(target.rglob("*")) if path.is_file()}
    manifest = {"original_repository": str(ROOT), "frozen_code_root": str(target), "files": files,
                "inputs": inputs, "from_zero": True, "resumed": False, "seed": SEED,
                "updates": args.updates, "num_envs": NUM_ENVS, "steps_per_update": STEPS,
                "max_samples": args.updates * NUM_ENVS * STEPS}
    save(out / "source_manifest.json", manifest)
    save(out / "status.json", {"status": "frozen", "pid": os.getpid(), "updates": args.updates})
    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "4")
    env.setdefault("MKL_NUM_THREADS", "4")
    os.execve(sys.executable, [sys.executable, "-u", str(target / RUNNER_REL), "--frozen-worker",
                             "--updates", str(args.updates), "--out-dir", str(out)], env)


def train(args):
    out = args.out_dir.resolve()
    manifest = json.loads((out / "source_manifest.json").read_text())
    assert ROOT == Path(manifest["frozen_code_root"])
    assert args.updates == manifest["updates"]
    for relative, expected in manifest["files"].items():
        assert sha(ROOT / relative) == expected, relative
    for item in manifest["inputs"]["runtime_assets"].values():
        assert sha(item["path"]) == item["sha256"], item["path"]
    os.chdir(manifest["original_repository"])
    sys.path.insert(0, str(ROOT / "mimickit"))
    import torch
    import run as production_run
    from learning.base_agent import BaseAgent
    torch.set_num_threads(4)
    total_samples = args.updates * NUM_ENVS * STEPS
    track = {}
    original_build = production_run.build_agent
    original_env = production_run.build_env
    original_iter = BaseAgent._train_iter
    original_log = BaseAgent._log_train_info

    def scalar(value):
        if torch.is_tensor(value):
            assert value.numel() == 1 and bool(torch.isfinite(value).all())
            value = value.detach().item()
        if isinstance(value, (float, int)):
            assert math.isfinite(value)
        return value

    def fingerprint(module, prefixes):
        digest = hashlib.sha256()
        for name, value in sorted(module.state_dict().items()):
            if name.startswith(prefixes):
                value = value.detach().cpu().contiguous()
                digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
                digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def layout(agent):
        model = agent._model
        assert int(model._monotone_schema_version) == 2
        assert not any(type(module).__name__ == "_SpectralNorm" for module in model._disc_energy.modules())
        diagonal = []
        for index, dim in zip((0, 2, 4), (7, 341, 512)):
            layer = model._disc_energy[index]
            parametrizations = layer.parametrizations.weight
            assert len(parametrizations) == 2
            assert type(parametrizations[0]).__name__ == "PositiveWeight"
            assert type(parametrizations[1]).__name__ == "PositiveDiagonalRescaling"
            assert parametrizations[1].log_q.numel() == dim
            diagonal.append(parametrizations[1].log_q)
        optimizer_params = [p for group in agent._disc_optimizer._optimizer.param_groups for p in group["params"]]
        assert len({id(p) for p in optimizer_params}) == len(optimizer_params)
        assert {id(p) for p in model.get_disc_params()} == {id(p) for p in optimizer_params}
        assert all(sum(p is q for p in optimizer_params) == 1 for q in diagonal)
        return diagonal

    def capture_env(*values, **kwargs):
        env = original_env(*values, **kwargs)
        track["env"] = env
        return env

    def capture_agent(*values, **kwargs):
        agent = original_build(*values, **kwargs)
        track["agent"] = agent
        diagonal = layout(agent)
        optimizers = {}
        for name, opt in agent._get_optimizers().items():
            assert isinstance(opt._optimizer, torch.optim.SGD) and opt.get_steps() == 0
            optimizers[name] = {"class": "SGD", "steps": 0,
                                "groups": [{k: v for k, v in group.items() if k != "params"}
                                           for group in opt._optimizer.param_groups]}
        initial = {"from_zero": True, "resumed": False, "schema": 2,
                   "actor_initial_sha256": fingerprint(agent._model, ("_actor_layers.", "_action_dist.")),
                   "critic_initial_sha256": fingerprint(agent._model, ("_critic_layers.", "_critic_out.")),
                   "disc_parameter_count": sum(p.numel() for p in agent._model.get_disc_params()),
                   "diagonal_parameter_count": sum(q.numel() for q in diagonal),
                   "optimizers": optimizers, "checkpoint_extra": agent._get_checkpoint_extra_state()}
        expected = json.loads((out / "expected_initialization.json").read_text())
        initial["baseline_initialization_comparison_available"] = expected is not None
        if expected is not None:
            assert all(initial[key] == expected[key] for key in ("actor_initial_sha256", "critic_initial_sha256"))
            initial["baseline_actor_critic_initialization_matches"] = True
        save(out / "initial_state.json", initial)
        return agent

    def timed_iter(agent):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.monotonic()
        info = original_iter(agent)
        torch.cuda.synchronize()
        track["iteration_seconds"] = time.monotonic() - started
        track["peak_torch_allocated_MB"] = torch.cuda.max_memory_allocated() / 2**20
        return info

    def pressure(agent, train_info):
        started = time.monotonic()
        model = agent._model
        previous = model.training
        model.eval()
        try:
            diagonal = layout(agent)
            for name, parameter in agent.named_parameters():
                assert bool(torch.isfinite(parameter).all()), name
                if parameter.grad is not None:
                    assert bool(torch.isfinite(parameter.grad).all()), name + ".grad"
            for key in ("disc_grad_penalty", "disc_pos_grad_penalty", "disc_neg_grad_penalty"):
                assert scalar(train_info[key]) == 0
            scale = agent._disc_obs_norm.get_abs_mean().detach()
            assert scale.shape == (7,) and bool(torch.isfinite(scale).all()) and bool((scale >= 0).all())
            effective_scale = scale.clamp_min(agent._disc_obs_norm._min_diff)
            assert bool((effective_scale >= 1e-4).all())
            with torch.no_grad():
                norms = [float(torch.linalg.matrix_norm(model._disc_energy[index].weight.double(), ord=2))
                         for index in (0, 2, 4)]
                assert max(norms) <= 1.0001, norms
                assert all(bool((model._disc_energy[index].weight >= 0).all()) for index in (0, 2, 4))
                for key in ("obs", "action", "reward", "disc_obs", "disc_obs_demo"):
                    for chunk in agent._exp_buffer.get_data_flat(key).split(4096):
                        assert bool(torch.isfinite(chunk).all()), key
                policy = agent._exp_buffer.get_data_flat("disc_obs")
                reference = agent._exp_buffer.get_data_flat("disc_obs_demo")
                sums = policy.new_zeros(7)
                for p, r in zip(policy.split(4096), reference.split(4096)):
                    errors = model.pair_to_error_metrics(p, r)
                    assert bool(torch.isfinite(errors).all()) and bool((errors >= 0).all())
                    assert float(errors[:, (1, 3)].max()) <= math.pi + 1e-4
                    sums += errors.sum(0)
                sample = model.pair_to_error_metrics(policy[:64], reference[:64])
                normalized = agent._disc_obs_norm.normalize(sample)
                assert bool(torch.isfinite(normalized).all()) and bool((normalized >= 0).all())
            z = normalized.detach().requires_grad_(True)
            score = model.eval_disc(z)
            gradient = -torch.autograd.grad(score.sum(), z)[0]
            assert bool(torch.isfinite(score).all()) and bool(torch.isfinite(gradient).all())
            assert float(gradient.min()) >= -2e-5
            assert float(gradient.norm(dim=-1).max()) <= 1.0001
            with torch.no_grad():
                zero_score = model.eval_disc(torch.zeros_like(z[:1]))
                assert torch.allclose(zero_score.flatten(), model._anchor_logit.reshape(-1), atol=1e-6, rtol=1e-6)
                improved = model.eval_disc(z.detach() * .99)
                assert bool((improved >= score.detach() - 2e-5).all())
                assert bool((score <= model._anchor_logit + 2e-5).all())
            return {"finite_checks_passed": True, "exact_spectral_norms": norms,
                    "normalized_error_Lipschitz_upper_bound": math.prod(norms),
                    "normalizer_scale": scale.cpu().tolist(),
                    "rollout_ERR_means": dict(zip(METRICS, (sums / len(policy)).cpu().tolist())),
                    "log_q_ranges": [{"min": float(q.detach().min()), "max": float(q.detach().max())} for q in diagonal],
                    "minimum_coordinate_energy_gradient": float(gradient.min()),
                    "maximum_score_gradient_L2": float(gradient.norm(dim=-1).max()),
                    "diagonal_optimizer_coverage": True,
                    "validation_overhead_seconds": time.monotonic() - started}
        finally:
            model.train(previous)

    def log(agent, train_info, test_info, env_diag_info, start_time, test_random_info=None):
        complete = int(agent._iter) + 1
        output = agent._iter % agent._iters_per_output == 0 or agent._sample_count >= total_samples
        assert 1 <= complete <= args.updates
        assert int(agent._sample_count) == complete * NUM_ENVS * STEPS
        assert scalar(train_info["disc_grad_penalty"]) == 0
        row = {"completed_updates": complete, "target_updates": args.updates,
               "samples": int(agent._sample_count), "target_samples": total_samples,
               "iteration_seconds": track["iteration_seconds"], "elapsed_seconds": time.time() - start_time,
               "GP": 0, "train_episode_length": scalar(train_info["mean_ep_len"]),
               "disc_BCE": scalar(train_info["disc_cls_loss"]),
               "reward_mean": scalar(train_info["disc_reward_mean"]),
               "reward_std": scalar(train_info["disc_reward_std"]),
               "advantage_std": scalar(train_info["adv_std"]),
               "peak_torch_allocated_MB": track["peak_torch_allocated_MB"],
               "optimizer_steps": {name: opt.get_steps() for name, opt in agent._get_optimizers().items()}}
        if output:
            track["last_phase_zero_test"] = {
                "completed_updates": complete, "episode_length": scalar(test_info["mean_ep_len"]),
                "ERR": {name: scalar(env_diag_info[name + "_err"]) for name in METRICS},
                "random_start_episode_length": (None if test_random_info is None else scalar(test_random_info["mean_ep_len"]))}
        row["last_phase_zero_test"] = track.get("last_phase_zero_test")
        if complete <= 3 or output:
            audit = {"completed_updates": complete, **pressure(agent, train_info)}
            append(out / "constraint_rounds.jsonl", audit)
            row["last_pressure_check"] = audit
            print("DIAGONAL_CHECK " + json.dumps(audit, sort_keys=True), flush=True)
        append(out / "training_rounds.jsonl", row)
        save(out / "progress.json", row)
        if complete == 1:
            save(out / "status.json", {"status": "running", "pid": os.getpid(), "from_zero": True,
                                       "updates": args.updates, "max_samples": total_samples, "GP": 0})
        return original_log(agent, train_info, test_info, env_diag_info, start_time,
                            test_random_info=test_random_info)

    def checkpoint_check(agent):
        """Verify saved model and SGD state reload; no resumed PPO is executed."""
        checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert checkpoint["trainer_state"]["next_iter"] == args.updates
        assert checkpoint["trainer_state"]["sample_count"] == total_samples
        saved_q = {name: value for name, value in checkpoint["model_state_dict"].items() if name.endswith(".log_q")}
        assert len(saved_q) == 3 and sum(q.numel() for q in saved_q.values()) == 860
        assert all(bool(torch.isfinite(q).all()) for q in saved_q.values())
        previous = agent._model.training
        agent._model.eval()
        try:
            with torch.no_grad():
                probe = torch.linspace(0, 2, 119, device=agent._device).reshape(17, 7)
                before = agent._model.eval_disc(probe).clone()
                agent.load_state_dict(checkpoint["model_state_dict"])
                agent._load_checkpoint_extra_state(checkpoint["agent_extra_state"])
                for name, opt in agent._get_optimizers().items():
                    expected_steps = opt.get_steps()
                    opt.load_state_dict(checkpoint["optimizer_state_dicts"][name])
                    assert opt.get_steps() == expected_steps
                after = agent._model.eval_disc(probe)
                assert torch.equal(before, after)
            layout(agent)
        finally:
            agent._model.train(previous)
        result = {"status": "passed", "saved_next_iter": args.updates,
                  "saved_samples": total_samples, "saved_diagonal_parameters": 860,
                  "model_and_optimizer_state_reload": True, "probe_max_absolute_difference": 0,
                  "continued_PPO_updates": 0, "physics_resume_tested": False}
        save(out / "checkpoint_check.json", result)
        return result

    production_run.build_env = capture_env
    production_run.build_agent = capture_agent
    BaseAgent._train_iter = timed_iter
    BaseAgent._log_train_info = log
    argv = ["diagonal_cmnn", "--mode", "train", "--num_envs", str(NUM_ENVS),
            "--visualize", "false", "--devices", "cuda:0", "--master_port", "6497",
            "--rand_seed", str(SEED), "--out_dir", str(out), "--save_int_models", "true",
            "--max_samples", str(total_samples)]
    for kind in CONFIGS:
        argv.extend(["--" + kind + "_config", str(out / ("frozen_" + kind + ".yaml"))])
    protocol = {"experiment": "CMNN7_positive_learned_diagonal_all_layers", "from_zero": True,
                "resumed": False, "initialization_checkpoint": None, "schema": 2, "GP": 0,
                "updates": args.updates, "num_envs": NUM_ENVS, "steps_per_update": STEPS,
                "max_samples": total_samples, "seed": SEED, "argv": argv,
                "frozen_code_root": str(ROOT), "training_code_monkeypatched": False,
                "observation_hooks": ["build_agent", "build_env", "train_iteration_timing", "logging"],
                "pressure_checks": "first_three_and_original_output_iterations",
                "original_output_interval": 100, "new_loss_weights": False,
                "smoke_pass_does_not_prove_control_advantage": True}
    save(out / "protocol.json", protocol)
    started = time.monotonic()
    save(out / "status.json", {"status": "initializing", "pid": os.getpid(), "from_zero": True,
                               "updates": args.updates, "GP": 0})
    try:
        production_run.main(argv)
        agent = track["agent"]
        assert int(agent._iter) == args.updates and int(agent._sample_count) == total_samples
        checkpoint_result = checkpoint_check(agent)
        for relative, expected in manifest["files"].items():
            assert sha(ROOT / relative) == expected, relative
        save(out / "status.json", {"status": "completed", "updates": args.updates,
                                   "samples": total_samples, "elapsed_seconds": time.monotonic() - started,
                                   "from_zero": True, "GP": 0, "frozen_sources_unchanged": True,
                                   "checkpoint_check": checkpoint_result})
    except BaseException as exc:
        save(out / "status.json", {"status": "failed", "error": repr(exc),
                                   "traceback": traceback.format_exc(), "seconds": time.monotonic() - started})
        raise
    finally:
        env = track.get("env")
        launcher = getattr(getattr(env, "_engine", None), "_app_launcher", None)
        if launcher is not None:
            launcher.app.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--preflight", action="store_true", help="Check files/configuration without torch or GPU")
    parser.add_argument("--frozen-worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if not 3 <= args.updates <= 2000:
        parser.error("--updates must be between 3 and 2000")
    if args.frozen_worker:
        train(args)
    else:
        freeze(args)


if __name__ == "__main__":
    main()
