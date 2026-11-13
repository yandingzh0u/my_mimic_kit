#!/usr/bin/env python3
"""Freeze a CMNN experiment or resume its full training checkpoint.

Examples (use the Isaac Lab Python environment):
  python tools/paper_benchmark/run_dare_diagonal.py --updates 30 --out-dir output/diagnostics/diagonal_smoke
  python tools/paper_benchmark/run_dare_diagonal.py --updates 2000 --out-dir output/paper_benchmark/diagonal_formal
  python tools/paper_benchmark/run_dare_diagonal.py --updates 2000 --out-dir output/paper_benchmark/diagonal_formal --resume

Resume uses the experiment's frozen training sources and configuration. Its
hooks only observe training; no model methods or objectives are replaced.
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
    import fcntl

    out = args.out_dir.resolve()
    run_lock = (out / ".training.lock").open("a")
    fcntl.flock(run_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
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
    from learning.dare_monotone_agent import DAREMonotoneAgent
    from learning.dare_monotone_model import (
        FEATURE_DIM, FEATURE_SLICES, MONOTONE_SCHEMA_VERSION, aggregate_error_metrics)
    torch.set_num_threads(4)
    total_samples = args.updates * NUM_ENVS * STEPS
    track = {}
    resume_checkpoint = None
    resume_dir = None
    if args.resume:
        resume_checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        restored_updates = int(resume_checkpoint["trainer_state"]["next_iter"])
        assert 0 < restored_updates < args.updates
        assert resume_checkpoint["trainer_state"]["sample_count"] == restored_updates * NUM_ENVS * STEPS
        assert resume_checkpoint["agent_extra_state"]["monotone_disc"]["version"] == MONOTONE_SCHEMA_VERSION
        for kind in CONFIGS:
            expected = resume_checkpoint["metadata"]["checkpoint_context"][kind + "_config_sha256"]
            assert sha(out / ("frozen_" + kind + ".yaml")) == expected
        segment = int(resume_checkpoint["trainer_state"].get("resume_count", 0)) + 1
        resume_dir = out / "resume_segments" / f"{segment:03d}"
        attempt = 1
        while resume_dir.exists():
            attempt += 1
            resume_dir = out / "resume_segments" / f"{segment:03d}_attempt{attempt:03d}"
        resume_dir.mkdir(parents=True, exist_ok=False)
        # Checkpoints are replaced atomically, so a hard link preserves this
        # recovery point without duplicating its replay-buffer storage.
        os.link(out / "checkpoint.pt", resume_dir / "checkpoint.pt")
        for name in ("progress.json", "status.json", "training_rounds.jsonl", "constraint_rounds.jsonl",
                     "train_metrics.jsonl", "log.txt", "protocol.json"):
            if (out / name).is_file():
                shutil.copyfile(out / name, resume_dir / name)
        shutil.copyfile(Path(__file__), resume_dir / "runner.py")
        # Keep the pre-shutdown records, but remove updates whose weights were
        # never checkpointed from the active timeline before appending a replay.
        for name, key, offset in (("training_rounds.jsonl", "completed_updates", 0),
                                  ("constraint_rounds.jsonl", "completed_updates", 0),
                                  ("train_metrics.jsonl", "iteration", 1)):
            path = out / name
            retained = [line for line in path.read_text().splitlines()
                        if int(json.loads(line)[key]) + offset <= restored_updates]
            path.write_text("\n".join(retained) + "\n")
        restored_row = json.loads((out / "training_rounds.jsonl").read_text().splitlines()[-1])
        assert restored_row["completed_updates"] == restored_updates
        track["last_phase_zero_test"] = restored_row["last_phase_zero_test"]
        save(out / "progress.json", restored_row)
        track["resumed_at_update"] = restored_updates
        print(f"FULL_CHECKPOINT_RESUME saved_updates={restored_updates} target_updates={args.updates}", flush=True)
    original_build = production_run.build_agent
    original_env = production_run.build_env
    original_train = production_run.train
    original_iter = BaseAgent._train_iter
    original_log = BaseAgent._log_train_info
    original_rewards = DAREMonotoneAgent._compute_rewards

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
        assert FEATURE_DIM == 64 and MONOTONE_SCHEMA_VERSION == 4
        assert int(model._monotone_schema_version) == MONOTONE_SCHEMA_VERSION
        assert not any(type(module).__name__ == "_SpectralNorm" for module in model._disc_energy.modules())
        diagonal = []
        for index, dim in zip((0, 2, 4), (FEATURE_DIM, 341, 512)):
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

    def fixed_probe(agent):
        # Deterministic normalized inputs consume no RNG and are independent
        # of the moving physical-error normalizer. Tracking anchored energy
        # separately prevents a changing scalar bias from hiding a frozen net.
        identity = torch.eye(FEATURE_DIM, device=agent._device)
        return torch.cat((identity.new_zeros((1, FEATURE_DIM)), identity,
                          identity.new_full((1, FEATURE_DIM), .1),
                          identity.new_ones((1, FEATURE_DIM)),
                          identity.new_full((1, FEATURE_DIM), 2)), 0)

    def anchored_probe_energy(agent):
        with torch.no_grad():
            score = agent._model.eval_disc(fixed_probe(agent)).squeeze(-1)
            return (agent._model._anchor_logit - score).detach().clone()

    def capture_env(*values, **kwargs):
        env = original_env(*values, **kwargs)
        track["env"] = env
        return env

    def capture_agent(*values, **kwargs):
        agent = original_build(*values, **kwargs)
        track["agent"] = agent
        diagonal = layout(agent)
        if args.resume:
            return agent
        optimizers = {}
        for name, opt in agent._get_optimizers().items():
            assert isinstance(opt._optimizer, torch.optim.SGD) and opt.get_steps() == 0
            optimizers[name] = {"class": "SGD", "steps": 0,
                                "groups": [{k: v for k, v in group.items() if k != "params"}
                                           for group in opt._optimizer.param_groups]}
        initial = {"from_zero": True, "resumed": False, "schema": MONOTONE_SCHEMA_VERSION,
                   "discriminator_input_dim": FEATURE_DIM, "physical_metric_count": len(METRICS),
                   "initial_normalizer_scale": agent._disc_obs_norm.get_abs_mean().detach().cpu().tolist(),
                   "initial_normalizer_count": int(agent._disc_obs_norm.get_count()),
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
        track["initial_probe_energy"] = anchored_probe_energy(agent)
        track["previous_probe_energy"] = track["initial_probe_energy"].clone()
        initial["fixed_probe_initial_energy"] = track["initial_probe_energy"].cpu().tolist()
        save(out / "initial_state.json", initial)
        return agent

    def continue_train(agent, **kwargs):
        # production_run calls this only after BaseAgent.resume has restored
        # all training state. Verify it before the first resumed rollout.
        assert agent._resume_pending
        assert agent._iter == restored_updates
        assert agent._sample_count == resume_checkpoint["trainer_state"]["sample_count"]
        layout(agent)
        for name, value in agent.state_dict().items():
            assert torch.equal(value.detach().cpu(), resume_checkpoint["model_state_dict"][name]), name
        steps = {}
        for name, opt in agent._get_optimizers().items():
            expected = resume_checkpoint["optimizer_state_dicts"][name]["steps"]
            assert opt.get_steps() == expected
            steps[name] = expected
        initial = json.loads((out / "initial_state.json").read_text())
        track["initial_probe_energy"] = torch.tensor(initial["fixed_probe_initial_energy"], device=agent._device)
        track["previous_probe_energy"] = anchored_probe_energy(agent)
        save(resume_dir / "restored_state.json", {
            "status": "verified", "restored_updates": restored_updates,
            "restored_samples": agent._sample_count, "optimizer_steps": steps,
            "normalizer_count": int(agent._disc_obs_norm.get_count()),
            "normalizer_scale": agent._disc_obs_norm.get_abs_mean().detach().cpu().tolist(),
            "model_state_exact_match": True, "schema": MONOTONE_SCHEMA_VERSION,
            "physics_state_restored": False, "new_rollout_after_resume": True})
        return original_train(agent=agent, **kwargs)

    def observe_first_rewards(agent):
        # Observe the first real rollout BEFORE the original reward computation,
        # discriminator optimizer or normalizer update. No RNG, state update,
        # diagnostic backward pass, data replacement or extra optimizer step.
        if "first_rollout_before_update" not in track:
            with torch.no_grad():
                policy = agent._exp_buffer.get_data_flat("disc_obs")
                reference = agent._exp_buffer.get_data_flat("disc_obs_demo")
                selection = torch.linspace(0, len(policy) - 1, min(4096, len(policy)),
                                           device=policy.device).long()
                features = agent._model.pair_to_error_metrics(policy[selection], reference[selection])
                normalized = agent._disc_obs_norm.normalize(features)
                score = agent._model.eval_disc(normalized).squeeze(-1)
                energy = agent._model._anchor_logit - score
                from learning.add_agent import calc_unscaled_disc_reward
                reward = calc_unscaled_disc_reward(score)
                quantiles = normalized.new_tensor([0., .5, .95, .99, 1.])
                distributions = {}
                for name, values in (("anchored_energy", energy), ("score", score),
                                     ("reward", reward),
                                     ("normalized_input_L2", normalized.norm(dim=-1))):
                    assert bool(torch.isfinite(values).all()), name
                    distributions[name] = dict(zip(
                        ("min", "P50", "P95", "P99", "max"),
                        torch.quantile(values, quantiles).cpu().tolist()))
                observation = {
                    "stage": "first_real_rollout_before_reward_optimizer_and_normalizer_update",
                    "sampled_rollout_states": len(selection),
                    "discriminator_optimizer_steps": agent._disc_optimizer.get_steps(),
                    "normalizer_count": int(agent._disc_obs_norm.get_count()),
                    "normalizer_scale": agent._disc_obs_norm.get_abs_mean().detach().cpu().tolist(),
                    "sampled_rollout_distributions": distributions,
                    "normalized_feature_quantiles_rows_min_P50_P95_P99_max":
                        torch.quantile(normalized, quantiles, dim=0).cpu().tolist(),
                    "reward_exact_zero_fraction": float((reward == 0).float().mean()),
                    "reward_upper_clamp_fraction": float((torch.sigmoid(score) >= .9999).float().mean()),
                }
                assert observation["discriminator_optimizer_steps"] == 0
                assert observation["normalizer_count"] == 0
                track["first_rollout_before_update"] = observation
                save(out / "first_rollout_before_update.json", observation)
                print("CMNN64_FIRST_ROLLOUT " + json.dumps(observation, sort_keys=True), flush=True)
        return original_rewards(agent)

    def timed_iter(agent):
        track["normalizer_scale_before_iteration"] = agent._disc_obs_norm.get_abs_mean().detach().clone()
        track["normalizer_count_before_iteration"] = int(agent._disc_obs_norm.get_count())
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
            real_gradients = {}
            for name, parameter in agent.named_parameters():
                assert bool(torch.isfinite(parameter).all()), name
                if parameter.grad is not None:
                    assert bool(torch.isfinite(parameter.grad).all()), name + ".grad"
            # These are gradients left by the actual final training minibatch,
            # before this audit computes input derivatives. No diagnostic loss
            # or additional optimizer step is introduced.
            for name, parameter in model._disc_energy.named_parameters():
                gradient = parameter.grad
                assert gradient is not None, "Missing real training gradient: " + name
                grad_norm = float(gradient.detach().double().norm())
                real_gradients[name] = {"L2": grad_norm,
                                        "max_abs": float(gradient.detach().abs().max())}
                if name.endswith(("weight.original", "log_q")):
                    assert grad_norm > 0, "Zero real training gradient: " + name
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
                    assert errors.shape[-1] == FEATURE_DIM
                    assert bool(torch.isfinite(errors).all()) and bool((errors >= 0).all())
                    assert float(errors[:, 1].max()) <= math.pi + 1e-4
                    assert float(errors[:, 16:30].max()) <= math.pi + 1e-4
                    metrics = aggregate_error_metrics(errors)
                    assert metrics.shape == (len(errors), 7)
                    sums += metrics.sum(0)
                selection = torch.linspace(0, len(policy) - 1, min(4096, len(policy)),
                                           device=policy.device).long()
                sample = model.pair_to_error_metrics(policy[selection], reference[selection])
                normalized = agent._disc_obs_norm.normalize(sample)
                assert normalized.shape[-1] == FEATURE_DIM
                assert bool(torch.isfinite(normalized).all()) and bool((normalized >= 0).all())
                rollout_score = model.eval_disc(normalized).squeeze(-1)
                rollout_energy = model._anchor_logit - rollout_score
                previous_scale = track["normalizer_scale_before_iteration"].clamp_min(
                    agent._disc_obs_norm._min_diff)
                previous_feature_scale = torch.empty(FEATURE_DIM, device=sample.device,
                                                     dtype=sample.dtype)
                for index, name in enumerate(METRICS):
                    part = FEATURE_SLICES[name]
                    previous_feature_scale[part] = previous_scale[index] * math.sqrt(
                        part.stop - part.start)
                # Same physical samples and post-update model: this comparison
                # isolates the normalizer update, not the optimizer's effect.
                previous_normalized = (sample / previous_feature_scale).clamp(
                    -agent._disc_obs_norm._clip, agent._disc_obs_norm._clip)
                previous_scale_score = model.eval_disc(previous_normalized).squeeze(-1)
                previous_scale_energy = model._anchor_logit - previous_scale_score
                assert bool(torch.isfinite(rollout_score).all())
                assert bool(torch.isfinite(rollout_energy).all())
                from learning.add_agent import calc_unscaled_disc_reward
                rollout_reward = calc_unscaled_disc_reward(rollout_score)
                reward_zero_fraction = float((rollout_reward == 0).float().mean())
                reward_cap_fraction = float((torch.sigmoid(rollout_score) >= .9999).float().mean())
                quantiles = normalized.new_tensor([0., .5, .95, .99, 1.])
                distributions = {
                    name: dict(zip(("min", "P50", "P95", "P99", "max"),
                                   torch.quantile(values, quantiles).cpu().tolist()))
                    for name, values in (("anchored_energy", rollout_energy),
                                         ("score", rollout_score), ("reward", rollout_reward),
                                         ("normalized_input_L2", normalized.norm(dim=-1)))}
                normalizer_effect = {
                    "comparison": "same_rollout_samples_and_same_postupdate_model",
                    "count_before_iteration": track["normalizer_count_before_iteration"],
                    "count_after_iteration": int(agent._disc_obs_norm.get_count()),
                    "scale_before_iteration": track["normalizer_scale_before_iteration"].cpu().tolist(),
                    "scale_after_iteration": scale.cpu().tolist(),
                    "score_before_normalizer_update": dict(zip(
                        ("min", "P50", "P95", "P99", "max"),
                        torch.quantile(previous_scale_score, quantiles).cpu().tolist())),
                    "energy_before_normalizer_update": dict(zip(
                        ("min", "P50", "P95", "P99", "max"),
                        torch.quantile(previous_scale_energy, quantiles).cpu().tolist())),
                    "score_after_normalizer_update": distributions["score"],
                    "energy_after_normalizer_update": distributions["anchored_energy"],
                    "score_change_max_abs": float((rollout_score - previous_scale_score).abs().max()),
                }
                feature_quantiles = torch.quantile(normalized, quantiles, dim=0).cpu().tolist()
                current_probe = anchored_probe_energy(agent)
                from_initial = (current_probe - track["initial_probe_energy"]).abs()
                since_previous = (current_probe - track["previous_probe_energy"]).abs()
                assert float(from_initial.max()) > 0, "Anchored energy stayed exactly at initialization"
                assert float(since_previous.max()) > 0, "Anchored energy did not change since previous pressure check"
                track["previous_probe_energy"] = current_probe.clone()
            z = normalized[:64].detach().requires_grad_(True)
            score = model.eval_disc(z)
            gradient = -torch.autograd.grad(score.sum(), z)[0]
            assert bool(torch.isfinite(score).all()) and bool(torch.isfinite(gradient).all())
            assert float(gradient.min()) >= -2e-5
            assert float(gradient.norm(dim=-1).max()) <= 1.0001
            # The unit bound is in normalized feature space. Report physical
            # coordinate response too; normalization can amplify raw units.
            inverse_feature_scales = agent._disc_obs_norm.normalize(torch.ones_like(z[:1]))
            physical_gradient = gradient * inverse_feature_scales
            with torch.no_grad():
                zero_score = model.eval_disc(torch.zeros_like(z[:1]))
                assert torch.allclose(zero_score.flatten(), model._anchor_logit.reshape(-1), atol=1e-6, rtol=1e-6)
                improved = model.eval_disc(z.detach() * .99)
                assert bool((improved >= score.detach() - 2e-5).all())
                assert bool((score <= model._anchor_logit + 2e-5).all())
                # A separate finite perturbation test covers all coordinates,
                # including feature axes absent from a particular rollout.
                coordinate_base = normalized[0:1].expand(FEATURE_DIM, -1)
                coordinate_improved = (coordinate_base -
                                       torch.eye(FEATURE_DIM, device=z.device) * .01).clamp_min(0)
                coordinate_score = model.eval_disc(coordinate_improved)
                base_score = model.eval_disc(coordinate_base)
                assert bool((coordinate_score >= base_score - 2e-5).all())
            return {"finite_checks_passed": True, "exact_spectral_norms": norms,
                    "discriminator_input_dim": FEATURE_DIM,
                    "normalized_error_Lipschitz_upper_bound": math.prod(norms),
                    "normalizer_scale": scale.cpu().tolist(),
                    "normalizer_update_effect": normalizer_effect,
                    "rollout_ERR_means": dict(zip(METRICS, (sums / len(policy)).cpu().tolist())),
                    "sampled_rollout_distributions": distributions,
                    "sampled_reward_exact_zero_fraction": reward_zero_fraction,
                    "sampled_reward_upper_clamp_fraction": reward_cap_fraction,
                    "normalized_feature_quantiles_rows_min_P50_P95_P99_max": feature_quantiles,
                    "real_last_training_parameter_gradients": real_gradients,
                    "fixed_probe_energy_change_from_initial_max": float(from_initial.max()),
                    "fixed_probe_energy_change_since_previous_check_max": float(since_previous.max()),
                    "log_q_ranges": [{"min": float(q.detach().min()), "max": float(q.detach().max())} for q in diagonal],
                    "minimum_coordinate_energy_gradient": float(gradient.min()),
                    "maximum_score_gradient_L2": float(gradient.norm(dim=-1).max()),
                    "mean_normalized_energy_gradient_by_feature": gradient.mean(0).detach().cpu().tolist(),
                    "mean_physical_energy_gradient_by_feature": physical_gradient.mean(0).detach().cpu().tolist(),
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
               "disc_positive_logit": scalar(train_info["disc_pos_logit"]),
               "disc_negative_logit": scalar(train_info["disc_neg_logit"]),
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
        if complete <= track.get("resumed_at_update", 0) + 3 or output:
            audit = {"completed_updates": complete, **pressure(agent, train_info)}
            append(out / "constraint_rounds.jsonl", audit)
            row["last_pressure_check"] = audit
            print("DIAGONAL_CHECK " + json.dumps(audit, sort_keys=True), flush=True)
        append(out / "training_rounds.jsonl", row)
        save(out / "progress.json", row)
        if complete == track.get("resumed_at_update", 0) + 1:
            save(out / "status.json", {"status": "running", "pid": os.getpid(), "from_zero": not args.resume,
                                       "resumed": args.resume, "restored_updates": track.get("resumed_at_update", 0),
                                       "updates": args.updates, "max_samples": total_samples, "GP": 0})
        return original_log(agent, train_info, test_info, env_diag_info, start_time,
                            test_random_info=test_random_info)

    def checkpoint_check(agent):
        """Verify saved model and SGD state reload; no resumed PPO is executed."""
        checkpoint = torch.load(out / "checkpoint.pt", map_location="cpu", weights_only=False)
        assert checkpoint["trainer_state"]["next_iter"] == args.updates
        assert checkpoint["trainer_state"]["sample_count"] == total_samples
        saved_q = {name: value for name, value in checkpoint["model_state_dict"].items() if name.endswith(".log_q")}
        diagonal_count = FEATURE_DIM + 341 + 512
        assert len(saved_q) == 3 and sum(q.numel() for q in saved_q.values()) == diagonal_count
        assert all(bool(torch.isfinite(q).all()) for q in saved_q.values())
        previous = agent._model.training
        agent._model.eval()
        try:
            with torch.no_grad():
                probe = torch.linspace(0, 2, 17 * FEATURE_DIM, device=agent._device).reshape(17, FEATURE_DIM)
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
                  "saved_samples": total_samples, "saved_diagonal_parameters": diagonal_count,
                  "model_and_optimizer_state_reload": True, "probe_max_absolute_difference": 0,
                  "continued_PPO_updates": 0, "physics_resume_tested": False}
        save(out / "checkpoint_check.json", result)
        return result

    production_run.build_env = capture_env
    production_run.build_agent = capture_agent
    BaseAgent._train_iter = timed_iter
    BaseAgent._log_train_info = log
    if args.resume:
        production_run.train = continue_train
    else:
        DAREMonotoneAgent._compute_rewards = observe_first_rewards
    argv = ["diagonal_cmnn", "--mode", "train", "--num_envs", str(NUM_ENVS),
            "--visualize", "false", "--devices", "cuda:0", "--master_port", "6497",
            "--rand_seed", str(SEED), "--out_dir", str(out), "--save_int_models", "true",
            "--max_samples", str(total_samples)]
    for kind in CONFIGS:
        argv.extend(["--" + kind + "_config", str(out / ("frozen_" + kind + ".yaml"))])
    if args.resume:
        argv.extend(["--resume_file", str(out / "checkpoint.pt")])
    protocol = {"experiment": "CMNN64_physical_errors_positive_learned_diagonal_group_rms", "from_zero": True,
                "resumed": False, "initialization_checkpoint": None,
                "schema": MONOTONE_SCHEMA_VERSION, "discriminator_input_dim": FEATURE_DIM,
                "shared_physical_normalizer_groups": len(METRICS), "GP": 0,
                "updates": args.updates, "num_envs": NUM_ENVS, "steps_per_update": STEPS,
                "max_samples": total_samples, "seed": SEED, "argv": argv,
                "frozen_code_root": str(ROOT), "training_code_monkeypatched": False,
                "observation_hooks": ["build_agent", "build_env", "train_iteration_timing", "logging",
                                      "first_real_rollout_before_reward_computation"],
                "pressure_checks": "first_three_and_original_output_iterations",
                "original_output_interval": 100, "new_loss_weights": False,
                "smoke_pass_does_not_prove_control_advantage": True}
    if args.resume:
        protocol.update({"from_zero": False, "resumed": True,
                         "restored_updates": restored_updates, "physics_state_restored": False,
                         "checkpoint_sha256": sha(out / "checkpoint.pt"),
                         "runner_sha256": sha(Path(__file__))})
        save(resume_dir / "resume_protocol.json", protocol)
    else:
        save(out / "protocol.json", protocol)
    started = time.monotonic()
    save(out / "status.json", {"status": "initializing", "pid": os.getpid(), "from_zero": not args.resume,
                               "resumed": args.resume, "restored_updates": track.get("resumed_at_update", 0),
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
                                   "from_zero": not args.resume, "resumed": args.resume,
                                   "GP": 0, "frozen_sources_unchanged": True,
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
    global ROOT

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--preflight", action="store_true", help="Check files/configuration without torch or GPU")
    parser.add_argument("--frozen-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--resume", action="store_true", help="Resume the full checkpoint using the original frozen sources")
    args = parser.parse_args()
    if not 3 <= args.updates <= 2000:
        parser.error("--updates must be between 3 and 2000")
    if args.resume:
        ROOT = (args.out_dir.resolve() / "source_snapshot").resolve()
        train(args)
    elif args.frozen_worker:
        train(args)
    else:
        freeze(args)


if __name__ == "__main__":
    main()
