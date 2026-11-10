#!/usr/bin/env python3
"""Frozen follow-up worker, including a separate full-scale preflight.

Copied as launch_train.py into each immutable experiment directory. The parent
waits for a successful three-iteration GPU preflight before starting a fresh
2000-iteration worker. Smoke weights are never loaded into the formal run.
"""
import argparse
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

OUT = Path(__file__).resolve().parent
STATE = {}
RUN = OUT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temp.replace(path)


def write_state():
    write_json(RUN / "execution.json", STATE)


def verify_frozen():
    audit = json.loads((OUT / "launch_audit.json").read_text())
    if sha(__file__) != audit["launcher_sha256"]:
        raise RuntimeError("Frozen launcher hash mismatch")
    manifest = json.loads((OUT / "source_manifest.json").read_text())
    for name, expected in manifest.items():
        if sha(OUT / "source_snapshot" / name) != expected:
            raise RuntimeError("Frozen source mismatch: " + name)
    for name, expected in audit["frozen_config_sha256"].items():
        if sha(OUT / ("frozen_" + name + ".yaml")) != expected:
            raise RuntimeError("Frozen config mismatch: " + name)
    for name, expected in audit["archived_runtime_assets_sha256"].items():
        if sha(OUT / "runtime_assets" / name) != expected:
            raise RuntimeError("Frozen runtime asset mismatch: " + name)
    return audit


def fingerprint(state):
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def worker(args, audit):
    global RUN
    import torch
    RUN = Path(args.output_dir).resolve() if args.output_dir else OUT
    RUN.mkdir(parents=True, exist_ok=True)
    protected = ("checkpoint.pt", "model.pt", "all_iterations.jsonl", "train_metrics.jsonl", "execution.json")
    if any((RUN / name).exists() for name in protected):
        raise RuntimeError("Refusing to overwrite/reuse existing training artifacts: " + str(RUN))
    if not args.smoke and (args.rounds, args.num_envs, args.device) != (2000, 8192, "cuda:0"):
        raise RuntimeError("Formal budget/device cannot be changed")
    STATE.update(status="starting", pid=os.getpid(), started_at_unix=time.time(),
                 job_id=audit["job_id"], from_scratch=True, smoke=args.smoke,
                 target_iterations=args.rounds, target_samples=args.rounds * args.num_envs * 32,
                 source_manifest_sha256=sha(OUT / "source_manifest.json"),
                 agent_config_sha256=sha(OUT / "frozen_agent.yaml"), startup_rounds_passed=False)
    write_state()
    (RUN / "train.pid").write_text(str(os.getpid()) + "\n")
    sys.path.insert(0, str(OUT / "source_snapshot" / "mimickit"))
    os.chdir(audit["repository_root"])
    import run as training
    from learning.add_agent import calc_unscaled_disc_reward
    from learning.dare_body_split_cpl_agent import DAREBodySplitCPLAgent
    from learning.dare_body_split_cpl_model import ConvexPotentialBlock
    from learning.dare_body_split_sn_model import hierarchical_parent_mean
    from learning.dare_monotone_agent import DAREMonotoneAgent
    torch.set_num_threads(2 if args.device == "cpu" else 4)
    original_build = training.build_agent

    def audited_build(arguments, env, device):
        agent = original_build(arguments, env, device)
        model = agent._model
        is_cpl = audit["variant"] == "cpl"
        if not isinstance(agent, DAREBodySplitCPLAgent if is_cpl else DAREMonotoneAgent):
            raise RuntimeError("Wrong production agent")
        if (agent._disc_grad_penalty != 0 or agent._normalizer_samples != 100000000
                or agent._phase_sampler_mode != "none" or agent._disc_group_negatives
                or getattr(env, "_reference_centered_actions", False)
                or hasattr(agent, "_disc_reward_scale") or hasattr(model, "_disc_logit_scale")
                or hasattr(agent, "_calibrate_disc_logit_scale")):
            raise RuntimeError("Unexpected GP, sampler, scaling, calibration or action configuration")
        if agent._disc_optimizer._optimizer._optimizer_step_post_hooks:
            raise RuntimeError("Unexpected discriminator projection hook")
        discriminator_ids = {id(p) for p in model.get_disc_params()}
        if discriminator_ids != {id(p) for p in agent._disc_optimizer._param_list}:
            raise RuntimeError("Incorrect discriminator optimizer coverage")
        actor_critic_ids = {id(p) for p in model.get_actor_params() + model.get_critic_params()}
        if discriminator_ids & actor_critic_ids:
            raise RuntimeError("Discriminator parameters shared with actor/critic")
        if is_cpl:
            groups = model.get_disc_error_groups()
            dims = {name: len(indices) for name, indices in groups}
            if dims != {"root_pose": 9, "body_pos": 45, "body_rot": 84, "velocity": 34}:
                raise RuntimeError("Incorrect fixed four-way partition")
            branches = model._disc_branches.branches
            units = dict(zip(dims, branches))
            depth = audit["cpl_depth"]
            for branch, (_, indices) in zip(branches, groups):
                blocks = [m for m in branch if isinstance(m, ConvexPotentialBlock)]
                if len(blocks) != depth:
                    raise RuntimeError("Incorrect CPL depth")
                if [(m.in_features, m.out_features) for m in (branch[0], branch[2], branch[-1])] != [
                        (len(indices), 341), (341, 512), (512, 1)]:
                    raise RuntimeError("Incorrect rectangular projections/head")
                if any(torch.nn.utils.parametrize.is_parametrized(m, "weight")
                       for m in (branch[0], branch[2], branch[-1])):
                    raise RuntimeError("User requested NO SN on projections/head")
                if any(len(b.linear.parametrizations.weight) != 1 or
                       type(b.linear.parametrizations.weight[0]).__name__ != "_SpectralNorm" for b in blocks):
                    raise RuntimeError("CPL internal tied matrix must retain historical SN")
            parameter_sets = [{id(p) for p in b.parameters()} for b in branches]
            if any(parameter_sets[i] & parameter_sets[j] for i in range(4) for j in range(i)):
                raise RuntimeError("CPL branches share parameters")
            STATE.update(group_dimensions=dims, cpl_depth=depth,
                         projection_head_constraint="none", cpl_internal_constraint="SN")
        else:
            units = {"monotone_energy": model._disc_energy}
            if tuple(agent._disc_obs_norm.get_shape()) != (7,):
                raise RuntimeError("Monotone discriminator requires seven-dimensional normalization")
            STATE.update(discriminator_input="seven_physical_ERRs", num_discriminators=1)
        initial = fingerprint(model.state_dict())
        was_training = model.training
        model.eval()
        with torch.no_grad():
            if is_cpl:
                probe = torch.linspace(-1, 1, 688, device=device).reshape(4, 172)
                expected = hierarchical_parent_mean(calc_unscaled_disc_reward(model.eval_disc(probe)))
                torch.testing.assert_close(agent._calc_disc_rewards(probe), expected, rtol=0, atol=0)
            else:
                probe = torch.linspace(.1, 2., 448, device=device).reshape(64, 7)
                before, after = agent._calc_disc_rewards(probe), agent._calc_disc_rewards(probe * .9)
                if not (after >= before - 1e-6).all():
                    raise RuntimeError("Production monotone reward ordering violated")
                for layer in model._disc_energy:
                    if isinstance(layer, torch.nn.Linear) and not (layer.weight >= 0).all():
                        raise RuntimeError("Negative monotone energy weight")
            if not torch.isfinite(agent._calc_disc_rewards(probe)).all():
                raise RuntimeError("Invalid production reward")
        model.train(was_training)
        if initial != fingerprint(model.state_dict()):
            raise RuntimeError("Reward validation mutated frozen-model state")
        STATE.update(status="running", next_iter=0, sample_count=0,
                     optimizer_parameter_coverage_exact=True, initial_model_sha256=initial,
                     discriminator_parameters=sum(p.numel() for p in model.get_disc_params()),
                     reward_mapping_verified=True, reward_multiplier="none", logit_calibration="none")
        write_state()
        original_iter = agent._train_iter

        def checked_iter():
            first = int(agent._iter) == 0
            previous = {name: fingerprint(dict(unit.named_parameters())) for name, unit in units.items()} if first else {}
            bias = model._anchor_logit.detach().clone() if first and not is_cpl else None
            started = time.monotonic()
            info = original_iter()
            STATE["last_train_iter_seconds"] = time.monotonic() - started
            if first:
                changed = {name: fingerprint(dict(unit.named_parameters())) != previous[name]
                           for name, unit in units.items()}
                if bias is not None:
                    changed["anchor_logit"] = not torch.equal(bias, model._anchor_logit)
                if not all(changed.values()):
                    raise RuntimeError("Discriminator components failed to update: " + str(changed))
                STATE["first_complete_update_changed"] = changed
            for key in ("actor_loss", "critic_loss", "disc_loss"):
                if key not in info:
                    raise RuntimeError("Missing loss: " + key)
            for key, value in info.items():
                if "loss" in key.lower() or key in ("disc_reward_mean", "disc_reward_std"):
                    if not torch.isfinite(torch.as_tensor(value)).all():
                        raise RuntimeError("Nonfinite training output: " + key)
            if not all(torch.isfinite(p).all() for p in model.parameters()):
                raise RuntimeError("Nonfinite model parameters")
            return info

        agent._train_iter = checked_iter
        original_log = agent._log_train_info

        def audited_log(*values, **keywords):
            original_log(*values, **keywords)
            row = {"iteration": int(agent._iter), "samples": int(agent._sample_count)}
            for key, entry in agent._logger.log_current_row.items():
                value = entry.val
                if torch.is_tensor(value) and value.numel() == 1:
                    value = value.item()
                if isinstance(value, numbers.Integral):
                    row[key] = int(value)
                elif isinstance(value, numbers.Real):
                    row[key] = float(value) if math.isfinite(value) else None
            for key in ("Actor_Loss", "Critic_Loss", "Disc_Loss"):
                if row.get(key) is None:
                    raise RuntimeError("Invalid logged loss: " + key)
            row["checked_train_iter_seconds"] = STATE["last_train_iter_seconds"]
            with (RUN / "all_iterations.jsonl").open("a") as stream:
                stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
            STATE.update(next_iter=row["iteration"] + 1, sample_count=row["samples"],
                         last_update_unix=time.time(), startup_rounds_passed=row["iteration"] >= 2)
            write_state()

        agent._log_train_info = audited_log
        return agent

    training.build_agent = audited_build
    arguments = {"mode": "train", "num_envs": args.num_envs, "devices": args.device,
                 "engine_config": str(OUT / "frozen_engine.yaml"),
                 "env_config": str(OUT / "frozen_env.yaml"), "agent_config": str(OUT / "frozen_agent.yaml"),
                 "max_samples": args.rounds * args.num_envs * 32, "rand_seed": 0,
                 "visualize": "false", "save_int_models": "true", "logger": "txt", "out_dir": str(RUN)}
    args_file = RUN / "launch_args.txt"
    args_file.write_text("".join(f"--{key} {value}\n" for key, value in arguments.items()))
    write_json(RUN / "command.json", {"executable": sys.executable, "training_args": arguments})
    training.main(["run.py", "--arg_file", str(args_file)])
    checkpoint = torch.load(RUN / "checkpoint.pt", map_location="cpu", weights_only=False)
    trainer = checkpoint["trainer_state"]
    if (trainer["next_iter"] != args.rounds or trainer["sample_count"] != args.rounds * args.num_envs * 32
            or trainer["resume_count"] != 0):
        raise RuntimeError("Incorrect final checkpoint budget/resume state")
    STATE.update(status="complete", finished_at_unix=time.time(), next_iter=trainer["next_iter"],
                 sample_count=trainer["sample_count"], resume_count=trainer["resume_count"])
    if args.device.startswith("cuda"):
        STATE["peak_torch_reserved_bytes"] = torch.cuda.max_memory_reserved()
    write_state()
    print("FOLLOWUP_TRAINING_COMPLETE", audit["job_id"], "smoke=" + str(args.smoke), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=8192)
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    audit = verify_frozen()
    if args.worker or args.smoke:
        return worker(args, audit)
    # This parent holds no Isaac/GPU context. Each child exits completely before
    # the next starts, releasing PhysX and CUDA allocations.
    if any((OUT / name).exists() for name in ("execution.json", "checkpoint.pt", "all_iterations.jsonl")):
        raise RuntimeError("Formal run has existing artifacts; refusing restart")
    smoke_dir = OUT / "preflight_fullscale"
    smoke_dir.mkdir(exist_ok=True)
    command = [sys.executable, "-u", str(Path(__file__).resolve()), "--worker", "--smoke",
               "--rounds", "3", "--num-envs", "8192", "--output-dir", str(smoke_dir)]
    with (smoke_dir / "console.log").open("x") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        raise RuntimeError("8192-environment GPU preflight failed; formal run blocked")
    report = json.loads((smoke_dir / "execution.json").read_text())
    if (report.get("status") != "complete" or report.get("next_iter") != 3
            or report.get("sample_count") != 786432 or report.get("resume_count") != 0
            or not report.get("startup_rounds_passed")):
        raise RuntimeError("Full-scale preflight did not pass")
    write_json(OUT / "fullscale_gate.json", {"status": "passed", "report": str(smoke_dir / "execution.json"),
               "source_manifest_sha256": sha(OUT / "source_manifest.json"), "passed_at_unix": time.time()})
    result = subprocess.run([sys.executable, "-u", str(Path(__file__).resolve()), "--worker"])
    if result.returncode:
        raise RuntimeError("Formal worker failed")


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        if STATE:
            STATE.update(status="failed", error=repr(error), failed_at_unix=time.time())
            write_state()
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
