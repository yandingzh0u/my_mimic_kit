"""Exercise the real DARE rollout/update path without starting a long run.

Run from the repository root with the IsaacLab Python interpreter. The agent,
environment, optimizer budgets, normalization, action path, and replay behavior
are unmodified. Only evaluation/checkpoint/logging orchestration is omitted.
"""

import argparse
import functools
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mimickit"))


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--env-config", default="data/envs/paper_benchmark/dare_crawl_env.yaml")
    parser.add_argument("--agent-config", default="data/agents/dare_10layer_cpl_climb_agent.yaml")
    parser.add_argument("--engine-config", default="data/engines/isaac_lab_engine.yaml")
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    if args.iterations < 1 or args.num_envs < 1:
        parser.error("iterations and num-envs must be positive")
    return args


def write_report(filename, report):
    temporary = filename.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(filename)


def main():
    args = arguments()
    os.chdir(ROOT)
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "pressure_report.json"
    report = {
        "status": "starting",
        "command": sys.argv,
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True).strip(),
        "num_envs": args.num_envs,
        "iterations_requested": args.iterations,
        "seed": args.seed,
        "device": args.device,
        "orchestration": "Original _train_iter; evaluation, checkpoint writing and logger orchestration omitted",
        "timing_note": "Includes synchronization and finite loss/gradient/parameter checks at every optimizer step",
        "frontend_diagnostics": "First 32 rollout residuals without random sampling; no_grad analysis after training timing and memory measurements",
        "config_sha256": {
            name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
            for name in ("agent_config", "env_config", "engine_config")
        },
        "iterations": [],
    }
    write_report(report_path, report)
    try:
        # Keep the regular entry point's simulator import and construction order.
        import run
        import torch
        from envs.base_env import DoneFlags
        from learning.base_agent import AgentMode

        run_args = run.load_args([
            "coordinate_disc_pressure.py", "--rand_seed", str(args.seed),
            "--agent_config", args.agent_config, "--env_config", args.env_config,
            "--engine_config", args.engine_config, "--visualize", "false",
        ])
        run.mp_util.init(0, 1, args.device, 6456)
        run.set_rand_seed(run_args)
        env = run.build_env(run_args, args.num_envs, args.device, False)
        agent = run.build_agent(run_args, env, args.device)
        agent.set_checkpoint_context(run.build_checkpoint_context(run_args))
        agent.set_mode(AgentMode.TRAIN)
        agent._curr_obs, agent._curr_info = agent._reset_envs()
        agent._init_train()
        frontend = agent._model._disc_layers.frontend
        with torch.no_grad():
            initial_weight = frontend.unit_weight().detach().clone()
            initial_bias = frontend.bias.detach().clone()

        cuda = torch.device(args.device).type == "cuda"

        def synchronize():
            if cuda:
                torch.cuda.synchronize(args.device)

        def tensors_finite(tensors):
            flags = [torch.isfinite(tensor.detach()).all() for tensor in tensors
                     if tensor is not None]
            return bool(torch.stack(flags).all().item()) if flags else True

        optimizers = agent._get_optimizers()
        expected_steps = {}
        minibatches = {}
        for name, budget, epochs in (
            ("_actor_optimizer", agent._actor_batch_size, agent._actor_epochs),
            ("_critic_optimizer", agent._critic_batch_size, agent._critic_epochs),
            ("_disc_optimizer", agent._disc_batch_size, agent._disc_epochs),
        ):
            minibatch = math.ceil(budget * args.num_envs)
            minibatches[name] = minibatch
            expected_steps[name] = math.ceil(
                agent._steps_per_iter * args.num_envs / minibatch) * epochs
        report["steps_per_iter"] = agent._steps_per_iter
        report["optimizer_minibatch_sizes"] = minibatches
        report["disc_negative_forward_batch_size"] = minibatches["_disc_optimizer"] * 2
        report["expected_optimizer_steps_per_iter"] = expected_steps
        report["parameters"] = sum(p.numel() for p in agent._model.parameters())
        report["gpu_name"] = torch.cuda.get_device_name(args.device) if cuda else None
        report["status"] = "running"
        current = {}

        def timed_method(name):
            original = getattr(agent, name)

            @functools.wraps(original)
            def measured(*pos, **kw):
                synchronize()
                start = time.perf_counter()
                result = original(*pos, **kw)
                synchronize()
                current["timings_seconds"][name] = (
                    current["timings_seconds"].get(name, 0.0)
                    + time.perf_counter() - start)
                if cuda:
                    current.setdefault("cuda_running_peak_after_stage_bytes", {})[name] = (
                        torch.cuda.max_memory_allocated(args.device))
                return result

            setattr(agent, name, measured)

        for name in ("_rollout_train", "_build_train_data", "_update_model",
                     "_update_critic", "_update_actor", "_update_disc"):
            timed_method(name)

        def checked_step(name, optimizer):
            original = optimizer.step

            @functools.wraps(original)
            def step(loss):
                if not tensors_finite([loss]):
                    raise FloatingPointError(name + " received nonfinite loss")
                result = original(loss)
                if not math.isfinite(optimizer.get_last_grad_norm()):
                    raise FloatingPointError(name + " produced nonfinite gradient norm")
                if not tensors_finite(optimizer._param_list):
                    raise FloatingPointError(name + " produced nonfinite parameters")
                if not tensors_finite(p.grad for p in optimizer._param_list):
                    raise FloatingPointError(name + " produced nonfinite gradients")
                current["max_gradient_norm"][name] = max(
                    current["max_gradient_norm"].get(name, 0.0),
                    optimizer.get_last_grad_norm())
                return result

            optimizer.step = step

        for name, optimizer in optimizers.items():
            checked_step(name, optimizer)

        @torch.no_grad()
        def frontend_diagnostics():
            weight = frontend.unit_weight()
            norm_error = float((weight.norm(dim=-1) - 1).abs().max().item())
            if norm_error > 1e-5:
                raise AssertionError("Effective token weights lost unit norm")
            weight_delta = torch.linalg.vector_norm(weight - initial_weight)
            bias_delta = torch.linalg.vector_norm(frontend.bias - initial_bias)
            diagnostics = {
                "token_unit_norm_max_error": norm_error,
                "token_weight_change_frobenius": float(weight_delta.item()),
                "token_weight_relative_change": float((weight_delta / initial_weight.norm()).item()),
                "token_bias_change_frobenius": float(bias_delta.item()),
            }
            for name, parameter in (("raw_weight", frontend.raw_weight),
                                    ("bias", frontend.bias)):
                gradient = parameter.grad
                finite = gradient is not None and tensors_finite([gradient])
                if not finite:
                    raise FloatingPointError("Missing/nonfinite token gradient: " + name)
                diagnostics[name + "_last_gradient_finite"] = finite
                diagnostics[name + "_last_gradient_norm"] = float(gradient.norm().item())

            demo = agent._exp_buffer.get_data_flat("disc_obs_demo")[:32]
            observed = agent._exp_buffer.get_data_flat("disc_obs")[:32]
            residual = agent._disc_obs_norm.normalize(demo - observed)
            tokens = residual.unsqueeze(-1) * weight + frontend.bias
            interacted = frontend.interact(tokens)
            centered = tokens - tokens.mean(dim=-2, keepdim=True)
            squared_norms = centered.square().sum(dim=-1, keepdim=True)
            distances = (squared_norms + squared_norms.transpose(-1, -2)
                         - 2 * torch.bmm(centered, centered.transpose(-1, -2)))
            affinity = torch.rsqrt(1 + distances.clamp_min(0))
            off_diagonal = ~torch.eye(frontend.input_dim, dtype=torch.bool,
                                      device=tokens.device)
            similarities = affinity[:, off_diagonal].flatten()
            quantiles = torch.quantile(similarities, tokens.new_tensor([.01, .1, .5, .9, .99]))
            displacement = interacted - tokens
            relative_displacement = displacement.flatten(1).norm(dim=-1) / tokens.flatten(1).norm(
                dim=-1).clamp_min(torch.finfo(tokens.dtype).tiny)
            token_displacements = displacement.norm(dim=-1)
            diagnostics.update({
                "diagnostic_observations": int(residual.shape[0]),
                "similarity_off_diagonal_mean": float(similarities.mean().item()),
                "similarity_off_diagonal_quantiles": dict(zip(
                    ("p01", "p10", "p50", "p90", "p99"), quantiles.tolist())),
                "interaction_relative_displacement_mean": float(relative_displacement.mean().item()),
                "interaction_relative_displacement_max": float(relative_displacement.max().item()),
                "interaction_token_displacement_norm_mean": float(token_displacements.mean().item()),
                "interaction_token_displacement_norm_max": float(token_displacements.max().item()),
            })
            return diagnostics

        for index in range(args.iterations):
            current = {"iteration": index + 1, "timings_seconds": {},
                       "max_gradient_norm": {}}
            previous_steps = {name: optimizer.get_steps()
                              for name, optimizer in optimizers.items()}
            synchronize()
            if cuda:
                torch.cuda.reset_peak_memory_stats(args.device)
            start = time.perf_counter()
            info = agent._train_iter()
            synchronize()
            current["wall_seconds"] = time.perf_counter() - start
            agent._sample_count = agent._update_sample_count()
            agent._iter += 1
            current["samples_total"] = int(agent._sample_count)
            current["optimizer_steps"] = {
                name: optimizer.get_steps() - previous_steps[name]
                for name, optimizer in optimizers.items()}
            if current["optimizer_steps"] != expected_steps:
                raise AssertionError("Unexpected optimizer update budget: "
                                     + str(current["optimizer_steps"]))
            scalar_info = {}
            # PPO intentionally emits NaN for a termination-conditioned mean,
            # standard deviation or quantile when that exact subset is empty.
            # Admit only those diagnostics, confirming both count and mask.
            empty_diagnostic_keys = set()
            done = agent._exp_buffer.get_data("done")
            for group, flag in (("null", DoneFlags.NULL.value),
                                ("fail", DoneFlags.FAIL.value),
                                ("succ", DoneFlags.SUCC.value),
                                ("time", DoneFlags.TIME.value)):
                prefix = "td_target_" + group
                count = info.get(prefix + "_count")
                if torch.is_tensor(count):
                    count = count.detach().item()
                if count == 0 and not bool((done == flag).any().item()):
                    empty_diagnostic_keys.update(prefix + "_" + statistic
                                                 for statistic in ("mean", "std", "p10", "p50", "p90"))
            undefined_metrics = {}
            for name, value in info.items():
                if torch.is_tensor(value):
                    if value.numel() != 1:
                        continue
                    value = value.detach().item()
                if isinstance(value, (int, float)):
                    if not math.isfinite(value):
                        if math.isnan(value) and name in empty_diagnostic_keys:
                            scalar_info[name] = None
                            undefined_metrics[name] = "Empty termination subset verified by count=0 and rollout done mask"
                            continue
                        else:
                            raise FloatingPointError("Nonfinite training metric: " + name)
                    scalar_info[name] = value
            current["training_metrics"] = scalar_info
            current["undefined_diagnostics"] = undefined_metrics
            current["all_parameters_finite"] = tensors_finite(agent._model.parameters())
            if not current["all_parameters_finite"]:
                raise FloatingPointError("Nonfinite model parameters")
            if cuda:
                current["cuda_peak_allocated_bytes"] = torch.cuda.max_memory_allocated(args.device)
                current["cuda_peak_reserved_bytes"] = torch.cuda.max_memory_reserved(args.device)
            current["frontend"] = frontend_diagnostics()
            report["iterations"].append(current)
            write_report(report_path, report)
            print("PRESSURE_ITER " + json.dumps(current, allow_nan=False), flush=True)

        report["status"] = "passed"
        write_report(report_path, report)
        print("PRESSURE_COMPLETE " + str(report_path), flush=True)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = type(error).__name__ + ": " + str(error)
        report["traceback"] = traceback.format_exc()
        if "current" in locals() and current:
            report["partial_iteration"] = current
        write_report(report_path, report)
        raise
if __name__ == "__main__":
    exit_code = 0
    try:
        main()
    except SystemExit as error:
        exit_code = error.code if isinstance(error.code, int) else 1
    except BaseException:
        traceback.print_exc()
        exit_code = 1
    # Match tools/render/render_motion_frames.py: Isaac Sim shutdown can hang
    # after all reports are saved, retaining GPU memory indefinitely.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)
