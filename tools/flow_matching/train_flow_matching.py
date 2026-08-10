import json
import random
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

from learning.flow_matching.flow_matching_model import FlowMatchingModel
from tools.diffusion_model.motion_prior_dataset import MotionPriorData


CHECKPOINT_VERSION = 1
MODEL_TYPE = "unconditional_flow_matching"


def fix_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def quantile_summary(values):
    values = values.detach().float().cpu()
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "mean": values.mean().item(),
        "std": values.std(unbiased=False).item(),
        "q05": quantiles[0].item(),
        "q25": quantiles[1].item(),
        "median": quantiles[2].item(),
        "q75": quantiles[3].item(),
        "q95": quantiles[4].item(),
    }


def build_validation_panel(
    model,
    raw_expert,
    num_obs_steps,
    frame_dim,
    num_joints,
    env_config,
    config,
    seed,
):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    raw_windows = raw_expert.reshape(-1, num_obs_steps, frame_dim)
    expert = model.normalize(raw_windows)
    expert_cpu = expert.detach().cpu()
    num_samples = expert_cpu.shape[0]

    permutations = torch.stack(
        [torch.randperm(num_obs_steps, generator=generator) for _ in range(num_samples)]
    )
    row_ids = torch.arange(num_samples).unsqueeze(1)
    temporal_shuffle = expert_cpu[row_ids, permutations]

    feature_noise_std = float(config.get("validation_feature_noise_std", 0.35))
    feature_noise = expert_cpu + feature_noise_std * torch.randn(
        expert_cpu.shape, generator=generator, dtype=expert_cpu.dtype
    )

    key_bodies = env_config.get("key_bodies", [])
    foot_body_indices = [
        index for index, body_name in enumerate(key_bodies) if "foot" in body_name.lower()
    ]
    if not foot_body_indices:
        raise ValueError("R1 foot-slide validation requires foot entries in key_bodies")
    root_position_dim = 3 if env_config.get("root_height_obs", True) else 2
    key_position_start = root_position_dim + 6 + 6 * (num_joints - 1)
    foot_xy_indices = [
        key_position_start + 3 * body_index + axis
        for body_index in foot_body_indices
        for axis in (0, 1)
    ]
    if max(foot_xy_indices) >= frame_dim:
        raise ValueError("computed foot feature indices exceed the discriminator frame size")

    foot_slide_distance = float(config.get("validation_foot_slide_distance", 0.25))
    angles = 2.0 * np.pi * torch.rand(
        (raw_windows.shape[0],), generator=generator, dtype=raw_windows.dtype
    )
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).to(
        raw_windows.device
    )
    progress = torch.linspace(
        0.0,
        1.0,
        num_obs_steps,
        device=raw_windows.device,
        dtype=raw_windows.dtype,
    )
    offsets = foot_slide_distance * progress[None, :, None] * directions[:, None, :]
    foot_slide_raw = raw_windows.clone()
    for body_index in foot_body_indices:
        xy_start = key_position_start + 3 * body_index
        foot_slide_raw[:, :, xy_start : xy_start + 2] += offsets
    foot_slide = model.normalize(foot_slide_raw).detach().cpu()

    severe_random_std = float(config.get("validation_severe_random_std", 3.0))
    severe_random = severe_random_std * torch.randn(
        expert_cpu.shape, generator=generator, dtype=expert_cpu.dtype
    )

    return {
        "expert": expert_cpu.reshape(num_samples, -1),
        "temporal_shuffle": temporal_shuffle.reshape(num_samples, -1),
        "feature_noise": feature_noise.reshape(num_samples, -1),
        "foot_slide": foot_slide.reshape(num_samples, -1),
        "severe_random": severe_random.reshape(num_samples, -1),
    }


@torch.no_grad()
def batched_mismatch(model, samples, times, base_noise, device, batch_size):
    mismatch = []
    for start in range(0, samples.shape[0], batch_size):
        batch = samples[start : start + batch_size].to(device)
        values = model.aggregate_mismatch(
            batch,
            times=times,
            base_noise=base_noise,
            use_ema=True,
        )
        if values.ndim != 1 or values.shape[0] != batch.shape[0]:
            raise RuntimeError(
                "aggregate_mismatch must return one scalar per sample; "
                f"got {tuple(values.shape)} for batch {tuple(batch.shape)}"
            )
        mismatch.append(values.detach().cpu())
    return torch.cat(mismatch)


@torch.no_grad()
def validate_prior(model, panel, times, base_noise, device, config):
    model.eval()
    eval_batch_size = int(config.get("validation_batch_size", config["batch_size"]))
    raw = {
        name: batched_mismatch(model, samples, times, base_noise, device, eval_batch_size)
        for name, samples in panel.items()
    }

    calibration_floor = float(config.get("calibration_floor", 1e-6))
    expert_scale = max(raw["expert"].median().item(), calibration_floor)
    calibrated = {name: values / expert_scale for name, values in raw.items()}

    expert = raw["expert"]
    perturbations = {}
    min_win_rate = float(config.get("validation_min_win_rate", 0.6))
    all_perturbations_pass = True
    for name in ("temporal_shuffle", "feature_noise", "foot_slide"):
        values = raw[name]
        win_rate = (expert < values).float().mean().item()
        median_ratio = values.median().item() / max(expert.median().item(), calibration_floor)
        passed = values.median().item() > expert.median().item() and win_rate >= min_win_rate
        perturbations[name] = {
            "expert_lower_win_rate": win_rate,
            "median_ratio_to_expert": median_ratio,
            "passed": passed,
        }
        all_perturbations_pass = all_perturbations_pass and passed

    severe = raw["severe_random"]
    severe_ratio = severe.median().item() / max(expert.median().item(), calibration_floor)
    severe_win_rate = (expert < severe).float().mean().item()
    severe_min_ratio = float(config.get("validation_severe_min_ratio", 2.0))
    severe_passed = severe_ratio >= severe_min_ratio and severe_win_rate >= min_win_rate
    perturbations["severe_random"] = {
        "expert_lower_win_rate": severe_win_rate,
        "median_ratio_to_expert": severe_ratio,
        "passed": severe_passed,
    }

    ratios = [entry["median_ratio_to_expert"] for entry in perturbations.values()]
    selection_score = min(ratios)
    metrics = {
        "expert_scale": expert_scale,
        "raw": {name: quantile_summary(values) for name, values in raw.items()},
        "calibrated": {name: quantile_summary(values) for name, values in calibrated.items()},
        "perturbations": perturbations,
        "gate_passed": all_perturbations_pass and severe_passed,
        "selection_score": selection_score,
    }
    model.train()
    return metrics


def checkpoint_payload(model, config, metadata, calibration, iteration, validation):
    return {
        "format_version": CHECKPOINT_VERSION,
        "model_type": MODEL_TYPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "model_state_dict": model.state_dict(),
        "model_config": dict(config),
        "metadata": dict(metadata),
        "calibration": {
            "times": calibration["times"].detach().cpu(),
            "base_noise": calibration["base_noise"].detach().cpu(),
            "expert_scale": float(calibration["expert_scale"]),
        },
        "offline_validation": validation,
    }


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def train(cfg_path, out_dir, device="cuda", max_iters=None):
    if not out_dir:
        raise ValueError("--out_dir must be non-empty")

    cfg_path = Path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if max_iters is not None:
        config["num_iterations"] = int(max_iters)

    seed = int(config.get("seed", 0))
    fix_seed(seed)
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out_dir / "source_config.yaml")

    with open(config["env_config"], "r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)

    dataset = MotionPriorData(config, device)
    obs_space = dataset.get_obs_space()
    num_obs_steps = int(env_config["num_disc_obs_steps"])
    input_dim = int(obs_space.shape[-1])
    if input_dim % num_obs_steps != 0:
        raise ValueError(f"input_dim={input_dim} is not divisible by H={num_obs_steps}")
    frame_dim = input_dim // num_obs_steps
    config["input_dim"] = input_dim
    config["input_channel"] = frame_dim

    with open(out_dir / "flow_config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    model = FlowMatchingModel(config, device)
    num_samples_stat = int(config.get("num_samples_stat", 10_000))
    normalization_samples = dataset.fetch_obs_demo(num_samples_stat).detach()
    model.update_normalizer(normalization_samples)
    model.to(device)

    validation_samples = int(config.get("validation_samples", 2_048))
    panel_seed = int(config.get("validation_seed", seed + 10_000))
    fix_seed(panel_seed)
    raw_expert = dataset.fetch_obs_demo(validation_samples).detach()
    panel = build_validation_panel(
        model,
        raw_expert,
        num_obs_steps,
        frame_dim,
        int(dataset._motion_lib.get_num_joints()),
        env_config,
        config,
        panel_seed,
    )

    times = torch.tensor(config.get("reward_times", [0.25, 0.5, 0.75]), device=device)
    if times.shape != (3,) or not torch.all((times > 0) & (times < 1)):
        raise ValueError("reward_times must contain exactly three values strictly between zero and one")
    reward_noise_samples = int(config.get("reward_noise_samples", 1))
    if reward_noise_samples not in (1, 2):
        raise ValueError("R1 supports K=1 or the explicit K=2 antithetic fallback")
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(int(config.get("reward_noise_seed", seed + 20_000)))
    first_noise = torch.randn(
        (1, num_obs_steps, frame_dim), generator=noise_generator
    )
    if reward_noise_samples == 2:
        base_noise = torch.cat((first_noise, -first_noise), dim=0)
    else:
        base_noise = first_noise
    base_noise = base_noise.to(device)

    fix_seed(seed)
    optimizer = optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    grad_clip_norm = float(config.get("grad_clip_norm", 1.0))
    batch_size = int(config["batch_size"])
    num_iterations = int(config["num_iterations"])
    output_iter = int(config.get("output_iter", 2_000))
    log_iter = int(config.get("log_iter", min(output_iter, 100)))

    metadata = {
        "input_dim": input_dim,
        "frame_dim": frame_dim,
        "window_steps": num_obs_steps,
        "time_embed_scale": float(config["time_embed_scale"]),
        "aggregation": "t_squared_weighted_mean",
        "reward_noise_samples": reward_noise_samples,
    }
    metrics_path = out_dir / "metrics.jsonl"
    metrics_path.unlink(missing_ok=True)
    (out_dir / "validation.jsonl").unlink(missing_ok=True)
    for checkpoint_name in ("model.pt", "flow_best.pt", "flow_last.pt"):
        (out_dir / checkpoint_name).unlink(missing_ok=True)
    (out_dir / "offline_validation.json").unlink(missing_ok=True)
    best_score = -float("inf")
    require_gate_pass = bool(config.get("require_gate_pass", True))
    published_checkpoint = False
    loss_sum = 0.0
    grad_norm_sum = 0.0
    window_count = 0

    model.train()
    print(
        f"Flow-SMP prior: H={num_obs_steps}, frame_dim={frame_dim}, input_dim={input_dim}, "
        f"parameters={sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M"
    )

    for iteration in range(1, num_iterations + 1):
        raw_samples = dataset.fetch_obs_demo(batch_size).detach()
        samples = model.normalize(raw_samples.reshape(batch_size, num_obs_steps, frame_dim)).reshape(
            batch_size, -1
        )

        optimizer.zero_grad(set_to_none=True)
        loss = model(samples)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite flow-matching loss at iteration {iteration}: {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        if config.get("model_ema", False):
            model.update_ema()

        loss_sum += loss.item()
        grad_norm_sum += float(grad_norm)
        window_count += 1

        if iteration % log_iter == 0 or iteration == num_iterations:
            record = {
                "iteration": iteration,
                "loss": loss_sum / window_count,
                "grad_norm": grad_norm_sum / window_count,
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(metrics_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
            loss_sum = 0.0
            grad_norm_sum = 0.0
            window_count = 0

        should_validate = iteration % output_iter == 0 or iteration == num_iterations
        if should_validate:
            validation = validate_prior(model, panel, times, base_noise, device, config)
            validation["iteration"] = iteration
            append_jsonl(out_dir / "validation.jsonl", validation)
            calibration = {
                "times": times,
                "base_noise": base_noise,
                "expert_scale": validation["expert_scale"],
            }
            payload = checkpoint_payload(
                model, config, metadata, calibration, iteration, validation
            )
            torch.save(payload, out_dir / "flow_last.pt")

            eligible_for_publish = validation["gate_passed"] or not require_gate_pass
            if eligible_for_publish and validation["selection_score"] > best_score:
                best_score = validation["selection_score"]
                torch.save(payload, out_dir / "flow_best.pt")
                torch.save(payload, out_dir / "model.pt")
                published_checkpoint = True

            with open(out_dir / "offline_validation.json", "w", encoding="utf-8") as stream:
                json.dump(validation, stream, indent=2, sort_keys=True)
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "gate_passed": validation["gate_passed"],
                        "selection_score": validation["selection_score"],
                        "expert_scale": validation["expert_scale"],
                        "perturbations": validation["perturbations"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if not published_checkpoint:
        raise RuntimeError(
            "offline mismatch gate never passed; no PPO-loadable model.pt was published"
        )

    return out_dir / "flow_best.pt"


def main():
    parser = ArgumentParser(description="Train the unconditional R1 Flow-SMP prior")
    parser.add_argument(
        "--cfg_path",
        default="tools/flow_matching/config/flow_single_clip.yaml",
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_iters", type=int)
    args = parser.parse_args()
    train(args.cfg_path, args.out_dir, device=args.device, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
