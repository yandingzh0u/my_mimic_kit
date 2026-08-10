from __future__ import annotations

import hashlib
import json
import math
import random
import shutil
import sys
from argparse import ArgumentParser
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.optim as optim
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

from learning.flow_matching.conditional_flow_matching_model import (
    CONDITIONAL_FLOW_FORMAT_VERSION,
    CONDITIONAL_FLOW_MODEL_TYPE,
    ConditionalFlowMatchingModel,
    conditional_checkpoint_payload,
)
from learning.skill_encoder.skill_encoder_model import LabelFreeSkillEncoder
from tools.flow_matching.conditional_motion_data import (
    CONTEXT_STEPS,
    MOTION_WINDOW_STEPS,
    ConditionalMotionWindowSampler,
)


ENCODER_FORMAT_VERSION = 1
ENCODER_MODEL_TYPE = "label_free_skill_encoder"
REQUIRED_COMPARISONS = {
    "matched_vs_wrong_random": ("matched", "wrong_random"),
    "matched_vs_wrong_semantic": ("matched", "wrong_semantic"),
    "matched_vs_null": ("matched", "null"),
    "matched_vs_temporal_shuffle": ("matched", "temporal_shuffle"),
    "matched_vs_feature_noise": ("matched", "feature_noise"),
    "matched_vs_foot_slide": ("matched", "foot_slide"),
    "matched_vs_severe_random": ("matched", "severe_random"),
    "wrong_random_vs_severe_random": ("wrong_random", "severe_random"),
    "wrong_semantic_vs_severe_random": ("wrong_semantic", "severe_random"),
}
SELECTION_COMPARISONS = (
    "matched_vs_null",
    "matched_vs_wrong_random",
    "matched_vs_wrong_semantic",
)


def fix_seed(seed: int) -> None:
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(dict(record), sort_keys=True) + "\n")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantile_summary(values: torch.Tensor) -> dict[str, float]:
    values = values.detach().float().cpu()
    quantiles = torch.quantile(
        values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "q05": float(quantiles[0]),
        "q25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "q75": float(quantiles[3]),
        "q95": float(quantiles[4]),
    }


def sample_null_mask(
    batch_size: int,
    probability: float,
    generator: torch.Generator,
    device: str | torch.device,
) -> torch.Tensor:
    """Sample trainer-owned NULL rows without touching the model API."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("NULL probability must be in [0,1]")
    mask = torch.rand(batch_size, generator=generator) < float(probability)
    return mask.to(device=torch.device(device))


def infer_smp_dimensions(
    obs_shape: Sequence[int], window_steps: int = MOTION_WINDOW_STEPS
) -> tuple[int, int]:
    """Resolve MimicKit's flattened discriminator space into H x F."""
    if len(obs_shape) != 1:
        raise ValueError(f"SMP discriminator observation must be flat, got {obs_shape}")
    input_dim = int(obs_shape[0])
    if input_dim <= 0 or input_dim % window_steps != 0:
        raise ValueError(
            f"flat SMP observation dim {input_dim} is not divisible by H={window_steps}"
        )
    return input_dim, input_dim // window_steps


def _validate_dataset_manifest(
    artifact_manifest: Mapping[str, Any], expected_manifest: Mapping[str, Any]
) -> None:
    if not isinstance(artifact_manifest, Mapping):
        raise RuntimeError("encoder artifact is missing its dataset manifest")
    for key in ("dataset_yaml_sha256", "canonical_manifest_sha256"):
        if artifact_manifest.get(key) != expected_manifest.get(key):
            raise RuntimeError(f"encoder/current dataset manifest mismatch: {key}")
    artifact_clips = artifact_manifest.get("clips")
    expected_clips = expected_manifest.get("clips")
    if not isinstance(artifact_clips, list) or not isinstance(expected_clips, list):
        raise RuntimeError("dataset manifests must carry per-clip records")
    if len(artifact_clips) != len(expected_clips):
        raise RuntimeError("encoder/current dataset manifests have different clip counts")
    for index, (artifact, expected) in enumerate(zip(artifact_clips, expected_clips)):
        for key in ("motion_id", "file", "sha256"):
            if artifact.get(key) != expected.get(key):
                raise RuntimeError(
                    f"encoder/current dataset clip {index} mismatch: {key}"
                )
        if not math.isclose(
            float(artifact.get("weight")),
            float(expected.get("weight")),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise RuntimeError(f"encoder/current dataset clip {index} mismatch: weight")
        if not math.isclose(
            float(artifact.get("length_seconds")),
            float(expected.get("length_seconds")),
            rel_tol=0.0,
            abs_tol=1e-6,
        ):
            raise RuntimeError(
                f"encoder/current dataset clip {index} mismatch: length_seconds"
            )


def load_frozen_encoder_artifact(
    artifact_path: str | Path,
    device: str | torch.device,
    *,
    expected_dataset_manifest: Mapping[str, Any] | None = None,
    expected_feature_schema: Mapping[str, Any] | None = None,
) -> tuple[LabelFreeSkillEncoder, dict[str, Any], dict[str, Any]]:
    """Strictly load a Gate-1-passing Stage-1A encoder and freeze it."""
    artifact_path = Path(artifact_path)
    if not artifact_path.is_file():
        raise FileNotFoundError(f"encoder artifact does not exist: {artifact_path}")
    payload = torch.load(artifact_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise RuntimeError("encoder artifact must contain a mapping")
    if payload.get("format_version") != ENCODER_FORMAT_VERSION:
        raise RuntimeError("unsupported encoder artifact format_version")
    if payload.get("model_type") != ENCODER_MODEL_TYPE:
        raise RuntimeError("encoder artifact has the wrong model_type")
    validation = payload.get("validation")
    if not isinstance(validation, Mapping) or validation.get("gate_passed") is not True:
        raise RuntimeError("encoder artifact did not pass Gate 1")
    schema = payload.get("encoder_schema")
    config = payload.get("model_config")
    state = payload.get("model_state_dict")
    if not isinstance(schema, Mapping) or not isinstance(config, Mapping):
        raise RuntimeError("encoder artifact is missing schema/config metadata")
    if not isinstance(state, Mapping):
        raise RuntimeError("encoder artifact is missing model_state_dict")
    required_schema = {
        "feature_dim": 44,
        "view_steps": CONTEXT_STEPS,
        "embedding_dim": 8,
    }
    for key, expected in required_schema.items():
        if int(schema.get(key, -1)) != expected:
            raise RuntimeError(f"encoder schema requires {key}={expected}")
    if expected_feature_schema is not None:
        if schema.get("feature_schema") != dict(expected_feature_schema):
            raise RuntimeError("encoder/current 44-D feature schemas do not match")
    artifact_manifest = schema.get("dataset_manifest")
    data_contract = payload.get("data_contract")
    if isinstance(data_contract, Mapping):
        contract_manifest = data_contract.get("dataset_manifest")
        if contract_manifest != artifact_manifest:
            raise RuntimeError("encoder artifact contains inconsistent dataset manifests")
    if expected_dataset_manifest is not None:
        _validate_dataset_manifest(artifact_manifest, expected_dataset_manifest)

    encoder = LabelFreeSkillEncoder(
        feature_dim=44,
        embedding_dim=8,
        hidden_dim=int(config["hidden_dim"]),
        num_layers=int(config["num_layers"]),
    )
    encoder.load_state_dict(state, strict=True)
    if any(not torch.isfinite(value).all() for value in encoder.state_dict().values()):
        raise RuntimeError("encoder artifact contains non-finite tensors")
    encoder.to(torch.device(device)).eval()
    encoder.requires_grad_(False)

    checkpoint_schema = deepcopy(dict(schema))
    checkpoint_schema["latent_dim"] = 8
    checkpoint_schema["runtime_embedding"] = "l2_normalize(y)"
    checkpoint_schema["dataset_manifest"] = checkpoint_dataset_manifest(
        {"dataset_manifest": artifact_manifest}
    )
    encoder_gate = {
        "artifact_path": str(artifact_path),
        "artifact_sha256": file_sha256(artifact_path),
        "format_version": ENCODER_FORMAT_VERSION,
        "model_type": ENCODER_MODEL_TYPE,
        "iteration": int(payload.get("iteration", -1)),
        "gate_passed": True,
        "validation": deepcopy(dict(validation)),
    }
    return encoder, checkpoint_schema, encoder_gate


def random_cross_clip_derangement_indices(
    motion_ids: torch.Tensor, generator: torch.Generator
) -> torch.Tensor:
    """Return a random permutation whose source/destination clips all differ."""
    motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device="cpu")
    unique_ids = torch.unique(motion_ids, sorted=True)
    if unique_ids.numel() < 2:
        raise ValueError("wrong-z audit requires at least two held-out clips")
    counts = torch.tensor([(motion_ids == motion_id).sum() for motion_id in unique_ids])
    if not torch.all(counts == counts[0]):
        raise ValueError("cross-clip derangement requires an equal-per-clip panel")

    shuffled = unique_ids[torch.randperm(unique_ids.numel(), generator=generator)]
    shift = int(
        torch.randint(1, unique_ids.numel(), (1,), generator=generator).item()
    )
    destination_for = {
        int(source): int(destination)
        for source, destination in zip(shuffled, torch.roll(shuffled, shifts=shift))
    }
    indices = torch.full_like(motion_ids, -1)
    for source_id in unique_ids.tolist():
        source_rows = torch.nonzero(motion_ids == source_id, as_tuple=False).flatten()
        destination_rows = torch.nonzero(
            motion_ids == destination_for[source_id], as_tuple=False
        ).flatten()
        destination_rows = destination_rows[
            torch.randperm(destination_rows.numel(), generator=generator)
        ]
        indices[source_rows] = destination_rows
    if torch.any(indices < 0) or torch.unique(indices).numel() != motion_ids.numel():
        raise RuntimeError("failed to construct a complete wrong-z derangement")
    if torch.any(motion_ids[indices] == motion_ids):
        raise RuntimeError("wrong-z derangement retained a source clip")
    return indices


def coarse_semantic_wrong_indices(
    audit_tags: Sequence[str], generator: torch.Generator
) -> torch.Tensor:
    """Choose the opposite heading-invariant run/walk audit class."""
    tags = [coarse_semantic_tag(tag) for tag in audit_tags]
    if len(set(tags)) < 2:
        raise ValueError("semantic wrong-z audit requires at least two coarse tags")
    result = torch.empty(len(tags), dtype=torch.long)
    for row, tag in enumerate(tags):
        candidates = torch.tensor(
            [index for index, candidate_tag in enumerate(tags) if candidate_tag != tag],
            dtype=torch.long,
        )
        selected = torch.randint(candidates.numel(), (1,), generator=generator)
        result[row] = candidates[int(selected)]
    return result


def coarse_semantic_tag(audit_tag: str) -> str:
    if audit_tag == "walk":
        return "walk"
    if audit_tag == "run" or audit_tag.startswith("run_"):
        return "run"
    raise ValueError(f"unsupported coarse semantic audit tag: {audit_tag}")


@torch.no_grad()
def _encode_contexts(
    encoder: LabelFreeSkillEncoder,
    contexts: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    embeddings = []
    for start in range(0, contexts.shape[0], batch_size):
        embeddings.append(encoder.runtime_z(contexts[start : start + batch_size]).cpu())
    latent = torch.cat(embeddings)
    torch.testing.assert_close(
        latent.norm(dim=-1),
        torch.ones(latent.shape[0]),
        rtol=1e-4,
        atol=1e-4,
    )
    return latent


@torch.no_grad()
def build_validation_panel(
    model: ConditionalFlowMatchingModel,
    encoder: LabelFreeSkillEncoder,
    sampler: ConditionalMotionWindowSampler,
    env_config: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    generator = sampler.make_generator(seed)
    paired = sampler.sample_equal_clip_panel(
        int(config.get("validation_samples_per_clip", 64)),
        "heldout",
        generator,
    )
    raw_window = paired["motion_window"]
    matched_latent = _encode_contexts(
        encoder,
        paired["context_features"],
        int(config.get("validation_batch_size", config["batch_size"])),
    )
    expert = model.normalize(raw_window).detach().cpu()
    num_samples, window_steps, frame_dim = expert.shape

    permutations = torch.stack(
        [torch.randperm(window_steps, generator=generator) for _ in range(num_samples)]
    )
    row_ids = torch.arange(num_samples).unsqueeze(1)
    temporal_shuffle = expert[row_ids, permutations]
    feature_noise = expert + float(
        config.get("validation_feature_noise_std", 0.35)
    ) * torch.randn(expert.shape, generator=generator, dtype=expert.dtype)

    key_bodies = list(env_config.get("key_bodies", []))
    foot_body_indices = [
        index for index, name in enumerate(key_bodies) if "foot" in name.lower()
    ]
    if not foot_body_indices:
        raise ValueError("conditional-flow validation requires foot entries in key_bodies")
    num_joints = int(sampler.dataset._motion_lib.get_num_joints())
    root_position_dim = 3 if env_config.get("root_height_obs", True) else 2
    key_position_start = root_position_dim + 6 + 6 * (num_joints - 1)
    if key_position_start + 3 * (max(foot_body_indices) + 1) > frame_dim:
        raise ValueError("computed foot feature indices exceed the SMP frame size")
    angles = 2.0 * math.pi * torch.rand(num_samples, generator=generator)
    directions = torch.stack((torch.cos(angles), torch.sin(angles)), dim=-1).to(
        raw_window.device
    )
    progress = torch.linspace(
        0.0,
        1.0,
        window_steps,
        device=raw_window.device,
        dtype=raw_window.dtype,
    )
    offsets = (
        float(config.get("validation_foot_slide_distance", 0.25))
        * progress[None, :, None]
        * directions[:, None, :]
    )
    foot_slide_raw = raw_window.clone()
    for body_index in foot_body_indices:
        xy_start = key_position_start + 3 * body_index
        foot_slide_raw[:, :, xy_start : xy_start + 2] += offsets
    foot_slide = model.normalize(foot_slide_raw).detach().cpu()
    severe_random = float(config.get("validation_severe_random_std", 3.0)) * torch.randn(
        expert.shape, generator=generator, dtype=expert.dtype
    )

    random_wrong_indices = random_cross_clip_derangement_indices(
        paired["motion_ids"], generator
    )
    semantic_wrong_indices = coarse_semantic_wrong_indices(
        paired["gate_audit_tags"], generator
    )
    if torch.any(paired["motion_ids"][random_wrong_indices] == paired["motion_ids"]):
        raise RuntimeError("random wrong-z panel contains a matched clip")
    semantic_wrong_tags = [
        coarse_semantic_tag(paired["gate_audit_tags"][i])
        for i in semantic_wrong_indices
    ]
    source_semantic_tags = [
        coarse_semantic_tag(tag) for tag in paired["gate_audit_tags"]
    ]
    if any(source == wrong for source, wrong in zip(source_semantic_tags, semantic_wrong_tags)):
        raise RuntimeError("semantic wrong-z panel contains a matched coarse label")

    variants = {
        "matched": {"samples": expert, "latent": matched_latent},
        "wrong_random": {
            "samples": expert,
            "latent": matched_latent[random_wrong_indices],
        },
        "wrong_semantic": {
            "samples": expert,
            "latent": matched_latent[semantic_wrong_indices],
        },
        "null": {"samples": expert, "latent": None},
        "temporal_shuffle": {
            "samples": temporal_shuffle,
            "latent": matched_latent,
        },
        "feature_noise": {"samples": feature_noise, "latent": matched_latent},
        "foot_slide": {"samples": foot_slide, "latent": matched_latent},
        "severe_random": {"samples": severe_random, "latent": matched_latent},
    }
    return {
        "variants": variants,
        "motion_ids": paired["motion_ids"],
        "audit_tags": paired["audit_tags"],
        "gate_audit_tags": paired["gate_audit_tags"],
        "coarse_semantic_tags": source_semantic_tags,
        "starts": paired["starts"],
        "context_times": paired["context_times"],
        "window_times": paired["window_times"],
        "random_wrong_indices": random_wrong_indices,
        "semantic_wrong_indices": semantic_wrong_indices,
        "audit_labels_used_for_training": False,
        "audit_labels_used_for_validation_only": True,
    }


@torch.no_grad()
def batched_mismatch(
    model: ConditionalFlowMatchingModel,
    samples: torch.Tensor,
    latent: torch.Tensor | None,
    times: torch.Tensor,
    base_noise: torch.Tensor,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    parts = []
    for start in range(0, samples.shape[0], batch_size):
        stop = start + batch_size
        batch = samples[start:stop].to(device).reshape(-1, model.input_dim)
        batch_latent = None if latent is None else latent[start:stop].to(device)
        values = model.aggregate_mismatch(
            batch,
            batch_latent,
            times,
            base_noise,
            use_ema=True,
        )
        if values.shape != (batch.shape[0],) or not torch.isfinite(values).all():
            raise RuntimeError("conditional aggregate_mismatch returned invalid values")
        parts.append(values.cpu())
    return torch.cat(parts)


def _comparison_thresholds(config: Mapping[str, Any], name: str) -> tuple[float, float]:
    gate_config = config.get("validation_gates", {})
    override = gate_config.get(name, {}) if isinstance(gate_config, Mapping) else {}
    min_win_rate = float(
        override.get(
            "min_win_rate", config.get("validation_min_paired_win_rate", 0.55)
        )
    )
    min_median_ratio = float(
        override.get(
            "min_median_ratio", config.get("validation_min_median_ratio", 1.02)
        )
    )
    if not 0.0 <= min_win_rate <= 1.0 or min_median_ratio <= 0.0:
        raise ValueError(f"invalid validation threshold for {name}")
    return min_win_rate, min_median_ratio


def normalized_comparison_margin(comparison: Mapping[str, Any]) -> float:
    return min(
        float(comparison["lower_paired_win_rate"])
        / max(float(comparison["min_win_rate"]), 1e-12),
        float(comparison["upper_to_lower_median_ratio"])
        / max(float(comparison["min_median_ratio"]), 1e-12),
    )


def conditional_selection_score(comparisons: Mapping[str, Mapping[str, Any]]) -> float:
    missing = set(SELECTION_COMPARISONS).difference(comparisons)
    if missing:
        raise KeyError(f"conditional selection comparisons are missing {sorted(missing)}")
    return min(
        normalized_comparison_margin(comparisons[name])
        for name in SELECTION_COMPARISONS
    )


@torch.no_grad()
def validate_prior(
    model: ConditionalFlowMatchingModel,
    panel: Mapping[str, Any],
    times: torch.Tensor,
    base_noise: torch.Tensor,
    device: torch.device,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model.eval()
    eval_batch_size = int(config.get("validation_batch_size", config["batch_size"]))
    raw = {
        name: batched_mismatch(
            model,
            variant["samples"],
            variant["latent"],
            times,
            base_noise,
            device,
            eval_batch_size,
        )
        for name, variant in panel["variants"].items()
    }
    floor = float(config.get("calibration_floor", 1e-6))
    conditional_expert_scale = max(float(raw["matched"].median()), floor)
    calibrated = {
        name: values / conditional_expert_scale for name, values in raw.items()
    }

    comparisons = {}
    for name, (lower_name, upper_name) in REQUIRED_COMPARISONS.items():
        lower = raw[lower_name]
        upper = raw[upper_name]
        min_win_rate, min_median_ratio = _comparison_thresholds(config, name)
        lower_median = float(lower.median())
        upper_median = float(upper.median())
        win_rate = float((lower < upper).float().mean())
        ratio = upper_median / max(lower_median, floor)
        ordered = lower_median < upper_median
        comparisons[name] = {
            "lower": lower_name,
            "upper": upper_name,
            "lower_median": lower_median,
            "upper_median": upper_median,
            "lower_paired_win_rate": win_rate,
            "upper_to_lower_median_ratio": ratio,
            "min_win_rate": min_win_rate,
            "min_median_ratio": min_median_ratio,
            "median_order_passed": ordered,
            "passed": ordered
            and win_rate >= min_win_rate
            and ratio >= min_median_ratio,
        }
    gate_passed = all(item["passed"] for item in comparisons.values())
    gate_margin_score = min(
        normalized_comparison_margin(item) for item in comparisons.values()
    )
    # Perturbation panels decide whether a checkpoint is publishable, but
    # several of their win rates quickly saturate at one. Select among passing
    # checkpoints using only the three tests that measure whether z is used.
    selection_score = conditional_selection_score(comparisons)
    result = {
        "conditional_expert_scale": conditional_expert_scale,
        "raw": {name: quantile_summary(values) for name, values in raw.items()},
        "calibrated": {
            name: quantile_summary(values) for name, values in calibrated.items()
        },
        "comparisons": comparisons,
        "gate_passed": gate_passed,
        "gate_margin_score": gate_margin_score,
        "selection_score": selection_score,
        "selection_comparisons": list(SELECTION_COMPARISONS),
        "paired_noise_and_times": True,
        "audit_labels_used_for_training": False,
        "coarse_semantic_labels_used_for_validation_only": True,
    }
    model.train()
    return result


def checkpoint_dataset_manifest(data_audit: Mapping[str, Any]) -> dict[str, Any]:
    source = data_audit["dataset_manifest"]
    clips = []
    for clip in source["clips"]:
        clips.append(
            {
                "motion_id": int(clip["motion_id"]),
                "file": str(clip["file"]),
                "weight": float(clip["weight"]),
                "length_seconds": float(clip["length_seconds"]),
                "sha256": str(clip["sha256"]),
            }
        )
    return {
        "dataset_yaml_sha256": str(source["dataset_yaml_sha256"]),
        "canonical_manifest_sha256": str(source["canonical_manifest_sha256"]),
        "clips": clips,
    }


def build_checkpoint(
    model: ConditionalFlowMatchingModel,
    encoder: LabelFreeSkillEncoder,
    *,
    encoder_schema: Mapping[str, Any],
    encoder_gate: Mapping[str, Any],
    data_audit: Mapping[str, Any],
    config: Mapping[str, Any],
    times: torch.Tensor,
    base_noise: torch.Tensor,
    iteration: int,
    validation: Mapping[str, Any],
) -> dict[str, Any]:
    if encoder_gate.get("gate_passed") is not True:
        raise RuntimeError("refusing to build a format-2 checkpoint without Encoder Gate 1")
    if validation.get("gate_passed") is not True:
        raise RuntimeError("refusing to build a format-2 checkpoint without offline Gate 2")
    calibration = {
        "times": times.detach().cpu(),
        "base_noise": base_noise.detach().cpu(),
        "conditional_expert_scale": float(
            validation["conditional_expert_scale"]
        ),
    }
    payload = conditional_checkpoint_payload(
        model,
        encoder,
        encoder_schema=encoder_schema,
        calibration=calibration,
        iteration=iteration,
    )
    payload["created_utc"] = datetime.now(timezone.utc).isoformat()
    metadata = payload["metadata"]
    dataset_manifest = checkpoint_dataset_manifest(data_audit)
    metadata["encoder_schema"]["dataset_manifest"] = deepcopy(dataset_manifest)
    metadata.update(
        {
            "latent_schema": {
                "dimension": 8,
                "encoder_output": "y",
                "runtime_condition": "z=l2_normalize(y)",
                "non_null_norm": "unit_l2",
                "semantic_labels_used_for_training": False,
            },
            "aggregation": "t_squared_weighted_mean",
            "K": int(config["reward_noise_samples"]),
            "reward_noise_samples": int(config["reward_noise_samples"]),
            "null_training_probability": float(config["p_null"]),
            "dataset_manifest": deepcopy(dataset_manifest),
            "paired_sampling": deepcopy(data_audit["paired_sampling"]),
        }
    )
    payload["encoder_gate"] = deepcopy(dict(encoder_gate))
    payload["offline_validation"] = deepcopy(dict(validation))
    if payload["format_version"] != CONDITIONAL_FLOW_FORMAT_VERSION:
        raise RuntimeError("conditional checkpoint helper changed format unexpectedly")
    if payload["model_type"] != CONDITIONAL_FLOW_MODEL_TYPE:
        raise RuntimeError("conditional checkpoint helper changed model type unexpectedly")
    if metadata["dataset_manifest"] != metadata["encoder_schema"]["dataset_manifest"]:
        raise RuntimeError("flow and encoder dataset manifests are not identical")
    return payload


def train(
    cfg_path: str | Path,
    out_dir: str | Path,
    device: str | torch.device = "cuda",
    max_iters: int | None = None,
) -> Path:
    cfg_path = Path(cfg_path)
    with cfg_path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if max_iters is not None:
        config["num_iterations"] = int(max_iters)
    if not out_dir:
        raise ValueError("out_dir must be non-empty")
    device = torch.device(device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    seed = int(config.get("seed", 0))
    fix_seed(seed)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg_path, out_dir / "source_config.yaml")
    for filename in (
        "metrics.jsonl",
        "validation.jsonl",
        "offline_validation.json",
        "offline_validation_best.json",
    ):
        (out_dir / filename).unlink(missing_ok=True)
    for filename in ("model.pt", "conditional_flow_best.pt", "conditional_flow_last.pt"):
        (out_dir / filename).unlink(missing_ok=True)

    sampler = ConditionalMotionWindowSampler(config, device)
    data_audit = sampler.data_audit()
    if data_audit["num_clips"] != int(config.get("expected_num_clips", 54)):
        raise RuntimeError("R2 data audit did not resolve the expected 54 clips")
    if data_audit["eligible_clips"] != data_audit["num_clips"]:
        raise RuntimeError("every R2 clip must be eligible for paired sampling")
    with (out_dir / "data_audit.json").open("w", encoding="utf-8") as stream:
        json.dump(data_audit, stream, indent=2, sort_keys=True)

    encoder, encoder_schema, encoder_gate = load_frozen_encoder_artifact(
        config["encoder_artifact"],
        device,
        expected_dataset_manifest=data_audit["dataset_manifest"],
        expected_feature_schema=sampler.feature_schema,
    )
    with (out_dir / "encoder_gate.json").open("w", encoding="utf-8") as stream:
        json.dump(encoder_gate, stream, indent=2, sort_keys=True)

    obs_shape = sampler.dataset.get_obs_space().shape
    input_dim, frame_dim = infer_smp_dimensions(obs_shape)
    config.update(
        {
            "input_dim": input_dim,
            "input_channel": frame_dim,
            "num_disc_obs_steps": MOTION_WINDOW_STEPS,
            "latent_dim": 8,
            "enforce_unit_latent": True,
        }
    )
    if not 0.0 <= float(config.get("p_null", 0.1)) <= 1.0:
        raise ValueError("p_null must be in [0,1]")
    with (out_dir / "conditional_flow_config.yaml").open(
        "w", encoding="utf-8"
    ) as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    model = ConditionalFlowMatchingModel(config, device).to(device)
    stats_generator = sampler.make_generator(int(config.get("stats_seed", seed + 1_000)))
    num_stats = int(config.get("num_samples_stat", 20_000))
    stats_batch = int(config.get("stats_batch_size", config["batch_size"]))
    for start in range(0, num_stats, stats_batch):
        pair = sampler.sample_pairs(
            min(stats_batch, num_stats - start), "train", stats_generator
        )
        model.update_normalizer(pair["motion_window"])

    with Path(config["env_config"]).open("r", encoding="utf-8") as stream:
        env_config = yaml.safe_load(stream)
    panel = build_validation_panel(
        model,
        encoder,
        sampler,
        env_config,
        config,
        int(config.get("validation_seed", seed + 10_000)),
    )
    times = torch.tensor(
        config.get("reward_times", [0.25, 0.5, 0.75]),
        device=device,
        dtype=torch.float32,
    )
    if times.shape != (3,) or not torch.all((times > 0.0) & (times < 1.0)):
        raise ValueError("reward_times must contain three values strictly inside (0,1)")
    reward_noise_samples = int(config.get("reward_noise_samples", 1))
    if reward_noise_samples not in (1, 2):
        raise ValueError("conditional Flow-SMP supports K=1 or K=2")
    config["reward_noise_samples"] = reward_noise_samples
    noise_generator = torch.Generator(device="cpu")
    noise_generator.manual_seed(int(config.get("reward_noise_seed", seed + 20_000)))
    first_noise = torch.randn((1, MOTION_WINDOW_STEPS, frame_dim), generator=noise_generator)
    base_noise = (
        torch.cat((first_noise, -first_noise), dim=0)
        if reward_noise_samples == 2
        else first_noise
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 0.0)),
    )
    batch_size = int(config["batch_size"])
    num_iterations = int(config["num_iterations"])
    log_iter = int(config.get("log_iter", 100))
    output_iter = int(config.get("output_iter", 2_000))
    grad_clip = float(config.get("grad_clip_norm", 1.0))
    p_null = float(config.get("p_null", 0.1))
    train_generator = sampler.make_generator(seed)
    null_generator = torch.Generator(device="cpu")
    null_generator.manual_seed(int(config.get("null_seed", seed + 30_000)))
    best_score = -float("inf")
    published = False
    running_loss = 0.0
    running_grad = 0.0
    running_null = 0.0
    running_count = 0

    print(
        f"Conditional Flow-SMP: clips={data_audit['num_clips']}, "
        f"train={len(data_audit['train_clips'])}, heldout={len(data_audit['heldout_clips'])}, "
        f"A={CONTEXT_STEPS}x44, W={MOTION_WINDOW_STEPS}x{frame_dim}, z=8, "
        f"p_null={p_null:.3f}",
        flush=True,
    )
    model.train()
    for iteration in range(1, num_iterations + 1):
        pair = sampler.sample_pairs(batch_size, "train", train_generator)
        with torch.no_grad():
            latent = encoder.runtime_z(pair["context_features"])
        if not torch.allclose(
            latent.norm(dim=-1),
            torch.ones(batch_size, device=device),
            rtol=1e-4,
            atol=1e-4,
        ):
            raise RuntimeError("frozen encoder returned a non-unit runtime latent")
        samples = model.normalize(pair["motion_window"]).reshape(batch_size, input_dim)
        null_mask = sample_null_mask(batch_size, p_null, null_generator, device)

        optimizer.zero_grad(set_to_none=True)
        loss = model(samples, latent, null_mask=null_mask)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"non-finite conditional flow loss at iteration {iteration}"
            )
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        if bool(config.get("model_ema", True)):
            model.update_ema()

        running_loss += float(loss)
        running_grad += float(grad_norm)
        running_null += float(null_mask.float().mean())
        running_count += 1
        if iteration % log_iter == 0 or iteration == num_iterations:
            record = {
                "iteration": iteration,
                "loss": running_loss / running_count,
                "grad_norm": running_grad / running_count,
                "null_fraction": running_null / running_count,
                "configured_p_null": p_null,
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_jsonl(out_dir / "metrics.jsonl", record)
            print(json.dumps(record, sort_keys=True), flush=True)
            running_loss = running_grad = running_null = 0.0
            running_count = 0

        if iteration % output_iter == 0 or iteration == num_iterations:
            validation = validate_prior(model, panel, times, base_noise, device, config)
            validation["iteration"] = iteration
            append_jsonl(out_dir / "validation.jsonl", validation)
            with (out_dir / "offline_validation.json").open(
                "w", encoding="utf-8"
            ) as stream:
                json.dump(validation, stream, indent=2, sort_keys=True)
            if validation["gate_passed"]:
                payload = build_checkpoint(
                    model,
                    encoder,
                    encoder_schema=encoder_schema,
                    encoder_gate=encoder_gate,
                    data_audit=data_audit,
                    config=config,
                    times=times,
                    base_noise=base_noise,
                    iteration=iteration,
                    validation=validation,
                )
                torch.save(payload, out_dir / "conditional_flow_last.pt")
                if validation["selection_score"] > best_score:
                    best_score = float(validation["selection_score"])
                    torch.save(payload, out_dir / "conditional_flow_best.pt")
                    torch.save(payload, out_dir / "model.pt")
                    with (out_dir / "offline_validation_best.json").open(
                        "w", encoding="utf-8"
                    ) as stream:
                        json.dump(validation, stream, indent=2, sort_keys=True)
                    published = True
            print(
                json.dumps(
                    {
                        "iteration": iteration,
                        "gate_passed": validation["gate_passed"],
                        "selection_score": validation["selection_score"],
                        "conditional_expert_scale": validation[
                            "conditional_expert_scale"
                        ],
                        "comparisons": validation["comparisons"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    if not published:
        raise RuntimeError(
            "conditional offline gate never passed; no PPO-loadable model.pt was published"
        )
    return out_dir / "model.pt"


def main() -> None:
    parser = ArgumentParser(description="Train the R2 Stage 1B conditional Flow-SMP prior")
    parser.add_argument(
        "--cfg_path",
        default="tools/flow_matching/config/conditional_flow_locomotion.yaml",
    )
    parser.add_argument(
        "--out_dir", default="output/flow_matching/conditional_locomotion"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_iters", type=int)
    args = parser.parse_args()
    train(args.cfg_path, args.out_dir, args.device, args.max_iters)


if __name__ == "__main__":
    main()
