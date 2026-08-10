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

from learning.skill_encoder.skill_encoder_model import (
    LabelFreeSkillEncoder,
    embedding_diagnostics,
    vicreg_loss,
)
from tools.skill_encoder.motion_view_data import PairedMotionViewSampler


MODEL_TYPE = "label_free_skill_encoder"
FORMAT_VERSION = 1


def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def append_jsonl(path, record):
    with open(path, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


@torch.no_grad()
def evaluate(model, heldout_pair, sampler, config):
    model.eval()
    eval_batch_size = int(config.get("validation_batch_size", 512))
    y_a_parts = []
    y_b_parts = []
    for start in range(0, heldout_pair["view_a"].shape[0], eval_batch_size):
        stop = start + eval_batch_size
        y_a_parts.append(model(heldout_pair["view_a"][start:stop]).cpu())
        y_b_parts.append(model(heldout_pair["view_b"][start:stop]).cpu())
    y_a = torch.cat(y_a_parts)
    y_b = torch.cat(y_b_parts)
    z_a = torch.nn.functional.normalize(y_a, dim=-1)
    z_b = torch.nn.functional.normalize(y_b, dim=-1)

    all_y = torch.cat((y_a, y_b), dim=0)
    diagnostics = embedding_diagnostics(all_y)
    pair_cosine = (z_a * z_b).sum(dim=-1)
    similarity = z_a @ z_b.T
    nearest_indices = similarity.argmax(dim=1)
    paired_indices = torch.arange(similarity.shape[0])
    paired_rank = 1 + (similarity > similarity.diagonal().unsqueeze(1)).sum(dim=1)
    motion_ids = heldout_pair["motion_ids"].cpu()
    nearest_same_clip = motion_ids[nearest_indices] == motion_ids

    audit_tags = [sampler.audit_tag(int(motion_id)) for motion_id in motion_ids]
    tag_names = sorted(set(audit_tags))
    tag_to_id = {tag: index for index, tag in enumerate(tag_names)}
    tag_ids = torch.tensor([tag_to_id[tag] for tag in audit_tags])
    mirror_families = heldout_pair["mirror_families"]
    family_to_id = {family: index for index, family in enumerate(sorted(set(mirror_families)))}
    family_ids = torch.tensor([family_to_id[family] for family in mirror_families])
    same_family_matrix = family_ids.unsqueeze(1) == family_ids.unsqueeze(0)
    cross_clip_similarity = similarity.masked_fill(same_family_matrix, -torch.inf)
    if not torch.isfinite(cross_clip_similarity.max(dim=1).values).all():
        raise RuntimeError("held-out panel lacks a cross-clip nearest-neighbor candidate")
    cross_clip_indices = cross_clip_similarity.argmax(dim=1)
    cross_clip_same_tag = tag_ids[cross_clip_indices] == tag_ids
    valid_candidates = ~same_family_matrix
    same_tag_candidates = tag_ids.unsqueeze(1) == tag_ids.unsqueeze(0)
    chance_per_query = (
        (same_tag_candidates & valid_candidates).sum(dim=1).float()
        / valid_candidates.sum(dim=1).float()
    )
    cross_clip_rate = float(cross_clip_same_tag.float().mean())
    chance_rate = float(chance_per_query.mean())
    per_tag_recall = {
        tag: float(cross_clip_same_tag[tag_ids == tag_id].float().mean())
        for tag, tag_id in tag_to_id.items()
    }
    per_tag_chance = {
        tag: float(chance_per_query[tag_ids == tag_id].mean())
        for tag, tag_id in tag_to_id.items()
    }
    cross_clip_macro = sum(per_tag_recall.values()) / len(per_tag_recall)
    chance_macro = sum(per_tag_chance.values()) / len(per_tag_chance)

    gate_tags = [sampler.gate_audit_tag(int(motion_id)) for motion_id in motion_ids]
    gate_tag_names = sorted(set(gate_tags))
    gate_tag_to_id = {tag: index for index, tag in enumerate(gate_tag_names)}
    gate_tag_ids = torch.tensor([gate_tag_to_id[tag] for tag in gate_tags])
    gate_same_tag = gate_tag_ids[cross_clip_indices] == gate_tag_ids
    gate_same_tag_candidates = gate_tag_ids.unsqueeze(1) == gate_tag_ids.unsqueeze(0)
    gate_chance_per_query = (
        (gate_same_tag_candidates & valid_candidates).sum(dim=1).float()
        / valid_candidates.sum(dim=1).float()
    )
    gate_per_tag_recall = {
        tag: float(gate_same_tag[gate_tag_ids == tag_id].float().mean())
        for tag, tag_id in gate_tag_to_id.items()
    }
    gate_per_tag_chance = {
        tag: float(gate_chance_per_query[gate_tag_ids == tag_id].mean())
        for tag, tag_id in gate_tag_to_id.items()
    }
    gate_macro = sum(gate_per_tag_recall.values()) / len(gate_per_tag_recall)
    gate_chance_macro = sum(gate_per_tag_chance.values()) / len(gate_per_tag_chance)

    probe_similarity = z_a @ z_a.T
    probe_similarity.fill_diagonal_(-torch.inf)
    probe_indices = probe_similarity.argmax(dim=1)
    clip_probe_accuracy = float((motion_ids[probe_indices] == motion_ids).float().mean())
    clip_probe_chance = float(
        (
            (motion_ids.unsqueeze(1) == motion_ids.unsqueeze(0)).sum(dim=1) - 1
        ).float().div(motion_ids.numel() - 1).mean()
    )
    phase_center = (
        heldout_pair["view_a_times"].mean(dim=1).cpu()
        / heldout_pair["motion_lengths"].cpu()
    ).remainder(1.0)
    phase_sin_cos = torch.stack(
        (
            torch.sin(2.0 * torch.pi * phase_center),
            torch.cos(2.0 * torch.pi * phase_center),
        ),
        dim=-1,
    )
    phase_gap = torch.abs(phase_center - phase_center[probe_indices])
    phase_gap = torch.minimum(phase_gap, 1.0 - phase_gap)
    shuffle_generator = torch.Generator(device="cpu")
    shuffle_generator.manual_seed(int(config.get("audit_shuffle_seed", 3000)))
    shuffled_phase = phase_center[
        torch.randperm(phase_center.numel(), generator=shuffle_generator)
    ]
    shuffled_gap = torch.abs(shuffled_phase - shuffled_phase[probe_indices])
    shuffled_gap = torch.minimum(shuffled_gap, 1.0 - shuffled_gap)
    phase_knn_mae = float(phase_gap.mean())
    phase_shuffle_mae = float(shuffled_gap.mean())

    same_clip_phase_gap = torch.abs(
        heldout_pair["view_a_times"][:, 0].cpu()
        - heldout_pair["view_b_times"][nearest_indices, 0].cpu()
    ) / heldout_pair["motion_lengths"].cpu()
    same_clip_phase_gap = same_clip_phase_gap[nearest_same_clip]

    min_std_gate = float(config.get("gate_min_dim_std", 0.05))
    min_rank_gate = float(config.get("gate_min_effective_rank", 4.0))
    tag_margin_gate = float(config.get("gate_cross_clip_tag_margin", 0.1))
    tag_rate_gate = float(config.get("gate_cross_clip_tag_min_rate", 0.0))
    pair_cosine_gate = float(config.get("gate_min_pair_cosine", 0.8))
    max_clip_lift = float(config.get("gate_max_clip_probe_lift", 0.75))
    max_phase_gain = float(config.get("gate_max_phase_knn_gain", 0.15))
    noncollapse_passed = (
        diagnostics["min_dim_std"] >= min_std_gate
        and diagnostics["effective_rank"] >= min_rank_gate
        and torch.isfinite(all_y).all().item()
    )
    cross_clip_audit_passed = (
        gate_macro >= gate_chance_macro + tag_margin_gate
        and gate_macro >= tag_rate_gate
    )
    invariance_passed = float(pair_cosine.mean()) >= pair_cosine_gate
    leakage_audit_passed = (
        clip_probe_accuracy - clip_probe_chance <= max_clip_lift
        and phase_shuffle_mae - phase_knn_mae <= max_phase_gain
        and not set(sampler.train_families).intersection(sampler.heldout_families)
    )
    gate_passed = (
        noncollapse_passed
        and invariance_passed
        and cross_clip_audit_passed
        and leakage_audit_passed
    )
    metrics = {
        **diagnostics,
        "pair_cosine_mean": float(pair_cosine.mean()),
        "pair_cosine_median": float(pair_cosine.median()),
        "paired_retrieval_top1": float((nearest_indices == paired_indices).float().mean()),
        "paired_retrieval_mean_rank": float(paired_rank.float().mean()),
        "nearest_neighbor_same_clip_rate": float(nearest_same_clip.float().mean()),
        "same_clip_nn_phase_gap_mean": (
            float(same_clip_phase_gap.mean()) if same_clip_phase_gap.numel() else None
        ),
        "same_clip_nn_phase_gap_median": (
            float(same_clip_phase_gap.median()) if same_clip_phase_gap.numel() else None
        ),
        "cross_clip_same_tag_top1": cross_clip_rate,
        "cross_clip_tag_chance": chance_rate,
        "cross_clip_tag_lift": cross_clip_rate - chance_rate,
        "cross_clip_per_tag_recall": per_tag_recall,
        "cross_clip_per_tag_chance": per_tag_chance,
        "cross_clip_macro_same_tag_top1": cross_clip_macro,
        "cross_clip_macro_chance": chance_macro,
        "hard_semantic_tags": "heading-invariant run vs walk",
        "hard_semantic_per_tag_recall": gate_per_tag_recall,
        "hard_semantic_per_tag_chance": gate_per_tag_chance,
        "hard_semantic_macro_top1": gate_macro,
        "hard_semantic_macro_chance": gate_chance_macro,
        "hard_semantic_macro_lift": gate_macro - gate_chance_macro,
        "clip_id_knn_top1": clip_probe_accuracy,
        "clip_id_knn_chance": clip_probe_chance,
        "clip_id_knn_lift": clip_probe_accuracy - clip_probe_chance,
        "phase_center_knn_circular_mae": phase_knn_mae,
        "phase_center_shuffle_circular_mae": phase_shuffle_mae,
        "phase_knn_gain_over_shuffle": phase_shuffle_mae - phase_knn_mae,
        "phase_probe": "leave-one-out kNN on (sin(2pi phase), cos(2pi phase)); circular MAE reported",
        "gate_min_dim_std": min_std_gate,
        "gate_min_effective_rank": min_rank_gate,
        "gate_cross_clip_tag_margin": tag_margin_gate,
        "gate_cross_clip_tag_min_rate": tag_rate_gate,
        "gate_min_pair_cosine": pair_cosine_gate,
        "gate_max_clip_probe_lift": max_clip_lift,
        "gate_max_phase_knn_gain": max_phase_gain,
        "noncollapse_passed": noncollapse_passed,
        "invariance_passed": invariance_passed,
        "cross_clip_audit_passed": cross_clip_audit_passed,
        "leakage_audit_passed": leakage_audit_passed,
        "gate_passed": gate_passed,
        "audit_labels_used_for_training": False,
        "audit_labels_used_for_split": True,
        "audit_tag_source": (
            "filename-only partition/audit tags; run/walk is hard, "
            "forward/left/right is diagnostic only"
        ),
    }
    selection_score = (
        min(diagnostics["min_dim_std"] / min_std_gate, 2.0)
        + min(diagnostics["effective_rank"] / min_rank_gate, 2.0)
        + float(pair_cosine.mean())
        + (gate_macro - gate_chance_macro)
    )
    metrics["selection_score"] = selection_score
    embeddings = {
        "y_a": y_a,
        "y_b": y_b,
        "z_a": z_a,
        "z_b": z_b,
        "motion_ids": motion_ids,
        "motion_lengths": heldout_pair["motion_lengths"].cpu(),
        "mirror_families": mirror_families,
        "audit_tags": audit_tags,
        "hard_semantic_tags": gate_tags,
        "context_starts": heldout_pair["context_starts"].cpu(),
        "view_a_times": heldout_pair["view_a_times"].cpu(),
        "view_b_times": heldout_pair["view_b_times"].cpu(),
        "phase_sin_cos": phase_sin_cos,
        "feature_schema": sampler.feature_schema,
        "semantic_labels_used_for_training": False,
        "audit_labels_used_for_training": False,
        "audit_labels_used_for_split": True,
    }
    nn_audit = {
        **metrics,
        "nearest_indices": nearest_indices.tolist(),
        "nearest_cosine": similarity.max(dim=1).values.tolist(),
        "paired_rank": paired_rank.tolist(),
        "query_motion_ids": motion_ids.tolist(),
        "nearest_motion_ids": motion_ids[nearest_indices].tolist(),
        "cross_clip_nearest_indices": cross_clip_indices.tolist(),
        "cross_clip_nearest_motion_ids": motion_ids[cross_clip_indices].tolist(),
        "query_audit_tags": audit_tags,
        "cross_clip_nearest_audit_tags": [
            audit_tags[index] for index in cross_clip_indices.tolist()
        ],
        "cross_clip_nearest_hard_semantic_tags": [
            gate_tags[index] for index in cross_clip_indices.tolist()
        ],
        "clip_probe_nearest_indices": probe_indices.tolist(),
        "phase_center": phase_center.tolist(),
    }
    model.train()
    return metrics, embeddings, nn_audit


def build_checkpoint(model, config, sampler, data_audit, iteration, validation):
    encoder_schema = {
        "feature_dim": sampler.feature_dim,
        "view_steps": sampler.view_steps,
        "embedding_dim": model.embedding_dim,
        "hidden_dim": int(config["hidden_dim"]),
        "num_layers": int(config["num_layers"]),
        "feature_schema": sampler.feature_schema,
        "dataset_manifest": data_audit["dataset_manifest"],
    }
    return {
        "format_version": FORMAT_VERSION,
        "model_type": MODEL_TYPE,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "iteration": int(iteration),
        "model_state_dict": model.state_dict(),
        "model_config": dict(config),
        "encoder_schema": encoder_schema,
        "metadata": {
            "view_steps": sampler.view_steps,
            "feature_dim": sampler.feature_dim,
            "embedding_dim": model.embedding_dim,
            "runtime_embedding": "l2_normalize(y)",
            "training_objective": "VICReg(y_a,y_b)",
            "views": "two contiguous non-overlapping views from one local context",
            "feature_schema": sampler.feature_schema,
            "semantic_labels_used_for_training": False,
            "audit_labels_used_for_training": False,
            "audit_labels_used_for_split": True,
            "split_policy": data_audit["split_policy"],
            "split_hard_semantic_counts": data_audit[
                "split_hard_semantic_counts"
            ],
        },
        "data_contract": {
            "eligible_motion_ids": sorted(
                entry["motion_id"]
                for entry in data_audit["train_clips"] + data_audit["heldout_clips"]
            ),
            "train_clips": data_audit["train_clips"],
            "heldout_clips": data_audit["heldout_clips"],
            "ineligible_clips": data_audit["ineligible_clips"],
            "motion_source": data_audit["motion_source"],
            "dataset_manifest": data_audit["dataset_manifest"],
            "audit_labels_used_for_training": False,
            "audit_labels_used_for_split": True,
            "split_policy": data_audit["split_policy"],
            "split_hard_semantic_counts": data_audit[
                "split_hard_semantic_counts"
            ],
            "split_seed": data_audit["split_seed"],
            "requested_heldout_clip_fraction_per_stratum": data_audit[
                "requested_heldout_clip_fraction_per_stratum"
            ],
            "min_heldout_families_per_tag": data_audit[
                "min_heldout_families_per_tag"
            ],
        },
        "validation": validation,
    }


def train(cfg_path, out_dir, device="cuda", max_iters=None):
    cfg_path = Path(cfg_path)
    with open(cfg_path, "r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if max_iters is not None:
        config["num_iterations"] = int(max_iters)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for filename in (
        "metrics.jsonl",
        "validation.jsonl",
        "validation_latest.json",
        "validation_best.json",
        "model.pt",
        "skill_encoder_best.pt",
        "skill_encoder_last.pt",
        "heldout_embeddings.pt",
        "nn_audit.json",
    ):
        (out_dir / filename).unlink(missing_ok=True)
    shutil.copy2(cfg_path, out_dir / "source_config.yaml")
    with open(out_dir / "skill_encoder_config.yaml", "w", encoding="utf-8") as stream:
        yaml.safe_dump(config, stream, sort_keys=False)

    seed = int(config.get("seed", 0))
    fix_seed(seed)
    device = torch.device(device)
    sampler = PairedMotionViewSampler(config, device)
    audit = sampler.data_audit()
    if audit["eligible_clips"] < 2 or len(audit["heldout_clips"]) < 1:
        raise RuntimeError("multi-clip train/heldout sampling is unavailable")
    with open(out_dir / "data_audit.json", "w", encoding="utf-8") as stream:
        json.dump(audit, stream, indent=2, sort_keys=True)
    with open(out_dir / "feature_schema.json", "w", encoding="utf-8") as stream:
        json.dump(sampler.feature_schema, stream, indent=2, sort_keys=True)

    model = LabelFreeSkillEncoder(
        feature_dim=sampler.feature_dim,
        embedding_dim=int(config.get("embedding_dim", 8)),
        hidden_dim=int(config.get("hidden_dim", 128)),
        num_layers=int(config.get("num_layers", 3)),
    ).to(device)
    stats_mean, stats_std = sampler.estimate_feature_stats(
        num_pairs=int(config.get("num_samples_stat", 8_192)),
        batch_size=int(config.get("stats_batch_size", 512)),
        seed=int(config.get("stats_seed", seed + 1_000)),
        min_std=float(config.get("normalizer_min_std", 0.05)),
    )
    model.set_feature_stats(stats_mean, stats_std)

    validation_generator = sampler.make_generator(
        int(config.get("validation_seed", seed + 2_000))
    )
    heldout_pair = sampler.sample_equal_clip_panel(
        int(config.get("validation_samples_per_clip", 128)),
        "heldout",
        validation_generator,
    )
    train_generator = sampler.make_generator(seed)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config.get("weight_decay", 1e-4)),
    )

    num_iterations = int(config["num_iterations"])
    batch_size = int(config["batch_size"])
    log_iter = int(config.get("log_iter", 100))
    output_iter = int(config.get("output_iter", 1_000))
    grad_clip = float(config.get("grad_clip_norm", 1.0))
    require_gate = bool(config.get("require_gate_pass", True))
    running = {"loss": 0.0, "invariance": 0.0, "variance": 0.0, "covariance": 0.0}
    running_grad = 0.0
    window_count = 0
    best_score = -float("inf")
    published = False

    print(
        f"Label-free skill encoder: clips={audit['num_clips']}, eligible={audit['eligible_clips']}, "
        f"train={len(audit['train_clips'])}, heldout={len(audit['heldout_clips'])}, "
        f"H_A={sampler.view_steps}, F={sampler.feature_dim}, y={model.embedding_dim}",
        flush=True,
    )

    model.train()
    for iteration in range(1, num_iterations + 1):
        pair = sampler.sample_pairs(batch_size, "train", train_generator)
        y_a = model(pair["view_a"])
        y_b = model(pair["view_b"])
        loss, components = vicreg_loss(
            y_a,
            y_b,
            invariance_weight=float(config.get("vicreg_invariance_weight", 25.0)),
            variance_weight=float(config.get("vicreg_variance_weight", 25.0)),
            covariance_weight=float(config.get("vicreg_covariance_weight", 1.0)),
            target_std=float(config.get("vicreg_target_std", 1.0)),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite VICReg loss at iteration {iteration}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        running["loss"] += float(loss)
        for name, value in components.items():
            running[name] += float(value)
        running_grad += float(grad_norm)
        window_count += 1

        if iteration % log_iter == 0 or iteration == num_iterations:
            record = {"iteration": iteration, "grad_norm": running_grad / window_count}
            record.update({name: value / window_count for name, value in running.items()})
            append_jsonl(out_dir / "metrics.jsonl", record)
            print(json.dumps(record, sort_keys=True), flush=True)
            running = {name: 0.0 for name in running}
            running_grad = 0.0
            window_count = 0

        if iteration % output_iter == 0 or iteration == num_iterations:
            validation, embeddings, nn_audit = evaluate(model, heldout_pair, sampler, config)
            validation["iteration"] = iteration
            append_jsonl(out_dir / "validation.jsonl", validation)
            payload = build_checkpoint(model, config, sampler, audit, iteration, validation)
            torch.save(payload, out_dir / "skill_encoder_last.pt")
            with open(out_dir / "validation_latest.json", "w", encoding="utf-8") as stream:
                json.dump(validation, stream, indent=2, sort_keys=True)

            eligible = validation["gate_passed"] or not require_gate
            if eligible and validation["selection_score"] > best_score:
                best_score = validation["selection_score"]
                published = True
                torch.save(payload, out_dir / "skill_encoder_best.pt")
                torch.save(payload, out_dir / "model.pt")
                torch.save(embeddings, out_dir / "heldout_embeddings.pt")
                with open(out_dir / "validation_best.json", "w", encoding="utf-8") as stream:
                    json.dump(validation, stream, indent=2, sort_keys=True)
                with open(out_dir / "nn_audit.json", "w", encoding="utf-8") as stream:
                    json.dump(nn_audit, stream, indent=2, sort_keys=True)
            print(json.dumps(validation, sort_keys=True), flush=True)

    if not published:
        raise RuntimeError("skill encoder non-collapse gate never passed; model.pt not published")
    return out_dir / "model.pt"


def main():
    parser = ArgumentParser(description="Train the R2 Stage 1A label-free skill encoder")
    parser.add_argument(
        "--cfg_path", default="tools/skill_encoder/config/skill_encoder_locomotion.yaml"
    )
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max_iters", type=int)
    args = parser.parse_args()
    train(args.cfg_path, args.out_dir, args.device, args.max_iters)


if __name__ == "__main__":
    main()
