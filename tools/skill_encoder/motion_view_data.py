from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
import yaml

from learning.skill_encoder.motion_features import (
    build_motion_dynamic_features,
    make_feature_schema,
)
from tools.diffusion_model.motion_prior_dataset import MotionPriorData


def required_context_span(view_steps: int, control_freq: int) -> float:
    if view_steps <= 0 or control_freq <= 0:
        raise ValueError("view_steps and control_freq must be positive")
    # The two H-step views occupy 2H input states; one extra state at the end
    # is required for the final forward-difference foot velocity.
    return float(2 * view_steps) / float(control_freq)


def build_nonoverlapping_view_times(
    context_starts: torch.Tensor,
    view_steps: int,
    control_freq: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dt = 1.0 / float(control_freq)
    offsets = torch.arange(view_steps, device=context_starts.device) * dt
    view_a = context_starts.unsqueeze(-1) + offsets
    view_b = context_starts.unsqueeze(-1) + view_steps * dt + offsets
    return view_a, view_b


def stratified_family_split(
    family_to_ids: dict[str, list[int]],
    family_to_tag: dict[str, str],
    heldout_fraction: float,
    seed: int,
    min_heldout_families_per_tag: int = 2,
) -> tuple[list[int], list[int], list[str], list[str]]:
    """Split mirror families within each audit dynamics stratum.

    The filename-derived tags only select a representative train/audit
    partition. They are never exposed to the encoder or its objective.
    At least two held-out families per tag are required so a cross-family
    nearest-neighbour audit has a valid same-tag candidate.
    """
    if not 0.0 < heldout_fraction < 1.0:
        raise ValueError("heldout_fraction must be strictly inside (0,1)")
    if min_heldout_families_per_tag < 2:
        raise ValueError("cross-family audit requires at least two held-out families")
    if set(family_to_ids) != set(family_to_tag):
        raise ValueError("every mirror family must have exactly one audit tag")

    tag_to_families: dict[str, list[str]] = {}
    for family, ids in family_to_ids.items():
        if not ids:
            raise ValueError(f"mirror family {family!r} is empty")
        tag_to_families.setdefault(family_to_tag[family], []).append(family)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    heldout_families: list[str] = []
    for tag in sorted(tag_to_families):
        families = sorted(tag_to_families[tag])
        if len(families) <= min_heldout_families_per_tag:
            raise RuntimeError(
                f"audit stratum {tag!r} needs at least "
                f"{min_heldout_families_per_tag + 1} mirror families"
            )
        order = torch.randperm(len(families), generator=generator).tolist()
        target_clips = max(
            1,
            round(sum(len(family_to_ids[family]) for family in families) * heldout_fraction),
        )
        selected: list[str] = []
        selected_clips = 0
        for family_index in order:
            if (
                len(selected) >= min_heldout_families_per_tag
                and selected_clips >= target_clips
            ):
                break
            # Always retain at least one family from every stratum for training.
            if len(selected) >= len(families) - 1:
                break
            family = families[family_index]
            selected.append(family)
            selected_clips += len(family_to_ids[family])
        if len(selected) < min_heldout_families_per_tag:
            raise RuntimeError(f"failed to build a cross-family audit for {tag!r}")
        heldout_families.extend(selected)

    heldout_family_set = set(heldout_families)
    train_families = sorted(set(family_to_ids) - heldout_family_set)
    heldout_families = sorted(heldout_families)
    train_ids = sorted(
        motion_id
        for family in train_families
        for motion_id in family_to_ids[family]
    )
    heldout_ids = sorted(
        motion_id
        for family in heldout_families
        for motion_id in family_to_ids[family]
    )
    return train_ids, heldout_ids, train_families, heldout_families


class PairedMotionViewSampler:
    """Samples two non-overlapping H-step views from one valid local context."""

    def __init__(self, config: dict, device: str | torch.device):
        self.device = torch.device(device)
        self.view_steps = int(config["view_steps"])
        self.control_freq = int(config["control_freq"])
        self.context_span = required_context_span(self.view_steps, self.control_freq)
        self.motion_manifest_path = Path(config["motion_file"])
        with self.motion_manifest_path.open("r", encoding="utf-8") as stream:
            manifest_doc = yaml.safe_load(stream)
        self.manifest_files = [entry["file"] for entry in manifest_doc["motions"]]
        self.manifest_weights = [float(entry["weight"]) for entry in manifest_doc["motions"]]
        self.dataset = MotionPriorData(config, self.device)
        self.motion_lib = self.dataset._motion_lib
        self.foot_body_names = tuple(
            config.get("foot_body_names", ["right_foot", "left_foot"])
        )
        if len(self.foot_body_names) != 2:
            raise ValueError("foot_body_names must contain right and left foot names")
        self.foot_body_ids = tuple(
            self.dataset._kin_char_model.get_body_id(name) for name in self.foot_body_names
        )
        self.ground_height = float(config.get("contact_ground_height", 0.0))
        self.contact_height_threshold = float(
            config.get("contact_height_threshold", 0.08)
        )
        self.contact_speed_threshold = float(
            config.get("contact_speed_threshold", 0.4)
        )
        self.feature_schema = make_feature_schema(
            self.foot_body_names,
            self.foot_body_ids,
            self.ground_height,
            self.contact_height_threshold,
            self.contact_speed_threshold,
        )

        dof_size = int(self.dataset._kin_char_model.get_dof_size())
        self.feature_dim = 6 + dof_size + 6 + 2 + 2
        if self.feature_dim != self.feature_schema["feature_dim"]:
            raise ValueError(
                f"feature schema expects {self.feature_schema['feature_dim']} dims, got {self.feature_dim}"
            )

        lengths = self.motion_lib.get_motion_lengths().detach().cpu()
        weights = self.motion_lib.get_motion_weights().detach().float().cpu()
        eligible_mask = lengths >= self.context_span
        self.eligible_ids = torch.nonzero(eligible_mask, as_tuple=False).flatten()
        self.ineligible_ids = torch.nonzero(~eligible_mask, as_tuple=False).flatten()
        if self.eligible_ids.numel() < 2:
            raise RuntimeError(
                "fewer than two clips can contain two non-overlapping H_A views"
            )

        family_to_ids = {}
        for motion_id in self.eligible_ids.tolist():
            family_to_ids.setdefault(self.mirror_family(motion_id), []).append(motion_id)
        self.heldout_fraction = float(config.get("heldout_clip_fraction", 0.2))
        self.split_seed = int(config.get("split_seed", 17))
        self.min_heldout_families_per_tag = int(
            config.get("min_heldout_families_per_tag", 2)
        )
        family_to_tag = {}
        for family, motion_ids in family_to_ids.items():
            tags = {self.gate_audit_tag(motion_id) for motion_id in motion_ids}
            if len(tags) != 1:
                raise RuntimeError(f"mirror family {family!r} spans multiple audit strata")
            family_to_tag[family] = tags.pop()
        train_ids, heldout_ids, train_families, heldout_families = (
            stratified_family_split(
                family_to_ids,
                family_to_tag,
                self.heldout_fraction,
                self.split_seed,
                self.min_heldout_families_per_tag,
            )
        )
        self.heldout_ids = torch.tensor(heldout_ids, dtype=torch.long)
        self.train_ids = torch.tensor(train_ids, dtype=torch.long)
        self.heldout_families = heldout_families
        self.train_families = train_families
        self.family_to_tag = family_to_tag
        self.lengths = lengths
        self.weights = weights

    def make_generator(self, seed: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return generator

    def sample_pairs(
        self,
        batch_size: int,
        split: str,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        candidate_ids = self._split_ids(split)
        candidate_weights = self.weights[candidate_ids]
        sampled_index = torch.multinomial(
            candidate_weights / candidate_weights.sum(),
            num_samples=batch_size,
            replacement=True,
            generator=generator,
        )
        motion_ids_cpu = candidate_ids[sampled_index]
        return self._sample_pairs_for_ids(motion_ids_cpu, generator)

    def sample_equal_clip_panel(
        self,
        samples_per_clip: int,
        split: str,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        if samples_per_clip <= 0:
            raise ValueError("samples_per_clip must be positive")
        motion_ids_cpu = self._split_ids(split).repeat_interleave(samples_per_clip)
        order = torch.randperm(motion_ids_cpu.numel(), generator=generator)
        return self._sample_pairs_for_ids(motion_ids_cpu[order], generator)

    def _sample_pairs_for_ids(
        self,
        motion_ids_cpu: torch.Tensor,
        generator: torch.Generator,
    ) -> dict[str, torch.Tensor]:
        batch_size = motion_ids_cpu.numel()
        available = self.lengths[motion_ids_cpu] - self.context_span
        context_starts_cpu = torch.rand(batch_size, generator=generator) * available
        view_a_times_cpu, view_b_times_cpu = build_nonoverlapping_view_times(
            context_starts_cpu,
            self.view_steps,
            self.control_freq,
        )

        motion_ids = motion_ids_cpu.to(self.device)
        view_a = self._features_at_times(motion_ids, view_a_times_cpu.to(self.device))
        view_b = self._features_at_times(motion_ids, view_b_times_cpu.to(self.device))
        return {
            "view_a": view_a,
            "view_b": view_b,
            "motion_ids": motion_ids_cpu,
            "motion_lengths": self.lengths[motion_ids_cpu],
            "mirror_families": [
                self.mirror_family(int(motion_id)) for motion_id in motion_ids_cpu
            ],
            "context_starts": context_starts_cpu,
            "view_a_times": view_a_times_cpu,
            "view_b_times": view_b_times_cpu,
        }

    @torch.no_grad()
    def estimate_feature_stats(
        self,
        num_pairs: int,
        batch_size: int,
        seed: int,
        min_std: float = 1e-3,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        generator = self.make_generator(seed)
        feature_sum = torch.zeros(self.feature_dim, device=self.device, dtype=torch.float64)
        feature_sum_sq = torch.zeros_like(feature_sum)
        count = 0
        while count < num_pairs * 2 * self.view_steps:
            pair_batch = min(
                batch_size,
                (num_pairs * 2 * self.view_steps - count + 2 * self.view_steps - 1)
                // (2 * self.view_steps),
            )
            pair = self.sample_pairs(pair_batch, "train", generator)
            features = torch.cat((pair["view_a"], pair["view_b"]), dim=0).reshape(
                -1, self.feature_dim
            )
            feature_sum += features.double().sum(dim=0)
            feature_sum_sq += features.double().square().sum(dim=0)
            count += features.shape[0]
        mean = feature_sum / count
        variance = (feature_sum_sq / count - mean.square()).clamp_min(min_std * min_std)
        return mean.float(), variance.sqrt().float()

    def data_audit(self) -> dict:
        clip_fingerprints = []
        ordered_digest = hashlib.sha256()
        for motion_id in range(self.lengths.numel()):
            motion_path = Path(self.motion_lib.get_motion_file(motion_id))
            sha256 = hashlib.sha256(motion_path.read_bytes()).hexdigest()
            entry = {
                "motion_id": motion_id,
                "file": str(motion_path),
                "size_bytes": motion_path.stat().st_size,
                "sha256": sha256,
                "length_seconds": float(self.lengths[motion_id]),
                "weight": self.manifest_weights[motion_id],
                "audit_tag": self.audit_tag(motion_id),
            }
            clip_fingerprints.append(entry)
            ordered_digest.update(
                f"{motion_id}:{motion_path}:{sha256}\n".encode("utf-8")
            )

        def records(ids):
            return [clip_fingerprints[int(motion_id)] for motion_id in ids.tolist()]

        canonical_manifest = [
            {
                "motion_id": entry["motion_id"],
                "file": entry["file"],
                "weight": entry["weight"],
                "length_seconds": entry["length_seconds"],
                "sha256": entry["sha256"],
            }
            for entry in clip_fingerprints
        ]
        canonical_sha256 = hashlib.sha256(
            json.dumps(canonical_manifest, separators=(",", ":"), sort_keys=True).encode(
                "utf-8"
            )
        ).hexdigest()
        dataset_yaml_sha256 = hashlib.sha256(
            self.motion_manifest_path.read_bytes()
        ).hexdigest()
        def semantic_counts(ids):
            tags = [self.gate_audit_tag(int(motion_id)) for motion_id in ids]
            families = [self.mirror_family(int(motion_id)) for motion_id in ids]
            return {
                tag: {
                    "clips": tags.count(tag),
                    "families": len(
                        {family for family, item_tag in zip(families, tags) if item_tag == tag}
                    ),
                }
                for tag in sorted(set(tags))
            }

        return {
            "num_clips": int(self.lengths.numel()),
            "total_seconds": float(self.lengths.sum()),
            "view_steps": self.view_steps,
            "control_freq": self.control_freq,
            "required_context_span_seconds": self.context_span,
            "eligible_clips": int(self.eligible_ids.numel()),
            "eligibility_fraction": float(self.eligible_ids.numel() / self.lengths.numel()),
            "train_clips": records(self.train_ids),
            "heldout_clips": records(self.heldout_ids),
            "ineligible_clips": records(self.ineligible_ids),
            "feature_schema": self.feature_schema,
            "semantic_labels_used_for_training": False,
            "audit_labels_used_for_training": False,
            "audit_labels_used_for_split": True,
            "split_policy": (
                "evaluation-tag-stratified, mirror-family-grouped fixed-seed split; "
                "tags are partition-only and never encoder inputs/targets"
            ),
            "split_hard_semantic_counts": {
                "train": semantic_counts(self.train_ids),
                "heldout": semantic_counts(self.heldout_ids),
            },
            "split_seed": self.split_seed,
            "requested_heldout_clip_fraction_per_stratum": self.heldout_fraction,
            "min_heldout_families_per_tag": self.min_heldout_families_per_tag,
            "train_mirror_families": self.train_families,
            "heldout_mirror_families": self.heldout_families,
            "mirror_family_overlap": sorted(
                set(self.train_families).intersection(self.heldout_families)
            ),
            "audit_tag_mapping": "filename-only coarse tags: walk/run_forward/run_left/run_right",
            "hard_gate_tag_mapping": "heading-invariant evaluation-only tags: walk/run",
            "motion_source": {
                "manifest": str(self.motion_manifest_path),
                "manifest_sha256": dataset_yaml_sha256,
                "canonical_manifest_sha256": canonical_sha256,
                "ordered_clips_sha256": ordered_digest.hexdigest(),
                "clips": clip_fingerprints,
            },
            "dataset_manifest": {
                "motion_file": str(self.motion_manifest_path),
                "clips": canonical_manifest,
                "files": list(self.manifest_files),
                "weights": list(self.manifest_weights),
                "lengths_seconds": self.lengths.tolist(),
                "sha256": hashlib.sha256(
                    self.motion_manifest_path.read_bytes()
                ).hexdigest(),
                "dataset_yaml_sha256": dataset_yaml_sha256,
                "canonical_manifest_sha256": canonical_sha256,
            },
        }

    def audit_tag(self, motion_id: int) -> str:
        """Filename tag used only for partitioning and offline audit."""
        filename = str(self.motion_lib.get_motion_file(int(motion_id)))
        if "long_walk" in filename:
            return "walk"
        for tag in ("run_forward", "run_left", "run_right"):
            if tag in filename:
                return tag
        raise ValueError(f"no coarse partition/audit tag for {filename}")

    def gate_audit_tag(self, motion_id: int) -> str:
        fine_tag = self.audit_tag(motion_id)
        return "walk" if fine_tag == "walk" else "run"

    def mirror_family(self, motion_id: int) -> str:
        path = Path(self.motion_lib.get_motion_file(int(motion_id)))
        stem = path.stem.removesuffix("_mirror")
        return str(path.with_name(stem))

    def _split_ids(self, split: str) -> torch.Tensor:
        if split == "train":
            return self.train_ids
        if split == "heldout":
            return self.heldout_ids
        raise ValueError("split must be 'train' or 'heldout'")

    def _features_at_times(
        self,
        motion_ids: torch.Tensor,
        times: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, num_steps = times.shape
        next_time = times[:, -1:] + 1.0 / self.control_freq
        state_times = torch.cat((times, next_time), dim=1)
        tiled_ids = motion_ids.unsqueeze(-1).expand(-1, num_steps + 1).reshape(-1)
        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel = self.motion_lib.calc_motion_frame(
            tiled_ids, state_times.reshape(-1)
        )
        root_pos = root_pos.reshape(batch_size, num_steps + 1, -1)
        root_rot = root_rot.reshape(batch_size, num_steps + 1, -1)
        root_vel = root_vel.reshape(batch_size, num_steps + 1, -1)
        root_ang_vel = root_ang_vel.reshape(batch_size, num_steps + 1, -1)
        joint_rot = joint_rot.reshape(batch_size, num_steps + 1, *joint_rot.shape[-2:])
        dof_vel = dof_vel.reshape(batch_size, num_steps + 1, -1)
        body_pos, _ = self.dataset._kin_char_model.forward_kinematics(
            root_pos.reshape(-1, 3),
            root_rot.reshape(-1, 4),
            joint_rot.reshape(-1, *joint_rot.shape[-2:]),
        )
        body_pos = body_pos.reshape(batch_size, num_steps + 1, body_pos.shape[-2], 3)
        foot_pos = body_pos[:, :, self.foot_body_ids, :]
        features = build_motion_dynamic_features(
            root_rot=root_rot[:, :num_steps],
            root_vel=root_vel[:, :num_steps],
            root_ang_vel=root_ang_vel[:, :num_steps],
            dof_vel=dof_vel[:, :num_steps],
            foot_pos=foot_pos,
            timestep=1.0 / self.control_freq,
            ground_height=self.ground_height,
            contact_height_threshold=self.contact_height_threshold,
            contact_speed_threshold=self.contact_speed_threshold,
        )
        if features.shape != (batch_size, num_steps, self.feature_dim):
            raise RuntimeError(f"unexpected dynamic feature shape {tuple(features.shape)}")
        if not torch.isfinite(features).all():
            raise FloatingPointError("motion dynamic features contain non-finite values")
        return features
