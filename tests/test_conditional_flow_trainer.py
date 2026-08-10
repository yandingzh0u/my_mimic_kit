from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

from learning.flow_matching.conditional_flow_matching_model import (
    ConditionalFlowMatchingModel,
)
from learning.skill_conditioned_flow_agent import SkillConditionedFlowAgent
from learning.skill_encoder.motion_features import make_feature_schema
from learning.skill_encoder.skill_encoder_model import LabelFreeSkillEncoder
from tools.flow_matching.conditional_motion_data import (
    ConditionalMotionWindowSampler,
    build_conditional_pair_times,
)
from tools.flow_matching.train_conditional_flow_matching import (
    build_checkpoint,
    coarse_semantic_tag,
    coarse_semantic_wrong_indices,
    conditional_selection_score,
    infer_smp_dimensions,
    load_frozen_encoder_artifact,
    random_cross_clip_derangement_indices,
    sample_null_mask,
)


def _flow_config():
    return {
        "arch_name": "ConditionalDiT",
        "num_disc_obs_steps": 10,
        "input_dim": 20,
        "input_channel": 2,
        "latent_dim": 8,
        "enforce_unit_latent": True,
        "num_layers": 1,
        "num_attention_heads": 1,
        "attention_head_dim": 8,
        "dropout": 0.0,
        "time_embed_scale": 49.0,
        "model_ema_decay": 0.9,
        "model_ema_steps": 1,
        "model_ema_update_after": 0,
        "normalizer_std_clip": 0.2,
        "reward_noise_samples": 1,
        "p_null": 0.1,
    }


def test_model_selection_tracks_condition_use_not_saturated_perturbation_gate():
    comparisons = {
        name: {
            "lower_paired_win_rate": 0.9,
            "min_win_rate": 0.5,
            "upper_to_lower_median_ratio": ratio,
            "min_median_ratio": 1.0,
        }
        for name, ratio in {
            "matched_vs_null": 1.2,
            "matched_vs_wrong_random": 1.5,
            "matched_vs_wrong_semantic": 1.7,
        }.items()
    }
    assert conditional_selection_score(comparisons) == pytest.approx(1.2)
    comparisons["matched_vs_null"]["upper_to_lower_median_ratio"] = 1.3
    assert conditional_selection_score(comparisons) == pytest.approx(1.3)


def test_exact_conditional_schedule_and_sampler_pair_do_not_overlap():
    assert infer_smp_dimensions((1140,)) == (1140, 114)
    with pytest.raises(ValueError, match="must be flat"):
        infer_smp_dimensions((10, 114))
    starts = torch.tensor([0.0, 1.25])
    context_times, window_times = build_conditional_pair_times(starts)
    assert context_times.shape == (2, 20)
    assert window_times.shape == (2, 10)
    torch.testing.assert_close(context_times[:, 0], starts)
    torch.testing.assert_close(context_times[:, -1] - starts, torch.full((2,), 19 / 30))
    torch.testing.assert_close(window_times[:, 0] - starts, torch.full((2,), 20 / 30))
    torch.testing.assert_close(window_times[:, -1] - starts, torch.full((2,), 29 / 30))
    assert torch.all(context_times[:, -1] < window_times[:, 0])

    captured = {}

    class FakeDataset:
        def _compute_smp_obs_demo(self, motion_ids, end_times):
            captured["window_end"] = end_times.cpu()
            return torch.zeros(motion_ids.numel(), 10, 114)

    sampler = ConditionalMotionWindowSampler.__new__(ConditionalMotionWindowSampler)
    sampler.lengths = torch.tensor([2.0, 3.0])
    sampler.pair_end_offset = 29 / 30
    sampler.view_steps = 20
    sampler.window_steps = 10
    sampler.control_freq = 30
    sampler.device = torch.device("cpu")
    sampler.dataset = FakeDataset()
    def fake_features(ids, times):
        captured["context_times"] = times.cpu()
        return torch.zeros(ids.numel(), 20, 44)

    sampler._features_at_times = fake_features
    sampler.mirror_family = lambda motion_id: f"family-{motion_id}"
    sampler.audit_tag = lambda motion_id: "walk" if motion_id == 0 else "run_left"
    sampler.gate_audit_tag = lambda motion_id: "walk" if motion_id == 0 else "run"

    pair = sampler._sample_conditional_pairs_for_ids(
        torch.tensor([0, 1]),
        torch.Generator().manual_seed(12),
        include_audit_labels=False,
    )
    assert pair["motion_window"].shape == (2, 10, 114)
    assert torch.all(pair["context_times"][:, -1] < pair["window_times"][:, 0])
    torch.testing.assert_close(captured["window_end"], pair["window_times"][:, -1])
    assert torch.all(pair["starts"] >= 0)
    assert torch.all(pair["starts"] <= sampler.lengths - 29 / 30)
    assert "audit_tags" not in pair and "gate_audit_tags" not in pair


def test_null_sampler_rate_and_conditional_model_api():
    generator = torch.Generator().manual_seed(77)
    mask = sample_null_mask(50_000, 0.1, generator, "cpu")
    assert mask.dtype == torch.bool and mask.shape == (50_000,)
    assert abs(float(mask.float().mean()) - 0.1) < 0.01
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        sample_null_mask(4, 1.1, generator, "cpu")

    model = ConditionalFlowMatchingModel(_flow_config(), "cpu")
    samples = torch.randn(8, 20)
    latent = torch.nn.functional.normalize(torch.randn(8, 8), dim=-1)
    loss = model(samples, latent, null_mask=mask[:8])
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_wrong_latent_indices_are_cross_clip_and_semantically_wrong():
    motion_ids = torch.tensor([0, 0, 1, 1, 2, 2, 3, 3])
    generator = torch.Generator().manual_seed(9)
    wrong = random_cross_clip_derangement_indices(motion_ids, generator)
    assert torch.unique(wrong).numel() == motion_ids.numel()
    assert torch.all(motion_ids[wrong] != motion_ids)

    # These are the sampler.gate_audit_tag outputs; fine heading labels never
    # define the semantic gate because the 44-D encoder is heading-invariant.
    tags = ["walk", "walk", "run", "run", "run", "run", "run", "run"]
    semantic_wrong = coarse_semantic_wrong_indices(tags, generator)
    assert all(
        coarse_semantic_tag(tags[index]) != coarse_semantic_tag(tags[int(other)])
        for index, other in enumerate(semantic_wrong)
    )


def _encoder_payload(gate_passed=True):
    encoder = LabelFreeSkillEncoder(44, embedding_dim=8, hidden_dim=32, num_layers=2)
    manifest = {
        "motion_file": "manifest.yaml",
        "dataset_yaml_sha256": "yaml-hash",
        "canonical_manifest_sha256": "canonical-hash",
        "clips": [
            {
                "motion_id": 0,
                "file": "clip.pkl",
                "weight": 1.0,
                "length_seconds": 2.0,
                "sha256": "clip-hash",
            }
        ],
    }
    feature_schema = {"name": "schema", "feature_dim": 44}
    schema = {
        "feature_dim": 44,
        "view_steps": 20,
        "embedding_dim": 8,
        "hidden_dim": 32,
        "num_layers": 2,
        "feature_schema": feature_schema,
        "dataset_manifest": manifest,
    }
    return {
        "format_version": 1,
        "model_type": "label_free_skill_encoder",
        "iteration": 123,
        "model_state_dict": encoder.state_dict(),
        "model_config": {"hidden_dim": 32, "num_layers": 2},
        "encoder_schema": schema,
        "data_contract": {"dataset_manifest": manifest},
        "validation": {"gate_passed": gate_passed, "selection_score": 2.0},
    }, manifest, feature_schema


def test_encoder_artifact_must_pass_gate_and_match_contract(tmp_path):
    payload, manifest, feature_schema = _encoder_payload()
    artifact = tmp_path / "encoder.pt"
    torch.save(payload, artifact)
    encoder, schema, gate = load_frozen_encoder_artifact(
        artifact,
        "cpu",
        expected_dataset_manifest=manifest,
        expected_feature_schema=feature_schema,
    )
    assert schema["latent_dim"] == 8
    assert gate["gate_passed"] is True and len(gate["artifact_sha256"]) == 64
    assert encoder.training is False
    assert all(not parameter.requires_grad for parameter in encoder.parameters())
    latent = encoder.runtime_z(torch.randn(3, 20, 44))
    torch.testing.assert_close(latent.norm(dim=-1), torch.ones(3))

    failed_payload, _, _ = _encoder_payload(gate_passed=False)
    torch.save(failed_payload, artifact)
    with pytest.raises(RuntimeError, match="did not pass Gate 1"):
        load_frozen_encoder_artifact(artifact, "cpu")

    mismatch_payload, _, _ = _encoder_payload()
    torch.save(mismatch_payload, artifact)
    bad_manifest = dict(manifest, canonical_manifest_sha256="other")
    with pytest.raises(RuntimeError, match="canonical_manifest_sha256"):
        load_frozen_encoder_artifact(
            artifact, "cpu", expected_dataset_manifest=bad_manifest
        )


def test_conditional_checkpoint_has_strict_r2_contract():
    model = ConditionalFlowMatchingModel(_flow_config(), "cpu")
    encoder = LabelFreeSkillEncoder(44, embedding_dim=8, hidden_dim=32, num_layers=2)
    encoder_schema = {
        "feature_dim": 44,
        "view_steps": 20,
        "embedding_dim": 8,
        "latent_dim": 8,
    }
    audit = {
        "dataset_manifest": {
            "motion_file": "manifest.yaml",
            "dataset_yaml_sha256": "yaml",
            "canonical_manifest_sha256": "canonical",
            "clips": [
                {
                    "motion_id": 0,
                    "file": "clip.pkl",
                    "weight": 1.0,
                    "length_seconds": 2.0,
                    "sha256": "clip",
                }
            ],
        },
        "paired_sampling": {"encoder_motion_window_overlap": False},
    }
    validation = {
        "conditional_expert_scale": 0.25,
        "gate_passed": True,
        "selection_score": 1.5,
    }
    payload = build_checkpoint(
        model,
        encoder,
        encoder_schema=encoder_schema,
        encoder_gate={"gate_passed": True},
        data_audit=audit,
        config=_flow_config(),
        times=torch.tensor([0.25, 0.5, 0.75]),
        base_noise=torch.zeros(1, 10, 2),
        iteration=9,
        validation=validation,
    )
    assert payload["format_version"] == 2
    assert payload["model_type"] == "conditional_flow_matching"
    assert set(payload).issuperset(
        {
            "model_state_dict",
            "encoder_state_dict",
            "model_config",
            "metadata",
            "calibration",
            "offline_validation",
            "encoder_gate",
        }
    )
    assert payload["metadata"]["latent_schema"]["non_null_norm"] == "unit_l2"
    assert payload["metadata"]["aggregation"] == "t_squared_weighted_mean"
    assert payload["metadata"]["K"] == 1
    assert set(payload["metadata"]["dataset_manifest"]) == {
        "clips",
        "dataset_yaml_sha256",
        "canonical_manifest_sha256",
    }
    assert payload["metadata"]["dataset_manifest"] == payload["metadata"][
        "encoder_schema"
    ]["dataset_manifest"]
    assert payload["metadata"]["dataset_manifest"]["clips"][0] == {
        "motion_id": 0,
        "file": "clip.pkl",
        "weight": 1.0,
        "length_seconds": 2.0,
        "sha256": "clip",
    }
    assert payload["calibration"]["conditional_expert_scale"] == 0.25


def test_checkpoint_passes_the_runtime_agents_exact_artifact_validator():
    model = ConditionalFlowMatchingModel(_flow_config(), "cpu")
    encoder = LabelFreeSkillEncoder(44, embedding_dim=8, hidden_dim=32, num_layers=2)
    manifest = {
        "clips": [
            {
                "motion_id": 0,
                "file": "clip.pkl",
                "weight": 1.0,
                "length_seconds": 2.0,
                "sha256": "clip",
            }
        ],
        "dataset_yaml_sha256": "yaml",
        "canonical_manifest_sha256": "canonical",
    }
    encoder_schema = {
        "feature_dim": 44,
        "view_steps": 20,
        "embedding_dim": 8,
        "latent_dim": 8,
        "hidden_dim": 32,
        "num_layers": 2,
        "feature_schema": make_feature_schema(),
        "dataset_manifest": manifest,
    }
    audit = {
        "dataset_manifest": manifest,
        "paired_sampling": {"encoder_motion_window_overlap": False},
    }
    validation = {
        "conditional_expert_scale": 0.25,
        "gate_passed": True,
        "selection_score": 1.5,
    }
    payload = build_checkpoint(
        model,
        encoder,
        encoder_schema=encoder_schema,
        encoder_gate={"gate_passed": True},
        data_audit=audit,
        config=_flow_config(),
        times=torch.tensor([0.25, 0.5, 0.75]),
        base_noise=torch.zeros(1, 10, 2),
        iteration=9,
        validation=validation,
    )

    class FakeDiscSpace:
        shape = (20,)

    class FakeRuntimeEnv:
        def get_disc_obs_space(self):
            return FakeDiscSpace()

        def get_skill_dataset_manifest(self):
            return manifest

    agent = SkillConditionedFlowAgent.__new__(SkillConditionedFlowAgent)
    agent._env = FakeRuntimeEnv()
    model_config, metadata, loaded_schema, calibration = agent._validate_artifact(
        payload
    )
    assert model_config["input_dim"] == 20
    assert metadata["dataset_manifest"] == loaded_schema["dataset_manifest"] == manifest
    assert calibration["conditional_expert_scale"] == 0.25

    if torch.cuda.is_available():
        cuda_payload = {
            **payload,
            "calibration": {
                **payload["calibration"],
                "times": payload["calibration"]["times"].cuda(),
                "base_noise": payload["calibration"]["base_noise"].cuda(),
                "conditional_expert_scale": torch.tensor(0.25, device="cuda"),
            },
        }
        _, _, _, cuda_calibration = agent._validate_artifact(cuda_payload)
        assert cuda_calibration["times"].is_cuda
