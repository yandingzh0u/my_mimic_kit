from pathlib import Path
import sys

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))

from learning.skill_conditioned_flow_agent import SkillConditionedFlowAgent
from learning.skill_conditioned_ppo_model import SkillConditionedPPOModel
from learning.skill_conditioned_runtime import (
    assert_dataset_manifest_equal,
    context_times,
    rsi_context_starts,
)
from util.arg_parser import ArgParser
import run as run_module


class _FakeEnv:
    def get_obs_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(5,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(-np.ones(2), np.ones(2), dtype=np.float32)

    def get_disc_obs_space(self):
        return spaces.Box(-np.inf, np.inf, shape=(20,), dtype=np.float32)

    def get_skill_dataset_manifest(self):
        return _manifest()


def _policy_config():
    return {
        "actor_net": "fc_2layers_128units",
        "critic_net": "fc_2layers_128units",
        "actor_init_output_scale": 0.01,
        "actor_std_type": "FIXED",
        "action_std": 0.2,
    }


def _manifest():
    return {
        "clips": [
            {
                "motion_id": 0,
                "file": "data/a.pkl",
                "weight": 1.0,
                "length_seconds": 2.0,
                "sha256": "a",
            }
        ],
        "dataset_yaml_sha256": "yaml",
        "canonical_manifest_sha256": "canonical",
    }


def test_skill_policy_conditions_actor_and_critic_on_eight_dimensional_z():
    model = SkillConditionedPPOModel(_policy_config(), _FakeEnv(), latent_dim=8)
    obs = torch.randn(4, 5)
    latent = F.normalize(torch.randn(4, 8), dim=-1)
    assert model.eval_actor(obs, latent).mode.shape == (4, 2)
    assert model.eval_critic(obs, latent).shape == (4, 1)
    rollout_obs = torch.randn(3, 4, 5)
    rollout_latent = F.normalize(torch.randn(3, 4, 8), dim=-1)
    assert model.eval_critic(rollout_obs, rollout_latent).shape == (3, 4, 1)
    with pytest.raises(ValueError, match="latent must have shape"):
        model.eval_actor(obs, torch.randn(4, 7))


def test_context_recovers_exact_a19_rsi_and_has_no_repeated_points():
    starts_expected = torch.tensor([0.0, 0.3, 2.0])
    reset = starts_expected + 19.0 / 30.0
    lengths = torch.tensor([2.0, 2.0, 3.0])
    starts = rsi_context_starts(reset, lengths)
    times = context_times(starts)
    assert times.shape == (3, 20)
    torch.testing.assert_close(times[:, 1:] - times[:, :-1], torch.full((3, 19), 1 / 30))
    torch.testing.assert_close(starts, starts_expected)
    torch.testing.assert_close(times[:, 19], reset)


def test_manifest_comparison_is_order_hash_weight_and_length_strict():
    clips = [
        {"motion_id": 0, "file": "data/a.pkl", "weight": 1.0, "length_seconds": 2.0, "sha256": "a"},
        {"motion_id": 1, "file": "data/b.pkl", "weight": 2.0, "length_seconds": 3.0, "sha256": "b"},
    ]
    manifest = {"clips": clips, "dataset_yaml_sha256": "yaml", "canonical_manifest_sha256": "canonical"}
    assert_dataset_manifest_equal(manifest, manifest)
    reordered = {**manifest, "clips": list(reversed(clips))}
    with pytest.raises(ValueError, match="clip 0"):
        assert_dataset_manifest_equal(manifest, reordered)
    length_drift = {**manifest, "clips": [dict(clips[0]), dict(clips[1])]}
    length_drift["clips"][0]["length_seconds"] += 2e-5
    with pytest.raises(ValueError, match="length"):
        assert_dataset_manifest_equal(manifest, length_drift)


def test_agent_accepts_only_the_exact_format_two_trainer_contract():
    manifest = _manifest()
    feature_schema = {
        "feature_dim": 44,
        "foot_body_names": ["right_foot", "left_foot"],
        "foot_body_ids": [11, 14],
        "contact_proxy": {
            "ground_height": 0.0,
            "height_threshold": 0.08,
            "speed_threshold": 0.4,
        },
    }
    encoder_schema = {
        "feature_dim": 44,
        "view_steps": 20,
        "embedding_dim": 8,
        "hidden_dim": 16,
        "num_layers": 1,
        "feature_schema": feature_schema,
        "dataset_manifest": manifest,
    }
    metadata = {
        "input_dim": 20,
        "frame_dim": 2,
        "window_steps": 10,
        "latent_dim": 8,
        "condition_mode": "continuous_or_null",
        "runtime_embedding": "l2_normalize(y)",
        "time_embed_scale": 49.0,
        "aggregation": "t_squared_weighted_mean",
        "reward_noise_samples": 1,
        "encoder_schema": encoder_schema,
        "dataset_manifest": manifest,
        "condition_schema": {
            "type": "continuous_latent_with_learned_null",
            "latent_dim": 8,
            "runtime_embedding": "l2_normalize(y)",
            "conditional_latent_norm": "unit_l2",
            "aggregation": "t_squared_weighted_mean",
        },
    }
    artifact = {
        "format_version": 2,
        "model_type": "conditional_flow_matching",
        "model_config": {
            "input_dim": 20,
            "latent_dim": 8,
            "num_disc_obs_steps": 10,
            "time_embed_scale": 49.0,
            "enforce_unit_latent": True,
        },
        "model_state_dict": {},
        "encoder_state_dict": {},
        "metadata": metadata,
        "calibration": {
            "conditional_expert_scale": 2.0,
            "times": [0.25, 0.5, 0.75],
            "base_noise": torch.zeros(1, 20),
        },
        "offline_validation": {"gate_passed": True},
        "encoder_gate": {"gate_passed": True},
    }
    agent = SkillConditionedFlowAgent.__new__(SkillConditionedFlowAgent)
    agent._env = _FakeEnv()
    _, returned_metadata, returned_encoder, _ = agent._validate_artifact(artifact)
    assert returned_metadata is metadata and returned_encoder is encoder_schema
    artifact["metadata"] = dict(metadata, dataset_manifest={**manifest, "dataset_yaml_sha256": "drift"})
    with pytest.raises(ValueError, match="dataset_manifest differ"):
        agent._validate_artifact(artifact)


class _FakePrior(nn.Module):
    def normalize(self, value):
        return value

    def aggregate_mismatch(self, x1, latent, times, base_noise, use_ema=True):
        SkillConditionedFlowAgent._assert_runtime_latent(latent)
        return torch.arange(1, x1.shape[0] + 1, device=x1.device, dtype=x1.dtype)


class _FakeBuffer:
    def __init__(self):
        self.flat = {
            "disc_obs": torch.randn(4, 20),
            "latent": F.normalize(torch.randn(4, 8), dim=-1),
            "reward": torch.full((4,), 99.0),
        }

    def get_data_flat(self, name):
        return self.flat[name]

    def set_data_flat(self, name, value):
        self.flat[name] = value


def test_r2_reward_replaces_task_reward_and_uses_conditional_scaled_mismatch():
    agent = SkillConditionedFlowAgent.__new__(SkillConditionedFlowAgent)
    nn.Module.__init__(agent)
    agent._exp_buffer = _FakeBuffer()
    agent._prior_model = _FakePrior()
    agent._flow_times = torch.tensor([0.25, 0.5, 0.75])
    agent._flow_base_noise = torch.zeros(1, 20)
    agent._conditional_expert_scale = torch.tensor(2.0)
    agent._flow_reward_alpha = 0.003
    agent._smp_eval_batch_size = 16
    info = agent._compute_rewards()
    expected = torch.exp(-0.003 * torch.arange(1.0, 5.0) / 2.0)
    torch.testing.assert_close(agent._exp_buffer.flat["reward"], expected)
    torch.testing.assert_close(info["conditional_reward_mean"], expected.mean())


def test_runtime_latent_contract_rejects_nonunit_or_nonfinite_commands():
    SkillConditionedFlowAgent._assert_runtime_latent(
        F.normalize(torch.randn(3, 8), dim=-1)
    )
    with pytest.raises(ValueError, match="unit L2 norm"):
        SkillConditionedFlowAgent._assert_runtime_latent(torch.ones(3, 8))
    bad = F.normalize(torch.randn(3, 8), dim=-1)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        SkillConditionedFlowAgent._assert_runtime_latent(bad)


def test_formal_args_cannot_inherit_the_legacy_smp_environment():
    parser = ArgParser()
    assert parser.load_file("args/skill_conditioned_flow_humanoid_args.txt")
    assert parser.parse_int("num_envs") == 4096
    assert parser.parse_string("env_config") == (
        "data/envs/skill_conditioned_location_humanoid_env.yaml"
    )
    assert parser.parse_string("agent_config") == (
        "data/agents/skill_conditioned_flow_humanoid_agent.yaml"
    )
    assert parser.parse_string("out_dir") == "output/skill_conditioned_flow/locomotion"


def test_external_skill_cli_requires_test_mode_and_exactly_one_clip_identity():
    class FakeAgent:
        command = None

        def set_skill_command(self, **kwargs):
            self.command = kwargs

    parser = ArgParser()
    parser.load_args(
        [
            "--skill_clip_sha256",
            "abc123",
            "--skill_context_start_sec",
            "0.4",
        ]
    )
    agent = FakeAgent()
    run_module.apply_skill_command(parser, agent, "test")
    assert agent.command == {
        "motion_path": None,
        "clip_sha256": "abc123",
        "context_start_sec": 0.4,
    }
    with pytest.raises(ValueError, match="only in test mode"):
        run_module.apply_skill_command(parser, agent, "train")

    invalid = ArgParser()
    invalid.load_args(
        [
            "--skill_motion_path",
            "data/a.pkl",
            "--skill_clip_sha256",
            "abc123",
            "--skill_context_start_sec",
            "0.4",
        ]
    )
    with pytest.raises(ValueError, match="exactly one"):
        run_module.apply_skill_command(invalid, agent, "test")
