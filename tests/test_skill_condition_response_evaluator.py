from pathlib import Path
import inspect
import sys

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "mimickit"))

from learning.base_agent import AgentMode
from learning.skill_conditioned_flow_agent import SkillConditionedFlowAgent
from tools.evaluation.evaluate_skill_condition_response import (
    compare_initial_states,
    evaluate_condition_response,
)


class _LegalContextEnv:
    def get_num_envs(self):
        return 1

    def get_expert_skill_context(self, **kwargs):
        if kwargs.get("motion_path") != "expert_a.pkl":
            raise ValueError("not in manifest")
        return {
            "features": torch.ones(1, 20, 44),
            "motion_id": 3,
            "motion_path": "expert_a.pkl",
            "clip_sha256": "sha-a",
            "context_start_sec": float(kwargs["context_start_sec"]),
        }


class _UnitEncoder(nn.Module):
    def runtime_z(self, features):
        value = features.mean(dim=(1, 2), keepdim=False)
        latent = torch.zeros(features.shape[0], 8, device=features.device)
        latent[:, 0] = value
        return F.normalize(latent, dim=-1)


class _EvalPrior(nn.Module):
    input_dim = 4

    def normalize(self, value):
        return value

    def aggregate_mismatch(self, x1, latent, times, base_noise, use_ema=True):
        return torch.square(x1[:, 0] - latent[:, 0]) + 0.5


def _bare_real_agent(mode=AgentMode.TEST):
    agent = SkillConditionedFlowAgent.__new__(SkillConditionedFlowAgent)
    nn.Module.__init__(agent)
    agent._mode = mode
    agent._env = _LegalContextEnv()
    agent._device = torch.device("cpu")
    agent._encoder_feature_schema = {"feature_dim": 44}
    agent._skill_encoder = _UnitEncoder()
    agent._current_latent = torch.zeros(1, 8)
    return agent


def test_evaluation_context_api_is_test_only_and_has_no_raw_latent_argument():
    signature = inspect.signature(SkillConditionedFlowAgent.set_evaluation_skill_context)
    assert "latent" not in signature.parameters

    agent = _bare_real_agent(mode=AgentMode.TRAIN)
    with pytest.raises(ValueError, match="only in test mode"):
        agent.set_evaluation_skill_context(
            motion_path="expert_a.pkl", context_start_sec=0.2
        )
    agent._mode = AgentMode.TEST
    applied = agent.set_evaluation_skill_context(
        motion_path="expert_a.pkl", context_start_sec=0.2
    )
    assert applied["clip_sha256"] == "sha-a"
    torch.testing.assert_close(agent._current_latent, torch.tensor([[1.0] + [0.0] * 7]))
    with pytest.raises(ValueError, match="not in manifest"):
        agent.set_evaluation_skill_context(
            motion_path="invented.pkl", context_start_sec=0.2
        )


def test_evaluation_window_scoring_uses_frozen_calibration_without_reward_alpha():
    agent = _bare_real_agent()
    agent._prior_model = _EvalPrior()
    agent._flow_times = torch.tensor([0.25, 0.5, 0.75])
    agent._flow_base_noise = torch.zeros(1, 4)
    agent._conditional_expert_scale = torch.tensor(2.0)
    agent._smp_eval_batch_size = 2
    result = agent.score_evaluation_windows(
        torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]),
        motion_path="expert_a.pkl",
        context_start_sec=0.2,
    )
    torch.testing.assert_close(result["raw_mismatch"], torch.tensor([0.5, 1.5]))
    torch.testing.assert_close(result["scaled_mismatch"], torch.tensor([0.25, 0.75]))


class _PairedFakeEnv:
    def __init__(self):
        self.agent = None
        self.reset()

    def reset(self):
        self.step_count = 0
        self.root = torch.zeros(1, 3)
        self.dof = torch.zeros(1, 2)
        self.body = torch.zeros(1, 2, 3)
        self.target = torch.tensor([[2.0, 0.0, 0.0]])

    def step(self, action):
        self.step_count += 1
        self.root[:, 0] += action[:, 0]
        self.dof[:, 0] += action[:, 0]
        self.body[:, :, 0] += action[:, :1]
        obs = self.root[:, :2].clone()
        disc = torch.stack(
            (self.root[:, 0], action[:, 0], self.dof[:, 0], torch.zeros(1)), dim=-1
        )
        done = torch.tensor([1 if self.step_count >= 12 else 0])
        return obs, torch.zeros(1), done, {"disc_obs": disc}

    def get_skill_evaluation_state(self, env_ids=None):
        return {
            "root_pos": self.root.clone(),
            "root_rot": torch.tensor([[0.0, 0.0, 0.0, 1.0]]),
            "root_vel": torch.zeros(1, 3),
            "root_ang_vel": torch.zeros(1, 3),
            "dof_pos": self.dof.clone(),
            "dof_vel": torch.zeros(1, 2),
            "body_pos": self.body.clone(),
            "target_pos": self.target.clone(),
            "observation": self.root[:, :2].clone(),
            "disc_observation": torch.zeros(1, 4),
            "time": torch.tensor([self.step_count / 30.0]),
        }


class _PairedFakeAgent:
    def __init__(self):
        self._env = _PairedFakeEnv()
        self._mode = AgentMode.TRAIN
        self._latent = 0.0

    def eval(self):
        return self

    def set_mode(self, mode):
        self._mode = mode

    def get_num_envs(self):
        return 1

    def set_skill_command(self, **kwargs):
        assert kwargs["motion_path"] == "reset.pkl"

    def _reset_envs(self):
        self._env.reset()
        return torch.zeros(1, 2), {"disc_obs": torch.zeros(1, 4)}

    @staticmethod
    def _z_for(path):
        return 1.0 if path == "expert_a.pkl" else -1.0

    def set_evaluation_skill_context(self, **kwargs):
        assert self._mode == AgentMode.TEST
        path = kwargs["motion_path"]
        if path not in ("expert_a.pkl", "expert_b.pkl"):
            raise ValueError("not in manifest")
        self._latent = self._z_for(path)
        latent = torch.tensor([self._latent] + [0.0] * 7)
        return {
            "motion_id": 0 if self._latent > 0 else 1,
            "motion_path": path,
            "clip_sha256": "sha-a" if self._latent > 0 else "sha-b",
            "context_start_sec": kwargs["context_start_sec"],
            "latent": latent,
        }

    def _decide_action(self, obs, info):
        return torch.tensor([[self._latent]]), {}

    def score_evaluation_windows(self, windows, **kwargs):
        path = kwargs["motion_path"]
        condition = self._z_for(path)
        windows = torch.as_tensor(windows).float()
        mismatch = torch.square(windows[:, 1] - condition) + 0.1
        return {
            "motion_id": 0 if condition > 0 else 1,
            "motion_path": path,
            "clip_sha256": "sha-a" if condition > 0 else "sha-b",
            "context_start_sec": kwargs["context_start_sec"],
            "raw_mismatch": mismatch,
            "scaled_mismatch": mismatch,
        }


def test_paired_evaluator_holds_initial_state_fixed_and_reports_response_and_w_a_test():
    report = evaluate_condition_response(
        _PairedFakeAgent(),
        reset_context={"motion_path": "reset.pkl", "context_start_sec": 0.0},
        context_a={"motion_path": "expert_a.pkl", "context_start_sec": 0.0},
        context_b={"motion_path": "expert_b.pkl", "context_start_sec": 0.0},
        rollout_steps=12,
        seed=7,
    )
    assert report["protocol"]["arbitrary_latent_injection"] is False
    assert report["initial_state_check"]["equal_within_tolerance"] is True
    assert report["behavior_difference"]["behavior_changed"] is True
    assert report["behavior_difference"]["root_endpoint_separation"] == pytest.approx(24.0)
    assert report["w_a_context_test"]["paired_matched_lower_rate"] == 1.0
    assert report["w_a_context_test"]["counterfactual_over_matched_mean_ratio"] > 10.0
    assert report["w_a_context_test"]["warmup_steps"] == 9
    assert report["w_a_context_test"]["scored_window_count"] == 3


def test_initial_state_comparison_detects_policy_relevant_drift():
    state_a = {"root": torch.zeros(3), "target": torch.zeros(2)}
    state_b = {"root": torch.zeros(3), "target": torch.tensor([0.0, 0.01])}
    comparison = compare_initial_states(state_a, state_b, tolerance=1e-6)
    assert comparison["equal_within_tolerance"] is False
    assert comparison["per_field_max_abs_difference"]["target"] == pytest.approx(0.01)


def test_mismatch_audit_rejects_branches_without_a_fully_policy_h10_window():
    with pytest.raises(RuntimeError, match="at least 10"):
        evaluate_condition_response(
            _PairedFakeAgent(),
            reset_context={"motion_path": "reset.pkl", "context_start_sec": 0.0},
            context_a={"motion_path": "expert_a.pkl", "context_start_sec": 0.0},
            context_b={"motion_path": "expert_b.pkl", "context_start_sec": 0.0},
            rollout_steps=9,
            seed=7,
        )
