"""The online screen must isolate curriculum from reward-shape changes."""
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/diagnostics"))
from dare_tolerance_learning_stress import (
    ERROR_NAMES, ToleranceProbeAgent, evaluate, mc_returns, screening_gate,
)
from kungfubot_tolerance_stress import AdaptiveTolerance, gap_reward


def test_mc_includes_terminal_reward_but_no_post_terminal_leak():
    rewards = torch.tensor([[1., 2.], [3., 4.], [100., 6.]])
    active = torch.tensor([[True, True], [True, True], [False, True]])
    expected = torch.tensor([[2.5, 5.5], [3., 7.], [0., 6.]])
    torch.testing.assert_close(mc_returns(rewards, active, .5), expected)


def comparison(ratio=.89, length=0., success=0.):
    item = dict(common_prefix_error_ratio={name: 1. for name in ERROR_NAMES},
                length_delta=length, success_delta=success)
    item["common_prefix_error_ratio"]["dof_vel"] = ratio
    return {mode: dict(item, common_prefix_error_ratio=item["common_prefix_error_ratio"].copy())
            for mode in ("phase0_mean", "rsi_sample")}


def test_gate_needs_two_repetitions_and_same_metric_in_both_modes():
    assert not screening_gate([comparison()])["passed"]
    assert screening_gate([comparison(), comparison()])["passed"]
    assert not screening_gate([comparison(), comparison(.95)])["passed"]
    other = comparison()
    other["rsi_sample"]["common_prefix_error_ratio"]["dof_vel"] = 1.
    other["rsi_sample"]["common_prefix_error_ratio"]["root_pos"] = .8
    assert not screening_gate([comparison(), other])["passed"]


def test_gate_rejects_hidden_tradeoffs_and_completion_loss():
    other = comparison()
    other["phase0_mean"]["common_prefix_error_ratio"]["body_pos"] = 1.06
    assert not screening_gate([comparison(), other])["passed"]
    assert not screening_gate([comparison(), comparison(length=-2.)])["passed"]
    assert not screening_gate([comparison(), comparison(success=-.03)])["passed"]


def test_gate_rejects_missing_modes_metrics_and_nonfinite_measurements():
    missing = comparison()
    missing.pop("rsi_sample")
    assert not screening_gate([missing, missing])["passed"]
    missing = comparison()
    missing["phase0_mean"]["common_prefix_error_ratio"].pop("root_pos")
    assert not screening_gate([comparison(), missing])["passed"]
    for invalid in (float("nan"), float("inf"), -1.):
        malformed = comparison()
        malformed["phase0_mean"]["common_prefix_error_ratio"]["root_pos"] = invalid
        assert not screening_gate([comparison(), malformed])["passed"]


class ToyModel:
    def eval_disc_raw(self, x):
        return (1 - x.square().sum(-1)).unsqueeze(-1)


class ToyAgent:
    _model = ToyModel()
    _pos_diff = torch.zeros(2)

    def __init__(self, variant):
        self.probe_variant = variant
        self.probe_tolerance = AdaptiveTolerance(2.)
        self.probe_last = {}

    def eval(self):
        return self

    def get_num_envs(self):
        return 2


def test_adaptive_mapping_uses_pre_update_sigma_in_time_major_order():
    data = torch.tensor([[1., 0.], [0., 1.], [.5, 0.], [0., .5]])
    toy = ToyAgent("adaptive")
    result = ToleranceProbeAgent._calc_disc_rewards(toy, data)
    expected = torch.cat([gap_reward(toy._model.eval_disc_raw(data[:2]).squeeze(), 1., 2.),
                          gap_reward(toy._model.eval_disc_raw(data[2:]).squeeze(), 1., 1.999)])
    torch.testing.assert_close(result, expected)
    assert toy.probe_tolerance.steps == 2
    fixed = ToyAgent("fixed")
    actual = ToleranceProbeAgent._calc_disc_rewards(fixed, data)
    torch.testing.assert_close(actual, gap_reward(toy._model.eval_disc_raw(data).squeeze(), 1., 2.))
    assert fixed.probe_tolerance.steps == 0
    assert fixed.probe_tolerance.sigma == 2.


def test_evaluation_resets_dead_replicas_without_reactivating_or_losing_terminal(monkeypatch):
    import dare_tolerance_learning_stress as module

    class Identity:
        def normalize(self, x):
            return x

        def unnormalize(self, x):
            return x

    class Environment:
        _rand_reset = True

        def __init__(self):
            self.step_index = 0
            self.partial_resets = []

        def reset(self, ids=None):
            if ids is not None:
                self.partial_resets.append(ids.tolist())
            return torch.zeros(2, 1), {}

        def get_motion_phase(self):
            return torch.zeros(2)

        def step(self, action):
            self.step_index += 1
            done = torch.tensor([1, 0] if self.step_index == 1 else [0, 2])
            return torch.zeros(2, 1), None, done, {
                "disc_obs_demo": torch.ones(2, 1), "disc_obs": torch.zeros(2, 1)}

    class Model:
        def eval_actor(self, obs):
            return torch.distributions.Normal(torch.zeros_like(obs), torch.ones_like(obs))

    class Agent:
        _device = "cpu"
        _obs_norm = Identity()
        _a_norm = Identity()
        _model = Model()
        _env = Environment()

        def eval(self):
            return self

        def _get_rng_state(self):
            return None

        def _set_rng_state(self, state):
            pass

        def get_num_envs(self):
            return 2

    agent = Agent()
    monkeypatch.setattr(module, "condition", lambda *args: None)
    monkeypatch.setattr(module, "release_cuda_cache", lambda: None)
    monkeypatch.setattr(module, "physics_errors", lambda env: (
        torch.ones(2, 7), None, torch.ones(2, 28), torch.ones(2, 3), None, None))
    scorer = lambda *args: (torch.ones(2), torch.ones(2))
    trace = evaluate(agent, True, True, 5, None, scorer)
    np.testing.assert_array_equal(trace["active"], [[True, True], [False, True]])
    np.testing.assert_array_equal(trace["done"], [[1, 0], [0, 2]])
    assert agent._env.partial_resets == [[0]]
    assert agent._env._rand_reset
