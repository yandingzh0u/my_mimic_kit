"""Contracts for the sequential CPL1/CPL7 Body-split ablations."""

import copy
from pathlib import Path

import gymnasium.spaces as spaces
import pytest
import torch
import torch.nn.functional as F
import yaml

from envs.add_env import build_disc_error_groups
from learning.add_agent import calc_unscaled_disc_reward
from learning.agent_builder import build_agent
from learning.dare_body_split_cpl_agent import DAREBodySplitCPLAgent
from learning.dare_body_split_cpl_model import (
    ConvexPotentialBlock, DAREBodySplitCPLModel, HIDDEN_WIDTHS,
)
from learning.dare_body_split_sn_model import (
    DAREBodySplitSNModel, hierarchical_parent_mean,
)


ROOT = Path(__file__).resolve().parents[1]


class Env:
    def __init__(self, frames=1):
        self.frames = frames

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1., 1., shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(172 * self.frames,))

    def get_disc_error_groups(self):
        return build_disc_error_groups(self.frames, 15, 15, 28, 172 * self.frames)

    def get_num_envs(self):
        return 4


def config(depth):
    return yaml.safe_load((ROOT / (
        "data/agents/dare_body_split_cpl{}_humanoid_agent.yaml".format(depth))).read_text())


def agent(depth):
    return DAREBodySplitCPLAgent(config(depth), Env(), "cpu")


@pytest.mark.parametrize("depth", [1, 7])
def test_agent_builder_dispatches_both_formal_configs(depth):
    filename = ROOT / f"data/agents/dare_body_split_cpl{depth}_humanoid_agent.yaml"
    instance = build_agent(str(filename), Env(), "cpu")
    assert isinstance(instance, DAREBodySplitCPLAgent)
    assert instance._model.get_disc_cpl_depth() == depth


@pytest.mark.parametrize("depth", [1, 7])
def test_configs_change_only_agent_name_and_cpl_depth(depth):
    baseline = yaml.safe_load((ROOT / "data/agents/dare_body_split_sn_humanoid_agent.yaml").read_text())
    candidate = config(depth)
    assert candidate.pop("agent_name") == "DAREBodySplitCPL"
    assert candidate["model"].pop("disc_cpl_depth") == depth
    baseline.pop("agent_name")
    assert candidate == baseline


@pytest.mark.parametrize("depth,frames", [(1, 1), (7, 1), (1, 3)])
def test_shapes_and_common_baseline_parameters_are_seed_identical(depth, frames):
    env = Env(frames)
    torch.manual_seed(49)
    baseline = DAREBodySplitSNModel(config(depth)["model"], env).eval()
    torch.manual_seed(49)
    candidate = DAREBodySplitCPLModel(config(depth)["model"], env)
    assert candidate.get_disc_cpl_depth() == depth
    assert candidate.get_disc_error_groups() == baseline.get_disc_error_groups()
    assert candidate.get_disc_group_width() == HIDDEN_WIDTHS[0]
    assert candidate.get_disc_group_total_width() == 4 * HIDDEN_WIDTHS[0]
    for a, b in zip(baseline.get_actor_params() + baseline.get_critic_params(),
                    candidate.get_actor_params() + candidate.get_critic_params()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for dimension, base, branch in zip((9, 45, 84, 34),
                                      baseline._disc_branches.branches,
                                      candidate._disc_branches.branches):
        assert isinstance(branch[1], torch.nn.ReLU)
        assert sum(isinstance(m, ConvexPotentialBlock) for m in branch) == depth
        assert [type(m) for m in branch[3:-1]] == [ConvexPotentialBlock] * depth
        assert [(m.in_features, m.out_features) for m in (branch[0], branch[2], branch[-1])] == [
            (dimension * frames, 341), (341, 512), (512, 1)]
        for base_layer, new_layer in zip((base[0], base[2], base[-1]),
                                        (branch[0], branch[2], branch[-1])):
            torch.testing.assert_close(new_layer.weight, base_layer.weight, rtol=0, atol=0)
            torch.testing.assert_close(new_layer.bias, base_layer.bias, rtol=0, atol=0)
            assert not hasattr(new_layer, "parametrizations")
        for block in branch[3:-1]:
            assert "weight" in block.linear.parametrizations
    candidate.eval()
    assert candidate.eval_disc(torch.randn(11, 172 * frames)).shape == (11, 4)


def test_historical_formula_and_one_power_iteration_per_forward(monkeypatch):
    block = ConvexPotentialBlock(9)
    parametrization = block.linear.parametrizations.weight[0]
    original = parametrization._power_method
    calls = []

    def count(weight, n_power_iterations):
        calls.append(n_power_iterations)
        return original(weight, n_power_iterations)

    monkeypatch.setattr(parametrization, "_power_method", count)
    block.train()
    inputs = torch.randn(13, 9)
    actual = block(inputs)
    assert calls == [1]
    block.eval()
    weight = block.linear.weight
    expected = inputs - 2.0 * F.linear(
        F.relu(F.linear(inputs, weight, block.linear.bias)), weight.T)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_tied_transpose_input_jacobian_matches_analytic_formula():
    block = ConvexPotentialBlock(7).double().eval()
    with torch.no_grad():
        block.linear.bias.copy_(torch.linspace(-.2, .3, 7, dtype=torch.double))
    x = torch.randn(7, dtype=torch.double, requires_grad=True)
    weight = block.linear.weight
    mask = (F.linear(x, weight, block.linear.bias) > 0).to(torch.double)
    expected = torch.eye(7, dtype=torch.double) - 2.0 * weight.T @ (mask[:, None] * weight)
    actual = torch.autograd.functional.jacobian(block, x)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("depth", [1, 7])
def test_cpu_classifier_step_reaches_every_parameter_and_reward_mapping_is_unchanged(depth):
    torch.manual_seed(17)
    a = agent(depth)
    raw = torch.randn(128, 172)
    a._disc_buffer.push({"disc_obs_demo": raw[:64].unsqueeze(1),
                         "disc_obs": torch.zeros_like(raw[:64]).unsqueeze(1)})
    batch = {"disc_obs_demo": raw[64:], "disc_obs": torch.zeros_like(raw[64:])}
    a.train()
    losses = a._compute_disc_loss(batch)
    assert all(torch.isfinite(value).all() for value in losses.values())
    assert "disc_grad_penalty" in losses and losses["disc_grad_penalty"] == 0
    a._disc_optimizer.step(losses["disc_loss"])
    assert a._disc_optimizer.get_steps() == 1
    for name, parameter in a._model._disc_branches.named_parameters():
        assert parameter.grad is not None, name
        assert torch.isfinite(parameter.grad).all(), name
        assert parameter.grad.abs().sum() > 0, name
        assert torch.isfinite(parameter).all(), name
    a.eval()
    torch.testing.assert_close(a._calc_disc_rewards(raw), hierarchical_parent_mean(
        calc_unscaled_disc_reward(a._model.eval_disc(raw))), rtol=0, atol=0)
    assert not hasattr(a, "_disc_reward_scale")
    assert not hasattr(a._model, "get_disc_logit_scale")


@pytest.mark.parametrize("depth", [1, 7])
def test_full_cpu_checkpoint_resume_and_cross_architecture_rejection(depth):
    a = agent(depth)
    raw = torch.randn(19, 172)

    def deterministic_step(instance):
        instance.train()
        pos = instance._model.eval_disc(torch.zeros(1, 172))
        neg = instance._model.eval_disc(raw)
        loss = .5 * (F.binary_cross_entropy_with_logits(
            pos, torch.ones_like(pos), reduction="none").mean(0)
            + F.binary_cross_entropy_with_logits(
                neg, torch.zeros_like(neg), reduction="none").mean(0)).sum()
        loss = loss + instance._disc_logit_reg * instance._model.get_disc_logit_weights().square().sum()
        instance._disc_optimizer.step(loss)

    deterministic_step(a)
    a.eval()
    saved = copy.deepcopy(a._get_checkpoint_extra_state())
    assert "body_split_cpl" in saved and "body_split_sn" not in saved and "multi_disc" not in saved
    restored = agent(depth).eval()
    restored.load_state_dict(copy.deepcopy(a.state_dict()), strict=True)
    restored._load_checkpoint_extra_state(saved)
    restored._disc_optimizer.load_state_dict(copy.deepcopy(a._disc_optimizer.state_dict()))
    assert restored._disc_optimizer.get_steps() == 1
    assert all("momentum_buffer" in item for item in
               restored._disc_optimizer._optimizer.state.values())
    torch.testing.assert_close(restored._calc_disc_rewards(raw), a._calc_disc_rewards(raw), rtol=0, atol=0)
    deterministic_step(a)
    deterministic_step(restored)
    assert a._disc_optimizer.get_steps() == restored._disc_optimizer.get_steps() == 2
    for key, value in a.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value, rtol=0, atol=0)
    invalid = copy.deepcopy(saved)
    invalid["body_split_cpl"]["cpl_depth"] = 8
    with pytest.raises(ValueError, match="different Body-split CPL semantics"):
        restored._load_checkpoint_extra_state(invalid)
    with pytest.raises(ValueError, match="different Body-split CPL semantics"):
        restored._load_checkpoint_extra_state({"body_split_sn": {}})
    state = copy.deepcopy(a._model.state_dict())
    state["_body_split_cpl_schema"][1] = 8
    with pytest.raises(RuntimeError, match="different Body-split CPL architecture"):
        restored._model.load_state_dict(state, strict=False)
    del state["_body_split_cpl_schema"]
    with pytest.raises(RuntimeError, match="different Body-split CPL architecture"):
        restored._model.load_state_dict(state, strict=False)


@pytest.mark.parametrize("depth", [1, 7])
def test_normalizer_completion_has_no_reward_or_model_side_effect(depth):
    a = agent(depth).eval()
    raw = torch.ones(128, 172)
    a._exp_buffer.set_data_flat("disc_obs_demo", raw)
    a._exp_buffer.set_data_flat("disc_obs", torch.zeros_like(raw))
    a._exp_buffer.set_data_flat("reward", raw.new_zeros(len(raw)))
    a._disc_buffer.push({"disc_obs_demo": raw[:16].unsqueeze(1),
                         "disc_obs": torch.zeros_like(raw[:16]).unsqueeze(1)})
    state = copy.deepcopy(a._model.state_dict())
    reward = a._calc_disc_rewards(raw)
    a._sample_count = a._normalizer_samples - 1
    a._compute_rewards()
    a._sample_count = a._normalizer_samples
    a._compute_rewards()
    torch.testing.assert_close(a._calc_disc_rewards(raw), reward, rtol=0, atol=0)
    for key, value in state.items():
        torch.testing.assert_close(a._model.state_dict()[key], value, rtol=0, atol=0)


@pytest.mark.parametrize("depth", [None, 0, 2, 8, 1.0, True])
def test_invalid_cpl_depth_fails_before_training(depth):
    cfg = config(1)["model"]
    cfg["disc_cpl_depth"] = depth
    with pytest.raises(ValueError, match="disc_cpl_depth"):
        DAREBodySplitCPLModel(cfg, Env())
