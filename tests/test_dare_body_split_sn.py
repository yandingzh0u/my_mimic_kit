"""Key contracts for the optional ordinary-SN Body-split variant."""

import copy
from pathlib import Path

import gymnasium.spaces as spaces
import pytest
import torch
import yaml

from envs.add_env import build_disc_error_groups
from learning.add_agent import calc_unscaled_disc_reward
from learning.dare_multi_model import DAREMultiModel
from learning.dare_body_split_sn_model import (
    DAREBodySplitSNModel, HIDDEN_WIDTHS, build_body_split_groups,
    hierarchical_parent_mean,
)
from learning.dare_body_split_sn_agent import DAREBodySplitSNAgent
from learning.agent_builder import build_agent


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "data/agents/dare_body_split_sn_humanoid_agent.yaml"


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


def config():
    return yaml.safe_load(CONFIG.read_text())


def agent():
    return DAREBodySplitSNAgent(config(), Env(), "cpu")


@pytest.mark.parametrize("frames", [1, 3])
def test_real_indices_cover_each_frame_and_keep_original_root_velocity(frames):
    env = Env(frames)
    groups = build_body_split_groups(env.get_disc_error_groups(), 172 * frames)
    assert [len(idx) for _, idx in groups] == [x * frames for x in (9, 45, 84, 34)]
    assert sorted(i for _, idx in groups for i in idx) == list(range(172 * frames))
    with pytest.raises(ValueError):
        build_body_split_groups(env.get_disc_error_groups()[:-1], 172 * frames)


def test_widths_are_341_512_not_divided_by_four_and_all_linear_layers_use_sn():
    model = DAREBodySplitSNModel(config()["model"], Env())
    assert model.get_disc_group_width() == 341
    assert model.get_disc_group_total_width() == 1364
    for dimension, branch in zip((9, 45, 84, 34), model._disc_branches.branches):
        linears = [m for m in branch.modules() if isinstance(m, torch.nn.Linear)]
        assert [(m.in_features, m.out_features) for m in linears] == [
            (dimension, 341), (341, 512), (512, 1)]
        assert all(hasattr(m, "parametrizations") and "weight" in m.parametrizations for m in linears)


def test_config_changes_only_agent_name_and_actor_critic_are_seed_identical():
    original = yaml.safe_load((ROOT / "data/agents/dare_multi_humanoid_agent.yaml").read_text())
    candidate = config()
    assert candidate.pop("agent_name") == "DAREBodySplitSN"
    original.pop("agent_name")
    assert candidate == original
    torch.manual_seed(11)
    baseline = DAREMultiModel(config()["model"], Env())
    torch.manual_seed(11)
    split = DAREBodySplitSNModel(config()["model"], Env())
    for a, b in zip(baseline.get_actor_params() + baseline.get_critic_params(),
                    split.get_actor_params() + split.get_critic_params()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


@pytest.mark.parametrize("batch_size", [0, 1, 3])
def test_mapped_hierarchical_reward_preserves_clipping_and_no_cross_group_gate(batch_size):
    a = agent()
    a._disc_eval_batch_size = batch_size
    a._model.eval_disc = lambda disc_obs: disc_obs
    logits = torch.tensor([[1., -8., 2., -1.], [1000., 1000., 1000., 1000.],
                           [-1000., -1000., -1000., -1000.]], requires_grad=True)
    actual = a._calc_disc_rewards(logits)
    expected = hierarchical_parent_mean(calc_unscaled_disc_reward(logits))
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (3,) and not actual.requires_grad
    assert not torch.allclose(actual, calc_unscaled_disc_reward(logits.mean(-1)))
    first = torch.tensor([[1., -2., -1., 0.]])
    second = first.clone()
    second[:, 2] = -30.
    increment = torch.tensor([[0., .1, 0., 0.]])
    first_delta = a._calc_disc_rewards(first + increment) - a._calc_disc_rewards(first)
    second_delta = a._calc_disc_rewards(second + increment) - a._calc_disc_rewards(second)
    torch.testing.assert_close(first_delta, second_delta, atol=1e-7, rtol=2e-5)


def fill_buffers(a, raw):
    a._exp_buffer.set_data_flat("disc_obs_demo", raw)
    a._exp_buffer.set_data_flat("disc_obs", torch.zeros_like(raw))
    a._exp_buffer.set_data_flat("reward", raw.new_zeros(len(raw)))
    a._disc_buffer.push({"disc_obs_demo": raw[:16].unsqueeze(1),
                         "disc_obs": torch.zeros_like(raw[:16]).unsqueeze(1)})


def test_full_cpu_agent_builder_loss_optimizer_reward_and_metadata_smoke():
    a = build_agent(str(CONFIG), Env(), "cpu")
    raw = torch.randn(128, 172)
    fill_buffers(a, raw)
    batch = {"disc_obs_demo": raw[:16], "disc_obs": torch.zeros_like(raw[:16])}
    a.train()
    losses = a._compute_disc_loss(batch)
    assert all(torch.isfinite(value).all() for value in losses.values())
    a._disc_optimizer.step(losses["disc_loss"])
    assert a._disc_optimizer.get_steps() == 1
    for branch in a._model._disc_branches.branches:
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in branch.parameters())
    a.eval()
    assert a._calc_disc_rewards(raw).shape == (128,)
    saved = copy.deepcopy(a._get_checkpoint_extra_state())
    assert "body_split_sn" in saved and "multi_disc" not in saved
    restored = agent()
    restored.load_state_dict(copy.deepcopy(a.state_dict()), strict=True)
    restored._load_checkpoint_extra_state(saved)
    restored._disc_optimizer.load_state_dict(copy.deepcopy(a._disc_optimizer.state_dict()))
    restored.eval()
    torch.testing.assert_close(restored._calc_disc_rewards(raw), a._calc_disc_rewards(raw), rtol=0, atol=0)
    invalid = copy.deepcopy(saved)
    invalid["body_split_sn"]["reward_aggregation"] = "mapped_mean_logit"
    with pytest.raises(ValueError, match="different Body-split"):
        restored._load_checkpoint_extra_state(invalid)
    with pytest.raises(ValueError, match="different Body-split"):
        restored._load_checkpoint_extra_state({"multi_disc": {}})


def test_normalizer_completion_does_not_change_model_logits_or_reward():
    a = agent().eval()
    raw = torch.ones(128, 172)
    fill_buffers(a, raw)
    state_before = copy.deepcopy(a._model.state_dict())
    a._sample_count = a._normalizer_samples - 1
    first = a._calc_disc_rewards(raw)
    a._compute_rewards()
    a._sample_count = a._normalizer_samples
    a._compute_rewards()
    second = a._calc_disc_rewards(raw)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    for key, value in state_before.items():
        torch.testing.assert_close(a._model.state_dict()[key], value, rtol=0, atol=0)
    assert not hasattr(a, "_disc_reward_scale")
    assert not hasattr(a, "_calibrate_disc_logit_scale")
    assert not hasattr(a._model, "get_disc_logit_scale")
