"""Behavioral checks for the changed discriminator training distribution."""

import copy
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F
import yaml

from envs.add_env import build_disc_error_groups
from learning.dare_agent import DAREAgent
from learning.diff_normalizer import DiffNormalizer
from learning.group_negatives import GroupNegativeSampler


GROUPS = (("root", (0, 2)), ("velocity", (1, 3)))


def sampler(seed=0):
    return GroupNegativeSampler(GROUPS, 4, "cpu", seed)


def test_half_of_each_source_is_masked_without_mutating_real_data():
    data = torch.arange(1, 41 * 4 + 1).reshape(41, 4).float()
    before = data.clone()
    result = sampler().sample(data, (20, 21))
    assert result.values.shape == data.shape
    assert (result.group_ids[:20] >= 0).sum() == 10
    assert (result.group_ids[20:] >= 0).sum() == 10
    assert result.valid.all()
    torch.testing.assert_close(data, before, rtol=0, atol=0)
    for row, group_id in enumerate(result.group_ids.tolist()):
        if group_id == -1:
            torch.testing.assert_close(result.values[row], data[row])
        else:
            indices = GROUPS[group_id][1]
            expected = torch.zeros(4)
            expected[list(indices)] = data[row, list(indices)]
            torch.testing.assert_close(result.values[row], expected)


def test_existing_noncontiguous_seven_groups_are_all_sampled():
    groups = build_disc_error_groups(3, 15, 15, 28, 3 * 172)
    draw = GroupNegativeSampler(groups, 3 * 172, "cpu", 5)
    data = torch.ones(1400, 3 * 172)
    result = draw.sample(data, (700, 700))
    counts = torch.bincount(result.group_ids[result.group_ids >= 0], minlength=7)
    assert counts.sum() == 700
    assert counts.min() > 60 and counts.max() < 140
    for group_id, (_, indices) in enumerate(groups):
        rows = result.values[result.group_ids == group_id]
        assert torch.all(rows[:, list(indices)] == 1)
        assert torch.all(rows.sum(-1) == len(indices))


def test_zero_group_falls_back_and_zero_real_row_is_not_negative():
    # Repeated real rows have only the root group nonzero; velocity requests
    # must stay real rather than turning into contradictory zero negatives.
    data = torch.tensor([[1., 0., 2., 0.]]).repeat(128, 1)
    data[:16] = 0
    result = sampler().sample(data, (64, 64))
    zero_requested = result.requested_group_ids == 1
    assert zero_requested.any()
    assert torch.all(result.group_ids[zero_requested] == -1)
    torch.testing.assert_close(result.values[zero_requested], data[zero_requested])
    assert not result.valid[:16].any()
    assert result.valid[16:].all()
    assert torch.all(result.values[result.group_ids >= 0].ne(0).any(-1))


def test_rng_stream_is_separate_and_checkpoint_restores_exact_next_batch():
    torch.manual_seed(917)
    global_before = torch.get_rng_state().clone()
    draw = sampler(23)
    data = torch.ones(100, 4)
    draw.sample(data, (50, 50))
    state = draw.state_dict()
    expected = draw.sample(data, (50, 50))
    restored = sampler(99)
    restored.load_state_dict(state)
    actual = restored.sample(data, (50, 50))
    assert torch.equal(torch.get_rng_state(), global_before)
    for field in ("values", "group_ids", "requested_group_ids", "valid"):
        assert torch.equal(getattr(actual, field), getattr(expected, field))


@pytest.mark.parametrize("groups", [
    (("a", (0, 1)), ("b", (1, 2, 3))),
    (("a", (0, 1)), ("b", (2,))),
    (("a", (0, 1)), ("b", (2, 4))),
    (("a", (0, 1)), ("a", (2, 3))),
])
def test_bad_group_layout_is_rejected(groups):
    with pytest.raises(ValueError):
        GroupNegativeSampler(groups, 4, "cpu", 0)


def test_shape_source_and_nonfinite_errors_are_not_hidden_by_masking():
    draw = sampler()
    with pytest.raises(ValueError, match="shape"):
        draw.sample(torch.ones(4, 3), (2, 2))
    with pytest.raises(ValueError, match="source sizes"):
        draw.sample(torch.ones(4, 4), ())
    with pytest.raises(ValueError, match="source sizes"):
        draw.sample(torch.ones(4, 4), (1, 2))
    for value in (float("nan"), float("inf")):
        data = torch.ones(4, 4)
        data[0, 0] = value
        with pytest.raises(ValueError, match="finite"):
            draw.sample(data, (2, 2))


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.2, -.3, .4, -.5]))
        self.bias = torch.nn.Parameter(torch.tensor(.1))
        self.calls = []

    def eval_disc(self, disc_obs):
        data = disc_obs
        self.calls.append(data.detach().clone())
        return (data @ self.weight + self.bias).unsqueeze(-1)

    def get_disc_logit_weights(self):
        return self.weight

    def get_disc_group_width(self):
        return self.bias.new_tensor(146.)

    def get_disc_group_total_width(self):
        return self.bias.new_tensor(1022.)

    def get_disc_logit_scale(self):
        return self.bias.new_tensor(1.)


class _Replay:
    def __init__(self, data):
        self.data = data

    def sample(self, count):
        assert count == len(self.data)
        return {"disc_obs_demo": self.data, "disc_obs": torch.zeros_like(self.data)}


def agent_and_batch(enabled=True, zeros=False):
    agent = object.__new__(DAREAgent)
    torch.nn.Module.__init__(agent)
    agent._device = "cpu"
    agent._model = _Model()
    agent._pos_diff = torch.zeros(4)
    agent._disc_obs_norm = DiffNormalizer((4,), "cpu")
    current = torch.zeros(20, 4) if zeros else torch.arange(1, 81).reshape(20, 4).float()
    replay = current + (0 if zeros else 1)
    agent._disc_buffer = _Replay(replay)
    agent._disc_logit_reg = .01
    agent._disc_eval_batch_size = 0
    agent._disc_reward_scale = 2.
    agent._disc_group_negatives = enabled
    agent._disc_group_negative_sampler = sampler() if enabled else None
    agent._phase_coverage_reset = False
    batch = {"disc_obs_demo": current, "disc_obs": torch.zeros_like(current)}
    return agent, batch


def test_loss_has_two_forwards_and_preserves_stats_replay_and_reward_path():
    agent, batch = agent_and_batch()
    norm_before = copy.deepcopy(agent._disc_obs_norm.state_dict())
    real_before = batch["disc_obs_demo"].clone()
    replay_before = agent._disc_buffer.data.clone()
    info = agent._compute_disc_loss(batch)
    assert len(agent._model.calls) == 2
    assert agent._model.calls[0].shape == (1, 4)
    assert agent._model.calls[1].shape == (40, 4)
    assert info["disc_group_negative_fraction"] == .5
    assert info["disc_group_negative_requested_fraction"] == .5
    assert info["disc_neg_valid_fraction"] == 1
    mixed = agent._model.calls[1]
    expected = .5 * (F.softplus(-agent._model.bias)
                      + F.softplus(mixed @ agent._model.weight + agent._model.bias).mean())
    torch.testing.assert_close(info["disc_cls_loss"], expected)
    info["disc_loss"].backward()
    assert torch.isfinite(agent._model.weight.grad).all()
    assert all(torch.isfinite(value).all() for value in info.values())
    torch.testing.assert_close(batch["disc_obs_demo"], real_before)
    torch.testing.assert_close(agent._disc_buffer.data, replay_before)
    for key, value in norm_before.items():
        torch.testing.assert_close(agent._disc_obs_norm.state_dict()[key], value)
    agent._calc_disc_rewards(real_before)
    torch.testing.assert_close(agent._model.calls[-1], real_before)


def test_all_zero_negatives_have_no_contradictory_negative_gradient():
    agent, batch = agent_and_batch(zeros=True)
    info = agent._compute_disc_loss(batch)
    assert info["disc_neg_valid_fraction"] == 0
    assert info["disc_group_negative_fraction"] == 0
    expected = .5 * F.softplus(-agent._model.bias)
    torch.testing.assert_close(info["disc_cls_loss"], expected)
    info["disc_loss"].backward()
    assert agent._model.bias.grad < 0
    assert all(torch.isfinite(value).all() for value in info.values())


def test_disabled_loss_keeps_real_batch_and_global_rng_unchanged():
    agent, batch = agent_and_batch(enabled=False)
    before = torch.get_rng_state().clone()
    info = agent._compute_disc_loss(batch)
    expected = torch.cat((batch["disc_obs_demo"], agent._disc_buffer.data))
    assert len(agent._model.calls) == 2
    torch.testing.assert_close(agent._model.calls[1], expected)
    assert torch.equal(before, torch.get_rng_state())
    assert not any(key.startswith("disc_group_negative_") for key in info)
    assert agent._get_checkpoint_extra_state() == {}


def test_agent_extra_state_round_trip_and_mode_mismatch_rejection():
    source, _ = agent_and_batch()
    data = torch.ones(40, 4)
    source._disc_group_negative_sampler.sample(data, (20, 20))
    saved = source._get_checkpoint_extra_state()
    restored, _ = agent_and_batch()
    restored._load_checkpoint_extra_state(saved)
    expected = source._disc_group_negative_sampler.sample(data, (20, 20))
    actual = restored._disc_group_negative_sampler.sample(data, (20, 20))
    assert torch.equal(actual.values, expected.values)
    assert torch.equal(actual.group_ids, expected.group_ids)
    with pytest.raises(ValueError, match="different structured-negative mode"):
        restored._load_checkpoint_extra_state({})
    disabled, _ = agent_and_batch(enabled=False)
    with pytest.raises(ValueError, match="different structured-negative mode"):
        disabled._load_checkpoint_extra_state(saved)
    disabled._load_checkpoint_extra_state({})


def test_enabled_config_diff_is_only_the_boolean_switch():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load((root / "data/agents/dare_humanoid_agent.yaml").read_text())
    candidate = yaml.safe_load((root / "data/agents/dare_group_negatives_humanoid_agent.yaml").read_text())
    assert candidate.pop("disc_group_negatives") is True
    assert candidate == baseline


@pytest.mark.parametrize("value", ["true", 1, None])
def test_switch_rejects_non_boolean_values(value):
    agent = object.__new__(DAREAgent)
    with pytest.raises(ValueError, match="must be a boolean"):
        agent._load_params({"disc_group_negatives": value})
