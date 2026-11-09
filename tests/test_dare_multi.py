"""Behavioral checks for the three-objective, independent SN discriminator."""

import copy
from pathlib import Path

import gymnasium.spaces as spaces
import pytest
import torch
import torch.nn.functional as F
import yaml

from envs.add_env import build_disc_error_groups
from learning.add_agent import calc_unscaled_disc_reward
from learning.dare_agent import DAREAgent, anchor_gap
from learning.dare_model import ANCHOR_GAP_TARGET, DAREModel
from learning.dare_multi_agent import DAREMultiAgent
from learning.dare_multi_model import DAREMultiModel, build_multi_disc_groups
from learning.diff_normalizer import DiffNormalizer


ROOT = Path(__file__).resolve().parents[1]


class _Env:
    def __init__(self, steps=1):
        self.steps = steps

    def get_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"), shape=(10,))

    def get_action_space(self):
        return spaces.Box(-1., 1., shape=(4,))

    def get_disc_obs_space(self):
        return spaces.Box(-float("inf"), float("inf"),
                          shape=(172 * self.steps,))

    def get_disc_error_groups(self):
        return build_disc_error_groups(self.steps, 15, 15, 28,
                                       172 * self.steps)


def _config():
    return {
        "actor_net": "fc_2layers_128units",
        "actor_init_output_scale": .01,
        "actor_std_type": "FIXED",
        "action_std": .05,
        "critic_net": "fc_2layers_128units",
        "disc_net": "fc_2layers_128units",
    }


@pytest.mark.parametrize("steps", [1, 3])
def test_three_groups_cover_actual_error_layout_across_frames(steps):
    env = _Env(steps)
    groups = build_multi_disc_groups(env.get_disc_error_groups(), 172 * steps)
    assert tuple(name for name, _ in groups) == (
        "root_pose", "body_pose", "velocity")
    assert [len(indices) for _, indices in groups] == [
        9 * steps, 129 * steps, 34 * steps]
    assert sorted(index for _, indices in groups for index in indices) == list(
        range(172 * steps))
    coordinate_sets = (
        set(range(0, 9)), set(range(9, 138)), set(range(138, 172)))
    for (_, indices), coordinates in zip(groups, coordinate_sets):
        assert set(indices) == {
            frame * 172 + coordinate
            for frame in range(steps) for coordinate in coordinates}


@pytest.mark.parametrize("kind", ["missing", "duplicate", "overlap", "range"])
def test_invalid_source_group_schema_is_rejected(kind):
    groups = list(_Env().get_disc_error_groups())
    if kind == "missing":
        groups.pop()
    elif kind == "duplicate":
        groups.append(groups[0])
    elif kind == "overlap":
        name, indices = groups[1]
        groups[1] = (name, (0,) + indices[1:])
    else:
        name, indices = groups[-1]
        groups[-1] = (name, indices[:-1] + (172,))
    with pytest.raises(ValueError):
        build_multi_disc_groups(groups, 172)


def test_three_independent_branches_are_sn_without_cpl_or_shared_parameters():
    model = DAREMultiModel(_config(), _Env()).eval()
    branches = model._disc_branches.branches
    assert len(branches) == 3
    parameter_ids = [set(map(id, branch.parameters())) for branch in branches]
    assert all(not parameter_ids[i] & parameter_ids[j]
               for i in range(3) for j in range(i))
    for branch, dimension in zip(branches, (9, 129, 34)):
        linears = [layer for layer in branch.modules()
                   if isinstance(layer, torch.nn.Linear)]
        assert len(linears) == 3
        assert linears[0].in_features == dimension
        assert linears[-1].out_features == 1
        assert all(hasattr(layer.parametrizations, "weight")
                   for layer in linears)
    assert all("ConvexPotential" not in type(child).__name__
               for child in model.modules())
    disc_ids = set(map(id, model.get_disc_params()))
    assert disc_ids == set.union(*parameter_ids)
    assert model.get_disc_logit_weights().numel() == 3 * 64


@pytest.mark.parametrize("steps", [1, 3])
def test_each_logit_is_unchanged_when_only_other_error_groups_change(steps):
    torch.manual_seed(29)
    model = DAREMultiModel(_config(), _Env(steps)).eval()
    data = torch.randn(6, 172 * steps)
    original = model.eval_disc_raw(data)
    assert original.shape == (6, 3)
    for branch_id, (_, indices) in enumerate(model.get_disc_error_groups()):
        changed = torch.randn_like(data) * 20
        changed[:, list(indices)] = data[:, list(indices)]
        torch.testing.assert_close(model.eval_disc_raw(changed)[:, branch_id],
                                   original[:, branch_id], rtol=0, atol=0)
        local = data.clone().requires_grad_()
        model.eval_disc_raw(local)[:, branch_id].sum().backward()
        excluded = sorted(set(range(local.shape[-1])) - set(indices))
        assert torch.count_nonzero(local.grad[:, excluded]) == 0


def test_actor_and_critic_initialize_identically_to_existing_dare_at_same_seed():
    torch.manual_seed(317)
    baseline = DAREModel(_config(), _Env())
    torch.manual_seed(317)
    candidate = DAREMultiModel(_config(), _Env())
    for left, right in (
            (baseline.get_actor_params(), candidate.get_actor_params()),
            (baseline.get_critic_params(), candidate.get_critic_params())):
        assert len(left) == len(right)
        for source, actual in zip(left, right):
            torch.testing.assert_close(source, actual, rtol=0, atol=0)


def test_training_mode_sn_forward_backward_is_finite_for_every_branch():
    model = DAREMultiModel(_config(), _Env()).train()
    positive = model.eval_disc(torch.zeros(1, 172))
    negative = model.eval_disc(torch.randn(32, 172))
    loss = .5 * (F.softplus(-positive).mean(0)
                 + F.softplus(negative).mean(0)).sum()
    loss.backward()
    assert torch.isfinite(loss)
    for branch in model._disc_branches.branches:
        gradients = [parameter.grad for parameter in branch.parameters()]
        assert gradients and all(gradient is not None for gradient in gradients)
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)


class _AnalyticModel(torch.nn.Module):
    """Three independent linear logits; no spectral-state side effects."""

    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([.3, -.4, .2]))
        self.bias = torch.nn.Parameter(torch.tensor([.1, -.2, .4]))
        self.calls = []
        self.register_buffer("scale", torch.ones(()))
        self.register_buffer("calibrated", torch.zeros((), dtype=torch.bool))

    def get_disc_error_groups(self):
        return (("root_pose", (0,)), ("body_pose", (1,)),
                ("velocity", (2,)))

    def eval_disc_raw(self, disc_obs):
        self.calls.append(disc_obs.detach().clone())
        return disc_obs * self.weight + self.bias

    def eval_disc(self, disc_obs):
        return self.scale * self.eval_disc_raw(disc_obs)

    def get_disc_logit_weights(self):
        return self.weight

    def get_disc_group_width(self):
        return self.weight.new_tensor(42.)

    def get_disc_group_total_width(self):
        return self.weight.new_tensor(126.)

    def get_disc_logit_scale(self):
        return self.scale.clone()

    def is_disc_logit_calibrated(self):
        return bool(self.calibrated)

    def set_disc_logit_scale(self, value):
        self.scale.fill_(value)
        self.calibrated.fill_(True)


class _Replay:
    def __init__(self, values):
        self.values = values

    def sample(self, count):
        assert count == len(self.values)
        return {"disc_obs_demo": self.values,
                "disc_obs": torch.zeros_like(self.values)}

    def get_sample_count(self):
        return len(self.values)

    def get_data_flat(self, name):
        if name == "disc_obs_demo":
            return self.values
        if name == "disc_obs":
            return torch.zeros_like(self.values)
        raise KeyError(name)


def _agent(current=None, replay=None):
    agent = object.__new__(DAREMultiAgent)
    torch.nn.Module.__init__(agent)
    agent._device = "cpu"
    agent._model = _AnalyticModel()
    agent._pos_diff = torch.zeros(3)
    agent._disc_obs_norm = DiffNormalizer((3,), "cpu")
    if current is None:
        current = torch.tensor([[1., 2., 3.], [2., 3., 4.]])
    if replay is None:
        replay = torch.tensor([[4., 3., 2.], [3., 2., 1.]])
    agent._disc_buffer = _Replay(replay)
    agent._exp_buffer = _Replay(current)
    agent._disc_logit_reg = 0.
    agent._disc_eval_batch_size = 0
    agent._disc_reward_scale = 2.
    agent._disc_group_negatives = False
    agent._disc_group_negative_sampler = None
    agent._phase_coverage_reset = False
    agent._iter = 386
    agent._calibration_gap_raw = float("nan")
    agent._calibration_gaps_raw = None
    batch = {"disc_obs_demo": current,
             "disc_obs": torch.zeros_like(current)}
    return agent, batch


def test_independent_bce_sum_preserves_each_branch_gradient_and_real_samples():
    agent, batch = _agent()
    before = copy.deepcopy(agent._disc_obs_norm.state_dict())
    data = torch.cat((batch["disc_obs_demo"], agent._disc_buffer.values))
    info = agent._compute_disc_loss(batch)
    assert len(agent._model.calls) == 2
    torch.testing.assert_close(agent._model.calls[0], torch.zeros(1, 3))
    torch.testing.assert_close(agent._model.calls[1], data)
    info["disc_loss"].backward()
    actual_weight = agent._model.weight.grad.clone()
    actual_bias = agent._model.bias.grad.clone()
    individual_losses = []
    for group_id in range(3):
        w = agent._model.weight[group_id].detach().clone().requires_grad_()
        b = agent._model.bias[group_id].detach().clone().requires_grad_()
        loss = .5 * (F.softplus(-b)
                     + F.softplus(data[:, group_id] * w + b).mean())
        loss.backward()
        individual_losses.append(loss.detach())
        torch.testing.assert_close(actual_weight[group_id], w.grad)
        torch.testing.assert_close(actual_bias[group_id], b.grad)
    torch.testing.assert_close(info["disc_cls_loss"],
                               torch.stack(individual_losses).mean())
    for name, value in before.items():
        torch.testing.assert_close(agent._disc_obs_norm.state_dict()[name], value)


def test_zero_local_errors_are_excluded_without_discarding_other_groups():
    current = torch.tensor([[0., 2., 3.], [0., 0., 4.]])
    agent, batch = _agent(current, current.clone())
    info = agent._compute_disc_loss(batch)
    info["disc_loss"].backward()
    # Root has no negative samples: only .5 * positive BCE remains.
    expected_root_bias_grad = -.5 * torch.sigmoid(-agent._model.bias[0])
    torch.testing.assert_close(agent._model.bias.grad[0], expected_root_bias_grad)
    assert agent._model.weight.grad[0] == 0
    assert agent._model.weight.grad[1] != 0
    assert agent._model.weight.grad[2] != 0
    assert all(torch.isfinite(value).all() for value in info.values())


def test_reward_maps_and_clips_each_head_before_averaging():
    agent, _ = _agent()
    values = torch.tensor([[-1000., 0., 1000.], [-1., 2., -3.]],
                          requires_grad=True)
    agent._model.eval_disc = lambda disc_obs: disc_obs
    actual = agent._calc_disc_rewards(values)
    expected = 2 * calc_unscaled_disc_reward(values.detach()).mean(-1)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2,) and not actual.requires_grad
    assert torch.isfinite(actual).all()
    assert not torch.allclose(actual,
                              2 * calc_unscaled_disc_reward(values.mean(-1)))


def test_public_calibration_uses_one_scale_from_mean_of_three_anchor_gaps():
    agent, _ = _agent()
    with torch.no_grad():
        agent._model.weight.copy_(torch.tensor([-.1, -.4, -.8]))
    before = anchor_gap(agent._model, agent._disc_obs_norm,
                        agent._pos_diff,
                        agent._exp_buffer.values,
                        agent._disc_buffer.values, 128)
    assert before > 0
    agent._calibrate_disc_logit_scale()
    torch.testing.assert_close(agent._model.scale,
                               torch.tensor(ANCHOR_GAP_TARGET / before))
    assert agent._model.is_disc_logit_calibrated()
    assert agent._model.training
    positive = agent._model.eval_disc(torch.zeros(1, 3)).mean()
    negative = .5 * (
        agent._model.eval_disc(agent._exp_buffer.values).mean()
        + agent._model.eval_disc(agent._disc_buffer.values).mean())
    torch.testing.assert_close(positive - negative,
                               torch.tensor(ANCHOR_GAP_TARGET))


def test_multi_model_checkpoint_round_trip_retains_logits_and_single_scale():
    source = DAREMultiModel(_config(), _Env()).eval()
    source.set_disc_logit_scale(4.25)
    restored = DAREMultiModel(_config(), _Env()).eval()
    restored.load_state_dict(copy.deepcopy(source.state_dict()), strict=True)
    data = torch.randn(7, 172)
    torch.testing.assert_close(restored.eval_disc(data), source.eval_disc(data),
                               rtol=0, atol=0)
    assert restored.get_disc_logit_scale().ndim == 0
    assert restored.is_disc_logit_calibrated()
    baseline = DAREModel(_config(), _Env())
    with pytest.raises(RuntimeError):
        restored.load_state_dict(baseline.state_dict(), strict=True)


def test_model_checkpoint_rejects_same_shape_but_different_group_indices():
    model = DAREMultiModel(_config(), _Env()).eval()
    saved = copy.deepcopy(model.state_dict())
    indices = saved["_disc_branches.group_indices_0"]
    indices[0], indices[1] = indices[1].clone(), indices[0].clone()
    with pytest.raises(RuntimeError, match="group schema differs"):
        model.load_state_dict(saved, strict=True)


def test_agent_resume_retains_calibration_and_rejects_different_semantics():
    source, _ = _agent()
    source._calibration_gap_raw = .75
    source._calibration_gaps_raw = [.5, .75, 1.]
    saved = source._get_checkpoint_extra_state()
    restored, _ = _agent()
    restored._load_checkpoint_extra_state(copy.deepcopy(saved))
    assert restored._calibration_gap_raw == .75
    assert restored._calibration_gaps_raw == [.5, .75, 1.]
    for key, value in (
            ("version", -1),
            ("reward_aggregation", "mean_of_logits"),
            ("loss_aggregation", "mean_of_independent_losses"),
            ("groups", (("root_pose", (1,)), ("body_pose", (0,)),
                        ("velocity", (2,))))):
        invalid = copy.deepcopy(saved)
        invalid["multi_disc"][key] = value
        with pytest.raises(ValueError, match="different multi-discriminator semantics"):
            restored._load_checkpoint_extra_state(invalid)
    with pytest.raises(ValueError, match="different multi-discriminator semantics"):
        restored._load_checkpoint_extra_state({})


def test_multi_agent_rejects_structured_negative_mode():
    agent = object.__new__(DAREMultiAgent)
    with pytest.raises(ValueError, match="real group errors"):
        agent._load_params({"disc_group_negatives": True})


def test_failed_calibration_keeps_scale_and_restores_training_mode():
    agent, _ = _agent()
    with torch.no_grad():
        agent._model.weight.fill_(1.)
    with pytest.raises(RuntimeError, match="positive mean separation"):
        agent._calibrate_disc_logit_scale()
    assert agent._model.training
    assert not agent._model.is_disc_logit_calibrated()
    assert agent._model.scale == 1


def test_multi_agent_config_only_switches_agent_and_preserves_training_defaults():
    baseline = yaml.safe_load(
        (ROOT / "data/agents/dare_humanoid_agent.yaml").read_text())
    candidate = yaml.safe_load(
        (ROOT / "data/agents/dare_multi_humanoid_agent.yaml").read_text())
    assert candidate["agent_name"] == "DAREMulti"
    candidate["agent_name"] = baseline["agent_name"]
    # An explicit false is allowed to document that synthetic negatives are off.
    assert candidate.pop("disc_group_negatives", False) is False
    assert candidate == baseline
    assert "disc_group_negatives" not in baseline


def test_original_dare_class_keeps_scalar_logit_api():
    model = DAREModel(_config(), _Env()).eval()
    assert model.eval_disc(torch.randn(5, 172)).shape == (5, 1)
    assert DAREMultiAgent is not DAREAgent
