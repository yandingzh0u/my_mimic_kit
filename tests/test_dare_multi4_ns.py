"""Behavioral and algebraic checks for four independent NS-maintained critics."""

import copy
from pathlib import Path

import gymnasium.spaces as spaces
import pytest
import torch
import torch.nn.functional as F
import yaml

from envs.add_env import build_disc_error_groups
from learning.add_agent import calc_unscaled_disc_reward
from learning.dare_model import DAREModel
from learning.dare_multi4_ns_agent import DAREMulti4NSAgent
from learning.dare_multi4_ns_model import (
    DAREMulti4NSModel, build_multi4_disc_groups, newton_schulz_column_step_,
)
from learning.diff_normalizer import DiffNormalizer


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
        "disc_net": "fc_2layers_1024units",
    }


def _bare_agent(model=None):
    agent = object.__new__(DAREMulti4NSAgent)
    torch.nn.Module.__init__(agent)
    agent._device = "cpu"
    agent._model = model if model is not None else DAREMulti4NSModel(
        _config(), _Env()).eval()
    agent._disc_group_negative_sampler = None
    agent._phase_coverage_reset = False
    return agent


def _build_optimizers(agent):
    optimizer = {"type": "SGD", "learning_rate": 2.5e-4,
                 "weight_decay": .0001}
    agent._build_optimizer({
        "actor_optimizer": dict(optimizer),
        "critic_optimizer": dict(optimizer),
        "disc_optimizer": dict(optimizer),
    })


def _hidden_layers(model):
    return [branch[index]
            for branch in model._disc_branches.branches
            for index in (0, 2)]


@pytest.mark.parametrize("steps", [1, 3])
def test_four_groups_follow_real_add_layout_including_all_frames(steps):
    groups = build_multi4_disc_groups(
        _Env(steps).get_disc_error_groups(), 172 * steps)
    assert [len(indices) for _, indices in groups] == [
        15 * steps, 45 * steps, 84 * steps, 28 * steps]
    # Root includes noncontiguous velocity coordinates in each actual frame.
    coordinates = (
        set(range(9)) | set(range(138, 144)),
        set(range(93, 138)),
        set(range(9, 93)),
        set(range(144, 172)),
    )
    for (_, indices), expected in zip(groups, coordinates):
        assert set(indices) == {
            frame * 172 + coordinate
            for frame in range(steps) for coordinate in expected}
    assert sorted(i for _, indices in groups for i in indices) == list(
        range(172 * steps))


@pytest.mark.parametrize("kind", ["missing", "duplicate", "overlap", "range"])
def test_invalid_source_group_partition_is_rejected(kind):
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
        build_multi4_disc_groups(groups, 172)


def test_wide_input_that_cannot_be_column_orthogonal_is_rejected():
    with pytest.raises(ValueError):
        DAREMulti4NSModel(_config(), _Env(4))


def test_hidden_linears_are_dense_column_orthogonal_and_only_head_has_sn():
    torch.manual_seed(511)
    model = DAREMulti4NSModel(_config(), _Env()).eval()
    branches = model._disc_branches.branches
    assert len(branches) == 4
    owned = [set(map(id, branch.parameters())) for branch in branches]
    assert all(not owned[i] & owned[j] for i in range(4) for j in range(i))
    assert set(map(id, model.get_disc_params())) == set.union(*owned)
    for branch, dimension in zip(branches, (15, 45, 84, 28)):
        linears = [layer for layer in branch if isinstance(layer, torch.nn.Linear)]
        assert [(layer.in_features, layer.out_features) for layer in linears] == [
            (dimension, 256), (256, 512), (512, 1)]
        assert isinstance(branch[1], torch.nn.ReLU)
        assert isinstance(branch[3], torch.nn.ReLU)
        assert hasattr(linears[-1], "parametrizations")
        assert hasattr(linears[-1].parametrizations, "weight")
        for layer in linears[:-1]:
            assert not hasattr(layer, "parametrizations")
            weight = layer.weight.detach()
            torch.testing.assert_close(weight.T @ weight,
                                       torch.eye(weight.shape[1]),
                                       atol=1e-6, rtol=1e-5)
            assert torch.count_nonzero(weight) > .99 * weight.numel()
            assert torch.count_nonzero(layer.bias) == 0
    assert int(model.get_ns_update_count()) == 0


@pytest.mark.parametrize("steps", [1, 3])
def test_logits_and_input_gradients_are_independent_between_error_groups(steps):
    torch.manual_seed(29)
    model = DAREMulti4NSModel(_config(), _Env(steps)).eval()
    data = torch.randn(6, 172 * steps)
    original = model.eval_disc_raw(data)
    assert original.shape == (6, 4)
    for group_id, (_, indices) in enumerate(model.get_disc_error_groups()):
        changed = torch.randn_like(data) * 20
        changed[:, list(indices)] = data[:, list(indices)]
        torch.testing.assert_close(model.eval_disc_raw(changed)[:, group_id],
                                   original[:, group_id], rtol=0, atol=0)
        local = data.clone().requires_grad_()
        model.eval_disc_raw(local)[:, group_id].sum().backward()
        excluded = sorted(set(range(local.shape[-1])) - set(indices))
        assert torch.count_nonzero(local.grad[:, excluded]) == 0


def test_actor_and_critic_are_identical_to_existing_dare_at_same_seed():
    torch.manual_seed(317)
    baseline = DAREModel(_config(), _Env())
    torch.manual_seed(317)
    candidate = DAREMulti4NSModel(_config(), _Env())
    for left, right in (
            (baseline.get_actor_params(), candidate.get_actor_params()),
            (baseline.get_critic_params(), candidate.get_critic_params())):
        for source, actual in zip(left, right):
            torch.testing.assert_close(source, actual, rtol=0, atol=0)
        assert len(left) == len(right)


def test_ns_reduces_known_singular_value_errors_but_cannot_restore_a_null_column():
    model = DAREMulti4NSModel(_config(), _Env()).eval()
    layer = _hidden_layers(model)[0]
    singular_values = torch.ones(layer.in_features)
    singular_values[:5] = torch.tensor([0., .1, .8, 1., 1.1])
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[:layer.in_features].copy_(torch.diag(singular_values))
        layer.bias.uniform_(-.2, .2)
    saved_biases = [item.bias.detach().clone() for item in _hidden_layers(model)]
    expected = .5 * singular_values * (3 - singular_values.square())
    before_error = (singular_values.square() - 1).abs()
    model.maintain_hidden_orthogonality_()
    actual = layer.weight[:layer.in_features].diag().detach()
    torch.testing.assert_close(actual, expected)
    after_error = (actual.square() - 1).abs()
    assert torch.all(after_error[1:] <= before_error[1:])
    assert torch.all(after_error[[1, 2, 4]] < before_error[[1, 2, 4]])
    assert torch.count_nonzero(layer.weight[:, 0]) == 0
    assert int(model.get_ns_update_count()) == 1
    assert all(parameter.grad is None for parameter in model.parameters())
    for original, item in zip(saved_biases, _hidden_layers(model)):
        torch.testing.assert_close(item.bias, original, rtol=0, atol=0)


def test_ns_arithmetic_is_fp32_even_under_outer_autocast():
    torch.manual_seed(928)
    weight = torch.randn(25, 8) * .1
    expected = 1.5 * weight - .5 * weight @ (weight.T @ weight)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        newton_schulz_column_step_(weight)
    torch.testing.assert_close(weight, expected, rtol=0, atol=0)


def test_training_mode_multiple_forwards_backpropagate_finite_gradients_in_all_branches():
    model = DAREMulti4NSModel(_config(), _Env()).train()
    loss = _classification_loss(model, torch.randn(32, 172))
    loss.backward()
    assert torch.isfinite(loss)
    for branch in model._disc_branches.branches:
        for parameter in branch.parameters():
            assert parameter.grad is not None
            assert torch.isfinite(parameter.grad).all()
        assert any(torch.count_nonzero(parameter.grad)
                   for parameter in branch.parameters())


def _classification_loss(model, data):
    return .5 * (F.softplus(-model.eval_disc(torch.zeros(1, 172))).mean(0)
                 + F.softplus(model.eval_disc(data)).mean(0)).sum()


def test_real_sgd_step_runs_exactly_one_ns_update_and_leaves_actor_critic_untouched():
    torch.manual_seed(983)
    agent = _bare_agent()
    _build_optimizers(agent)
    agent._install_disc_ns_hook()  # Reinstallation must not double the update.
    reference = copy.deepcopy(agent._model).eval()
    optimizer = torch.optim.SGD(reference.get_disc_params(), lr=2.5e-4,
                                momentum=.9, weight_decay=.0001)
    untouched = [parameter.detach().clone() for parameter in (
        agent._model.get_actor_params() + agent._model.get_critic_params())]
    data = torch.randn(16, 172)
    optimizer.zero_grad()
    _classification_loss(reference, data).backward()
    optimizer.step()
    expected = [layer.weight.detach().clone() for layer in _hidden_layers(reference)]
    # Independent matrix formula for the raw SGD-updated weights.
    expected = [1.5 * w - .5 * (w @ w.T) @ w for w in expected]
    agent._disc_optimizer.step(_classification_loss(agent._model, data))
    assert agent._disc_optimizer.get_steps() == 1
    assert int(agent._model.get_ns_update_count()) == 1
    for actual, target in zip(_hidden_layers(agent._model), expected):
        torch.testing.assert_close(actual.weight, target, atol=2e-7, rtol=2e-5)
    for actual, source in zip(agent._model._disc_branches.branches,
                              reference._disc_branches.branches):
        for index in (0, 2):
            torch.testing.assert_close(actual[index].bias, source[index].bias,
                                       rtol=0, atol=0)
        torch.testing.assert_close(actual[-1].parametrizations.weight.original,
                                   source[-1].parametrizations.weight.original,
                                   rtol=0, atol=0)
    for actual, saved in zip(
            agent._model.get_actor_params() + agent._model.get_critic_params(),
            untouched):
        torch.testing.assert_close(actual, saved, rtol=0, atol=0)


def test_resume_restores_ns_count_optimizer_momentum_and_next_update():
    torch.manual_seed(137)
    source = _bare_agent()
    _build_optimizers(source)
    data = torch.randn(8, 172)
    source._disc_optimizer.step(_classification_loss(source._model, data))
    saved_model = copy.deepcopy(source._model.state_dict())
    saved_optimizer = copy.deepcopy(source._disc_optimizer.state_dict())
    restored = _bare_agent()
    _build_optimizers(restored)
    restored._model.load_state_dict(saved_model, strict=True)
    restored._disc_optimizer.load_state_dict(saved_optimizer)
    assert int(restored._model.get_ns_update_count()) == 1
    assert not hasattr(restored._model, "get_disc_logit_scale")
    for agent in (source, restored):
        agent._disc_optimizer.step(_classification_loss(agent._model, data))
        assert agent._disc_optimizer.get_steps() == 2
        assert int(agent._model.get_ns_update_count()) == 2
    for key, expected in source._model.state_dict().items():
        torch.testing.assert_close(restored._model.state_dict()[key], expected,
                                   rtol=0, atol=0)


class _AnalyticModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([-.1, -.4, -.8, -.3]))
        self.bias = torch.nn.Parameter(torch.tensor([.1, -.2, .4, -.1]))
        self.calls = []

    def get_disc_error_groups(self):
        return tuple((name, (index,)) for index, name in enumerate(
            ("root", "body_pos", "body_rot", "dof_vel")))

    def eval_disc_raw(self, disc_obs):
        self.calls.append(disc_obs.detach().clone())
        return disc_obs * self.weight + self.bias

    def eval_disc(self, disc_obs):
        return self.eval_disc_raw(disc_obs)

    def get_disc_logit_weights(self):
        return self.weight

    def get_disc_group_width(self):
        return self.weight.new_tensor(256.)

    def get_disc_group_total_width(self):
        return self.weight.new_tensor(1024.)


    def get_ns_update_count(self):
        return torch.zeros((), dtype=torch.int64)


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


def _analytic_agent():
    agent = _bare_agent(_AnalyticModel())
    agent._pos_diff = torch.zeros(4)
    agent._disc_obs_norm = DiffNormalizer((4,), "cpu")
    current = torch.tensor([[0., 2., 3., 4.], [0., 3., 4., 5.]])
    replay = torch.tensor([[0., 3., 2., 1.], [0., 2., 1., 4.]])
    agent._exp_buffer = _Replay(current)
    agent._disc_buffer = _Replay(replay)
    agent._disc_logit_reg = 0.
    agent._disc_eval_batch_size = 0
    agent._iter = 386
    return agent, {"disc_obs_demo": current,
                   "disc_obs": torch.zeros_like(current)}


def test_four_bce_gradients_match_independent_classifiers_with_real_local_negatives():
    agent, batch = _analytic_agent()
    data = torch.cat((agent._exp_buffer.values, agent._disc_buffer.values))
    info = agent._compute_disc_loss(batch)
    assert len(agent._model.calls) == 2
    torch.testing.assert_close(agent._model.calls[0], torch.zeros(1, 4))
    torch.testing.assert_close(agent._model.calls[1], data)
    info["disc_loss"].backward()
    losses = []
    for index in range(4):
        w = agent._model.weight[index].detach().clone().requires_grad_()
        b = agent._model.bias[index].detach().clone().requires_grad_()
        valid = data[:, index] != 0
        negative = (F.softplus(data[valid, index] * w + b).mean()
                    if valid.any() else 0. * w)
        loss = .5 * (F.softplus(-b) + negative)
        loss.backward()
        losses.append(loss.detach())
        torch.testing.assert_close(agent._model.weight.grad[index], w.grad)
        torch.testing.assert_close(agent._model.bias.grad[index], b.grad)
    torch.testing.assert_close(info["disc_loss"].detach(), torch.stack(losses).sum())
    torch.testing.assert_close(info["disc_cls_loss"], torch.stack(losses).mean())


def test_rewards_are_mean_of_four_individually_mapped_and_clipped_logits():
    agent, _ = _analytic_agent()
    values = torch.tensor([[-1000., 0., 1., 1000.], [-1., 2., -3., .5]],
                          requires_grad=True)
    agent._model.eval_disc = lambda disc_obs: disc_obs
    actual = agent._calc_disc_rewards(values)
    expected = calc_unscaled_disc_reward(values.detach()).mean(-1)
    torch.testing.assert_close(actual, expected)
    assert actual.shape == (2,) and not actual.requires_grad
    assert not torch.allclose(actual,
                              calc_unscaled_disc_reward(values.mean(-1)))


def test_checkpoint_rejects_different_group_indices_and_old_scalar_model():
    model = DAREMulti4NSModel(_config(), _Env()).eval()
    saved = copy.deepcopy(model.state_dict())
    indices = saved["_disc_branches.group_indices_0"]
    indices[0], indices[1] = indices[1].clone(), indices[0].clone()
    with pytest.raises(RuntimeError):
        model.load_state_dict(saved, strict=True)
    with pytest.raises(RuntimeError):
        model.load_state_dict(DAREModel(_config(), _Env()).state_dict(), strict=False)


@pytest.mark.parametrize("key,value", [
    ("ns_schema_version", torch.tensor(-1, dtype=torch.int64)),
    ("ns_update_count", torch.tensor(-1, dtype=torch.int64)),
    ("ns_update_count", torch.tensor(3., dtype=torch.float32)),
])
def test_checkpoint_rejects_wrong_constraint_schema_or_invalid_maintenance_count(key, value):
    model = DAREMulti4NSModel(_config(), _Env()).eval()
    saved = copy.deepcopy(model.state_dict())
    saved["_disc_branches." + key] = value
    with pytest.raises(RuntimeError):
        model.load_state_dict(saved, strict=False)


def test_agent_resume_rejects_changed_reward_loss_and_ns_semantics():
    source = _bare_agent()
    saved = source._get_checkpoint_extra_state()
    restored = _bare_agent()
    restored._load_checkpoint_extra_state(copy.deepcopy(saved))
    assert saved["dare_reward"]["version"] == 2
    assert saved["multi4_ns"]["version"] == 2
    assert "multi4_ns" in saved
    semantic_state = saved["multi4_ns"]
    for key in ("version", "groups", "reward_aggregation", "loss_aggregation",
                "hidden_widths", "hidden_constraint", "head_constraint",
                "ns_update_count"):
        assert key in semantic_state
        invalid = copy.deepcopy(saved)
        invalid["multi4_ns"][key] = "incompatible"
        with pytest.raises((ValueError, TypeError)):
            restored._load_checkpoint_extra_state(invalid)
    with pytest.raises(ValueError):
        restored._load_checkpoint_extra_state({})


def test_formal_config_keeps_optimizer_ppo_normalizer_reward_and_sampling_defaults():
    root = Path(__file__).resolve().parents[1]
    baseline = yaml.safe_load(
        (root / "data/agents/dare_multi_humanoid_agent.yaml").read_text())
    candidate = yaml.safe_load(
        (root / "data/agents/dare_multi4_ns_humanoid_agent.yaml").read_text())
    assert candidate["agent_name"] == "DAREMulti4NS"
    candidate["agent_name"] = baseline["agent_name"]
    assert candidate == baseline
