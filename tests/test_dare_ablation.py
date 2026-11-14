"""Declared A/B ablations: input semantics, constraints and training state."""

import copy

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from learning.add_agent import calc_unscaled_disc_reward
from learning.dare_ablation_agent import DAREAblationAgent
from learning.dare_ablation_model import DAREAblationModel
from learning.dare_monotone_model import DAREMonotoneModel, PositiveWeight
from learning.diff_normalizer import DiffNormalizer
from learning.diagonal_normalization import SignedDiagonalRescaling
from learning.physical_error_normalizer import PhysicalErrorNormalizer
from test_dare_monotone import Env, config, fill_buffers, physical_pair


VARIANTS = ("no_a", "no_b", "no_ab")


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def ablation_config(variant):
    candidate = config()
    candidate["agent_name"] = "DAREAblation"
    candidate["ablation_variant"] = variant
    return candidate


@pytest.mark.parametrize("variant", VARIANTS)
def test_variants_have_exact_declared_inputs_and_constraints(variant):
    env = Env()
    candidate = ablation_config(variant)
    agent = DAREAblationAgent(candidate, env, "cpu")
    model = agent._model
    assert model.variant == variant
    assert model.physical_errors == (variant == "no_b")
    assert model.diagonal_enabled == (variant == "no_a")
    assert model.input_dim == (64 if variant == "no_b" else 172)
    assert agent._pos_diff.shape == (model.input_dim,)
    (policy, _), (reference, _) = physical_pair(env, count=32)
    inputs = agent._pair_disc_inputs(policy, reference)
    expected = (model.pair_to_error_metrics(policy, reference) if variant == "no_b"
                else reference - policy)
    torch.testing.assert_close(inputs, expected, rtol=0, atol=0)
    assert inputs.shape == (32, model.input_dim)
    assert (inputs >= 0).all() if variant == "no_b" else (inputs < 0).any()
    assert isinstance(agent._disc_obs_norm,
                      PhysicalErrorNormalizer if variant == "no_b" else DiffNormalizer)
    assert agent._disc_obs_norm.get_shape() == ((7,) if variant == "no_b" else (172,))
    linears = [layer for layer in model._disc_energy if isinstance(layer, nn.Linear)]
    assert [(l.in_features, l.out_features) for l in linears] == [
        (model.input_dim, 341), (341, 512), (512, 1)]
    assert linears[-1].bias is None
    for layer in linears:
        if variant == "no_a":
            assert len(layer.parametrizations.weight) == 1
            assert isinstance(layer.parametrizations.weight[0], SignedDiagonalRescaling)
            assert (layer.weight < 0).any() and (layer.weight > 0).any()
            assert torch.linalg.matrix_norm(layer.weight.double(), ord=2) <= 1 + 2e-6
        elif variant == "no_b":
            assert len(layer.parametrizations.weight) == 1
            assert isinstance(layer.parametrizations.weight[0], PositiveWeight)
            assert (layer.weight >= 0).all()
        else:
            assert not parametrize.is_parametrized(layer)
            assert (layer.weight < 0).any() and (layer.weight > 0).any()
    torch.testing.assert_close(model.eval_disc(torch.zeros(4, model.input_dim)),
                               model._anchor_logit.expand(4, 1), rtol=0, atol=1e-7)
    assert model.eval_disc(torch.zeros(0, model.input_dim)).shape == (0, 1)
    with pytest.raises(ValueError, match=str(model.input_dim)):
        model.eval_disc(torch.zeros(2, 344))


@pytest.mark.parametrize("variant", VARIANTS)
def test_original_actor_critic_initialization_and_production_defaults_untouched(variant):
    env = Env()
    torch.manual_seed(71)
    full = DAREMonotoneModel(config()["model"], env)
    torch.manual_seed(71)
    model = DAREAblationModel(config()["model"] | {"ablation_variant": variant}, env)
    for a, b in zip(full.get_actor_params() + full.get_critic_params(),
                    model.get_actor_params() + model.get_critic_params()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert config()["agent_name"] == "DAREMonotone"
    assert full._disc_energy[0].in_features == 64
    assert len(full._disc_energy[0].parametrizations.weight) == 2


def test_b_ablation_matches_initial_effective_weights_biases_and_scores():
    env = Env()
    torch.manual_seed(77)
    bounded = DAREAblationModel(config()["model"] | {"ablation_variant": "no_a"}, env)
    torch.manual_seed(77)
    plain = DAREAblationModel(config()["model"] | {"ablation_variant": "no_ab"}, env)
    for constrained, raw in zip(bounded._disc_energy, plain._disc_energy):
        if isinstance(raw, nn.Linear):
            torch.testing.assert_close(constrained.weight, raw.weight, rtol=0, atol=0)
            if raw.bias is not None:
                torch.testing.assert_close(constrained.bias, raw.bias, rtol=0, atol=0)
    probe = torch.randn(32, bounded.input_dim)
    torch.testing.assert_close(bounded.eval_disc(probe), plain.eval_disc(probe), rtol=0, atol=0)
    torch.manual_seed(79)
    full = DAREMonotoneModel(config()["model"], env)
    torch.manual_seed(79)
    unbounded = DAREAblationModel(config()["model"] | {"ablation_variant": "no_b"}, env)
    for constrained, raw in zip(full._disc_energy, unbounded._disc_energy):
        if isinstance(raw, nn.Linear):
            torch.testing.assert_close(constrained.weight, raw.weight, rtol=0, atol=0)
            if raw.bias is not None:
                torch.testing.assert_close(constrained.bias, raw.bias, rtol=0, atol=0)
    probe = torch.rand(32, unbounded.input_dim)
    torch.testing.assert_close(full.eval_disc(probe), unbounded.eval_disc(probe), rtol=0, atol=0)
    # The initialization does not survive as an optimizer-time constraint.
    with torch.no_grad():
        unbounded._disc_energy[0].parametrizations.weight.original.mul_(10)
        plain._disc_energy[0].weight.fill_(-2)
    assert torch.linalg.matrix_norm(unbounded._disc_energy[0].weight.double(), ord=2) > 1
    assert (unbounded._disc_energy[0].weight >= 0).all()
    assert not parametrize.is_parametrized(plain._disc_energy[0])
    assert (plain._disc_energy[0].weight == -2).all()


@pytest.mark.parametrize("variant", VARIANTS)
def test_first_rollout_initializes_correct_history_and_reward_uses_original_mapping(variant):
    env = Env()
    agent = DAREAblationAgent(ablation_config(variant), env, "cpu").eval()
    (policy, _), (reference, _) = physical_pair(env)
    fill_buffers(agent, policy, reference)
    raw = agent._pair_disc_inputs(policy, reference)
    agent._compute_rewards()
    expected_scales = (agent._model.aggregate_error_metrics(raw).mean(0)
                       if variant == "no_b" else raw.abs().mean(0))
    torch.testing.assert_close(agent._disc_obs_norm.get_abs_mean(), expected_scales)
    assert int(agent._disc_obs_norm.get_count()) == len(policy)
    normalized = agent._disc_obs_norm.normalize(raw)
    expected_reward = calc_unscaled_disc_reward(agent._model.eval_disc(normalized).squeeze(-1))
    torch.testing.assert_close(agent._exp_buffer.get_data_flat("reward"), expected_reward)
    torch.testing.assert_close(agent.calc_disc_reward_from_obs(policy, reference), expected_reward)
    assert not hasattr(agent, "_disc_reward_scale")
    if variant != "no_b":
        assert torch.equal(torch.sign(normalized), torch.sign(reference - policy))


@pytest.mark.parametrize("variant", VARIANTS)
def test_bce_optimizer_updates_are_finite_without_gp_autograd(variant, monkeypatch):
    torch.manual_seed(73)
    agent = DAREAblationAgent(ablation_config(variant), Env(), "cpu")
    (policy, _), (reference, _) = physical_pair(agent._env)
    fill_buffers(agent, policy, reference)
    agent._compute_rewards()
    batch = {"disc_obs": policy[:16], "disc_obs_demo": reference[:16]}
    before = [parameter.detach().clone() for parameter in agent._model.get_disc_params()]
    def forbidden_gp(*args, **kwargs):
        raise AssertionError("GP differentiation must never be evaluated")
    monkeypatch.setattr(torch.autograd, "grad", forbidden_gp)
    for _ in range(3):
        info = agent._compute_disc_loss(batch)
        assert all(torch.isfinite(value).all() for value in info.values())
        for key in ("disc_grad_penalty", "disc_neg_grad_penalty", "disc_pos_grad_penalty"):
            assert info[key].item() == 0
        # Retain the declared original head penalty, no added loss terms.
        torch.testing.assert_close(info["disc_loss"], info["disc_cls_loss"]
                                   + agent._disc_logit_reg * info["disc_logit_loss"])
        agent._disc_optimizer.step(info["disc_loss"])
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in agent._model.get_disc_params())
    assert any(not torch.equal(old, new) for old, new in
               zip(before, agent._model.get_disc_params()))


@pytest.mark.parametrize("variant", VARIANTS)
def test_checkpoint_roundtrip_pending_history_and_cross_variant_rejection(variant, tmp_path):
    agent = DAREAblationAgent(ablation_config(variant), Env(), "cpu").eval()
    (policy, _), (reference, _) = physical_pair(agent._env)
    inputs = agent._pair_disc_inputs(policy, reference)
    agent._disc_obs_norm.record(inputs[:17])
    checkpoint = tmp_path / (variant + ".pt")
    agent.save_checkpoint(checkpoint, next_iter=4)
    restored = DAREAblationAgent(ablation_config(variant), Env(), "cpu").eval()
    restored.resume(checkpoint)
    assert restored._iter == 4
    for current in (agent, restored):
        current._disc_obs_norm.record(inputs[17:])
        current._disc_obs_norm.update()
    torch.testing.assert_close(agent._disc_obs_norm.get_abs_mean(),
                               restored._disc_obs_norm.get_abs_mean(), rtol=0, atol=0)
    torch.testing.assert_close(agent.calc_disc_reward_from_obs(policy, reference),
                               restored.calc_disc_reward_from_obs(policy, reference), rtol=0, atol=0)
    for other in set(VARIANTS) - {variant}:
        target = DAREAblationAgent(ablation_config(other), Env(), "cpu")
        with pytest.raises(RuntimeError, match="ablation variant/schema"):
            target.load_state_dict(copy.deepcopy(agent.state_dict()), strict=False)
        with pytest.raises(ValueError, match="ablation semantics"):
            target._load_checkpoint_extra_state(copy.deepcopy(agent._get_checkpoint_extra_state()))
    invalid = copy.deepcopy(agent.state_dict())
    invalid.pop("_model._ablation_variant_code")
    with pytest.raises(RuntimeError, match="ablation variant/schema"):
        restored.load_state_dict(invalid, strict=False)
    legacy_raw_init = copy.deepcopy(agent.state_dict())
    legacy_raw_init["_model._ablation_schema_version"] = torch.tensor(1, dtype=torch.int64)
    with pytest.raises(RuntimeError, match="ablation variant/schema"):
        restored.load_state_dict(legacy_raw_init, strict=False)
    legacy_metadata = copy.deepcopy(agent._get_checkpoint_extra_state())
    legacy_metadata["ablation_disc"].pop("initialization")
    with pytest.raises(ValueError, match="ablation semantics"):
        restored._load_checkpoint_extra_state(legacy_metadata)


@pytest.mark.parametrize("variant", VARIANTS)
def test_forbidden_gp_and_reward_scales_fail_explicitly(variant):
    for name, value in (("disc_grad_penalty", .1), ("disc_reward_scale", 2),
                        ("disc_group_negatives", True)):
        candidate = ablation_config(variant)
        candidate[name] = value
        with pytest.raises(ValueError):
            DAREAblationAgent(candidate, Env(), "cpu")
