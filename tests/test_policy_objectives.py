"""Published SPO objective, exact default-PPO compatibility and integration."""
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))
from learning.policy_objectives import policy_objective, validate_policy_objective
from test_ppo_phasebalanced import _agent


def test_default_is_bitwise_original_ppo_including_gradients():
    ratio = torch.tensor([.4, .9, 1., 1.1, 1.6], requires_grad=True)
    advantage = torch.tensor([2., -3., 0., 1., -2.])
    old = torch.minimum(ratio * advantage, advantage * ratio.clamp(.8, 1.2))
    actual = policy_objective(ratio, advantage, .2)
    assert torch.equal(old, actual)
    assert torch.equal(torch.autograd.grad(old.sum(), ratio)[0],
                       torch.autograd.grad(actual.sum(), ratio)[0])


def test_spo_matches_official_formula_and_corrects_overshoot():
    ratio = torch.tensor([1.4, .6, 1., 1.], dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([1., -1., 2., -2.], dtype=torch.float64)
    result = policy_objective(ratio, adv, .2, "spo")
    expected = ratio * adv - adv.abs() * (ratio - 1)**2 / .4
    torch.testing.assert_close(result, expected)
    derivative = torch.autograd.grad(result.sum(), ratio)[0]
    torch.testing.assert_close(derivative, torch.tensor([-1., 1., 2., -2.], dtype=torch.float64))
    ppo_derivative = torch.autograd.grad(policy_objective(ratio, adv, .2).sum(), ratio)[0]
    torch.testing.assert_close(ppo_derivative[:2], torch.zeros(2, dtype=torch.float64))


def test_spo_initial_gradient_equals_ppo_and_zero_adv_stays_zero():
    ratio = torch.ones(5, dtype=torch.float64, requires_grad=True)
    adv = torch.tensor([-3., -.4, 0., .8, 5.], dtype=torch.float64)
    gradients = [torch.autograd.grad(policy_objective(ratio, adv, .2, name).sum(), ratio)[0]
                 for name in ("ppo", "spo")]
    assert torch.equal(*gradients)
    torch.testing.assert_close(gradients[0], adv)


@pytest.mark.parametrize("name,epsilon", [("other", .2), ("spo", 0.),
                                         ("ppo", float("nan")), ("spo", 1.)])
def test_invalid_configuration_rejected(name, epsilon):
    with pytest.raises(ValueError):
        validate_policy_objective(name, epsilon)


def test_phase_weighted_spo_preserves_action_mask_and_regularizer():
    agent = _agent()
    agent._actor_objective = "spo"
    batch = dict(agent._exp_buffer._flat_buffers)
    # One deterministic action must not enter the policy objective.
    batch["rand_action_mask"] = batch["rand_action_mask"].clone()
    batch["rand_action_mask"][0] = 0
    keep = batch["rand_action_mask"] == 1
    dist = agent._model.eval_actor(batch["obs"][keep])
    ratio = (dist.log_prob(batch["action"][keep]) - batch["a_logp"][keep]).exp()
    weights = batch["phase_weight"][keep]
    expected = -agent._weighted_mean(
        policy_objective(ratio, batch["adv"][keep], .2, "spo"), weights)
    bounds = agent._compute_action_bound_loss(dist)
    if bounds is not None:
        expected = expected + agent._action_bound_weight * agent._weighted_mean(bounds, weights)
    info = agent._compute_actor_loss(batch)
    torch.testing.assert_close(info["actor_loss"], expected)
    torch.testing.assert_close(torch.autograd.grad(info["actor_loss"], agent._model.weight)[0],
                               torch.autograd.grad(expected, agent._model.weight)[0])
    assert float(info["actor_objective_spo"]) == 1
    assert not info["ppo_flat_frac"].requires_grad
