from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from learning.trust_region_mean import mean_kl, project_mean, projection_regression, validate_parameters
from dare_trpl_learning import TrustRegionDARE, configure_variant, KL_BOUND, REGRESSION_WEIGHT
from test_ppo_phasebalanced import _agent


def test_projection_has_exact_identity_region_and_gaussian_kl_bound():
    torch.manual_seed(1941)
    old = torch.randn(400, 28, dtype=torch.float64)
    logstd = torch.linspace(-4, -.5, 28, dtype=torch.float64).expand_as(old)
    mean = (old + torch.randn_like(old) * .2).requires_grad_()
    mean.data[:4] = old[:4]
    actual, before = project_mean(mean, old, logstd, .05)
    assert torch.equal(actual[before <= .05], mean[before <= .05])
    distance = mean_kl(actual, old, logstd)
    assert distance.max() <= .05 + 1e-13
    torch.testing.assert_close(distance[before > .05], torch.full_like(distance[before > .05], .05))
    actual.sum().backward()
    assert torch.isfinite(mean.grad).all()


def test_projection_gradient_is_differentiable_inside_and_outside():
    old = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
    mean = torch.tensor([[.01, -.02], [1., 2.]], dtype=torch.float64, requires_grad=True)
    logstd = torch.zeros_like(old, requires_grad=True)
    assert torch.autograd.gradcheck(lambda x: project_mean(x, old, logstd, .05)[0], (mean,))
    projected, _ = project_mean(mean, old, logstd, .05)
    projected.sum().backward()
    assert old.grad is None and logstd.grad is None
    torch.testing.assert_close(mean.grad[0], torch.ones(2, dtype=mean.dtype))


def test_regression_target_is_detached_and_pulls_raw_mean_back():
    mean = torch.tensor([[.3, .4]], requires_grad=True)
    projected, _ = project_mean(mean, torch.zeros_like(mean), torch.zeros_like(mean), .005)
    penalty = projection_regression(mean, projected, torch.zeros_like(mean)).sum()
    grad = torch.autograd.grad(penalty, mean)[0]
    torch.testing.assert_close(grad, mean - projected.detach())
    assert (grad * mean).sum() > 0


def test_disabled_component_preserves_existing_loss_and_gradient_bitwise():
    agent = _agent()
    batch = dict(agent._exp_buffer._flat_buffers)
    expected = agent._compute_actor_loss(batch)["actor_loss"]
    grad = torch.autograd.grad(expected, agent._model.weight)[0]
    agent.__class__ = TrustRegionDARE
    actual = agent._compute_actor_loss(batch)["actor_loss"]
    assert torch.equal(expected, actual)
    assert torch.equal(grad, torch.autograd.grad(actual, agent._model.weight)[0])


def test_component_respects_phase_weights_action_mask_and_raw_bounds():
    agent = _agent()
    agent.__class__ = TrustRegionDARE
    agent._trpl_active = True
    batch = dict(agent._exp_buffer._flat_buffers)
    batch["rand_action_mask"] = batch["rand_action_mask"].clone()
    batch["rand_action_mask"][1::3] = 0
    with torch.no_grad():
        old_dist = agent._model.eval_actor(batch["obs"])
        batch["trpl_old_mean"] = old_dist.mean - .2
        batch["trpl_old_logstd"] = old_dist.logstd.clone()
    mask = batch["rand_action_mask"] == 1
    raw = agent._model.eval_actor(batch["obs"][mask])
    old, std = batch["trpl_old_mean"][mask], batch["trpl_old_logstd"][mask]
    projected, _ = project_mean(raw.mean, old, std, KL_BOUND)
    from learning.distribution_gaussian_diag import DistributionGaussianDiag
    dist = DistributionGaussianDiag(projected, std)
    ratio = (dist.log_prob(batch["action"][mask]) - batch["a_logp"][mask]).exp()
    expected = agent._weighted_mean(-ratio * batch["adv"][mask] + REGRESSION_WEIGHT * projection_regression(raw.mean, projected, std), batch["phase_weight"][mask])
    expected += agent._action_bound_weight * agent._weighted_mean(agent._compute_action_bound_loss(raw), batch["phase_weight"][mask])
    actual = agent._compute_actor_loss(batch)["actor_loss"]
    assert actual.requires_grad
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(torch.autograd.grad(actual, agent._model.weight)[0], torch.autograd.grad(expected, agent._model.weight)[0])


def test_optimizer_controls_are_explicit_and_do_not_change_input_configuration():
    original = dict(actor_optimizer=dict(type="SGD", learning_rate=1e-4))
    agent = SimpleNamespace()
    for name in ("ppo", "trpl_sgd", "ppo_adam", "trpl"):
        config, skip = configure_variant(agent, original, name)
        assert agent._trpl_active == name.startswith("trpl")
        assert bool(skip) == (name in ("ppo_adam", "trpl"))
        assert config["actor_optimizer"]["learning_rate"] == 1e-4
    assert original == dict(actor_optimizer=dict(type="SGD", learning_rate=1e-4))
    with pytest.raises(ValueError):
        configure_variant(agent, original, "unknown")


def test_old_policy_capture_precedes_update_and_checks_rollout_logprobs(monkeypatch):
    from learning.ppo_agent import PPOAgent
    agent = _agent()
    agent.__class__ = TrustRegionDARE
    agent._trpl_active = True
    obs = agent._exp_buffer.get_data_flat("obs")
    with torch.no_grad():
        old = agent._model.eval_actor(obs)
        agent._exp_buffer.set_data_flat("a_logp", old.log_prob(agent._exp_buffer.get_data_flat("action")))
    def update(current, batch_size, num_steps):
        torch.testing.assert_close(current._exp_buffer.get_data_flat("trpl_old_mean"), old.mean)
        assert not current._exp_buffer.get_data_flat("trpl_old_mean").requires_grad
        return {"checked": True}
    monkeypatch.setattr(PPOAgent, "_update_actor", update)
    assert agent._update_actor(16, 1) == {"checked": True}
    agent._exp_buffer.get_data_flat("a_logp").add_(.1)
    with pytest.raises(RuntimeError, match="Old policy/log-prob mismatch"):
        agent._update_actor(16, 1)


@pytest.mark.parametrize("bound,weight", [(0, 8), (.05, 0), (float("nan"), 8), (.05, float("inf"))])
def test_invalid_parameters_fail_closed(bound, weight):
    with pytest.raises(ValueError):
        validate_parameters(bound, weight)
