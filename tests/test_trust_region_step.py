"""Real-parameter trust region: full-data gradient/HVP and fail-closed rollback."""
import copy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from learning.trust_region_step import (
    assign_vector, backtracking, conjugate_gradient, flat_gradient,
)
from learning.trust_region_mean import mean_kl
import dare_trpo_learning as trpo
from test_ppo_phasebalanced import _agent


def test_conjugate_gradient_matches_spd_solution_and_zero_rhs():
    matrix = torch.tensor([[4., 1.], [1., 3.]], dtype=torch.float64)
    rhs = torch.tensor([1., 2.], dtype=torch.float64)
    actual, info = conjugate_gradient(lambda x: matrix @ x, rhs)
    torch.testing.assert_close(actual, torch.linalg.solve(matrix, rhs))
    assert info["cg_steps"] == 2 and info["cg_positive_curvature"]
    actual, info = conjugate_gradient(lambda x: matrix @ x, rhs * 0)
    assert torch.equal(actual, rhs * 0) and info["cg_steps"] == 0


def test_nonpositive_or_nonfinite_curvature_cannot_be_hidden():
    value, info = conjugate_gradient(lambda x: -x, torch.ones(2))
    assert not info["cg_positive_curvature"] and torch.equal(value, torch.zeros(2))
    with pytest.raises(RuntimeError, match="Nonfinite Fisher"):
        conjugate_gradient(lambda x: x * float("nan"), torch.ones(2))


def test_backtracking_measures_raw_parameters_and_checks_actual_kl():
    parameter = torch.nn.Parameter(torch.tensor([0.], dtype=torch.float64))
    def evaluate():
        return dict(objective=float(parameter.detach().item()), kl=float(.5 * parameter.detach().square().item()))
    result = backtracking([parameter], torch.tensor([1.], dtype=torch.float64),
                          evaluate, kl_limit=.01)
    assert result["accepted"] and result["trials"] == 10
    assert 0 < result["final"]["kl"] <= .01 and parameter.item() > 0
    assert result["step_fraction"] == .8 ** 9


@pytest.mark.parametrize("failure", ["no_improvement", "nonfinite", "exception"])
def test_rejected_or_interrupted_search_restores_exact_original_bits(failure):
    parameters = [torch.nn.Parameter(torch.tensor([.125, -.375])),
                  torch.nn.Parameter(torch.tensor([[.625]]))]
    original = torch.nn.utils.parameters_to_vector(parameters).detach().clone()
    def evaluate():
        current = torch.nn.utils.parameters_to_vector(parameters).detach()
        changed = not torch.equal(current, original)
        if changed and failure == "exception":
            raise RuntimeError("interrupted evaluation")
        objective = float("nan") if changed and failure == "nonfinite" else -float((current - original).square().sum())
        return dict(objective=objective, kl=0.)
    if failure == "exception":
        with pytest.raises(RuntimeError, match="interrupted"):
            backtracking(parameters, torch.ones_like(original), evaluate)
    else:
        assert not backtracking(parameters, torch.ones_like(original), evaluate)["accepted"]
    assert torch.equal(original, torch.nn.utils.parameters_to_vector(parameters))


def test_invalid_vector_is_rejected_before_any_parameter_is_changed():
    params = [torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(1))]
    with pytest.raises(ValueError):
        assign_vector(params, torch.zeros(2))
    assert all(torch.equal(p, torch.ones_like(p)) for p in params)


def test_inactive_trpo_delegates_unchanged_ppo_update(monkeypatch):
    from learning.ppo_agent import PPOAgent
    agent = _agent()
    agent.__class__ = trpo.RawTrustRegionDARE
    monkeypatch.setattr(PPOAgent, "_update_actor", lambda current, size, steps: (size, steps))
    assert agent._update_actor(16, 40) == (16, 40)


def _trpo_agent(chunk):
    agent = _agent()
    agent.__class__ = trpo.RawTrustRegionDARE
    agent._trpo_active = True
    agent._env.get_num_envs = lambda: chunk
    agent._model.get_actor_params = lambda: [agent._model.weight]
    agent._model.get_critic_params = lambda: []
    agent._actor_optimizer = SimpleNamespace(get_steps=lambda: 0)
    obs = agent._exp_buffer.get_data_flat("obs")
    actions = agent._exp_buffer.get_data_flat("action")
    with torch.no_grad():
        agent._exp_buffer.set_data_flat("a_logp", agent._model.eval_actor(obs).log_prob(actions))
    agent._exp_buffer.get_data_flat("rand_action_mask")[1::3] = 0
    return agent


def test_full_rollout_chunked_gradient_fisher_and_mask_match_manual_formula(monkeypatch):
    agent = _trpo_agent(7)  # Multiple chunks, last partial chunk.
    data = agent._exp_buffer._flat_buffers
    obs, weights = data["obs"], data["phase_weight"]
    old = agent._model.eval_actor(obs)
    keep = data["rand_action_mask"] == 1
    # At the old policy, unclipped TRPO and original PPO gradients coincide.
    expected = -agent._compute_actor_loss(dict(data))["actor_loss"]
    gradient = flat_gradient(expected, [agent._model.weight]).detach()
    old_mean = old.mean.detach()
    def kl_of_weight(weight):
        return agent._weighted_mean(mean_kl(obs @ weight, old_mean, old.logstd), weights)
    matrix = torch.autograd.functional.hessian(kl_of_weight, agent._model.weight).reshape(2, 2)
    matrix += torch.eye(2) * trpo.DAMPING
    sampled = []
    sample = agent._exp_buffer.sample
    def spy(size):
        sampled.append(size)
        return sample(size)
    monkeypatch.setattr(agent._exp_buffer, "sample", spy)
    def solve(operator, rhs, steps):
        torch.testing.assert_close(rhs, gradient, rtol=2e-5, atol=2e-5)
        actual_matrix = torch.stack([operator(v) for v in torch.eye(2)], dim=1)
        torch.testing.assert_close(actual_matrix, matrix, rtol=2e-5, atol=2e-5)
        return torch.linalg.solve(actual_matrix, rhs), dict(cg_steps=2,
            cg_relative_residual_squared=0., cg_positive_curvature=True)
    monkeypatch.setattr(trpo, "conjugate_gradient", solve)
    result = agent._update_actor(16, 2)
    assert sampled == [16, 16] and agent._actor_optimizer.get_steps() == 0
    assert result["trpo_full_weighted_kl"] <= trpo.KL_LIMIT
    assert result["trpo_accepted"] == 1 and result["trpo_objective_gain"] > 0
    assert not all(keep)  # Deterministic actions really were excluded from loss.


def test_chunking_does_not_change_accepted_raw_policy():
    agents = [_trpo_agent(chunk) for chunk in (7, 48)]
    results = [a._update_actor(16, 2) for a in agents]
    assert all(r["trpo_accepted"] == 1 for r in results)
    torch.testing.assert_close(agents[0]._model.weight, agents[1]._model.weight,
                               rtol=3e-5, atol=3e-6)


@pytest.mark.parametrize("bad_value", [.1, float("nan")])
def test_misaligned_old_logprob_fails_before_mutating_actor(bad_value):
    agent = _trpo_agent(7)
    original = agent._model.weight.detach().clone()
    agent._exp_buffer.get_data_flat("a_logp")[0] += bad_value
    with pytest.raises(RuntimeError, match="Old policy/log-prob mismatch"):
        agent._update_actor(16, 2)
    assert torch.equal(original, agent._model.weight)


def test_variant_preserves_input_config_and_declares_unused_optimizer():
    original = dict(actor_optimizer=dict(type="SGD", learning_rate=1e-4))
    before = copy.deepcopy(original)
    agent = SimpleNamespace()
    config, skip = trpo.configure_variant(agent, original, "trpo")
    assert original == before and not skip and agent._trpo_active
    assert "unused" in config["diagnostic_actor_update"]
    trpo.configure_variant(agent, original, "ppo")
    assert not agent._trpo_active
