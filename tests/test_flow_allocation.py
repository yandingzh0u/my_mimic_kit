"""Mathematical gates for the isolated allocation candidate."""
from pathlib import Path

import numpy as np
import pytest
import torch
from types import SimpleNamespace
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
sys.path.insert(0, str(ROOT / "tools/diagnostics"))
import learning.trajectory_allocation as allocation
from dare_flow_allocation import EpisodeFlow


def test_neyman_solution_and_unequal_episode_lengths():
    visits = np.diag([1., 4., 16.])
    fit = allocation.solve_allocation(visits, [1., 4., 9.])
    desired = np.array([1, 2, 3]) / 6
    assert np.allclose(fit["predicted_occupancy"], desired, atol=1e-6)
    rates = desired / np.diag(visits)
    assert np.allclose(fit["reset_probabilities"], rates / rates.sum(), atol=1e-6)


def test_physical_flow_can_make_unconstrained_neyman_target_unreachable():
    fit = allocation.solve_allocation([[1., 10.], [0., 1.]], [100., 1.])
    assert np.allclose(fit["reset_probabilities"], [1, 0], atol=1e-6)
    assert np.allclose(fit["predicted_occupancy"], [1/11, 10/11], atol=1e-6)
    assert not np.allclose(fit["predicted_occupancy"], [10/11, 1/11])


def test_no_manual_phase_identity_and_invariance_to_global_moment_scale():
    visits = np.array([[5, 3, 2], [0, 7, 2], [0, 0, 8.]])
    moments = np.array([2., 3., 5.])
    fit = allocation.solve_allocation(visits, moments)
    order = np.array([2, 0, 1])
    permuted = allocation.solve_allocation(visits[order][:, order], moments[order] * 37)
    assert np.allclose(np.array(fit["reset_probabilities"])[order],
                       permuted["reset_probabilities"], atol=1e-5)


def test_zero_signal_retains_original_uniform_reset():
    fit = allocation.solve_allocation(np.diag([1., 5.]), [0, 0])
    assert fit["zero_signal"]
    assert np.allclose(fit["reset_probabilities"], [.5, .5])


def test_global_moment_scale_does_not_overflow():
    fit = allocation.solve_allocation(np.eye(2), [1e308, 1e308])
    assert np.allclose(fit["reset_probabilities"], [.5, .5])


def test_independent_certificate_can_accept_solver_false_status(monkeypatch):
    monkeypatch.setattr(allocation, "minimize", lambda *a, **kw: SimpleNamespace(
        x=np.array([.5, .5]), success=False, message="Synthetic false failure"))
    fit = allocation.solve_allocation(np.eye(2), [1., 1.])
    assert not fit["numerical_certificate"]["solver_success"]
    assert fit["numerical_certificate"]["relative_optimality_gap"] == 0


def test_bad_solver_optimum_is_refined_with_independent_certificate(monkeypatch):
    monkeypatch.setattr(allocation, "minimize", lambda *a, **kw: SimpleNamespace(
        x=np.array([.9, .1]), success=True, message="Synthetic false success"))
    fit = allocation.solve_allocation(np.eye(2), [1., 1.])
    assert np.allclose(fit["reset_probabilities"], [.5, .5], atol=1e-6)
    assert fit["numerical_certificate"]["refinements"] > 0
    assert fit["numerical_certificate"]["relative_optimality_gap"] <= 1e-6


def test_head_score_norm_agrees_with_autograd_including_bias():
    torch.manual_seed(9071)
    head = torch.nn.Linear(5, 3).double()
    hidden = torch.randn(7, 5, dtype=torch.float64)
    means = head(hidden)
    std = torch.full_like(means, .2)
    actions = means.detach() + std * torch.randn_like(means)
    advantages = torch.randn(7, dtype=torch.float64)
    predicted = allocation.head_gradient_squared_norm(hidden, actions, means, std, advantages)
    for i in range(7):
        logp = -.5 * ((actions[i] - means[i]) / std[i]).square().sum()
        gradients = torch.autograd.grad(logp * advantages[i], list(head.parameters()), retain_graph=True)
        actual = sum(g.square().sum() for g in gradients)
        assert torch.allclose(predicted[i], actual, atol=1e-10)


def test_missing_phase_moments_only_allowed_for_diagnostic_logging():
    data = dict(obs=torch.ones(4, 2), action=torch.ones(4, 2),
                adv=torch.ones(4), diag_phase=torch.full((4,), .1))
    agent = SimpleNamespace(
        _device="cpu", get_num_envs=lambda: 4,
        _exp_buffer=SimpleNamespace(get_data_flat=lambda key: data[key]),
        _obs_norm=SimpleNamespace(normalize=lambda x: x),
        _a_norm=SimpleNamespace(normalize=lambda x: x),
        _model=SimpleNamespace(_actor_layers=lambda x: x,
                              _action_dist=lambda x: torch.distributions.Normal(x * 0, x)),
    )
    with pytest.raises(RuntimeError, match="Incomplete"):
        allocation.actor_moments(agent, 2)
    assert np.array_equal(allocation.actor_moments(agent, 2, require_complete=False), [6., 0.])


@pytest.mark.parametrize("visits,moments", [([[0, 0], [0, 1]], [1, 1]),
                                           ([[1, -1], [0, 1]], [1, 1]),
                                           ([[1, 0], [1, 0]], [1, 1]),
                                           ([[1, 0], [0, 1]], [1, -1]),
                                           ([[1, 0], [0, 1]], [1])])
def test_invalid_or_unreachable_inputs_are_not_silently_fitted(visits, moments):
    with pytest.raises(ValueError):
        allocation.solve_allocation(visits, moments)


class FakeEnv:
    def __init__(self):
        self.phase = torch.linspace(.05, .95, 8)
        self._motion_lib = type("MotionLib", (), {
            "get_motion_length": lambda self, ids: torch.ones_like(ids, dtype=torch.float32),
        })()

    def get_motion_phase(self):
        return self.phase

    def get_num_envs(self):
        return len(self.phase)

    def _sample_motion_times(self, n):
        return torch.zeros(n, dtype=torch.long), torch.full((n,), .75)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.phase[ids] = .75
        return self.phase.clone(), {}

    def step(self, actions):
        self.phase += .01
        return self.phase.clone(), torch.zeros(8), actions.long(), {}


def test_full_population_reset_and_eval_do_not_use_candidate_birth_distribution():
    env = FakeEnv()
    flow = EpisodeFlow(env, 2, 73)
    flow.probabilities = torch.tensor([1., 0.])
    flow.collect = True
    _, full = env._sample_motion_times(8)
    _, partial = env._sample_motion_times(7)
    assert torch.all(full == .75)
    assert torch.all((partial >= 0) & (partial < .5))
    flow.collect = False
    assert torch.all(env._sample_motion_times(7)[1] == .75)
    flow.close()
    assert env._sample_motion_times.__func__ is FakeEnv._sample_motion_times


def test_visit_statistics_commit_only_terminated_complete_episodes():
    env = FakeEnv()
    flow = EpisodeFlow(env, 2, 73)
    flow.collect = True
    flow.starts[:] = torch.arange(8) // 4
    env.step(torch.zeros(8))
    assert not flow.pending and flow.flush() is None
    env.step(torch.ones(8))
    visits = flow.flush()
    assert np.allclose(visits, [[2, 0], [0, 2]])
    env.reset(torch.tensor([0, 7]))
    assert torch.all(flow.visits[[0, 7]] == 0)
    flow.close()


def test_episode_fifo_keeps_most_recent_entries_when_one_flush_overflows():
    env = FakeEnv()
    flow = EpisodeFlow(env, 2, 73)
    entries = torch.stack((torch.arange(1, 10), torch.zeros(9))).T.float()
    flow.pending.append((torch.zeros(9, dtype=torch.long), entries))
    flow.pending.append((torch.ones(1, dtype=torch.long), torch.tensor([[0., 1.]])))
    visits = flow.flush()
    assert flow.capacity == 4 and flow.sizes.tolist() == [4, 1]
    assert np.allclose(visits, [[7.5, 0], [0, 1]])
    flow.close()
