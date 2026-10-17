import math
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))
from learning.reppo import (CategoricalQ, Duals, actor_objective,
                            frozen_parameters, gaussian_labels, soft_lambda_returns)
from learning.distribution_gaussian_diag import DistributionGaussianDiag as Gaussian


def test_hl_gauss_mass_and_action_derivative():
    torch.manual_seed(1)
    q = CategoricalQ(3, 2, -10, 10, width=16, bins=51)
    x, a = torch.randn(7, 3), torch.randn(7, 2, requires_grad=True)
    labels = gaussian_labels(torch.linspace(-100, 100, 7), q.edges)
    torch.testing.assert_close(labels.sum(-1), torch.ones(7))
    assert (labels >= 0).all()
    g = torch.autograd.grad(q.value(x, a).sum(), a)[0]
    h = 1e-3
    with torch.no_grad():
        perturb = torch.zeros_like(a)
        perturb[:, 0] = h
        finite = (q.value(x, a + perturb) - q.value(x, a - perturb)) / (2 * h)
    torch.testing.assert_close(g[:, 0], finite, atol=4e-4, rtol=.03)


def test_terminal_no_reset_leak_and_time_bootstrap():
    r = torch.tensor([[1., 2., 3.], [100., 200., 300.]])
    done = torch.tensor([[1, 2, 3], [0, 0, 0]])
    value, logp = torch.full_like(r, 10), torch.full_like(r, 2)
    target = soft_lambda_returns(r, value, logp, done, gamma=.9, lam=.95, alpha=.5)
    torch.testing.assert_close(target[0], torch.tensor([1., 2., 11.1]))
    torch.testing.assert_close(target[1], r[1] + 8.1)


def test_entropy_in_both_lambda_branches_exact_constant_case():
    r = torch.ones(3, 2)
    q = torch.full_like(r, 5.)
    p = torch.full_like(r, 2.)
    done = torch.zeros_like(r, dtype=torch.long)
    # At lambda=1 every transition contains entropy of each future action once.
    target = soft_lambda_returns(r, q, p, done, gamma=.9, lam=1., alpha=.5)
    torch.testing.assert_close(target[-1], torch.full((2,), 4.6))
    torch.testing.assert_close(target[-2], 1 + .9 * (target[-1] - 1))
    torch.testing.assert_close(soft_lambda_returns(r, q, p, done, gamma=.9, lam=0., alpha=.5),
                               torch.full_like(r, 4.6))


class LinearQ(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([2., -3.]))

    def value(self, obs, action):
        return action @ self.weight


def test_frozen_q_still_gives_pathwise_actor_gradient_and_restores_flags():
    mean = nn.Parameter(torch.zeros(4, 2))
    logstd = nn.Parameter(torch.full((4, 2), math.log(.05)))
    dist = Gaussian(mean, logstd)
    old = Gaussian(mean.detach().clone(), logstd.detach().clone())
    q, duals = LinearQ(), Duals()
    with frozen_parameters(q):
        loss, _ = actor_objective(dist, old, q, torch.zeros(4, 3), torch.zeros(4, 2),
                                   duals, torch.ones(4), target_entropy=-1)
        loss.backward()
    torch.testing.assert_close(mean.grad, -q.weight.detach().expand(4, 2) / 4)
    assert q.weight.grad is None and q.weight.requires_grad
    assert logstd.grad is not None and duals.log_alpha.grad < 0
    assert duals.log_beta.grad > 0  # KL below limit -> decrease multiplier.
    with pytest.raises(RuntimeError):
        with frozen_parameters(q):
            raise RuntimeError("test")
    assert q.weight.requires_grad


def test_outside_kl_region_actor_is_pulled_back_not_towards_q():
    mean = nn.Parameter(torch.ones(4, 2))
    dist = Gaussian(mean, torch.zeros(4, 2))
    old = Gaussian(torch.zeros(4, 2), torch.zeros(4, 2))
    q, duals = LinearQ(), Duals()
    with frozen_parameters(q):
        loss, info = actor_objective(dist, old, q, torch.zeros(4, 3), torch.zeros(4, 2),
                                      duals, torch.ones(4))
        loss.backward()
    assert info["reppo_kl_clipped_fraction"] == 1
    assert (mean.grad > 0).all() and duals.log_beta.grad < 0


def test_zero_alpha_matches_existing_td_lambda_with_terminal_replacement():
    from learning.rl_util import compute_td_lambda_return
    torch.manual_seed(7)
    r, v = torch.rand(32, 7), torch.rand(32, 7)
    done = torch.randint(0, 4, (32, 7))
    v[(done == 1) | (done == 2)] = 0
    original = compute_td_lambda_return(r, v, done, .99, .95)
    target = soft_lambda_returns(r, v, torch.randn_like(v), done, alpha=0)
    torch.testing.assert_close(original, target)
