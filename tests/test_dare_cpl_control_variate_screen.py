from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_cpl_control_variate_screen import (
    beta_from_independent_batches, linear_slope, stein_mean_control, td_lambda_target,
)


def test_stein_correction_has_zero_population_mean_and_not_naive_gradient_addition():
    torch.manual_seed(918)
    std = torch.tensor([.05, .07, .11])
    epsilon = torch.randn(200000, 3) * std
    slope = torch.tensor([1., -2., .3])
    z = stein_mean_control(epsilon, std, slope)
    assert (z.mean(0).abs() < 5 * z.std(0) / len(z) ** .5).all()
    assert (z + slope).mean(0).norm() > 1.


def test_stein_surrogate_autograd_matches_fixed_std_output_identity():
    mu = torch.tensor([.1, -.2], requires_grad=True)
    epsilon = torch.tensor([.03, -.05])
    std = torch.tensor([.05, .08])
    slope = torch.tensor([.2, -1.])
    action = mu.detach() + epsilon
    dist = torch.distributions.Normal(mu, std)
    logp = dist.log_prob(action).sum()
    loss = logp * (epsilon @ slope) - mu @ slope
    actual = torch.autograd.grad(loss, mu)[0]
    torch.testing.assert_close(actual, stein_mean_control(epsilon, std, slope))


def test_terminal_stops_td_lambda_and_retains_terminal_reward():
    reward = torch.tensor([[1., 2.], [999., 3.]])
    following = torch.tensor([[0., 4.], [777., 0.]])
    done = torch.tensor([[1, 0], [1, 2]])
    actual = td_lambda_target(reward, following, done, .9, .5)
    torch.testing.assert_close(actual, torch.tensor([1., 2. + .9 * (.5 * 4 + .5 * 3)]))


def test_linear_fit_and_variance_coefficient_recover_exact_independent_problem():
    torch.manual_seed(19)
    x = torch.randn(1024, 4)
    slope = torch.tensor([.2, 1., -.5, 2.])
    torch.testing.assert_close(linear_slope(x, 5 + x @ slope), slope)
    z = torch.randn(48, 16)
    constant = torch.randn(16)
    assert abs(beta_from_independent_batches(2 * z + constant, z) - 2.) < 1e-6
