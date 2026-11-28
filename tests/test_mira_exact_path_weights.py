"""CPU checks for the experimental whole-network ReLU path constraint."""

import math

import pytest
import torch
from torch.nn import functional as F

from PAPER.tools.benchmark.mira_exact_path_weights import (
    enforce_nonnegative_path_contraction_,
    exact_positive_path_gain,
)


def _energy(z, weights, biases):
    for weight, bias in zip(weights[:-1], biases[:-1]):
        z = F.relu(F.linear(z, weight, bias))
    return F.linear(z, weights[-1], biases[-1]).squeeze(-1)


def _log_reward(energy):
    # Analytically identical to the ADD reward clamp, without the cancellation
    # of log(1 - sigmoid(-energy)) when the reward becomes small.
    return F.softplus(-energy).clamp_max(math.log(1e4)).log()


def test_constraint_clips_all_layers_but_only_shrinks_the_final_row():
    first = torch.nn.Parameter(torch.tensor([[3., -4.], [0., 2.]], dtype=torch.float64))
    second = torch.nn.Parameter(torch.tensor([[.5, .25], [-.1, .4]], dtype=torch.float64))
    final = torch.nn.Parameter(torch.tensor([[2., 1.]], dtype=torch.float64))
    chain = [first, second, final]

    gain = enforce_nonnegative_path_contraction_(chain)

    # After negative entries are removed, the path vector is (3, 1.8).
    expected_gain = math.hypot(3., 1.8)
    torch.testing.assert_close(gain, torch.tensor(expected_gain, dtype=torch.float64))
    torch.testing.assert_close(first, torch.diag(first.new_tensor([3., 2.])))
    torch.testing.assert_close(second, second.new_tensor([[.5, .25], [0., .4]]))
    torch.testing.assert_close(final, final.new_tensor([[2., 1.]]) / expected_gain)
    torch.testing.assert_close(exact_positive_path_gain(chain), gain.new_tensor(1.))
    assert all(weight.is_leaf and weight.requires_grad for weight in chain)
    torch.testing.assert_close(torch.linalg.matrix_norm(first.detach(), ord=2), gain.new_tensor(3.))


@pytest.mark.parametrize("active", [0., .5])
def test_feasible_chain_with_a_zero_column_is_unchanged(active):
    chain = [torch.tensor([[active, 0.], [0., 0.]]), torch.tensor([[1., 1.]])]
    original = [weight.clone() for weight in chain]
    gain = enforce_nonnegative_path_contraction_(chain)
    assert gain.item() == active
    for repaired, unchanged in zip(chain, original):
        torch.testing.assert_close(repaired, unchanged, rtol=0, atol=0)


def test_path_calculation_and_repair_do_not_overflow_for_large_fp32_weights():
    chain = [torch.tensor([[1e24, -1e24], [0., 2e24]]),
             torch.tensor([[1e12, 1e12]]), torch.tensor([[1e30]])]

    initial_gain = enforce_nonnegative_path_contraction_(chain)

    assert initial_gain.dtype == torch.float64 and torch.isfinite(initial_gain)
    assert initial_gain > torch.finfo(torch.float32).max
    assert all(torch.isfinite(weight).all() and (weight >= 0).all() for weight in chain)
    torch.testing.assert_close(exact_positive_path_gain(chain), initial_gain.new_tensor(1.),
                               rtol=1e-6, atol=0)


def test_global_bound_covers_relu_regions_and_upper_reward_clipping():
    random = torch.Generator().manual_seed(187)
    shapes = ((9, 5), (7, 9), (1, 7))
    weights = [torch.randn(shape, generator=random, dtype=torch.float64) for shape in shapes]
    enforce_nonnegative_path_contraction_(weights)
    gain = exact_positive_path_gain(weights)
    biases = [torch.linspace(-1., 1., 9, dtype=gain.dtype),
              torch.linspace(.2, -.7, 7, dtype=gain.dtype), gain.new_tensor([-20.])]
    x, y = torch.rand((2, 500, 5), generator=random, dtype=torch.float64)
    scales = torch.logspace(-4, 2, 500, dtype=gain.dtype).unsqueeze(1)
    x, y = x * scales, y * scales
    distance = torch.linalg.vector_norm(y - x, dim=1)
    x_energy, y_energy = _energy(x, weights, biases), _energy(y, weights, biases)
    x_reward, y_reward = _log_reward(x_energy), _log_reward(y_energy)

    assert ((y_energy - x_energy).abs() <= gain * distance + 1e-12).all()
    assert ((y_reward - x_reward).abs() <= gain * distance + 1e-12).all()
    assert ((x_reward - math.log(math.log(1e4))).abs() <= 1e-12).any()
    assert (x_reward < 0).any()
    lower_reward = _log_reward(_energy(torch.minimum(x, y), weights, biases))
    assert (lower_reward >= torch.maximum(x_reward, y_reward) - 1e-12).all()


def test_bound_is_tight_on_a_ray_after_negative_biases_stop_masking_paths():
    dtype = torch.float64
    weights = [torch.tensor([[2., 0., 0.], [0., 1., .3]], dtype=dtype),
               torch.tensor([[1.2, .1], [.2, .7]], dtype=dtype),
               torch.tensor([[.4, .1]], dtype=dtype)]
    biases = [torch.tensor([-3., 1.], dtype=dtype), torch.tensor([-2., -.4], dtype=dtype),
              torch.tensor([-20.], dtype=dtype)]
    gain = exact_positive_path_gain(weights)
    # These path coefficients are obtained by summing the four paths by hand.
    direction = torch.tensor([1., .11, .033], dtype=dtype)
    direction /= torch.linalg.vector_norm(direction)
    x, y = 140 * direction, 142 * direction
    e_x, e_y = _energy(x, weights, biases), _energy(y, weights, biases)

    assert F.relu(biases[0])[0] == 0  # One branch was initially inactive.
    torch.testing.assert_close((e_y - e_x) / 2, gain, rtol=1e-12, atol=0)
    torch.testing.assert_close((_log_reward(e_x) - _log_reward(e_y)) / 2, gain,
                               rtol=1e-12, atol=0)


def test_non_scalar_output_is_rejected_before_weights_are_modified():
    weight = torch.tensor([[-1.], [1.]])
    with pytest.raises(ValueError, match="last layer must have one output"):
        enforce_nonnegative_path_contraction_([weight])
    assert weight[0, 0] == -1


@pytest.mark.parametrize("target_gain", [.73, 2.8])
def test_dense_relu_scale_balance_attains_path_norm_product_without_changing_energy(target_gain):
    random = torch.Generator().manual_seed(830)
    dtype = torch.float64
    shapes = ((4, 3), (5, 4), (1, 5))
    weights = [torch.rand(shape, generator=random, dtype=dtype) + .05 for shape in shapes]
    weights[0] *= torch.tensor([.01, 30., 2., .6], dtype=dtype)[:, None]
    weights[1] *= torch.tensor([2., .5, 30., .06, .9], dtype=dtype)[:, None]
    weights[-1] *= target_gain / exact_positive_path_gain(weights)
    biases = [torch.linspace(-2, 1, shape[0], dtype=dtype) for shape in shapes]

    # Incoming and outgoing path sums determine each hidden change of units.
    backward = [torch.ones(1, dtype=dtype)]
    for weight in reversed(weights):
        backward.append(weight.T @ backward[-1])
    backward.reverse()
    forward = [backward[0]]
    for weight in weights:
        forward.append(weight @ forward[-1])
    scales = [(b / f).sqrt() for b, f in zip(backward[:-1], forward[:-1])]
    scales.append(torch.ones(1, dtype=dtype))  # Keep the output units and bias.
    balanced = [weight * scales[i + 1][:, None] / scales[i][None, :]
                for i, weight in enumerate(weights)]
    balanced_biases = [bias * scales[i + 1] for i, bias in enumerate(biases)]

    original_norms = [torch.linalg.matrix_norm(weight, ord=2) for weight in weights]
    norms = torch.stack([torch.linalg.matrix_norm(weight, ord=2) for weight in balanced])
    assert original_norms[0] > 1 and original_norms[1] > 1
    torch.testing.assert_close(norms, torch.tensor([1., 1., target_gain], dtype=dtype),
                               rtol=1e-12, atol=0)
    torch.testing.assert_close(exact_positive_path_gain(balanced), norms.prod(), rtol=1e-12,
                               atol=0)

    z = torch.randn((500, 3), generator=random, dtype=dtype)
    z = torch.cat((z, z.abs(), torch.zeros((1, 3), dtype=dtype)))
    torch.testing.assert_close(_energy(z, balanced, balanced_biases), _energy(z, weights, biases),
                               rtol=1e-12, atol=1e-12)
