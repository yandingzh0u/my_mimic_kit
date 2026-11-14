"""Numerical and optimizer contracts for signed diagonal normalization."""

import copy

import pytest
import torch
from torch import nn
from torch.nn.utils import parametrize

from learning.diagonal_normalization import (
    PositiveDiagonalRescaling,
    SignedDiagonalRescaling,
)


SHAPES = ((341, 172), (512, 341), (1, 512))


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _signed(shape, dtype=torch.float64):
    generator = torch.Generator().manual_seed(9137 + sum(shape))
    return torch.randn(shape, dtype=dtype, generator=generator)


def _norm(weight):
    return torch.linalg.svdvals(weight.detach().double())[0].item()


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_signed_gram_formula_and_positive_special_case(shape):
    weight = _signed(shape)
    normalization = SignedDiagonalRescaling(shape[1]).double()
    with torch.no_grad():
        normalization.log_q.copy_(torch.linspace(-4., 4., shape[1], dtype=weight.dtype))
    q = normalization.log_q.exp()
    t = ((weight.T @ weight).abs() @ q) / q
    expected = weight / t.sqrt().unsqueeze(0)
    actual = normalization(weight)
    torch.testing.assert_close(actual, expected, rtol=3e-12, atol=2e-12)
    assert _norm(actual) <= 1. + 3e-12
    assert torch.equal(torch.sign(actual), torch.sign(weight))

    positive = weight.abs() + .05
    positive_normalization = PositiveDiagonalRescaling(shape[1]).double()
    with torch.no_grad():
        positive_normalization.log_q.copy_(normalization.log_q)
    torch.testing.assert_close(normalization(positive), positive_normalization(positive),
                               rtol=3e-12, atol=2e-12)


@pytest.mark.parametrize("shape", SHAPES)
def test_perron_initialization_is_deterministic_and_matches_positive_case(shape):
    weight = _signed(shape)
    normalization = SignedDiagonalRescaling(shape[1]).double()
    before = torch.random.get_rng_state().clone()
    normalization.initialize_from_weight(weight)
    assert torch.equal(torch.random.get_rng_state(), before)
    assert torch.isfinite(normalization.log_q).all()
    assert _norm(normalization(weight)) <= 1. + 3e-12
    duplicate = SignedDiagonalRescaling(shape[1]).double()
    duplicate.initialize_from_weight(weight)
    torch.testing.assert_close(duplicate.log_q, normalization.log_q, rtol=0, atol=0)

    positive = weight.abs() + .05
    positive_normalization = PositiveDiagonalRescaling(shape[1]).double()
    positive_normalization.initialize_from_weight(positive)
    normalization.initialize_from_weight(positive)
    torch.testing.assert_close(normalization(positive), positive_normalization(positive),
                               rtol=3e-11, atol=2e-12)


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_random_q_spectral_bound_and_q_scale_gauge(shape, dtype):
    weight = _signed(shape, dtype)
    normalization = SignedDiagonalRescaling(shape[1]).to(dtype=dtype)
    with torch.no_grad():
        normalization.log_q.copy_(torch.linspace(-6., 6., shape[1], dtype=dtype))
    actual = normalization(weight)
    assert torch.isfinite(actual).all()
    assert _norm(actual) <= 1. + (5e-6 if dtype == torch.float32 else 3e-12)
    with torch.no_grad():
        normalization.log_q.add_(17.)
    torch.testing.assert_close(normalization(weight), actual,
                               rtol=2e-5 if dtype == torch.float32 else 2e-11,
                               atol=1e-7 if dtype == torch.float32 else 2e-12)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("zero_pattern", ("none", "column", "row", "all"))
def test_extreme_q_and_zero_weights_have_finite_forward_backward(dtype, zero_pattern):
    weight = _signed((11, 7), dtype)
    if zero_pattern == "column":
        weight[:, -1] = 0
    elif zero_pattern == "row":
        weight[0] = 0
    elif zero_pattern == "all":
        weight.zero_()
    weight.requires_grad_(True)
    normalization = SignedDiagonalRescaling(7).to(dtype=dtype)
    with torch.no_grad():
        normalization.log_q.fill_(-1000.)
        normalization.log_q[-1] = 1000.
    actual = normalization(weight)
    assert torch.isfinite(actual).all()
    assert _norm(actual) <= 1. + (1e-5 if dtype == torch.float32 else 3e-12)
    if zero_pattern == "all":
        assert torch.count_nonzero(actual) == 0
    else:
        assert actual.norm() > 1e-6
    direction = torch.linspace(-.7, 1.1, weight.numel(), dtype=dtype).reshape_as(weight)
    (actual * direction).sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert normalization.log_q.grad is not None
    assert torch.isfinite(normalization.log_q.grad).all()


def test_signed_weight_and_q_pass_double_gradcheck():
    normalization = SignedDiagonalRescaling(4).double()
    weight = _signed((6, 4)).requires_grad_(True)
    log_q = torch.tensor([-.3, .7, .1, -.2], dtype=torch.float64, requires_grad=True)

    def functional(current_weight, current_q):
        return torch.func.functional_call(normalization,
                                          {"log_q": current_q}, (current_weight,))

    assert torch.autograd.gradcheck(functional, (weight, log_q), eps=1e-6,
                                    atol=2e-5, rtol=2e-4)


def test_zero_element_in_live_column_retains_signed_linear_gradient():
    weight = _signed((6, 4))
    weight[0, 0] = 0
    weight.requires_grad_(True)
    normalization = SignedDiagonalRescaling(4).double()
    actual = normalization(weight)
    actual[0, 0].backward()
    expected = ((weight.detach().T @ weight.detach()).abs().sum(1)[0]).rsqrt()
    torch.testing.assert_close(weight.grad[0, 0], expected, rtol=1e-12, atol=1e-12)
    assert weight.grad[0, 0] > 0


@pytest.mark.parametrize("dtype,small,large", [
    (torch.float32, 1e-30, 1e30), (torch.float64, 1e-200, 1e200)])
def test_global_raw_weight_scale_is_a_gauge_at_large_dynamic_range(dtype, small, large):
    normalization = SignedDiagonalRescaling(7).to(dtype=dtype)
    weight = _signed((11, 7), dtype)
    baseline = normalization(weight)
    for scale in (small, large):
        current = (weight * scale).requires_grad_(True)
        actual = normalization(current)
        torch.testing.assert_close(actual, baseline,
                                   rtol=2e-5 if dtype == torch.float32 else 2e-11,
                                   atol=1e-7 if dtype == torch.float32 else 2e-12)
        actual.sum().backward()
        assert torch.isfinite(current.grad).all()


def test_parametrized_linear_checkpoint_roundtrip_and_optimizer_coverage():
    def make_layer():
        layer = nn.Linear(7, 11).double()
        normalization = SignedDiagonalRescaling(7).double()
        normalization.initialize_from_weight(layer.weight)
        parametrize.register_parametrization(layer, "weight", normalization)
        return layer

    layer = make_layer()
    optimizer = torch.optim.SGD(layer.parameters(), lr=.01)
    q = layer.parametrizations.weight[0].log_q
    assert sum(parameter is q for group in optimizer.param_groups
               for parameter in group["params"]) == 1
    layer(torch.randn(5, 7, dtype=torch.float64)).square().sum().backward()
    assert q.grad is not None and torch.isfinite(q.grad).all() and q.grad.norm() > 0
    optimizer.step()
    saved = copy.deepcopy(layer.state_dict())
    assert not any(name.endswith(("._u", "._v")) for name in saved)
    restored = make_layer()
    restored.load_state_dict(saved, strict=True)
    torch.testing.assert_close(restored.weight, layer.weight, rtol=0, atol=0)
    torch.testing.assert_close(restored.bias, layer.bias, rtol=0, atol=0)
    del saved["parametrizations.weight.0.log_q"]
    with pytest.raises(RuntimeError, match="log_q"):
        restored.load_state_dict(saved, strict=True)
