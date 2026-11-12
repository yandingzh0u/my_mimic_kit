"""Independent CPU numerical audit of positive SLL linear normalization.

These tests cover the normalization's mathematical and checkpoint contracts;
they do not assert that a bounded scorer improves PPO or tracking accuracy.
"""

import copy

import pytest
import torch

from learning.diagonal_normalization import PositiveDiagonalRescaling
from learning.dare_monotone_model import positive_diagonal_linear


SHAPES = ((341, 7), (512, 341), (1, 512))


@pytest.fixture(autouse=True)
def single_cpu_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _positive(shape, dtype=torch.float64):
    generator = torch.Generator().manual_seed(8121 + shape[0] + shape[1])
    return torch.rand(shape, dtype=dtype, generator=generator) + .05


def _norm(weight):
    return torch.linalg.svdvals(weight.detach().double())[0].item()


@pytest.mark.parametrize("shape", SHAPES)
def test_perron_initialization_matches_exact_sn_on_all_production_shapes(shape):
    weight = _positive(shape)
    normalization = PositiveDiagonalRescaling(shape[1]).double()
    rng_before = torch.random.get_rng_state().clone()
    normalization.initialize_from_weight(weight)
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    actual = normalization(weight)
    expected = weight / torch.linalg.svdvals(weight)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    assert all((tensor >= 0).all() for tensor in (actual,))
    assert torch.isfinite(actual).all()


@pytest.mark.parametrize("shape", SHAPES)
def test_unit_q_reproduces_aol_and_q_has_nonzero_finite_gradient(shape):
    weight = _positive(shape).requires_grad_(True)
    normalization = PositiveDiagonalRescaling(shape[1]).double()
    with torch.no_grad():
        normalization.log_q.zero_()
    actual = normalization(weight)
    expected = weight / (weight.T @ weight).sum(1).sqrt()[None, :]
    torch.testing.assert_close(actual, expected, rtol=2e-11, atol=2e-12)
    direction = torch.linspace(-1., 1., weight.numel(), dtype=weight.dtype).reshape_as(weight)
    (actual * direction).sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert normalization.log_q.grad is not None
    assert torch.isfinite(normalization.log_q.grad).all()
    assert normalization.log_q.grad.norm() > 1e-8


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_random_q_keeps_nonnegative_spectral_bound_and_scale_gauge(shape, dtype):
    weight = _positive(shape, dtype)
    normalization = PositiveDiagonalRescaling(shape[1]).to(dtype=dtype)
    with torch.no_grad():
        normalization.log_q.copy_(torch.linspace(-4., 4., shape[1], dtype=dtype))
    actual = normalization(weight)
    assert (actual >= 0).all() and torch.isfinite(actual).all()
    assert _norm(actual) <= 1. + (4e-6 if dtype == torch.float32 else 2e-12)
    with torch.no_grad():
        normalization.log_q.add_(17.)
    torch.testing.assert_close(normalization(weight), actual,
                               rtol=1e-5 if dtype == torch.float32 else 2e-11,
                               atol=1e-7 if dtype == torch.float32 else 2e-12)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
@pytest.mark.parametrize("zero_pattern", ("none", "column", "row", "all"))
def test_extreme_q_and_zero_weights_have_finite_forward_backward_and_hard_bound(dtype, zero_pattern):
    weight = _positive((11, 7), dtype)
    if zero_pattern == "column":
        # The dominant q sits on the zero column: directly exponentiating q
        # after subtracting max would underflow every active column to zero.
        weight[:, -1] = 0
    elif zero_pattern == "row":
        weight[0] = 0
    elif zero_pattern == "all":
        weight.zero_()
    weight.requires_grad_(True)
    normalization = PositiveDiagonalRescaling(7).to(dtype=dtype)
    with torch.no_grad():
        normalization.log_q.fill_(-1000.)
        normalization.log_q[-1] = 1000.
    actual = normalization(weight)
    assert torch.isfinite(actual).all() and (actual >= 0).all()
    assert _norm(actual) <= 1. + (1e-5 if dtype == torch.float32 else 3e-12)
    if zero_pattern == "all":
        assert torch.count_nonzero(actual) == 0
    else:
        assert actual.norm() > 1e-6
    direction = torch.linspace(-.7, 1.1, actual.numel(), dtype=dtype).reshape_as(actual)
    (actual * direction).sum().backward()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()
    assert normalization.log_q.grad is not None
    assert torch.isfinite(normalization.log_q.grad).all()


def test_positive_dense_normalization_passes_double_gradcheck():
    normalization = PositiveDiagonalRescaling(4).double()
    log_q = torch.tensor([-.3, .7, .1, -.2], dtype=torch.float64,
                         requires_grad=True)
    weight = _positive((6, 4)).requires_grad_(True)
    def functional(current_weight, current_q):
        return torch.func.functional_call(normalization,
                                           {"log_q": current_q}, (current_weight,))
    assert torch.autograd.gradcheck(functional, (weight, log_q), eps=1e-6,
                                    atol=2e-5, rtol=2e-4)


@pytest.mark.parametrize("dtype,small,large", [
    (torch.float32, 1e-30, 1e30), (torch.float64, 1e-200, 1e200)])
def test_global_weight_radius_is_a_gauge_even_at_large_dynamic_range(dtype, small, large):
    normalization = PositiveDiagonalRescaling(7).to(dtype=dtype)
    original = _positive((11, 7), dtype)
    baseline = normalization(original)
    for radius in (small, large):
        weight = (original * radius).requires_grad_(True)
        actual = normalization(weight)
        assert torch.isfinite(actual).all()
        torch.testing.assert_close(actual, baseline,
                                   rtol=1e-5 if dtype == torch.float32 else 2e-11,
                                   atol=1e-7 if dtype == torch.float32 else 2e-12)
        actual.sum().backward()
        assert torch.isfinite(weight.grad).all()


def test_parameter_and_checkpoint_roundtrip_include_learned_q():
    layer = positive_diagonal_linear(7, 11).double()
    parameters = dict(layer.named_parameters())
    q_names = [name for name in parameters if name.endswith("log_q")]
    assert len(q_names) == 1
    assert parameters[q_names[0]].shape == (7,)
    assert not any(name.endswith(("._u", "._v")) for name in layer.state_dict())
    with torch.no_grad():
        parameters[q_names[0]].add_(torch.linspace(-.3, .5, 7, dtype=torch.float64))
    expected = layer(torch.randn(5, 7, dtype=torch.float64))
    saved = copy.deepcopy(layer.state_dict())
    restored = positive_diagonal_linear(7, 11).double()
    restored.load_state_dict(saved, strict=True)
    torch.testing.assert_close(restored.weight, layer.weight, rtol=0, atol=0)
    torch.testing.assert_close(restored.bias, layer.bias, rtol=0, atol=0)
    assert expected.shape == (5, 11)
    missing_q = copy.deepcopy(saved)
    missing_q.pop(q_names[0])
    with pytest.raises(RuntimeError, match="log_q"):
        restored.load_state_dict(missing_q, strict=True)


def test_all_three_factory_shapes_preserve_sign_and_bound_without_sn():
    for out_features, in_features in SHAPES:
        layer = positive_diagonal_linear(in_features, out_features,
                                         bias=out_features != 1)
        assert isinstance(layer, torch.nn.Linear)
        assert layer.weight.shape == (out_features, in_features)
        assert (layer.weight >= 0).all()
        assert _norm(layer.weight) <= 1. + 4e-6
        assert not any(name.endswith(("._u", "._v")) for name in layer.state_dict())
