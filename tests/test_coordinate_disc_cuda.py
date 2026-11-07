"""CUDA correctness checks for the exact fused coordinate interaction."""

import warnings

import pytest
import torch

pytest.importorskip("triton")
from learning.coordinate_disc import CoordinateInteractionEncoder
from learning.coordinate_disc_cuda import interact_cuda


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="CUDA is required for fused interaction")


def _explicit(tokens):
    difference = tokens.unsqueeze(-2) - tokens.unsqueeze(-3)
    affinity = torch.rsqrt(1 + difference.square().sum(dim=-1))
    return tokens - (affinity.unsqueeze(-1) * difference).sum(dim=-2) / (
        2 * tokens.shape[-2])


@pytest.mark.parametrize("count", [1, 7, 172])
@pytest.mark.parametrize("scale", [0.01, 1.0, 100.0])
def test_fused_values_and_gradients_match_float64_reference(count, scale):
    torch.manual_seed(14)
    tokens = (scale * torch.randn(2, count, 6, device="cuda")).requires_grad_()
    probe = torch.randn_like(tokens)
    reference_tokens = tokens.detach().double().requires_grad_()
    reference = _explicit(reference_tokens)
    actual = interact_cuda(tokens)
    actual_grad = torch.autograd.grad(actual, tokens, probe)[0]
    expected_grad = torch.autograd.grad(reference, reference_tokens,
                                        probe.double())[0]
    torch.testing.assert_close(actual.double(), reference, rtol=2e-6,
                               atol=4e-7 * max(1, scale))
    torch.testing.assert_close(actual_grad.double(), expected_grad,
                               rtol=2e-6, atol=4e-7)


def test_noncontiguous_inputs_and_output_gradients():
    torch.manual_seed(23)
    tokens = torch.randn(2, 7, 12, device="cuda")[..., ::2].requires_grad_()
    probe = torch.randn(2, 6, 7, device="cuda").transpose(-1, -2)
    assert not tokens.is_contiguous() and not probe.is_contiguous()
    actual = interact_cuda(tokens)
    expected = _explicit(tokens)
    actual_grad = torch.autograd.grad(actual, tokens, probe)[0]
    expected_grad = torch.autograd.grad(expected, tokens, probe)[0]
    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=4e-7)
    torch.testing.assert_close(actual_grad, expected_grad, rtol=2e-6, atol=4e-7)


@pytest.mark.parametrize("offset", [0.0, 3.5])
def test_repeated_tokens_have_correct_finite_derivative(offset):
    tokens = torch.full((2, 7, 6), offset, device="cuda", requires_grad=True)
    actual = interact_cuda(tokens)
    torch.testing.assert_close(actual, tokens, rtol=0, atol=0)
    probe = torch.randn_like(tokens)
    gradient = torch.autograd.grad(actual, tokens, probe)[0]
    expected = 0.5 * probe + 0.5 * probe.mean(dim=-2, keepdim=True)
    torch.testing.assert_close(gradient, expected, rtol=2e-6, atol=4e-7)
    assert torch.isfinite(gradient).all()


def test_fused_global_bounds_and_local_jacobian():
    torch.manual_seed(3)
    first = torch.randn(2, 172, 6, device="cuda")
    second = torch.randn_like(first)
    before = (first - second).flatten(1).norm(dim=-1)
    after = (interact_cuda(first) - interact_cuda(second)).flatten(1).norm(dim=-1)
    assert torch.all(after >= 0.5 * before - 2e-5)
    assert torch.all(after <= before + 2e-5)
    tokens = torch.randn(3, 6, device="cuda", requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(interact_cuda, tokens)
    jacobian = jacobian.reshape(18, 18).double().cpu()
    torch.testing.assert_close(jacobian, jacobian.T, rtol=2e-6, atol=2e-7)
    eigenvalues = torch.linalg.eigvalsh(jacobian)
    assert eigenvalues.min() >= 0.5 - 3e-7
    assert eigenvalues.max() <= 1.0 + 3e-7


def test_second_derivatives_match_explicit_autograd():
    torch.manual_seed(41)
    # Noncontiguous input also checks that the saved contiguous conversion
    # remains connected to the original input for higher derivatives.
    tokens = torch.randn(2, 3, 12, device="cuda")[..., ::2].requires_grad_()
    direction = torch.randn_like(tokens)
    actual_grad = torch.autograd.grad(interact_cuda(tokens).square().sum(),
                                     tokens, create_graph=True)[0]
    actual_hvp = torch.autograd.grad(actual_grad, tokens, direction)[0]
    expected_grad = torch.autograd.grad(_explicit(tokens).square().sum(),
                                       tokens, create_graph=True)[0]
    expected_hvp = torch.autograd.grad(expected_grad, tokens, direction)[0]
    torch.testing.assert_close(actual_grad, expected_grad, rtol=3e-6, atol=5e-7)
    torch.testing.assert_close(actual_hvp, expected_hvp, rtol=5e-6, atol=1e-6)


def test_numerical_first_and_second_derivatives():
    torch.manual_seed(71)
    tokens = torch.randn(1, 2, 6, device="cuda", requires_grad=True)
    # The production kernel is float32; larger perturbations are necessary
    # for finite differences. Float64 reference comparisons above are tighter.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Input #.*double precision.*")
        assert torch.autograd.gradcheck(interact_cuda, (tokens,), eps=2e-3,
                                        atol=3e-4, rtol=3e-3)
        assert torch.autograd.gradgradcheck(interact_cuda, (tokens,), eps=2e-3,
                                            atol=5e-4, rtol=5e-3)


def test_empty_batch_and_leading_dimensions():
    empty = torch.empty(0, 172, 6, device="cuda", requires_grad=True)
    output = interact_cuda(empty)
    assert output.shape == empty.shape
    output.sum().backward()
    assert empty.grad.shape == empty.shape
    tokens = torch.randn(2, 3, 7, 6, device="cuda")
    torch.testing.assert_close(interact_cuda(tokens), _explicit(tokens),
                               rtol=2e-6, atol=4e-7)


def test_device_initialization_preserves_weights_and_random_state():
    encoder = CoordinateInteractionEncoder(172, 6)
    expected = {key: value.clone() for key, value in encoder.state_dict().items()}
    cpu_rng = torch.random.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state().clone()
    assert encoder.cuda() is encoder
    for key, value in encoder.state_dict().items():
        torch.testing.assert_close(value.cpu(), expected[key], rtol=0, atol=0)
    assert torch.equal(torch.random.get_rng_state(), cpu_rng)
    assert torch.equal(torch.cuda.get_rng_state(), cuda_rng)
    # Moving to an unsupported CUDA dtype and back leaves initialization
    # bookkeeping out of both the parameter set and the checkpoint format.
    assert encoder.double().raw_weight.dtype == torch.float64
    assert encoder.float().raw_weight.dtype == torch.float32
    assert set(encoder.state_dict()) == set(expected)


def test_inference_matches_training_forward_without_recording_gradients():
    tokens = torch.randn(2, 172, 6, device="cuda", requires_grad=True)
    training = interact_cuda(tokens)
    with torch.no_grad():
        inference = interact_cuda(tokens)
    assert training.requires_grad
    assert not inference.requires_grad
    torch.testing.assert_close(inference, training, rtol=0, atol=0)
