"""Numerical checks of the coordinate interaction's mathematical contract."""

import pytest
import torch

from learning.coordinate_disc import CoordinateInteractionEncoder


def _explicit_interaction(tokens):
    difference = tokens.unsqueeze(-3) - tokens.unsqueeze(-2)
    affinity = torch.rsqrt(1 + difference.square().sum(dim=-1))
    return tokens + (affinity.unsqueeze(-1) * difference).sum(dim=-2) / (
        2 * tokens.shape[-2])


def test_interaction_matches_explicit_formula_and_gradients():
    torch.manual_seed(13)
    encoder = CoordinateInteractionEncoder(7, 3).double()
    tokens = torch.randn(5, 7, 3, dtype=torch.double, requires_grad=True)
    actual = encoder.interact(tokens)
    expected = _explicit_interaction(tokens)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    probe = torch.randn_like(tokens)
    actual_grad = torch.autograd.grad((actual * probe).sum(), tokens)[0]
    expected_grad = torch.autograd.grad((expected * probe).sum(), tokens)[0]
    torch.testing.assert_close(actual_grad, expected_grad,
                               rtol=1e-11, atol=1e-12)


def test_interaction_gradcheck_includes_affinity_derivative():
    encoder = CoordinateInteractionEncoder(3, 2).double()
    tokens = torch.randn(2, 3, 2, dtype=torch.double, requires_grad=True)
    assert torch.autograd.gradcheck(encoder.interact, (tokens,),
                                    eps=1e-6, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("offset", [0.0, 3.5])
def test_identical_tokens_preserve_values_with_correct_finite_derivative(offset):
    encoder = CoordinateInteractionEncoder(4, 3).double()
    tokens = torch.full((2, 4, 3), offset, dtype=torch.double,
                        requires_grad=True)
    actual = encoder.interact(tokens)
    torch.testing.assert_close(actual, tokens, rtol=0, atol=0)
    probe = torch.randn_like(tokens)
    gradient = torch.autograd.grad((actual * probe).sum(), tokens)[0]
    # At equal tokens the pair Hessians are I: J = .5 I + .5 mean.
    expected_gradient = 0.5 * probe + 0.5 * probe.mean(dim=-2, keepdim=True)
    torch.testing.assert_close(gradient, expected_gradient,
                               rtol=1e-12, atol=1e-12)
    assert torch.isfinite(gradient).all()


def test_complete_jacobian_is_symmetric_with_eigenvalues_in_half_to_one():
    torch.manual_seed(10)
    encoder = CoordinateInteractionEncoder(4, 3).double()
    tokens = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    jacobian = torch.autograd.functional.jacobian(
        encoder.interact, tokens).reshape(12, 12)
    torch.testing.assert_close(jacobian, jacobian.T, rtol=1e-11, atol=1e-12)
    eigenvalues = torch.linalg.eigvalsh(jacobian)
    assert eigenvalues.min() >= 0.5 - 1e-12
    assert eigenvalues.max() <= 1.0 + 1e-12
    # A common translation of every token is an unchanged direction.
    translations = torch.eye(3, dtype=torch.double).repeat(4, 1)
    torch.testing.assert_close(jacobian @ translations, translations,
                               rtol=1e-11, atol=1e-12)


def test_global_distance_bounds_at_actual_coordinate_count():
    torch.manual_seed(21)
    encoder = CoordinateInteractionEncoder(172, 6).double()
    for scale in (0.0, 0.01, 1.0, 100.0):
        first = scale * torch.randn(3, 172, 6, dtype=torch.double)
        second = scale * torch.randn_like(first) + torch.randn_like(first)
        before = (first - second).flatten(1).norm(dim=-1)
        after = (encoder.interact(first)
                 - encoder.interact(second)).flatten(1).norm(dim=-1)
        assert torch.all(after >= 0.5 * before - 1e-10)
        assert torch.all(after <= before + 1e-10)


def test_unit_embedding_retains_lengths_and_norms_after_optimizer_updates():
    torch.manual_seed(17)
    encoder = CoordinateInteractionEncoder(11, 6).double()
    optimizer = torch.optim.SGD(encoder.parameters(), lr=0.2, momentum=0.9)
    original_weight = encoder.raw_weight.detach().clone()
    first = torch.randn(8, 11, dtype=torch.double)
    second = torch.randn_like(first)
    for _ in range(3):
        optimizer.zero_grad()
        output = encoder(first)
        output.square().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in encoder.parameters())
        optimizer.step()
        torch.testing.assert_close(encoder.unit_weight().norm(dim=-1),
                                   torch.ones(11, dtype=torch.double),
                                   rtol=1e-12, atol=1e-12)
    assert not torch.equal(original_weight, encoder.raw_weight)
    token_difference = (first - second).unsqueeze(-1) * encoder.unit_weight()
    torch.testing.assert_close(token_difference.flatten(1).norm(dim=-1),
                               (first - second).norm(dim=-1),
                               rtol=1e-12, atol=1e-12)
    encoded_distance = (encoder(first) - encoder(second)).norm(dim=-1)
    input_distance = (first - second).norm(dim=-1)
    assert torch.all(encoded_distance >= 0.5 * input_distance - 1e-10)
    assert torch.all(encoded_distance <= input_distance + 1e-10)


def test_exact_zero_raw_row_has_unit_fallback_and_finite_gradients():
    encoder = CoordinateInteractionEncoder(3, 6).double()
    with torch.no_grad():
        encoder.raw_weight[0].zero_()
    torch.testing.assert_close(encoder.unit_weight()[0],
                               torch.tensor([1., 0., 0., 0., 0., 0.],
                                            dtype=torch.double))
    encoder(torch.randn(2, 3, dtype=torch.double)).square().sum().backward()
    assert torch.isfinite(encoder.raw_weight.grad).all()
    assert torch.isfinite(encoder.bias.grad).all()


@pytest.mark.parametrize("scale", [1e-30, 1e30])
def test_unit_weights_retain_nonzero_direction_at_extreme_float32_scales(scale):
    encoder = CoordinateInteractionEncoder(2, 6)
    direction = torch.arange(1., 7.)
    with torch.no_grad():
        encoder.raw_weight[:] = scale * direction
    expected = direction / direction.norm()
    effective = encoder.unit_weight()
    torch.testing.assert_close(effective, expected.expand_as(effective),
                               rtol=2e-7, atol=2e-7)
    assert torch.isfinite(effective).all()
    encoder(torch.ones(1, 2)).sum().backward()
    assert torch.isfinite(encoder.raw_weight.grad).all()


def test_chunking_preserves_values_gradients_and_batch_shapes():
    torch.manual_seed(19)
    encoder = CoordinateInteractionEncoder(5, 3).double()
    tokens = torch.randn(2, 4, 5, 3, dtype=torch.double, requires_grad=True)
    encoder._interaction_batch_size = 2
    chunked = encoder.interact(tokens)
    encoder._interaction_batch_size = 100
    complete = encoder.interact(tokens)
    torch.testing.assert_close(chunked, complete, rtol=1e-12, atol=1e-12)
    chunked_grad = torch.autograd.grad(chunked.square().sum(), tokens)[0]
    complete_grad = torch.autograd.grad(complete.square().sum(), tokens)[0]
    torch.testing.assert_close(chunked_grad, complete_grad,
                               rtol=1e-12, atol=1e-12)
    with torch.no_grad():
        inference = encoder.interact(tokens)
    torch.testing.assert_close(inference, complete.detach())
    assert encoder(torch.randn(2, 4, 5, dtype=torch.double)).shape == (2, 4, 15)
    assert encoder(torch.empty(0, 5, dtype=torch.double)).shape == (0, 15)


def test_centering_preserves_common_translations():
    torch.manual_seed(25)
    encoder = CoordinateInteractionEncoder(5, 3).double()
    tokens = torch.randn(2, 5, 3, dtype=torch.double)
    translation = torch.full((2, 1, 3), 1e6, dtype=torch.double)
    torch.testing.assert_close(encoder.interact(tokens + translation),
                               encoder.interact(tokens) + translation,
                               rtol=0, atol=3e-10)


@pytest.mark.parametrize("shape", [(2, 4, 6), (2, 5, 4)])
def test_interaction_rejects_incorrect_token_shape(shape):
    encoder = CoordinateInteractionEncoder(5, 6)
    with pytest.raises(ValueError, match="tokens must end"):
        encoder.interact(torch.randn(*shape))
