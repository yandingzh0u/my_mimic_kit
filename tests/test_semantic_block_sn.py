import pytest
import torch

from learning.semantic_block_sn import SemanticBlockEqualizedLinear


GROUPS = (
    ("root", tuple(range(0, 3))),
    ("body", tuple(range(3, 15))),
    ("velocity", tuple(range(15, 19))),
)


def build_layer(out_features=24):
    torch.manual_seed(11)
    return SemanticBlockEqualizedLinear(
        in_features=19,
        out_features=out_features,
        groups=GROUPS)


def test_first_forward_equalizes_exact_block_gains_and_composite_gain():
    layer = build_layer().eval()
    layer(torch.randn(8, 19))
    block_values = layer.exact_effective_spectral_values()
    torch.testing.assert_close(
        block_values, block_values.mean().expand_as(block_values),
        rtol=3e-4, atol=2e-5)
    torch.testing.assert_close(
        layer.exact_composite_spectral_value(), torch.tensor(1.0),
        rtol=3e-4, atol=2e-5)


def test_forward_backward_is_finite_and_reaches_every_block():
    layer = build_layer().train()
    inputs = torch.randn(32, 19, requires_grad=True)
    loss = torch.square(layer(inputs)).mean()
    loss.backward()
    assert torch.isfinite(inputs.grad).all()
    assert torch.isfinite(layer.weight.grad).all()
    for _, indices in GROUPS:
        assert torch.count_nonzero(layer.weight.grad[:, indices]) > 0


def test_equal_top_gain_does_not_equalize_total_energy():
    # This is a deliberate limitation check: blocks with more orthogonal
    # coordinate directions retain more Frobenius/expected Gaussian energy.
    groups = (("small", tuple(range(2))),
              ("large", tuple(range(2, 10))))
    layer = SemanticBlockEqualizedLinear(10, 10, groups).eval()
    with torch.no_grad():
        layer.weight.copy_(torch.eye(10))
    layer(torch.randn(2, 10))
    weight = layer.normalized_weight()
    small = torch.linalg.matrix_norm(weight[:, :2], ord="fro")
    large = torch.linalg.matrix_norm(weight[:, 2:], ord="fro")
    torch.testing.assert_close(large / small, torch.tensor(2.0),
                               rtol=2e-4, atol=2e-5)


def test_invalid_partition_fails_loudly():
    with pytest.raises(ValueError, match="partition"):
        SemanticBlockEqualizedLinear(
            4, 8, (("a", (0, 1)), ("b", (1, 2))))
