import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_cpl_frontend_activation_probe import (
    PairSort, FrontendActivationCPL, activation_jvp, directional_probe,
)
from dare_shared_reference import ConvexPotentialBlock


def test_pair_sort_preserves_norm_jvp_and_lipschitz_bound():
    torch.manual_seed(3101)
    for width in [4, 5]:
        x = torch.randn(23, width, dtype=torch.double)
        other = torch.randn_like(x)
        tangent = torch.randn_like(x)
        layer = PairSort()
        actual, jvp = activation_jvp(layer, x, tangent)
        values, reference = torch.autograd.functional.jvp(layer, x, tangent)
        torch.testing.assert_close(actual, values)
        torch.testing.assert_close(jvp, reference)
        torch.testing.assert_close(actual.norm(dim=-1), x.norm(dim=-1))
        torch.testing.assert_close(jvp.norm(dim=-1), tangent.norm(dim=-1))
        assert ((layer(x) - layer(other)).norm(dim=-1)
                <= (x - other).norm(dim=-1) + 1e-12).all()


def test_frontend_replacement_keeps_full_coordinates_parameters_and_cpl_activation():
    groups = [(f'group{i}', [i, i + 7]) for i in range(7)]
    original = FrontendActivationCPL(groups, False, False).eval()
    changed = FrontendActivationCPL(groups, True, True).eval()
    changed.load_state_dict(original.state_dict(), strict=True)
    assert sum(p.numel() for p in original.parameters()) == sum(p.numel() for p in changed.parameters())
    for index, encoder in enumerate(changed.layers.encoders):
        assert isinstance(encoder[1], PairSort)
        assert getattr(changed.layers, f'group_indices_{index}').tolist() == groups[index][1]
    assert isinstance(changed.layers.trunk[1], PairSort)
    for block in changed.layers.trunk[2:]:
        assert isinstance(block, ConvexPotentialBlock)
        assert isinstance(block.activation, torch.nn.ReLU)
    diff = torch.randn(8, 14, requires_grad=True)
    changed.loss(diff).backward()
    assert torch.isfinite(diff.grad).all()
    for parameter in changed.parameters():
        assert parameter.grad is None or torch.isfinite(parameter.grad).all()


def test_directional_probe_matches_autograd_and_norm_preserving_frontend():
    groups = [(f'group{i}', [i, i + 7]) for i in range(7)]
    model = FrontendActivationCPL(groups, True, True).double().eval()
    diff = torch.randn(11, 14, dtype=torch.double)
    probe = directional_probe(model, diff)
    _, ids = groups[2]
    tangent = torch.zeros_like(diff)
    tangent[:, ids] = diff[:, ids] / diff[:, ids].norm(dim=-1, keepdim=True)
    _, reference = torch.autograd.functional.jvp(model.logits, diff, tangent)
    assert abs(probe['group2']['signed_logit_radial_derivative'] - float(reference.mean())) < 1e-10
    for entry in probe.values():
        assert abs(entry['encoder_activation_retention'] - 1) < 1e-12
        assert abs(entry['fusion_activation_retention'] - 1) < 1e-12
