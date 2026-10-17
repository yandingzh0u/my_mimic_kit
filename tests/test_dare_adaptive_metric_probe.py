import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_adaptive_metric_probe import AdaptiveMetric, allocation_audit, restore_models


GROUPS = [('position', [0, 1, 2]), ('pose', [3, 4, 5, 6]), ('velocity', [7, 8])]


def test_common_initial_cost_and_rewards():
    fixed = AdaptiveMetric(GROUPS, False).double()
    learned = AdaptiveMetric(GROUPS, True).double()
    diff = torch.randn(32, 9, dtype=torch.double)
    torch.testing.assert_close(fixed.reward(diff), learned.reward(diff), rtol=0, atol=0)
    n = len(GROUPS) ** .5
    expected = sum(diff[:, ids].norm(dim=-1) / len(ids) ** .5 for _, ids in GROUPS) / (n + 1 / n)
    torch.testing.assert_close(learned.cost(diff), expected)


def test_learned_metric_remains_invertible_nonexpansive_and_group_monotone():
    model = AdaptiveMetric(GROUPS).double()
    with torch.no_grad():
        for metric in model.metrics:
            metric.diagonal.normal_()
            metric.off_diagonal.normal_()
        model.coefficients.normal_()
    diff = torch.randn(64, 9, dtype=torch.double)
    original = model.reward(diff)
    for _, ids in GROUPS:
        changed = diff.clone()
        changed[:, ids] *= .5
        assert (model.reward(changed) > original).all()
    for metric in model.metrics:
        singular_values = torch.linalg.svdvals(metric.weight())
        assert singular_values[-1] > 0
        assert singular_values[0] <= 1 + 1e-10
    assert model.weights().norm() <= 1 + 1e-10


def test_anisotropy_is_representable_without_a_fixed_relative_slope_floor():
    model = AdaptiveMetric(GROUPS)
    with torch.no_grad():
        model.metrics[1].diagonal[0] = 5
        model.coefficients[0] = -8
        model.coefficients[1] = 3
    a = torch.zeros(1, 9)
    b = torch.zeros(1, 9)
    a[0, 3] = 1
    b[0, 4] = 1
    assert model.cost(a) > model.cost(b)
    assert model.weights()[1] / model.weights()[0] > 1000


def test_zero_and_nonzero_bce_gradients_are_finite():
    model = AdaptiveMetric(GROUPS)
    assert model.cost(torch.zeros(8, 9)).eq(0).all()
    diff = torch.cat((torch.zeros(4, 9), torch.randn(28, 9)))
    loss = model.loss(diff)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())


def test_weight_continuation_preserves_both_maps_and_sources(tmp_path):
    source = {name: AdaptiveMetric(GROUPS, learn)
              for name, learn in [('adaptive_rms', False), ('full_metric', True)]}
    with torch.no_grad():
        source['full_metric'].metrics[1].off_diagonal.fill_(.2)
        source['adaptive_rms'].anchor.fill_(.5)
    paths = {}
    for name, model in source.items():
        path = tmp_path / f'seed5_{name}.pt'
        torch.save(model.state_dict(), path)
        paths[name] = path.read_bytes()
    restored = {name: AdaptiveMetric(GROUPS, learn)
                for name, learn in [('adaptive_rms', False), ('full_metric', True)]}
    hashes = restore_models(restored, tmp_path, 5)
    diff = torch.randn(32, 9)
    for name in source:
        torch.testing.assert_close(restored[name].reward(diff), source[name].reward(diff), rtol=0, atol=0)
        assert len(hashes[name]) == 64
        assert (tmp_path / f'seed5_{name}.pt').read_bytes() == paths[name]


def test_allocation_marginal_matches_binary_loss_derivative():
    model = AdaptiveMetric(GROUPS).double()
    diff = torch.randn(64, 9, dtype=torch.double)
    weights = model.weights().detach().requires_grad_()
    loss = .5 * (torch.nn.functional.softplus(-model.anchor)
        + torch.nn.functional.softplus(model.anchor - (model.components(diff) * weights).sum(-1)).mean())
    expected, = torch.autograd.grad(loss, weights)
    audit = allocation_audit(model, diff)
    measured = torch.tensor(audit['negative_loss_marginal'], dtype=torch.double)
    torch.testing.assert_close(-expected, measured, rtol=1e-12, atol=1e-12)
