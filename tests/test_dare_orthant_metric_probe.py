import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_orthant_metric_probe import OrthantMetric, OrthantReward, evaluate


GROUPS = [('position', [0, 4, 5]), ('pose', [1, 2, 3, 6]), ('velocity', [7, 8])]


def test_four_arms_have_exactly_equal_initial_maps_and_parameters():
    models = [OrthantReward(GROUPS, symmetric, independent).double()
              for symmetric in (False, True) for independent in (False, True)]
    diff = torch.randn(32, 9, dtype=torch.double)
    expected = models[0].reward(diff)
    count = sum(p.numel() for p in models[0].parameters())
    for model in models:
        torch.testing.assert_close(model.reward(diff), expected, rtol=0, atol=0)
        assert sum(p.numel() for p in model.parameters()) == count


def test_metric_is_definite_homogeneous_and_coordinate_monotone():
    metric = OrthantMetric(4, False).double()
    with torch.no_grad():
        metric.positive.normal_()
        metric.negative.normal_()
    diff = torch.randn(64, 4, dtype=torch.double)
    values = metric(diff)
    assert (values > 0).all()
    assert metric(torch.zeros_like(diff)).eq(0).all()
    torch.testing.assert_close(metric(.25 * diff), .25 * values)
    for i in range(4):
        changed = diff.clone()
        changed[:, i] *= .5
        assert (metric(changed) < values).all()
    for matrix in metric.matrices():
        assert matrix.norm() <= 1 + 1e-12


def test_symmetric_control_and_asymmetric_coordinate_response():
    symmetric = OrthantMetric(3, True).double()
    asymmetric = OrthantMetric(3, False).double()
    with torch.no_grad():
        symmetric.positive.normal_()
        symmetric.negative.normal_()
        asymmetric.load_state_dict(symmetric.state_dict())
    diff = torch.randn(32, 3, dtype=torch.double)
    torch.testing.assert_close(symmetric(diff), symmetric(-diff), rtol=0, atol=0)
    assert (asymmetric(diff) - asymmetric(-diff)).abs().max() > 1e-3


def test_independent_budget_changes_only_selected_group():
    model = OrthantReward(GROUPS, False, True)
    before = model.weights().detach().clone()
    with torch.no_grad():
        model.coefficients[0] = 5
    torch.testing.assert_close(model.weights()[1:], before[1:], rtol=0, atol=0)


def test_bce_gradients_and_zero_input_are_finite():
    model = OrthantReward(GROUPS, False, False)
    diff = torch.cat((torch.zeros(4, 9), torch.randn(28, 9)))
    model.loss(diff).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.cost(torch.zeros(8, 9)).eq(0).all()
    audit = evaluate(model, diff)
    assert all(group['negative_fraction'] == 0 for group in audit['groups'])
    assert all(group['coordinate_negative_fraction'] == 0 for group in audit['groups'])
