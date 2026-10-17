import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_independent_metric_probe import IndependentMetric


GROUPS = [('position', [0, 1, 2]), ('pose', [3, 4, 5, 6]), ('velocity', [7, 8])]


def test_loss_is_balanced_per_present_group_not_divided_by_group_count():
    model = IndependentMetric(GROUPS).double()
    diff = torch.randn(32, 9, dtype=torch.double)
    expected = .5 * (F.softplus(-model.anchor) + F.softplus(model.logits(diff)).mean(0)).sum()
    torch.testing.assert_close(model.loss(diff), expected)


def test_exactly_tracked_groups_are_not_negative_examples():
    model = IndependentMetric(GROUPS)
    diff = torch.zeros(16, 9)
    diff[:, :3] = torch.randn(16, 3)
    model.loss(diff).backward()
    assert model.anchor.grad[0] != 0
    assert model.anchor.grad[1:].eq(0).all()
    assert model.coefficients.grad[1:].eq(0).all()


def test_sensitivity_budget_does_not_transfer_between_groups():
    model = IndependentMetric(GROUPS)
    before = model.weights().detach().clone()
    with torch.no_grad():
        model.coefficients[0] = 5
    after = model.weights()
    torch.testing.assert_close(before[1:], after[1:], rtol=0, atol=0)
    assert after[0] == 1


def test_all_group_contractions_raise_actual_aggregate_reward():
    model = IndependentMetric(GROUPS).double()
    with torch.no_grad():
        for metric in model.metrics:
            metric.diagonal.normal_()
            metric.off_diagonal.normal_()
        model.coefficients.normal_()
        model.anchor.normal_()
    diff = torch.randn(64, 9, dtype=torch.double)
    original = model.reward(diff)
    for _, ids in GROUPS:
        changed = diff.clone()
        changed[:, ids] *= .5
        assert (model.reward(changed) > original).all()
    assert torch.isfinite(model.reward(torch.zeros(8, 9, dtype=torch.double))).all()
