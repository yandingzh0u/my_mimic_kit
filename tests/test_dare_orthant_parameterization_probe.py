import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_orthant_metric_probe import OrthantReward
from dare_orthant_parameterization_probe import ProjectedReward, optimizer_for, audit


GROUPS = [('position', [0, 4, 5]), ('pose', [1, 2, 3, 6]), ('velocity', [7, 8])]


def test_projected_and_softplus_functions_start_identical():
    diff = torch.randn(32, 9)
    for symmetric in (False, True):
        softplus = OrthantReward(GROUPS, symmetric, False)
        projected = ProjectedReward(GROUPS, symmetric)
        torch.testing.assert_close(softplus.reward(diff), projected.reward(diff), rtol=0, atol=0)
        assert sum(p.numel() for p in softplus.parameters()) == sum(p.numel() for p in projected.parameters())


def test_projection_and_zero_input_gradient_remain_finite():
    model = ProjectedReward(GROUPS, False)
    optimizer = optimizer_for(model)
    diff = torch.cat((torch.zeros(4, 9), torch.randn(28, 9)))
    model.loss(diff).backward()
    optimizer.step()
    with torch.no_grad():
        model.metrics[0].positive[0, 0] = -1
    model.project()
    assert model.metrics[0].positive[0, 0] == 0
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.cost(torch.zeros(8, 9)).eq(0).all()
    assert audit(model, diff)['zero_cost_nonzero_input_fraction'] == 0


def test_equal_regularization_on_effective_geometry_not_log_coordinates():
    for model in (OrthantReward(GROUPS, False, False), ProjectedReward(GROUPS, False)):
        groups = optimizer_for(model).param_groups
        assert groups[0]['weight_decay'] == 0
        assert groups[1]['weight_decay'] == 1e-4


def test_initial_softplus_effective_gradient_chain_is_verified():
    source = OrthantReward(GROUPS, False, False).double()
    direct = ProjectedReward(GROUPS, False).double()
    with torch.no_grad():
        for left, right in zip(source.metrics, direct.metrics):
            right.positive.copy_(torch.nn.functional.softplus(left.positive))
            right.negative.copy_(torch.nn.functional.softplus(left.negative))
    diff = torch.randn(32, 9, dtype=torch.double)
    torch.testing.assert_close(source.reward(diff), direct.reward(diff), rtol=0, atol=0)
    source.loss(diff).backward()
    direct.loss(diff).backward()
    for left, right in zip(source.metrics, direct.metrics):
        derivative = left.positive.sigmoid()
        torch.testing.assert_close(left.positive.grad, derivative * right.positive.grad,
                                   rtol=1e-10, atol=1e-12)
