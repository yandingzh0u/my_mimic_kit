"""Contracts for independent screening, not a PBHC training reproduction."""
from pathlib import Path
import sys

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
sys.path.insert(0, str(ROOT / "tools/diagnostics"))
from kungfubot_tolerance_stress import (
    AdaptiveTolerance, FrozenMapping, gap_reward, normalization_cancellation,
)


def test_ema_and_min_match_published_origin_update():
    controller = AdaptiveTolerance(2.)
    controller.update(torch.tensor([0., 2.]))
    assert controller.ema == pytest.approx(1.999)
    assert controller.sigma == pytest.approx(1.999)
    previous = controller.sigma
    controller.update(torch.tensor([100., 100.]))
    assert controller.ema > previous
    assert controller.sigma == previous
    assert controller.steps == 2


@pytest.mark.parametrize("initial,alpha", [(0., .1), (-1., .1), (float('nan'), .1), (1., 0.), (1., 1.1)])
def test_invalid_controller_is_rejected(initial, alpha):
    with pytest.raises(ValueError):
        AdaptiveTolerance(initial, alpha)


def test_reward_formula_monotonicity_finiteness_and_anchor_clipping():
    logits = torch.tensor([-1e6, -2., 0., 1., 10.], dtype=torch.double)
    reward = gap_reward(logits, 1., 2.)
    torch.testing.assert_close(reward, 2 * torch.exp(-(1 - logits).clamp_min(0) / 2))
    assert torch.isfinite(reward).all()
    assert (reward[1:] >= reward[:-1]).all()
    assert reward[-1] == reward[-2] == 2
    with pytest.raises(ValueError):
        gap_reward(logits, 1., 0.)


def test_scaling_before_refit_mean_absolute_normalizer_cancels():
    torch.manual_seed(9)
    result = normalization_cancellation(torch.randn(512, 172, dtype=torch.double))
    assert result['normalized_input_abs_max'] < 1e-12


class ToyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.))

    def logits(self, pairs):
        return self.scale * (1 - pairs[:, 0].square().sum(-1))


def test_frozen_scorer_does_not_update_controller_from_evaluation_data():
    model = ToyModel()
    mapping = FrozenMapping(model, 2.)
    inputs = torch.randn(7, 2, 172)
    original = mapping.reward(inputs)
    mapping.reward(inputs * 100)
    torch.testing.assert_close(original, mapping.reward(inputs))
    assert mapping.sigma == 2.


def test_gap_derivative_includes_inverse_tolerance():
    logits = torch.tensor([-2., 0.], dtype=torch.double, requires_grad=True)
    reward = gap_reward(logits, 1., 2.)
    actual = torch.autograd.grad(reward.sum(), logits)[0]
    torch.testing.assert_close(actual, reward.detach() / 2.)
