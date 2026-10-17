"""Actual DARE trajectory gradients, not linear-logit assumptions."""
from pathlib import Path
import json
import sys

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_preference_distance_audit import flat_gradient, pair_graph, summarize
from dare_preference_stress import discounted_score
from learning.add_agent import calc_unscaled_disc_reward


class TinyReward(nn.Module):
    def __init__(self, spectral=False):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(3, 5), nn.Tanh())
        self.head = nn.Linear(5, 1)
        if spectral:
            nn.utils.parametrizations.spectral_norm(self.head)


def test_effective_head_jacobian_includes_reward_derivative_and_bias():
    torch.manual_seed(2)
    model = TinyReward().double().eval()
    inputs = torch.randn(2, 3, 4, 3, dtype=torch.double)
    weights = torch.tensor([[1., .9, .81, 0.], [1., .9, 0., 0.], [1., .9, .81, .729]], dtype=torch.double)
    gap, effective, _ = pair_graph(model, inputs, weights)
    actual = flat_gradient(gap, tuple(model.head.parameters()))
    torch.testing.assert_close(effective, actual, rtol=1e-12, atol=1e-12)
    assert actual[-1].abs() > 0  # A nonlinear reward means bias need not cancel.


def test_actual_gap_gradient_matches_finite_difference_with_spectral_head():
    torch.manual_seed(3)
    model = TinyReward(spectral=True).double().eval()
    inputs = torch.randn(2, 2, 4, 3, dtype=torch.double)
    weights = torch.ones(2, 4, dtype=torch.double)
    params = tuple(model.parameters())
    gap, _, _ = pair_graph(model, inputs, weights)
    analytic = flat_gradient(gap, params)
    direction = torch.randn_like(analytic)
    direction /= direction.norm()
    originals = [p.detach().clone() for p in params]
    def displaced(sign):
        offset = 0
        with torch.no_grad():
            for p, original in zip(params, originals):
                p.copy_(original + sign * 1e-5 * direction[offset:offset+p.numel()].reshape_as(p))
                offset += p.numel()
        value, _, _ = pair_graph(model, inputs, weights)
        return value.detach()
    difference = (displaced(1) - displaced(-1)) / 2e-5
    with torch.no_grad():
        for p, original in zip(params, originals):
            p.copy_(original)
    torch.testing.assert_close(analytic @ direction, difference, atol=1e-8, rtol=1e-6)


def test_masked_nan_does_not_enter_score_or_gradient():
    torch.manual_seed(4)
    model = TinyReward().double().eval()
    inputs = torch.randn(2, 2, 4, 3, dtype=torch.double)
    weights = torch.tensor([[1., .9, 0., 0.], [1., .9, .81, 0.]], dtype=torch.double)
    good, effective, distance = pair_graph(model, inputs, weights)
    poisoned = inputs.clone()
    poisoned[:, weights == 0] = float("nan")
    actual, actual_effective, actual_distance = pair_graph(model, poisoned, weights)
    torch.testing.assert_close(actual, good)
    torch.testing.assert_close(actual_effective, effective)
    torch.testing.assert_close(actual_distance, distance)


def test_score_is_original_capped_reward_and_not_time_average():
    model = TinyReward().double().eval()
    inputs = torch.randn(2, 2, 4, 3, dtype=torch.double)
    weights = torch.ones(2, 4, dtype=torch.double)
    with torch.no_grad():
        model.head.weight.zero_()
        model.head.bias.fill_(20.)
    gap, effective, _ = pair_graph(model, inputs, weights)
    torch.testing.assert_close(gap, torch.zeros_like(gap))
    torch.testing.assert_close(effective, torch.zeros_like(effective))
    rewards = 2 * calc_unscaled_disc_reward(torch.zeros(2, 2, 4, dtype=torch.double))
    value = discounted_score(rewards, weights[None])
    torch.testing.assert_close(value, torch.full((2,), 8 * torch.log(torch.tensor(2.)).item(), dtype=torch.double), rtol=1e-7, atol=1e-7)


def test_empty_misranked_partition_is_reported_not_fabricated():
    result = summarize([dict(gap=1., agrees=True, full_gap_gradient_norm=1.)])
    assert result["misranked"]["gap"]["count"] == 0
    assert result["misranked_in_bottom_full_gradient_quartile"]["total"] == 0
    json.dumps(result)
    json.dumps(summarize([dict(gap=-1., agrees=False, full_gap_gradient_norm=.1)]))
