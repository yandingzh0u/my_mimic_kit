"""Contracts for the diagnostic prototype, not production training changes."""
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from reference_conditioned_dare_stress import (
    KINDS, ReferenceReward, context_statistics, preservation,
)
from temporal_dare_stress import TemporalReward


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(972)
    original = TemporalReward([("root", tuple(range(3))), ("body", tuple(range(3, 172)))]).double().eval()
    diff, reference = torch.randn(12, 172, dtype=torch.double), torch.randn(12, 172, dtype=torch.double)
    mean, scale = context_statistics(reference)
    models = {kind: ReferenceReward(original, kind, mean, scale).double().eval() for kind in KINDS}
    return original, diff, reference, models


def test_all_controls_preserve_initial_function_and_input_gradients():
    original, diff, reference, models = fixture()
    values = preservation(original, models, diff, reference)
    for kind, result in values.items():
        assert result["logit_abs_max"] < 1e-7, (kind, result)
        assert result["reward_abs_max"] < 1e-7, (kind, result)
        assert result["diff_gradient_abs_max"] < 1e-7, (kind, result)
        assert result["initial_context_gradient_abs_max"] == 0, (kind, result)


def test_anchor_is_context_independent_after_nonzero_modulation():
    _, diff, reference, models = fixture()
    model = models["conditional"]
    with torch.no_grad():
        model.modulator[-1].weight.normal_(0, .1)
        model.modulator[-1].bias.normal_(0, .1)
    logits = model.logits(torch.zeros_like(diff), reference)
    torch.testing.assert_close(logits, model.anchor.expand_as(logits), atol=1e-12, rtol=1e-12)
    assert model.direction(diff, reference).norm(dim=-1).max() <= 1 + 1e-12
    assert (model.logits(diff, reference) - model.logits(diff, reference.flip(0))).abs().max() > 1e-6


def test_capacity_and_shuffled_controls_have_identical_parameter_budget():
    _, diff, reference, models = fixture()
    counts = {kind: sum(p.numel() for p in models[kind].parameters())
              for kind in ("capacity", "conditional", "shuffled")}
    assert len(set(counts.values())) == 1
    model = models["capacity"]
    with torch.no_grad():
        model.modulator[-1].weight.normal_(0, .1)
    torch.testing.assert_close(model.logits(diff, reference), model.logits(diff, reference.flip(0)))


def test_positive_and_negative_context_marginals_are_matched():
    _, diff, reference, models = fixture()
    model = models["concat"]
    seen = []
    handle = model.layers.register_forward_pre_hook(lambda _, args: seen.append(args[0].detach().clone()))
    model.loss(diff, reference).backward()
    handle.remove()
    assert len(seen) == 2
    torch.testing.assert_close(seen[0][:, 172:], seen[1][:, 172:])
    assert seen[0][:, :172].count_nonzero() == 0
    assert torch.isfinite(model.layers.encoders[0][0].parametrizations.weight.original.grad).all()


def test_constant_reference_coordinates_are_finite_and_zero_centered():
    ref = torch.randn(16, 172)
    ref[:, :3] = 4
    mean, scale = context_statistics(ref)
    normalized = (ref - mean) / scale
    assert torch.isfinite(normalized).all()
    assert normalized[:, :3].count_nonzero() == 0


def test_conditioning_branch_receives_gradients_and_learns_after_zero_initialization():
    _, diff, reference, models = fixture()
    model = models["conditional"].train()
    optimizer = torch.optim.SGD(model.parameters(), lr=.01)
    initial = model.modulator[-1].weight.detach().clone()
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = model.loss(diff, reference)
        loss.backward()
        assert model.modulator[-1].weight.grad.abs().max() > 0
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        optimizer.step()
    assert not torch.equal(initial, model.modulator[-1].weight)
    assert model.modulator[0].weight.grad.abs().max() > 0
    model.eval()
    with torch.no_grad():
        assert (model.reward(diff, reference) - model.reward(diff, reference.flip(0))).abs().max() > 0
        zero = model.logits(torch.zeros_like(diff), reference)
        torch.testing.assert_close(zero, model.anchor.expand_as(zero), atol=1e-12, rtol=1e-12)
