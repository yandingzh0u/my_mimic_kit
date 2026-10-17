"""Contracts for the isolated dynamics-geometry candidate, not policy gains."""
from pathlib import Path
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_dynamics_geometry_stress import (
    BoundedLinear, StateEncoder, GeometryReward, ICVF, episode_links,
    expectile_loss, make_transition_data, sample_goals, icvf_loss,
)
from temporal_dare_stress import TemporalReward


def fixture():
    torch.set_num_threads(2)
    torch.manual_seed(10160)
    groups = [(str(i), tuple(range(24 * i, min(172, 24 * (i + 1)) if i < 6 else 172))) for i in range(7)]
    original = TemporalReward(groups).double().eval()
    encoder = StateEncoder().double()
    diff, ref = torch.randn(10, 172, dtype=torch.double), torch.randn(10, 172, dtype=torch.double)
    center, scale = torch.zeros(172, dtype=torch.double), torch.ones(172, dtype=torch.double)
    return original, encoder, diff, ref, center, scale


def test_exact_encoder_spectral_bound_and_frozen_materialization():
    torch.manual_seed(1)
    encoder = StateEncoder(7, 12, 3).double()
    with torch.no_grad():
        for m in encoder.net:
            if isinstance(m, BoundedLinear):
                m.weight.mul_(9)
    x, y = torch.randn(500, 7, dtype=torch.double), torch.randn(500, 7, dtype=torch.double)
    assert ((encoder(x) - encoder(y)).norm(dim=-1) <= (x - y).norm(dim=-1) + 1e-12).all()
    frozen = encoder.frozen_copy()
    assert all(not p.requires_grad for p in frozen.parameters())
    assert not any(isinstance(m, BoundedLinear) for m in frozen.modules())
    torch.testing.assert_close(encoder(x), frozen(x), atol=1e-12, rtol=1e-12)
    for m in frozen.net:
        if isinstance(m, nn.Linear):
            assert torch.linalg.matrix_norm(m.weight, ord=2) <= 1 + 1e-12


def test_function_preserving_lift_copies_sn_state_and_gradients():
    original, encoder, diff, ref, center, scale = fixture()
    lifted = GeometryReward(original, encoder.frozen_copy(), center, scale, "dynamics").eval()
    x = diff.clone().requires_grad_()
    expected = original.head(original.layers(x)).squeeze(-1)
    expected_grad = torch.autograd.grad(expected.sum(), x)[0]
    y, context = diff.clone().requires_grad_(), ref.clone().requires_grad_()
    actual = lifted.logits(y, context)
    grad, context_grad = torch.autograd.grad(actual.sum(), (y, context))
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-11)
    torch.testing.assert_close(grad, expected_grad, atol=1e-11, rtol=1e-11)
    assert context_grad.count_nonzero() == 0


def test_raw_coordinates_retained_and_zero_anchor_remains_context_independent():
    original, encoder, diff, ref, center, scale = fixture()
    model = GeometryReward(original, encoder.frozen_copy(), center, scale, "dynamics").eval()
    with torch.no_grad():
        model.layers.trunk[0].parametrizations.weight.original[:, -146:].normal_(0, .01)
    zeros = torch.zeros_like(diff)
    assert model.feature(zeros, ref).count_nonzero() == 0
    anchor = model.logits(zeros, ref)
    torch.testing.assert_close(anchor, anchor[:1].expand_as(anchor), atol=1e-12, rtol=1e-12)
    seen = []
    h = model.layers.register_forward_pre_hook(lambda m, a: seen.append(a[0].detach()))
    model.logits(diff, ref)
    h.remove()
    torch.testing.assert_close(seen[0][:, :172], diff)


def test_episode_links_keep_terminal_next_but_do_not_cross_reset_or_censor():
    # Two environments, six time slices, reset env0 before slice3, env1 before slice2.
    ep = torch.tensor([[0, 0], [0, 0], [0, 1], [1, 1], [1, 1], [1, 1]]).flatten()
    links = episode_links(ep, 2)
    assert torch.equal(ep[links["current"]], ep[links["next"]])
    assert 2 in links["current"].tolist()  # env0 slice1 -> terminal slice2 retained
    idx = links["current"].tolist().index(2)
    assert bool(links["terminal"][idx])
    assert not (links["next"] >= 10).any()  # final status unknown
    assert links["end"][0].item() == 4
    assert links["end"][1].item() == 3
    with pytest.raises(ValueError):
        episode_links(torch.arange(7), 2)


def test_future_goals_stay_in_recorded_episode_and_data_chunks(monkeypatch):
    ep = torch.tensor([[0, 0], [0, 0], [1, 0], [1, 1], [1, 1]]).flatten()
    d = dict(episode=ep, diff=torch.ones(10, 3), reference=torch.ones(10, 3) * 4)
    data = make_transition_data([d, d], torch.ones(3) * 2, torch.ones(3), 2, torch.ones(3) * 4)
    # (reference-center)/target_scale - cached_diff*cached_scale/target_scale
    torch.testing.assert_close(data["states"], torch.full((20, 3), -.5))
    assert (data["end"][:10] < 10).all() and (data["end"][10:] >= 10).all()
    # Direct deterministic future range, independent of random-global goals.
    for sid in range(20):
        end = data["end"][sid].item()
        for goal in range(sid, end + 1, 2):
            assert ep[sid % 10] == ep[goal % 10]
    sampled = sample_goals(data, torch.arange(20).repeat(10), torch.Generator().manual_seed(1))
    assert sampled.min() >= 0 and sampled.max() < 20
    # Force the future-goal branch, not the intentionally global random arm.
    monkeypatch.setattr(torch, "rand", lambda *shape, **kw: torch.full(shape, .5, device=kw.get("device")))
    starts = torch.arange(20).repeat(10)
    future = sample_goals(data, starts, torch.Generator().manual_seed(3))
    assert ((future - starts) % 2 == 0).all()
    assert (future >= starts).all() and (future <= data["end"][starts]).all()
    assert torch.equal(ep[starts % 10], ep[future % 10])


def test_bce_updates_extra_branch_but_cannot_change_frozen_representation():
    original, encoder, diff, ref, center, scale = fixture()
    model = GeometryReward(original, encoder.frozen_copy(), center, scale, "dynamics").train()
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    opt = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.01)
    for _ in range(2):
        opt.zero_grad(set_to_none=True)
        model.loss(diff, ref).backward()
        assert model.layers.trunk[0].parametrizations.weight.original.grad[:, -146:].abs().max() > 0
        opt.step()
    for k, value in model.encoder.state_dict().items():
        torch.testing.assert_close(value, before[k], atol=0, rtol=0)
    assert all(p.grad is None for p in model.encoder.parameters())


def test_bce_only_control_really_updates_the_same_bounded_encoder():
    original, encoder, diff, ref, center, scale = fixture()
    model = GeometryReward(original, encoder, center, scale, "bce_only").train()
    assert all(p.requires_grad for p in model.encoder.parameters())
    before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
    opt = torch.optim.SGD(model.parameters(), lr=.01)
    for _ in range(3):
        opt.zero_grad(set_to_none=True)
        model.loss(diff, ref).backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        opt.step()
    assert any(not torch.equal(v, before[k]) for k, v in model.encoder.state_dict().items())


def test_icvf_td_has_no_target_gradients_and_expectile_sign_is_correct():
    torch.manual_seed(2)
    model, old = ICVF(5, 8, 3).double(), ICVF(5, 8, 3).double()
    batch = {k: torch.randn(10, 5, dtype=torch.double) for k in ("s", "n", "g", "z")}
    batch.update(reward=torch.ones(10, dtype=torch.double), goal_reward=torch.zeros(10, dtype=torch.double),
                 terminal=torch.zeros(10, dtype=torch.double))
    icvf_loss(model, old, batch).backward()
    assert any(p.grad is not None and p.grad.abs().max() > 0 for p in model.phi.parameters())
    assert all(p.grad is None for p in old.parameters())
    value = expectile_loss(torch.zeros(2), torch.ones(2), torch.tensor([1., -1.]), .9)
    torch.testing.assert_close(value, torch.tensor(.5))


def test_checkpoint_specific_normalization_not_mature_cache_scale():
    original, encoder, diff, ref, center, scale = fixture()
    cached = scale * 3
    model = GeometryReward(original, None, center, scale, "original", cached).eval()
    expected = original.head(original.layers(3 * diff)).squeeze(-1)
    torch.testing.assert_close(model.logits(diff, ref), expected)
