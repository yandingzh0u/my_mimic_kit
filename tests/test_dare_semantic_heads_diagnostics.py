import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_semantic_heads_probe import SemanticHeads, calc_unscaled_disc_reward


def model():
    groups = [(f"group{i}", [i]) for i in range(7)]
    return SemanticHeads(groups, parameter_budget=250000)


def test_each_semantic_score_depends_only_on_its_own_error_group():
    disc = model().eval()
    x = torch.randn(5, 7)
    before = disc.logits(x)
    changed = x.clone()
    changed[:, 1:] += 1
    after = disc.logits(changed)
    torch.testing.assert_close(before[:, 0], after[:, 0], atol=0, rtol=0)


def test_semantic_head_input_gradient_excludes_other_groups():
    disc = model().eval()
    x = torch.randn(5, 7, requires_grad=True)
    gradient = torch.autograd.grad(disc.logits(x)[:, 0].sum(), x)[0]
    assert torch.isfinite(gradient).all()
    assert torch.count_nonzero(gradient[:, 1:]) == 0


def test_semantic_reward_uses_same_bounded_reward_function_for_all_groups():
    disc = model().eval()
    x = torch.randn(5, 7)
    expected = (2 * calc_unscaled_disc_reward(disc.logits(x))).mean(-1)
    torch.testing.assert_close(disc.reward(x), expected)
    assert disc.reward(x).shape == (5,)
    assert torch.all(disc.reward(x) >= 0)
    assert torch.all(disc.reward(x) <= -2 * torch.log(torch.tensor(1e-4)))


def test_zero_groups_are_not_trained_as_negative_examples():
    disc = model().train()
    loss = disc.loss(torch.zeros(5, 7))
    assert loss.item() == 0
    loss.backward()
    assert all(p.grad is None or torch.count_nonzero(p.grad) == 0
               for p in disc.parameters())


def test_independent_supervision_is_finite_with_partly_zero_groups():
    disc = model().train()
    x = torch.randn(5, 7)
    x[:, :2] = 0
    loss = disc.loss(x)
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all()
               for p in disc.parameters())


def test_each_bank_receives_the_original_single_discriminator_gradient():
    disc = model().eval()
    x = torch.randn(5, 7)
    parameter = disc.heads[0].bias
    joint_gradient = torch.autograd.grad(disc.loss(x), parameter)[0]
    positive = disc.logits(torch.zeros_like(x[:1]))[:, 0]
    negative = disc.logits(x)[:, 0]
    single_loss = .5 * (F.softplus(-positive).mean()
                        + F.softplus(negative).mean())
    single_gradient = torch.autograd.grad(single_loss, parameter)[0]
    torch.testing.assert_close(joint_gradient, single_gradient)
