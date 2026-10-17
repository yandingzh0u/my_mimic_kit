import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/diagnostics'))
from dare_monotone_cpl_probe import CertifiedCPL, MonotoneCPL, PooledCPL, certified_weight
from dare_monotone_formal_rescore import FrozenFormalDisc, GROUP_NAMES
from dare_monotone_replay_coverage import audit
from learning.monotone_cpl import AnchorLogit, MonotoneGroupCost


def groups():
    return [('root', [0, 1, 2]), ('pose', [3, 4, 5, 6]), ('velocity', [7, 8])]


def test_certified_weight_has_true_spectral_norm_below_one():
    for shape in [(1, 8), (8, 3), (16, 16)]:
        weight = certified_weight(torch.randn(*shape) * 10)
        assert torch.linalg.matrix_norm(weight, 2) <= 1.00001


def test_cpl_jacobian_is_nonexpansive_with_certified_weights():
    block = CertifiedCPL(8).double()
    x = torch.randn(8, dtype=torch.double)
    jac = torch.autograd.functional.jacobian(block, x)
    assert torch.linalg.matrix_norm(jac, 2) <= 1.00001


def test_group_cost_has_nonzero_lower_response_bound():
    model = MonotoneCPL(groups(), width=8).double()
    errors = torch.rand(64, 3, dtype=torch.double, requires_grad=True) * 5
    cost = model.cost(errors)
    partial = torch.autograd.grad(cost.sum(), errors)[0]
    n = 3 ** .5
    assert (partial >= (1 - 1 / n) / (n + 1 / n) - 1e-8).all()
    assert (partial <= (1 + 1 / n) / (n + 1 / n) + 1e-8).all()


def test_all_group_contractions_raise_the_original_reward():
    model = MonotoneCPL(groups(), width=8).eval()
    diff = torch.randn(64, 9)
    original = model.reward(diff)
    for _, ids in groups():
        changed = diff.clone()
        changed[:, ids] *= .5
        assert (model.reward(changed) > original).all()
    assert (model.logits(diff) < model.logits(torch.zeros_like(diff))).all()


def test_anchor_is_a_learned_bce_logit_not_a_fixed_reward_target():
    model = MonotoneCPL(groups(), width=8)
    diff = torch.zeros(4, 9)
    torch.testing.assert_close(model.logits(diff), model.anchor.expand(4, 1))
    loss = model.loss(torch.randn(16, 9))
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert model.anchor.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.blocks.parameters())


def test_pooled_control_has_identical_learnable_parameters():
    monotone = MonotoneCPL(groups(), width=8)
    pooled = PooledCPL(groups(), width=8)
    pooled.load_state_dict(monotone.state_dict())
    assert sum(p.numel() for p in monotone.parameters()) == sum(p.numel() for p in pooled.parameters())


def frozen_formal_model():
    groups = [(name, [i, i + 7]) for i, name in enumerate(GROUP_NAMES)]
    cost = MonotoneGroupCost(groups, width=8, depth=4)
    head = AnchorLogit()
    state = {f'_model._disc_layers.{key}': value for key, value in cost.state_dict().items()}
    state['_model._disc_logits.bias'] = head.bias.detach().clone()
    state['_disc_obs_norm._mean_abs'] = torch.ones(14)
    state['_disc_obs_norm._count'] = torch.tensor([12])
    return FrozenFormalDisc(state)


def test_replay_audit_is_read_only_and_checks_every_nonzero_group():
    model = frozen_formal_model()
    diff = torch.ones(16, 14)
    before = {key: value.clone() for key, value in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    result = audit(model, diff)
    assert len(result['groups']) == 7
    for row in result['groups']:
        assert row['eligible_rows'] == 16
        assert row['gain_mean'] > 0
        assert row['negative_fraction'] == 0
        assert row['logit_bound_violations'] == 0
    torch.testing.assert_close(diff, torch.ones_like(diff))
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key])


def test_replay_audit_does_not_call_perfect_tracking_a_missing_signal():
    result = audit(frozen_formal_model(), torch.zeros(4, 14))
    for row in result['groups']:
        assert row['eligible_rows'] == 0
        assert row['gain_mean'] is None
        assert row['negative_fraction'] is None
        assert row['logit_bound_violations'] == 0
