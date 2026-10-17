"""Transport-teacher contracts; no simulator, checkpoints, or GPU required."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_fmp_teacher_screen import (
    ZeroAnchorTransport, bounded_reward, calibration, control_pass, gate,
    physical_direction, self_duals, squared_cost,
)


def teacher(bank, epsilon=.4):
    a, b, certificate = self_duals(bank, epsilon)
    assert certificate["relative_marginal_error"] < 1e-8
    return ZeroAnchorTransport(bank[None], a[None], b[None], [epsilon]), a, b


def test_duals_and_simplified_score_equal_complete_cross_minus_two_self_terms():
    torch.set_num_threads(2)
    bank = torch.tensor([[.1, .4], [.8, -.2], [-.6, .3]], dtype=torch.float64)
    model, a, b = teacher(bank)
    x = torch.tensor([[.2, -.1], [.3, .9]], dtype=bank.dtype)
    origin = torch.zeros(1, 2, dtype=bank.dtype)

    def full(q):
        cost = squared_cost(q, bank)
        cross = -.5 * q.square().sum(-1)
        left = .4 * torch.logsumexp((a[None] - cost) / .4 - np.log(len(bank)), -1)
        right = .4 * torch.logsumexp((b[None] - cost) / .4 - np.log(len(bank)), -1)
        return cross - .5 * (left + right)

    torch.testing.assert_close(model.score(x), full(x) - full(origin), atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(model.score(origin), torch.zeros(1, dtype=bank.dtype))


def test_both_self_terms_have_correct_gradient_and_concavity():
    bank = torch.tensor([[.1, .4], [.8, -.2], [-.6, .3]], dtype=torch.float64)
    model, _, _ = teacher(bank)
    x = torch.tensor([[.2, -.1], [.3, .9]], dtype=bank.dtype, requires_grad=True)
    actual = torch.autograd.grad(model.score(x).sum(), x)[0]
    torch.testing.assert_close(actual, model.gradient(x), atol=1e-12, rtol=1e-12)
    hessian = torch.autograd.functional.hessian(lambda q: model.score(q[None]).sum(), x[0])
    assert torch.linalg.eigvalsh(hessian).max() <= 1e-10


def test_source_self_correction_is_not_radial_and_zero_is_not_necessarily_maximum():
    bank = torch.tensor([[1., 0.], [2., 0.]], dtype=torch.float64)
    model, _, _ = teacher(bank)
    origin = bank.new_zeros(1, 2)
    assert model.gradient(origin)[0, 0] < 0
    # Same norm, different sign: this is the FULL teacher, not plain distance.
    probes = bank.new_tensor([[-.001, 0.], [.001, 0.]])
    assert model.score(probes)[0] > 0 > model.score(probes)[1]


def test_dual_gauge_and_permutation_do_not_change_score():
    bank = torch.tensor([[.1], [.8], [-.6]], dtype=torch.float64)
    model, a, b = teacher(bank)
    other = ZeroAnchorTransport(bank[None], (a + 20)[None], (b - 20)[None], [.4])
    x = torch.linspace(-2, 2, 9, dtype=bank.dtype)[:, None]
    torch.testing.assert_close(model.score(x), other.score(x), atol=1e-12, rtol=1e-12)
    permuted, _, _ = teacher(bank.flip(0))
    torch.testing.assert_close(model.score(x), permuted.score(x), atol=1e-8, rtol=1e-8)


def test_calibration_is_finite_monotone_and_offset_invariant():
    score = torch.tensor([-1., 0., 2., 4.])
    fitted = calibration(score)
    reward = bounded_reward(score, fitted)
    assert bool((reward[1:] > reward[:-1]).all())
    torch.testing.assert_close(reward, bounded_reward(score + 200, calibration(score + 200)))
    assert torch.isfinite(bounded_reward(torch.ones(4), calibration(torch.ones(4)))).all()


def test_physical_direction_and_repeat_guard_reject_tradeoffs_and_changed_terminal():
    base = np.ones(7)
    assert physical_direction(base, base * .9, np.array([10]), np.array([10]))[0] == 1
    changed = base * .9
    changed[0] = 1.1
    assert physical_direction(base, changed, np.array([10]), np.array([10]))[0] == 0
    trace = dict(active=np.array([[True], [True], [False]]),
                 done=np.array([[0], [2], [0]]), norm_diff=np.zeros((3, 1, 2)))
    repeat = {key: value.copy() for key, value in trace.items()}
    repeat["norm_diff"][2] = np.nan  # Ignore poisoned post-termination physics.
    assert control_pass(trace, repeat, slice(None))[0]
    repeat["done"][1] = 1
    assert not control_pass(trace, repeat, slice(None))[0]


def test_gate_does_not_accept_one_repeat_one_horizon_or_new_harms():
    good = dict(contrasts={name: dict(harmed=0, net_corrections=3) for name in ("frozen_add", "radial")})
    pair = {"32": good, "full": good}
    assert gate([pair, pair])["passed"]
    assert not gate([pair])["passed"]
    assert not gate([{"full": good}, pair])["passed"]
    bad = dict(contrasts={"frozen_add": dict(harmed=1, net_corrections=5)})
    assert not gate([pair, {"32": bad, "full": good}])["passed"]
