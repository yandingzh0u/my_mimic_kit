import copy
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_reppo_stress import (FIT_SEEDS, FUTURE_SEEDS, WINDOWS, ERROR_NAMES,
                               KINDS, gate, monte_carlo_q)


def test_mc_q_excludes_current_entropy_and_finished_frames():
    reward = torch.tensor([[1., 2.], [3., 999.], [5., 999.]])
    active = torch.tensor([[True, True], [True, False], [True, False]])
    logp = torch.tensor([[2., 2.], [2., float("nan")], [2., float("nan")]])
    raw, soft = monte_carlo_q(reward, logp, active, gamma=.5, alpha=1.)
    torch.testing.assert_close(raw[:, 0], torch.tensor([3.75, 5.5, 5.]))
    torch.testing.assert_close(soft[:, 0], torch.tensor([2.25, 4.5, 5.]))
    torch.testing.assert_close(soft[:, 1], torch.tensor([2., 0., 0.]))
    assert torch.isfinite(soft).all()


def cases():
    good = dict(valid_replicas=8, replicas=8, length_delta=0., success_delta=0.,
        repeat_reward_drift=0., reward_delta=dict(mean=.2, ci95=[.1,.3]),
        errors={name:dict(relative=-.02) for name in ERROR_NAMES})
    results = {k:copy.deepcopy(good) for k in KINDS}
    for k in ("negative_soft_q", "random_1", "random_2"):
        results[k]["reward_delta"]["mean"] = .1
    return [dict(label=f"{fit}_{window}_{future}",fit_seed=fit,window=window,future_seed=future,
                 results=copy.deepcopy(results))
            for fit in FIT_SEEDS for window in WINDOWS for future in FUTURE_SEEDS]


def test_screen_cannot_pass_missing_cases_controls_or_physical_regression():
    fitting = [dict(seed=s, kind=k, heldout_r2=.5, heldout_outside_support=0.)
               for s in FIT_SEEDS for k in ("raw", "soft")]
    data = cases()
    assert gate(data, fitting)["passed"]
    assert not gate(data[:-1], fitting)["passed"]
    assert not gate(data[1:]+[copy.deepcopy(data[1])], fitting)["passed"]
    for key, value in (("valid_replicas", 7), ("length_delta", -1.), ("success_delta", -.01)):
        broken = copy.deepcopy(data)
        broken[0]["results"]["soft_q"][key] = value
        assert not gate(broken, fitting)["passed"]
    data[0]["results"]["soft_q"]["errors"]["dof_vel"]["relative"] = .0001
    assert not gate(data, fitting)["passed"]


def test_mean_positive_reward_is_not_sufficient_and_random_control_matters():
    data = cases()
    data[0]["results"]["soft_q"]["reward_delta"]["ci95"][0] = -.001
    assert not gate(data, [])["passed"]
    data = cases()
    data[0]["results"]["random_1"]["reward_delta"]["mean"] = .21
    assert not gate(data, [])["passed"]
