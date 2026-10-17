import math
from pathlib import Path
import sys

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_multinegative_screen import contrastive_loss, objective, safety_gate
from dare_multinegative_learning import eligible_arms, replay_control


@pytest.mark.parametrize("mapping", ["logit", "paper"])
def test_vectorized_matches_explicit_windows(mapping):
    p = torch.tensor([1.2], dtype=torch.double, requires_grad=True)
    n = torch.tensor([-1.1, .3, -4., .2], dtype=torch.double, requires_grad=True)
    actual = contrastive_loss(p, n, mapping, 3)
    pp, nn = (p.sigmoid() / .07, n.sigmoid() / .07) if mapping == "paper" else (p, n)
    expected = torch.stack([torch.logsumexp(torch.cat((pp, nn[[(i+j) % 4 for j in range(3)]])), 0) - pp[0]
                            for i in range(4)]).mean()
    torch.testing.assert_close(actual, expected)
    for a, b in zip(torch.autograd.grad(actual, (p, n), retain_graph=True),
                    torch.autograd.grad(expected, (p, n))):
        torch.testing.assert_close(a, b)


def test_native_logit_gauge_and_log_odds_identity():
    p, n = torch.tensor([1.]), torch.tensor([-.5, -1., -2.])
    torch.testing.assert_close(contrastive_loss(p, n), contrastive_loss(p + 7, n + 7))
    f = torch.linspace(-5, 5, 20, dtype=torch.double)
    torch.testing.assert_close(torch.logit(f.sigmoid()), f)
    torch.testing.assert_close(2 * torch.atanh(2 * f.sigmoid() - 1), f)


def test_group_competition_and_hard_negative_gradient():
    p = torch.zeros(1, dtype=torch.double, requires_grad=True)
    n = torch.tensor([-3., 1., -3.], dtype=torch.double, requires_grad=True)
    loss = contrastive_loss(p, n, group_size=3)
    gp, gn = torch.autograd.grad(loss, (p, n))
    assert gp < 0 and (gn > 0).all() and gn[1] > gn[0] and gn[1] > gn[2]
    torch.testing.assert_close(gp.sum() + gn.sum(), torch.zeros((), dtype=torch.double), atol=1e-12, rtol=0)
    zeros = torch.zeros(20)
    assert abs(float(contrastive_loss(torch.zeros(1), zeros)) - math.log(16)) < 1e-6


@pytest.mark.parametrize("mapping", ["logit", "paper"])
def test_extreme_logits_finite(mapping):
    p = torch.tensor([1e4], requires_grad=True)
    n = torch.tensor([-1e4, 1e4, 0.], requires_grad=True)
    loss = contrastive_loss(p, n, mapping)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(p.grad).all() and torch.isfinite(n.grad).all()


def test_auxiliary_keeps_bce_and_shared_gauge_anchor():
    p, n = torch.tensor([1.]), torch.tensor([-2., -.5])
    base, bce, _ = objective(p, n, "bce")
    torch.testing.assert_close(base, bce)
    for arm in ("bce_nce_logit", "bce_nce_paper"):
        total, bce, nce = objective(p, n, arm)
        torch.testing.assert_close(total, bce + nce)
    with pytest.raises(ValueError):
        objective(p, n, "unknown")


def test_safety_is_not_usefulness_and_rejects_harms_or_collapse():
    stats = {h: {"contrasts": {c: {"harmed": 0} for c in ("bce", "frozen_add")}}
             for h in ("32", "full")}
    heldout = dict(all_finite=True, reward_std=.01)
    decision = safety_gate(stats, heldout)
    assert decision["passed"] and "UNPROVEN" in decision["reason"]
    stats["full"]["contrasts"]["bce"]["harmed"] = 1
    assert not safety_gate(stats, heldout)["passed"]
    assert not safety_gate({}, heldout)["passed"]
    assert not safety_gate({}, dict(all_finite=True, reward_std=0))["passed"]


def test_online_eligibility_requires_complete_paired_repeats():
    rows = [dict(iteration=2999, seed=seed, safety={
        "bce_nce_logit": dict(passed=True), "bce_nce_paper": dict(passed=(seed == 1))})
        for seed in (1, 2)]
    assert eligible_arms(dict(complete=True, cases=rows)) == ["bce", "bce_nce_logit"]
    with pytest.raises(ValueError):
        eligible_arms(dict(complete=False, cases=rows))
    with pytest.raises(ValueError):
        eligible_arms(dict(complete=True, cases=rows[:1]))


def test_replay_control_rejects_matching_rewards_but_different_physics():
    import numpy as np
    from envs.base_env import DoneFlags
    steps, count = 4, 3
    trace = dict(active=np.ones((steps, count), dtype=bool), reward=np.ones((steps, count)),
                 done=np.zeros((steps, count), dtype=int), errors=np.ones((steps, count, 7)),
                 root_abs=np.ones((steps, count, 3)), knee_abs=np.ones((steps, count, 2)),
                 initial_obs=np.zeros((count, 4)), initial_phase=np.zeros(count))
    trace["done"][-1] = DoneFlags.SUCC.value
    assert replay_control(trace, trace)["passed"]
    changed = dict(trace, errors=trace["errors"] * 1.01)
    assert not replay_control(trace, changed)["passed"]


@pytest.mark.parametrize("arm", ["bce", "bce_nce_logit", "bce_nce_paper"])
def test_online_adapter_uses_same_zero_anchor_and_current_replay(arm):
    from types import SimpleNamespace
    from dare_multinegative_learning import MultiNegativeAgent
    calls = []
    class FakeModel:
        def eval_disc_raw(self, inputs):
            calls.append(inputs.detach().clone())
            return inputs.sum(-1, keepdim=True)

        def get_disc_group_width(self):
            return 146

        def get_disc_group_total_width(self):
            return 1022

    current = dict(disc_obs_demo=torch.ones(4, 3), disc_obs=torch.zeros(4, 3))
    previous = dict(disc_obs_demo=torch.zeros(4, 3), disc_obs=torch.ones(4, 3))
    fake = SimpleNamespace(_model=FakeModel(), _pos_diff=torch.zeros(3), objective_arm=arm,
        _disc_buffer=SimpleNamespace(sample=lambda count: previous),
        _disc_obs_norm=SimpleNamespace(normalize=lambda value: value / 2),
        _compute_disc_acc=lambda n, p: ((n < 0).float().mean(), (p > 0).float().mean()))
    metrics = MultiNegativeAgent._compute_disc_loss(fake, current)
    assert len(calls) == 2 and torch.equal(calls[0], torch.zeros(1, 3))
    assert len(calls[1]) == 8 and (calls[1] == .5).all(-1).sum() == 4
    assert (calls[1] == -.5).all(-1).sum() == 4
    expected, bce, nce = objective(torch.zeros(1), calls[1].sum(-1), arm)
    torch.testing.assert_close(metrics["disc_loss"], expected)
    torch.testing.assert_close(metrics["disc_cls_loss"], bce)
    torch.testing.assert_close(metrics["disc_nce_loss"], nce)


def test_secondary_precision_analysis_masks_dead_frames_and_matches_phase():
    import numpy as np
    from summarize_multinegative_learning import secondary_metrics, hotspot_comparison
    active = np.ones((80, 2), dtype=bool)
    active[60:, 1] = False
    errors = np.ones((80, 2, 7))
    knees = np.ones((80, 2, 2))
    errors[~active] = np.nan
    knees[~active] = np.nan
    trace = dict(active=active, errors=errors, knee_abs=knees, initial_phase=np.zeros(2))
    metrics = secondary_metrics(trace)
    assert metrics["active_frames"] == 140 and metrics["dof_mean_velocity_frame_p95"] == 1
    changed = dict(trace, errors=errors * .8)
    comparison = hotspot_comparison(trace, changed)
    assert comparison["frames"] == 60
    assert all(abs(value - .8) < 1e-10 for value in comparison["ratios"].values())
    with pytest.raises(ValueError):
        hotspot_comparison(trace, dict(changed, initial_phase=np.ones(2)))


def test_own_episode_mean_audit_uses_float64_reductions():
    import numpy as np
    from summarize_multinegative_learning import own_episode_means
    active = np.ones((10000, 2), dtype=bool)
    active[-1, 1] = False
    errors = np.full((10000, 2, 7), .1, dtype=np.float32)
    knees = np.full((10000, 2, 2), .3, dtype=np.float32)
    errors[~active] = np.nan
    knees[~active] = np.nan
    means = own_episode_means(dict(active=active, errors=errors, knee_abs=knees))
    assert all(value == float(np.float32(.1)) for value in means["errors"].values())
    assert means["knee_abs"] == [float(np.float32(.3))] * 2


def test_training_sampler_initialization_is_independent_of_prior_branch_rng():
    from types import SimpleNamespace
    from dare_multinegative_learning import initialize_training_random_state
    from learning.experience_buffer import ExperienceBuffer
    experience = ExperienceBuffer(4, 8, "cpu")
    replay = ExperienceBuffer(7, 1, "cpu")
    fake = SimpleNamespace(_exp_buffer=experience, _disc_buffer=replay,
        _build_optimizer=lambda config: None, _init_train=experience.clear)
    before = initialize_training_random_state(fake, {}, 1234)
    torch.rand(413)
    experience.set_total_samples(experience.get_capacity())
    replay.set_total_samples(replay.get_capacity())
    experience.sample(4)
    replay.sample(3)
    after = initialize_training_random_state(fake, {}, 1234)
    assert before == after
    assert before != initialize_training_random_state(fake, {}, 5678)


def test_gradient_shape_audit_distinguishes_scaling_and_reweighting():
    from audit_multinegative_gradients import gradient_shape
    gradient = torch.tensor([1., 2., 3.])
    same = gradient_shape(gradient, 5 * gradient)
    assert abs(same["cosine"] - 1) < 1e-12
    assert same["best_scalar"] == 5 and same["relative_residual"] == 0
    changed = gradient_shape(gradient, gradient.flip(0))
    assert changed["relative_residual"] > .5
    with pytest.raises(ValueError):
        gradient_shape(gradient, torch.zeros_like(gradient))
