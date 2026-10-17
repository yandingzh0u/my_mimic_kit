"""Checks for the CPU-only evidence audit, not evidence of learning benefits."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


path = Path(__file__).resolve().parents[1] / "tools/diagnostics/compare_flow_learning.py"
spec = importlib.util.spec_from_file_location("flow_compare", path)
compare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compare)


def trace():
    return dict(active=np.array([[True, True], [True, False], [False, False]]),
                reward=np.array([[1., 2.], [3., np.nan], [np.nan, np.nan]]),
                done=np.array([[0, 1], [2, 1], [2, 1]]),
                errors=np.ones((3, 2, 7)), initial_obs=np.zeros((2, 3)),
                initial_phase=np.array([0., .5]))


def test_returns_use_terminal_reward_but_ignore_inactive_values():
    result = compare.actual_returns(trace(), .5)
    assert np.array_equal(result["length"], [2, 1])
    assert np.array_equal(result["success"], [True, False])
    assert np.allclose(result["discounted"], [2.5, 2])
    assert np.allclose(result["undiscounted"], [4, 2])


def test_no_update_is_exact_zero_control():
    result = compare.paired_comparison(trace(), trace(), .99)
    for key in ("discounted", "undiscounted", "length", "success"):
        assert result[key]["paired_delta"] == 0
        assert result[key]["conditional_paired_se"] == 0
    assert all(value["relative_delta"] == 0 for value in result["common_prefix_errors"].values())


def test_different_episode_lengths_use_only_common_physical_prefix():
    before, after = trace(), trace()
    after["active"][1, 0] = False
    after["done"][0, 0] = 1
    after["errors"][0] *= 2
    after["errors"][1:] = 1e9
    result = compare.paired_comparison(before, after, .99)
    assert result["length"]["paired_delta"] == -.5
    assert all(value["relative_delta"] == 1 for value in result["common_prefix_errors"].values())


def test_unpaired_visible_initial_states_are_rejected():
    before, after = trace(), trace()
    after["initial_phase"][0] = .1
    with pytest.raises(ValueError, match="do not match"):
        compare.paired_comparison(before, after, .99)


def test_nonfinite_active_reward_is_rejected():
    data = trace()
    data["reward"][0, 0] = np.nan
    with pytest.raises(ValueError, match="Non-finite"):
        compare.actual_returns(data, .99)


def test_torn_final_jsonl_record_does_not_invent_terminal_state(tmp_path):
    path = tmp_path / "rows.jsonl"
    path.write_text('{"offset": 1}\n{"offset":')
    assert compare.read_rows(path) == [dict(offset=1)]


def test_first_rollout_audit_detects_changed_optimizer_measurement():
    before = dict(metrics=dict(actor_loss=.1), allocation=dict(coverage={}, gradient_second_moments=[1]))
    after = dict(metrics=dict(actor_loss=.2), allocation=dict(coverage={}, gradient_second_moments=[1]))
    result = compare.first_rollout_audit(before, after)
    assert not result["metrics_exact"]
    assert result["moments_exact"]
    assert result["metric_differences"] == dict(actor_loss=[.1, .2])


def test_summary_comparison_does_not_invent_paired_confidence():
    before = dict(discounted=10, undiscounted=15, length=70, success=0,
                  tracking_errors={key: 2. for key in compare.ERROR_NAMES})
    after = dict(discounted=11, undiscounted=16, length=71, success=0,
                 tracking_errors={key: 1. for key in compare.ERROR_NAMES})
    result = compare.summary_comparison(before, after)
    assert result["discounted"] == dict(baseline=10, candidate=11, delta=1)
    assert result["episode_length_dependent_error_means"]["root_pos"]["relative_delta"] == -.5
    assert "conditional_paired_se" not in result["discounted"]


def test_matched_points_exclude_initial_and_unobserved_budgets():
    stats = dict(discounted=10, undiscounted=15, length=70, success=0,
                 tracking_errors={key: 2. for key in compare.ERROR_NAMES})
    modes = {key: stats for key in ("rsi_sample", "phase0_mean")}
    left = dict(evaluations=[dict(offset=offset, modes=modes) for offset in (0, 100, 200)])
    right = dict(evaluations=[dict(offset=offset, modes=modes) for offset in (0, 100)])
    points = compare.matched_summary_points(left, right)
    assert set(points) == {100}
    assert points[100]["phase0_mean"]["discounted"]["delta"] == 0


def test_missing_phase_is_preserved_not_hidden_by_finite_weight_average():
    rows = [dict(offset=1, allocation=dict(coverage=dict(
        fractions=[0., 1.], kish_ess_fraction=0., max_phase_weight=None,
        first_bin_fraction=0.)))]
    result = compare.summarize_rows(rows)
    assert result["max_phase_weight"] == dict(mean=None, minimum=None, maximum=None,
                                               missing_due_to_absent_phase=1)
    assert result["kish_ess_fraction"]["mean"] == 0.
