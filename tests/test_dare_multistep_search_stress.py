from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_multistep_search_stress import (
    candidate_bank, common_error_means, control_validity, objectives, select_candidates,
    stats, summarize_validation,
)
from dare_multistep_search_rescore import common_rescore, pool_stats


def test_bank_matched_single_multi_and_antithetic_sequences():
    bank, meta = candidate_bank(2)
    assert bank.shape == (17, 32, 28)
    np.testing.assert_array_equal(bank[0], 0.)
    np.testing.assert_array_equal(bank[1:9, 0], bank[9:, 0])
    np.testing.assert_array_equal(bank[9:, 1:], 0.)
    for i in range(1, len(bank), 2):
        np.testing.assert_array_equal(bank[i], -bank[i + 1])
    np.testing.assert_array_equal(bank, candidate_bank(2)[0])
    assert meta[1]["kind"] == "multi" and meta[9]["kind"] == "single"


def test_objective_terminal_reward_included_bootstrap_only_alive():
    trace = dict(active=np.array([[1, 1], [1, 1], [0, 1]], dtype=bool),
                 done=np.array([[0, 0], [1, 0], [0, 0]]),
                 reward=np.array([[2., 1.], [3., 1.], [np.nan, 1.]]),
                 next_value=np.array([[7., 10.], [0., 10.], [np.nan, 10.]]))
    raw, boot = objectives(trace, 3, .9)
    np.testing.assert_allclose(raw, [4.7, 2.71])
    np.testing.assert_allclose(boot, [4.7, 10.])


def test_selection_ignores_physical_labels_and_keeps_unchanged_option():
    _, meta = candidate_bank(1)
    trace = dict(active=np.ones((32, len(meta)), dtype=bool),
                 done=np.zeros((32, len(meta)), dtype=int),
                 reward=np.zeros((32, len(meta))),
                 next_value=np.zeros((32, len(meta))))
    trace["reward"][:, 0] = 1.
    assert all(row["index"] == 0 for row in select_candidates(trace, meta, 1, .99))
    trace["reward"][:, 1] = 2.
    rows = select_candidates(trace, meta, 1, .99)
    assert [r["index"] for r in rows] == [1, 1, 1, 0, 0]


def test_control_excludes_replay_drift_even_when_termination_matches():
    base = dict(active=np.ones((4, 2), dtype=bool), errors=np.ones((4, 2, 7)))
    repeat = {k: v.copy() for k, v in base.items()}
    repeat["errors"][2, 1, 0] += .001
    valid, _ = control_validity(base, repeat)
    np.testing.assert_array_equal(valid, [True, False])
    np.testing.assert_array_equal(control_validity(base, repeat, False)[0], [False, False])


def test_selection_uses_paired_baseline_and_identical_control_valid_noises():
    _, meta = candidate_bank(1)
    n = len(meta) * 2
    trace = dict(active=np.ones((32, n), dtype=bool),
                 done=np.zeros((32, n), dtype=int),
                 reward=np.ones((32, n)), next_value=np.zeros((32, n)))
    baseline = {k: v.copy() for k, v in trace.items()}
    # Candidate 1 appears spectacular only on excluded noise index 1.
    trace["reward"][:, 3] = 1000.
    trace["reward"][:, 4] = 2.
    rows = select_candidates(trace, meta, 2, .99, baseline, np.array([True, False]))
    assert [r["index"] for r in rows] == [2, 2, 2, 0, 0]
    assert all(r["common_valid_noise_replicas"] == 1 for r in rows)


def test_common_prefix_does_not_count_nan_after_failure():
    base = dict(active=np.ones((4, 1), dtype=bool), errors=np.ones((4, 1, 7)))
    changed = dict(active=np.array([[1], [1], [0], [0]], dtype=bool),
                   errors=np.ones((4, 1, 7)))
    changed["errors"][2:] = np.nan
    left, right = common_error_means(base, changed, .99)
    np.testing.assert_array_equal(left, right)


def test_untouched_third_replay_noise_failure_excludes_every_candidate():
    base = dict(active=np.ones((4, 12), dtype=bool),
                errors=np.ones((4, 12, 7)), reward=np.ones((4, 12)),
                next_value=np.zeros((4, 12)), done=np.zeros((4, 12), dtype=int))
    base["done"][-1] = 1
    repeat = {k: v.copy() for k, v in base.items()}
    changed = {k: v.copy() for k, v in base.items()}
    changed["errors"][2, 0, 0] += .1
    selections = [dict(name=str(i), index=1) for i in range(5)]
    rows = summarize_validation(base, repeat, changed, selections, 2, .99, True)
    assert all(r["control_valid_replicas"] == 1 for r in rows)
    assert rows[0]["return_delta"]["mean"] == 0.


def test_pooled_moments_match_concatenated_noise_repetitions():
    left, right = np.array([1., 2., 3.]), np.array([8., 9.])
    actual = pool_stats([stats(left), stats(right)])
    expected = stats(np.concatenate((left, right)))
    assert actual["n"] == expected["n"]
    np.testing.assert_allclose([actual["mean"], actual["se"]],
                               [expected["mean"], expected["se"]])


def test_fair_rescore_applies_union_of_control_failures_to_all_methods():
    base = dict(active=np.ones((4, 18), dtype=bool),
                errors=np.ones((4, 18, 7)), reward=np.ones((4, 18)),
                next_value=np.zeros((4, 18)), done=np.zeros((4, 18), dtype=int))
    base["done"][-1] = 1
    repeat = {k: v.copy() for k, v in base.items()}
    changed = {k: v.copy() for k, v in base.items()}
    # Noise index 1 fails only in method group 2; all methods must drop it.
    repeat["errors"][2, 7, 0] += .1
    selections = [dict(name=str(i), index=1) for i in range(5)]
    result = common_rescore(base, repeat, changed, selections, 3, .99, True)
    assert result["kept_noise_indices"] == [0, 2]
    assert all(r["control_valid_replicas"] == 2 for r in result["results"])
