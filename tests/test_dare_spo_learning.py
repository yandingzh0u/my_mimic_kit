import copy
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_spo_learning import strict_gate, paired_error_intervals, physical_statistics, ERROR_NAMES, MODES


def _cases():
    value = dict(common_prefix_error_ratio={k: .94 for k in ERROR_NAMES},
                 own_episode_error_ratio={k: .94 for k in ERROR_NAMES},
                 paired_error_intervals={k: dict(upper95=-.04) for k in ERROR_NAMES},
                 length_delta=0., success_delta=0.)
    modes = {mode: copy.deepcopy(value) for mode, _, _ in MODES}
    return [dict(seed=seed, complete=True,
                 controls={mode: dict(passed=True) for mode, _, _ in MODES},
                 branches=dict(spo={reference: copy.deepcopy(modes)
                     for reference in ("versus_initial", "versus_ppo")})) for seed in (1, 2)]


def test_gate_requires_both_seeds_all_modes_and_unmodified_original_control():
    cases = _cases()
    assert strict_gate(cases)["passed"]
    assert not strict_gate(cases[:1])["passed"]
    cases[1]["seed"] = 1
    assert not strict_gate(cases)["passed"]
    cases = _cases()
    del cases[0]["branches"]["spo"]["versus_initial"]
    assert not strict_gate(cases)["passed"]


def test_gate_rejects_small_error_regression_and_uncertain_nonregression():
    for reference in ("versus_initial", "versus_ppo"):
        cases = _cases()
        row = cases[0]["branches"]["spo"][reference][MODES[0][0]]
        row["own_episode_error_ratio"]["dof_vel"] = 1.0001
        assert not strict_gate(cases)["passed"]
        row["own_episode_error_ratio"]["dof_vel"] = .99
        row["paired_error_intervals"]["dof_vel"]["upper95"] = .001
        assert not strict_gate(cases)["passed"]
        row["paired_error_intervals"]["dof_vel"]["upper95"] = float("nan")
        assert not strict_gate(cases)["passed"]


def test_gate_rejects_completion_loss_missing_physics_or_no_update_failure():
    cases = _cases()
    cases[0]["branches"]["spo"]["versus_ppo"][MODES[0][0]]["success_delta"] = -.0001
    assert not strict_gate(cases)["passed"]
    cases = _cases()
    del cases[0]["branches"]["spo"]["versus_ppo"][MODES[0][0]]["own_episode_error_ratio"]["root_pos"]
    assert not strict_gate(cases)["passed"]
    cases = _cases()
    cases[0]["controls"][MODES[0][0]]["passed"] = False
    assert not strict_gate(cases)["passed"]


def test_named_component_must_also_beat_a_matched_optimizer_control():
    cases = _cases()
    for case in cases:
        candidate = case["branches"].pop("spo")
        candidate["versus_ppo_adam"] = copy.deepcopy(candidate["versus_ppo"])
        case["branches"]["trpl"] = candidate
    refs = ("versus_initial", "versus_ppo", "versus_ppo_adam")
    assert strict_gate(cases, "trpl", refs)["passed"]
    cases[0]["branches"]["trpl"]["versus_ppo_adam"][MODES[0][0]]["common_prefix_error_ratio"]["root_pos"] = 1.01
    assert not strict_gate(cases, "trpl", refs)["passed"]


def test_cluster_intervals_use_whole_episodes_not_independent_frames():
    left = dict(active=np.ones((4, 5), dtype=bool), errors=np.ones((4, 5, 7)))
    right = dict(active=left["active"], errors=left["errors"] * .9)
    result = paired_error_intervals(left, right)
    for row in result.values():
        assert abs(row["relative_delta"] + .1) < 1e-10
        assert row["relative_se"] < 1e-10
    # Truncating the right episode must not include masked error spikes.
    right["active"] = right["active"].copy()
    right["active"][2:, 0] = False
    right["errors"][2:, 0] = 999
    for row in paired_error_intervals(left, right).values():
        assert abs(row["relative_delta"] + .1) < 1e-10


def test_physical_statistics_use_fp64_and_mask_finished_episodes():
    active = np.array([[True, True], [True, False]])
    errors = np.full((2, 2, 7), .1, dtype=np.float32)
    errors[1, 1] = 1000
    trace = dict(active=active, errors=errors, done=np.array([[0, 1], [2, 0]]),
                 root_abs=np.ones((2, 2, 3)), knee_abs=np.ones((2, 2, 2)))
    stats = physical_statistics(trace)
    assert stats["length"] == 1.5 and stats["success"] == .5
    assert all(value == float(np.float32(.1)) for value in stats["errors"].values())
