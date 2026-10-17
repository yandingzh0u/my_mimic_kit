"""CPU-only mathematical contracts for the isolated reset-flow diagnostic."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

path = Path(__file__).resolve().parents[1] / "tools/diagnostics/phasebalanced_reset_flow_probe.py"
spec = importlib.util.spec_from_file_location("reset_flow_probe", path)
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


def test_diagonal_episode_lengths_require_rate_not_probability_fit():
    visits = np.diag([1., 2., 4., 8.])
    probs, predicted, residual = probe.fit_reset_probabilities(visits)
    assert np.allclose(probs, np.array([8, 4, 2, 1]) / 15)
    assert np.allclose(predicted, .25)
    assert residual < 1e-10


def test_traversal_already_covers_every_phase_from_first_bin():
    probs, predicted, residual = probe.fit_reset_probabilities(np.triu(np.ones((8, 8))))
    assert np.allclose(probs, [1, 0, 0, 0, 0, 0, 0, 0])
    assert np.allclose(predicted, .125)
    assert residual < 1e-10


def test_infeasible_uniform_target_is_reported_not_hidden():
    probs, predicted, residual = probe.fit_reset_probabilities([[1, 2], [0, 1]])
    assert residual > 0
    assert not np.allclose(predicted, .5)
    assert np.allclose(probs, [1, 0])


@pytest.mark.parametrize("visits", [[[0, 0], [0, 1]], [[1, -1], [0, 1]],
                                   [[1, float("nan")], [0, 1]], [[1, 2, 3]]])
def test_invalid_operator_is_rejected(visits):
    with pytest.raises(ValueError):
        probe.fit_reset_probabilities(visits)


def test_weight_ess_is_one_only_for_uniform_occupancy():
    assert probe.occupancy_metrics([100] * 8)["kish_ess_fraction"] == 1
    metrics = probe.occupancy_metrics([1] + [100] * 7)
    assert metrics["kish_ess_fraction"] < .1
    assert probe.occupancy_metrics([0, 1])["kish_ess_fraction"] == 0


def test_uniform_complete_cycles_do_not_imply_uniform_training_windows():
    # Identical periodic trajectories started together cover the full motion
    # uniformly, yet a short batch sees only two of eight phases. This is a
    # sampler timing counterexample, not a policy-learning failure.
    phases = np.tile(np.arange(8), (128, 1))
    full = probe.occupancy_metrics(np.bincount(phases.flatten(), minlength=8))
    short = probe.occupancy_metrics(np.bincount(phases[:, :2].flatten(), minlength=8))
    assert full["kish_ess_fraction"] == 1
    assert short["kish_ess_fraction"] == 0
    assert short["total_variation_from_uniform"] == .75
