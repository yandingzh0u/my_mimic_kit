from __future__ import annotations

import numpy as np
import pytest

from tools.phase_transition_critic.offline_gate import (
    GateThresholds,
    evaluate_gate,
)
from tools.phase_transition_critic.rollout_contract import (
    validate_transition_bundle,
)


def _transition_bundle(episodes: int = 4, steps: int = 3, dim: int = 5):
    episode_id = np.repeat(np.arange(episodes, dtype=np.int64), steps)
    step_index = np.tile(np.arange(steps, dtype=np.int64), episodes)
    x_t = np.stack(
        [episode_id, step_index, episode_id + step_index, step_index**2, episode_id * 0],
        axis=-1,
    ).astype(np.float32)
    x_t1 = x_t.copy()
    r_t = 0.5 * x_t
    r_t1 = r_t.copy()
    for episode in range(episodes):
        rows = np.flatnonzero(episode_id == episode)
        x_t1[rows[:-1]] = x_t[rows[1:]]
        r_t1[rows[:-1]] = r_t[rows[1:]]
        x_t1[rows[-1]] = x_t[rows[-1]] + 1
        r_t1[rows[-1]] = r_t[rows[-1]] + 0.5
    return {
        "x_t": x_t,
        "x_t1": x_t1,
        "r_t": r_t,
        "r_t1": r_t1,
        "episode_id": episode_id,
        "step_index": step_index,
        "phase": step_index.astype(np.float32) / steps,
        "alive": np.ones(episodes * steps, dtype=np.bool_),
    }


def _score_bundle(episodes: int = 20, steps: int = 4):
    ids = np.repeat(np.arange(episodes, dtype=np.int64), steps)
    alive = np.ones(ids.size, dtype=np.bool_)
    offset = np.tile(np.linspace(-0.01, 0.01, steps), episodes)
    return {
        "reference_score": np.zeros(ids.size),
        "reference_episode_id": ids,
        "reference_alive": alive,
        "success_score": -0.25 + offset,
        "success_episode_id": ids,
        "success_alive": alive,
        "shortcut_score": -1.0 + offset,
        "shortcut_episode_id": ids,
        "shortcut_alive": alive,
        "reference_phase_shuffled_score": -0.75 + offset,
        "reference_phase_shuffled_alive": alive,
        "pose_sensitivity": np.full(ids.size, 0.55),
        "velocity_sensitivity": np.full(ids.size, 0.45),
        "next_error_sensitivity": np.full(ids.size, 0.60),
        "motion_error_sensitivity": np.full(ids.size, 0.40),
        "gp_norm": np.linspace(0.8, 1.2, 100),
    }


def test_transition_contract_accepts_contiguous_unlabelled_rollout():
    report = validate_transition_bundle(_transition_bundle())
    assert report["num_episodes"] == 4
    assert report["phi_dim"] == 5
    assert report["continuity_links"] == 8


def test_transition_contract_rejects_behavior_label_and_broken_continuity():
    labelled = _transition_bundle()
    labelled["completion"] = np.ones(labelled["alive"].shape, dtype=np.bool_)
    with pytest.raises(ValueError, match="behavior labels"):
        validate_transition_bundle(labelled)

    broken = _transition_bundle()
    broken["x_t1"][0, 0] += 1
    with pytest.raises(ValueError, match="continuity"):
        validate_transition_bundle(broken)


def test_gate_passes_strictly_ranked_balanced_scores():
    report = evaluate_gate(
        _score_bundle(), GateThresholds(min_episodes=16, bootstrap_samples=200)
    )
    assert report["pass"] is True
    assert report["decision"] == "GO"
    assert report["checks"]["success_over_shortcut"]["pair_probability"] == 1.0


def test_gate_rejects_shortcut_misranking():
    bundle = _score_bundle()
    bundle["shortcut_score"] = np.full_like(bundle["shortcut_score"], -0.1)
    report = evaluate_gate(
        bundle, GateThresholds(min_episodes=16, bootstrap_samples=200)
    )
    assert report["pass"] is False
    assert report["checks"]["success_over_shortcut"]["pass"] is False


def test_gate_rejects_position_only_sensitivity():
    bundle = _score_bundle()
    bundle["pose_sensitivity"][:] = 1.0
    bundle["velocity_sensitivity"][:] = 1e-9
    report = evaluate_gate(
        bundle, GateThresholds(min_episodes=16, bootstrap_samples=200)
    )
    assert report["pass"] is False
    assert report["checks"]["pose_velocity_sensitivity"]["pass"] is False


def test_gate_rejects_next_state_only_sensitivity():
    bundle = _score_bundle()
    bundle["next_error_sensitivity"][:] = 1.0
    bundle["motion_error_sensitivity"][:] = 1e-9
    report = evaluate_gate(
        bundle, GateThresholds(min_episodes=16, bootstrap_samples=200)
    )
    assert report["pass"] is False
    assert report["checks"]["next_motion_error_sensitivity"]["pass"] is False


def test_gate_rejects_nonfinite_valid_score():
    bundle = _score_bundle()
    bundle["success_score"][0] = np.nan
    with pytest.raises(ValueError, match="NaN/Inf"):
        evaluate_gate(
            bundle, GateThresholds(min_episodes=16, bootstrap_samples=200)
        )
