"""Return/termination checks for the standalone physical branch diagnostic."""

from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools/diagnostics"))
from phasebalanced_physical_chain_probe import make_actions, paired_errors, returns
from learning.rl_util import compute_td_lambda_return


def test_probe_td_lambda_matches_actual_training_with_terminal_boundaries():
    rng = np.random.default_rng(361)
    reward = rng.random((40, 3)).astype(np.float32)
    next_value = rng.random((40, 3)).astype(np.float32) * 20
    done = np.zeros((40, 3), dtype=np.int64)
    done[7, 0], done[21, 1], done[39, 2] = 1, 2, 2
    active = np.ones((40, 3), dtype=bool)
    active[8:, 0], active[22:, 1] = False, False
    next_value[done != 0] = 0
    arrays = dict(reward=reward, next_value=next_value,
                  done=done, active=active)
    result = returns(arrays, .99, .95)
    training = compute_td_lambda_return(
        torch.from_numpy(reward[:32] * active[:32]),
        torch.from_numpy(next_value[:32] * active[:32]),
        torch.from_numpy(done[:32]), .99, .95)
    np.testing.assert_allclose(result["td32"], training[0].numpy(), rtol=1e-6)
    np.testing.assert_array_equal(result["length"], [8, 22, 40])
    np.testing.assert_array_equal(result["flags"], [1, 2, 2])
    # Absorbing/post-failure simulation must not contribute to the return.
    changed = {k: v.copy() for k, v in arrays.items()}
    changed["reward"][~active] = 1e6
    np.testing.assert_array_equal(result["mc"], returns(changed, .99, .95)["mc"])
    changed["reward"][~active] = np.nan
    changed["next_value"][~active] = np.nan
    np.testing.assert_array_equal(result["mc"], returns(changed, .99, .95)["mc"])
    np.testing.assert_allclose(result["td32"], returns(changed, .99, .95)["td32"])


def test_actual_reward_includes_failure_transition_and_no_later_rewards():
    arrays = dict(reward=np.array([[2.], [3.], [999.]]),
                  next_value=np.array([[10.], [0.], [999.]]),
                  done=np.array([[0], [1], [0]]),
                  active=np.array([[True], [True], [False]]))
    result = returns(arrays, .9, 1.)
    np.testing.assert_allclose(result["mc"], [4.7])
    np.testing.assert_allclose(result["td32"], result["mc"])
    np.testing.assert_allclose(result["td1"], [11.])


def test_common_prefix_error_does_not_reward_shorter_survival():
    left = dict(active=np.ones((4, 1), dtype=bool))
    right = dict(active=np.array([[True], [True], [False], [False]]))
    for key, dimensions in (("errors", 7), ("root_abs", 3), ("knee_abs", 2)):
        left[key] = np.ones((4, 1, dimensions))
        right[key] = np.ones((4, 1, dimensions))
        right[key][2:] = np.nan
    result = paired_errors(left, right, .99)
    np.testing.assert_array_equal(result["errors"], np.zeros((1, 7)))
    # Survival is checked separately in summarize; comparison uses only states
    # observed by both trajectories, not post-termination physics.


def test_antithetic_action_candidates_are_reproducible_without_global_rng_use():
    np.random.seed(367)
    before = np.random.get_state()
    actions, labels = make_actions(28, 3, [.025, .05], 371)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert actions.shape == (29, 28)
    assert labels[0] == "unchanged"
    np.testing.assert_array_equal(actions[0], np.zeros(28))
    for index in range(1, len(actions), 2):
        np.testing.assert_array_equal(actions[index], -actions[index + 1])
    np.testing.assert_array_equal(actions, make_actions(28, 3, [.025, .05], 371)[0])
