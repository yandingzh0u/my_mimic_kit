from pathlib import Path
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_reward_drift_stress import drift_metrics, lambda_windows, returns


def test_returns_excludes_padding_and_keeps_terminal_bootstrap():
    reward = np.array([[1., 1.], [2., 1000.], [3., 1000.]])
    active = np.array([[True, True], [True, False], [True, False]])
    terminal = np.array([[0., 7.], [0., 0.], [4., 0.]])
    actual = returns(reward, active, .5, terminal)
    np.testing.assert_allclose(actual, [[3.25, 4.5], [4.5, 0.], [5., 0.]])


def test_returns_rejects_cross_episode_join():
    with pytest.raises(ValueError, match="separate episodes"):
        returns(np.ones((3, 1)), np.array([[True], [False], [True]]), .99)


def test_lambda_windows_matches_production_recursion():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))
    from learning.rl_util import compute_td_lambda_return
    generator = np.random.default_rng(501)
    reward = generator.normal(size=(65, 4))
    value = generator.normal(size=(65, 4))
    done = np.zeros((65, 4), dtype=int)
    done[12, 0] = 1
    done[31, 1] = 2
    expected = np.concatenate([compute_td_lambda_return(
        torch.tensor(reward[t:t + 32]), torch.tensor(value[t:t + 32]),
        torch.tensor(done[t:t + 32]), .99, .95).numpy() for t in (0, 32, 64)])
    # Production intentionally casts reset masks to FP32 even with FP64 inputs.
    np.testing.assert_allclose(lambda_windows(reward, value, done, np.ones_like(done, dtype=bool)),
                               expected, atol=1e-6, rtol=2e-6)


def test_zero_update_has_zero_drift():
    reward = np.ones((40, 8))
    active = np.ones_like(reward, dtype=bool)
    trace = dict(active=active, value=reward * 10, next_value=reward * 10,
                 done=np.zeros_like(reward, dtype=int), terminal_value=reward * 0,
                 initial_phase=np.arange(8) / 8)
    result = drift_metrics(reward, reward.copy(), trace)
    assert result["return_change_rms"] == 0
    assert result["initial_value_mse_change"]["mean"] == 0
    assert result["first32_td_lambda_sign_flip_fraction"] == 0


def test_terminal_bootstrap_unchanged_under_reward_difference():
    reward = np.ones((3, 2))
    active = np.array([[True, True], [True, False], [True, False]])
    terminal = np.ones_like(reward) * 12
    change = returns(reward + .2, active, .9, terminal) - returns(reward, active, .9, terminal)
    expected = returns(np.ones_like(reward) * .2, active, .9)
    np.testing.assert_allclose(change, expected)
