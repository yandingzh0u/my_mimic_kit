"""Official sampler-formula and experiment-control tests; no simulator imports."""

from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from dare_failure_sampling import FailureAdaptiveFlow


class Env:
    def __init__(self):
        self.phase = torch.tensor([.1, .2, .3, .4, .6, .7, .8, .9])
        self._motion_lib = type("MotionLib", (), {
            "get_motion_length": lambda self, ids: torch.ones_like(ids, dtype=torch.float32),
        })()

    def get_motion_phase(self):
        return self.phase

    def get_num_envs(self):
        return len(self.phase)

    def _sample_motion_times(self, n):
        return torch.zeros(n, dtype=torch.long), torch.full((n,), .75)

    def reset(self, env_ids=None):
        return self.phase, {}

    def step(self, flags):
        return self.phase, torch.zeros(8), flags, {}


def test_additive_floor_is_not_constant_uniform_mix():
    flow = FailureAdaptiveFlow(Env(), 2, 73)
    assert torch.allclose(flow.failure_probabilities(), torch.tensor([.5, .5]))
    flow.failure_ema[:] = torch.tensor([0., 2.])
    assert torch.allclose(flow.failure_probabilities(), torch.tensor([.05, 2.05]) / 2.1)
    flow.close()


def test_paper_three_tap_lookback_samples_before_failure():
    flow = FailureAdaptiveFlow(Env(), 4, 73, kernel_size=3)
    flow.failure_ema[:] = torch.tensor([0., 1., 0., 0.])
    raw = torch.tensor([.8 + .025 * 2.44, 1. + .025 * 2.44,
                        .025 * 2.44, .025 * 2.44])
    assert torch.allclose(flow.failure_probabilities(), raw / raw.sum())
    assert flow.failure_probabilities()[0] > .4
    flow.close()


def test_paper_kernel_keeps_released_replicated_right_boundary():
    flow = FailureAdaptiveFlow(Env(), 4, 73, kernel_size=3)
    flow.failure_ema[:] = 0.
    assert torch.allclose(flow.failure_probabilities(), torch.full((4,), .25))
    flow.close()


def test_failure_location_not_initial_bin_and_success_not_failure():
    env = Env()
    flow = FailureAdaptiveFlow(env, 2, 73)
    flow.collect = True
    flow.starts[:] = 0
    flags = torch.tensor([0, 2, 3, 0, 1, 1, 0, 0])
    env.step(flags)
    assert torch.equal(flow.current_failed, torch.tensor([0., 2.]))
    assert not flow.failure_ema.any()  # one-step lag at reset
    env.step(torch.zeros(8))
    assert torch.allclose(flow.failure_ema, torch.tensor([0., .002]))
    assert torch.equal(flow.total_failures, torch.tensor([0., 2.]))
    env.step(torch.zeros(8))
    assert torch.allclose(flow.failure_ema, torch.tensor([0., .002 * .999]))
    flow.close()


def test_first_rollout_full_population_and_eval_preserve_original_rsi():
    env = Env()
    flow = FailureAdaptiveFlow(env, 2, 73)
    flow.collect = True
    flow.failure_ema[:] = torch.tensor([1000., 0.])
    assert torch.all(env._sample_motion_times(7)[1] == .75)
    flow.enabled = True
    assert torch.all(env._sample_motion_times(8)[1] == .75)
    assert torch.all(env._sample_motion_times(7)[1] < .5)
    flow.collect = False
    before = flow.failure_ema.clone()
    env.step(torch.ones(8))
    assert torch.equal(flow.failure_ema, before)
    assert torch.all(env._sample_motion_times(7)[1] == .75)
    flow.close()


def test_private_sampling_does_not_consume_additional_global_rng():
    env = Env()
    flow = FailureAdaptiveFlow(env, 2, 73)
    flow.collect = flow.enabled = True
    before = torch.random.get_rng_state().clone()
    env._sample_motion_times(7)
    assert torch.equal(torch.random.get_rng_state(), before)
    flow.close()
