"""Training/reset/checkpoint contracts for the DARE collection allocator."""

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
sys.path.insert(0, str(ROOT / "tools/diagnostics"))
from envs import base_env
from envs.add_env import ADDEnv
from envs.deepmimic_env import DeepMimicEnv
from learning.trajectory_allocation import TrajectoryAllocator, solve_allocation
from stress_flow_allocation import sparse_operator


def test_previous_numerical_failure_has_certified_solution():
    rng = np.random.default_rng(951501)
    for _ in range(987):
        visits = sparse_operator(rng)
        moments = np.exp(rng.uniform(-3., 3., 8))
    fit = solve_allocation(visits, moments)
    assert fit["numerical_certificate"]["relative_optimality_gap"] <= 1e-6
    assert fit["normalized_second_moment_objective"] <= fit["baseline_second_moment_objective"]


def test_unobserved_episode_origins_do_not_create_a_visit_model():
    flow = TrajectoryAllocator(8, 2, "cpu", 9)
    flow.restart(None, torch.zeros(8))
    flow.record(torch.zeros(8))
    flow.finish(torch.ones(8))
    assert flow.flush() is None
    assert flow.sample_phases(4) is None


def test_zero_signal_update_retains_uniform_reference_starts():
    flow = TrajectoryAllocator(8, 2, "cpu", 9)
    flow.update(np.diag([1., 10.]), [0., 0.])
    torch.testing.assert_close(flow.probabilities, torch.tensor([.5, .5]))


def test_sampler_uses_private_rng_and_restores_it_from_checkpoint():
    flow = TrajectoryAllocator(8, 2, "cpu", 9)
    flow.probabilities = torch.tensor([.25, .75])
    global_rng = torch.get_rng_state().clone()
    state = flow.state_dict()
    expected = flow.sample_phases(20)
    assert torch.equal(global_rng, torch.get_rng_state())
    restored = TrajectoryAllocator(8, 2, "cpu", 99)
    restored.load_state_dict(state)
    torch.testing.assert_close(restored.sample_phases(20), expected, rtol=0., atol=0.)


def test_checkpoint_preserves_episode_fifo_but_discards_partial_physics():
    flow = TrajectoryAllocator(8, 2, "cpu", 9)
    phase = torch.arange(8).float() / 8
    flow.restart(None, phase)
    flow.record(phase)
    flow.finish(torch.ones(8))
    matrix = flow.flush()
    flow.record(phase)
    flow.update(matrix, [1., 4.])
    restored = TrajectoryAllocator(8, 2, "cpu", 99)
    restored.load_state_dict(flow.state_dict())
    assert np.array_equal(flow.sizes, restored.sizes)
    assert np.array_equal(flow.cursors, restored.cursors)
    assert np.array_equal(matrix, restored.flush())
    assert restored.updates == 1 and not restored.visits.any()


@pytest.mark.parametrize("mode,n,expected_adaptive", [
    (base_env.EnvMode.TRAIN, 7, True),
    (base_env.EnvMode.TRAIN, 8, False),
    (base_env.EnvMode.TRAIN, 0, False),
    (base_env.EnvMode.TEST, 7, False),
])
def test_only_partial_training_resets_use_allocation(monkeypatch, mode, n, expected_adaptive):
    calls = []
    def original(self, count):
        calls.append("original")
        return torch.zeros(count, dtype=torch.long), torch.full((count,), .75)
    monkeypatch.setattr(DeepMimicEnv, "_sample_motion_times", original)
    env = object.__new__(ADDEnv)
    env._mode, env._rand_reset = mode, True
    env.get_num_envs = lambda: 8
    env._motion_lib = SimpleNamespace(get_motion_length=lambda ids: torch.full_like(ids, 2.))
    def sampler(count):
        calls.append("allocation")
        return torch.full((count,), .125)
    env.set_motion_start_sampler(sampler)
    _, times = env._sample_motion_times(n)
    assert calls == (["original", "allocation"] if expected_adaptive else ["original"])
    torch.testing.assert_close(times, torch.full((n,), .25 if expected_adaptive else .75))


def test_invalid_solver_point_is_not_silently_changed_to_uniform(monkeypatch):
    import learning.trajectory_allocation as module
    monkeypatch.setattr(module, "minimize", lambda *args, **kwargs: SimpleNamespace(x=np.array([-.1, 1.1])))
    with pytest.raises(RuntimeError, match="infeasible"):
        solve_allocation(np.eye(2), [1., 1.])
