"""Evaluation must not select the initialization of the next training rollout."""
from pathlib import Path
from types import SimpleNamespace
import sys
from unittest import mock

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))

import envs.base_env as base_env
import envs.deepmimic_env as deepmimic_env
import learning.base_agent as base_agent


class _PhaseEnv:
    def __init__(self):
        self._device = "cpu"
        self._rand_reset = True
        self._mode = base_env.EnvMode.TRAIN
        self._test_random_start = False
        self._motion_lib = mock.Mock()
        self._motion_lib.sample_motions.return_value = torch.zeros(4).long()
        self._motion_lib.sample_time.return_value = torch.tensor(
            [0.1, 0.3, 0.6, 0.9])
        self.resets = []

    def set_mode(self, mode):
        self._mode = mode

    def set_test_random_start(self, enabled):
        self._test_random_start = enabled

    def reset(self, env_ids=None):
        _, self.phase = deepmimic_env.DeepMimicEnv._sample_motion_times(self, 4)
        self.resets.append((self._mode, self.phase.clone()))
        return self.phase, {}

    def record_diagnostics(self):
        return {}


class _LoopAgent(base_agent.BaseAgent):
    def __init__(self):
        torch.nn.Module.__init__(self)
        self._env = _PhaseEnv()
        self._mode = base_agent.AgentMode.TRAIN
        self._resume_pending = False
        self._elapsed_train_time = 0.0
        self._config = {}
        self._iter = 0
        self._sample_count = 0
        self._iters_per_output = 1
        self._test_episodes = 1
        self._last_test_info = {}
        self._last_test_random_info = {}
        self._train_return_tracker = SimpleNamespace(reset=lambda: None)
        self.rollout_starts = []
        self.test_starts = []

    def _build_logger(self, *args, **kwargs):
        return SimpleNamespace(print_log=lambda: None, write_log=lambda: None)

    def _init_train(self):
        pass

    def _train_iter(self):
        self.set_mode(base_agent.AgentMode.TRAIN)
        self.rollout_starts.append(self._env.phase.clone())
        return {}

    def _update_sample_count(self):
        return self._sample_count + 1

    def _rollout_test(self, num_episodes):
        self.test_starts.append(self._env.phase.clone())
        return {}

    def _log_train_info(self, *args, **kwargs):
        pass

    def _write_train_metrics_jsonl(self):
        pass

    def _output_train_model(self, *args):
        pass


@pytest.mark.parametrize("mode", tuple(base_agent.AgentMode))
@pytest.mark.parametrize("training", (False, True))
def test_evaluation_restores_caller_mode_and_test_flag(mode, training):
    agent = _LoopAgent()
    agent.set_mode(mode)
    agent.train(training)
    agent._env.set_test_random_start(True)
    with mock.patch("util.mp_util.get_num_procs", return_value=1):
        agent.test_model(1, random_start=False)
    assert agent._mode == mode
    assert agent._env._mode == base_env.EnvMode[mode.name]
    assert agent.training == training
    assert agent._env._test_random_start is True
    assert torch.count_nonzero(agent.test_starts[0]) == 0


def test_evaluation_restores_mode_even_when_rollout_raises():
    agent = _LoopAgent()
    agent._rollout_test = mock.Mock(side_effect=RuntimeError("test failure"))
    with mock.patch("util.mp_util.get_num_procs", return_value=1):
        with pytest.raises(RuntimeError, match="test failure"):
            agent.test_model(1, random_start=True)
    assert agent._mode == base_agent.AgentMode.TRAIN
    assert agent._env._mode == base_env.EnvMode.TRAIN
    assert agent.training
    assert agent._env._test_random_start is False


def test_training_resets_in_train_mode_after_every_evaluation(tmp_path):
    agent = _LoopAgent()
    # A previous standalone test must not select the first training reset.
    agent.set_mode(base_agent.AgentMode.TEST)
    with mock.patch("util.mp_util.get_num_procs", return_value=1):
        agent.train_model(3, str(tmp_path), False, "txt")
    random_phase = torch.tensor([0.1, 0.3, 0.6, 0.9])
    assert len(agent.rollout_starts) == 3
    for phase in agent.rollout_starts:
        torch.testing.assert_close(phase, random_phase)
    for random_start, start0 in zip(agent.test_starts[::2],
                                    agent.test_starts[1::2]):
        torch.testing.assert_close(random_start, random_phase)
        assert torch.count_nonzero(start0) == 0
    train_resets = [phase for mode, phase in agent._env.resets
                    if mode == base_env.EnvMode.TRAIN]
    assert len(train_resets) == 4
    assert agent._mode == base_agent.AgentMode.TRAIN
