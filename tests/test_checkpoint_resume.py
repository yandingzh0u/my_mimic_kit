from pathlib import Path
import random
import sys

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
MIMICKIT = ROOT / "mimickit"
if str(MIMICKIT) not in sys.path:
    sys.path.insert(0, str(MIMICKIT))

import learning.base_agent as base_agent
import learning.experience_buffer as experience_buffer
import learning.mp_optimizer as mp_optimizer
import learning.ppo_agent as ppo_agent
import learning.distribution_gaussian_diag as distribution_gaussian_diag
from util.logger import Logger


class _TinyEnv:
    def __init__(self, num_envs=2):
        self._num_envs = num_envs

    def get_obs_space(self):
        return spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32)

    def get_action_space(self):
        return spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)

    def get_num_envs(self):
        return self._num_envs


class _TinyAgent(base_agent.BaseAgent):
    def _build_model(self, config):
        self._model = torch.nn.Linear(3, 2)

    def _build_optimizer(self, config):
        self._optimizer = mp_optimizer.MPOptimizer(
            {"type": "Adam", "learning_rate": 1e-3},
            list(self._model.parameters()))

    def _get_exp_buffer_length(self):
        return self._steps_per_iter

    def _sync_optimizer(self):
        self._optimizer.sync()

    def _decide_action(self, obs, info):
        raise NotImplementedError

    def _update_model(self):
        raise NotImplementedError


def _config():
    return {
        "discount": 0.99,
        "iters_per_output": 2,
        "test_episodes": 1,
        "steps_per_iter": 2,
    }


def test_replay_push_over_capacity_and_round_trip():
    buffer = experience_buffer.ExperienceBuffer(
        buffer_length=5, batch_size=1, device="cpu")
    values = torch.arange(14, dtype=torch.float32).view(7, 1, 2)
    buffer.push({"disc_obs": values})

    assert buffer.get_sample_count() == 5
    assert buffer.get_total_samples() == 7
    assert torch.equal(buffer.get_data("disc_obs"), values[-5:])

    restored = experience_buffer.ExperienceBuffer(
        buffer_length=5, batch_size=1, device="cpu")
    restored.load_state_dict(buffer.state_dict())
    assert restored.get_total_samples() == 7
    assert restored._buffer_head == buffer._buffer_head
    assert restored._sample_buf_head == buffer._sample_buf_head
    assert torch.equal(restored.get_data("disc_obs"),
                       buffer.get_data("disc_obs"))
    assert torch.equal(restored._sample_buf, buffer._sample_buf)


def test_full_checkpoint_restores_training_state_and_keeps_model_compatible(
        tmp_path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    agent = _TinyAgent(_config(), _TinyEnv(), "cpu")

    # Exercise exact continuation of Normalizer's unclamped second moment,
    # which is deliberately absent from the lightweight model state_dict.
    nearly_constant_obs = torch.tensor(
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    agent._obs_norm.record(nearly_constant_obs)
    agent._obs_norm.update()
    expected_mean_sq = agent._obs_norm._mean_sq.clone()

    loss = torch.square(agent._model(torch.ones(1, 3))).sum()
    agent._optimizer.step(loss)
    agent._iter = 7
    agent._sample_count = 1234
    agent._exp_buffer.set_total_samples(617)
    # Advance the on-policy permutation so a weights/optimizer-only restore
    # would choose different minibatches on its next update.
    agent._exp_buffer._sample_rand_idx(3)
    expected_next_sample_idx = agent._exp_buffer._sample_buf[
        agent._exp_buffer._sample_buf_head:].clone()
    agent._elapsed_train_time = 12.5
    agent._last_output_sample_count = 1234
    agent._last_output_wall_time = 12.5
    agent._last_test_info = {
        "mean_return": 2.0,
        "mean_ep_len": 9.0,
        "num_eps": 4,
    }
    agent._disc_buffer = experience_buffer.ExperienceBuffer(
        buffer_length=4, batch_size=1, device="cpu")
    replay_values = torch.randn(3, 1, 3)
    agent._disc_buffer.push({"disc_obs": replay_values})

    model_file = tmp_path / "model.pt"
    checkpoint_file = tmp_path / "checkpoint.pt"
    agent.save(model_file)
    agent.save_checkpoint(checkpoint_file, next_iter=8)
    expected_rng = (random.random(), np.random.rand(), torch.rand(1))

    # The deployment artifact remains a plain torch module state_dict.
    model_state = torch.load(model_file, map_location="cpu", weights_only=True)
    assert "checkpoint_version" not in model_state
    assert all(torch.is_tensor(val) for val in model_state.values())
    weights_only_agent = _TinyAgent(_config(), _TinyEnv(), "cpu")
    weights_only_agent.load(model_file)
    for before, after in zip(agent._model.parameters(),
                             weights_only_agent._model.parameters()):
        assert torch.equal(before, after)

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)

    restored = _TinyAgent(_config(), _TinyEnv(), "cpu")
    restored._disc_buffer = experience_buffer.ExperienceBuffer(
        buffer_length=4, batch_size=1, device="cpu")
    restored.resume(checkpoint_file)

    assert restored._iter == 8
    assert restored._sample_count == 1234
    assert restored._resume_pending is True
    assert restored._optimizer.get_steps() == agent._optimizer.get_steps()
    assert restored._elapsed_train_time == pytest.approx(12.5)
    assert restored._last_test_info == agent._last_test_info
    assert torch.equal(restored._obs_norm._mean_sq, expected_mean_sq)
    assert torch.equal(restored._disc_buffer.get_data("disc_obs"),
                       agent._disc_buffer.get_data("disc_obs"))
    for before, after in zip(agent._model.parameters(),
                             restored._model.parameters()):
        assert torch.equal(before, after)

    actual_rng = (random.random(), np.random.rand(), torch.rand(1))
    assert actual_rng[0] == expected_rng[0]
    assert actual_rng[1] == expected_rng[1]
    assert torch.equal(actual_rng[2], expected_rng[2])

    # Matching the next Adam update verifies moment estimates, not only the
    # optimizer's bookkeeping counter.
    update_obs = torch.tensor([[0.25, -0.5, 1.0]])
    update_target = torch.tensor([[0.1, -0.2]])
    original_loss = torch.square(
        agent._model(update_obs) - update_target).sum()
    restored_loss = torch.square(
        restored._model(update_obs) - update_target).sum()
    original_grad_norm = agent._optimizer.step(original_loss)
    restored_grad_norm = restored._optimizer.step(restored_loss)
    assert restored_grad_norm == pytest.approx(original_grad_norm)
    for before, after in zip(agent._model.parameters(),
                             restored._model.parameters()):
        assert torch.equal(before, after)

    restored._init_train()
    assert restored._iter == 8
    assert restored._sample_count == 1234
    assert restored._exp_buffer.get_total_samples() == 617
    restored_next_sample_idx = restored._exp_buffer._sample_buf[
        restored._exp_buffer._sample_buf_head:].clone()
    assert torch.equal(restored_next_sample_idx, expected_next_sample_idx)


def test_resume_rejects_weights_only_and_incompatible_num_envs(tmp_path):
    agent = _TinyAgent(_config(), _TinyEnv(num_envs=2), "cpu")
    model_file = tmp_path / "model.pt"
    checkpoint_file = tmp_path / "checkpoint.pt"
    agent.save(model_file)
    agent.save_checkpoint(checkpoint_file, next_iter=1)

    with pytest.raises(ValueError, match="weights-only"):
        agent.resume(model_file)

    incompatible = _TinyAgent(_config(), _TinyEnv(num_envs=3), "cpu")
    with pytest.raises(ValueError, match="num_envs mismatch"):
        incompatible.resume(checkpoint_file)


def test_resume_rejects_different_configuration_content(tmp_path):
    agent = _TinyAgent(_config(), _TinyEnv(), "cpu")
    agent.set_checkpoint_context({"env_config_sha256": "abc"})
    checkpoint_file = tmp_path / "checkpoint.pt"
    agent.save_checkpoint(checkpoint_file, next_iter=1)

    incompatible = _TinyAgent(_config(), _TinyEnv(), "cpu")
    incompatible.set_checkpoint_context({"env_config_sha256": "def"})
    with pytest.raises(ValueError, match="configuration mismatch"):
        incompatible.resume(checkpoint_file)


def test_text_logger_appends_without_truncating_or_duplicate_header(tmp_path):
    log_file = tmp_path / "log.txt"
    first = Logger()
    first.configure_output_file(log_file)
    first.log("Iteration", 0)
    first.log("Samples", 10)
    first.write_log()
    first.output_file.close()

    resumed = Logger()
    resumed.configure_output_file(log_file, append=True)
    resumed.log("Iteration", 1)
    resumed.log("Samples", 20)
    resumed.write_log()
    resumed.output_file.close()

    lines = log_file.read_text().splitlines()
    assert len(lines) == 3
    assert lines[0].split() == ["Iteration", "Samples"]
    assert lines[1].split() == ["0", "10"]
    assert lines[2].split() == ["1", "20"]

    changed = Logger()
    changed.configure_output_file(log_file, append=True)
    changed.log("Iteration", 2)
    changed.log("Samples", 30)
    changed.log("Resume_Count", 1)
    changed.write_log()
    changed.output_file.close()

    same_changed = Logger()
    same_changed.configure_output_file(log_file, append=True)
    same_changed.log("Iteration", 3)
    same_changed.log("Samples", 40)
    same_changed.log("Resume_Count", 2)
    same_changed.write_log()
    same_changed.output_file.close()

    lines = log_file.read_text().splitlines()
    assert lines[-3].split() == ["Iteration", "Samples", "Resume_Count"]
    assert lines[-2].split() == ["2", "30", "1"]
    assert lines[-1].split() == ["3", "40", "2"]


def test_policy_audit_computes_exact_gaussian_kl():
    class _FixedActor:
        def eval_actor(self, obs):
            return distribution_gaussian_diag.DistributionGaussianDiag(
                mean=obs[:, :2], logstd=torch.zeros_like(obs[:, :2]))

    agent = object.__new__(ppo_agent.PPOAgent)
    object.__setattr__(agent, "_model", _FixedActor())
    obs = torch.tensor([[0.25, -0.5, 1.0], [0.0, 0.75, -1.0]])
    old = {
        "mean": obs[:, :2].clone(),
        "logstd": torch.zeros(2, 2),
    }
    info = agent._compute_policy_audit(obs, old)
    assert info["policy_kl"].item() == pytest.approx(0.0, abs=1e-7)
    assert info["action_mean_update_rms"].item() == pytest.approx(0.0)
    assert info["action_mean_bound_frac"].item() == pytest.approx(0.0)
