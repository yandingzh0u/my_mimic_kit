"""Phasebalanced uses uniform minibatches with inverse-frequency weights."""
from pathlib import Path
from types import SimpleNamespace
import sys

import gymnasium.spaces as spaces
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "mimickit"))
from learning.distribution_gaussian_diag import DistributionGaussianDiag
from learning.experience_buffer import ExperienceBuffer
from learning.ppo_agent import PPOAgent
from learning.algorithms.ppo import PPO
from util import torch_util


class _Identity:
    def normalize(self, value):
        return value


class _Actor(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[0.18], [-0.02]]))

    def eval_actor(self, obs):
        mean = obs @ self.weight
        return DistributionGaussianDiag(mean, torch.full_like(mean, -1.0))


def _agent(counts=(1, 2, 3, 4, 5, 6, 7, 20)):
    agent = object.__new__(PPOAgent)
    torch.nn.Module.__init__(agent)
    agent._device = "cpu"
    agent._steps_per_iter = 1
    agent._env = SimpleNamespace(
        get_num_envs=lambda: sum(counts),
        get_action_space=lambda: spaces.Box(-1.0, 1.0, shape=(1,)))
    agent._exp_buffer = ExperienceBuffer(1, sum(counts), "cpu")
    agent._exp_buffer.set_total_samples(sum(counts))
    bins = torch.repeat_interleave(torch.arange(8), torch.tensor(counts))
    obs = torch.stack((bins.float() + 1, bins.float() * 0.2), dim=-1)
    agent._model = _Actor()
    agent._obs_norm = _Identity()
    agent._a_norm = _Identity()
    agent._algorithm = PPO({"ppo_clip_ratio": 0.2, "norm_adv_clip": 4.0})
    agent._action_bound_weight = 10.0
    agent._action_entropy_weight = 0.0
    agent._action_reg_weight = 0.0
    action = (bins.float() * 0.1 - 0.3).unsqueeze(-1)
    with torch.no_grad():
        old_logp = agent._model.eval_actor(obs).log_prob(action) + 0.03
    fields = {"obs": obs, "diag_phase": (bins.float() + 0.5) / 8,
              "sample_id": torch.arange(len(bins)), "action": action,
              "a_logp": old_logp, "adv": bins.float() * 0.3 - 0.7,
              "rand_action_mask": torch.ones(len(bins))}
    for name, values in fields.items():
        agent._exp_buffer.set_data(name, values.unsqueeze(0))
    weights, _ = agent._build_phase_weights()
    agent._exp_buffer.set_data("phase_weight", weights)
    return agent


@pytest.mark.parametrize("counts", ((1, 2, 3, 4, 5, 6, 7, 20),
                                   (1, 0, 0, 4, 0, 0, 0, 20)))
def test_phase_weights_equalize_only_occupied_bins(counts):
    agent = _agent(counts)
    weights, info = agent._build_phase_weights()
    bins = (agent._exp_buffer.get_data("diag_phase") * 8).long()
    present = sum(count > 0 for count in counts)
    torch.testing.assert_close(weights.mean(), torch.tensor(1.0))
    torch.testing.assert_close(info["phase_bins_present"], torch.tensor(float(present)))
    for phase_id, count in enumerate(counts):
        expected = sum(counts) / present if count else 0.0
        torch.testing.assert_close(weights[bins == phase_id].sum(), torch.tensor(expected))


def test_uniform_sampler_preserves_rollout_weights_and_sample_alignment():
    agent = _agent()
    weights = agent._exp_buffer.get_data_flat("phase_weight").clone()
    # A full permutation retains occupancy rather than imposing equal counts.
    batch = agent._exp_buffer.sample(weights.numel())
    bins = (batch["diag_phase"] * 8).long()
    assert torch.bincount(bins, minlength=8).tolist() == [1, 2, 3, 4, 5, 6, 7, 20]
    torch.testing.assert_close(batch["phase_weight"], weights[batch["sample_id"]])
    torch.testing.assert_close(batch["obs"],
                               agent._exp_buffer.get_data_flat("obs")[batch["sample_id"]])
    torch.testing.assert_close(agent._exp_buffer.get_data_flat("phase_weight"), weights)


def test_uniform_full_batch_matches_weighted_actor_loss_and_gradient():
    agent = _agent()
    full = dict(agent._exp_buffer._flat_buffers)
    expected = agent._compute_actor_loss(full)["actor_loss"]
    expected_grad = torch.autograd.grad(expected, agent._model.weight)[0]
    batch = agent._exp_buffer.sample(agent._exp_buffer.get_sample_count())
    actual = agent._compute_actor_loss(batch)["actor_loss"]
    actual_grad = torch.autograd.grad(actual, agent._model.weight)[0]
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_grad, expected_grad)


def test_non_motion_agent_has_unit_phase_weights():
    agent = _agent()
    del agent._exp_buffer._buffers["diag_phase"]
    del agent._exp_buffer._flat_buffers["diag_phase"]
    weights, info = agent._build_phase_weights()
    torch.testing.assert_close(weights, torch.ones_like(weights))
    torch.testing.assert_close(info["phase_bins_present"], torch.tensor(1.0))


def test_policy_audit_samples_all_timesteps_without_changing_training_rng():
    agent = _agent()
    agent._exp_buffer = ExperienceBuffer(32, 512, "cpu")
    ids = torch.arange(32 * 512).reshape(32, 512).float()
    agent._exp_buffer.set_data("obs", torch.stack((ids, ids * 0), dim=-1))
    agent._exp_buffer.set_data("diag_phase", ids / ids.numel())
    before = torch.get_rng_state().clone()
    obs, old_dist, phase = agent._build_policy_audit_batch()
    assert torch.equal(before, torch.get_rng_state())
    assert obs.shape == (4096, 2)
    assert len(torch.unique((obs[:, 0] / 512).long())) == 32
    torch.testing.assert_close(phase, obs[:, 0] / ids.numel())
    assert not old_dist["mean"].requires_grad
    assert not old_dist["logstd"].requires_grad
    info = agent._compute_policy_audit(obs, old_dist, phase)
    torch.testing.assert_close(info["policy_kl"], torch.tensor(0.0))


def test_metric_accumulators_are_independent_and_leave_loss_unchanged():
    metrics = {}
    for _ in range(32):
        loss = torch.tensor(0.2, requires_grad=True)
        torch_util.add_torch_dict({"disc_loss": loss,
                                  "disc_cls_loss": loss.detach()}, metrics)
        torch.testing.assert_close(loss, torch.tensor(0.2))
        loss.backward()
        torch.testing.assert_close(loss.grad, torch.tensor(1.0))
    torch_util.scale_torch_dict(1 / 32, metrics)
    for value in metrics.values():
        torch.testing.assert_close(value, torch.tensor(0.2))
        assert not value.requires_grad
    assert metrics["disc_loss"].data_ptr() != metrics["disc_cls_loss"].data_ptr()


def test_loss_logging_does_not_change_sgd_updates():
    final_parameters = []
    for log_enabled in (False, True):
        parameter = torch.nn.Parameter(torch.tensor(0.5))
        optimizer = torch.optim.SGD([parameter], lr=0.01, momentum=0.9)
        metrics, losses = {}, []
        for _ in range(32):
            optimizer.zero_grad()
            loss = (parameter - 0.75).square()
            losses.append(loss.detach().clone())
            loss.backward()
            optimizer.step()
            if log_enabled:
                torch_util.add_torch_dict({"disc_loss": loss,
                                          "disc_cls_loss": loss.detach()}, metrics)
        if log_enabled:
            torch_util.scale_torch_dict(1 / 32, metrics)
            for value in metrics.values():
                torch.testing.assert_close(value, torch.stack(losses).mean())
        final_parameters.append(parameter.detach().clone())
    assert torch.equal(*final_parameters)
