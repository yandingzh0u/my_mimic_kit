"""Coverage, finite-population target and actual PPO integration checks."""
from pathlib import Path
import sys
from unittest import mock

import gymnasium.spaces as spaces
import numpy as np
import pytest
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mimickit"))
from learning.phase_sampler import PhaseSampler, control_cells, supported_resolution
from learning.experience_buffer import ExperienceBuffer
from learning.ppo_agent import PPOAgent
from learning.dare_agent import DAREAgent


@pytest.mark.parametrize("duration,cells", [(0, 1), (1/300, 1), (1, 30), (299/30, 299), (100, 3000)])
def test_resolution(duration, cells):
    assert control_cells(duration, 1/30) == cells


@pytest.mark.parametrize("frames,budget", [(1, 1), (299, 32), (3000, 128), (10**9, 17)])
def test_no_duplicates_or_starvation_and_correct_mass(frames, budget):
    phase = torch.linspace(0, 1, 129, dtype=torch.float64)
    eligible = torch.arange(len(phase)) % 3 != 1
    for b in (budget, budget + 7, 200):
        k = supported_resolution(phase, frames, [b], [eligible])
        sampler = PhaseSampler(phase, k, eligible)
        idx, weight, quota = sampler.sample(b)
        assert len(idx) == min(b, int(eligible.sum()))
        assert idx.unique().numel() == len(idx)
        assert bool(eligible[idx].all())
        assert bool(((quota >= 1) & (quota <= sampler.counts)).all())
        assert int(quota.sum()) == len(idx)
        mass = torch.zeros(sampler.num_bins, dtype=torch.float64).scatter_add_(0, sampler.bin_ids[idx], weight)
        torch.testing.assert_close(mass[sampler.present_bins], sampler.target_mass)
        if b >= int(eligible.sum()):
            full = torch.zeros_like(phase).scatter_add_(0, idx, weight)
            torch.testing.assert_close(full, sampler.full_weights())


def test_under_supported_pools_are_rejected_instead_of_refunded():
    phase = torch.tensor([.01] + [.4]*50 + [.8]*50)
    sampler = PhaseSampler(phase, 3)
    with pytest.raises(ValueError, match="supported resolution"):
        sampler.sample(12)
    k = supported_resolution(phase, 3, [12], [None])
    indices, weights, _ = PhaseSampler(phase, k).sample(12)
    assert len(indices.unique()) == 12
    assert float(weights.sum()) == pytest.approx(1)


def test_supported_equal_width_bins_have_equal_target_mass():
    sampler = PhaseSampler(torch.tensor([.01]*4 + [.4]*6 + [.8]*10), 3)
    torch.testing.assert_close(sampler.target_mass, torch.full((3,), 1/3, dtype=torch.float64))
    idx, weight, _ = sampler.sample(9)
    for bin_id in range(3):
        assert float(weight[sampler.bin_ids[idx] == bin_id].sum()) == pytest.approx(1/3)


def test_empty_bins_require_resolution_selection():
    with pytest.raises(ValueError, match="supported resolution"):
        PhaseSampler(torch.tensor([0., .1, .9, 1.]), 4)


def test_empty_eligibility_can_skip_actor_without_nan():
    sampler = PhaseSampler(torch.linspace(0, 1, 10), 8, torch.zeros(10, dtype=torch.bool))
    idx, w, q = sampler.sample(8)
    assert len(idx) == len(w) == len(q) == 0
    assert sampler.full_weights().sum() == 0
    assert sampler.coverage["occupied_bins"] == 0


@pytest.mark.parametrize("phase", [torch.tensor([]), torch.tensor([float("nan")]),
                                   torch.tensor([float("inf")]), torch.tensor([-.1]), torch.tensor([1.1])])
def test_invalid_population_is_rejected(phase):
    with pytest.raises(ValueError):
        PhaseSampler(phase, 1)


def test_impossible_minibatch_is_rejected():
    sampler = PhaseSampler(torch.linspace(0, 1, 10), 10)
    with pytest.raises(ValueError, match="cover all"):
        sampler.sample(9)


def test_monte_carlo_estimates_the_same_finite_population_target():
    torch.manual_seed(72)
    phase = torch.tensor([.01]*5 + [.4]*7 + [.8]*21)
    sampler = PhaseSampler(phase, 3)
    values = torch.randn(len(phase), 5, dtype=torch.float64)
    exact = (sampler.full_weights()[:, None] * values).sum(0)
    estimates = []
    for _ in range(6000):
        idx, weight, _ = sampler.sample(12)
        estimates.append((values[idx] * weight[:, None]).sum(0))
    estimates = torch.stack(estimates)
    se = estimates.std(0) / len(estimates)**.5
    assert bool(((estimates.mean(0) - exact).abs() < 5*se + 1e-12).all())


class _Env:
    def get_num_envs(self): return 8
    def get_obs_space(self): return spaces.Box(-1, 1, (3,), dtype=np.float32)
    def get_action_space(self): return spaces.Box(-1, 1, (2,), dtype=np.float32)
    def get_motion_sampling_spec(self): return ([299/30], 1/30)
    def get_reward_bounds(self): return (0, 1)
    def get_reward_succ(self): return 1
    def get_reward_fail(self): return 0
    def get_disc_obs_space(self): return spaces.Box(-np.inf, np.inf, (172,), dtype=np.float32)

class _SamplingPPO(PPOAgent):
    # Exercise DARE's actual sampling methods with a small actor/critic.
    _build_phase_weights = DAREAgent._build_phase_weights
    _actor_population_weights = DAREAgent._actor_population_weights
    _sample_ppo_batch = DAREAgent._sample_ppo_batch

    def _load_params(self, config):
        super()._load_params(config)
        durations, dt = self._env.get_motion_sampling_spec()
        self._phase_control_cells = control_cells(durations[0], dt)


def _agent(agent_type=_SamplingPPO):
    config = yaml.safe_load((ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml").read_text())
    config["model"].update(actor_net="fc_2layers_128units", critic_net="fc_2layers_128units")
    config.update(actor_epochs=1, critic_epochs=1, steps_per_iter=4)
    with mock.patch("util.mp_util.get_num_procs", return_value=1):
        agent = agent_type(config, _Env(), "cpu")
    shape = (4, 8)
    obs = torch.randn(*shape, 3)
    dist = agent._model.eval_actor(agent._obs_norm.normalize(obs))
    actions = dist.sample().detach()
    for name, data in dict(obs=obs, next_obs=obs*.9, reward=torch.rand(*shape),
                           done=torch.zeros(shape, dtype=torch.int), diag_phase=torch.linspace(0, 1, 32).view(shape),
                           rand_action_mask=(torch.arange(32).view(shape) % 3 != 1).float(),
                           action=actions, a_logp=dist.log_prob(actions).detach()).items():
        agent._exp_buffer.set_data(name, data)
    agent._exp_buffer.set_total_samples(32)
    agent.eval()
    return agent


def test_actual_ppo_target_normalization_losses_and_updates():
    agent = _agent()
    with torch.no_grad():
        info = agent._build_train_data()
        raw_adv = agent._exp_buffer.get_data_flat("tar_val") - agent._model.eval_critic(
            agent._obs_norm.normalize(agent._exp_buffer.get_data_flat("obs"))).squeeze(-1)
        weights = agent._actor_phase_sampler.full_weights().float()
        mean = (weights*raw_adv).sum()
        torch.testing.assert_close(info["adv_mean"], mean)
    for actor, b in ((True, 32), (False, 16)):
        batch = agent._sample_ppo_batch(b, actor=actor)
        if actor:
            assert bool((batch["rand_action_mask"] == 1).all())
        assert len(batch["obs"]) == min(b, 21 if actor else 32)
        assert batch["phase_weight"].sum().item() == pytest.approx(1)
        loss = agent._compute_actor_loss(batch) if actor else agent._compute_critic_loss(batch)
        assert all(torch.isfinite(v).all() for v in loss.values())
    before = [p.detach().clone() for p in agent._model.parameters()]
    update = agent._update_model()
    assert all(torch.isfinite(v).all() for v in update.values())
    assert any(not torch.equal(a, b) for a, b in zip(before, agent._model.parameters()))


def test_zero_eligible_actor_is_skipped_and_critic_still_trains():
    agent = _agent()
    agent._exp_buffer.get_data("rand_action_mask").zero_()
    with torch.no_grad():
        info = agent._build_train_data()
    assert info["adv_mean"] == info["adv_std"] == 0
    update = agent._update_model()
    assert update["actor_skipped_no_eligible"] == 1
    assert torch.isfinite(update["critic_loss"])


def test_ordinary_ppo_remains_uniform():
    agent = _agent(agent_type=PPOAgent)
    weights, _ = agent._build_phase_weights()
    torch.testing.assert_close(weights, torch.ones_like(weights))
    with mock.patch.object(agent._exp_buffer, "sample", return_value={"original": True}) as sample:
        assert agent._sample_ppo_batch(16, actor=True) == {"original": True}
        sample.assert_called_once_with(16)


def test_indexed_gather_preserves_order():
    buffer = ExperienceBuffer(2, 3, "cpu")
    buffer.set_data("value", torch.arange(6).view(2, 3))
    torch.testing.assert_close(buffer.sample_indices(torch.tensor([5, 1, 4]))["value"], torch.tensor([5, 1, 4]))


def test_seeded_network_initialization_is_unchanged():
    torch.manual_seed(0)
    baseline = _agent(agent_type=PPOAgent)
    torch.manual_seed(0)
    adaptive = _agent()
    for a, b in zip(baseline._model.parameters(), adaptive._model.parameters()):
        assert torch.equal(a, b)


def test_dare_has_one_sampling_path_without_mode_parameters():
    agent = _agent(agent_type=DAREAgent)
    assert agent._phase_control_cells == 299
    config = yaml.safe_load((ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml").read_text())
    assert "phase_sampler" not in config
    assert "phase_sampler_bins" not in config
    assert "phase_coverage_reset" not in config


def test_selected_config_preserves_completed_supported_hyperparameters():
    current = yaml.safe_load((ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml").read_text())
    reference_path = ROOT / "output/paper_benchmark/dare_10layer_cpl_supported_climb_3000_8192_seed0_headless_20261025/agent_config.yaml"
    if not reference_path.is_file():
        pytest.skip("Historical run is not present in this checkout")
    reference = yaml.safe_load(reference_path.read_text())
    assert reference.pop("phase_sampler") == "adaptive_supported"
    assert reference.pop("phase_coverage_reset") is False
    assert current == reference
