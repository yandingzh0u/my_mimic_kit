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
from learning.phase_sampler import PhaseSampler, control_cells
from learning.experience_buffer import ExperienceBuffer
from learning.ppo_agent import PPOAgent
from util.phase_coverage import PhaseCoverage
import envs.base_env as base_env
from envs.deepmimic_env import DeepMimicEnv


@pytest.mark.parametrize("duration,cells", [(0, 1), (1/300, 1), (1, 30), (299/30, 299), (100, 3000)])
def test_resolution(duration, cells):
    assert control_cells(duration, 1/30) == cells


@pytest.mark.parametrize("frames,budget", [(1, 1), (299, 32), (3000, 128), (10**9, 17)])
def test_no_duplicates_or_starvation_and_correct_mass(frames, budget):
    phase = torch.linspace(0, 1, 129, dtype=torch.float64)
    eligible = torch.arange(len(phase)) % 3 != 1
    sampler = PhaseSampler(phase, frames, budget, eligible)
    for b in (budget, budget + 7, 200):
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


def test_sparse_pool_refunds_are_used_by_other_phases():
    phase = torch.tensor([.01] + [.4]*50 + [.8]*50)
    sampler = PhaseSampler(phase, 3, 12)
    for _ in range(50):
        idx, weight, quota = sampler.sample(12)
        assert quota[0] == 1 and len(idx.unique()) == 12
        assert quota[1] >= 4 and quota[2] >= 4
        assert float(weight.sum()) == pytest.approx(1)


def test_final_weights_use_time_width_not_equal_merged_bin_mass():
    sampler = PhaseSampler(torch.tensor([.01, .21, .41, .61, .81]), 5, 4)
    torch.testing.assert_close(sampler.target_mass, torch.tensor([.4, .2, .2, .2], dtype=torch.float64))
    idx, weight, _ = sampler.sample(4)
    assert float(weight[sampler.bin_ids[idx] == 0].sum()) == pytest.approx(.4)


def test_empty_eligibility_can_skip_actor_without_nan():
    sampler = PhaseSampler(torch.linspace(0, 1, 10), 299, 8, torch.zeros(10, dtype=torch.bool))
    idx, w, q = sampler.sample(8)
    assert len(idx) == len(w) == len(q) == 0
    assert sampler.full_weights().sum() == 0
    assert sampler.coverage["occupied_bins"] == 0


@pytest.mark.parametrize("phase", [torch.tensor([]), torch.tensor([float("nan")]),
                                   torch.tensor([float("inf")]), torch.tensor([-.1]), torch.tensor([1.1])])
def test_invalid_population_is_rejected(phase):
    with pytest.raises(ValueError):
        PhaseSampler(phase, 299, 8)


def test_impossible_minibatch_is_rejected():
    sampler = PhaseSampler(torch.linspace(0, 1, 10), 10, 10)
    with pytest.raises(ValueError, match="cover all"):
        sampler.sample(9)


def test_monte_carlo_estimates_the_same_finite_population_target():
    torch.manual_seed(72)
    phase = torch.tensor([.01] + [.4]*7 + [.8]*21)
    sampler = PhaseSampler(phase, 3, 12)
    values = torch.randn(len(phase), 5, dtype=torch.float64)
    exact = (sampler.full_weights()[:, None] * values).sum(0)
    estimates = []
    for _ in range(6000):
        idx, weight, _ = sampler.sample(12)
        estimates.append((values[idx] * weight[:, None]).sum(0))
    estimates = torch.stack(estimates)
    se = estimates.std(0) / len(estimates)**.5
    assert bool(((estimates.mean(0) - exact).abs() < 5*se + 1e-12).all())


@pytest.mark.parametrize("cells", [1, 7, 299, 3000, 100000])
def test_coverage_and_exact_resume_after_float32_conversion(cells):
    torch.manual_seed(7)
    coverage = PhaseCoverage(cells)
    lengths = torch.full((2*cells+17,), cells/30, dtype=torch.float32)
    coverage.draw_times(lengths[:13])
    saved = coverage.state_dict()
    rng = torch.get_rng_state()
    times = coverage.draw_times(lengths)
    native = ((times/lengths).double()*cells).long().clamp_max(cells-1)
    for start in (0, 1, cells//2):
        assert len(native[start:start+cells].unique()) == cells
    replica = PhaseCoverage(cells)
    replica.load_state_dict(saved)
    torch.set_rng_state(rng)
    assert torch.equal(replica.draw_times(lengths), times)


class _Env:
    def get_num_envs(self): return 8
    def get_obs_space(self): return spaces.Box(-1, 1, (3,), dtype=np.float32)
    def get_action_space(self): return spaces.Box(-1, 1, (2,), dtype=np.float32)
    def get_motion_sampling_spec(self): return ([299/30], 1/30)
    def get_reward_bounds(self): return (0, 1)
    def get_reward_succ(self): return 1
    def get_reward_fail(self): return 0
    def enable_phase_coverage_resets(self, cells): self._coverage = PhaseCoverage(cells)
    def get_phase_coverage_state(self): return self._coverage.state_dict()
    def load_phase_coverage_state(self, state): self._coverage.load_state_dict(state)


def _agent(adaptive=True, coverage=False):
    config = yaml.safe_load((ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml").read_text())
    config.update(model=dict(actor_net="fc_2layers_128units", critic_net="fc_2layers_128units",
                             actor_init_output_scale=.01, actor_std_type="FIXED", action_std=.05),
                  actor_epochs=1, critic_epochs=1, steps_per_iter=4)
    if adaptive:
        config["phase_sampler"] = "adaptive_twostage"
        config["phase_coverage_reset"] = coverage
    with mock.patch("util.mp_util.get_num_procs", return_value=1):
        agent = PPOAgent(config, _Env(), "cpu")
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


def test_legacy_phase_balancing_and_default_sampling_unchanged():
    agent = _agent(adaptive=False)
    weights, _ = agent._build_phase_weights()
    torch.testing.assert_close(weights, torch.ones_like(weights))  # four observations per 8-bin phase
    with mock.patch.object(agent._exp_buffer, "sample", return_value={"original": True}) as sample:
        assert agent._sample_ppo_batch(16, actor=True) == {"original": True}
        sample.assert_called_once_with(16)


def test_indexed_gather_preserves_order():
    buffer = ExperienceBuffer(2, 3, "cpu")
    buffer.set_data("value", torch.arange(6).view(2, 3))
    torch.testing.assert_close(buffer.sample_indices(torch.tensor([5, 1, 4]))["value"], torch.tensor([5, 1, 4]))


def test_evaluations_do_not_advance_training_coverage():
    env = _Env()
    env._motion_lib = mock.Mock()
    env._motion_lib.sample_motions.side_effect = lambda n: torch.zeros(n, dtype=torch.long)
    env._motion_lib.get_motion_length.side_effect = lambda ids: torch.ones_like(ids, dtype=torch.float)
    env._motion_lib.sample_time.side_effect = lambda ids: torch.rand(len(ids))
    env._phase_coverage = PhaseCoverage(30)
    env._rand_reset, env._device = True, "cpu"
    env._mode, env._test_random_start = base_env.EnvMode.TEST, False
    state = env._phase_coverage.state_dict()
    _, t = DeepMimicEnv._sample_motion_times(env, 50)
    assert t.sum() == 0
    env._test_random_start = True
    DeepMimicEnv._sample_motion_times(env, 50)
    assert env._phase_coverage.state_dict() == state
    env._mode = base_env.EnvMode.TRAIN
    DeepMimicEnv._sample_motion_times(env, 50)
    assert env._phase_coverage.cursor == 20


def test_seeded_network_initialization_is_unchanged():
    torch.manual_seed(0)
    baseline = _agent(adaptive=False)
    torch.manual_seed(0)
    adaptive = _agent(adaptive=True, coverage=True)
    for a, b in zip(baseline._model.parameters(), adaptive._model.parameters()):
        assert torch.equal(a, b)


def test_agent_checkpoint_hook_restores_reset_coverage():
    agent = _agent(coverage=True)
    coverage = agent._env._coverage
    coverage.draw_times(torch.full((8217,), 299/30))
    state, rng = agent._get_checkpoint_extra_state(), torch.get_rng_state()
    expected = coverage.draw_times(torch.full((917,), 299/30))
    agent._load_checkpoint_extra_state(state)
    torch.set_rng_state(rng)
    assert torch.equal(coverage.draw_times(torch.full((917,), 299/30)), expected)


def test_new_benchmark_only_enables_sampler_and_coverage():
    baseline = yaml.safe_load((ROOT / "data/agents/dare_10layer_cpl_climb_agent.yaml").read_text())
    candidate = yaml.safe_load((ROOT / "data/agents/dare_10layer_cpl_climb_twostage_agent.yaml").read_text())
    assert candidate.pop("phase_sampler") == "adaptive_twostage"
    assert candidate.pop("phase_coverage_reset") is True
    assert baseline == candidate
