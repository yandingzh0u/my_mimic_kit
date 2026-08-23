import math

import torch

import learning.aligned_add_agent as aligned_add_agent
from learning.phase_scalarization import (
    compute_phase_importance_weights,
    compute_phase_statistics,
)
import util.mp_util as mp_util


def normalize_add_reward(disc_reward, reward_scale):
    """Express the stock ADD reward in its scale-free native units."""
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    return disc_reward / reward_scale


def compute_entropic_phase_probabilities(phase_utility, present, beta):
    """Closed-form adversarial distribution for the entropic soft minimum."""
    phase_utility = phase_utility.reshape(-1)
    present = present.reshape(-1).to(dtype=torch.bool)
    if phase_utility.shape != present.shape:
        raise ValueError("phase_utility and present must have equal shape")
    if beta <= 0.0:
        raise ValueError("et_beta must be positive")
    if not torch.any(present):
        raise ValueError("at least one phase must be observed")

    logits = -float(beta) * phase_utility
    logits = logits.masked_fill(~present, -torch.inf)
    return torch.softmax(logits, dim=0)


def compute_entropic_softmin(phase_utility, present, beta):
    """Return -log(mean(exp(-beta * J))) / beta over observed phases."""
    phase_utility = phase_utility.reshape(-1)
    present = present.reshape(-1).to(dtype=torch.bool)
    if phase_utility.shape != present.shape:
        raise ValueError("phase_utility and present must have equal shape")
    if beta <= 0.0:
        raise ValueError("et_beta must be positive")
    observed = phase_utility[present]
    if observed.numel() == 0:
        raise ValueError("at least one phase must be observed")

    log_num_phases = math.log(float(observed.numel()))
    return -(torch.logsumexp(-float(beta) * observed, dim=0)
             - log_num_phases) / float(beta)


class ETAlignedADDAgent(aligned_add_agent.AlignedADDAgent):
    """Entropy-regularized temporal scalarization of stock ADD rewards."""

    def __init__(self, config, env, device):
        super().__init__(config=config, env=env, device=device)
        self._et_num_phases = env.get_mm_num_phases()
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._et_beta = float(config["et_beta"])
        if self._et_beta <= 0.0:
            raise ValueError("et_beta must be positive")
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record("et_phase_idx", next_info["mm_phase_idx"])
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        phase_idx = self._exp_buffer.get_data_flat("et_phase_idx")

        obs_diff = disc_obs_demo - disc_obs
        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        # Use the inherited ADD reward verbatim. The phase adversary and PPO
        # therefore optimize the same utility rather than two logit mappings.
        disc_r = super()._calc_disc_rewards(norm_obs_diff)
        phase_sample_utility = normalize_add_reward(
            disc_r, self._disc_reward_scale)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        phase_sums, phase_counts = compute_phase_statistics(
            phase_sample_utility.detach(), phase_idx, self._et_num_phases)
        phase_sums = mp_util.reduce_sum(phase_sums)
        phase_counts = mp_util.reduce_sum(phase_counts)
        present = phase_counts > 0
        phase_utility = phase_sums / torch.clamp_min(phase_counts, 1.0)

        phase_probability = compute_entropic_phase_probabilities(
            phase_utility, present, self._et_beta)
        phase_weights = compute_phase_importance_weights(
            phase_probability, phase_counts, phase_idx)
        weighted_disc_r = disc_r * phase_weights
        weighted_reward_std, weighted_reward_mean = torch.std_mean(
            weighted_disc_r)

        r = (self._task_reward_weight * task_r
             + self._disc_reward_weight * weighted_disc_r)
        self._exp_buffer.set_data_flat("reward", r)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(obs_diff)

        observed_utility = phase_utility[present]
        observed_probability = phase_probability[present]
        tiny = torch.finfo(phase_probability.dtype).tiny
        entropy = -torch.sum(
            observed_probability
            * torch.log(torch.clamp_min(observed_probability, tiny)))
        num_present = torch.sum(present).to(dtype=phase_probability.dtype)
        normalized_entropy = torch.where(
            num_present > 1.0, entropy / torch.log(num_present),
            torch.ones_like(entropy))
        probability_ratio = (torch.max(observed_probability)
                             / torch.min(observed_probability))
        softmin = compute_entropic_softmin(
            phase_utility, present, self._et_beta)

        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "et_weighted_reward_mean": weighted_reward_mean,
            "et_weighted_reward_std": weighted_reward_std,
            "et_phase_utility_min": torch.min(observed_utility),
            "et_phase_utility_mean": torch.mean(observed_utility),
            "et_phase_utility_max": torch.max(observed_utility),
            "et_phase_utility_softmin": softmin,
            "et_phase_count_min": torch.min(phase_counts[present]),
            "et_phase_count_max": torch.max(phase_counts[present]),
            "et_phase_missing_fraction": 1.0 - torch.mean(present.float()),
            "et_reward_weight_mean": torch.mean(phase_weights),
            "et_reward_weight_std": torch.std(phase_weights),
            "et_reward_weight_max": torch.max(phase_weights),
            "et_probability_max": torch.max(observed_probability),
            "et_probability_ratio": probability_ratio,
            "et_probability_entropy": entropy,
            "et_probability_normalized_entropy": normalized_entropy,
            "et_probability_effective_support": torch.exp(entropy),
            "et_probability_argmax": torch.argmax(
                phase_probability).float(),
            "et_beta": torch.tensor(self._et_beta, device=self._device),
        }
        return info
