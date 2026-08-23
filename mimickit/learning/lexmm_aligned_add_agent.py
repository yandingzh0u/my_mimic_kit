import math

import torch

import learning.aligned_add_agent as aligned_add_agent
from learning.phase_scalarization import (
    compute_phase_importance_weights,
    compute_phase_statistics,
)
import util.mp_util as mp_util
import util.torch_util as torch_util


def lexicographic_concentration(horizon_iters):
    """Theory-selected exponential concentration c = log(T) / 4."""
    horizon_iters = int(horizon_iters)
    if horizon_iters <= 0:
        raise ValueError("lexmm_horizon_iters must be positive")
    return 0.25 * math.log(float(horizon_iters))


def add_reward_and_quality_from_logits(logits, reward_scale):
    """Return the stock ADD reward and its bounded discriminator quality."""
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    quality = torch.sigmoid(logits)
    # Match AMP/ADD's numerical cap exactly; the bounded quality itself remains
    # sigmoid(f) rather than inheriting the cap at very large positive logits.
    disc_reward = -torch.log(torch.clamp_min(1.0 - quality, 0.0001))
    disc_reward = reward_scale * disc_reward
    return disc_reward, quality


def compute_lexicographic_probabilities(phase_quality, present,
                                        horizon_iters):
    """Exponential lexicographic surrogate over current phase risks."""
    phase_quality = phase_quality.reshape(-1)
    present = present.reshape(-1).to(dtype=torch.bool)
    if phase_quality.shape != present.shape:
        raise ValueError("phase_quality and present must have equal shape")
    if not torch.any(present):
        raise ValueError("at least one phase must be observed")

    concentration = lexicographic_concentration(horizon_iters)
    risk_logits = concentration * (1.0 - phase_quality)
    risk_logits = risk_logits.masked_fill(~present, -torch.inf)
    return torch.softmax(risk_logits, dim=0)


class LexMMAlignedADDAgent(aligned_add_agent.AlignedADDAgent):
    """Lexicographic temporal scalarization over stock ADD rewards."""

    def __init__(self, config, env, device):
        super().__init__(config=config, env=env, device=device)
        self._lexmm_num_phases = env.get_mm_num_phases()
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._lexmm_horizon_iters = int(config["lexmm_horizon_iters"])
        self._lexmm_concentration = lexicographic_concentration(
            self._lexmm_horizon_iters)
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record("lexmm_phase_idx", next_info["mm_phase_idx"])
        return

    def _calc_disc_rewards_and_quality(self, norm_disc_obs):
        with torch.no_grad():
            disc_inputs = {"disc_obs": norm_disc_obs}
            logits = torch_util.eval_minibatch(
                self._model.eval_disc, disc_inputs,
                self._disc_eval_batch_size).squeeze(-1)
            return add_reward_and_quality_from_logits(
                logits, self._disc_reward_scale)

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        phase_idx = self._exp_buffer.get_data_flat("lexmm_phase_idx")

        obs_diff = disc_obs_demo - disc_obs
        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        disc_r, bounded_quality = self._calc_disc_rewards_and_quality(
            norm_obs_diff)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        # q=sigmoid(f) is the bounded discriminator quality associated with
        # stock ADD's s*softplus(f) reward, not a hand-designed physical metric.
        phase_sums, phase_counts = compute_phase_statistics(
            bounded_quality.detach(), phase_idx, self._lexmm_num_phases)
        phase_sums = mp_util.reduce_sum(phase_sums)
        phase_counts = mp_util.reduce_sum(phase_counts)
        present = phase_counts > 0
        phase_quality = phase_sums / torch.clamp_min(phase_counts, 1.0)

        phase_probability = compute_lexicographic_probabilities(
            phase_quality, present, self._lexmm_horizon_iters)
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

        observed_quality = phase_quality[present]
        observed_probability = phase_probability[present]
        tiny = torch.finfo(phase_probability.dtype).tiny
        entropy = -torch.sum(
            observed_probability
            * torch.log(torch.clamp_min(observed_probability, tiny)))
        num_present = torch.sum(present).to(dtype=phase_probability.dtype)
        normalized_entropy = torch.where(
            num_present > 1.0, entropy / torch.log(num_present),
            torch.ones_like(entropy))
        effective_support = torch.exp(entropy)
        probability_ratio = (torch.max(observed_probability)
                             / torch.min(observed_probability))

        info = {
            # Preserve stock names so the discriminator is directly auditable.
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "lexmm_weighted_reward_mean": weighted_reward_mean,
            "lexmm_weighted_reward_std": weighted_reward_std,
            "lexmm_phase_quality_min": torch.min(observed_quality),
            "lexmm_phase_quality_mean": torch.mean(observed_quality),
            "lexmm_phase_quality_max": torch.max(observed_quality),
            "lexmm_phase_count_min": torch.min(phase_counts[present]),
            "lexmm_phase_count_max": torch.max(phase_counts[present]),
            "lexmm_phase_missing_fraction": 1.0 - torch.mean(present.float()),
            "lexmm_reward_weight_mean": torch.mean(phase_weights),
            "lexmm_reward_weight_std": torch.std(phase_weights),
            "lexmm_reward_weight_max": torch.max(phase_weights),
            "lexmm_probability_max": torch.max(observed_probability),
            "lexmm_probability_ratio": probability_ratio,
            "lexmm_probability_ratio_bound": torch.tensor(
                math.exp(self._lexmm_concentration), device=self._device),
            "lexmm_probability_entropy": entropy,
            "lexmm_probability_normalized_entropy": normalized_entropy,
            "lexmm_probability_effective_support": effective_support,
            "lexmm_probability_argmax": torch.argmax(
                phase_probability).float(),
            "lexmm_concentration": torch.tensor(
                self._lexmm_concentration, device=self._device),
        }
        return info
