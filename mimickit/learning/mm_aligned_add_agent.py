import torch

import learning.aligned_add_agent as aligned_add_agent
import util.mp_util as mp_util


def compute_phase_statistics(reward, phase_idx, num_phases):
    """Return per-phase reward sums and counts for flat rollout tensors."""
    reward = reward.reshape(-1)
    phase_idx = phase_idx.reshape(-1).to(dtype=torch.long)
    if reward.shape[0] != phase_idx.shape[0]:
        raise ValueError("reward and phase_idx must contain the same samples")
    if num_phases <= 0:
        raise ValueError("num_phases must be positive")
    if phase_idx.numel() > 0:
        if torch.any(phase_idx < 0) or torch.any(phase_idx >= num_phases):
            raise ValueError("phase_idx is outside the configured phase range")

    sums = torch.zeros(num_phases, device=reward.device, dtype=reward.dtype)
    counts = torch.zeros(num_phases, device=reward.device, dtype=reward.dtype)
    sums.scatter_add_(0, phase_idx, reward)
    counts.scatter_add_(0, phase_idx, torch.ones_like(reward))
    return sums, counts


def compute_phase_importance_weights(phase_lambda, phase_counts, phase_idx):
    """Compute lambda_j/rho_j with unit rollout mean by construction."""
    phase_lambda = phase_lambda.reshape(-1)
    phase_counts = phase_counts.reshape(-1)
    phase_idx = phase_idx.reshape(-1).to(dtype=torch.long)
    if phase_lambda.shape != phase_counts.shape:
        raise ValueError("phase_lambda and phase_counts must have equal shape")
    if torch.any(phase_counts[phase_idx] <= 0):
        raise ValueError("every sampled phase must have a positive count")

    total_count = torch.sum(phase_counts)
    if total_count <= 0:
        raise ValueError("phase_counts must contain samples")
    phase_probability = phase_counts / total_count
    return phase_lambda[phase_idx] / phase_probability[phase_idx]


def exponentiated_dual_update(phase_lambda, phase_quality, present,
                              step_size, quality_scale):
    """Exponentiated-gradient ascent for the minimum phase objective.

    Lower-quality phases receive more mass. Missing phases are assigned the
    observed mean for this update, which leaves their relative log-weight
    unchanged instead of inventing an observation.
    """
    if step_size <= 0.0:
        raise ValueError("step_size must be positive")
    if quality_scale <= 0.0:
        raise ValueError("quality_scale must be positive")
    if not torch.any(present):
        raise ValueError("at least one phase must be observed")

    observed_mean = torch.mean(phase_quality[present])
    update_quality = torch.where(present, phase_quality, observed_mean)
    normalized_quality = update_quality / quality_scale
    normalized_quality = normalized_quality - torch.mean(normalized_quality)

    tiny = torch.finfo(phase_lambda.dtype).tiny
    log_lambda = torch.log(torch.clamp_min(phase_lambda, tiny))
    log_lambda = log_lambda - step_size * normalized_quality
    return torch.softmax(log_lambda, dim=0)


class MMAlignedADDAgent(aligned_add_agent.AlignedADDAgent):
    """Max-min phase-balanced policy optimization over stock ADD rewards."""

    def __init__(self, config, env, device):
        super().__init__(config=config, env=env, device=device)
        num_phases = env.get_mm_num_phases()
        phase_lambda = torch.full(
            (num_phases,), 1.0 / num_phases,
            device=self._device, dtype=torch.float32)
        # A buffer is saved by both model.pt and checkpoint.pt, so strict
        # continuation restores the dual state without a parallel mechanism.
        self.register_buffer("_mm_phase_lambda", phase_lambda)
        return

    def _load_params(self, config):
        super()._load_params(config)
        self._mm_dual_step_size = float(config["mm_dual_step_size"])
        if self._mm_dual_step_size <= 0.0:
            raise ValueError("mm_dual_step_size must be positive")
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record("mm_phase_idx", next_info["mm_phase_idx"])
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        phase_idx = self._exp_buffer.get_data_flat("mm_phase_idx")

        obs_diff = disc_obs_demo - disc_obs
        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        disc_r = self._calc_disc_rewards(norm_obs_diff)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        phase_sums, phase_counts = compute_phase_statistics(
            disc_r.detach(), phase_idx, self._mm_phase_lambda.numel())
        phase_sums = mp_util.reduce_sum(phase_sums)
        phase_counts = mp_util.reduce_sum(phase_counts)
        present = phase_counts > 0
        phase_quality = phase_sums / torch.clamp_min(phase_counts, 1.0)

        # Use the current dual iterate for this PPO rollout.  The update below
        # becomes the weight for the next rollout, avoiding a look-ahead step.
        phase_weights = compute_phase_importance_weights(
            self._mm_phase_lambda.detach(), phase_counts, phase_idx)
        weighted_disc_r = disc_r * phase_weights
        weighted_reward_std, weighted_reward_mean = torch.std_mean(
            weighted_disc_r)

        r = (self._task_reward_weight * task_r
             + self._disc_reward_weight * weighted_disc_r)
        self._exp_buffer.set_data_flat("reward", r)

        with torch.no_grad():
            next_lambda = exponentiated_dual_update(
                self._mm_phase_lambda, phase_quality, present,
                self._mm_dual_step_size, self._disc_reward_scale)
            self._mm_phase_lambda.copy_(next_lambda)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(obs_diff)

        observed_quality = phase_quality[present]
        entropy = -torch.sum(
            self._mm_phase_lambda
            * torch.log(torch.clamp_min(
                self._mm_phase_lambda,
                torch.finfo(self._mm_phase_lambda.dtype).tiny)))
        effective_support = torch.exp(entropy)
        normalized_entropy = entropy / torch.log(torch.tensor(
            float(self._mm_phase_lambda.numel()), device=self._device))

        info = {
            # Preserve the stock ADD names for direct log comparison.
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "mm_weighted_reward_mean": weighted_reward_mean,
            "mm_weighted_reward_std": weighted_reward_std,
            "mm_phase_quality_min": torch.min(observed_quality),
            "mm_phase_quality_mean": torch.mean(observed_quality),
            "mm_phase_quality_max": torch.max(observed_quality),
            "mm_phase_count_min": torch.min(phase_counts[present]),
            "mm_phase_count_max": torch.max(phase_counts[present]),
            "mm_phase_missing_fraction": 1.0 - torch.mean(present.float()),
            "mm_reward_weight_mean": torch.mean(phase_weights),
            "mm_reward_weight_std": torch.std(phase_weights),
            "mm_reward_weight_max": torch.max(phase_weights),
            "mm_lambda_max": torch.max(self._mm_phase_lambda),
            "mm_lambda_entropy": entropy,
            "mm_lambda_normalized_entropy": normalized_entropy,
            "mm_lambda_effective_support": effective_support,
            "mm_lambda_argmax": torch.argmax(self._mm_phase_lambda).float(),
        }
        return info
