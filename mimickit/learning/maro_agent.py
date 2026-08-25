import torch

import learning.add_agent as add_agent
from learning.phase_scalarization import (
    compute_phase_statistics,
)
import util.mp_util as mp_util


def normalize_adversarial_reward(disc_reward, reward_scale):
    """Express the learned tracking reward in scale-free native units."""
    if reward_scale <= 0.0:
        raise ValueError("reward_scale must be positive")
    return disc_reward / reward_scale


def _normalize_phase_occupancy(phase_occupancy, reference):
    phase_occupancy = phase_occupancy.reshape(-1).to(
        device=reference.device, dtype=reference.dtype)
    if not torch.all(torch.isfinite(phase_occupancy)):
        raise ValueError("phase_occupancy must be finite")
    if torch.any(phase_occupancy < 0.0):
        raise ValueError("phase_occupancy must be nonnegative")
    total = torch.sum(phase_occupancy)
    if total <= 0.0:
        raise ValueError("at least one phase must be observed")
    return phase_occupancy / total


def compute_phase_density(phase_utility, phase_occupancy, beta):
    """Return the log-stable adversarial density p/rho on observed phases."""
    phase_utility = phase_utility.reshape(-1)
    if not torch.all(torch.isfinite(phase_utility)):
        raise ValueError("phase_utility must be finite")
    phase_occupancy = _normalize_phase_occupancy(
        phase_occupancy, phase_utility)
    if phase_utility.shape != phase_occupancy.shape:
        raise ValueError(
            "phase_utility and phase_occupancy must have equal shape")
    if beta <= 0.0:
        raise ValueError("maro_beta must be positive")

    present = phase_occupancy > 0.0
    log_normalizer = torch.logsumexp(
        torch.log(phase_occupancy[present])
        - float(beta) * phase_utility[present], dim=0)
    density = torch.zeros_like(phase_utility)
    density[present] = torch.exp(
        -float(beta) * phase_utility[present] - log_normalizer)
    return density


def compute_phase_probabilities(phase_utility, phase_occupancy, beta):
    """Solve the occupancy-anchored entropy-regularized inner problem."""
    phase_utility = phase_utility.reshape(-1)
    phase_occupancy = _normalize_phase_occupancy(
        phase_occupancy, phase_utility)
    density = compute_phase_density(
        phase_utility, phase_occupancy, beta)
    return phase_occupancy * density


def compute_entropic_softmin(phase_utility, phase_occupancy, beta):
    """Return the occupancy-anchored entropic phase utility."""
    phase_utility = phase_utility.reshape(-1)
    phase_occupancy = _normalize_phase_occupancy(
        phase_occupancy, phase_utility)
    if phase_utility.shape != phase_occupancy.shape:
        raise ValueError(
            "phase_utility and phase_occupancy must have equal shape")
    if beta <= 0.0:
        raise ValueError("maro_beta must be positive")

    present = phase_occupancy > 0.0
    log_terms = (torch.log(phase_occupancy[present])
                 - float(beta) * phase_utility[present])
    return -torch.logsumexp(log_terms, dim=0) / float(beta)


def compute_discounted_rollout_return(reward, discount):
    """Return each environment's truncated discounted sum for a [T, N] batch."""
    if reward.ndim != 2:
        raise ValueError("reward must have shape [time, environments]")
    if not 0.0 <= discount <= 1.0:
        raise ValueError("discount must lie in [0, 1]")
    steps = torch.arange(
        reward.shape[0], device=reward.device, dtype=reward.dtype)
    weights = torch.pow(
        torch.as_tensor(discount, device=reward.device, dtype=reward.dtype),
        steps)
    return torch.sum(reward * weights.unsqueeze(-1), dim=0)


def one_based_phase_argmax(phase_probability):
    """Return the paper-facing, one-based id of the most emphasized phase."""
    phase_probability = phase_probability.reshape(-1)
    if phase_probability.numel() == 0:
        raise ValueError("phase_probability must be nonempty")
    return torch.argmax(phase_probability) + 1


class MAROAgent(add_agent.ADDAgent):
    """Optimize learned tracking rewards across reference-motion phases."""

    def __init__(self, config, env, device):
        super().__init__(config=config, env=env, device=device)
        self._maro_num_phases = env.get_maro_num_phases()
        self._maro_num_phases_is_auto = env.get_maro_num_phases_is_auto()
        return

    def _get_return_log_keys(self):
        # BaseAgent's online trackers see the environment diagnostic reward,
        # before MARO constructs the reward sequence consumed by PPO.
        return ("Maro_Environment_Train_Return",
                "Maro_Environment_Test_Return")

    def _load_params(self, config):
        super()._load_params(config)
        phase_prior = str(config.get("maro_phase_prior", ""))
        if phase_prior != "occupancy":
            raise ValueError(
                "maro_phase_prior must be 'occupancy' for MARO")
        self._maro_phase_prior = phase_prior
        self._maro_beta = float(config["maro_beta"])
        if self._maro_beta <= 0.0:
            raise ValueError("maro_beta must be positive")
        return

    def _record_data_post_step(self, next_obs, r, done, next_info):
        super()._record_data_post_step(next_obs, r, done, next_info)
        self._exp_buffer.record(
            "maro_phase_idx", next_info["maro_phase_idx"])
        return

    def _compute_rewards(self):
        task_r = self._exp_buffer.get_data_flat("reward")
        disc_obs = self._exp_buffer.get_data_flat("disc_obs")
        disc_obs_demo = self._exp_buffer.get_data_flat("disc_obs_demo")
        phase_idx = self._exp_buffer.get_data_flat("maro_phase_idx")

        obs_diff = disc_obs_demo - disc_obs
        norm_obs_diff = self._disc_obs_norm.normalize(obs_diff)
        # The policy and phase objective consume exactly the same learned
        # adversarial reward; no tracking, contact, or completion term enters.
        disc_r = super()._calc_disc_rewards(norm_obs_diff)
        phase_sample_utility = normalize_adversarial_reward(
            disc_r, self._disc_reward_scale)
        disc_reward_std, disc_reward_mean = torch.std_mean(disc_r)

        phase_sums, phase_counts = compute_phase_statistics(
            phase_sample_utility.detach(), phase_idx,
            self._maro_num_phases)
        phase_sums = mp_util.reduce_sum(phase_sums)
        phase_counts = mp_util.reduce_sum(phase_counts)
        present = phase_counts > 0
        phase_utility = phase_sums / torch.clamp_min(phase_counts, 1.0)
        phase_occupancy = phase_counts / torch.sum(phase_counts)

        phase_probability = compute_phase_probabilities(
            phase_utility, phase_occupancy, self._maro_beta)
        phase_density = compute_phase_density(
            phase_utility, phase_occupancy, self._maro_beta)
        # Use the closed-form density directly instead of dividing two small
        # probabilities.  This is exactly p_j/rho_j on the observed support.
        phase_weights = phase_density[phase_idx]
        if not torch.all(torch.isfinite(phase_weights)):
            raise FloatingPointError("MARO produced non-finite phase weights")
        weighted_disc_r = disc_r * phase_weights
        if not torch.all(torch.isfinite(weighted_disc_r)):
            raise FloatingPointError("MARO produced non-finite policy rewards")
        weighted_reward_std, weighted_reward_mean = torch.std_mean(
            weighted_disc_r)

        r = (self._task_reward_weight * task_r
             + self._disc_reward_weight * weighted_disc_r)
        self._exp_buffer.set_data_flat("reward", r)

        ppo_reward_std, ppo_reward_mean = torch.std_mean(r)
        rollout_reward = self._exp_buffer.get_data("reward")
        discounted_rollout_return = compute_discounted_rollout_return(
            rollout_reward, self._discount)
        (discounted_rollout_return_std,
         discounted_rollout_return_mean) = torch.std_mean(
             discounted_rollout_return)

        if self._need_normalizer_update():
            self._disc_obs_norm.record(obs_diff)

        observed_utility = phase_utility[present]
        observed_probability = phase_probability[present]
        observed_density = phase_density[present]
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
        density_ratio = (torch.max(observed_density)
                         / torch.min(observed_density))
        softmin = compute_entropic_softmin(
            phase_utility, phase_occupancy, self._maro_beta)

        info = {
            "disc_reward_mean": disc_reward_mean,
            "disc_reward_std": disc_reward_std,
            "maro_weighted_reward_mean": weighted_reward_mean,
            "maro_weighted_reward_std": weighted_reward_std,
            "maro_ppo_consumed_reward_mean": ppo_reward_mean,
            "maro_ppo_consumed_reward_std": ppo_reward_std,
            "maro_ppo_consumed_discounted_rollout_return_mean": (
                discounted_rollout_return_mean),
            "maro_ppo_consumed_discounted_rollout_return_std": (
                discounted_rollout_return_std),
            "maro_phase_utility_min": torch.min(observed_utility),
            "maro_phase_utility_mean": torch.mean(observed_utility),
            "maro_phase_utility_max": torch.max(observed_utility),
            "maro_phase_utility_softmin": softmin,
            "maro_phase_count_min": torch.min(phase_counts[present]),
            "maro_phase_count_max": torch.max(phase_counts[present]),
            "maro_phase_missing_fraction": 1.0 - torch.mean(present.float()),
            "maro_reward_weight_mean": torch.mean(phase_weights),
            "maro_reward_weight_std": torch.std(phase_weights),
            "maro_reward_weight_max": torch.max(phase_weights),
            "maro_probability_max": torch.max(observed_probability),
            "maro_probability_ratio": probability_ratio,
            "maro_adversarial_density_ratio": density_ratio,
            "maro_probability_entropy": entropy,
            "maro_probability_normalized_entropy": normalized_entropy,
            "maro_probability_effective_support": torch.exp(entropy),
            # Paper phase indices are one-based.  Keep that convention in the
            # public log field rather than exposing the tensor's zero-based id.
            "maro_probability_argmax_phase": (
                one_based_phase_argmax(phase_probability).float()),
            "maro_adversarial_argmax_phase": (
                one_based_phase_argmax(phase_density).float()),
            "maro_beta": torch.tensor(self._maro_beta, device=self._device),
            "maro_num_phases": torch.tensor(
                self._maro_num_phases, device=self._device),
            "maro_num_phases_auto": torch.tensor(
                float(self._maro_num_phases_is_auto), device=self._device),
            "maro_occupancy_anchored": torch.ones(
                (), device=self._device),
        }
        return info
