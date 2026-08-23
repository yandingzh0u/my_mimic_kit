import torch


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


def compute_phase_importance_weights(phase_probability, phase_counts,
                                     phase_idx):
    """Compute p_j/rho_j with unit rollout mean when p sums to one."""
    phase_probability = phase_probability.reshape(-1)
    phase_counts = phase_counts.reshape(-1)
    phase_idx = phase_idx.reshape(-1).to(dtype=torch.long)
    if phase_probability.shape != phase_counts.shape:
        raise ValueError("phase_probability and phase_counts must have equal shape")
    if torch.any(phase_counts[phase_idx] <= 0):
        raise ValueError("every sampled phase must have a positive count")

    total_count = torch.sum(phase_counts)
    if total_count <= 0:
        raise ValueError("phase_counts must contain samples")
    rollout_probability = phase_counts / total_count
    return (phase_probability[phase_idx]
            / rollout_probability[phase_idx])
