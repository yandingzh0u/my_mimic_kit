"""Continuous phase-density balancing for PPO.

The old scheme split [0, 1) into a fixed number of bins and weighted each
sample by N / (K * n_k). That is a histogram density estimate with a hand-set
resolution. Here the same objective (each moment of the motion contributes
equally) is computed from a smoothed density estimate:

    q(phi) = smoothed rollout occupancy
    w_i    = (1 / q(phi_i)) / mean(1 / q)

The smoothing width is chosen by cross-validation over disjoint environment
groups, so the resolution comes from the data instead of a fixed bin count.
Boundaries are reflected, never wrapped: a non-cyclic motion does not connect
its end to its start.
"""
import torch

GRID = 512
BANDWIDTHS = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0)
EPS = 1e-9


def _gaussian_smooth(counts, sigma):
    if sigma <= 0.0:
        return counts
    radius = max(1, int(3.0 * sigma))
    offsets = torch.arange(-radius, radius + 1, device=counts.device,
                           dtype=counts.dtype)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    kernel = kernel / kernel.sum()
    index = torch.arange(-radius, counts.numel() + radius, device=counts.device)
    index = index.abs()
    index = torch.where(index >= counts.numel(),
                        2 * counts.numel() - 2 - index, index)
    padded = counts[index.clamp(0, counts.numel() - 1)]
    return torch.nn.functional.conv1d(
        padded.view(1, 1, -1), kernel.view(1, 1, -1)).view(-1)


def _density(counts, samples, sigma):
    smoothed = _gaussian_smooth(counts, sigma)
    return smoothed / max(samples, 1.0)


def compute(phase, bandwidths=BANDWIDTHS, grid=GRID):
    """Return per-transition weights (mean 1) and diagnostics."""
    shape = phase.shape
    flat = torch.nan_to_num(phase.flatten().to(torch.float32), nan=0.0,
                            posinf=1.0, neginf=0.0).clamp(0.0, 1.0 - 1e-6)
    num = flat.numel()
    index = (flat * grid).long().clamp(0, grid - 1)
    if shape.numel() == num and phase.dim() == 2:
        env = torch.arange(phase.shape[1], device=phase.device).repeat(
            phase.shape[0]).to(phase.device)
        group_a = (env % 2 == 0)
        group_b = ~group_a
    else:
        group_a = torch.arange(num, device=phase.device) % 2 == 0
        group_b = ~group_a
    counts_a = torch.bincount(index[group_a], minlength=grid).to(torch.float64)
    counts_b = torch.bincount(index[group_b], minlength=grid).to(torch.float64)
    n_a = float(group_a.sum())
    n_b = float(group_b.sum())
    best_sigma, best_score = None, None
    for sigma in bandwidths:
        dens_a = _density(counts_a, n_a, sigma)[index[group_b]]
        dens_b = _density(counts_b, n_b, sigma)[index[group_a]]
        score = (torch.log(dens_a + EPS).mean()
                 + torch.log(dens_b + EPS).mean()) * 0.5
        score = float(score)
        if best_score is None or score > best_score:
            best_score, best_sigma = score, sigma
    counts = (counts_a + counts_b).to(torch.float64)
    dens = _density(counts, float(num), best_sigma)
    weight = 1.0 / dens.clamp_min(EPS)
    weight = weight[index]
    weight = weight / weight.mean()
    total = float(weight.numel())
    ess = float(weight.sum() ** 2 / torch.sum(weight * weight))
    info = {
        "phase_weight_mean": weight.mean(),
        "phase_weight_max": weight.max(),
        "phase_bins_present": torch.tensor(float((counts > 0).sum()),
                                           device=phase.device),
        "phase_bandwidth": torch.tensor(best_sigma / grid, device=phase.device),
        "phase_ess": torch.tensor(ess / total, device=phase.device),
    }
    return weight.reshape(shape), info
