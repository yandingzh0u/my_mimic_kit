"""Reliability-shrunk phase balancing for PPO.

Each reference frame f gets a weight that blends uniform sampling with
per-frame equalisation, where the mixing coefficient is the statistical
reliability of that frame's policy-gradient signal rather than its sample
count:

    z_i    = A_i (a_i - mu(o_i)) / sigma(o_i)^2        # per-sample signal
    u_f    = (n_a n_b / n^2) ||z_a - z_b||^2           # two-group noise
    tau^2  = mean_f( ||z_f - z||^2 - u_f )_+           # between-frame signal
    alpha_f= tau^2 / (tau^2 + u_f)                     # reliability in [0, 1]
    w_i    = (1 - mean(alpha)) + N alpha_f / (F n_f)   # mean weight 1

Frames come from the reference motion length, so no grid, bin count or
bandwidth is chosen by hand.  With alpha = 0 everywhere this is plain PPO;
with alpha = 1 everywhere it is exact per-frame equalisation.
"""
import torch


@torch.no_grad()
def compute(phase, score, num_ref_frames):
    """Return per-transition weights (mean 1) and diagnostics."""
    frames = max(int(num_ref_frames), 1)
    steps, envs = phase.shape
    flat_phase = torch.nan_to_num(phase.reshape(-1).to(torch.float32),
                                  nan=0.0, posinf=1.0, neginf=0.0)
    index = (flat_phase.clamp(0.0, 1.0 - 1e-6) * frames).long().clamp(0, frames - 1)
    z = score.reshape(flat_phase.numel(), -1).to(torch.float32).detach()
    env_ids = torch.arange(envs, device=phase.device).repeat(steps)
    group_a = env_ids % 2 == 0
    group_b = ~group_a

    def aggregate(mask):
        ids = index[mask]
        counts = torch.bincount(ids, minlength=frames).to(z.dtype)
        sums = z.new_zeros((frames, z.shape[-1]))
        sums.index_add_(0, ids, z[mask])
        return counts, sums / counts.clamp_min(1.0)[:, None]

    n_a, mean_a = aggregate(group_a)
    n_b, mean_b = aggregate(group_b)
    n = n_a + n_b
    valid = n > 0
    both = (n_a > 0) & (n_b > 0)
    noise = torch.zeros(frames, device=z.device, dtype=z.dtype)
    if bool(both.any()):
        gap = ((mean_a[both] - mean_b[both]) ** 2).sum(-1)
        noise[both] = (n_a[both] * n_b[both] / n[both].square()) * gap
    mean_f = torch.zeros_like(mean_a)
    if bool(valid.any()):
        mean_f[valid] = ((mean_a[valid] * n_a[valid, None]
                          + mean_b[valid] * n_b[valid, None])
                         / n[valid, None])
        spread = ((mean_f[valid] - mean_f[valid].mean(0)) ** 2).sum(-1)
        tau2 = torch.clamp((spread - noise[valid]).mean(), min=0.0)
    else:
        tau2 = torch.zeros((), device=z.device, dtype=z.dtype)
    alpha = torch.zeros(frames, device=z.device, dtype=z.dtype)
    if float(tau2) > 0.0 and bool(both.any()):
        alpha[both] = tau2 / (tau2 + noise[both])
    alpha_mean = alpha.sum() / float(frames)
    weight = ((1.0 - alpha_mean)
              + (float(z.shape[0]) / (frames * n[index].clamp_min(1.0))) * alpha[index])
    weight = torch.where(valid[index], weight, torch.ones_like(weight))
    weight = weight / weight.mean()
    total = float(weight.numel())
    ess = float(weight.sum() ** 2 / torch.sum(weight * weight))
    info = {
        "phase_weight_mean": weight.mean(),
        "phase_weight_max": weight.max(),
        "phase_bins_present": torch.tensor(float(valid.sum()), device=z.device),
        "phase_alpha_mean": alpha_mean,
        "phase_tau2": tau2,
        "phase_ess": torch.tensor(ess / total, device=z.device),
    }
    return weight.reshape(steps, envs), info
