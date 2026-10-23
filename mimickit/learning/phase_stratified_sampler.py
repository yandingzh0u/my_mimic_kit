"""Phase-stratified minibatch sampling for PPO.

The sampler changes only which rollout indices enter a minibatch.  For every
active phase bin it draws an equal quota and assigns each draw weight 1/m_k.
The existing weighted-mean losses then estimate the mean of the per-phase
means without adding another inverse-occupancy factor.
"""

from __future__ import annotations

import torch


class PhaseStratifiedSampler:
    def __init__(self, phase: torch.Tensor, num_bins: int = 8):
        if phase.ndim != 2 or num_bins <= 0:
            raise ValueError("phase must be [time, env] and num_bins positive")
        self._device = phase.device
        flat = torch.nan_to_num(phase.reshape(-1).to(torch.float32),
                                nan=0.0, posinf=1.0, neginf=0.0)
        flat = flat.clamp(0.0, 1.0 - torch.finfo(flat.dtype).eps)
        self._bins = (flat * int(num_bins)).to(torch.long)
        self._num_bins = int(num_bins)
        self._last = {}

    def sample(self, batch_size: int, eligible: torch.Tensor | None = None):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if eligible is None:
            eligible = torch.ones_like(self._bins, dtype=torch.bool)
        eligible = eligible.reshape(-1).to(device=self._device, dtype=torch.bool)
        if eligible.numel() != self._bins.numel():
            raise ValueError("eligible mask must match the rollout")

        active = [b for b in range(self._num_bins)
                  if bool(torch.any(eligible & (self._bins == b)))]
        if not active:
            raise ValueError("no eligible transition for phase sampling")

        if batch_size < len(active):
            # This branch is only for tiny smoke tests; normal PPO batches
            # are much larger than the number of phase bins.
            perm = torch.randperm(len(active), device=self._device)
            active = [active[int(i)] for i in perm[:batch_size]]
            quotas = [1] * len(active)
        else:
            base, remainder = divmod(int(batch_size), len(active))
            quotas = [base + (i < remainder) for i in range(len(active))]

        indices = []
        weights = []
        counts = []
        for bin_id, quota in zip(active, quotas):
            pool = torch.nonzero(eligible & (self._bins == bin_id),
                                 as_tuple=False).flatten()
            draw = pool[torch.randint(pool.numel(), (quota,), device=self._device)]
            indices.append(draw)
            weights.append(torch.full((quota,), 1.0 / float(quota),
                                      device=self._device, dtype=torch.float32))
            counts.append(quota)

        result_indices = torch.cat(indices, dim=0)
        result_weights = torch.cat(weights, dim=0)
        order = torch.randperm(result_indices.numel(), device=self._device)
        result_indices = result_indices[order]
        result_weights = result_weights[order]
        self._last = {
            "active_bins": len(active),
            "counts": counts,
            "max_reuse": int(torch.bincount(
                result_indices, minlength=self._bins.numel()).max()),
            "unsampled_fraction": float(
                1.0 - torch.unique(result_indices).numel() / self._bins.numel()),
        }
        return result_indices, result_weights

    def diagnostics(self):
        return dict(self._last)
