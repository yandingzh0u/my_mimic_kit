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


class EpochPhaseStratifiedSampler:
    """Phase-stratified sampling with full coverage per PPO epoch.

    Every eligible transition is used exactly once per epoch: each bin's pool is
    shuffled once per epoch and split into ``num_batches`` disjoint chunks, one
    chunk per minibatch.  All occupied bins appear in every minibatch and each
    bin contributes the same total weight, so the phase-uniform objective is
    unchanged while duplicated draws disappear.

    No tuning parameter is introduced: the number of batches per epoch follows
    from the rollout size and the minibatch size.
    """

    def __init__(self, phase: torch.Tensor, num_bins: int = 8,
                 eligible: torch.Tensor | None = None):
        if phase.ndim != 2 or num_bins <= 0:
            raise ValueError("phase must be [time, env] and num_bins positive")
        self._device = phase.device
        flat = torch.nan_to_num(phase.reshape(-1).to(torch.float32),
                                nan=0.0, posinf=1.0, neginf=0.0)
        flat = flat.clamp(0.0, 1.0 - torch.finfo(flat.dtype).eps)
        bins = (flat * int(num_bins)).to(torch.long)
        if eligible is None:
            eligible = torch.ones_like(bins, dtype=torch.bool)
        else:
            eligible = eligible.reshape(-1).to(device=self._device,
                                               dtype=torch.bool)
            if eligible.numel() != bins.numel():
                raise ValueError("eligible mask size mismatch")
        pools = [torch.nonzero(eligible & (bins == k), as_tuple=False).flatten()
                 for k in range(int(num_bins))]
        self._pools = [p for p in pools if p.numel() > 0]
        if not self._pools:
            raise ValueError("no eligible transitions")
        self._counts = [int(p.numel()) for p in self._pools]
        self.num_samples = int(sum(self._counts))
        # Usage counters are indexed by rollout position, not by pool slot.
        self._eligible = torch.cat(self._pools, dim=0)
        self._rare = min(range(len(self._pools)), key=lambda k: self._counts[k])
        self._num_batches = None
        self._batch_in_epoch = 0
        self._calls = 0
        self._chunks = None
        self._epochs = 0
        self._usage = torch.zeros(int(bins.numel()), device=self._device,
                                  dtype=torch.long)

    def _shuffle_epoch(self):
        batches = int(self._num_batches)
        self._chunks = []
        for pool in self._pools:
            perm = pool[torch.randperm(pool.numel(), device=self._device)]
            count = int(pool.numel())
            # Spread the bin over exactly `batches` chunks: base + one extra for
            # the first `count % batches` batches.  Every sample is used once.
            base, extra = divmod(count, batches)
            if base == 0 and extra < batches:
                raise ValueError(
                    "phase bin smaller than the number of batches per epoch")
            sizes = [base + (1 if i < extra else 0) for i in range(batches)]
            self._chunks.append(list(torch.split(perm, sizes)))
        self._batch_in_epoch = 0
        self._epochs += 1

    def sample(self, batch_size: int):
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self._num_batches is None:
            self._num_batches = int(-(-self.num_samples // int(batch_size)))
            for count in self._counts:
                if count < self._num_batches:
                    raise ValueError(
                        "phase bin has fewer samples than batches per epoch")
        if self._chunks is None or self._batch_in_epoch >= self._num_batches:
            self._shuffle_epoch()

        indices = []
        weights = []
        num_bins = len(self._pools)
        for bin_id, chunks in enumerate(self._chunks):
            chunk = chunks[self._batch_in_epoch]
            indices.append(chunk)
            weights.append(torch.full((chunk.numel(),),
                                      1.0 / float(num_bins * chunk.numel()),
                                      device=self._device, dtype=torch.float32))
        self._batch_in_epoch += 1
        self._calls += 1

        result_indices = torch.cat(indices, dim=0)
        result_weights = torch.cat(weights, dim=0)
        order = torch.randperm(result_indices.numel(), device=self._device)
        self._usage.index_add_(0, result_indices,
                               torch.ones_like(result_indices))
        return result_indices[order], result_weights[order]

    def diagnostics(self):
        rare_pool = self._pools[self._rare]
        eligible_usage = self._usage[self._eligible]
        used = eligible_usage > 0
        info = {
            "epochs": float(self._epochs),
            "batches_per_epoch": float(self._num_batches or 0),
            "num_batches": float(self._calls),
            "active_bins": float(len(self._pools)),
            "max_reuse": (float(eligible_usage.max().item())
                          if self.num_samples else 0.0),
            "unique_fraction": float(used.float().mean().item()),
            "rare_bin_count": float(self._counts[self._rare]),
            "rare_bin_mean_reuse": float(
                self._usage[rare_pool].float().mean().item()),
        }
        return info
