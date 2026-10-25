import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools/diagnostics"))
from adaptive_phase_candidate import AdaptivePhaseSampler, CapacityAwarePhaseSampler, capped_quota_plan


@pytest.mark.parametrize("counts,batch", [([1, 100, 100], 12), ([1, 2, 40, 80], 32),
    ([1, 1, 1], 3), ([3, 7], 9), ([1, 1000], 999), ([1, 1, 10000], 10001)])
def test_no_useful_original_quota_is_removed(counts, batch):
    for cursor in range(len(counts)):
        for refund in range(len(counts)):
            quota, _, _, original = capped_quota_plan(counts, batch, cursor, refund)
            assert sum(quota) == batch
            assert all(1 <= m <= n for m, n in zip(quota, counts))
            assert all(m >= min(o, n) for m, o, n in zip(quota, original, counts))


@pytest.mark.parametrize("batch", [4, 7, 16, 124, 125, 2000])
def test_fixed_target_no_starvation_and_maximal_unique_count(batch):
    counts = [1, 3, 40, 80]
    phase = torch.cat([torch.full((n,), (b+.5)/4, dtype=torch.float64)
                       for b, n in enumerate(counts)])
    eligible = torch.ones(len(phase), dtype=torch.bool)
    sampler = CapacityAwarePhaseSampler(phase, frame_count=4, batch_budget=4, eligible=eligible)
    values = torch.arange(len(phase), dtype=torch.float64).sin()
    exact = (sampler.full_weights()*values).sum()
    for _ in range(40):
        idx, w, quota = sampler.sample(batch)
        assert len(idx) == batch
        assert len(idx.unique()) == min(batch, len(phase))
        assert (quota >= 1).all()
        mass = torch.zeros(4, dtype=torch.float64).scatter_add_(0, sampler.bin_ids[idx], w)
        assert torch.allclose(mass, sampler.target_mass, atol=1e-12, rtol=0)
        if batch >= len(phase):
            per_transition = torch.zeros(len(phase), dtype=torch.float64).scatter_add_(0, idx, w)
            assert torch.allclose(per_transition, sampler.full_weights(), atol=1e-12, rtol=0)
            assert torch.allclose((w*values[idx]).sum(), exact, atol=1e-12, rtol=0)


def test_identical_when_no_bin_repeats():
    phase = torch.linspace(0, 1, 1000, dtype=torch.float64)
    old = AdaptivePhaseSampler(phase, frame_count=10, batch_budget=30)
    new = CapacityAwarePhaseSampler(phase, frame_count=10, batch_budget=30)
    g0 = torch.Generator().manual_seed(123)
    g1 = torch.Generator().manual_seed(123)
    for _ in range(20):
        i0, w0, q0 = old.sample(33, generator=g0)
        i1, w1, q1 = new.sample(33, generator=g1)
        assert torch.equal(i0, i1)
        assert torch.equal(w0, w1)
        assert torch.equal(q0, q1)


def test_mask_is_preserved():
    phase = torch.linspace(0, 1, 1000, dtype=torch.float64)
    eligible = torch.arange(1000) % 3 == 0
    sampler = CapacityAwarePhaseSampler(phase, frame_count=299, batch_budget=100,
                                        eligible=eligible)
    for batch in [100, 200, 1000]:
        idx, w, _ = sampler.sample(batch)
        assert bool(eligible[idx].all())
        assert abs(float(w.sum())-1) < 1e-12


@pytest.mark.parametrize("counts,batch", [([], 1), ([0, 2], 2), ([1, 2], 1),
                                         ([1, 2], True), ([1., 2], 3)])
def test_invalid_contracts_fail(counts, batch):
    with pytest.raises(ValueError):
        capped_quota_plan(counts, batch)
