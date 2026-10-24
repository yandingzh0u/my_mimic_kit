import torch

from learning.phase_stratified_sampler import EpochPhaseStratifiedSampler


def test_full_coverage_without_replacement():
    phase = torch.tensor([
        [0.01, 0.26, 0.51, 0.76],
        [0.03, 0.31, 0.55, 0.79],
    ])
    sampler = EpochPhaseStratifiedSampler(phase, num_bins=4)
    for _ in range(3):
        seen = []
        for _ in range(2):
            indices, weights = sampler.sample(4)
            bins = (phase.flatten()[indices] * 4).long()
            for k in range(4):
                mask = bins == k
                assert bool(mask.any())
                torch.testing.assert_close(
                    weights[mask].sum(), torch.tensor(0.25))
            seen.extend(indices.tolist())
        assert len(seen) == 8
        assert len(set(seen)) == 8
    train_info = sampler.diagnostics()
    assert train_info["max_reuse"] == 3.0
    assert train_info["unique_fraction"] == 1.0
    assert train_info["active_bins"] == 4.0


def test_uneven_bins_still_cover_everything():
    # Bin 0 has 6 samples, bins 1-3 have 4 each: 18 samples, batch 6 -> 3 batches.
    phase = torch.tensor([[
        0.01, 0.02, 0.03, 0.04, 0.05, 0.06,
        0.30, 0.31, 0.32, 0.33,
        0.55, 0.56, 0.57, 0.58,
        0.80, 0.81, 0.82, 0.83,
    ]])
    sampler = EpochPhaseStratifiedSampler(phase, num_bins=4)
    seen = []
    for _ in range(3):
        indices, weights = sampler.sample(6)
        seen.extend(indices.tolist())
        bins = (phase.flatten()[indices] * 4).long()
        for k in range(4):
            mask = bins == k
            assert bool(mask.any())
            torch.testing.assert_close(weights[mask].sum(), torch.tensor(0.25))
    assert len(seen) == 18
    assert len(set(seen)) == 18
    assert sampler.diagnostics()["unique_fraction"] == 1.0


def test_eligible_mask_is_respected():
    phase = torch.tensor([[0.01, 0.26, 0.51, 0.76]])
    eligible = torch.tensor([True, False, True, False])
    sampler = EpochPhaseStratifiedSampler(phase, num_bins=4, eligible=eligible)
    assert sampler.num_samples == 2
    indices, _ = sampler.sample(2)
    assert sorted(indices.tolist()) == [0, 2]
