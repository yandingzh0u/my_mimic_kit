"""Invariant tests for the isolated, opt-in research candidate."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools/diagnostics"))
from adaptive_phase_candidate import AdaptivePhaseSampler, PhaseCoverageRotor, control_cells


@pytest.mark.parametrize("duration,expected", [(1/300, 1), (1, 30), (299/30, 299), (100, 3000)])
def test_control_resolution(duration, expected):
    assert control_cells(duration, 1/30) == expected


@pytest.mark.parametrize("frames,budget", [(1, 1), (299, 32768), (3000, 7), (10**9, 17)])
def test_quota_and_weights(frames, budget):
    phase = torch.linspace(0, 1, 129, dtype=torch.float64)
    sampler = AdaptivePhaseSampler(phase, frame_count=frames, batch_budget=budget)
    extras = torch.zeros(sampler.num_present, dtype=torch.long)
    for _ in range(19):
        idx, weight, quota = sampler.sample(budget)
        count = torch.bincount(sampler.bin_ids[idx], minlength=sampler.num_bins)
        assert (count[sampler.present_bins] >= 1).all()
        assert int(count.sum()) == budget
        assert int(quota.max() - quota.min()) <= 1
        assert abs(float(weight.sum()) - 1) < 1e-12
        for j, b in enumerate(sampler.present_bins.tolist()):
            assert abs(float(weight[sampler.bin_ids[idx] == b].sum()) - float(sampler.target_mass[j])) < 1e-12
            use = torch.bincount(idx[sampler.bin_ids[idx] == b], minlength=len(phase))
            pool = use[sampler.bin_ids == b]
            assert int(pool.max() - pool.min()) <= 1
        extras += quota - budget // sampler.num_present
        assert int(extras.max() - extras.min()) <= 1


def test_mask_empty_cells_and_endpoints_are_explicit():
    phase = torch.tensor([0., .01, .02, .99, 1.], dtype=torch.float64)
    eligible = torch.tensor([True, False, True, False, True])
    sampler = AdaptivePhaseSampler(phase, frame_count=299, batch_budget=100, eligible=eligible)
    idx, _, _ = sampler.sample(100)
    assert eligible[idx].all()
    assert sampler.coverage["missing_control_cells"] == 296
    assert sampler.coverage["empty_bins"] > 0
    assert sampler.bin_ids[-1] == sampler.num_bins - 1


@pytest.mark.parametrize("phase", [torch.tensor([]), torch.tensor([float('nan')]),
                                  torch.tensor([float('inf')]), torch.tensor([-.01]),
                                  torch.tensor([1.01]), torch.tensor([1])])
def test_invalid_data_rejected(phase):
    with pytest.raises(ValueError):
        AdaptivePhaseSampler(phase, frame_count=299, batch_budget=10)


def test_impossible_coverage_is_not_silently_claimed():
    sampler = AdaptivePhaseSampler(torch.linspace(0, 1, 10), frame_count=10, batch_budget=10)
    with pytest.raises(ValueError, match="cover every"):
        sampler.sample(9)
    with pytest.raises(ValueError, match="no eligible"):
        AdaptivePhaseSampler(torch.ones(10), frame_count=10, batch_budget=10,
                             eligible=torch.zeros(10, dtype=torch.bool))


@pytest.mark.parametrize("frames", [1, 2, 7, 299, 1024, 10007])
def test_rotor_any_window_coverage_and_resume(frames):
    rotor = PhaseCoverageRotor(frames, seed=123)
    a, ia = rotor.draw(frames // 3)
    state = rotor.state_dict()
    b, ib = rotor.draw(3 * frames + 7)
    replica = PhaseCoverageRotor(frames, seed=999)
    replica.load_state_dict(state)
    rb, rib = replica.draw(3 * frames + 7)
    assert torch.equal(rb, b) and torch.equal(rib, ib)
    for start in [0, 1, frames // 2, frames + 1]:
        assert len(torch.unique(ib[start:start + frames])) == frames
    assert torch.equal(torch.floor(b * frames).long(), ib)
    assert ((a >= 0) & (a < 1)).all()


def test_exact_average_when_quota_is_multiple_of_pool():
    phase = torch.tensor([.01, .02, .03, .60, .90, .91], dtype=torch.float64)
    values = torch.tensor([1., 2., 9., 21., -4., 7.], dtype=torch.float64)
    sampler = AdaptivePhaseSampler(phase, frame_count=3, batch_budget=18)
    exact = (sampler.full_weights() * values).sum()
    for _ in range(100):
        idx, weight, _ = sampler.sample(18)
        assert torch.allclose((weight * values[idx]).sum(), exact, atol=1e-12, rtol=0)


def test_merged_bins_use_time_width_and_rotor_covers_every_bin():
    # Five native cells cannot be divided into four equal-width cell groups.
    rotor = PhaseCoverageRotor(5, seed=17)
    phase, native = rotor.draw(5)
    sampler = AdaptivePhaseSampler(phase, frame_count=5, batch_budget=4)
    assert sampler.coverage["empty_bins"] == 0
    assert torch.allclose(sampler.target_mass, torch.tensor([.4, .2, .2, .2], dtype=torch.float64))
    assert torch.equal(sampler.bin_ids, native * 4 // 5)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_low_precision_endpoints_and_long_motion(dtype):
    phase = torch.tensor([0, .001, .5, .99, 1], dtype=dtype)
    sampler = AdaptivePhaseSampler(phase, frame_count=10**9, batch_budget=3)
    idx, weight, _ = sampler.sample(3)
    assert (idx >= 0).all() and (idx < len(phase)).all()
    assert abs(float(weight.sum()) - 1) < 1e-12


@pytest.mark.parametrize("frames", [1, 5, 17, 299, 3000])
def test_complete_native_reset_cycle_covers_every_merged_stratum(frames):
    phase, _ = PhaseCoverageRotor(frames, seed=103).draw(frames)
    for budget in [1, 2, 3, 7, 8, 13, 32, 128]:
        sampler = AdaptivePhaseSampler(phase, frame_count=frames, batch_budget=budget)
        assert sampler.coverage["empty_bins"] == 0
        assert sampler.coverage["covered_partition_time_fraction"] == 1


@pytest.mark.parametrize("frames", [1, 5, 299, 100000])
def test_real_float32_timestamp_conversion_preserves_requested_cells(frames):
    rotor = PhaseCoverageRotor(frames, seed=41)
    duration = torch.full((min(20000, frames*100),), frames/30, dtype=torch.float32)
    times, cells = rotor.draw_times(duration)
    actual = ((times/duration).double()*frames).floor().long().clamp_max(frames-1)
    assert torch.equal(actual, cells)


def test_unrepresentable_float32_timestamps_do_not_claim_full_coverage():
    rotor = PhaseCoverageRotor(10**9, seed=41)
    with pytest.raises(ValueError, match="cannot represent"):
        rotor.draw_times(torch.full((1000,), 10**9/30, dtype=torch.float32))
