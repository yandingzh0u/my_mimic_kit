"""Adversarial tests of the production sampler, including real GPU batch sizes."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mimickit"))
from learning.phase_sampler import PhaseSampler
from util.phase_coverage import PhaseCoverage


def sampling(cases, seed):
    rng = np.random.default_rng(seed)
    generator = torch.Generator().manual_seed(seed)
    total = batches = 0
    for case in range(cases):
        n = int(rng.integers(1, 1025))
        frames = [1, 3, 8, 299, 3000, 10**9][case % 6]
        budget = int(rng.integers(1, 2049))
        phase = torch.rand(n, dtype=torch.float64, generator=generator)
        if case % 4 == 0:
            phase[:-1] = 0
            phase[-1] = 1
        if case % 4 == 1:
            phase = phase.square().square().square()
        eligible = torch.rand(n, generator=generator) > .3
        if case % 17 == 0:
            eligible[:] = False
        sampler = PhaseSampler(phase, frames, budget, eligible)
        for size in (budget, budget+1, max(budget, n), budget+17):
            idx, weight, quota = sampler.sample(size, generator)
            assert len(idx) == min(size, int(eligible.sum()))
            assert len(idx.unique()) == len(idx)
            assert bool(eligible[idx].all())
            assert bool(((quota >= 1) & (quota <= sampler.counts)).all())
            assert int(quota.sum()) == len(idx)
            assert bool(torch.isfinite(weight).all()) and bool((weight > 0).all())
            mass = torch.zeros(sampler.num_bins, dtype=torch.float64).scatter_add_(0, sampler.bin_ids[idx], weight)
            torch.testing.assert_close(mass[sampler.present_bins], sampler.target_mass, atol=1e-12, rtol=0)
            if size >= len(sampler.indices):
                population = torch.zeros(n, dtype=torch.float64).scatter_add_(0, idx, weight)
                torch.testing.assert_close(population, sampler.full_weights(), atol=1e-12, rtol=0)
            total += len(idx)
            batches += 1
        if (case+1) % 5000 == 0:
            print(json.dumps(dict(cases=case+1, minibatches=batches, indices=total)), flush=True)
    return dict(distributions=cases, minibatches=batches, sampled_indices=total,
                starvation_violations=0, duplicate_violations=0, weight_violations=0)


def monte_carlo(repeats, seed):
    generator = torch.Generator().manual_seed(seed)
    counts = [1, 3, 40, 80]
    phase = torch.cat([torch.full((n,), (k+.5)/4) for k, n in enumerate(counts)])
    values = torch.randn(sum(counts), 12, dtype=torch.float64, generator=generator)
    values *= 1 + 4*phase[:, None]
    sampler = PhaseSampler(phase, 4, 16)
    target = (sampler.full_weights()[:, None] * values).sum(0)
    estimates = []
    for _ in range(repeats):
        idx, weight, _ = sampler.sample(16, generator)
        estimates.append((values[idx] * weight[:, None]).sum(0))
    estimates = torch.stack(estimates)
    se = estimates.std(0) / repeats**.5
    z = ((estimates.mean(0)-target).abs()/se).max()
    assert z < 5
    return dict(repeats=repeats, dimensions=12, maximum_absolute_bias_in_standard_errors=float(z),
                mse=float((estimates-target).square().sum(1).mean()))


def reset_coverage(seed):
    torch.manual_seed(seed)
    draws = 0
    for cells in [1, 2, 7, 8, 299, 3000, 10007, 100000]:
        for _ in range(10):
            coverage = PhaseCoverage(cells)
            coverage.draw_times(torch.full((int(torch.randint(0, cells+1, ())),), cells/30))
            lengths = torch.full((2*cells+3,), cells/30, dtype=torch.float32)
            state, rng = coverage.state_dict(), torch.get_rng_state()
            times = coverage.draw_times(lengths)
            actual = ((times/lengths).double()*cells).long().clamp_max(cells-1)
            for start in (0, 1, cells//3, cells):
                assert len(actual[start:start+cells].unique()) == cells
            coverage.load_state_dict(state)
            torch.set_rng_state(rng)
            assert torch.equal(coverage.draw_times(lengths), times)
            draws += len(times)
    return dict(draws=draws, window_checks=320, violations=0)


def gpu(seed):
    if not torch.cuda.is_available():
        raise RuntimeError("GPU stress is required for this experiment")
    generator = torch.Generator(device="cuda:0").manual_seed(seed)
    counts = np.maximum(1, np.geomspace(5000, 1, 299).astype(int))
    counts[-10:] = np.arange(1, 11)
    counts[0] += 262144 - counts.sum()
    phase = (torch.repeat_interleave(torch.arange(299, device="cuda:0"), torch.as_tensor(counts, device="cuda:0")).double()+.5)/299
    sampler = PhaseSampler(phase, 299, 16384)
    report = dict(population=len(phase), control_cells=299, batches={})
    for b in (16384, 32768):
        timings, minimum = [], b
        for repeat in range(100):
            torch.cuda.synchronize()
            start = time.perf_counter()
            idx, weight, quota = sampler.sample(b, generator)
            torch.cuda.synchronize()
            timings.append(1000*(time.perf_counter()-start))
            minimum = min(minimum, len(idx.unique()))
            assert minimum == b and bool((quota >= 1).all()) and bool((quota <= sampler.counts).all())
            mass = torch.zeros(299, device=phase.device, dtype=torch.float64).scatter_add_(0, sampler.bin_ids[idx], weight)
            torch.testing.assert_close(mass, sampler.target_mass, atol=1e-12, rtol=0)
        report["batches"][str(b)] = dict(minimum_unique=minimum, median_ms=float(np.median(timings[10:])))
    coverage = PhaseCoverage(299)
    times = coverage.draw_times(torch.full((8192,), 299/30, device="cuda:0"))
    assert len(((times/(299/30)).double()*299).long().unique()) == 299
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=25000)
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.out.mkdir(parents=True, exist_ok=True)
    report = dict(seed=args.seed, sampling=sampling(args.cases, args.seed),
                  monte_carlo=monte_carlo(args.repeats, args.seed),
                  reset_coverage=reset_coverage(args.seed), gpu=gpu(args.seed))
    (args.out / "stress.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
