"""Stress capacity redistribution without changing the phase target."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from adaptive_phase_candidate import AdaptivePhaseSampler, CapacityAwarePhaseSampler, capped_quota_plan


def old_coefficient(n, m):
    if n == 1:
        return 0.
    r = m % n
    return r*(n-r)/(m*m*(n-1))


def audit_allocations(cases, seed):
    rng = np.random.default_rng(seed)
    allocations = 0
    for case in range(cases):
        k = int(rng.integers(1, 130))
        counts = np.exp(rng.uniform(0, np.log(10**6), k)).astype(np.int64).clip(1).tolist()
        if case % 3 == 0:
            counts[:max(1, k//3)] = [1]*max(1, k//3)
        total = sum(counts)
        batch = int(rng.integers(k, max(k+1, min(total+10, 32769))))
        cursor = int(rng.integers(k))
        refund = int(rng.integers(k))
        quota, _, _, original = capped_quota_plan(counts, batch, cursor, refund)
        assert sum(quota) == batch
        assert all(m >= 1 for m in quota)
        assert all(m >= min(o, n) for m, o, n in zip(quota, original, counts))
        if batch <= total:
            assert all(m <= n for m, n in zip(quota, counts))
            for n, old, new in zip(counts, original, quota):
                assert old_coefficient(n, new) <= old_coefficient(n, old) + 1e-14
        else:
            assert all(m >= n for m, n in zip(quota, counts))
        allocations += k
        if (case+1) % 20000 == 0:
            print(json.dumps(dict(allocation_cases=case+1)), flush=True)
    return dict(cases=cases, per_bin_checks=allocations, violations=0)


def audit_sampling(cases, seed):
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    draws = batches = 0
    for case in range(cases):
        n = int(rng.integers(1, 513))
        frames = [1, 3, 8, 299, 3000, 10**9][case % 6]
        budget = int(rng.integers(1, 1025))
        phase = torch.rand(n, dtype=torch.float64, generator=gen)
        if case % 3 == 0:
            phase[:-1] = 0
            phase[-1] = 1
        mask = torch.rand(n, generator=gen) > .2
        mask[0] = True
        old = AdaptivePhaseSampler(phase, frame_count=frames, batch_budget=budget, eligible=mask)
        new = CapacityAwarePhaseSampler(phase, frame_count=frames, batch_budget=budget, eligible=mask)
        assert torch.equal(old.full_weights(), new.full_weights())
        for _ in range(3):
            size = budget + int(rng.integers(5))
            idx, weights, quota = new.sample(size, generator=gen)
            assert len(idx) == size
            assert bool(mask[idx].all())
            assert idx.unique().numel() == min(size, new.indices.numel())
            assert bool((quota > 0).all())
            assert bool(torch.isfinite(weights).all()) and bool((weights > 0).all())
            mass = torch.zeros(new.num_bins, dtype=torch.float64).scatter_add_(0, new.bin_ids[idx], weights)
            assert torch.allclose(mass[new.present_bins], new.target_mass, atol=1e-12, rtol=0)
            if size >= new.indices.numel():
                population_mass = torch.zeros(n, dtype=torch.float64).scatter_add_(0, idx, weights)
                assert torch.allclose(population_mass, new.full_weights(), atol=1e-12, rtol=0)
            else:
                assert bool((quota <= new.counts).all())
            batches += 1
            draws += size
        if (case+1) % 2000 == 0:
            print(json.dumps(dict(sampling_cases=case+1, sampled_indices=draws)), flush=True)
    return dict(cases=cases, minibatches=batches, sampled_indices=draws,
                starvation_violations=0, weight_violations=0, duplicate_violations=0)


def monte_carlo(repeats, seed):
    counts = [1, 3, 40, 80]
    gen = torch.Generator().manual_seed(seed)
    values = torch.randn(sum(counts), 12, dtype=torch.float64, generator=gen)
    phase = torch.cat([torch.full((n,), (b+.5)/4, dtype=torch.float64)
                       for b, n in enumerate(counts)])
    values *= (1+4*phase[:, None])
    result = {}
    for name, cls in [("balanced", AdaptivePhaseSampler), ("capacity", CapacityAwarePhaseSampler)]:
        sampler = cls(phase, frame_count=4, batch_budget=16)
        generator = torch.Generator().manual_seed(seed)
        exact = (sampler.full_weights()[:, None]*values).sum(0)
        sigma = [float(values[sampler.bin_ids == b].var(0, unbiased=False).sum()) for b in range(4)]
        errors, predicted, unique = [], [], []
        means = torch.zeros(12, dtype=torch.float64)
        for _ in range(repeats):
            idx, weights, quota = sampler.sample(16, generator=generator)
            estimate = (weights[:, None]*values[idx]).sum(0)
            errors.append(float((estimate-exact).square().sum()))
            means += estimate
            unique.append(len(idx.unique()))
            predicted.append(sum(old_coefficient(n, m)*v/16 for n, m, v in zip(counts, quota.tolist(), sigma)))
        actual, theory = float(np.mean(errors)), float(np.mean(predicted))
        assert abs(actual/theory-1) < .08
        result[name] = dict(empirical_mse=actual, predicted_mse=theory,
                            minimum_unique=min(unique), mean_bias_norm=float((means/repeats-exact).norm()))
    result["ratio"] = result["capacity"]["empirical_mse"]/result["balanced"]["empirical_mse"]
    return result


def gpu_benchmark(seed):
    if not torch.cuda.is_available():
        return dict(skipped=True)
    rng = np.random.default_rng(seed)
    counts = (np.geomspace(2000, 1, 299)*3).astype(np.int64).clip(1)
    counts[-10:] = np.arange(1, 11)
    counts[0] += 262144-int(counts.sum())
    cells = torch.repeat_interleave(torch.arange(299, device="cuda:0"), torch.tensor(counts, device="cuda:0"))
    phase = (cells.double()+.5)/299
    report = dict(samples=len(phase), control_cells=299, cases={})
    for name, cls in [("balanced", AdaptivePhaseSampler), ("capacity", CapacityAwarePhaseSampler)]:
        sampler = cls(phase, frame_count=299, batch_budget=16384)
        gen = torch.Generator(device="cuda:0").manual_seed(seed)
        result = {}
        for size in [16384, 32768]:
            elapsed, uniques = [], []
            for _ in range(25):
                torch.cuda.synchronize()
                start = time.perf_counter()
                idx, weight, quota = sampler.sample(size, generator=gen)
                torch.cuda.synchronize()
                elapsed.append(1000*(time.perf_counter()-start))
                uniques.append(idx.unique().numel())
                assert abs(float(weight.sum())-1) < 1e-12
                if name == "capacity":
                    assert uniques[-1] == size and bool((quota <= sampler.counts).all())
            result[str(size)] = dict(sample_median_ms=float(np.median(elapsed[5:])), minimum_unique=min(uniques))
        report["cases"][name] = result
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--allocation-cases", type=int, default=100000)
    parser.add_argument("--sampling-cases", type=int, default=10000)
    parser.add_argument("--repeats", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261025)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = dict(scope="capacity allocation and same-target sampling; not RL training", seed=args.seed)
    report["allocations"] = audit_allocations(args.allocation_cases, args.seed)
    report["sampling"] = audit_sampling(args.sampling_cases, args.seed)
    report["monte_carlo"] = monte_carlo(args.repeats, args.seed)
    report["gpu"] = gpu_benchmark(args.seed)
    (args.out/"capacity_stress.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(complete=True)), flush=True)


if __name__ == "__main__":
    main()
