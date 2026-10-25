"""Adversarial sampling audit, independent of Isaac/DARE and RL training."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch

from adaptive_phase_candidate import AdaptivePhaseSampler, PhaseCoverageRotor


def fuzz(cases, batches, seed):
    rng = np.random.default_rng(seed)
    gen = torch.Generator().manual_seed(seed)
    frames_options = [1, 2, 3, 7, 8, 9, 30, 299, 3000, 100000, 10**9]
    draws = 0
    start = time.perf_counter()
    for case in range(cases):
        frames = frames_options[case % len(frames_options)]
        n = int(rng.integers(1, 257))
        budget = int(rng.integers(1, 258))
        if case % 101 == 0:
            budget = [16384, 32768][case % 2]
        mode = case % 7
        if mode == 0:
            phase = torch.rand(n, dtype=torch.float64, generator=gen)
        elif mode == 1:
            phase = torch.zeros(n, dtype=torch.float64)
            phase[-1] = 1
        elif mode == 2:
            phase = torch.rand(n, dtype=torch.float64, generator=gen).pow(20)
        elif mode == 3:
            phase = torch.full((n,), .5, dtype=torch.float64)
        elif mode == 4:
            phase = torch.randint(0, 2, (n,), generator=gen).double()
        elif mode == 5:
            phase = torch.linspace(0, 1, n, dtype=torch.float64)
        else:
            phase = torch.randint(0, 7, (n,), generator=gen).double() / 6
        mask = torch.rand(n, generator=gen) > .3
        mask[int(rng.integers(n))] = True
        sampler = AdaptivePhaseSampler(phase, frame_count=frames, batch_budget=budget, eligible=mask)
        assert sampler.num_bins <= min(frames, budget, n)
        accumulated_extras = torch.zeros(sampler.num_present, dtype=torch.long)
        for _ in range(batches):
            size = budget + int(rng.integers(0, 5))
            idx, weight, quota = sampler.sample(size, generator=gen)
            assert len(idx) == size and bool(mask[idx].all())
            count = torch.bincount(sampler.bin_ids[idx], minlength=sampler.num_bins)
            assert torch.equal(count[sampler.present_bins], quota)
            assert int(quota.min()) >= 1
            assert int(quota.max() - quota.min()) <= 1
            assert bool(torch.isfinite(weight).all()) and bool((weight > 0).all())
            assert abs(float(weight.sum()) - 1) < 1e-12
            mass = torch.zeros(sampler.num_bins, dtype=torch.float64)
            mass.scatter_add_(0, sampler.bin_ids[idx], weight)
            assert bool((mass[sampler.present_bins] - sampler.target_mass).abs().max() < 1e-12)
            use = torch.bincount(idx, minlength=n)[sampler.indices]
            q = quota[sampler.labels] // sampler.counts[sampler.labels]
            assert bool(((use == q) | (use == q + 1)).all())
            # There is no starvation even with changing batch sizes.  The
            # extra slots, which are optional, are also fair over time.
            accumulated_extras += quota - size // sampler.num_present
            assert int(accumulated_extras.max() - accumulated_extras.min()) <= 1
            draws += size
        if (case + 1) % 1000 == 0:
            print(json.dumps(dict(fuzz_cases=case+1, minibatches=(case+1)*batches,
                                  sampled_indices=draws)), flush=True)
    return dict(cases=cases, minibatches=cases*batches, sampled_indices=draws,
                starvation_violations=0, weight_violations=0,
                reuse_violations=0, seconds=time.perf_counter()-start)


def variance_audit(repeats, seed):
    gen = torch.Generator().manual_seed(seed)
    counts = [1, 2, 7, 53]
    phase = torch.cat([torch.full((n,), (b+.5)/4, dtype=torch.float64)
                       for b, n in enumerate(counts)])
    values = torch.randn(sum(counts), dtype=torch.float64, generator=gen)
    values += 10 * torch.sin(phase * 9)
    result = []
    for size in [4, 5, 7, 16, 64, 4096]:
        sampler = AdaptivePhaseSampler(phase, frame_count=4, batch_budget=size)
        exact = float((sampler.full_weights() * values).sum())
        estimates, predicted, wr = [], [], []
        variances = torch.stack([values[sampler.bin_ids == b].var(unbiased=False)
                                 for b in sampler.present_bins.tolist()])
        for _ in range(repeats):
            idx, w, quota = sampler.sample(size, generator=gen)
            estimates.append(float((w * values[idx]).sum()))
            v, vwr = 0., 0.
            for j, (n, m) in enumerate(zip(counts, quota.tolist())):
                r = m % n
                if n > 1:
                    v += float(variances[j]) * r * (n-r) / (m*m*(n-1)*16)
                vwr += float(variances[j]) / (m*16)
            predicted.append(v)
            wr.append(vwr)
        mean = float(np.mean(estimates))
        observed_var = float(np.var(estimates, ddof=1))
        predicted_var = float(np.mean(predicted))
        stderr = math.sqrt(predicted_var/repeats)
        assert abs(mean-exact) <= 6*stderr + 1e-12
        assert predicted_var <= float(np.mean(wr)) + 1e-12
        if predicted_var > 1e-20:
            assert abs(observed_var/predicted_var - 1) < .15
        result.append(dict(batch=size, repeats=repeats, target=exact, mean=mean,
                           bias_z=(mean-exact)/max(stderr, 1e-30),
                           empirical_variance=observed_var, predicted_variance=predicted_var,
                           with_replacement_variance=float(np.mean(wr))))
    return result


def rotor_audit(seed):
    generator = np.random.default_rng(seed)
    total_windows = requests = 0
    for frames in list(range(1, 65)) + [127, 299, 1024, 10007, 100000]:
        rotor = PhaseCoverageRotor(frames, seed=seed+frames)
        cells, phases = [], []
        remaining = 3*frames + 19
        while remaining:
            n = min(remaining, int(generator.integers(1, max(2, frames))))
            p, c = rotor.draw(n)
            phases.append(p)
            cells.append(c)
            remaining -= n
            requests += n
        cells, phases = torch.cat(cells), torch.cat(phases)
        assert torch.equal((phases*frames).floor().long(), cells)
        starts = range(2*frames) if frames < 65 else generator.integers(0, 2*frames, size=128)
        for offset in starts:
            assert torch.unique(cells[offset:offset+frames]).numel() == frames
            total_windows += 1
    # Test billion-cell metadata without allocating a billion-element table.
    rotor = PhaseCoverageRotor(10**9, seed=seed)
    p, c = rotor.draw(10000)
    assert c.unique().numel() == 10000
    assert torch.equal((p*10**9).floor().long(), c)
    return dict(windows_checked=total_windows, reset_requests=requests+10000,
                starvation_violations=0, largest_motion_cells=10**9)


def gpu_audit(seed):
    if not torch.cuda.is_available():
        return dict(skipped="CUDA unavailable")
    device = "cuda:0"
    gen = torch.Generator(device=device).manual_seed(seed)
    counts = [70000, 60000, 40000, 30000, 25000, 20000, 12508, 4636]
    phase = torch.cat([(b + torch.rand(n, device=device, generator=gen))/8
                       for b, n in enumerate(counts)])
    output = dict(device=torch.cuda.get_device_name(), samples=len(phase), cases=[])
    for frames in [8, 299, 3000, 100000, 10**9]:
        torch.cuda.synchronize()
        start = time.perf_counter()
        sampler = AdaptivePhaseSampler(phase, frame_count=frames, batch_budget=16384)
        torch.cuda.synchronize()
        build_ms = 1000*(time.perf_counter()-start)
        for size in [16384, 32768]:
            times = []
            for _ in range(23):
                torch.cuda.synchronize()
                start = time.perf_counter()
                idx, weight, quota = sampler.sample(size, generator=gen)
                torch.cuda.synchronize()
                times.append(1000*(time.perf_counter()-start))
                assert bool((quota > 0).all())
                assert abs(float(weight.sum()) - 1) < 1e-12
                assert len(idx) == size
            output["cases"].append(dict(frame_count=frames, batch=size,
                                         coverage=sampler.coverage, build_ms=build_ms,
                                         sample_median_ms=float(np.median(times[3:]))))
    # Historical eight-bin independent-with-replacement implementation.
    pools = [(phase*8).long().eq(b).nonzero().flatten() for b in range(8)]
    times = []
    for _ in range(23):
        torch.cuda.synchronize()
        start = time.perf_counter()
        idx = torch.cat([p[torch.randint(len(p), (4096,), device=device, generator=gen)] for p in pools])
        idx = idx[torch.randperm(len(idx), device=device, generator=gen)]
        torch.cuda.synchronize()
        times.append(1000*(time.perf_counter()-start))
    output["historical_cached_8bin_actor_median_ms"] = float(np.median(times[3:]))
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--cases", type=int, default=10000)
    parser.add_argument("--batches", type=int, default=12)
    parser.add_argument("--monte-carlo", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--resume", action="store_true", help="reuse an already completed fuzz stage")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    report = dict(seed=args.seed, scope="sampling invariants and frozen-data estimates; not RL training")
    if args.resume:
        report = json.loads((args.out/"stress.json").read_text())
        assert report["seed"] == args.seed
    else:
        report["fuzz"] = fuzz(args.cases, args.batches, args.seed)
    (args.out/"stress.json").write_text(json.dumps(report, indent=2)+"\n")
    report["variance"] = variance_audit(args.monte_carlo, args.seed)
    report["reset_rotor"] = rotor_audit(args.seed)
    report["gpu"] = gpu_audit(args.seed)
    n, m = 4636, 4096
    report["historical_rarest_bin_example"] = dict(pool=n, quota=m,
        expected_unique_with_replacement=n*(-math.expm1(m*math.log1p(-1/n))),
        balanced_unique=m, variance_ratio_balanced_over_wr=(n-m)/(n-1))
    (args.out/"stress.json").write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(dict(complete=True, path=str(args.out/"stress.json"))), flush=True)


if __name__ == "__main__":
    main()
