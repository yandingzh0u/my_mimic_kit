"""Isolated prototype: homogeneous distance head and exact BCE intercept.

This is a research gate, not a change to the DARE training configuration.
Geometry guarantees apply for each fixed network and fixed normalizer.
Offline BCE fitting does not establish closed-loop control performance.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from dare_paper_stress import ROOT, attacks, load_disc, radial_probe, stats


@torch.no_grad()
def optimal_intercept(distances):
    """Minimize balanced zero-vs-residual BCE over one scalar intercept.

    Its derivative vanishes at sigmoid(b) + mean(sigmoid(b-d)) = 1.
    The root is bracketed by min(d)/2 and max(d)/2.  Bisection is stopped
    by floating-point exhaustion, not a tunable optimization tolerance.
    """
    d = distances.detach().to(torch.float64)
    lo, hi = d.min() / 2, d.max() / 2
    for _ in range(64):
        mid = (lo + hi) / 2
        # Compare the two positive sides in log space. Subtracting 1 from
        # sigmoid(b) would lose the root for well-separated large distances.
        log_positive_error = F.logsigmoid(-mid)
        log_negative_error = torch.logsumexp(F.logsigmoid(mid - d), dim=0) - math.log(d.numel())
        sign = log_negative_error - log_positive_error
        lo = torch.where(sign < 0, mid, lo)
        hi = torch.where(sign >= 0, mid, hi)
    return ((lo + hi) / 2).to(distances.dtype)


class RayDisc(torch.nn.Module):
    def __init__(self, source):
        super().__init__()
        self._disc_layers = copy.deepcopy(source._disc_layers)
        # Remove, rather than merely initialize, every hidden bias.
        for layer in self._disc_layers.modules():
            if isinstance(layer, torch.nn.Linear):
                layer.register_parameter("bias", None)
        self.register_buffer("intercept", source._disc_logits.bias.new_zeros(()))

    def distance(self, x):
        return torch.linalg.vector_norm(self._disc_layers(x), dim=-1)

    def forward(self, x):
        return self.intercept - self.distance(x)

    @torch.no_grad()
    def calibrate(self, distances):
        self.intercept.copy_(optimal_intercept(distances))


def balanced_bce(zero, negative):
    return .5 * (F.softplus(-zero).mean() + F.softplus(negative).mean())


def diagnostics(model, train, heldout):
    model.eval()
    if isinstance(model, RayDisc):
        with torch.no_grad():
            model.calibrate(model.distance(train))
    with torch.no_grad():
        f0 = model(heldout.new_zeros(1, heldout.shape[-1]))
        values = model(heldout)
        tiny = heldout[:256] * 1e-6
        euler_x = heldout[:256].detach().clone()
        norm = euler_x.norm(dim=-1, keepdim=True)
        random = F.normalize(torch.randn_like(heldout), dim=-1) * heldout.norm(dim=-1, keepdim=True)
        # A deliberately invalid criterion: coordinatewise contraction does
        # not follow from radial monotonicity. Keep it visible in the report.
        contraction = torch.rand_like(heldout)
        contracted_values = model(contraction * heldout)
        result = dict(
            zero_logit=float(f0), heldout_logits=stats(values),
            heldout_balanced_bce=float(balanced_bce(f0, values)),
            heldout_negative_accuracy=float((values < 0).float().mean()),
            heldout_radial=radial_probe(model, heldout),
            random_radial=radial_probe(model, random),
            independent_coordinate_contraction_violation=float(
                (contracted_values < values - 1e-5).float().mean()),
            tiny_inputs_are_finite=bool(torch.isfinite(model(tiny)).all()))
    if isinstance(model, RayDisc):
        p = euler_x.requires_grad_(True)
        d = model.distance(p)
        grad, = torch.autograd.grad(d.sum(), p)
        result["euler_absolute_error"] = float(((grad * p).sum(-1) - d).abs().max())
        with torch.no_grad():
            result["homogeneity_relative_error"] = float(
                (model.distance(3.7 * euler_x) - 3.7 * d).abs().max()
                / (3.7 * d).abs().max().clamp_min(1e-12))
            result["calibration_root_residual"] = float(
                (torch.sigmoid(model.intercept)
                 + torch.sigmoid(model.intercept - model.distance(train)).mean() - 1).abs())
            result["distance_over_input_norm"] = stats(d / norm.squeeze(-1).clamp_min(1e-12))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(20261024)
    source, state, ckpt = load_disc(args.archive / "checkpoint.pt", args.device)
    buffers = ckpt["replay_buffer_states"]["_disc_buffer"]["buffers"]
    raw = (buffers["disc_obs_demo"][:, 0] - buffers["disc_obs"][:, 0]).to(args.device)
    x = raw / state["_disc_obs_norm._mean_abs"].to(args.device).clamp_min(1e-4)
    split = torch.randperm(len(x), device=x.device)
    train, heldout = x[split[:160000]], x[split[160000:]]
    # Use a bounded diagnostic slice but an independent fitting split.
    diag_train, diag_test = train[:8192], heldout[:2048]
    final_ray = RayDisc(source).eval().requires_grad_(False)
    report = dict(protocol="frozen residual replay; random split, not independent trajectories",
                  training_rows=len(train), heldout_rows=len(heldout),
                  original_final=diagnostics(source, diag_train, diag_test),
                  ray_from_final_features=diagnostics(final_ray, diag_train, diag_test),
                  fitting=[])
    for seed in args.seeds:
        torch.manual_seed(seed)
        early, _, _ = load_disc(args.archive / "int_models/model_0000000000.pt", args.device)
        ray = RayDisc(early)
        initial = {
            "baseline": diagnostics(early, diag_train, diag_test),
            "ray": diagnostics(ray, diag_train, diag_test)}
        curve = []
        for name, model in [("baseline", early), ("ray", ray)]:
            model.requires_grad_(True).train()
            opt = torch.optim.SGD(model.parameters(), lr=2.5e-4, weight_decay=1e-4)
            # Equalize replay sample order and SGD steps for the two heads.
            generator = torch.Generator(device=args.device).manual_seed(seed)
            start = time.perf_counter()
            for step in range(args.steps):
                idx = torch.randint(len(train), (1024,), device=args.device, generator=generator)
                batch = train[idx]
                opt.zero_grad(set_to_none=True)
                if isinstance(model, RayDisc):
                    d = model.distance(batch)
                    model.calibrate(d)
                    pos = model.intercept.reshape(1)
                    neg = model.intercept - d
                else:
                    pos = model(batch.new_zeros(1, batch.shape[-1]))
                    neg = model(batch)
                loss = balanced_bce(pos, neg)
                loss.backward()
                opt.step()
                if step in [0, args.steps // 2, args.steps - 1]:
                    curve.append(dict(method=name, step=step, bce=float(loss.detach())))
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            final = diagnostics(model, diag_train, diag_test)
            final["fit_seconds"] = elapsed
            report["fitting"].append(dict(seed=seed, method=name, initial=initial[name], final=final))
            print(json.dumps(dict(seed=seed, method=name,
                                  initial_bce=initial[name]["heldout_balanced_bce"],
                                  final_bce=final["heldout_balanced_bce"],
                                  final_negative_accuracy=final["heldout_negative_accuracy"],
                                  fit_seconds=elapsed)), flush=True)
        report.setdefault("fit_curves", []).append(dict(seed=seed, curve=curve))
        (args.out / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Wrote", args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
