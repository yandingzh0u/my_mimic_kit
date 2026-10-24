"""Read-only stress tests of DARE geometry on archived checkpoints.

No simulator, optimizer of the policy, or production source is modified.
The replay inputs are used for diagnostics, not independent evaluation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "mimickit"))
from learning.dare_model import GroupSeparableDiscLayers  # noqa: E402


class Disc(torch.nn.Module):
    def __init__(self, state, load_weights=True):
        super().__init__()
        self.groups = tuple(
            (str(i), state[f"_disc_layers.group_indices_{i}"].tolist())
            for i in range(7))
        self._disc_layers = GroupSeparableDiscLayers(
            self.groups, 1024, [512] * 8, torch.nn.ReLU)
        head = torch.nn.Linear(512, 1)
        torch.nn.init.uniform_(head.weight, -1., 1.)
        torch.nn.init.zeros_(head.bias)
        self._disc_logits = torch.nn.utils.parametrizations.spectral_norm(head)
        if load_weights:
            self.load_state_dict(state, strict=True)

    def forward(self, x):
        return self._disc_logits(self._disc_layers(x)).squeeze(-1)


def load_disc(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    state = checkpoint.get("model_state_dict", checkpoint)
    disc_state = {k.removeprefix("_model."): v for k, v in state.items()
                  if k.startswith("_model._disc_")}
    model = Disc(disc_state).to(device).eval().requires_grad_(False)
    return model, state, checkpoint


def stats(x):
    x = x.detach().float().flatten()
    q = torch.quantile(x, x.new_tensor([0.05, 0.5, 0.95]))
    return dict(mean=float(x.mean()), min=float(x.min()), max=float(x.max()),
                p05=float(q[0]), median=float(q[1]), p95=float(q[2]))


def reward(logits):
    return -2 * torch.log(torch.clamp_min(1 - torch.sigmoid(logits), 1e-4))


def radial_probe(model, x):
    scales = x.new_tensor([0., 1e-4, 1e-3, .01, .03, .1, .25, .5, 1., 2., 4., 16.])
    with torch.no_grad():
        y = torch.stack([model(s * x) for s in scales])
    change = y[1:] - y[:-1]
    return dict(scales=scales.tolist(), mean_logits=y.mean(1).tolist(),
                nonmonotone_ray_fraction=float((change > 1e-5).any(0).float().mean()),
                interpolation_nonmonotone_fraction=float((change[:8] > 1e-5).any(0).float().mean()),
                half_to_full_violation_fraction=float((y[8] > y[7] + 1e-5).float().mean()),
                full_above_anchor_fraction=float((y[8] > y[0] + 1e-5).float().mean()),
                above_anchor_fraction=float((y > y[:1] + 1e-5).any(0).float().mean()),
                largest_outward_logit_increase=float(change.max()))


def attacks(model, x, steps=60):
    zero = x.new_zeros(1, x.shape[1]).requires_grad_(True)
    f0 = model(zero)
    g0, = torch.autograd.grad(f0.sum(), zero)
    direction = F.normalize(g0, dim=-1)
    radii = x.new_tensor([0., 1e-6, 1e-5, 1e-4, .001, .003, .01, .03, .1, .3, 1.])
    with torch.no_grad():
        along_gradient = model(radii[:, None] * direction)
    norm = torch.linalg.vector_norm(x, dim=-1)
    levels = torch.quantile(norm, norm.new_tensor([.05, .5, .95]))
    result = dict(origin_logit=float(f0), origin_gradient_norm=float(g0.norm()),
                  gradient_ray_radii=radii.tolist(),
                  gradient_ray_logits=along_gradient.tolist(), sphere=[])
    for radius in levels:
        v = torch.nn.Parameter(F.normalize(torch.randn_like(x[:64]), dim=-1))
        opt = torch.optim.Adam([v], lr=.03)
        for _ in range(steps):
            opt.zero_grad(set_to_none=True)
            (-model(radius * F.normalize(v, dim=-1)).mean()).backward()
            opt.step()
        with torch.no_grad():
            points = radius * F.normalize(v, dim=-1)
            best = model(points)
        p = points.detach().requires_grad_(True)
        values = model(p)
        grads, = torch.autograd.grad(values.sum(), p)
        result["sphere"].append(dict(
            radius=float(radius), logits=stats(best),
            above_anchor_fraction=float((best > f0.detach() + 1e-5).float().mean()),
            outward_gradient_fraction=float(((grads * p).sum(-1) > 1e-5).float().mean())))
    return result


def geometry(model, raw, scale, x):
    report = {}
    for name, idx in model.groups:
        i = torch.tensor(idx, device=x.device)
        z = x.index_select(-1, i)
        report[name] = dict(dim=len(idx), mean_abs=float(z.abs().mean()),
                            squared_norm=stats(z.square().sum(-1)),
                            floor_coordinates=int((scale[i] <= 1e-4).sum()))
    # Test conversion of input AND fitted scale; the floor prevents exact
    # invariance when converted scales cross 1e-4.
    units = {}
    for factor in [1e-6, 1e-3, .1, 10., 1e3, 1e6]:
        altered = (factor * raw) / (factor * scale).clamp_min(1e-4)
        with torch.no_grad():
            delta = (model(altered) - model(x)).abs()
        units[str(factor)] = dict(input_max_abs_error=float((altered-x).abs().max()),
                                   logit_abs_error=stats(delta))
    return dict(groups=report, unit_conversion=units)


def spectral(model):
    result = {}
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            weight = module.weight.detach()
            sv = torch.linalg.svdvals(weight)
            result[name] = float(sv[0])
    return result


def loss_alias_probe():
    import util.torch_util as torch_util
    aggregate = {}
    for _ in range(32):
        loss = torch.tensor(1., requires_grad=True)
        torch_util.add_torch_dict(
            {"disc_loss": loss, "disc_cls_loss": loss.detach()}, aggregate)
    torch_util.scale_torch_dict(1/32, aggregate)
    return {k: float(v) for k, v in aggregate.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=ROOT / "output/paper_benchmark/dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--samples", type=int, default=4096)
    parser.add_argument("--attack-steps", type=int, default=60)
    args = parser.parse_args()
    torch.manual_seed(20261024)
    torch.set_num_threads(4)
    args.out.mkdir(parents=True, exist_ok=True)
    base, state, ckpt = load_disc(args.archive / "checkpoint.pt", args.device)
    buf = ckpt["replay_buffer_states"]["_disc_buffer"]["buffers"]
    n = buf["disc_obs"].shape[0]
    idx = torch.randperm(n)[:args.samples]
    raw = (buf["disc_obs_demo"][idx, 0] - buf["disc_obs"][idx, 0]).to(args.device)
    scale = state["_disc_obs_norm._mean_abs"].to(args.device)
    x = raw / scale.clamp_min(1e-4)
    report = dict(seed=20261024, source=str(args.archive),
                  samples=len(idx), input_dim=x.shape[-1],
                  floor=1e-4, normalizer_count=int(state["_disc_obs_norm._count"]))
    report["loss_logging_alias_probe_expected_one"] = loss_alias_probe()
    report["normalization"] = geometry(base, raw, scale, x)
    report["checkpoints"] = {}
    for iteration in [0, 100, 400, 1600, 1700, 2999]:
        path = args.archive / f"int_models/model_{iteration:010d}.pt"
        model, snapshot, _ = load_disc(path, args.device)
        z = raw / snapshot["_disc_obs_norm._mean_abs"].to(args.device).clamp_min(1e-4)
        random = F.normalize(torch.randn_like(z), dim=-1) * z.norm(dim=-1, keepdim=True)
        data = dict(on_shared_final_replay=radial_probe(model, z),
                    on_random_directions=radial_probe(model, random),
                    attacks=attacks(model, z, args.attack_steps),
                    spectral_norms=spectral(model))
        report["checkpoints"][str(iteration)] = data
        (args.out / "summary.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(dict(iteration=iteration,
                             replay=data["on_shared_final_replay"],
                             random=data["on_random_directions"],
                             origin_gradient=data["attacks"]["origin_gradient_norm"],
                             max_spectral=max(data["spectral_norms"].values()),
                             sphere=data["attacks"]["sphere"])), flush=True)
    print("Wrote", args.out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
