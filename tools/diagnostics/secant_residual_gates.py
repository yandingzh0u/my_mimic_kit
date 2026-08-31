#!/usr/bin/env python3
"""Cheap, read-only gates for the proposed Secant-Residual ADD objective.

The script never creates an Isaac environment and never mutates a checkpoint.
It reuses discriminator replay residuals and normalizer state saved in training
checkpoints, runs two analytic counterexamples, and probes the proposed
classification-first correction on cloned discriminator parameters.
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


class DenseDisc(torch.nn.Module):
    def __init__(self, shapes, spectral_norm):
        super().__init__()
        first = torch.nn.Linear(shapes[0][1], shapes[0][0])
        second = torch.nn.Linear(shapes[1][1], shapes[1][0])
        self._disc_layers = torch.nn.Sequential(
            first, torch.nn.ReLU(), second, torch.nn.ReLU())
        self._disc_logits = torch.nn.Linear(shapes[2][1], shapes[2][0])
        if spectral_norm:
            torch.nn.utils.parametrizations.spectral_norm(first)
            torch.nn.utils.parametrizations.spectral_norm(second)
            torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)

    def forward(self, x):
        return self._disc_logits(self._disc_layers(x)).squeeze(-1)


class GroupLayers(torch.nn.Module):
    def __init__(self, state, spectral_norm):
        super().__init__()
        group_ids = sorted({
            int(key.split(".")[2])
            for key in state
            if key.startswith("_disc_layers.encoders.")
        })
        self.encoders = torch.nn.ModuleList()
        for group_id in group_ids:
            index_key = f"_disc_layers.group_indices_{group_id}"
            self.register_buffer(
                f"group_indices_{group_id}",
                torch.empty_like(state[index_key]))
            prefix = f"_disc_layers.encoders.{group_id}.0"
            shape = _linear_shape(state, prefix)
            layer = torch.nn.Linear(shape[1], shape[0])
            if spectral_norm:
                torch.nn.utils.parametrizations.spectral_norm(layer)
            self.encoders.append(
                torch.nn.Sequential(layer, torch.nn.ReLU()))

        trunk_ids = sorted({
            int(key.split(".")[2])
            for key in state
            if key.startswith("_disc_layers.trunk.")
            and (key.endswith(".weight")
                 or key.endswith(".parametrizations.weight.original"))
        })
        trunk = []
        for layer_id in trunk_ids:
            prefix = f"_disc_layers.trunk.{layer_id}"
            shape = _linear_shape(state, prefix)
            layer = torch.nn.Linear(shape[1], shape[0])
            if spectral_norm:
                torch.nn.utils.parametrizations.spectral_norm(layer)
            trunk.extend((layer, torch.nn.ReLU()))
        self.trunk = torch.nn.Sequential(*trunk)

    def forward(self, x):
        encoded = []
        for group_id, encoder in enumerate(self.encoders):
            indices = getattr(self, f"group_indices_{group_id}")
            encoded.append(encoder(torch.index_select(x, -1, indices)))
        return self.trunk(torch.cat(encoded, dim=-1))


class GroupDisc(torch.nn.Module):
    def __init__(self, state, spectral_norm):
        super().__init__()
        self._disc_layers = GroupLayers(state, spectral_norm)
        shape = _linear_shape(state, "_disc_logits")
        self._disc_logits = torch.nn.Linear(shape[1], shape[0])
        if spectral_norm:
            torch.nn.utils.parametrizations.spectral_norm(self._disc_logits)

    def forward(self, x):
        return self._disc_logits(self._disc_layers(x)).squeeze(-1)


def _linear_shape(state, prefix):
    plain = f"{prefix}.weight"
    original = f"{prefix}.parametrizations.weight.original"
    weight = state[plain] if plain in state else state[original]
    return tuple(weight.shape)


def _disc_state(checkpoint):
    return {
        key.removeprefix("_model."): value
        for key, value in checkpoint["model_state_dict"].items()
        if key.startswith("_model._disc_")
    }


def build_disc(checkpoint, device, trainable=False):
    state = _disc_state(checkpoint)
    spectral_norm = any("parametrizations.weight.original" in key
                        for key in state)
    grouped = any(key.startswith("_disc_layers.encoders.")
                  for key in state)
    if grouped:
        model = GroupDisc(state, spectral_norm)
    else:
        shapes = (
            _linear_shape(state, "_disc_layers.0"),
            _linear_shape(state, "_disc_layers.2"),
            _linear_shape(state, "_disc_logits"),
        )
        model = DenseDisc(shapes, spectral_norm)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    if not trainable:
        model.requires_grad_(False)
    return model, {"spectral_norm": spectral_norm, "grouped": grouped}


def load_checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def replay_residuals(checkpoint):
    buffers = checkpoint["replay_buffer_states"]["_disc_buffer"]["buffers"]
    obs = buffers["disc_obs"].squeeze(1)
    demo = buffers["disc_obs_demo"].squeeze(1)
    return demo - obs


def normalize(checkpoint, raw):
    mean_abs = checkpoint["model_state_dict"][
        "_disc_obs_norm._mean_abs"]
    return raw / torch.clamp_min(mean_abs, 1e-4)


def reward_from_logits(logits):
    probability = torch.sigmoid(logits)
    return -2.0 * torch.log(torch.clamp_min(1.0 - probability, 1e-4))


def secant_metrics(model, diff, path_steps, chunk_size):
    totals = {
        "samples": 0,
        "reward_zero": 0.0,
        "reward_negative": 0.0,
        "endpoint_gap": 0.0,
        "necessary_energy": 0.0,
        "radial_excess": 0.0,
        "tangential_energy": 0.0,
        "full_excess": 0.0,
        "total_gradient_energy": 0.0,
        "identity_abs_error": 0.0,
        "reward_cap_fraction": 0.0,
    }
    midpoints = ((torch.arange(path_steps, device=diff.device) + 0.5)
                 / path_steps)
    for start in range(0, diff.shape[0], chunk_size):
        x = diff[start:start + chunk_size]
        norm = torch.linalg.vector_norm(x, dim=-1)
        valid = norm > 1e-6
        if not bool(torch.any(valid)):
            continue
        x = x[valid]
        norm = norm[valid]
        v = x / norm.unsqueeze(-1)
        with torch.no_grad():
            reward_zero = reward_from_logits(model(torch.zeros_like(x)))
            reward_negative = reward_from_logits(model(x))
            c = (reward_negative - reward_zero) / norm

        radial_excess = torch.zeros_like(norm)
        tangential = torch.zeros_like(norm)
        total_energy = torch.zeros_like(norm)
        cap_count = torch.zeros_like(norm)
        for midpoint in midpoints:
            point = (midpoint * x).detach().requires_grad_(True)
            logits = model(point)
            reward = reward_from_logits(logits)
            gradient = torch.autograd.grad(reward.sum(), point)[0]
            radial_slope = torch.sum(gradient * v, dim=-1)
            gradient_energy = torch.sum(torch.square(gradient), dim=-1)
            radial_excess += torch.square(radial_slope - c) / path_steps
            tangential += torch.clamp_min(
                gradient_energy - torch.square(radial_slope), 0.0
            ) / path_steps
            total_energy += gradient_energy / path_steps
            cap_count += (logits > -math.log(1e-4)).float() / path_steps

        necessary = torch.square(c)
        full_excess = radial_excess + tangential
        identity_error = torch.abs(
            total_energy - necessary - full_excess)
        count = x.shape[0]
        totals["samples"] += count
        values = {
            "reward_zero": reward_zero,
            "reward_negative": reward_negative,
            "endpoint_gap": reward_zero - reward_negative,
            "necessary_energy": necessary,
            "radial_excess": radial_excess,
            "tangential_energy": tangential,
            "full_excess": full_excess,
            "total_gradient_energy": total_energy,
            "identity_abs_error": identity_error,
            "reward_cap_fraction": cap_count,
        }
        for name, value in values.items():
            totals[name] += float(value.sum().item())

    samples = totals.pop("samples")
    result = {name: value / samples for name, value in totals.items()}
    result["samples"] = samples
    result["radial_share_of_full_excess"] = (
        result["radial_excess"] / max(result["full_excess"], 1e-12))
    result["tangential_share_of_full_excess"] = (
        result["tangential_energy"] / max(result["full_excess"], 1e-12))
    result["full_excess_over_total"] = (
        result["full_excess"]
        / max(result["total_gradient_energy"], 1e-12))
    return result


def gate_one(specs, shared_label, sample_count, path_steps, device, seed):
    checkpoints = {
        label: load_checkpoint(Path(path)) for label, path in specs.items()
    }
    generator = torch.Generator().manual_seed(seed)
    shared_raw_all = replay_residuals(checkpoints[shared_label])
    shared_indices = torch.randperm(
        shared_raw_all.shape[0], generator=generator)[:sample_count]
    shared_raw = shared_raw_all[shared_indices]
    report = {}
    for label, checkpoint in checkpoints.items():
        model, architecture = build_disc(checkpoint, device)
        self_raw_all = replay_residuals(checkpoint)
        self_indices = torch.randperm(
            self_raw_all.shape[0], generator=generator)[:sample_count]
        streams = {
            "self_replay": self_raw_all[self_indices],
            f"shared_{shared_label}_replay": shared_raw,
        }
        stream_report = {}
        for stream_name, raw in streams.items():
            diff = normalize(checkpoint, raw).to(device)
            stream_report[stream_name] = secant_metrics(
                model, diff, path_steps=path_steps, chunk_size=32)
        report[label] = {
            "checkpoint": specs[label],
            "trainer_state": checkpoint.get("trainer_state", {}),
            "architecture": architecture,
            "streams": stream_report,
        }
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return report


def gate_two(dimension=172, directions=20000, seed=0):
    w = torch.tensor([3.0, 4.0], dtype=torch.float64)
    e1 = torch.tensor([1.0, 0.0], dtype=torch.float64)
    e2 = torch.tensor([0.0, 1.0], dtype=torch.float64)

    def affine_excess(v):
        tangent = w - torch.dot(w, v) * v
        return float(torch.dot(tangent, tangent).item())

    generator = torch.Generator().manual_seed(seed)
    high_w = torch.randn(dimension, generator=generator, dtype=torch.float64)
    rays = torch.randn(
        directions, dimension, generator=generator, dtype=torch.float64)
    rays /= torch.linalg.vector_norm(rays, dim=-1, keepdim=True)
    projections = torch.sum(rays * high_w, dim=-1)
    empirical_ratio = float(torch.mean(
        torch.sum(torch.square(high_w), dim=-1)
        - torch.square(projections)).item() / torch.sum(torch.square(high_w)))
    expected_ratio = 1.0 - 1.0 / dimension

    rho = 1.0
    cone = []
    for slope in (1.0, 10.0, 100.0):
        # For R(x)=M-slope*||x|| and every t in (0,1], grad=-slope*v,
        # while the endpoint secant is the same vector.
        cone.append({
            "slope": slope,
            "full_secant_excess": 0.0,
            "gradient_energy": slope * slope,
            "endpoint_gap": slope * rho,
        })

    return {
        "affine_two_ray": {
            "gradient": w.tolist(),
            "ray_e1_full_excess": affine_excess(e1),
            "ray_e2_full_excess": affine_excess(e2),
            "ray_e1_radial_excess": 0.0,
            "ray_e2_radial_excess": 0.0,
            "interpretation": (
                "Full-vector excess penalizes the gradient needed by the "
                "other ray although the reward is affine and spike-free."),
        },
        "isotropic_high_dimensional": {
            "dimension": dimension,
            "num_directions": directions,
            "empirical_full_excess_fraction": empirical_ratio,
            "analytic_fraction": expected_ratio,
        },
        "radial_cone": cone,
        "full_vector_global_claim_passes": False,
        "radial_only_preserves_affine_cross_ray_signal": True,
        "radial_only_controls_absolute_sharpness": False,
    }


def _flat(parameters, attribute="data"):
    values = []
    for parameter in parameters:
        value = parameter if attribute == "data" else getattr(parameter, attribute)
        values.append(value.reshape(-1))
    return torch.cat(values)


def _set_flat(parameters, vector):
    offset = 0
    with torch.no_grad():
        for parameter in parameters:
            count = parameter.numel()
            parameter.copy_(vector[offset:offset + count].view_as(parameter))
            offset += count


def classification_loss(model, diff):
    positive = model(torch.zeros_like(diff))
    negative = model(diff)
    loss = 0.5 * (
        F.binary_cross_entropy_with_logits(
            positive, torch.ones_like(positive))
        + F.binary_cross_entropy_with_logits(
            negative, torch.zeros_like(negative)))
    return loss + 0.01 * torch.sum(torch.square(model._disc_logits.weight))


def secant_loss(model, diff, t, kind):
    point = (t * diff).requires_grad_(True)
    reward_path = 2.0 * F.softplus(model(point))
    gradient = torch.autograd.grad(
        reward_path.sum(), point, create_graph=True)[0]
    with torch.no_grad():
        reward_zero = 2.0 * F.softplus(model(torch.zeros_like(diff)))
        reward_negative = 2.0 * F.softplus(model(diff))
        norm = torch.clamp_min(
            torch.linalg.vector_norm(diff, dim=-1), 1e-6)
        v = diff / norm.unsqueeze(-1)
        c = (reward_negative - reward_zero) / norm
    radial = torch.sum(gradient * v, dim=-1)
    if kind == "radial":
        return torch.mean(torch.square(radial - c))
    target = c.unsqueeze(-1) * v
    return torch.mean(torch.sum(torch.square(gradient - target), dim=-1))


def evaluate_losses(model, diff, t, kind):
    cls = float(classification_loss(model, diff).detach().item())
    sec = float(secant_loss(model, diff, t, kind).detach().item())
    return cls, sec


def optimizer_probe_once(checkpoint, raw, kind, device, seed):
    model, architecture = build_disc(checkpoint, device, trainable=True)
    if architecture["grouped"]:
        raise ValueError("Optimizer probe requires the dense checkpoint")
    model.eval()
    parameters = list(model.parameters())
    optimizer = torch.optim.SGD(
        parameters, lr=2.5e-4, momentum=0.9, weight_decay=1e-4)
    optimizer.load_state_dict(copy.deepcopy(
        checkpoint["optimizer_state_dicts"]["_disc_optimizer"]["optimizer"]))
    base = _flat(parameters).detach().clone()
    diff = normalize(checkpoint, raw).to(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    t = torch.rand(
        (diff.shape[0], 1), generator=generator, device=device)

    primary_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    cls_current_tensor = classification_loss(model, diff)
    cls_current_tensor.backward()
    g_cls = _flat(parameters, "grad").detach().clone()
    optimizer.step()
    proposal = _flat(parameters).detach().clone()
    p_cls = proposal - base
    primary_seconds = time.perf_counter() - primary_start
    cls_proposal, sec_proposal = evaluate_losses(model, diff, t, kind)

    _set_flat(parameters, base)
    correction_start = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    sec_current_tensor = secant_loss(model, diff, t, kind)
    g_sec_values = torch.autograd.grad(sec_current_tensor, parameters)
    g_sec = torch.cat([value.reshape(-1) for value in g_sec_values]).detach()
    projection = torch.dot(g_cls, g_sec) / torch.clamp_min(
        torch.dot(g_cls, g_cls), 1e-20)
    q = -(g_sec - projection * g_cls)
    q_norm = torch.linalg.vector_norm(q)
    p_norm = torch.linalg.vector_norm(p_cls)
    max_alpha = float((p_norm / torch.clamp_min(q_norm, 1e-20)).item())

    candidates = []
    tolerance = 1e-8 * max(1.0, abs(cls_proposal))
    for fraction in (0.0, 0.0625, 0.125, 0.25, 0.5, 1.0):
        candidate = proposal + (fraction * max_alpha) * q
        _set_flat(parameters, candidate)
        cls_value, sec_value = evaluate_losses(model, diff, t, kind)
        candidates.append({
            "fraction": fraction,
            "classification_loss": cls_value,
            "secant_loss": sec_value,
            "accepted": cls_value <= cls_proposal + tolerance,
        })
    accepted = [value for value in candidates if value["accepted"]]
    selected = min(accepted, key=lambda value: value["secant_loss"])
    correction_seconds = time.perf_counter() - correction_start

    _set_flat(parameters, base)
    cls_current, sec_current = evaluate_losses(model, diff, t, kind)
    return {
        "kind": kind,
        "classification_current": cls_current,
        "classification_proposal": cls_proposal,
        "secant_current": sec_current,
        "secant_proposal": sec_proposal,
        "p_cls_norm": float(p_norm.item()),
        "q_norm": float(q_norm.item()),
        "g_cls_dot_q": float(torch.dot(g_cls, q).item()),
        "selected": selected,
        "nonzero_accepted": any(
            value["accepted"] and value["fraction"] > 0
            for value in candidates),
        "candidates": candidates,
        "primary_seconds": primary_seconds,
        "correction_seconds": correction_seconds,
        "cost_ratio_correction_over_primary": (
            correction_seconds / max(primary_seconds, 1e-12)),
    }


def gate_three(checkpoint_path, batches, batch_size, device, seed):
    checkpoint = load_checkpoint(Path(checkpoint_path))
    raw_all = replay_residuals(checkpoint)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(
        raw_all.shape[0], generator=generator)[:batches * batch_size]
    selected_raw = raw_all[indices]
    report = {"checkpoint": checkpoint_path, "methods": {}}
    for kind in ("full", "radial"):
        results = []
        for batch_id in range(batches):
            start = batch_id * batch_size
            raw = selected_raw[start:start + batch_size]
            results.append(optimizer_probe_once(
                checkpoint, raw, kind, device, seed + batch_id))
        nonzero = sum(value["selected"]["fraction"] > 0 for value in results)
        accepted_any = sum(value["nonzero_accepted"] for value in results)
        report["methods"][kind] = {
            "batches": results,
            "selected_nonzero_fraction": nonzero / batches,
            "any_nonzero_accepted_fraction": accepted_any / batches,
            "mean_cost_ratio_correction_over_primary": sum(
                value["cost_ratio_correction_over_primary"]
                for value in results) / batches,
            "mean_selected_secant_change_vs_proposal": sum(
                value["selected"]["secant_loss"]
                - value["secant_proposal"] for value in results) / batches,
            "mean_selected_cls_change_vs_proposal": sum(
                value["selected"]["classification_loss"]
                - value["classification_proposal"] for value in results
            ) / batches,
        }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--sample-count", type=int, default=128)
    parser.add_argument("--path-steps", type=int, default=4)
    parser.add_argument("--probe-batches", type=int, default=4)
    parser.add_argument("--probe-batch-size", type=int, default=64)
    args = parser.parse_args()
    device = torch.device(args.device)
    specs = {
        "official_gp2_ptoff_failed": (
            "output/paper_benchmark/add_climb_2k_8192_seed0/checkpoint.pt"),
        "dense_full_sn_pton_current": (
            "output/dense_full_sn_climb_pton_2k_8912_seed0/checkpoint.pt"),
        "a30_group_sn_pton_success": (
            "/home/y/.local/share/Trash/files/"
            "gadd_group_separable_sn_climb_2k_8912_seed0/pt_on/"
            "checkpoint.pt"),
        "a30_group_sn_ptoff_failed": (
            "/home/y/.local/share/Trash/files/"
            "gadd_group_separable_sn_climb_2k_8912_seed0/pt_off/"
            "checkpoint.pt"),
    }
    for path in specs.values():
        if not Path(path).is_file():
            raise FileNotFoundError(path)

    report = {
        "gate_1_checkpoint_geometry": gate_one(
            specs, shared_label="dense_full_sn_pton_current",
            sample_count=args.sample_count, path_steps=args.path_steps,
            device=device, seed=0),
        "gate_2_function_counterexamples": gate_two(),
        "gate_3_optimizer_probe": gate_three(
            specs["dense_full_sn_pton_current"],
            batches=args.probe_batches, batch_size=args.probe_batch_size,
            device=device, seed=1234),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
