#!/usr/bin/env python3
"""Freeze an isolated Sill-1997 min-max MIRA Backflip pressure test.

The default freezes a snapshot and runs CPU/static checks only. GPU training
requires explicit ``--launch`` after the manifest is reviewed.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import shutil
import sys
import run_mira_diagonal as trainer
ROOT = trainer.ROOT
MODEL = Path("mimickit/learning/mira_monotone_model.py")
AGENT = Path("mimickit/learning/mira_monotone_agent.py")
HELPER = Path("mimickit/learning/mira_sill_minmax.py")
RUNNER_REL = trainer.RUNNER_REL
SCHEMA = 32

def replace_once(path, old, new):
    path = Path(path); source = path.read_text()
    if source.count(old) != 1: raise ValueError(f"Expected one patch target in {path}: {old[:100]!r}")
    path.write_text(source.replace(old, new))

def patch_model(snapshot):
    model = snapshot / MODEL
    replace_once(model, "from learning.weight_constraints import enforce_nonnegative_contraction_, spectral_norm",
                 "from learning.weight_constraints import spectral_norm\nfrom learning.mira_sill_minmax import SillMinMaxEnergy")
    old_build = '''        self._disc_energy = nn.Sequential(
            constrained_linear(self._pair_metrics.feature_dim, HIDDEN_WIDTHS[0]), nn.ReLU(),
            constrained_linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]), nn.ReLU(),
            constrained_linear(HIDDEN_WIDTHS[1], 1))'''
    new_build = '''        self._disc_energy = SillMinMaxEnergy(
            self._pair_metrics.feature_dim, plane_count=SILL_PLANE_COUNT, groups=SILL_GROUP_COUNT)'''
    replace_once(model, old_build, new_build)
    replace_once(model, "MONOTONE_SCHEMA_VERSION = 29",
                 "MONOTONE_SCHEMA_VERSION = 32\nSILL_PLANE_COUNT = 341\nSILL_GROUP_COUNT = 31")
    old_enforce = '''    def enforce_disc_weight_constraints(self):
        for layer in self._disc_energy:
            if isinstance(layer, nn.Linear):
                enforce_nonnegative_contraction_(layer.weight)'''
    replace_once(model, old_enforce, '''    def enforce_disc_weight_constraints(self):
        # Effective row slopes are projected in every forward pass.
        self._disc_energy.enforce_constraints_()''')
    replace_once(model, "        return self._disc_energy[-1].weight.flatten()",
                 "        return self._disc_energy.effective_slopes.flatten()")
    old_load = '''        else:
            for index, layer in enumerate(self._disc_energy):
                key = prefix + f"_disc_energy.{index}.weight"
                weight = state_dict.get(key)
                if isinstance(layer, nn.Linear) and weight is not None and weight.shape == layer.weight.shape:
                    # Stored weights are used directly: do not accept a checkpoint
                    # that bypasses the training constraint.
                    if (not weight.is_floating_point() or not torch.isfinite(weight).all()
                            or (weight < 0).any()
                            or spectral_norm(weight) > 1 + 8 * torch.finfo(weight.dtype).eps):
                        error_msgs.append("Invalid nonnegative contraction in MIRA checkpoint: " + key)'''
    new_load = '''        else:
            key = prefix + "_disc_energy.raw_slopes"
            raw = state_dict.get(key)
            if raw is not None and raw.shape == self._disc_energy.raw_slopes.shape:
                effective = SillMinMaxEnergy.effective_from_raw(raw)
                if (not raw.is_floating_point() or not torch.isfinite(raw).all()
                        or not torch.isfinite(effective).all() or (effective < 0).any()
                        or (torch.linalg.vector_norm(effective, dim=-1) > 1 + 8 * torch.finfo(raw.dtype).eps).any()):
                    error_msgs.append("Invalid Sill effective row slopes in MIRA checkpoint: " + key)'''
    replace_once(model, old_load, new_load)
    agent = snapshot / AGENT
    replace_once(agent, '"first_layer": "direct_nonnegative_dense_no_group_budget",',
                 '"first_layer": "sill_minmax_341_planes_31_groups",')
    replace_once(agent, '"weight_constraint": "post_sgd_nonnegative_clip_uniform_spectral_cap",',
                 '"weight_constraint": "Sill_abs_row_unit_projection_in_forward",')

def patch_runner(snapshot):
    runner = snapshot / RUNNER_REL
    old_layout = '''        model = agent._model
        assert FEATURE_DIM == 64
        assert int(model._monotone_schema_version) == MONOTONE_SCHEMA_VERSION
        layers = []
        for index, dim_in, dim_out in zip(layer_indices, dimensions, dimensions[1:]):
            layer = model._disc_energy[index]
            assert type(layer) is torch.nn.Linear
            assert layer.weight.shape == (dim_out, dim_in) and layer.bias.shape == (dim_out,)
            layers.append(layer)
        assert all(type(model._disc_energy[index]) is torch.nn.ReLU for index in (1, 3))
        optimizer_params = [p for group in agent._disc_optimizer._optimizer.param_groups for p in group["params"]]
        assert len({id(p) for p in optimizer_params}) == len(optimizer_params)
        assert ({id(p) for layer in layers for p in (layer.weight, layer.bias)}
                == {id(p) for p in model.get_disc_params()}
                == {id(p) for p in optimizer_params})
        return [layer.weight for layer in layers]'''
    new_layout = '''        model = agent._model
        assert FEATURE_DIM == 64
        assert int(model._monotone_schema_version) == MONOTONE_SCHEMA_VERSION
        energy = model._disc_energy
        assert energy.plane_count == 341 and energy.group_count == 31
        assert energy.raw_slopes.shape == (341, FEATURE_DIM)
        params = [energy.raw_slopes, energy.plane_bias, energy.output_bias]
        optimizer_params = [p for group in agent._disc_optimizer._optimizer.param_groups for p in group["params"]]
        assert {id(p) for p in params} == {id(p) for p in model.get_disc_params()} == {id(p) for p in optimizer_params}
        return [energy.effective_slopes]'''
    replace_once(runner, old_layout, new_layout)
    old_verify = '''        norms, ranges = [], []
        for weight in weights:
            assert bool(torch.isfinite(weight).all()) and bool((weight >= 0).all())
            # Independent audit, deliberately use CPU double SVD rather than
            # the training update's numerical method.
            norms.append(float(torch.linalg.matrix_norm(weight.detach().to("cpu", torch.float64), ord=2)))
            ranges.append({"min": float(weight.min()), "max": float(weight.max()),
                           "zero_fraction": float((weight == 0).float().mean())})
        assert all(norm <= 1 + 4 * torch.finfo(weight.dtype).eps for norm, weight in zip(norms, weights)), norms
        return norms, ranges'''
    new_verify = '''        norms, ranges = [], []
        for weight in weights:
            assert bool(torch.isfinite(weight).all()) and bool((weight >= 0).all())
            rows = torch.linalg.vector_norm(weight.detach().to("cpu", torch.float64), dim=1)
            norms.append(float(rows.max()))
            ranges.append({"min": float(weight.min()), "max": float(weight.max()),
                           "zero_fraction": float((weight == 0).float().mean()),
                           "row_norm_max": float(rows.max()), "row_norm_mean": float(rows.mean())})
            assert bool((rows <= 1 + 8 * torch.finfo(weight.dtype).eps).all()), norms
        return norms, ranges'''
    replace_once(runner, old_verify, new_verify)
    old_checkpoint = '''        saved_weights = [checkpoint["model_state_dict"][f"_model._disc_energy.{index}.weight"]
                         for index in layer_indices]
        for weight, dim_in, dim_out in zip(saved_weights, dimensions, dimensions[1:]):
            assert weight.shape == (dim_out, dim_in)
        norms, _ = verify_weights(saved_weights)'''
    new_checkpoint = '''        saved_raw = checkpoint["model_state_dict"]["_model._disc_energy.raw_slopes"]
        assert saved_raw.shape == (341, FEATURE_DIM)
        saved_weights = [saved_raw]
        norms, _ = verify_weights(layout(agent))'''
    replace_once(runner, old_checkpoint, new_checkpoint)
    replace_once(runner, '"experiment": "MIRA64_direct_nonnegative_unit_operator_norm_weights"',
                 '"experiment": "MIRA64_Sill_1997_minmax_monotone_energy"')

def freeze(out, updates):
    out = Path(out)
    if out.exists(): raise FileExistsError(out)
    trainer.freeze(argparse.Namespace(out_dir=out, motion="backflip", preflight=False, freeze_only=True, updates=updates))
    snapshot = out / "source_snapshot"
    shutil.copyfile(Path(__file__).with_name(HELPER.name), snapshot / HELPER)
    patch_model(snapshot); patch_runner(snapshot)
    manifest_path = out / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sill_architecture"] = {"reference": "Sill_1997", "planes": 341, "groups": 31,
                                     "planes_per_group": 11, "energy": "output_bias_plus_min_max_affine",
                                     "effective_slope": "abs(raw)/max(1,row_l2)", "constraint": "forward_exact",
                                     "disc_logit_reg_scope": "all_effective_plane_slopes",
                                     "disc_logit_reg_expression": "sum(square(get_disc_logit_weights()))_unchanged"}
    manifest["files"] = {str(path.relative_to(snapshot)): trainer.sha(path) for path in sorted(snapshot.rglob("*")) if path.is_file()}
    trainer.save(manifest_path, manifest)
    for relative in (MODEL, AGENT, RUNNER_REL, HELPER): compile((snapshot / relative).read_text(), str(relative), "exec")

def cpu_checks(out):
    import importlib.util, torch
    snapshot = Path(out) / "source_snapshot"
    spec = importlib.util.spec_from_file_location("sill_check", snapshot / HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.manual_seed(8121); energy = module.SillMinMaxEnergy(64, 341, 31)
    x = torch.rand(256, 64); y = torch.rand_like(x)
    energy.raw_slopes.data[0, 0] = -3
    effective = energy.effective_slopes
    rows = torch.linalg.vector_norm(effective, dim=-1)
    assert bool((effective >= 0).all()) and bool((rows <= 1 + 1e-6).all())
    assert bool((energy(x * .9) <= energy(x) + 1e-6).all())
    assert bool(((energy(x) - energy(y)).abs() <= (x - y).norm(dim=-1, keepdim=True) + 3e-5).all())
    probe = torch.rand(4, 64); before = energy(probe)
    state = energy.state_dict(); restored = module.SillMinMaxEnergy(64, 341, 31); restored.load_state_dict(state)
    assert torch.equal(before, restored(probe))
    return {"status":"passed", "planes":341, "groups":31, "planes_per_group":11,
            "max_effective_row_norm":float(rows.max()), "monotone_and_lipschitz":True, "state_reload":True}

def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=100); parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--launch", action="store_true", help="Explicitly allow GPU training")
    args = parser.parse_args()
    if not 3 <= args.updates <= 2000: parser.error("--updates must be between 3 and 2000")
    out = args.out_dir.resolve(); freeze(out, args.updates); result = cpu_checks(out); trainer.save(out / "sill_cpu_checks.json", result)
    print(json.dumps({"status":"frozen_and_cpu_checked", "out_dir":str(out), **result}, indent=2))
    if args.freeze_only or not args.launch:
        print("GPU launch withheld; review source_manifest.json and sill_cpu_checks.json first."); return
    env = os.environ.copy(); env.setdefault("OMP_NUM_THREADS", "4"); env.setdefault("MKL_NUM_THREADS", "4")
    os.execve(sys.executable, [sys.executable, "-u", str(out / RUNNER_REL), "--frozen-worker", "--motion", "backflip", "--updates", str(args.updates), "--out-dir", str(out)], env)
if __name__ == "__main__": main()
