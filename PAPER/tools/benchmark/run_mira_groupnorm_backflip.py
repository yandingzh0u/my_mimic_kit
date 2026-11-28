#!/usr/bin/env python3
"""Isolated three-update MIRA group-weighted norm smoke.

The production tree is never modified.  A frozen source snapshot replaces only
MIRA's discriminator with E(z)=c+sum_g a_g ||z_g||_2, a>=0, ||a||_2<=1.
BCE, ADD reward, PPO, physical-error extraction and all non-discriminator
training settings remain from the maintained runner.
"""
from __future__ import annotations
import argparse, json, shutil, sys
from pathlib import Path
import run_mira_diagonal as trainer

ROOT = trainer.ROOT
MODEL = Path("mimickit/learning/mira_monotone_model.py")
AGENT = Path("mimickit/learning/mira_monotone_agent.py")
HELPER = Path("mimickit/learning/mira_groupnorm_layer.py")
RUNNER_REL = trainer.RUNNER_REL
SCHEMA = 33


def replace_once(path, old, new):
    path = Path(path); source = path.read_text()
    if source.count(old) != 1:
        raise ValueError(f"Expected one patch target in {path}: {old[:100]!r}")
    path.write_text(source.replace(old, new))


def patch_model(snapshot):
    model = snapshot / MODEL
    replace_once(model,
        "from learning.weight_constraints import enforce_nonnegative_contraction_, spectral_norm",
        "from learning.weight_constraints import spectral_norm\nfrom learning.mira_groupnorm_layer import GroupWeightedNormEnergy")
    old_build = '''        self._disc_energy = nn.Sequential(
            constrained_linear(self._pair_metrics.feature_dim, HIDDEN_WIDTHS[0]), nn.ReLU(),
            constrained_linear(HIDDEN_WIDTHS[0], HIDDEN_WIDTHS[1]), nn.ReLU(),
            constrained_linear(HIDDEN_WIDTHS[1], 1))'''
    new_build = '''        self._disc_energy = GroupWeightedNormEnergy(self._pair_metrics.feature_dim)'''
    replace_once(model, old_build, new_build)
    replace_once(model, "MONOTONE_SCHEMA_VERSION = 29",
                 f"MONOTONE_SCHEMA_VERSION = {SCHEMA}")
    old_enforce = '''    def enforce_disc_weight_constraints(self):
        for layer in self._disc_energy:
            if isinstance(layer, nn.Linear):
                enforce_nonnegative_contraction_(layer.weight)'''
    replace_once(model, old_enforce, '''    def enforce_disc_weight_constraints(self):
        self._disc_energy.enforce_constraints_()''')
    replace_once(model, "        return self._disc_energy[-1].weight.flatten()",
                 "        return self._disc_energy.effective_group_weights.flatten()")
    # Replace only the final model checkpoint validator (the pair metric has its own loader).
    source = model.read_text()
    marker = "    def _load_from_state_dict(self, state_dict, prefix, local_metadata,\n                              strict, missing_keys, unexpected_keys, error_msgs):"
    begin = source.rfind("\n" + marker)
    if begin < 0:
        raise ValueError("Could not find model checkpoint loader")
    replacement = '''
    def _load_from_state_dict(self, state_dict, prefix, local_metadata,
                              strict, missing_keys, unexpected_keys, error_msgs):
        version = state_dict.get(prefix + "_monotone_schema_version")
        if (version is None or version.dtype != torch.int64 or version.numel() != 1
                or int(version.item()) != MONOTONE_SCHEMA_VERSION):
            error_msgs.append("Cannot load a different monotone physical-error schema")
        else:
            raw = state_dict.get(prefix + "_disc_energy.raw_group_weights")
            if raw is not None:
                effective = GroupWeightedNormEnergy.effective_group_weights.fget(
                    self._disc_energy)
                if (not raw.is_floating_point() or not torch.isfinite(raw).all()
                        or raw.shape != (7,) or not torch.isfinite(effective).all()
                        or (effective < 0).any()
                        or torch.linalg.vector_norm(effective) > 1 + 8 * torch.finfo(raw.dtype).eps):
                    error_msgs.append("Invalid group-weighted norm parameters in MIRA checkpoint")
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys, error_msgs)
'''
    model.write_text(source[:begin] + "\n" + replacement)
    # Keep source syntax valid and remove now-unused direct-layer helper imports only by harmless retention.


def patch_agent(snapshot):
    agent = snapshot / AGENT
    replace_once(agent, '"first_layer": "direct_nonnegative_dense_no_group_budget",',
                 '"first_layer": "seven_group_weighted_l2_norm",')
    replace_once(agent, '"activation": "relu",',
                 '"activation": "group_l2_norm",')
    replace_once(agent, '"weight_constraint": "post_sgd_nonnegative_clip_uniform_spectral_cap",',
                 '"weight_constraint": "nonnegative_unit_l2_group_weights_in_forward",')


def patch_runner(snapshot):
    runner = snapshot / RUNNER_REL
    source = runner.read_text()
    # Replace local layout() function body through the next decorator.
    begin = source.index("    def layout(agent):")
    end = source.index("    @torch.no_grad()\n    def verify_weights", begin)
    layout = '''    def layout(agent):
        model = agent._model
        assert FEATURE_DIM == 64
        assert int(model._monotone_schema_version) == MONOTONE_SCHEMA_VERSION
        energy = model._disc_energy
        assert energy.feature_dim == FEATURE_DIM
        params = [energy.raw_group_weights, energy.output_bias]
        optimizer_params = [p for group in agent._disc_optimizer._optimizer.param_groups for p in group["params"]]
        assert {id(p) for p in params} == {id(p) for p in model.get_disc_params()} == {id(p) for p in optimizer_params}
        return [energy.effective_group_weights]

'''
    runner.write_text(source[:begin] + layout + source[end:])
    source = runner.read_text()
    begin = source.index("    @torch.no_grad()\n    def verify_weights")
    end = source.index("    def fixed_probe", begin)
    verify = '''    @torch.no_grad()
    def verify_weights(weights):
        assert len(weights) == 1
        weight = weights[0]
        assert bool(torch.isfinite(weight).all()) and bool((weight >= 0).all())
        norm = float(torch.linalg.vector_norm(weight.detach().to("cpu", torch.float64)))
        assert norm <= 1 + 8 * torch.finfo(weight.dtype).eps, norm
        ranges = [{"min": float(weight.min()), "max": float(weight.max()),
                   "zero_fraction": float((weight == 0).float().mean())}]
        return [norm], ranges

'''
    runner.write_text(source[:begin] + verify + source[end:])
    source = runner.read_text()
    # Checkpoint audit reads raw scalar vector and then audits effective live model.
    old = '''        saved_weights = [checkpoint["model_state_dict"][f"_model._disc_energy.{index}.weight"]
                         for index in layer_indices]
        for weight, dim_in, dim_out in zip(saved_weights, dimensions, dimensions[1:]):
            assert weight.shape == (dim_out, dim_in)
        norms, _ = verify_weights(saved_weights)'''
    new = '''        saved_raw = checkpoint["model_state_dict"]["_model._disc_energy.raw_group_weights"]
        assert saved_raw.shape == (7,)
        saved_weights = [agent._model._disc_energy.effective_group_weights.detach()]
        norms, _ = verify_weights(saved_weights)'''
    replace_once(runner, old, new)
    replace_once(runner, '"experiment": "MIRA64_direct_nonnegative_unit_operator_norm_weights"',
                 '"experiment": "MIRA64_group_weighted_l2_norm_energy"')
    # The dimensions variables remain for generic probes but no longer describe the discriminator layout.


def freeze(out: Path, updates: int):
    if out.exists():
        raise FileExistsError(out)
    trainer.freeze(argparse.Namespace(out_dir=out, motion="backflip", preflight=False,
                                     freeze_only=True, updates=updates))
    snapshot = out / "source_snapshot"
    shutil.copyfile(Path(__file__).with_name(HELPER.name), snapshot / HELPER)
    patch_model(snapshot); patch_agent(snapshot); patch_runner(snapshot)
    manifest_path = out / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["group_norm_architecture"] = {
        "energy": "output_bias_plus_sum_group_l2_norm",
        "groups": [[0,1],[1,2],[2,16],[16,30],[30,58],[58,61],[61,64]],
        "coefficients": "abs(raw)/max(1,l2_norm(abs(raw)))",
        "constraint": "nonnegative_and_l2_unit_ball_in_forward",
        "loss_reward_ppo": "unchanged"}
    manifest["files"] = {str(path.relative_to(snapshot)): trainer.sha(path)
                         for path in sorted(snapshot.rglob("*")) if path.is_file()}
    trainer.save(manifest_path, manifest)
    for relative in (MODEL, AGENT, RUNNER_REL, HELPER):
        compile((snapshot / relative).read_text(), str(relative), "exec")


def cpu_checks(out: Path):
    import importlib.util, torch
    snapshot = Path(out) / "source_snapshot"
    spec = importlib.util.spec_from_file_location("groupnorm_check", snapshot / HELPER)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    torch.manual_seed(8121)
    energy = module.GroupWeightedNormEnergy(64)
    x = torch.rand(256, 64); y = torch.rand_like(x)
    before = energy(x); improved = x * .9
    assert bool((energy(improved) <= before + 1e-6).all())
    assert bool(((energy(x) - energy(y)).abs() <= (x-y).norm(dim=-1, keepdim=True) + 3e-5).all())
    w = energy.effective_group_weights
    assert bool((w >= 0).all()) and float(w.norm()) <= 1 + 1e-6
    state = energy.state_dict(); restored = module.GroupWeightedNormEnergy(64); restored.load_state_dict(state)
    torch.testing.assert_close(energy(x), restored(x), rtol=0, atol=0)
    return {"status":"passed", "groups":7, "effective_group_l2":float(w.norm()),
            "monotone_and_lipschitz":True, "state_reload":True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--freeze-only", action="store_true")
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()
    if not 3 <= args.updates <= 2000:
        parser.error("--updates must be between 3 and 2000")
    out = args.out_dir.resolve(); freeze(out, args.updates)
    result = cpu_checks(out); trainer.save(out / "groupnorm_cpu_checks.json", result)
    print(json.dumps({"status":"frozen_and_cpu_checked", "out_dir":str(out), **result}, indent=2))
    if args.freeze_only or not args.launch:
        print("GPU launch withheld; review source_manifest.json and groupnorm_cpu_checks.json first.")
        return
    env = __import__("os").environ.copy(); env.setdefault("OMP_NUM_THREADS", "4"); env.setdefault("MKL_NUM_THREADS", "4")
    import os
    os.execve(sys.executable, [sys.executable, "-u", str(out / "source_snapshot" / RUNNER_REL), "--frozen-worker",
                               "--motion", "backflip", "--updates", str(args.updates), "--out-dir", str(out)], env)

if __name__ == "__main__": main()
