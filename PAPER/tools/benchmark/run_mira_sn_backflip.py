#!/usr/bin/env python3
"""Run the matched nonnegative exact spectral-normalization control.

The current production source is frozen first. Only the effective-weight
normalization in the frozen model is changed from the q-free diagonal formula
to P / ||P||_2, keeping P=V^2, initialization, loss, PPO, and seed identical.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[3]
UPDATES = 2000
OUT = ROOT / "output/mira_sn_backflip_2000_8192_seed0_20261128"
BASE_RUNNER = ROOT / "PAPER/tools/benchmark/run_mira_diagonal.py"
RUNNER_REL = Path("PAPER/tools/benchmark/run_mira_diagonal.py")


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def patch_snapshot():
    model_path = OUT / "source_snapshot/mimickit/learning/mira_monotone_model.py"
    model = model_path.read_text()
    model = model.replace("MONOTONE_SCHEMA_VERSION = 30", "MONOTONE_SCHEMA_VERSION = 31", 1)
    old = """        column_budget = positive.T @ positive.sum(dim=1)
        live = column_budget > 0
        safe_budget = torch.where(live, column_budget, torch.ones_like(column_budget))
        normalized = positive * safe_budget.rsqrt().unsqueeze(0)
        return torch.where(live.unsqueeze(0), normalized, torch.zeros_like(normalized)).to(
            self.raw_weight.dtype)
"""
    new = """        spectral = torch.linalg.matrix_norm(positive, ord=2)
        normalized = positive / spectral.clamp_min(torch.finfo(torch.float64).tiny)
        return normalized.to(self.raw_weight.dtype)
"""
    if old not in model:
        raise RuntimeError("q-free diagonal normalization body was not found in frozen model")
    model_path.write_text(model.replace(old, new, 1))

    agent_path = OUT / "source_snapshot/mimickit/learning/mira_monotone_agent.py"
    agent = agent_path.read_text()
    if "forward_q_free_diagonal_normalization" not in agent:
        raise RuntimeError("MIRA normalization metadata was not found in frozen agent")
    agent_path.write_text(agent.replace(
        "forward_q_free_diagonal_normalization", "forward_exact_spectral_normalization", 1))

    runner_path = OUT / "source_snapshot" / RUNNER_REL
    runner = runner_path.read_text()
    if "MIRA64_q_free_nonnegative_normalized_weights" not in runner:
        raise RuntimeError("q-free experiment protocol was not found in frozen runner")
    runner_path.write_text(runner.replace(
        "MIRA64_q_free_nonnegative_normalized_weights",
        "MIRA64_nonnegative_exact_spectral_normalization", 1))

    manifest_path = OUT / "source_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["experiment"] = "MIRA64_nonnegative_exact_spectral_normalization"
    manifest["files"] = {
        relative: sha(OUT / "source_snapshot" / relative)
        for relative in manifest["files"]
    }
    save(manifest_path, manifest)


def main():
    if (OUT / "source_manifest.json").exists():
        raise RuntimeError(f"Refusing to reuse {OUT}")
    subprocess.run([
        sys.executable, "-u", str(BASE_RUNNER), "--updates", str(UPDATES),
        "--motion", "backflip", "--out-dir", str(OUT), "--freeze-only",
    ], cwd=ROOT, check=True)
    patch_snapshot()
    worker = OUT / "source_snapshot" / RUNNER_REL
    os.execve(sys.executable, [sys.executable, "-u", str(worker), "--frozen-worker",
                               "--motion", "backflip", "--updates", str(UPDATES),
                               "--out-dir", str(OUT)], os.environ.copy())


if __name__ == "__main__":
    main()
