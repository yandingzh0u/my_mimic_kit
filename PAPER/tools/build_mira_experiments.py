"""Build MIRA assets from measured seed-0 training-log evaluations.

Use the same six motions. MIRA Climb uses 3000 updates; other results use 2000.
Missing measurements remain TODO; only current measured assets are generated.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "PAPER"
BENCHMARK = ROOT / "output"
GENERATED = PAPER / "generated"
UPDATES = 2000
SAMPLES_PER_UPDATE = 8192 * 32
MAX_SAMPLES = UPDATES * SAMPLES_PER_UPDATE
TERMINAL_UPDATES = {("MIRA", "climb"): 3000}
MOTIONS = ("roll", "climb", "backflip", "crawl", "getup_facedown", "spinkick")
MOTION_NAMES = ("Roll", "Climb", "Backflip", "Crawl", "Get-Up", "Spin Kick")
HORIZON_STEPS = dict(zip(MOTIONS, (60, 299, 60, 77, 91, 60)))
METHODS = ("AMP", "DeepMimic", "ADD", "MIRA")
METRICS = (
    ("Root_Pos_Err", "Root pos.", "m"),
    ("Root_Rot_Err", "Root rot.", "rad"),
    ("Body_Pos_Err", "Body pos.", "m"),
    ("Body_Rot_Err", "Body rot.", "rad"),
    ("Dof_Vel_Err", "DoF vel.", "rad/s"),
    ("Root_Vel_Err", "Root vel.", "m/s"),
    ("Root_Ang_Vel_Err", "Root ang. vel.", "rad/s"),
)
GRID = np.array([1, *range(101, 2000, 100), 2000]) * SAMPLES_PER_UPDATE
GRID_PROGRESS = GRID / MAX_SAMPLES
ABLATION_ROOT = BENCHMARK / "dare_ablation_backflip_matched_init_2000_8192_seed0_20261114"
VARIANTS = (
    ("MIRA", None),
    ("w/o Monotone Modeling", "no_a"),
    ("w/o Diagonal Norm.", "no_b"),
    ("w/o Both Components", "no_ab"),
)


def load_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_path(method: str, motion: str) -> Path:
    if method == "MIRA":
        candidates = sorted(BENCHMARK.glob(
            f"dare_cmnn64_group_rms_{motion}_2000_8192_seed0_*"), reverse=True)
        for path in candidates:
            source = path / "train_metrics.jsonl"
            samples = TERMINAL_UPDATES.get((method, motion), UPDATES) * SAMPLES_PER_UPDATE
            if source.exists() and any(sample_count(row) == samples
                                       for row in load_lines(source)):
                return path
        return BENCHMARK / f"dare_cmnn64_group_rms_{motion}_2000_8192_seed0_MISSING"
    return BENCHMARK / f"{method.lower()}_{motion}_2k_8192_seed0"


def sample_count(row: dict) -> int:
    return int(row.get("Samples", row.get("samples", -1)))


def read_run(path: Path, motion: str, updates: int = UPDATES) -> dict | None:
    source = path / "train_metrics.jsonl"
    if not source.exists():
        return None
    rows = load_lines(source)
    samples_used = updates * SAMPLES_PER_UPDATE
    final = [r for r in rows if sample_count(r) == samples_used]
    if not final:
        return None
    deduplicated = {}
    for row in rows:
        samples = int(row.get("Samples", row.get("samples", -1)))
        if 0 < samples <= samples_used:
            deduplicated[samples] = row
    ordered = [deduplicated[s] for s in sorted(deduplicated)]
    pressure_file = path / "constraint_rounds.jsonl"
    pressure = []
    if pressure_file.exists():
        checks = {int(r["completed_updates"]): r for r in load_lines(pressure_file)
                  if int(r["completed_updates"]) <= updates}
        pressure = [checks[k] for k in sorted(checks)]
    result = {
        "path": str(source.relative_to(ROOT)),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": 0,
        "completed_updates_used": updates,
        "samples_used": samples_used,
        "logged_iteration": int(final[-1].get("Iteration", final[-1].get("iteration", updates - 1))),
        "logged_resume_segment": int(final[-1].get("resume_segment", 0)),
        "evaluation_source": "in-training reference-start evaluation",
        "independent_evaluation_used": False,
        "test_episodes": int(final[-1].get("Test_Episodes", 0)),
        "final": {key: float(final[-1][key]) for key, _, _ in METRICS},
        "test_episode_length": float(final[-1]["Test_Episode_Length"]),
        "nominal_horizon_steps": HORIZON_STEPS[motion],
        "rows": ordered,
        "pressure": pressure,
    }
    checkpoint = path / "int_models" / f"model_{updates - 1:010d}.pt"
    if checkpoint.exists():
        result["terminal_policy_checkpoint"] = str(checkpoint.relative_to(ROOT))
        result["terminal_policy_checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    if not all(math.isfinite(v) for v in result["final"].values()):
        raise ValueError(f"Non-finite tracking errors: {source}")
    if not math.isfinite(result["test_episode_length"]):
        raise ValueError(f"Non-finite test length: {source}")
    result["complete"] = result["test_episode_length"] >= HORIZON_STEPS[motion] - 1e-3
    independent = path / "eval/final/summary.json"
    result["independent_summary_available"] = independent.exists()
    if independent.exists():
        result["independent_summary_path"] = str(independent.relative_to(ROOT))
    status_file = path / "status.json"
    if status_file.exists():
        status = json.loads(status_file.read_text())
        result["run_status"] = status.get("status")
        result["total_run_updates"] = status.get("updates")
    if pressure_file.exists():
        result["pressure_source"] = str(pressure_file.relative_to(ROOT))
        result["pressure_sha256"] = hashlib.sha256(pressure_file.read_bytes()).hexdigest()
    return result


def measured_curve(run: dict, metric: str) -> np.ndarray:
    rows = run["rows"]
    x = np.array([r.get("Samples", r.get("samples")) for r in rows])
    target_samples = GRID_PROGRESS * run["samples_used"]
    if target_samples[0] < x[0] or target_samples[-1] > x[-1]:
        raise ValueError(f"Incomplete curve support: {run['path']}")
    values = np.array([r[metric] for r in rows])
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite curve: {run['path']} / {metric}")
    return np.interp(target_samples, x, values)


def run_curves(run: dict, metric: str) -> np.ndarray:
    return measured_curve(run, metric)[None, :]


def aggregate(runs: list[dict | None], metric: str,
              expected_motions: int = 6) -> np.ndarray | None:
    if len(runs) != expected_motions or not runs or any(run is None for run in runs):
        return None
    return np.mean([run_curves(run, metric) for run in runs], axis=0)


def cell(curves: np.ndarray | None, best: bool = False) -> str:
    if curves is None:
        return r"\tbd"
    center = float(curves[0, -1])
    number = f"{center:.4f}"
    return "$" + (r"\mathbf{" + number + "}" if best else number) + "$"


def write_main_table(runs: dict) -> None:
    lines = [
        r"% Generated by PAPER/tools/build_mira_experiments.py; do not hand-edit.",
        r"\begin{table*}[!t]",
        r"\caption{Tracking errors. (a) Equal-weight",
        r"macro-averages over six motions. (b) MIRA results for each motion.",
        r"All entries use measured seed 0. Lower errors are better.",
        r"Bold values mark the lowest error for each metric in panel (a) only.",
    ]
    lines += [
        r"MIRA Climb$^\dagger$ uses 3000 updates; other results use 2000.",
        r"Errors cover executed timesteps.}",
        r"\label{tab:main}", r"\centering\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\textbf{(a) Six-motion macro-average}\\[2pt]",
        r"\begin{tabular}{lcccc}\toprule",
        r"Metric $\downarrow$ & AMP & DeepMimic & ADD & MIRA$^\dagger$ \\", r"\midrule",
    ]
    for metric, label, unit in METRICS:
        curves = [aggregate([runs[m][motion] for motion in MOTIONS], metric) for m in METHODS]
        available = [float(curve[0, -1]) for curve in curves if curve is not None]
        best = min(available) if available else None
        values = [cell(curve, curve is not None and float(curve[0, -1]) == best) for curve in curves]
        lines.append(f"{label} [{unit}] & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule\end{tabular}", r"\par\vspace{5pt}",
              r"\textbf{(b) MIRA per-motion tracking}\\[2pt]", r"\scriptsize",
              r"\setlength{\tabcolsep}{3.5pt}",
              r"\resizebox{\textwidth}{!}{\begin{tabular}{lccccccc}\toprule",
              r"Motion & Root pos. & Root rot. & Body pos. & Body rot. & DoF vel. & Root vel. & Root ang. vel. \\",
              r"\midrule"]
    for motion, name in zip(MOTIONS, MOTION_NAMES):
        run = runs["MIRA"][motion]
        values = []
        for metric, _, _ in METRICS:
            curve = None if run is None else run_curves(run, metric)
            values.append(cell(curve))
        label = name + (r"$^\dagger$" if motion == "climb" else "")
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule\end{tabular}}", r"\end{table*}"]
    (GENERATED / "mira_main_table.tex").write_text("\n".join(lines) + "\n")


def write_ablation_table(ablations: dict) -> None:
    lines = [
        r"\begin{table*}[!t]",
        r"\caption{Backflip component analysis at 2000 updates. Completion is",
        r"reported as a check or cross from the measured seed-0 reference-start test.",
        r"Errors of incomplete motions cover only the executed portion.",
    ]
    lines += [r"}", r"\label{tab:ablation}", r"\centering\scriptsize",
              r"\setlength{\tabcolsep}{3pt}",
              r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccc}\toprule",
              r"Configuration & Root pos. & Root rot. & Body pos. & Body rot. & DoF vel. & Root vel. & Root ang. vel. & Complete \\",
              r"\midrule"]
    for name, _ in VARIANTS:
        run = ablations[name]
        values = [cell(run_curves(run, metric),
                       run["final"][metric] == min(r["final"][metric] for r in ablations.values()))
                  for metric, _, _ in METRICS]
        complete = r"$\checkmark$" if run["test_episode_length"] >= 60 - 1e-3 else r"$\times$"
        lines.append((r"\textbf{MIRA}" if name == "MIRA" else name) + " & " + " & ".join(values) + " & " + complete + r" \\")
    lines += [r"\bottomrule\end{tabular}}", r"\end{table*}"]
    (GENERATED / "mira_ablation_table.tex").write_text("\n".join(lines) + "\n")


def export_data(runs: dict, ablations: dict) -> None:
    measured = {"default_training_updates": UPDATES, "motion_names": list(MOTION_NAMES),
                "default_training_samples": MAX_SAMPLES, "motion_ids": list(MOTIONS),
                "terminal_budget_exceptions": {"MIRA/climb": {"updates": 3000, "samples": 3000 * SAMPLES_PER_UPDATE}},
                "budget_note": "MIRA Climb uses 3000 updates; other results use 2000.",
                "metric_names": [metric for metric, _, _ in METRICS],
                "nominal_horizon_steps": HORIZON_STEPS,
                "actual_seed_ids": [0], "synthetic_data_used": False,
                "evaluation_source": "in-training reference-start evaluation",
                "independent_evaluation_used": False,
                "aggregation": "Equal-weight arithmetic mean of the same six motions; one measured seed per configuration.",
                "learning_curve_alignment": "Fraction of each run's actual terminal training budget.",
                "learning_curve_motions": [motion for motion in MOTIONS
                                           if all(runs[method][motion] is not None for method in METHODS)],
                "runs": {}, "ablations": {}}
    for method in METHODS:
        measured["runs"][method] = {}
        for motion in MOTIONS:
            run = runs[method][motion]
            measured["runs"][method][motion] = "TODO" if run is None else {
                k: v for k, v in run.items() if k not in {"rows", "pressure"}}
    for name, run in ablations.items():
        measured["ablations"][name] = {k: v for k, v in run.items() if k not in {"rows", "pressure"}}
    (GENERATED / "mira_measured_seed0.json").write_text(json.dumps(measured, indent=2, allow_nan=False) + "\n")
    means = {
        method: {metric: (None if (curve := aggregate(list(runs[method].values()), metric)) is None
                          else float(curve[0, -1])) for metric, _, _ in METRICS}
        for method in METHODS
    }
    audit = {
        "source": "mira_measured_seed0.json",
        "budget_note": measured["budget_note"],
        "default_training_updates": UPDATES,
        "terminal_budget_exceptions": measured["terminal_budget_exceptions"],
        "macro_average": means,
        "full_horizon_motion_count": {
            method: sum(run is not None and run["complete"] for run in method_runs.values())
            for method, method_runs in runs.items()},
        "per_motion": {
            motion: {
                "nominal_horizon_steps": HORIZON_STEPS[motion],
                "methods": {method: measured["runs"][method][motion] for method in METHODS},
            } for motion in MOTIONS},
        "ablation_final": measured["ablations"],
        "limitations": [
            "One measured seed per configuration; no measured seed uncertainty or significance claim.",
            "Metrics use logged in-training reference-start tests, not new independent evaluation results.",
            "Existing baseline eval/final/summary.json files use a separate protocol and are excluded.",
            "Early-terminated episode errors cover only executed timesteps; full-horizon completion is a survival check, not a task-specific success classifier.",
            "MIRA Climb uses the measured 3000-update record; all other terminal results use 2000 updates. These are not equal-budget terminal comparisons.",
            "Learning curves align each run by its own training-budget fraction; this is not equal-sample efficiency.",
            "Removing monotone modeling jointly changes representation, normalization, and weight signs; its effects cannot be attributed to just one change.",
            "Sampled score-gradient norms in different input representations do not establish comparable physical-space sensitivity.",
            "No synthetic seed placeholders enter active tables, curves, or aggregates.",
            "Additional independent training seeds and a unified independent checkpoint evaluation remain TODO.",
        ],
    }
    for name, run in ablations.items():
        audit["ablation_final"][name]["diagnostic_final"] = {
            "sampled_max_score_gradient_L2": run["pressure"][-1]["maximum_score_gradient_L2"],
            "sampled_numerically_zero_reward_fraction": run["pressure"][-1]["sampled_reward_exact_zero_fraction"],
            "negative_logit": run["rows"][-1]["Disc_Neg_Logit"],
        }
    (GENERATED / "mira_results_audit.json").write_text(json.dumps(audit, indent=2, allow_nan=False) + "\n")
    report = [
        "# MIRA measured seed-0 result audit",
        "",
        "MIRA Climb uses 3000 PPO updates (786,432,000 environment samples); all other results "
        "use 2000 updates (524,288,000 samples). Each run uses 8192 training environments and seed 0. "
        "The data are logged reference-start tests. "
        "They are not newly collected independent evaluations or multi-seed averages.",
        "",
        "| Metric | " + " | ".join(METHODS) + " |",
        "| --- | " + " | ".join("---:" for _ in METHODS) + " |",
    ]
    for metric, label, unit in METRICS:
        values = ["TODO" if means[method][metric] is None else f"{means[method][metric]:.6f}"
                  for method in METHODS]
        report.append(f"| {label} [{unit}] | " + " | ".join(values) + " |")
    report += ["", "## Per-motion measured values",
               "", "The four methods are reported using raw values with their actual budgets. Lower tracking errors are better; "
               "incomplete-trajectory errors cover only the executed portion.",
               "", "| Motion | Method | Updates | Samples | Root pos. | Root rot. | Body pos. | Body rot. | DoF vel. | Root vel. | Root ang. vel. | Complete |",
               "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |"]
    for motion, name in zip(MOTIONS, MOTION_NAMES):
        for method in METHODS:
            run = runs[method][motion]
            values = ["TODO" if run is None else f"{run['final'][metric]:.6f}"
                      for metric, _, _ in METRICS]
            complete = "TODO" if run is None else "✓" if run["complete"] else "✗"
            budget = "TODO | TODO" if run is None else f'{run["completed_updates_used"]} | {run["samples_used"]}'
            report.append("| " + name + " | " + method + " | " + budget + " | " + " | ".join(values)
                          + " | " + complete + " |")
    report += ["", "## Logged test lengths",
               "", "A completion mark in the paper only checks whether the mean logged test length "
               "reaches the nominal motion horizon (tolerance 0.001 step). It is not a task-specific success rate.",
               "", "| Motion | Horizon | AMP | DeepMimic | ADD | MIRA |",
               "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for motion, name in zip(MOTIONS, MOTION_NAMES):
        lengths = ["TODO" if runs[method][motion] is None else f'{runs[method][motion]["test_episode_length"]:.3f}'
                   for method in METHODS]
        report.append("| " + name + f" | {HORIZON_STEPS[motion]} | " + " | ".join(lengths) + " |")
    report += ["", "## Backflip ablations",
               "", "| Configuration | Body pos. | Body rot. | DoF vel. | Test length | Sampled gradient max | Numerically zero reward |",
               "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for name, run in ablations.items():
        diagnostic = audit["ablation_final"][name]["diagnostic_final"]
        report.append(f'| {name} | {run["final"]["Body_Pos_Err"]:.5f} | {run["final"]["Body_Rot_Err"]:.5f} | '
                      f'{run["final"]["Dof_Vel_Err"]:.5f} | {run["test_episode_length"]:.3f} | '
                      f'{diagnostic["sampled_max_score_gradient_L2"]:.3f} | '
                      f'{100 * diagnostic["sampled_numerically_zero_reward_fraction"]:.2f}% |')
    report += ["", "## Interpretation and limitations", ""]
    report += ["- " + limitation for limitation in audit["limitations"]]
    report += ["", "Source paths and SHA-256 hashes are stored in mira_measured_seed0.json. "
               "All per-motion raw values are available in mira_per_motion_seed0.csv.", ""]
    (GENERATED / "mira_results_audit.md").write_text("\n".join(report))
    curves = {
        "grid_samples": GRID.tolist(), "motion_ids": list(MOTIONS), "seed": 0,
        "grid_samples_note": "Legacy 2000-update sample schedule only; actual sample locations vary by run.",
        "grid_training_progress": GRID_PROGRESS.tolist(),
        "alignment": "Fraction of each run's actual terminal budget; not equal-sample efficiency.",
        "budget_note": measured["budget_note"],
        "per_run_grid_samples": {
            method: {motion: ("TODO" if run is None else (GRID_PROGRESS * run["samples_used"]).tolist())
                     for motion, run in method_runs.items()}
            for method, method_runs in runs.items()},
        "source": "in-training reference-start evaluation",
        "macro_mean": {
            method: {metric: (None if (curve := aggregate(list(runs[method].values()), metric)) is None
                              else curve[0].tolist()) for metric, _, _ in METRICS}
            for method in METHODS},
        "per_motion": {
            method: {motion: ("TODO" if run is None else {
                metric: measured_curve(run, metric).tolist() for metric, _, _ in METRICS})
                for motion, run in method_runs.items()}
            for method, method_runs in runs.items()},
    }
    (GENERATED / "mira_learning_curve_data.json").write_text(json.dumps(curves, indent=2, allow_nan=False) + "\n")
    with (GENERATED / "mira_per_motion_seed0.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["motion", "method", "seed", "ppo_updates", "samples",
                         *[metric for metric, _, _ in METRICS],
                         "test_episode_length", "nominal_horizon_steps", "complete", "source"])
        for motion in MOTIONS:
            for method in METHODS:
                run = runs[method][motion]
                if run is not None:
                    writer.writerow([motion, method, 0, run["completed_updates_used"], run["samples_used"],
                                     *[run["final"][metric] for metric, _, _ in METRICS],
                                     run["test_episode_length"], HORIZON_STEPS[motion],
                                     run["complete"], run["path"]])


def main() -> None:
    GENERATED.mkdir(parents=True, exist_ok=True)
    runs = {method: {motion: read_run(run_path(method, motion), motion,
                                    TERMINAL_UPDATES.get((method, motion), UPDATES))
                     for motion in MOTIONS} for method in METHODS}
    ablations = {name: runs["MIRA"]["backflip"] if directory is None else read_run(ABLATION_ROOT / "formal" / directory, "backflip")
                 for name, directory in VARIANTS}
    if any(run is None for run in ablations.values()):
        raise ValueError("Expected completed Backflip ablation runs are missing; do not fabricate them.")
    write_main_table(runs)
    write_ablation_table(ablations)
    export_data(runs, ablations)
    print(json.dumps({"main_coverage": {m: sum(run is not None for run in v.values()) for m, v in runs.items()},
                      "actual_seed_ids": [0], "synthetic_data_used": False,
                      "assets": "current LaTeX tables and measured data exports"}))


if __name__ == "__main__":
    main()
