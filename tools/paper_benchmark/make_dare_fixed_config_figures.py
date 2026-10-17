"""Render the evidence-aligned DARE draft from its two recorded climb runs."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BENCH = ROOT / "output/paper_benchmark"
RUNS = {
    "CPL-DARE baseline": "dare_10layer_cpl_resetfix_climb_3000_8192_seed0_headless_20261010",
    "Phase-balanced CPL-DARE": "dare_10layer_cpl_phasebalanced_climb_3000_8192_seed0_headless_20261012",
}
ERRORS = (
    "Root_Pos_Err", "Root_Rot_Err", "Body_Pos_Err", "Body_Rot_Err",
    "Dof_Vel_Err", "Root_Vel_Err", "Root_Ang_Vel_Err",
)
PANELS = (
    ("Test_Episode_Length_Start0", "(a) Full-motion execution", "Executed steps"),
    ("Root_Pos_Err", "(b) Root position", "Error [m]"),
    ("Dof_Vel_Err", "(c) Joint velocity", "Error [rad/s]"),
)


def load_run(dirname):
    path = BENCH / dirname / "train_metrics.jsonl"
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
    iterations = [row["Iteration"] for row in rows]
    samples = [row["Samples"] for row in rows]
    if not rows or iterations != sorted(set(iterations)):
        raise ValueError(f"Nonunique or unsorted iterations: {path}")
    if samples != sorted(set(samples)):
        raise ValueError(f"Nonunique or unsorted sample budgets: {path}")
    if iterations[-1] != 2999 or len(rows) != 31:
        raise ValueError(f"Unexpected run coverage: {path}")
    for row in rows:
        for key in (*ERRORS, "Test_Episode_Length_Start0"):
            if not np.isfinite(row[key]):
                raise ValueError(f"Nonfinite {key} at {row['Iteration']}: {path}")
    return rows, {
        "path": str(path.relative_to(ROOT)),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "records": len(rows),
        "last_iteration": iterations[-1],
        "training_seed": 0,
    }


def main():
    loaded = {label: load_run(dirname) for label, dirname in RUNS.items()}
    reference_samples = next(iter(loaded.values()))[0]
    for rows, _ in loaded.values():
        if [r["Samples"] for r in rows] != [r["Samples"] for r in reference_samples]:
            raise ValueError("Runs do not share recorded sample budgets")

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.titlesize": 9, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35))
    colors = ("#636363", "#1764AB")
    for axis, (key, title, ylabel) in zip(axes, PANELS):
        for (label, (rows, _)), color in zip(loaded.items(), colors):
            axis.plot([r["Samples"] / 1e8 for r in rows],
                      [r[key] for r in rows], color=color, linewidth=1.2,
                      marker="o", markersize=2.0, label=label)
        axis.set(title=title, xlabel=r"Transitions [$10^8$]", ylabel=ylabel,
                 xlim=(0, 8.0))
        axis.grid(alpha=0.22, linewidth=0.5)
    axes[0].axhline(299, color="#777777", linewidth=0.7, linestyle=":")
    axes[0].set_ylim(0, 315)
    axes[1].set_ylim(bottom=0)
    axes[2].set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, fontsize=7, ncol=2,
               loc="upper center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.5, w_pad=1.3)

    figures = ROOT / "PAPER/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures / "dare_fixed_config_climb.pdf")
    fig.savefig(figures / "dare_fixed_config_climb.png", dpi=240)
    plt.close(fig)

    labels = list(loaded)
    baseline, balanced = (loaded[label][0][-1] for label in labels)
    report = {
        "protocol": "Two seed-0 historical runs; no multi-seed mean or error bars.",
        "sources": {label: metadata for label, (_, metadata) in loaded.items()},
        "first_observed_full_motion": {},
        "final_errors": {},
    }
    for label, (rows, _) in loaded.items():
        full = next(r for r in rows if r["Test_Episode_Length_Start0"] >= 298.99)
        report["first_observed_full_motion"][label] = {
            "iteration": full["Iteration"], "samples": full["Samples"],
        }
    for key in ERRORS:
        report["final_errors"][key] = {
            "baseline": baseline[key], "phasebalanced": balanced[key],
            "reduction_percent": 100 * (1 - balanced[key] / baseline[key]),
        }
    out = figures / "dare_fixed_config_climb_sources.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
