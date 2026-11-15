"""Build paper tables and figures without inventing unrun motion results.

Run from any directory. The optional synthetic mode supplies explicitly marked
layout deviations for two seed slots of measured runs; it never fills a missing
motion. Measured and synthetic data are exported separately for replacement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "PAPER"
BENCHMARK = ROOT / "output/paper_benchmark"
GENERATED = PAPER / "generated"
FIGURES = PAPER / "figures"
UPDATES = 2000
SAMPLES_PER_UPDATE = 8192 * 32
MAX_SAMPLES = UPDATES * SAMPLES_PER_UPDATE
MOTIONS = ("roll", "climb", "backflip", "crawl", "getup_facedown", "spinkick")
MOTION_NAMES = ("Roll", "Climb", "Backflip", "Crawl", "Get-Up", "Spin Kick")
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
COLORS = {"AMP": "#75808b", "DeepMimic": "#4875b0", "ADD": "#c78b36", "MIRA": "#128279"}
ABLATION_ROOT = BENCHMARK / "dare_ablation_backflip_matched_init_2000_8192_seed0_20261114"
VARIANTS = (
    ("MIRA", None, "#128279"),
    ("w/o Monotone Modeling", "no_a", "#4875b0"),
    ("w/o Diagonal Norm.", "no_b", "#c78b36"),
    ("w/o Both Components", "no_ab", "#9867a9"),
)


def load_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def run_path(method: str, motion: str) -> Path:
    if method == "MIRA":
        return BENCHMARK / f"dare_cmnn64_group_rms_{motion}_2000_8192_seed0_20261113"
    return BENCHMARK / f"{method.lower()}_{motion}_2k_8192_seed0"


def read_run(path: Path) -> dict | None:
    source = path / "train_metrics.jsonl"
    if not source.exists():
        return None
    rows = load_lines(source)
    # Use the exact common budget, including when a run was later extended.
    final = [r for r in rows if int(r.get("Samples", r.get("samples", -1))) == MAX_SAMPLES]
    if not final:
        return None
    deduplicated = {}
    for row in rows:
        samples = int(row.get("Samples", row.get("samples", -1)))
        if 0 < samples <= MAX_SAMPLES:
            deduplicated[samples] = row
    ordered = [deduplicated[s] for s in sorted(deduplicated)]
    pressure_file = path / "constraint_rounds.jsonl"
    pressure = []
    if pressure_file.exists():
        checks = {int(r["completed_updates"]): r for r in load_lines(pressure_file)
                  if int(r["completed_updates"]) <= UPDATES}
        pressure = [checks[k] for k in sorted(checks)]
    result = {
        "path": str(source.relative_to(ROOT)),
        "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "seed": 0,
        "completed_updates_used": UPDATES,
        "final": {key: float(final[-1][key]) for key, _, _ in METRICS},
        "test_episode_length": float(final[-1]["Test_Episode_Length"]),
        "rows": ordered,
        "pressure": pressure,
    }
    if pressure_file.exists():
        result["pressure_source"] = str(pressure_file.relative_to(ROOT))
        result["pressure_sha256"] = hashlib.sha256(pressure_file.read_bytes()).hexdigest()
    return result


def measured_curve(run: dict, metric: str) -> np.ndarray:
    rows = run["rows"]
    x = np.array([r.get("Samples", r.get("samples")) for r in rows])
    if GRID[0] < x[0] or GRID[-1] > x[-1]:
        raise ValueError(f"Incomplete curve support: {run['path']}")
    return np.interp(GRID, x, [r[metric] for r in rows])


def placeholder_curves(run: dict, metric: str, synthetic: bool) -> np.ndarray:
    center = measured_curve(run, metric)
    if not synthetic:
        return center[None, :]
    # Symmetric layout values preserve the measured seed-0 center exactly.
    # They are not independent training runs or estimated seed uncertainty.
    identifier = f"MIRA-layout-20261114:{run['path']}:{metric}"
    seed = int.from_bytes(hashlib.sha256(identifier.encode()).digest()[:8], "big")
    rng = np.random.default_rng(seed)
    noise = np.convolve(rng.normal(size=len(GRID) + 4), np.ones(5) / 5, mode="valid")
    noise = np.clip(noise * 0.045 + rng.uniform(0.025, 0.06), -0.12, 0.12)
    return np.stack((center, center * (1 + noise), center * (1 - noise)))


def aggregate(runs: list[dict | None], metric: str, synthetic: bool,
              expected_motions: int = 6) -> np.ndarray | None:
    # Main six-motion tables retain TODO; preliminary curves explicitly use
    # the same available three-motion cohort for every compared method.
    if len(runs) != expected_motions or not runs or any(run is None for run in runs):
        return None
    return np.mean([placeholder_curves(run, metric, synthetic) for run in runs], axis=0)


def cell(curves: np.ndarray | None, synthetic: bool) -> str:
    if curves is None:
        return r"\tbd"
    center = float(curves[0, -1])
    if not synthetic:
        return f"${center:.4f}$"
    dummy_sd = float(np.std(curves[:, -1], ddof=1))
    deviation_text = f"{dummy_sd:.4f}" if dummy_sd >= 5e-5 else f"{dummy_sd:.5f}"
    return (f"${center:.4f}\\!\\pm\\!"
            f"\\textcolor{{orange!70!black}}{{{deviation_text}}}^{{\\dagger}}$")


def write_main_table(runs: dict, synthetic: bool) -> None:
    lines = [
        r"% Generated by PAPER/tools/build_mira_experiments.py; do not hand-edit.",
        r"\begin{table*}[t]",
        r"\caption{Tracking errors at the common 2000-update budget. (a) Equal-weight",
        r"macro-averages over all six motions; incomplete aggregates remain TODO.",
        r"(b) Available MIRA motion results and pending entries. Lower is better.",
    ]
    if synthetic:
        lines += [r"\textcolor{orange!70!black}{$\dagger$: Random layout deviations, not measured",
                  r"seed variability; centers retain the measured seed-0 values.}"]
    lines += [
        r"Climb$^\ddagger$ uses the 2000-update checkpoint, before its extension;",
        r"its evaluation terminates early. Errors cover executed timesteps.}",
        r"\label{tab:main}", r"\centering\small",
        r"\setlength{\tabcolsep}{8pt}",
        r"\textbf{(a) Six-motion macro-average}\\[2pt]",
        r"\begin{tabular}{lcccc}\toprule",
        r"Metric $\downarrow$ & AMP & DeepMimic & ADD & MIRA \\", r"\midrule",
    ]
    for metric, label, unit in METRICS:
        values = [cell(aggregate([runs[m][motion] for motion in MOTIONS], metric, synthetic), synthetic)
                  for m in METHODS]
        lines.append(f"{label} [{unit}] & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule\end{tabular}", r"\par\vspace{5pt}",
              r"\textbf{(b) MIRA motion coverage}\\[2pt]", r"\scriptsize",
              r"\setlength{\tabcolsep}{3.5pt}",
              r"\resizebox{\textwidth}{!}{\begin{tabular}{lccccccc}\toprule",
              r"Motion & Root pos. & Root rot. & Body pos. & Body rot. & DoF vel. & Root vel. & Root ang. vel. \\",
              r"\midrule"]
    for motion, name in zip(MOTIONS, MOTION_NAMES):
        run = runs["MIRA"][motion]
        values = [cell(None if run is None else placeholder_curves(run, metric, synthetic), synthetic)
                  for metric, _, _ in METRICS]
        label = name + (r"$^\ddagger$" if motion == "climb" else "")
        lines.append(label + " & " + " & ".join(values) + r" \\")
    lines += [r"\bottomrule\end{tabular}}", r"\end{table*}"]
    (GENERATED / "mira_main_table.tex").write_text("\n".join(lines) + "\n")


def write_ablation_table(ablations: dict, synthetic: bool) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\caption{Backflip component analysis at 2000 updates. Completion is",
        r"reported as a check or cross from the measured seed-0 reference-start test.",
        r"Errors of incomplete motions cover only the executed portion.",
    ]
    if synthetic:
        lines += [r"\textcolor{orange!70!black}{$\dagger$: Synthetic layout deviations;",
                  r"additional seed repetitions remain pending.}"]
    lines += [r"}", r"\label{tab:ablation}", r"\centering\scriptsize",
              r"\setlength{\tabcolsep}{3pt}",
              r"\resizebox{\textwidth}{!}{\begin{tabular}{lcccccccc}\toprule",
              r"Configuration & Root pos. & Root rot. & Body pos. & Body rot. & DoF vel. & Root vel. & Root ang. vel. & Complete \\",
              r"\midrule"]
    for name, _, _ in VARIANTS:
        run = ablations[name]
        values = [cell(placeholder_curves(run, metric, synthetic), synthetic) for metric, _, _ in METRICS]
        complete = r"$\checkmark$" if run["test_episode_length"] >= 60 - 1e-3 else r"$\times$"
        lines.append((r"\textbf{MIRA}" if name == "MIRA" else name) + " & " + " & ".join(values) + " & " + complete + r" \\")
    lines += [r"\bottomrule\end{tabular}}", r"\end{table*}"]
    (GENERATED / "mira_ablation_table.tex").write_text("\n".join(lines) + "\n")


def plot_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 8.5,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.18,
        "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def save_figure(fig, name: str) -> None:
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_macro_curves(runs: dict, synthetic: bool) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    shared_motions = [motion for motion in MOTIONS
                      if all(runs[method][motion] is not None for method in METHODS)]
    selected = (("Body_Pos_Err", "Body position error (m)"),
                ("Body_Rot_Err", "Body rotation error (rad)"),
                ("Dof_Vel_Err", "Joint velocity error (rad/s)"))
    for ax, (metric, ylabel) in zip(axes, selected):
        for method in METHODS:
            curves = aggregate([runs[method][motion] for motion in shared_motions],
                               metric, synthetic, expected_motions=len(shared_motions))
            if curves is None:
                continue
            center = curves[0]
            ax.plot(GRID / 1e8, center, color=COLORS[method], lw=1.65)
            if synthetic:
                deviation = np.std(curves, axis=0, ddof=1)
                ax.fill_between(GRID / 1e8, center - deviation, center + deviation,
                                color=COLORS[method], alpha=0.17, linewidth=0)
        ax.set(xlabel=r"Environment samples ($\times 10^8$)", ylabel=ylabel, xlim=(0, 5.25))
        ax.set_yscale("log")
    handles = []
    for method in METHODS:
        handles.append(Line2D([], [], color=COLORS[method], lw=1.8,
                              label=method))
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.005), ncol=4,
               frameon=False, fontsize=7.4, columnspacing=1.1)
    names = [MOTION_NAMES[MOTIONS.index(m)] for m in shared_motions]
    fig.text(0.5, 0.095,
             f"Current {len(shared_motions)}-motion mean: " + ", ".join(names),
             ha="center", color="#555555", fontsize=7)
    fig.text(0.5, 0.015,
             "DRAFT: random seed bands; measured seed-0 centers. Remaining motion runs: TODO."
             if synthetic else "Measured seed 0. Every method uses the same available motion cohort.",
             ha="center", color="#9c5f0d" if synthetic else "#555555", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.31, top=0.81, wspace=0.44)
    save_figure(fig, "mira_learning_macro_draft")


def plot_ablation_diagnostics(ablations: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55))
    for name, _, color in VARIANTS:
        run = ablations[name]
        checks = run["pressure"]
        x = np.array([r["completed_updates"] for r in checks]) * SAMPLES_PER_UPDATE / 1e8
        axes[0].plot(x, [r["maximum_score_gradient_L2"] for r in checks], color=color, lw=1.6)
        axes[2].plot(x, [100 * r["sampled_reward_exact_zero_fraction"] for r in checks], color=color, lw=1.6)
        rows = run["rows"]
        axes[1].plot([r["Samples"] / 1e8 for r in rows], [r["Disc_Neg_Logit"] for r in rows],
                     color=color, lw=1.6)
    axes[0].axhline(1, color="#777777", ls="--", lw=0.8)
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"Sampled max. $\|\nabla_z s\|_2$")
    axes[1].set_ylabel("Mean negative-sample logit")
    axes[2].set_ylabel("Numerically zero rewards (%)")
    axes[2].set_ylim(-2, 102)
    for ax in axes:
        ax.set(xlabel=r"Environment samples ($\times 10^8$)", xlim=(0, 5.25))
    handles = [Line2D([], [], color=color, lw=1.8, label=name) for name, _, color in VARIANTS]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.01), ncol=2,
               frameon=False, fontsize=7.4, columnspacing=1.4)
    fig.text(0.5, 0.015, "Backflip diagnostics: measured seed 0 only; no synthetic observations.",
             ha="center", color="#555555", fontsize=7)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.24, top=0.76, wspace=0.46)
    save_figure(fig, "mira_ablation_diagnostics")


def plot_frame_placeholders() -> None:
    # Earlier DARE renders are deliberately not relabelled as MIRA executions.
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 1.85))
    for ax, name in zip(axes.flat, MOTION_NAMES):
        ax.set_axis_off()
        ax.add_patch(Rectangle((0.01, 0.02), 0.98, 0.95, transform=ax.transAxes,
                               facecolor="#f6f7f8", edgecolor="#b3bac2", ls="--", lw=0.8))
        ax.text(0.5, 0.70, name, ha="center", va="center", fontsize=9, weight="bold")
        ax.text(0.5, 0.38, "TODO: current MIRA rollout frames", ha="center", va="center",
                fontsize=7, color="#b13d37")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01, hspace=0.16, wspace=0.05)
    save_figure(fig, "mira_motion_frames_todo")


def export_data(runs: dict, ablations: dict, synthetic: bool) -> None:
    measured = {"common_updates": UPDATES, "motion_names": list(MOTION_NAMES),
                "actual_seed_ids": [0], "target_seed_ids": [0, 1, 2],
                "aggregation": "Tables require six motions. Preliminary learning curves use the common measured motion cohort for every method.",
                "learning_curve_motions": [motion for motion in MOTIONS
                                           if all(runs[method][motion] is not None for method in METHODS)],
                "runs": {}, "ablations": {}}
    placeholders = {"WARNING": "SYNTHETIC LAYOUT VALUES, NOT EXPERIMENTAL OBSERVATIONS",
                    "purpose": "Two explicitly marked seed-slot placeholders for measured runs only.",
                    "synthetic_mode": synthetic, "grid_samples": GRID.tolist(), "runs": {}}
    for method in METHODS:
        measured["runs"][method] = {}
        for motion in MOTIONS:
            run = runs[method][motion]
            measured["runs"][method][motion] = "TODO" if run is None else {
                k: v for k, v in run.items() if k not in {"rows", "pressure"}}
    for name, run in ablations.items():
        measured["ablations"][name] = {k: v for k, v in run.items() if k not in {"rows", "pressure"}}
    if synthetic:
        unique = {run["path"]: run for method_runs in runs.values() for run in method_runs.values()
                  if run is not None}
        unique.update({run["path"]: run for run in ablations.values()})
        for path, run in unique.items():
            placeholders["runs"][path] = {
                metric: {"synthetic_slot_1": curves[1].tolist(), "synthetic_slot_2": curves[2].tolist()}
                for metric, _, _ in METRICS
                for curves in [placeholder_curves(run, metric, True)]}
    (GENERATED / "mira_measured_seed0.json").write_text(json.dumps(measured, indent=2, allow_nan=False) + "\n")
    (GENERATED / "mira_synthetic_seed_placeholders.json").write_text(json.dumps(placeholders, indent=2, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-seed-placeholders", action="store_true",
                        help="Add visibly marked random layout deviations, never missing-motion data.")
    args = parser.parse_args()
    GENERATED.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    runs = {method: {motion: read_run(run_path(method, motion)) for motion in MOTIONS} for method in METHODS}
    ablations = {name: runs["MIRA"]["backflip"] if directory is None else read_run(ABLATION_ROOT / "formal" / directory)
                 for name, directory, _ in VARIANTS}
    if any(run is None for run in ablations.values()):
        raise ValueError("Expected completed Backflip ablation runs are missing; do not fabricate them.")
    write_main_table(runs, args.synthetic_seed_placeholders)
    write_ablation_table(ablations, args.synthetic_seed_placeholders)
    plot_style()
    plot_macro_curves(runs, args.synthetic_seed_placeholders)
    plot_ablation_diagnostics(ablations)
    plot_frame_placeholders()
    export_data(runs, ablations, args.synthetic_seed_placeholders)
    print(json.dumps({"main_coverage": {m: sum(run is not None for run in v.values()) for m, v in runs.items()},
                      "synthetic_layout_mode": args.synthetic_seed_placeholders,
                      "figures": ["mira_learning_macro_draft", "mira_ablation_diagnostics", "mira_motion_frames_todo"]}))


if __name__ == "__main__":
    main()
