#!/usr/bin/env python3
"""Generate paper-ready plots from the completed Roll experiments.

This script is deliberately read-only with respect to experiment outputs.  It
uses only the stored text logs and behavior NPZ files; it does not import the
simulator or load a policy checkpoint.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np


REPO = Path(__file__).resolve().parents[3]
FIG_DIR = REPO / "paper" / "icra2027_cpmd" / "figures"
DATA_DIR = REPO / "paper" / "icra2027_cpmd" / "data"

RUNS = {
    "First-order": REPO / "output" / "liesig_l1_roll_cycle_1k_seed0",
    "CPMD": REPO / "output" / "liesig_placebo_roll_cycle_1k_seed0",
}

# This baseline was intentionally deleted from output/ but remains recoverable.
# Prefer a restored copy if one is added later; the fallback keeps the current
# evidence reproducible without mutating the trash or the experiment folders.
ADD_NPZ_CANDIDATES = (
    REPO / "output" / "add_roll_contact_et_1k_seed0" / "roll_behavior.npz",
    Path("/home/y/.local/share/Trash/files/add_roll_contact_et_1k_seed0/roll_behavior.npz"),
)

COLORS = {
    "ADD": "#9A9A9A",
    "First-order": "#E69F00",
    "CPMD": "#0072B2",
}

DISPLAY = {
    "ADD": "ADD",
    "First-order": "First-order",
    "CPMD": "CPMD",
}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.0,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.2,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.5,
            "lines.markersize": 3.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def find_add_npz() -> Path | None:
    return next((path for path in ADD_NPZ_CANDIDATES if path.exists()), None)


def parse_log(path: Path) -> dict[str, np.ndarray]:
    lines = path.read_text().splitlines()
    header = lines[0].split()
    rows: list[list[str]] = []
    for line in lines[1:]:
        fields = line.split()
        if fields and fields[0].isdigit() and len(fields) == len(header):
            rows.append(fields)
    if not rows:
        raise RuntimeError(f"No complete rows in {path}")
    columns = list(zip(*rows))
    return {
        name: np.asarray(values, dtype=np.float64)
        for name, values in zip(header, columns)
    }


def save_both(fig: mpl.figure.Figure, stem: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"{stem}.pdf")
    fig.savefig(FIG_DIR / f"{stem}.png")
    plt.close(fig)


def winding_distribution_figure(records: dict[str, dict[str, np.ndarray]]) -> None:
    names = list(records)
    positions = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(3.45, 2.20))
    values = [records[name]["winding_ratio"] for name in names]
    violins = ax.violinplot(
        values,
        positions=positions,
        widths=0.78,
        showmeans=False,
        showmedians=False,
        showextrema=False,
        bw_method=0.16,
    )
    for body, name in zip(violins["bodies"], names):
        body.set_facecolor(COLORS[name])
        body.set_edgecolor("none")
        body.set_alpha(0.75)
    q25 = [np.quantile(v, 0.25) for v in values]
    q75 = [np.quantile(v, 0.75) for v in values]
    med = [np.median(v) for v in values]
    ax.vlines(positions, q25, q75, color="black", lw=1.7, zorder=3)
    ax.scatter(positions, med, s=12, color="white", edgecolor="black", lw=0.6, zorder=4)
    ax.axhline(1.0, color="#444444", lw=0.8, ls=":", label="reference")
    ax.axhline(0.5, color="#C44E52", lw=0.8, ls="--", label="shortcut threshold")
    ax.set_ylabel("Directed winding ratio")
    ax.set_xticks(positions, [DISPLAY[name] for name in names])
    ax.set_ylim(-0.22, 1.20)
    ax.legend(frameon=False, loc="center right", handlelength=1.8)
    ax.text(0.02, 0.965, "single-seed pilot", transform=ax.transAxes, va="top", color="#555555")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.7)
    fig.subplots_adjust(bottom=0.22, left=0.17, right=0.98, top=0.98)
    save_both(fig, "winding_distribution")


def behavior_summary_figure(records: dict[str, dict[str, np.ndarray]]) -> None:
    names = list(records)
    positions = np.arange(len(names))
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.05, 2.18),
        gridspec_kw={"width_ratios": [0.82, 1.48]},
    )

    ax = axes[0]
    rates = [100.0 * np.mean(records[name]["winding_ratio"] < 0.5) for name in names]
    bars = ax.bar(positions, rates, width=0.68, color=[COLORS[name] for name in names])
    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width() / 2, rate + 2.0, f"{rate:.1f}%", ha="center", va="bottom")
    ax.set_ylabel("Shortcut rate (%)")
    ax.set_xticks(positions, [DISPLAY[name] for name in names], rotation=12, ha="right")
    ax.set_ylim(0, 116)
    ax.text(-0.17, 1.03, "(a)", transform=ax.transAxes, fontweight="bold")

    ax = axes[1]
    for name in names:
        rec = records[name]
        ax.scatter(
            rec["height_rmse"],
            rec["winding_ratio"],
            s=8,
            alpha=0.42,
            color=COLORS[name],
            edgecolors="none",
            label=name,
        )
    ax.axhline(0.5, color="#C44E52", lw=0.8, ls="--")
    ax.set_xlabel("Root-height RMSE (m)")
    ax.set_ylabel("Directed winding ratio")
    ax.set_xlim(left=0)
    ax.set_ylim(-0.22, 1.20)
    ax.legend(frameon=False, loc="upper right", markerscale=1.7)
    ax.text(-0.13, 1.03, "(b)", transform=ax.transAxes, fontweight="bold")
    ax.text(0.02, 0.965, "single-seed pilot", transform=ax.transAxes, va="top", color="#555555")

    for ax in axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#DDDDDD", lw=0.45, alpha=0.7)
    fig.subplots_adjust(wspace=0.34, bottom=0.24, left=0.09, right=0.99, top=0.97)
    save_both(fig, "behavior_summary")


def training_figure() -> None:
    logs = {name: parse_log(run / "log.txt") for name, run in RUNS.items()}
    metrics = (
        ("Body_Pos_Err", "Body position error"),
        ("Root_Rot_Err", "Root rotation error"),
        ("Root_Ang_Vel_Err", "Root angular-velocity error"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(7.05, 2.03))
    for panel, (ax, (metric, ylabel)) in enumerate(zip(axes, metrics)):
        for name, log in logs.items():
            ax.plot(
                log["Samples"] / 1e6,
                log[metric],
                marker="o",
                color=COLORS[name],
                label=name,
            )
        ax.set_xlabel("Environment samples (million)")
        ax.set_ylabel(ylabel)
        ax.grid(color="#DDDDDD", lw=0.45, alpha=0.7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(-0.20, 1.04, f"({chr(ord('a') + panel)})", transform=ax.transAxes, fontweight="bold")
    axes[0].legend(frameon=False, loc="upper right")
    fig.subplots_adjust(wspace=0.42)
    fig.text(0.995, 0.995, "single-seed pilot", ha="right", va="top", color="#555555")
    save_both(fig, "training_curves")


def method_overview_figure() -> None:
    """Conceptual overview; formulas are typeset by Matplotlib for portability."""
    fig, ax = plt.subplots(figsize=(7.05, 2.72))
    ax.set_xlim(0, 14.0)
    ax.set_ylim(0, 6.0)
    ax.axis("off")

    def box(
        x: float,
        y: float,
        w: float,
        h: float,
        text: str,
        *,
        color: str,
        lw: float = 0.9,
        fontsize: float = 7.2,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=color,
            edgecolor="#333333",
            linewidth=lw,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize)

    def arrow(x0: float, y0: float, x1: float, y1: float, *, color: str = "#444444") -> None:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.9,
                color=color,
            )
        )

    ax.text(0.0, 5.55, "Difference-first representation", fontsize=9.0, fontweight="bold", va="center")
    box(0.10, 4.05, 1.62, 0.72, "reference history", color="#E8F1F8")
    box(0.10, 2.62, 1.62, 0.72, "policy history", color="#FBEADB")
    arrow(1.72, 4.41, 2.55, 3.88)
    arrow(1.72, 2.98, 2.55, 3.51)
    box(2.55, 3.22, 1.48, 0.92, "subtract\n$\\Delta m_t$", color="#F2F2F2")
    arrow(4.03, 3.68, 4.77, 3.68)
    box(4.77, 3.22, 1.40, 0.92, "reward model", color="#F2F2F2")
    ax.text(3.10, 2.43, "shared motion context is removed", ha="center", color="#B33A3A", fontsize=7.3)

    ax.plot([6.48, 6.48], [0.42, 5.62], color="#C8C8C8", lw=0.8)

    ax.text(6.80, 5.55, "Context-preserving motion differential", fontsize=9.0, fontweight="bold", va="center")
    box(6.82, 4.10, 1.28, 0.70, "reference", color="#E8F1F8")
    box(6.82, 2.78, 1.28, 0.70, "policy", color="#FBEADB")
    arrow(8.10, 4.45, 8.54, 4.45)
    arrow(8.10, 3.13, 8.54, 3.13)
    box(8.54, 4.10, 1.30, 0.70, r"$m_t^{\mathrm{ref}}$", color="#E8F1F8")
    box(8.54, 2.78, 1.30, 0.70, r"$m_t^{\mathrm{sim}}$", color="#FBEADB")
    arrow(9.84, 4.45, 10.30, 4.45)
    arrow(9.84, 3.13, 10.30, 3.13)
    box(10.30, 4.10, 1.56, 0.70, r"$\Psi(m_t^{\mathrm{ref}})$", color="#D8EAF7")
    box(10.30, 2.78, 1.56, 0.70, r"$\Psi(m_t^{\mathrm{sim}})$", color="#FAE0C8")
    arrow(11.08, 4.10, 11.08, 2.25)
    arrow(11.08, 2.78, 11.08, 2.25)
    box(9.55, 1.42, 3.06, 0.82, r"$\Delta\Psi_t=\Psi(m_t^{ref})-\Psi(m_t^{sim})$", color="#DCEFE5", fontsize=6.9)
    arrow(12.61, 1.83, 12.90, 1.83)
    box(12.90, 1.42, 1.00, 0.82, "reward\nmodel", color="#DCEFE5", fontsize=6.8)

    ax.text(7.10, 0.58, r"$m_t=\rho m_{t-1}+\xi_t$", ha="center", color="#444444", fontsize=6.9)
    ax.text(9.82, 0.58, r"$\Psi(m)=[m,\;\operatorname{vech}_{i<j}(\frac{1}{2}mm^\top)]$", ha="center", color="#444444", fontsize=6.9)
    ax.text(12.74, 0.58, r"$\Delta\Psi_t=0$ for exact tracking", ha="center", color="#2D6A4F", fontsize=6.9)
    fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.02)
    save_both(fig, "method_overview")


def write_behavior_table(records: dict[str, dict[str, np.ndarray]], sources: dict[str, Path]) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    fields = (
        "method",
        "source",
        "episodes",
        "winding_mean",
        "winding_std",
        "winding_median",
        "disp_mean",
        "height_rmse_mean_m",
        "height_rmse_median_m",
        "upright_rate",
        "shortcut_count",
        "shortcut_rate",
        "full_horizon_count",
    )
    with (DATA_DIR / "roll_behavior_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name, rec in records.items():
            shortcut = rec["winding_ratio"] < 0.5
            writer.writerow(
                {
                    "method": name,
                    "source": str(sources[name]),
                    "episodes": len(rec["winding_ratio"]),
                    "winding_mean": f"{np.mean(rec['winding_ratio']):.8f}",
                    "winding_std": f"{np.std(rec['winding_ratio']):.8f}",
                    "winding_median": f"{np.median(rec['winding_ratio']):.8f}",
                    "disp_mean": f"{np.mean(rec['disp_ratio']):.8f}",
                    "height_rmse_mean_m": f"{np.mean(rec['height_rmse']):.8f}",
                    "height_rmse_median_m": f"{np.median(rec['height_rmse']):.8f}",
                    "upright_rate": f"{np.mean(rec['upright']):.8f}",
                    "shortcut_count": int(np.sum(shortcut)),
                    "shortcut_rate": f"{np.mean(shortcut):.8f}",
                    "full_horizon_count": int(np.sum(rec["ep_len"] == 300)),
                }
            )


def main() -> None:
    configure_style()
    sources = {
        name: run / "roll_behavior.npz"
        for name, run in RUNS.items()
    }
    add_npz = find_add_npz()
    if add_npz is not None:
        sources = {"ADD": add_npz, **sources}
    records = {name: load_npz(path) for name, path in sources.items()}
    winding_distribution_figure(records)
    behavior_summary_figure(records)
    training_figure()
    method_overview_figure()
    write_behavior_table(records, sources)


if __name__ == "__main__":
    main()
