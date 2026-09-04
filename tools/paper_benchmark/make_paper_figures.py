"""Build the paper's snapshot strips and learning-curve figure."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
FIG_DIR = ROOT / "PAPER" / "figures"
BENCH_DIR = ROOT / "output" / "paper_benchmark"


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as stream:
        for line in stream:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def plot_learning_curves(out_pdf: Path, out_png: Path) -> None:
    methods = {
        "AMP": "amp",
        "DeepMimic": "deepmimic",
        "ADD": "add",
        "DARE": "dare",
    }
    colors = {
        "AMP": "#777777",
        "DeepMimic": "#4c78a8",
        "ADD": "#f28e2b",
        "DARE": "#d62728",
    }
    styles = {"AMP": "--", "DeepMimic": "-.", "ADD": ":", "DARE": "-"}
    panels = [
        ("roll", "Body_Pos_Err", "Roll - Body Position", r"$E_{\mathrm{body\text{-}pos}}$"),
        ("roll", "Root_Rot_Err", "Roll - Root Rotation", r"$E_{\mathrm{root\text{-}rot}}$"),
        ("climb", "Body_Pos_Err", "Climb - Body Position", r"$E_{\mathrm{body\text{-}pos}}$"),
        ("climb", "Root_Rot_Err", "Climb - Root Rotation", r"$E_{\mathrm{root\text{-}rot}}$"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.1, 4.4), sharex=True)
    for ax, (motion, metric, title, ylabel) in zip(axes.flat, panels):
        for label, method in methods.items():
            path = BENCH_DIR / f"{method}_{motion}_2k_8192_seed0" / "train_metrics.jsonl"
            if not path.exists():
                continue
            rows = load_jsonl(path)
            xs = np.asarray([row["Samples"] for row in rows], dtype=float) / 1e8
            ys = np.asarray([row[metric] for row in rows], dtype=float)
            order = np.argsort(xs)
            ax.plot(xs[order], ys[order], label=label, color=colors[label],
                    linestyle=styles[label], linewidth=1.8)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.7)
        ax.tick_params(labelsize=7)
        ax.set_xlim(left=0)
    axes[1, 0].set_xlabel("Environment samples ($\\times 10^8$)", fontsize=8)
    axes[1, 1].set_xlabel("Environment samples ($\\times 10^8$)", fontsize=8)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.94), pad=0.8)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _blue_bbox(image: Image.Image) -> tuple[int, int, int, int] | None:
    arr = np.asarray(image.convert("RGB"))
    red, green, blue = [arr[..., i].astype(np.int16) for i in range(3)]
    mask = (blue > 85) & (blue > red + 25) & (blue > green + 5)
    ys, xs = np.where(mask)
    if len(xs) < 10:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def make_snapshot_strip(src: Path, dst: Path, frames: int) -> None:
    source = Image.open(src).convert("RGB")
    tile_width = source.width // frames
    tiles = []
    for index in range(frames):
        tile = source.crop((index * tile_width, 0, (index + 1) * tile_width, source.height))
        bbox = _blue_bbox(tile)
        if bbox is None:
            bbox = (0, 0, tile.width, tile.height)
        x0, y0, x1, y1 = bbox
        # Match the output aspect ratio while keeping the humanoid close to
        # the frame. This makes the simulated character the visual subject.
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        box_w = max(8.0, (x1 - x0) * 1.12)
        box_h = max(8.0, (y1 - y0) * 1.12)
        target_ratio = 170.0 / 145.0
        if box_w / box_h < target_ratio:
            box_w = box_h * target_ratio
        else:
            box_h = box_w / target_ratio
        x0 = max(0, int(round(cx - box_w / 2)))
        y0 = max(0, int(round(cy - box_h / 2)))
        x1 = min(tile.width, int(round(cx + box_w / 2)))
        y1 = min(tile.height, int(round(cy + box_h / 2)))
        crop = tile.crop((x0, y0, x1, y1)).resize((160, 135), Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", (170, 145), (231, 239, 246))
        canvas.paste(crop, (5, 5))
        tiles.append(canvas)
    result = Image.new("RGB", (170 * frames, 145), (231, 239, 246))
    for index, tile in enumerate(tiles):
        result.paste(tile, (170 * index, 0))
    result.save(dst, optimize=True)


def build_snapshot_strips() -> None:
    strips = {
        "ph_climb_10.png": ("dare_climb_10.png", 10),
        "ph_getup_10.png": ("dare_getup_10.png", 10),
        "ph_roll_5.png": ("dare_roll_5.png", 5),
        "ph_backflip_5.png": ("dare_backflip_5.png", 5),
        "ph_crawl_5.png": ("dare_crawl_5.png", 5),
        "ph_spinkick_5.png": ("dare_spinkick_5.png", 5),
    }
    for source_name, (target_name, frames) in strips.items():
        make_snapshot_strip(FIG_DIR / source_name, FIG_DIR / target_name, frames)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fig-dir", type=Path, default=FIG_DIR)
    args = parser.parse_args()
    args.fig_dir.mkdir(parents=True, exist_ok=True)
    plot_learning_curves(args.fig_dir / "fig_learning_curves.pdf",
                         args.fig_dir / "fig_learning_curves.png")
    build_snapshot_strips()


if __name__ == "__main__":
    main()
