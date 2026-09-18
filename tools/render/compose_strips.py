#!/usr/bin/env python3
"""Compose per-motion frame folders into a stacked strip figure.

Each row is one motion, laid out left to right like
`PAPER/figures/dare_climb_10.png`; rows are stacked top to bottom.

Example
-------
    python tools/render/compose_strips.py \
        --row "Climb:output/render/strips/climb_slope" \
        --row "Carry Box:output/render/strips/carry_box" \
        --row "Sit on Sofa:output/render/strips/sit_sofa" \
        --row "Backflip:output/render/strips/backflip" \
        --out output/render/strips/g1_skills.png
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--row", action="append", required=True,
                   metavar="LABEL:DIR", help="one motion; repeatable, in row order")
    p.add_argument("--out", required=True)
    p.add_argument("--cell-in", type=float, default=1.10,
                   help="width of one frame cell, in inches")
    p.add_argument("--gap", type=float, default=0.02,
                   help="gap between cells, as a fraction of the cell width")
    p.add_argument("--label-in", type=float, default=1.05,
                   help="width reserved for the row label, in inches")
    p.add_argument("--no-labels", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def load_row(directory: str):
    import imageio.v2 as iio

    files = sorted(glob.glob(os.path.join(directory, "frame_*.png")))
    if not files:
        raise SystemExit(f"no frame_*.png in {directory}")
    return [iio.imread(f)[:, :, :3] for f in files]


def main() -> None:
    args = parse_args()
    rows = []
    for spec in args.row:
        label, _, directory = spec.partition(":")
        rows.append((label, load_row(directory)))

    n_rows = len(rows)
    n_cols = max(len(f) for _l, f in rows)
    img_h, img_w = rows[0][1][0].shape[:2]
    aspect = img_w / img_h

    cell_w = args.cell_in
    cell_h = cell_w / aspect
    gap = args.gap * cell_w
    label_w = 0.0 if args.no_labels else args.label_in

    fig_w = label_w + n_cols * cell_w + (n_cols - 1) * gap
    fig_h = n_rows * cell_h + (n_rows - 1) * gap

    fig = plt.figure(figsize=(fig_w, fig_h))

    for r, (label, frames) in enumerate(rows):
        # rows are laid out top to bottom
        y = (n_rows - 1 - r) * (cell_h + gap) / fig_h
        for c, img in enumerate(frames):
            x = (label_w + c * (cell_w + gap)) / fig_w
            ax = fig.add_axes([x, y, cell_w / fig_w, cell_h / fig_h])
            ax.imshow(img)
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        if not args.no_labels:
            fig.text(label_w * 0.5 / fig_w, y + 0.5 * cell_h / fig_h, label,
                     ha="center", va="center", fontsize=12,
                     color="#1f2a37", fontweight="bold")

    out_dir = os.path.dirname(os.path.abspath(args.out))
    os.makedirs(out_dir or ".", exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, facecolor="white")
    stem, _ = os.path.splitext(args.out)
    fig.savefig(stem + ".pdf", facecolor="white")
    print(f"wrote {args.out} ({n_rows} rows x {n_cols} frames, "
          f"{fig_w:.1f}x{fig_h:.1f} in)")
    print(f"wrote {stem}.pdf")


if __name__ == "__main__":
    main()
