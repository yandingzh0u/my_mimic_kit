#!/usr/bin/env python3
"""Compose inspected simulator frame strips into a two-row paper teaser."""

import argparse
import os

import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--frames", default="paper/icra2027_cpmd/figures/roll_frames")
    p.add_argument("--out", default="paper/icra2027_cpmd/figures/roll_teaser")
    return p.parse_args()


def frame_files(folder, prefix):
    return sorted(os.path.join(folder, f) for f in os.listdir(folder)
                  if f.startswith(prefix + "_") and "_step" in f and f.endswith(".png"))


def main():
    args = parse_args()
    mpl.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})
    rows = [("instantaneous", "State differential"),
            ("cpmd", "CPMD (ours)")]
    files = [frame_files(args.frames, key) for key, _ in rows]
    if any(len(x) != 6 for x in files):
        raise RuntimeError("Expected six inspected frames for each row")

    fig, axes = plt.subplots(2, 6, figsize=(7.1, 2.35))
    for r, ((_, label), row_files) in enumerate(zip(rows, files)):
        for c, path in enumerate(row_files):
            axes[r, c].imshow(Image.open(path))
            axes[r, c].axis("off")
            if r == 0:
                axes[r, c].set_title(f"{0.4*c:.1f} s", fontsize=7, pad=1)
        axes[r, 0].text(0.03, 0.93, label, transform=axes[r, 0].transAxes,
                        ha="left", va="top", fontsize=7, fontweight="bold",
                        color="white",
                        bbox={"facecolor": "black", "alpha": 0.68,
                              "pad": 1.5, "edgecolor": "none"})
    fig.subplots_adjust(left=0.002, right=0.998, top=0.88, bottom=0.02,
                        wspace=0.02, hspace=0.04)
    fig.savefig(args.out + ".pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(args.out + ".png", dpi=250, bbox_inches="tight", pad_inches=0.01)
    print("wrote", args.out + ".{pdf,png}")


if __name__ == "__main__":
    main()
