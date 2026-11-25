#!/usr/bin/env python3
"""Figure 1: crossing-fibre graph construction and hardware-aware QAOA workflow.

Panels a-c are built from the released data for HCP subject 103414
(data/hcp/representative_103414/):
  anatomy_coronal_patch_localization.png  coronal T1 with TractSeg right CST (orange),
                                          JHU right superior corona radiata (blue) and
                                          the predefined 5x1x5 patch (red); radiological
                                          display, subject right on the left
  mrview_fod_glyphs_coronal_2048.png      MRtrix mrview rendering of the 5x5 FOD patch
  real_graph_*                            the 50-node Ising graph (drawn as in Figure 3)
Panels d-f are the vector schematic from make_figure1_schematic.py.

Usage: python src/figures/make_figure1.py --out figures/main
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle

import make_figure1_schematic as sch
import make_figure3_representative_hcp as f3
import make_main_figures as mf

MM = mf.MM
DATA = mf.ROOT / "data/hcp/representative_103414"
W, H = 183.0, 119.0           # figure size in mm
IMG = 52.0                    # side of the square image panels in mm
COLS = (0.0, 61.0, 122.0)     # left edge of the three top-row columns in mm
IMG_Y = 61.5                  # bottom of the top-row images in mm
BOTTOM_H = W * 62 / 224       # height of the d-f block (same scale as the schematic)

# Overlay colours as rendered in the anatomy image.
CST, SCR, PATCH = "#FFB400", "#00A6EC", "#F6511D"


def image_axes(fig, col):
    x0 = COLS[col] + (61.0 - IMG) / 2
    return fig.add_axes([x0 / W, IMG_Y / H, IMG / W, IMG / H])


def header(ov, col, letter, text):
    x = COLS[col] + 0.8
    ov.text(x, 115.1, letter, fontsize=8, fontweight="bold", va="bottom", ha="left")
    ov.text(x + 3.7, 115.3, text, fontsize=7, va="bottom", ha="left")


def caption(ov, col, text, y=59.1):
    ov.text(COLS[col] + 30.5, y, text, fontsize=5.6, ha="center", va="top")


def inline_legend(ov, col, items, y=54.1):
    """items: (kind, colour, label) with kind 'patch', 'line' or 'star'; laid out on one row."""
    widths = [4.2 + 1.15 * len(lab) for _, _, lab in items]
    x = COLS[col] + 30.5 - sum(widths) / 2
    for (kind, colour, lab), w in zip(items, widths):
        if kind == "patch":
            ov.add_patch(Rectangle((x, y - 0.9), 2.6, 1.8, fc=colour, ec="none"))
        elif kind == "line":
            ov.plot([x, x + 2.8], [y, y], color=colour, lw=1.4, solid_capstyle="butt")
        else:
            ov.plot(x + 1.3, y, marker="*", ms=4.2, color=colour, mec="none")
        ov.text(x + 3.5, y, lab, fontsize=5.6, va="center", ha="left")
        x += w


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=mf.ROOT / "figures/main")
    a = ap.parse_args()
    mf.set_style()

    fig = plt.figure(figsize=(W * MM, H * MM))
    ov = fig.add_axes([0, 0, 1, 1], zorder=5)
    ov.set_xlim(0, W)
    ov.set_ylim(0, H)
    ov.axis("off")
    ov.patch.set_alpha(0)

    # a anatomy
    ax = image_axes(fig, 0)
    ax.imshow(plt.imread(DATA / "anatomy_coronal_patch_localization.png"), interpolation="lanczos")
    ax.axis("off")
    for txt, xpos, ha in (("R", 0.03, "left"), ("L", 0.97, "right")):
        ax.text(xpos, 0.97, txt, transform=ax.transAxes, color="white", fontsize=6.5,
                fontweight="bold", ha=ha, va="top")
    header(ov, 0, "a", "Anatomical localization")
    caption(ov, 0, "HCP 103414, coronal slice, radiological view")
    inline_legend(ov, 0, [("patch", CST, "Right CST"), ("patch", SCR, "Right SCR"),
                          ("patch", PATCH, "5 × 1 × 5 patch")])

    # b FOD glyphs
    ax = image_axes(fig, 1)
    ax.imshow(plt.imread(DATA / "mrview_fod_glyphs_coronal_2048.png"), interpolation="lanczos")
    ax.axis("off")
    header(ov, 1, "b", "Local diffusion orientations")
    caption(ov, 1, "5 × 5 voxels; two retained CSD peak axes per voxel")

    # c graph, drawn exactly as in Figure 3c
    nodes, edges, fields, *_ = f3.load()
    _, _, dx, dy, pos = f3.geometry(nodes)
    ax = image_axes(fig, 2)
    f3.draw_graph(ax, edges, pos, fields, dx=dx, dy=dy)
    f3.frame(ax)
    header(ov, 2, "c", "Sparse Ising graph")
    n_cont = int((edges.edge_type == "continuation").sum())
    n_conf = int((edges.edge_type == "conflict").sum())
    caption(ov, 2, f"{len(nodes)} nodes · {n_cont} continuation · {n_conf} conflict edges")
    inline_legend(ov, 2, [("line", f3.CONT, "Continuation"), ("line", f3.CONFLICT, "Conflict"),
                          ("star", mf.INK, "Boundary field")])

    # d-f schematic
    bx = fig.add_axes([0, 0, 1, BOTTOM_H / H])
    bx.set_xlim(0, 224)
    bx.set_ylim(-4, 58)
    bx.axis("off")
    sch.panel_d(bx, 1, 0)
    sch.panel_e(bx, 62, 0)
    sch.panel_f(bx, 153, 0)

    mf.save(fig, a.out, "figure1_crossing_fibre_qaoa_overview")


if __name__ == "__main__":
    main()
