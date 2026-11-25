#!/usr/bin/env python3
"""Vector schematic for Figure 1 panels d-f (QAOA circuit, calibration, evaluation).

The panel functions are used by make_figure1.py, which builds the complete figure.
Run on its own, this script writes only panels d-f. All numbers follow the Methods.

Usage: python src/figures/make_figure1_schematic.py --out figures/main
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

import make_main_figures as mf

MM = 1 / 25.4
INK, GREY, LINE = mf.INK, "#6E6E6E", "#9A9A9A"
FILL = "#F4F4F4"
UC, UB = "#DCEAF5", "#FBE3D6"


def box(ax, x, y, w, h, title, body="", fc=FILL, ec=LINE, lw=0.6, tsize=6.3, bsize=5.4, title_bold=True):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
                                fc=fc, ec=ec, lw=lw, zorder=2))
    ax.text(x + w / 2, y + h - 2.6, title, ha="center", va="top", fontsize=tsize,
            fontweight="bold" if title_bold else "normal", zorder=3)
    if body:
        ax.text(x + w / 2, y + h - 7.2, body, ha="center", va="top", fontsize=bsize,
                linespacing=1.35, zorder=3, color=INK)


def arrow(ax, p, q, rad=0.0):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=6, lw=0.7, color=GREY,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, zorder=1))


def label(ax, x, y, letter, title):
    ax.text(x, y, letter, fontsize=8, fontweight="bold", va="bottom", ha="left")
    ax.text(x + 4.5, y + 0.2, title, fontsize=7, va="bottom", ha="left")


def panel_d(ax, x0, y0):
    label(ax, x0, y0 + 52, "d", "Depth-one QAOA")
    ys = [y0 + 42, y0 + 31, y0 + 12]
    names = ["$q_0$", "$q_1$", "$q_{49}$"]
    for y, n in zip(ys, names):
        ax.text(x0 + 1, y, n + r" $|0\rangle$", ha="left", va="center", fontsize=6)
        ax.plot([x0 + 10, x0 + 55], [y, y], color=INK, lw=0.6, zorder=1)
        ax.add_patch(Rectangle((x0 + 12, y - 2.6), 5.2, 5.2, fc="white", ec=INK, lw=0.6, zorder=2))
        ax.text(x0 + 14.6, y, "H", ha="center", va="center", fontsize=6, zorder=3)
        ax.add_patch(Rectangle((x0 + 47.5, y - 2.6), 6, 5.2, fc="white", ec=INK, lw=0.6, zorder=2))
        th = np.linspace(np.pi * 0.1, np.pi * 0.9, 30)
        ax.plot(x0 + 50.5 + 2.1 * np.cos(th), y - 1.4 + 2.1 * np.sin(th), color=INK, lw=0.5, zorder=3)
        ax.plot([x0 + 50.5, x0 + 52.3], [y - 1.4, y + 1.6], color=INK, lw=0.5, zorder=3)
    for dy in (-2.2, 0, 2.2):
        ax.plot(x0 + 14.6, y0 + 21.5 + dy, "o", ms=0.9, color=INK)
    ax.add_patch(Rectangle((x0 + 21, y0 + 8), 11.5, 38, fc=UC, ec=mf.HERON, lw=0.7, zorder=2))
    ax.text(x0 + 26.75, y0 + 27, r"$U_C(\gamma)$", ha="center", va="center", fontsize=6.5, zorder=3)
    ax.add_patch(Rectangle((x0 + 34.5, y0 + 8), 11.5, 38, fc=UB, ec=mf.NIGHTHAWK, lw=0.7, zorder=2))
    ax.text(x0 + 40.25, y0 + 27, r"$U_B(\beta)$", ha="center", va="center", fontsize=6.5, zorder=3)
    ax.text(x0 + 28, y0 + 2.5, "Cost layer " + r"$U_C(\gamma)$" + ", mixer " + r"$U_B(\beta)$" + "\np = 1 · 50 qubits · Z-basis readout",
            ha="center", va="center", fontsize=5.5, color=GREY, linespacing=1.4)


def panel_e(ax, x0, y0):
    label(ax, x0, y0 + 52, "e", "Calibration and compilation")
    box(ax, x0, y0 + 20, 25, 28, "Local noiseless",
        "Statevector, six\n3 × 3 graphs\n64 Sobol starts\n+ Nelder–Mead\n" + r"$\gamma_0$ = 0.777, $\beta_0$ = 1.174")
    box(ax, x0 + 30, y0 + 20, 25, 28, "QPU calibration",
        "seed-2000 phantom\n5 × 5 grid × 500 shots\nquadratic fit\n2,000-shot\nconfirmation")
    box(ax, x0 + 60, y0 + 35, 26, 13, "Heron · ibm_aachen",
        r"fractional $R_{ZZ}$" + "\n" + r"$\gamma$ = 0.7104, $\beta$ = 1.2263", fc="#EAF2F9", ec=mf.HERON,
        tsize=5.4, bsize=5.2)
    box(ax, x0 + 60, y0 + 20, 26, 13, "Nighthawk · ibm_berlin",
        "controlled-Z\n" + r"$\gamma$ = 0.7072, $\beta$ = 1.1743", fc="#FCEEE6", ec=mf.NIGHTHAWK,
        tsize=5.4, bsize=5.2)
    arrow(ax, (x0 + 25.3, y0 + 34), (x0 + 29.7, y0 + 34))
    arrow(ax, (x0 + 55.3, y0 + 36), (x0 + 59.7, y0 + 41.5))
    arrow(ax, (x0 + 55.3, y0 + 32), (x0 + 59.7, y0 + 26.5))
    box(ax, x0, y0 + 1, 86, 15, "Frozen for all evaluation graphs",
        "RCM relabelling → 10 calibration-aware transpilations (optimization level 3)\n"
        "→ reject circuits with any two-qubit error > 3% · no instance-specific retuning", fc="white", tsize=5.8, bsize=5.3)


def panel_f(ax, x0, y0):
    label(ax, x0, y0 + 52, "f", "Raw-sample evaluation")
    box(ax, x0, y0 + 26, 21, 22, "QPU sampling", "SamplerV2\nXY4 decoupling\nno twirling\n10,000 shots")
    box(ax, x0 + 25, y0 + 26, 21, 22, "Raw bitstrings", "01010…11001\n11100…00110\n00011…10100\n50 bits each",
        fc="white")
    box(ax, x0 + 50, y0 + 26, 21, 22, "Ising energy", r"$s = 1-2b$" + "\n" + r"$E(s)$ per shot" + "\nno repair or\npost-selection")
    arrow(ax, (x0 + 21.3, y0 + 37), (x0 + 24.7, y0 + 37))
    arrow(ax, (x0 + 46.3, y0 + 37), (x0 + 49.7, y0 + 37))
    metrics = ["Mean gap", "CVaR-10% gap", "Best gap", "Best assignment"]
    for i, m in enumerate(metrics):
        cx = x0 + (i % 2) * 36
        cy = y0 + 12 - (i // 2) * 10
        ax.add_patch(FancyBboxPatch((cx, cy), 34.5, 7.5, boxstyle="round,pad=0,rounding_size=1.2",
                                    fc="white", ec=LINE, lw=0.6))
        ax.text(cx + 17.25, cy + 3.75, m, ha="center", va="center", fontsize=6)
    arrow(ax, (x0 + 60.5, y0 + 25.7), (x0 + 60.5, y0 + 20))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=mf.ROOT / "figures/main")
    a = ap.parse_args()
    mf.set_style()
    fig = plt.figure(figsize=(183 * MM, 183 * 62 / 224 * MM * 1.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 224)
    ax.set_ylim(-4, 58)
    ax.axis("off")
    panel_d(ax, 1, 0)
    panel_e(ax, 62, 0)
    panel_f(ax, 153, 0)
    mf.save(fig, a.out, "figure1_panels_def_schematic")


if __name__ == "__main__":
    main()
