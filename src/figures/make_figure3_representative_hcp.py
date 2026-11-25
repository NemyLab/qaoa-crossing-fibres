#!/usr/bin/env python3
"""Figure 3: representative HCP patch (subject 103414) from graph to QPU sample.

Inputs (all in data/hcp/representative_103414/):
  mrview_fod_glyphs_coronal.png        MRtrix mrview rendering of the 5x5 FOD patch
                                       (radiological display: subject right on the left)
  real_graph_nodes_rcm.csv / real_graph_edges_rcm.csv / real_graph_solution_arrays.npz
  real_fractional_hardware_results.json / real_fractional_hardware_counts.json.gz
  real_fractional_best_assignments.npz

Panels: a FOD glyphs; b unlabelled candidate axes; c sparse Ising graph; d exact MILP
assignment on the graph; e exact labels in voxel space; f energy distribution of the
10,000 Heron samples versus 10,000 uniform assignments; g best measured QPU assignment
on the graph; h the same in voxel space. Energies are recomputed from the graph and
checked against the stored values.

Usage:
  python src/figures/make_figure3_representative_hcp.py --out figures/main
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

import make_main_figures as mf

DATA = mf.ROOT / "data/hcp/representative_103414"
MM = mf.MM
POP_A, POP_B = "#0072B2", "#E69F00"
CONFLICT, CONT = "#C8372D", "#6A8CB5"
QPU, UNIF = "#009E73", "#BDBDBD"
NODE_OFF = np.array([0.17, 0.13])
UNIFORM_SEED = 803414


# ------------------------------------------------------------------ data
def load():
    nodes = pd.read_csv(DATA / "real_graph_nodes_rcm.csv").sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(DATA / "real_graph_edges_rcm.csv")
    arr = np.load(DATA / "real_graph_solution_arrays.npz")
    fields = np.asarray(arr["fields_pre_rcm"], float)[nodes["old_node_id"].to_numpy(int)]
    exact = np.asarray(arr["exact_spins_rcm"], int)
    res = json.loads((DATA / "real_fractional_hardware_results.json").read_text())
    row = next(r for r in res["strategies"] if r["angle_strategy"] == "hardware_calibrated")
    with gzip.open(DATA / "real_fractional_hardware_counts.json.gz", "rt") as fh:
        counts = json.load(fh)["hardware_calibrated"]
    best = np.asarray(np.load(DATA / "real_fractional_best_assignments.npz")["hardware_calibrated_best_spins_rcm"], int)
    return nodes, edges, fields, exact, row, counts, best


def energy_fn(edges, fields, n):
    i, j = edges["i"].to_numpy(int), edges["j"].to_numpy(int)
    w = edges["weight"].to_numpy(float)
    sgn = np.where(edges["edge_type"].to_numpy() == "continuation", -1.0, 1.0)

    def energy(spins):
        s = np.atleast_2d(spins).astype(float)
        return -(s @ fields) + (s[:, i] * s[:, j]) @ (sgn * w)
    return energy


def bitstrings_to_spins(keys, n):
    bits = np.array([[int(c) for c in k[::-1]] for k in keys], dtype=np.int8)
    assert bits.shape[1] == n
    return 1 - 2 * bits


def align(spins, ref):
    return -spins if np.mean(-spins == ref) > np.mean(spins == ref) else spins


# ------------------------------------------------------------------ geometry
def geometry(nodes):
    x = 4.0 - nodes["col"].to_numpy(float)          # radiological: subject right on the left
    y = 4.0 - nodes["row"].to_numpy(float)          # row 0 is superior
    dx, dy = -nodes["dir_x"].to_numpy(float), nodes["dir_z"].to_numpy(float)
    norm = np.hypot(dx, dy)
    dx, dy = dx / norm, dy / norm
    peak = nodes["peak_type"].str.extract(r"(\d+)$")[0].astype(int).to_numpy()
    sign = np.where(peak == 0, -1.0, 1.0)
    pos = np.column_stack([x + sign * NODE_OFF[0], y + sign * NODE_OFF[1]])
    return x, y, dx, dy, pos


def frame(ax):
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylim(-0.6, 4.6)
    ax.set_aspect("equal")
    ax.axis("off")


def voxel_tiles(ax):
    for gx in range(5):
        for gy in range(5):
            ax.add_patch(plt.Rectangle((gx - 0.44, gy - 0.44), 0.88, 0.88, fc="#F3F3F3", ec="none", zorder=0))


def draw_axes(ax, x, y, dx, dy, colours, lw=1.6, half=0.36):
    segs = [[(xi - half * a, yi - half * b), (xi + half * a, yi + half * b)] for xi, yi, a, b in zip(x, y, dx, dy)]
    ax.add_collection(LineCollection(segs, colors=colours, linewidths=lw, capstyle="round", zorder=2))


def draw_graph(ax, edges, pos, fields, spins=None, mismatch=None, dx=None, dy=None):
    wmax = edges.loc[edges.edge_type == "continuation", "weight"].max()
    for e in edges.itertuples(index=False):
        p, q = pos[int(e.i)], pos[int(e.j)]
        conflict = e.edge_type == "conflict"
        ok = True
        if spins is not None:
            same = spins[int(e.i)] == spins[int(e.j)]
            ok = (not same) if conflict else same
        if conflict:
            lw, col = 1.6, CONFLICT
        else:
            lw, col = 0.35 + 1.5 * float(e.weight) / wmax, CONT
        if not ok:
            ax.plot([p[0], q[0]], [p[1], q[1]], color="#8C8C8C", lw=0.8, ls=(0, (1, 1.2)), zorder=1)
        else:
            ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=lw, alpha=0.9 if conflict else 0.75,
                    solid_capstyle="round", zorder=1)
    if spins is None:
        fc = ["white"] * len(pos)
    else:
        fc = [POP_A if s > 0 else POP_B for s in spins]
    ax.scatter(pos[:, 0], pos[:, 1], s=16, c=fc, edgecolors=mf.INK, linewidths=0.45, zorder=3)
    if dx is not None:
        h = 0.075
        segs = [[(px - h * a, py - h * b), (px + h * a, py + h * b)] for (px, py), a, b in zip(pos, dx, dy)]
        ax.add_collection(LineCollection(segs, colors="white" if spins is not None else mf.INK,
                                         linewidths=0.6, zorder=4))
    anchors = np.flatnonzero(np.abs(fields) > 1e-12)
    ax.scatter(pos[anchors, 0] - 0.13, pos[anchors, 1] + 0.13, marker="*", s=12, c=mf.INK, lw=0, zorder=5)
    if mismatch is not None:
        m = np.flatnonzero(mismatch)
        ax.scatter(pos[m, 0], pos[m, 1], s=58, facecolors="none", edgecolors=mf.INK, linewidths=0.7, zorder=4)


def caption(ax, text):
    ax.text(0.5, -0.03, text, transform=ax.transAxes, ha="center", va="top", fontsize=5.6, color=mf.INK)


def title(ax, letter, text):
    ax.text(-0.02, 1.07, letter, transform=ax.transAxes, fontsize=8, fontweight="bold", va="bottom")
    ax.text(0.09, 1.07, text, transform=ax.transAxes, fontsize=6.8, va="bottom")


# ------------------------------------------------------------------ figure
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=mf.ROOT / "figures/main")
    a = ap.parse_args()
    mf.set_style()

    nodes, edges, fields, exact, row, counts, best = load()
    n = len(nodes)
    energy = energy_fn(edges, fields, n)
    e_exact = float(energy(exact)[0])
    e_best = float(energy(best)[0])
    assert abs(e_exact - row["exact_energy"]) < 1e-6, (e_exact, row["exact_energy"])
    assert abs(e_best - row["best_energy"]) < 1e-6, (e_best, row["best_energy"])

    keys = list(counts)
    reps = np.array([counts[k] for k in keys])
    e_qpu = np.repeat(energy(bitstrings_to_spins(keys, n)), reps)
    assert abs(e_qpu.mean() - row["mean_energy"]) < 1e-6
    rng = np.random.default_rng(UNIFORM_SEED)
    e_unif = energy(1 - 2 * rng.integers(0, 2, size=(10000, n)))

    exact_d = exact.copy()
    best_d = align(best, exact_d)
    mismatch = best_d != exact_d
    agree = 1 - mismatch.mean()
    assert abs(agree - row["best_agreement_with_exact_up_to_sign"]) < 1e-9
    gap = (e_best - e_exact) / (-e_exact)

    x, y, dx, dy, pos = geometry(nodes)
    vox_mis = pd.Series(mismatch).groupby([nodes["row"], nodes["col"]]).any()

    fig = plt.figure(figsize=(183 * MM, 104 * MM))
    gs = fig.add_gridspec(2, 4, left=0.01, right=0.995, top=0.93, bottom=0.105, wspace=0.12, hspace=0.34)
    ax = [fig.add_subplot(gs[r, c]) for r in range(2) for c in range(4)]

    # a FOD glyphs
    img = plt.imread(DATA / "mrview_fod_glyphs_coronal.png")
    ax[0].imshow(img, extent=(-0.5, 4.5, -0.5, 4.5), interpolation="lanczos")
    frame(ax[0])
    for txt, xy, ha in (("R", (-0.55, 4.55), "left"), ("L", (4.55, 4.55), "right")):
        ax[0].text(*xy, txt, ha=ha, va="bottom", fontsize=5.6, color=mf.GREY)
    title(ax[0], "a", "Fibre-orientation distributions")
    caption(ax[0], "HCP 103414, right CST / SCR, 5 × 5 voxels")

    # b candidate axes
    voxel_tiles(ax[1])
    draw_axes(ax[1], x, y, dx, dy, ["#4D4D4D"] * n)
    frame(ax[1])
    title(ax[1], "b", "Unlabelled candidate axes")
    caption(ax[1], "Two strongest CSD peaks per voxel")

    # c graph
    draw_graph(ax[2], edges, pos, fields, dx=dx, dy=dy)
    frame(ax[2])
    title(ax[2], "c", "Sparse Ising graph")
    n_cont = int((edges.edge_type == "continuation").sum())
    n_conf = int((edges.edge_type == "conflict").sum())
    caption(ax[2], f"{n} nodes · {n_cont} continuation · {n_conf} conflict edges")

    # d exact graph
    draw_graph(ax[3], edges, pos, fields, spins=exact_d, dx=dx, dy=dy)
    frame(ax[3])
    title(ax[3], "d", "Exact assignment (MILP)")
    caption(ax[3], f"E = {e_exact:.2f}".replace("-", "−") + " · 100% agreement with TOM")

    # e exact voxel space
    voxel_tiles(ax[4])
    draw_axes(ax[4], x, y, dx, dy, [POP_A if s > 0 else POP_B for s in exact_d])
    frame(ax[4])
    title(ax[4], "e", "Exact fibre assignment")
    caption(ax[4], "MILP labels on the measured peak axes")

    # f energy histogram
    axf = ax[5]
    bins = np.linspace(min(e_exact, e_qpu.min()) - 1, max(e_unif.max(), e_qpu.max()) + 1, 60)
    axf.hist(e_unif, bins=bins, density=True, color=UNIF, alpha=0.9, lw=0, label="Uniform (10,000)")
    axf.hist(e_qpu, bins=bins, density=True, color=QPU, alpha=0.85, lw=0, label="Heron QAOA (10,000 shots)")
    ymax = axf.get_ylim()[1] * 1.75
    for val, ls, lab in ((e_exact, "-", "Exact optimum"), (e_best, (0, (4, 1.5)), "Best QPU sample"),
                         (e_qpu.mean(), (0, (1, 1.2)), "QPU mean")):
        axf.axvline(val, ymax=0.6, color=mf.INK, lw=0.8, ls=ls, label=lab)
    axf.set_ylim(0, ymax)
    axf.set_xlabel("Ising energy, E")
    axf.set_ylabel("Density")
    axf.legend(loc="upper right", fontsize=5.2, handlelength=1.8, borderaxespad=0.0, labelspacing=0.25)
    pf = axf.get_position()
    axf.set_position([pf.x0 + 0.045, pf.y0 + 0.035, pf.width - 0.05, pf.height - 0.035])
    axf.text(-0.27, 1.07, "f", transform=axf.transAxes, fontsize=8, fontweight="bold", va="bottom")
    axf.text(-0.15, 1.07, "QAOA sampling on Heron", transform=axf.transAxes, fontsize=6.8, va="bottom")

    # g best QPU graph
    draw_graph(ax[6], edges, pos, fields, spins=best_d, mismatch=mismatch, dx=dx, dy=dy)
    frame(ax[6])
    title(ax[6], "g", "Best measured QPU assignment")
    caption(ax[6], f"Gap {gap:.3f} · {agree:.0%} agreement with exact")

    # h best QPU voxel space
    voxel_tiles(ax[7])
    draw_axes(ax[7], x, y, dx, dy, [POP_A if s > 0 else POP_B for s in best_d])
    for (r, c), bad in vox_mis.items():
        if bad:
            ax[7].add_patch(plt.Rectangle((4 - c - 0.44, 4 - r - 0.44), 0.88, 0.88, fc="none", ec=mf.INK,
                                          lw=0.7, zorder=3))
    frame(ax[7])
    title(ax[7], "h", "Best QPU fibre assignment")
    caption(ax[7], "Outlined voxels differ from the exact solution")

    handles = [
        Line2D([], [], color=CONFLICT, lw=1.6, label="Within-voxel conflict"),
        Line2D([], [], color=CONT, lw=1.2, label="Continuation (width $\\propto$ weight)"),
        Line2D([], [], color="#8C8C8C", lw=0.8, ls=(0, (1, 1.2)), label="Unsatisfied edge"),
        Line2D([], [], marker="*", ls="", color=mf.INK, ms=4, label="Boundary field"),
        Line2D([], [], marker="o", ls="", mfc=POP_A, mec=mf.INK, mew=0.4, ms=4, label="Population 1"),
        Line2D([], [], marker="o", ls="", mfc=POP_B, mec=mf.INK, mew=0.4, ms=4, label="Population 2"),
        Line2D([], [], marker="o", ls="", mfc="none", mec=mf.INK, mew=0.7, ms=5.5, label="Differs from exact"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=7, bbox_to_anchor=(0.5, -0.01), fontsize=5.6,
               columnspacing=1.4, handletextpad=0.4)
    mf.save(fig, a.out, "figure3_representative_hcp_patch")
    print(f"exact {e_exact:.4f} best {e_best:.4f} mean {e_qpu.mean():.4f} gap {gap:.4f} agree {agree:.2f}")


if __name__ == "__main__":
    main()
