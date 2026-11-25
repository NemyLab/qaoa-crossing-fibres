#!/usr/bin/env python3
"""Regenerate main-text Figures 2, 4 and 5 from the released source data.

All three figures share one journal style (Nature Portfolio): Arial-metric
sans-serif, 5-7 pt text, bold lowercase 8 pt panel letters, no in-figure
titles, and the Okabe-Ito colour palette. Outputs are vector PDF and SVG
(editable text) plus 600-dpi PNG/TIFF.

Inputs (repository-relative):
  data/phantom/phantom_study_level_results.csv   -> Figure 2
  data/hcp/hcp_subject_level_results.csv         -> Figure 4
  analysis/hcp_baselines/hcp_nonspatial_baselines_summary.csv  -> Figure 4 (random reference)
  data/classical/figure5_source_data.csv         -> Figure 5

Usage:
  python src/figures/make_main_figures.py --out figures/main
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
MM = 1 / 25.4

HERON = "#0072B2"
NIGHTHAWK = "#D55E00"
GREY = "#7F7F7F"
LIGHT = "#C8C8C8"
INK = "#1A1A1A"


def set_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 6.5,
        "axes.labelsize": 6.5,
        "axes.titlesize": 7,
        "xtick.labelsize": 6,
        "ytick.labelsize": 6,
        "legend.fontsize": 6,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
        "mathtext.default": "regular",
    })


def panel_label(ax, letter, x=-0.16, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=8, fontweight="bold",
            va="bottom", ha="left")


def ci95(x):
    x = np.asarray(x, float)
    m = x.mean()
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return m, m - h, m + h


def swarm(values, width=0.055, tol=None, max_half=0.13):
    """Deterministic beeswarm offsets: points within `tol` share a row and are spread symmetrically."""
    v = np.asarray(values, float)
    if tol is None:
        tol = 0.012 * (np.nanmax(v) - np.nanmin(v) or 1.0)
    off = np.zeros(len(v))
    order = np.argsort(v, kind="stable")
    groups, cur = [], [order[0]]
    for i in order[1:]:
        if abs(v[i] - v[cur[0]]) <= tol:
            cur.append(i)
        else:
            groups.append(cur)
            cur = [i]
    groups.append(cur)
    for g in groups:
        k = len(g)
        w = min(width, 2 * max_half / (k - 1)) if k > 1 else 0.0
        for j, i in enumerate(g):
            off[i] = (j - (k - 1) / 2) * w
    return off


def holm(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def fmt_p(p):
    if p < 0.001:
        return "P < 0.001"
    if p >= 0.995:
        return "P = 1.00"
    return f"P = {p:.3f}" if p < 0.1 else f"P = {p:.2f}"


def save(fig, out: Path, name: str):
    out.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "svg", "png"):
        fig.savefig(out / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    fig.savefig(out / f"{name}.tiff", bbox_inches="tight", pad_inches=0.02,
                pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)


# ---------------------------------------------------------------- Figure 2
CONDITIONS = ["90° original", "90° rotated 45°", "60° original", "60° rotated 45°"]
TICKS = ["90°\noriginal", "90°\nrotated 45°", "60°\noriginal", "60°\nrotated 45°"]
POS = np.array([0.0, 1.0, 2.4, 3.4])


def figure2(out: Path):
    d = pd.read_csv(ROOT / "data/phantom/phantom_study_level_results.csv")
    d["exec_penalty"] = d["qpu_mean_normalized_gap"] - d["ideal_mean_normalized_gap"]
    metrics = [
        ("qpu_mean_normalized_gap", "Mean normalized gap, $\\hat{G}_{QPU}$", "a", False),
        ("exec_penalty", "Execution penalty, $\\Delta G_{exec}$", "b", False),
        ("qpu_cvar10_normalized_gap", "CVaR-10% normalized gap, $\\hat{C}_{0.10}$", "c", False),
        ("best_agreement_with_exact", "Best-sample agreement with MILP (%)", "d", True),
    ]
    subtitles = {"a": "QPU sampling", "b": "Hardware penalty vs ideal p = 1",
                 "c": "Low-energy tail", "d": "Best measured assignment"}
    fig, axes = plt.subplots(2, 2, figsize=(183 * MM, 118 * MM))
    rows = []
    for ax, (col, ylab, letter, pct) in zip(axes.flat, metrics):
        pvals, tops = [], []
        for c, x0 in zip(CONDITIONS, POS):
            h = d[(d.condition == c) & (d.platform == "Heron")].set_index("unit_id")[col]
            n = d[(d.condition == c) & (d.platform == "Nighthawk")].set_index("unit_id")[col]
            h, n = h.align(n, join="inner")
            sc = 100 if pct else 1
            tol = 0.5 if pct else 0.011 * (d[col].max() - d[col].min())
            xh = x0 - 0.17 + swarm(h * sc, width=0.04, tol=tol, max_half=0.1)
            xn = x0 + 0.17 + swarm(n * sc, width=0.04, tol=tol, max_half=0.1)
            for a, b, xa, xb in zip(h * sc, n * sc, xh, xn):
                ax.plot([xa, xb], [a, b], color=LIGHT, lw=0.4, zorder=1)
            ax.scatter(xh, h * sc, s=9, color=HERON, edgecolor="white", lw=0.3, zorder=2)
            ax.scatter(xn, n * sc, s=9, marker="s", color=NIGHTHAWK, edgecolor="white", lw=0.3, zorder=2)
            for vals, xm, colr, mk in ((h, x0 - 0.17, HERON, "o"), (n, x0 + 0.17, NIGHTHAWK, "s")):
                m, lo, hi = ci95(vals * sc)
                ax.errorbar(xm + (-0.11 if mk == "o" else 0.11), m, yerr=[[m - lo], [hi - m]], fmt="D",
                            ms=3.2, color=INK, mfc=colr, mec=INK, mew=0.5, elinewidth=0.7, capsize=1.6, zorder=3)
            pvals.append(stats.ttest_rel(n, h).pvalue)
            tops.append(max(h.max(), n.max()) * sc)
            rows.append(dict(metric=col, condition=c, heron_mean=h.mean(), nighthawk_mean=n.mean(),
                             paired_t_p=pvals[-1]))
        padj = holm(pvals)
        for r, pa in zip(rows[-4:], padj):
            r["paired_t_p_holm"] = pa
        lo_, hi_ = ax.get_ylim()
        span = hi_ - lo_
        for x0, top, pa in zip(POS, tops, padj):
            y = top + 0.05 * span
            ax.plot([x0 - 0.17, x0 - 0.17, x0 + 0.17, x0 + 0.17], [y - 0.015 * span, y, y, y - 0.015 * span],
                    color=INK, lw=0.5)
            ax.text(x0, y + 0.012 * span, fmt_p(pa), ha="center", va="bottom", fontsize=5.5)
        ax.set_ylim(lo_, hi_ + 0.12 * span)
        ax.axvline(1.7, color=LIGHT, lw=0.5, ls=(0, (3, 2)))
        ax.set_xticks(POS, TICKS)
        ax.set_xlim(-0.5, 3.9)
        ax.set_ylabel(ylab)
        ax.yaxis.grid(True, color="#EDEDED", lw=0.4)
        ax.set_axisbelow(True)
        ax.set_title(subtitles[letter], loc="left", fontsize=7, pad=8)
        panel_label(ax, letter, x=-0.13, y=1.03)
        if pct:
            ax.set_ylim(40, 100 + 0.12 * 60)
            ax.set_yticks(range(40, 101, 10))
    handles = [
        Line2D([], [], marker="o", ls="", color=HERON, mec="white", ms=4, label="Heron (ibm_aachen)"),
        Line2D([], [], marker="s", ls="", color=NIGHTHAWK, mec="white", ms=4, label="Nighthawk (ibm_berlin)"),
        Line2D([], [], marker="D", ls="", mfc="white", mec=INK, ms=3.5, label="Mean ± 95% CI"),
        Line2D([], [], color=LIGHT, lw=0.8, label="Same seed and condition"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.015))
    fig.tight_layout(rect=(0, 0.035, 1, 1), h_pad=1.6, w_pad=2.5)
    save(fig, out, "figure2_heron_vs_nighthawk")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- Figure 4
def figure4(out: Path):
    d = pd.read_csv(ROOT / "data/hcp/hcp_subject_level_results.csv")
    base = pd.read_csv(ROOT / "analysis/hcp_baselines/hcp_nonspatial_baselines_summary.csv").set_index("method")
    rand = float(base.loc["random_feasible_expected", "mean"]) * 100
    cols = [("best_agreement_with_exact_up_to_sign", "Agreement with\nexact MILP", HERON),
            ("best_tom_agreement_confident_voxels", "Agreement with\nTOM (confident)", "#009E73"),
            ("best_conflict_satisfaction_fraction", "Within-voxel\nconflicts satisfied", GREY)]
    fig, ax = plt.subplots(figsize=(89 * MM, 62 * MM))
    xs = np.arange(3)
    offs = {c: swarm(d[c].to_numpy(float) * 100, width=0.06, tol=0.5, max_half=0.2) for c, _, _ in cols[:2]}
    offs[cols[2][0]] = swarm(d[cols[2][0]].to_numpy(float) * 100, width=0.06, tol=0.5, max_half=0.12)
    for i in range(len(d)):
        ax.plot([xs[0] + offs[cols[0][0]][i], xs[1] + offs[cols[1][0]][i]],
                d.loc[i, [cols[0][0], cols[1][0]]].to_numpy(float) * 100,
                color="#E3E3E3", lw=0.35, zorder=1)
    for x, (c, _, colr) in zip(xs, cols):
        v = d[c].to_numpy(float) * 100
        ax.scatter(x + offs[c], v, s=10, color=colr, edgecolor="white", lw=0.3, zorder=2, alpha=0.9)
        m, lo, hi = ci95(v)
        ax.errorbar(x + 0.34, m, yerr=[[m - lo], [hi - m]], fmt="D", ms=3.4, mfc=colr, mec=INK,
                    mew=0.5, color=INK, elinewidth=0.7, capsize=1.6, zorder=3)
        ax.text(x + 0.44, m, f"{m:.1f}%", va="center", ha="left", fontsize=5.5)
    full = int((d[cols[2][0]] >= 1 - 1e-9).sum())
    ax.text(2, 93.5, f"{full}/{len(d)} at 100%", ha="center", va="top", fontsize=5.5, color=GREY)
    ax.plot([-0.4, 1.4], [rand, rand], color=INK, lw=0.6, ls=(0, (3, 2)))
    ax.text(1.42, rand, f"Random feasible\n({rand:.0f}%)", va="center", ha="left", fontsize=5.5, color=GREY)
    ax.set_xticks(xs, [c[1] for c in cols])
    ax.set_xlim(-0.5, 2.85)
    ax.set_ylim(40, 102)
    ax.set_yticks(range(40, 101, 10))
    ax.set_ylabel("Best measured QPU assignment (%)")
    ax.yaxis.grid(True, color="#EDEDED", lw=0.4)
    ax.set_axisbelow(True)
    save(fig, out, "figure4_hcp_best_assignment_quality")


# ---------------------------------------------------------------- Figure 5
METHODS = [
    ("Hardware-calibrated QAOA", "#0072B2", "o", "-", 1.3, True),
    ("Uniform random sampling", GREY, "s", (0, (1, 1.2)), 0.9, False),
    ("First-improvement greedy", "#E69F00", "^", (0, (4, 1.5)), 0.9, False),
    ("Best-improvement greedy", "#009E73", "D", (0, (5, 1.5, 1, 1.5)), 0.9, False),
    ("Simulated annealing", "#CC79A7", "v", (0, (2.5, 1.5)), 0.9, False),
]


def figure5(out: Path):
    d = pd.read_csv(ROOT / "data/classical/figure5_source_data.csv")
    fig, ax = plt.subplots(figsize=(89 * MM, 66 * MM))
    for name, colr, mk, ls, lw, filled in METHODS:
        s = d[d.method == name].sort_values("budget")
        ax.errorbar(s.budget, s.mean_best_normalized_gap,
                    yerr=[s.mean_best_normalized_gap - s.ci95_low, s.ci95_high - s.mean_best_normalized_gap],
                    color=colr, ls=ls, lw=lw, marker=mk, ms=3.6, mfc=colr if filled else "white",
                    mec=colr, mew=0.8, elinewidth=0.6, capsize=1.5, label=name, zorder=3 if filled else 2)
    ax.axvspan(90, 560, color="#F2F2F2", zorder=0, lw=0)
    ax.text(225, 0.86, "QAOA lowest", ha="center", va="top", fontsize=5.5, color=GREY)
    ax.set_xscale("log")
    budgets = sorted(d.budget.unique())
    ax.set_xticks(budgets, [f"{b:,}" for b in budgets])
    ax.xaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax.set_xlim(85, 5900)
    ax.set_ylim(0, 0.88)
    ax.set_xlabel("Objective-function evaluations, B")
    ax.set_ylabel("Best-observed normalized gap, $\\hat{G}_{best}(B)$")
    ax.yaxis.grid(True, color="#EDEDED", lw=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", handlelength=2.6, borderaxespad=0.2)
    save(fig, out, "figure5_classical_performance_context")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=ROOT / "figures/main")
    a = ap.parse_args()
    set_style()
    stats2 = figure2(a.out)
    stats2.to_csv(a.out / "figure2_paired_tests.csv", index=False)
    figure4(a.out)
    figure5(a.out)
    print(stats2.round(4).to_string())


if __name__ == "__main__":
    main()
