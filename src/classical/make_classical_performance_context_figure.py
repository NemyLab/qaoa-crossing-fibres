#!/usr/bin/env python3
"""Create the main-text classical performance-context figure.

The script reads the held-out Step 18 benchmark results, averages repeated
runs within each seed/method/budget combination, and then summarizes the 15
held-out seeds with a mean and two-sided 95% t confidence interval.

Default input
-------------
results/step18_classical_robustness/robustness_results_by_seed_budget.csv

Default outputs
---------------
paper_figures_classical_performance_context/
    figure_classical_performance_context.png
    figure_classical_performance_context.pdf
    figure_classical_performance_context.svg
    figure_classical_performance_context_source_data.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


EXPECTED_BUDGETS = [100, 256, 500, 1000, 2000, 5000]
EXPECTED_SEEDS = list(range(2001, 2016))

METHOD_ORDER = [
    "Hardware-calibrated QAOA",
    "Uniform random sampling",
    "First-improvement greedy",
    "Best-improvement greedy",
    "Simulated annealing",
]

# Step 18 also evaluated a transferred-center QPU parameter strategy. That
# auxiliary sensitivity analysis is intentionally omitted from this main-text
# comparison, which concerns hardware-calibrated QAOA and the four classical
# controls.
EXCLUDED_METHOD_LABELS = {"qpu_transferred_center"}

STYLE = {
    "Hardware-calibrated QAOA": dict(
        color="#0072B2", marker="o", linestyle="-", linewidth=2.4, zorder=5
    ),
    "Uniform random sampling": dict(
        color="#7F7F7F", marker="s", linestyle=(0, (2, 2)), linewidth=1.7, zorder=2
    ),
    "First-improvement greedy": dict(
        color="#E69F00", marker="^", linestyle="--", linewidth=1.7, zorder=3
    ),
    "Best-improvement greedy": dict(
        color="#009E73", marker="D", linestyle="-.", linewidth=1.7, zorder=4
    ),
    "Simulated annealing": dict(
        color="#CC79A7", marker="v", linestyle=(0, (5, 2)), linewidth=1.7, zorder=4
    ),
}


def normalized_column_name(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(df: pd.DataFrame, candidates: list[str], label: str) -> str:
    for candidate in candidates:
        if candidate in df.columns:
            return candidate
    raise ValueError(
        f"Could not identify the {label} column. Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def canonical_method(value: object) -> str:
    raw = str(value).strip()
    name = normalized_column_name(raw)

    # Step 18 stores the measured QAOA curve as
    # ``qpu_hardware_calibrated`` rather than including "qaoa" in the label.
    if "qaoa" in name or name in {
        "qpu_hardware_calibrated",
        "hardware_calibrated_qpu",
        "hardware_calibrated",
    }:
        return "Hardware-calibrated QAOA"
    if "uniform" in name or name in {"random", "random_sampling"}:
        return "Uniform random sampling"
    if "anneal" in name or name in {"sa", "simulated_annealing"}:
        return "Simulated annealing"
    if "greedy" in name and any(token in name for token in ("best", "steepest")):
        return "Best-improvement greedy"
    if "greedy" in name and any(token in name for token in ("first", "simple")):
        return "First-improvement greedy"
    if name in {"greedy", "greedy_first_improvement"}:
        return "First-improvement greedy"

    raise ValueError(f"Unrecognized method label in input CSV: {raw!r}")


def load_and_summarize(csv_path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    raw = pd.read_csv(csv_path)
    raw.columns = [normalized_column_name(c) for c in raw.columns]

    method_col = find_column(
        raw, ["method", "algorithm", "approach", "sampler"], "method"
    )
    budget_col = find_column(
        raw,
        ["budget", "evaluation_budget", "query_budget", "n_evaluations", "queries", "b"],
        "budget",
    )
    seed_col = find_column(
        raw, ["seed", "instance_seed", "problem_seed", "instance"], "seed"
    )
    gap_col = find_column(
        raw,
        [
            "mean_best_normalized_gap",
            "best_normalized_gap_mean",
            "mean_best_gap",
            "best_gap_mean",
            "best_normalized_gap",
            "best_gap",
            "g_best",
        ],
        "best-observed normalized gap",
    )

    normalized_methods = raw[method_col].map(normalized_column_name)
    excluded_present = sorted(
        set(normalized_methods).intersection(EXCLUDED_METHOD_LABELS)
    )
    raw = raw.loc[~normalized_methods.isin(EXCLUDED_METHOD_LABELS)].copy()

    data = pd.DataFrame(
        {
            "method": raw[method_col].map(canonical_method),
            "budget": pd.to_numeric(raw[budget_col], errors="raise").astype(int),
            "seed": pd.to_numeric(raw[seed_col], errors="raise").astype(int),
            "best_gap": pd.to_numeric(raw[gap_col], errors="raise"),
        }
    )

    # If Step 18 stores individual repetitions, obtain one value per independent
    # phantom instance first. If it is already by-seed, this is a no-op.
    by_seed = (
        data.groupby(["method", "budget", "seed"], as_index=False, observed=True)
        .agg(seed_mean_best_gap=("best_gap", "mean"), n_repetitions=("best_gap", "size"))
    )

    rows: list[dict[str, float | int | str]] = []
    for (method, budget), group in by_seed.groupby(
        ["method", "budget"], sort=False, observed=True
    ):
        values = group["seed_mean_best_gap"].to_numpy(dtype=float)
        n = len(values)
        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if n > 1 else np.nan
        sem = sd / np.sqrt(n) if n > 1 else np.nan
        critical = float(t.ppf(0.975, n - 1)) if n > 1 else np.nan
        half_width = critical * sem if n > 1 else np.nan
        rows.append(
            {
                "method": method,
                "budget": int(budget),
                "n_seeds": n,
                "mean_best_normalized_gap": mean,
                "sd_across_seeds": sd,
                "sem_across_seeds": sem,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
                "min_repetitions_per_seed": int(group["n_repetitions"].min()),
                "max_repetitions_per_seed": int(group["n_repetitions"].max()),
            }
        )

    summary = pd.DataFrame(rows)
    summary["method"] = pd.Categorical(
        summary["method"], categories=METHOD_ORDER, ordered=True
    )
    summary = summary.sort_values(["method", "budget"]).reset_index(drop=True)

    detected = {
        "method": method_col,
        "budget": budget_col,
        "seed": seed_col,
        "gap": gap_col,
        "excluded_methods": ", ".join(excluded_present) or "none",
    }
    validate_design(by_seed, summary)
    return summary, detected


def validate_design(by_seed: pd.DataFrame, summary: pd.DataFrame) -> None:
    methods = set(summary["method"].astype("object"))
    missing_methods = set(METHOD_ORDER) - methods
    if missing_methods:
        raise ValueError(f"Missing expected methods: {sorted(missing_methods)}")

    budgets = sorted(summary["budget"].unique().tolist())
    if budgets != EXPECTED_BUDGETS:
        raise ValueError(
            f"Unexpected budgets: {budgets}; expected {EXPECTED_BUDGETS}."
        )

    for method in METHOD_ORDER:
        for budget in EXPECTED_BUDGETS:
            observed = sorted(
                by_seed.loc[
                    (by_seed["method"] == method) & (by_seed["budget"] == budget),
                    "seed",
                ].unique().tolist()
            )
            if observed != EXPECTED_SEEDS:
                missing = sorted(set(EXPECTED_SEEDS) - set(observed))
                extra = sorted(set(observed) - set(EXPECTED_SEEDS))
                raise ValueError(
                    f"Seed mismatch for {method}, B={budget}: "
                    f"missing={missing}, extra={extra}."
                )


def set_paper_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "legend.fontsize": 8.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def make_figure(summary: pd.DataFrame, out_dir: Path) -> None:
    set_paper_style()
    fig, ax = plt.subplots(figsize=(6.65, 4.35))

    for method in METHOD_ORDER:
        group = summary.loc[summary["method"] == method].sort_values("budget")
        x = group["budget"].to_numpy(dtype=float)
        y = group["mean_best_normalized_gap"].to_numpy(dtype=float)
        lower = y - group["ci95_low"].to_numpy(dtype=float)
        upper = group["ci95_high"].to_numpy(dtype=float) - y
        style = STYLE[method]

        ax.errorbar(
            x,
            y,
            yerr=np.vstack([lower, upper]),
            label=method,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=style["linewidth"],
            markersize=5.2,
            markerfacecolor="white" if method != "Hardware-calibrated QAOA" else style["color"],
            markeredgecolor=style["color"],
            markeredgewidth=1.1,
            elinewidth=1.0,
            capsize=2.5,
            capthick=1.0,
            zorder=style["zorder"],
        )

    ax.set_xscale("log")
    ax.set_xticks(EXPECTED_BUDGETS)
    ax.set_xticklabels(["100", "256", "500", "1,000", "2,000", "5,000"])
    ax.set_xlabel(r"Evaluation budget, $B$")
    ax.set_ylabel(
        "Mean best-observed normalized gap, "
        r"$\widehat{G}_{\mathrm{best}}(B)$"
    )
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper right", handlelength=2.8)
    ax.margins(x=0.035)

    fig.tight_layout(pad=0.7)
    prefix = out_dir / "figure_classical_performance_context"
    fig.savefig(prefix.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(prefix.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot Step 18 QAOA-versus-classical finite-budget performance."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/step18_classical_robustness/"
            "robustness_results_by_seed_budget.csv"
        ),
        help="Step 18 by-seed benchmark CSV.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper_figures_classical_performance_context"),
        help="Output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.is_file():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary, detected = load_and_summarize(args.input)
    source_path = (
        args.out_dir / "figure_classical_performance_context_source_data.csv"
    )
    summary.to_csv(source_path, index=False, float_format="%.10g")
    make_figure(summary, args.out_dir)

    print(f"Input: {args.input}")
    print("Detected columns: " + ", ".join(f"{k}={v}" for k, v in detected.items()))
    print(f"Validated held-out seeds: {EXPECTED_SEEDS[0]}-{EXPECTED_SEEDS[-1]}")
    print(f"Validated budgets: {EXPECTED_BUDGETS}")
    print(f"Saved outputs in: {args.out_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
