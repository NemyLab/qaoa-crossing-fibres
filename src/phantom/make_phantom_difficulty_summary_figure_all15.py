#!/usr/bin/env python3
"""Create a compact four-condition phantom summary figure.

The figure compares:
    1. 90-degree phantom
    2. 90-degree phantom globally rotated by 45 degrees
    3. 60-degree phantom
    4. 60-degree phantom globally rotated by 45 degrees

Panels:
    A. Exact MILP agreement with the known phantom labels
    B. Ideal p=1 QAOA expected normalized gap
    C. QPU CVaR10 normalized gap (configurable to mean gap)
    D. Best measured QPU assignment agreement with the exact MILP solution

The primary QPU analysis uses the same angle strategy and seeds in every
condition. Individual seed values, paired orientation lines, and mean with
95% confidence intervals are shown. Each panel and the complete 2x2 figure are exported as PNG, PDF, and SVG.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy.stats import t


CONDITIONS = (
    ("90_original", "90°\noriginal", "curved_90", 0.0),
    ("90_rotated", "90°\nrotated 45°", "curved_90_rot45", 1.0),
    ("60_original", "60°\noriginal", "curved_60", 2.6),
    ("60_rotated", "60°\nrotated 45°", "curved_60_rot45", 3.6),
)

# Same colorblind-safe family used in the established crossing-fiber figures.
CONDITION_COLOURS = {
    "90_original": "#0072B2",
    "90_rotated": "#56B4E9",
    "60_original": "#D55E00",
    "60_rotated": "#E69F00",
}

PANEL_SPECS = (
    {
        "label": "A",
        "metric": "exact_solution_label_accuracy",
        "title": "Exact reconstruction of phantom labels",
        "ylabel": "Exact MILP label accuracy",
        "accuracy": True,
        "filename": "panel_A_exact_milp_label_accuracy",
        "direction": "Higher is better",
    },
    {
        "label": "B",
        "metric": "ideal_expected_normalized_gap",
        "title": "Ideal p=1 QAOA objective quality",
        "ylabel": "Expected normalized gap",
        "accuracy": False,
        "filename": "panel_B_ideal_qaoa_expected_gap",
        "direction": "Lower is better",
    },
    {
        "label": "C",
        "metric": "__QPU_GAP_METRIC__",
        "title": "__QPU_GAP_TITLE__",
        "ylabel": "__QPU_GAP_YLABEL__",
        "accuracy": False,
        "filename": "panel_C_qpu_gap",
        "direction": "Lower is better",
    },
    {
        "label": "D",
        "metric": "best_agreement_with_exact",
        "title": "Best measured QPU assignment",
        "ylabel": "Agreement with exact MILP",
        "accuracy": True,
        "filename": "panel_D_best_qpu_agreement_with_exact",
        "direction": "Higher is better",
    },
)


def configure_matplotlib(use_latex: bool) -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9.0,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.2,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "text.usetex": bool(use_latex),
        }
    )


def resolve_file(path: Path) -> Path:
    """Resolve an explicitly supplied path or find one unique matching basename."""
    path = path.expanduser()
    if path.is_file():
        return path.resolve()

    matches = sorted(
        candidate.resolve()
        for candidate in Path.cwd().rglob(path.name)
        if candidate.is_file()
    )
    if len(matches) == 1:
        print(f"Resolved {path} -> {matches[0]}")
        return matches[0]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {path}. Supply its location explicitly on the command line."
        )
    formatted = "\n  ".join(str(item) for item in matches)
    raise FileNotFoundError(
        f"{path} was not found exactly and its basename is ambiguous:\n  {formatted}\n"
        "Supply the intended full path explicitly."
    )


def read_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(
        resolve_file(path),
        dtype={"best_bitstring": str, "angle_strategy": str},
    )
    if "seed" not in frame.columns or "angle_strategy" not in frame.columns:
        raise ValueError(f"{path}: missing seed or angle_strategy column")
    frame["seed"] = pd.to_numeric(frame["seed"], errors="raise").astype(int)
    return frame


def read_results_many(paths: Iterable[Path]) -> pd.DataFrame:
    """Read and concatenate non-overlapping QPU result cohorts."""
    frames = []
    for path in paths:
        frame = read_results(path)
        frame["source_file"] = str(resolve_file(path))
        frames.append(frame)
    if not frames:
        raise ValueError("At least one result CSV must be supplied")
    return pd.concat(frames, ignore_index=True)


def build_graph_for_agreement(seed: int, geometry: str):
    """Rebuild a graph only when an older CSV lacks best_agreement_with_exact."""
    try:
        from step5_curved_signal_graph import build_curved_signal_graph, exact_milp
        from step9_common import get_rcm_order, reorder_graph
    except ImportError as exc:
        raise RuntimeError(
            "An input CSV lacks best_agreement_with_exact, so the script must "
            "reconstruct it. Run from crossing_fibres_step7 where "
            "step5_curved_signal_graph.py and step9_common.py are available."
        ) from exc

    graph, _diagnostics = build_curved_signal_graph(
        grid_size=5,
        n_gradients=60,
        snr=10.0,
        b_value=2000.0,
        geometry=geometry,
        fraction_mode="imbalanced",
        degree_cap=3,
        conflict_weight=1.0,
        anchor_weight=0.35,
        sigma_orientation_deg=18.0,
        sigma_displacement_deg=22.0,
        minimum_continuation_weight=0.01,
        seed=int(seed),
    )
    graph = reorder_graph(graph, get_rcm_order(graph))
    exact_spins, _energy, _solver_info = exact_milp(graph)
    return graph, np.asarray(exact_spins, dtype=np.int8)


def reconstruct_best_agreement(row: pd.Series, geometry: str) -> float:
    try:
        from step9_common import qiskit_bitstring_to_spins
    except ImportError as exc:
        raise RuntimeError(
            "Run from crossing_fibres_step7 so step9_common.py is importable."
        ) from exc

    graph, exact_spins = build_graph_for_agreement(int(row["seed"]), geometry)
    if "best_bitstring" not in row or pd.isna(row["best_bitstring"]):
        raise ValueError(
            f"Seed {row['seed']}: best_agreement_with_exact and best_bitstring "
            "are both unavailable."
        )
    bitstring = str(row["best_bitstring"]).strip().zfill(int(graph.n_nodes))
    measured = np.asarray(
        qiskit_bitstring_to_spins(bitstring, int(graph.n_nodes)),
        dtype=np.int8,
    )
    return float(
        max(
            np.mean(measured == exact_spins),
            np.mean(-measured == exact_spins),
        )
    )


def prepare_condition(
    frame: pd.DataFrame,
    *,
    key: str,
    label: str,
    geometry: str,
    strategy: str,
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    selected = frame[
        frame["seed"].isin(seeds)
        & (frame["angle_strategy"].astype(str) == strategy)
    ].copy()

    counts = selected.groupby("seed").size()
    missing = [seed for seed in seeds if int(counts.get(seed, 0)) == 0]
    duplicated = {
        int(seed): int(count)
        for seed, count in counts.items()
        if int(count) != 1
    }
    if missing or duplicated:
        raise ValueError(
            f"{label}: expected one {strategy!r} row for every seed {list(seeds)}. "
            f"Missing={missing}, non-unique={duplicated}"
        )

    selected = selected.set_index("seed").loc[list(seeds)].reset_index()

    required = {
        "exact_solution_label_accuracy",
        "ideal_expected_normalized_gap",
        "mean_normalized_gap",
        "cvar10_normalized_gap",
    }
    absent = sorted(required.difference(selected.columns))
    if absent:
        raise ValueError(f"{label}: missing required columns {absent}")

    if (
        "best_agreement_with_exact" not in selected.columns
        or selected["best_agreement_with_exact"].isna().any()
    ):
        selected["best_agreement_with_exact"] = [
            reconstruct_best_agreement(row, geometry)
            for _, row in selected.iterrows()
        ]

    selected["condition_key"] = key
    selected["condition"] = label.replace("\n", " ")
    selected["geometry_expected"] = geometry
    return selected


def make_long_table(args: argparse.Namespace) -> pd.DataFrame:
    original = read_results_many(
        [args.original_heldout, args.original_replication]
    )
    source_frames = {
        "90_original": original,
        "90_rotated": read_results_many(args.rotated90),
        "60_original": read_results_many(args.normal60),
        "60_rotated": read_results_many(args.rotated60),
    }

    seeds = tuple(int(seed) for seed in args.seeds)
    prepared = []
    for key, label, geometry, _x in CONDITIONS:
        prepared.append(
            prepare_condition(
                source_frames[key],
                key=key,
                label=label,
                geometry=geometry,
                strategy=args.strategy,
                seeds=seeds,
            )
        )
    return pd.concat(prepared, ignore_index=True)


def save_figure(
    figure: plt.Figure,
    output_base: Path,
    formats: Iterable[str],
    dpi: int,
) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        figure.savefig(
            output_base.with_suffix(f".{extension}"),
            dpi=dpi,
            bbox_inches="tight",
        )


def adjusted_panel_specs(qpu_metric: str) -> list[dict]:
    specs = [dict(spec) for spec in PANEL_SPECS]
    if qpu_metric == "cvar10_normalized_gap":
        title = "Low-energy QPU sampling performance"
        ylabel = "CVaR$_{10}$ normalized gap"
    else:
        title = "Overall QPU sampling performance"
        ylabel = "Mean normalized gap"

    for spec in specs:
        if spec["label"] == "C":
            spec["metric"] = qpu_metric
            spec["title"] = title
            spec["ylabel"] = ylabel
    return specs


def draw_metric_panel(
    ax: plt.Axes,
    table: pd.DataFrame,
    spec: dict,
    seeds: tuple[int, ...],
) -> None:
    metric = str(spec["metric"])
    x_by_key = {key: float(x) for key, _label, _geometry, x in CONDITIONS}
    label_by_key = {key: label for key, label, _geometry, _x in CONDITIONS}

    # Paired lines show the effect of global rotation within each angle.
    for seed in seeds:
        for pair in (("90_original", "90_rotated"), ("60_original", "60_rotated")):
            y_values = []
            x_values = []
            for key in pair:
                row = table[
                    (table["condition_key"] == key) & (table["seed"] == seed)
                ]
                y_values.append(float(row.iloc[0][metric]))
                x_values.append(x_by_key[key])
            ax.plot(
                x_values,
                y_values,
                color="#B8B8B8",
                linewidth=0.7,
                alpha=0.42,
                zorder=1,
            )

    # Seed observations and mean with 95% confidence interval.
    for key, _label, _geometry, x in CONDITIONS:
        values = (
            table.loc[table["condition_key"] == key, metric]
            .astype(float)
            .to_numpy()
        )
        colour = CONDITION_COLOURS[key]

        # Small deterministic horizontal offsets keep coincident values visible.
        offsets = np.linspace(-0.07, 0.07, len(values))
        ax.scatter(
            np.full(len(values), x) + offsets,
            values,
            s=34,
            color=colour,
            edgecolor="white",
            linewidth=0.6,
            alpha=0.9,
            zorder=3,
        )

        mean = float(np.mean(values))
        sd = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        ci95 = (
            float(t.ppf(0.975, len(values) - 1) * sd / np.sqrt(len(values)))
            if len(values) > 1 else 0.0
        )
        ax.errorbar(
            x,
            mean,
            yerr=ci95,
            fmt="D",
            markersize=6.2,
            markerfacecolor=colour,
            markeredgecolor="#111111",
            markeredgewidth=0.8,
            ecolor="#111111",
            elinewidth=1.1,
            capsize=3.0,
            zorder=4,
        )

    ax.axvline(1.8, color="#D5D5D5", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(
        [x for _key, _label, _geometry, x in CONDITIONS],
        [label for _key, label, _geometry, _x in CONDITIONS],
    )
    ax.set_xlim(-0.42, 4.02)
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_title(str(spec["title"]), loc="left", pad=8, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    values_all = table[metric].astype(float).to_numpy()
    if bool(spec["accuracy"]):
        lower = max(0.0, np.floor((float(np.min(values_all)) - 0.08) * 10.0) / 10.0)
        ax.set_ylim(lower, 1.025)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    else:
        upper = max(0.1, float(np.max(values_all)) * 1.18)
        ax.set_ylim(0.0, upper)

    ax.text(
        0.99,
        0.98,
        str(spec["direction"]),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.8,
        color="#555555",
    )
    ax.text(
        -0.11,
        1.08,
        str(spec["label"]),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=13,
        fontweight="bold",
    )


def summary_table(long_table: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    rows = []
    for key, label, geometry, _x in CONDITIONS:
        subset = long_table[long_table["condition_key"] == key]
        for spec in specs:
            values = subset[str(spec["metric"])].astype(float)
            rows.append(
                {
                    "condition_key": key,
                    "condition": label.replace("\n", " "),
                    "geometry": geometry,
                    "panel": spec["label"],
                    "metric": spec["metric"],
                    "n_seeds": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "sem": float(values.sem(ddof=1)),
                    "ci95_low": float(
                        values.mean()
                        - t.ppf(0.975, int(values.size) - 1) * values.sem(ddof=1)
                    ),
                    "ci95_high": float(
                        values.mean()
                        + t.ppf(0.975, int(values.size) - 1) * values.sem(ddof=1)
                    ),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def paired_rotation_table(long_table: pd.DataFrame, specs: list[dict]) -> pd.DataFrame:
    rows = []
    for angle, original_key, rotated_key in (
        ("90°", "90_original", "90_rotated"),
        ("60°", "60_original", "60_rotated"),
    ):
        for spec in specs:
            metric = str(spec["metric"])
            original = (
                long_table[long_table["condition_key"] == original_key]
                .set_index("seed")[metric]
                .astype(float)
            )
            rotated = (
                long_table[long_table["condition_key"] == rotated_key]
                .set_index("seed")[metric]
                .astype(float)
            )
            difference = rotated - original
            rows.append(
                {
                    "angle": angle,
                    "panel": spec["label"],
                    "metric": metric,
                    "difference_definition": "rotated minus original",
                    "n_pairs": int(difference.size),
                    "mean_difference": float(difference.mean()),
                    "sd_difference": float(difference.std(ddof=1)),
                    "sem_difference": float(difference.sem(ddof=1)),
                    "ci95_low": float(
                        difference.mean()
                        - t.ppf(0.975, int(difference.size) - 1)
                        * difference.sem(ddof=1)
                    ),
                    "ci95_high": float(
                        difference.mean()
                        + t.ppf(0.975, int(difference.size) - 1)
                        * difference.sem(ddof=1)
                    ),
                    "seed_differences": json.dumps(
                        {str(int(seed)): float(value) for seed, value in difference.items()}
                    ),
                }
            )
    return pd.DataFrame(rows)


def caption_text(qpu_metric: str, strategy: str, seeds: tuple[int, ...]) -> str:
    panel_c = (
        "QPU CVaR10 normalized gap"
        if qpu_metric == "cvar10_normalized_gap"
        else "QPU mean normalized gap"
    )
    return (
        "Controlled phantom difficulty and orientation generalization. "
        "The original and globally 45-degree-rotated 90-degree and 60-degree "
        "crossing-fiber phantoms were evaluated using the same selected hardware "
        f"seeds ({', '.join(str(seed) for seed in seeds)}) and the {strategy.replace('_', ' ')} "
        "p=1 angle strategy. Panels show (A) agreement of the exact MILP solution "
        "with known phantom labels, (B) ideal p=1 QAOA expected normalized gap, "
        f"(C) {panel_c}, and (D) agreement of the best measured QPU assignment "
        "with the exact MILP solution. Small circles represent individual seeds, "
        "grey lines pair original and rotated instances of the same seed, and "
        "diamonds with error bars show mean and 95% confidence intervals. Lower normalized gaps and "
        "higher agreements indicate better performance."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-heldout",
        type=Path,
        default=Path("heldout_hardware_results.csv"),
    )
    parser.add_argument(
        "--original-replication",
        type=Path,
        default=Path("replication_hardware_results.csv"),
    )
    parser.add_argument(
        "--rotated90",
        nargs="+",
        type=Path,
        default=[
            Path("rotated90_qpu_preflight/rotated90_hardware_results.csv"),
            Path("rotated90_qpu_remaining/rotated90_hardware_results.csv"),
        ],
        help="One or more non-overlapping rotated-90 hardware result CSVs.",
    )
    parser.add_argument(
        "--normal60",
        nargs="+",
        type=Path,
        default=[
            Path("curved60_qpu_preflight/curved60_hardware_results.csv"),
            Path("curved60_qpu_remaining/curved60_hardware_results.csv"),
        ],
        help="One or more non-overlapping normal-60 hardware result CSVs.",
    )
    parser.add_argument(
        "--rotated60",
        nargs="+",
        type=Path,
        default=[
            Path("curved60rot45_qpu_preflight/curved60rot45_hardware_results.csv"),
            Path("curved60rot45_qpu_remaining/curved60rot45_hardware_results.csv"),
        ],
        help="One or more non-overlapping rotated-60 hardware result CSVs.",
    )
    parser.add_argument(
        "--strategy",
        default="hardware_calibrated",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(2001, 2016)),
    )
    parser.add_argument(
        "--qpu-gap-metric",
        choices=["cvar10_normalized_gap", "mean_normalized_gap"],
        default="cvar10_normalized_gap",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper_figures_phantom_summary"),
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--latex", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_matplotlib(args.latex)
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = tuple(int(seed) for seed in args.seeds)
    long_table = make_long_table(args)
    specs = adjusted_panel_specs(args.qpu_gap_metric)

    # Export source values before plotting.
    export_columns = [
        "condition_key",
        "condition",
        "geometry_expected",
        "seed",
        "angle_strategy",
        "exact_solution_label_accuracy",
        "ideal_expected_normalized_gap",
        "mean_normalized_gap",
        "cvar10_normalized_gap",
        "best_agreement_with_exact",
    ]
    long_table[export_columns].to_csv(
        args.out / "phantom_summary_seed_values.csv",
        index=False,
    )
    means = summary_table(long_table, specs)
    means.to_csv(args.out / "phantom_summary_means_95ci.csv", index=False)
    paired = paired_rotation_table(long_table, specs)
    paired.to_csv(args.out / "phantom_summary_rotation_differences.csv", index=False)

    # Individual panels.
    for spec in specs:
        figure, axis = plt.subplots(figsize=(4.8, 3.8))
        draw_metric_panel(axis, long_table, spec, seeds)
        save_figure(
            figure,
            args.out / str(spec["filename"]),
            args.formats,
            args.dpi,
        )
        plt.close(figure)

    # Combined 2x2 manuscript figure.
    figure, axes = plt.subplots(
        2,
        2,
        figsize=(10.2, 7.7),
    )
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.145,
        top=0.900,
        wspace=0.260,
        hspace=0.300,
    )
    for axis, spec in zip(axes.flat, specs):
        draw_metric_panel(axis, long_table, spec, seeds)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="#777777",
            markeredgecolor="white",
            markersize=6.5,
            label="Individual seed",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="-",
            color="#111111",
            markerfacecolor="#777777",
            markeredgecolor="#111111",
            markersize=6.5,
            label="Mean ± 95% CI",
        ),
        Line2D(
            [0],
            [0],
            color="#B8B8B8",
            linewidth=1.0,
            label="Paired rotation comparison",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    figure.suptitle(
        "Controlled phantom difficulty and orientation generalization",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(
        figure,
        args.out / "figure_phantom_difficulty_orientation_summary",
        args.formats,
        args.dpi,
    )
    plt.close(figure)

    caption = caption_text(args.qpu_gap_metric, args.strategy, seeds)
    (args.out / "figure_caption.txt").write_text(caption + "\n", encoding="utf-8")

    metadata = {
        "strategy": args.strategy,
        "seeds": list(seeds),
        "qpu_gap_metric": args.qpu_gap_metric,
        "conditions": [
            {"key": key, "label": label.replace("\n", " "), "geometry": geometry}
            for key, label, geometry, _x in CONDITIONS
        ],
        "outputs": {
            "combined_figure": "figure_phantom_difficulty_orientation_summary",
            "seed_values": "phantom_summary_seed_values.csv",
            "means_95ci": "phantom_summary_means_95ci.csv",
            "paired_rotation_differences": "phantom_summary_rotation_differences.csv",
            "caption": "figure_caption.txt",
        },
    }
    (args.out / "figure_summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    display = means.pivot(
        index=["condition"],
        columns="metric",
        values="mean",
    )
    print("\nCondition means:")
    print(display.to_string())
    print(f"\nFigures and editable data written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
