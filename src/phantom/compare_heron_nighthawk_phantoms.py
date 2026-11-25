#!/usr/bin/env python3
"""Formal Heron-versus-Nighthawk comparison across four phantom conditions.

The analysis treats each backend as a complete backend-tailored workflow:
  * Heron / ibm_aachen: frozen `hardware_calibrated` angles
  * Nighthawk / ibm_berlin: frozen `nighthawk_calibrated` angles

The same seeds (2001-2015) and graph instances are paired across backends.
This is therefore an end-to-end workflow comparison, not an isolated causal
comparison of processor architectures because backend, native gate mode,
calibration epoch, layout, and angle pair differ.

Outputs
-------
Figures:
  figure_heron_vs_nighthawk_phantoms.{png,pdf,svg}
  panel_A_mean_gap.{png,pdf,svg}
  panel_B_hardware_penalty.{png,pdf,svg}
  panel_C_cvar10_gap.{png,pdf,svg}
  panel_D_best_agreement.{png,pdf,svg}

Tables:
  backend_comparison_seed_values.csv
  backend_comparison_descriptives.csv
  backend_comparison_paired_contrasts.csv
  backend_comparison_rm_anova.csv
  backend_comparison_primary_table.csv
  backend_comparison_primary_table.tex

Text/metadata:
  backend_comparison_results.md
  figure_caption.txt
  backend_comparison_summary.json

Run from crossing_fibres_step7 beside:
  make_phantom_difficulty_summary_figure_all15.py
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.anova import AnovaRM
from statsmodels.stats.multitest import multipletests

try:
    import make_phantom_difficulty_summary_figure_all15 as base
except ImportError as exc:
    raise SystemExit(
        "Could not import make_phantom_difficulty_summary_figure_all15.py. "
        "Place this script beside the established all-15 Heron plotter and "
        "run it from crossing_fibres_step7."
    ) from exc


BACKENDS = ("Heron", "Nighthawk")
BACKEND_COLOURS = {
    "Heron": "#0072B2",
    "Nighthawk": "#D55E00",
}
BACKEND_MARKERS = {
    "Heron": "o",
    "Nighthawk": "s",
}

CONDITIONS = tuple(base.CONDITIONS)
CONDITION_ORDER = tuple(item[0] for item in CONDITIONS)
CONDITION_META = {
    key: {
        "label": label.replace("\n", " "),
        "plot_label": label,
        "geometry": geometry,
        "x": float(x),
        "crossing_angle": "90" if key.startswith("90") else "60",
        "orientation": "rotated" if key.endswith("rotated") else "original",
    }
    for key, label, geometry, x in CONDITIONS
}

PANEL_SPECS = (
    {
        "label": "A",
        "metric": "mean_normalized_gap",
        "title": "Overall QPU sampling performance",
        "ylabel": "QPU Mean normalized gap ",
        "accuracy": False,
        "filename": "panel_A_mean_gap",
        "direction": "Lower is better",
    },
    {
        "label": "B",
        "metric": "hardware_minus_ideal_expected_gap",
        "title": "Hardware execution penalty",
        "ylabel": "Hardware minus ideal expected gap",
        "accuracy": False,
        "filename": "panel_B_hardware_penalty",
        "direction": "Lower is better",
    },
    {
        "label": "C",
        "metric": "cvar10_normalized_gap",
        "title": "Low-energy QPU sampling performance",
        "ylabel": "CVaR$_{10}$ normalized gap",
        "accuracy": False,
        "filename": "panel_C_cvar10_gap",
        "direction": "Lower is better",
    },
    {
        "label": "D",
        "metric": "best_agreement_with_exact",
        "title": "Best measured QPU assignment",
        "ylabel": "Agreement with exact MILP",
        "accuracy": True,
        "filename": "panel_D_best_agreement",
        "direction": "Higher is better",
    },
)

ANALYSIS_METRICS = (
    "mean_normalized_gap",
    "hardware_minus_ideal_expected_gap",
    "cvar10_normalized_gap",
    "best_normalized_gap",
    "best_agreement_with_exact",
    "best_label_accuracy",
    "ideal_expected_normalized_gap",
)


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return repr(value)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    # Heron inputs.
    parser.add_argument("--heron-original-heldout", type=Path, required=True)
    parser.add_argument("--heron-original-replication", type=Path, required=True)
    parser.add_argument("--heron-rotated90", nargs="+", type=Path, required=True)
    parser.add_argument("--heron-normal60", nargs="+", type=Path, required=True)
    parser.add_argument("--heron-rotated60", nargs="+", type=Path, required=True)
    parser.add_argument("--heron-strategy", default="hardware_calibrated")

    # Nighthawk inputs.
    parser.add_argument("--nighthawk-original90", nargs="+", type=Path, required=True)
    parser.add_argument("--nighthawk-rotated90", nargs="+", type=Path, required=True)
    parser.add_argument("--nighthawk-normal60", nargs="+", type=Path, required=True)
    parser.add_argument("--nighthawk-rotated60", nargs="+", type=Path, required=True)
    parser.add_argument("--nighthawk-strategy", default="nighthawk_calibrated")

    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(range(2001, 2016)),
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["png", "pdf", "svg"],
        choices=["png", "pdf", "svg"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--latex", action="store_true")
    return parser.parse_args()


def read_backend_conditions(
    *,
    backend: str,
    strategy: str,
    original_paths: Iterable[Path],
    rotated90_paths: Iterable[Path],
    normal60_paths: Iterable[Path],
    rotated60_paths: Iterable[Path],
    seeds: tuple[int, ...],
) -> pd.DataFrame:
    source_frames = {
        "90_original": base.read_results_many(original_paths),
        "90_rotated": base.read_results_many(rotated90_paths),
        "60_original": base.read_results_many(normal60_paths),
        "60_rotated": base.read_results_many(rotated60_paths),
    }

    prepared: list[pd.DataFrame] = []
    for key, label, geometry, _x in CONDITIONS:
        selected = base.prepare_condition(
            source_frames[key],
            key=key,
            label=label,
            geometry=geometry,
            strategy=strategy,
            seeds=seeds,
        )
        selected["backend"] = backend
        selected["backend_strategy"] = strategy
        selected["crossing_angle"] = CONDITION_META[key]["crossing_angle"]
        selected["orientation"] = CONDITION_META[key]["orientation"]

        if "best_normalized_gap" not in selected.columns:
            raise ValueError(
                f"{backend}, {key}: best_normalized_gap is required"
            )
        if "best_label_accuracy" not in selected.columns:
            raise ValueError(
                f"{backend}, {key}: best_label_accuracy is required"
            )

        if (
            "hardware_minus_ideal_expected_gap" not in selected.columns
            or selected["hardware_minus_ideal_expected_gap"].isna().any()
        ):
            selected["hardware_minus_ideal_expected_gap"] = (
                selected["mean_normalized_gap"].astype(float)
                - selected["ideal_expected_normalized_gap"].astype(float)
            )

        prepared.append(selected)

    return pd.concat(prepared, ignore_index=True)


def validate_balanced_design(frame: pd.DataFrame, seeds: tuple[int, ...]) -> None:
    expected = {
        (backend, condition, seed)
        for backend in BACKENDS
        for condition in CONDITION_ORDER
        for seed in seeds
    }
    observed = {
        (str(row.backend), str(row.condition_key), int(row.seed))
        for row in frame.itertuples()
    }
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    duplicates = (
        frame.groupby(["backend", "condition_key", "seed"])
        .size()
        .reset_index(name="n")
    )
    duplicates = duplicates.loc[duplicates["n"] != 1]

    if missing or extra or not duplicates.empty:
        raise ValueError(
            "Backend comparison design is not balanced.\n"
            f"Missing: {missing[:20]}\n"
            f"Extra: {extra[:20]}\n"
            f"Non-unique:\n{duplicates.to_string(index=False)}"
        )


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
            dpi=dpi if extension == "png" else None,
            bbox_inches="tight",
        )


def mean_ci(values: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return mean, mean, mean
    half_width = float(
        stats.t.ppf(0.975, len(values) - 1)
        * stats.sem(values)
    )
    return mean, mean - half_width, mean + half_width


def draw_backend_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    spec: dict[str, Any],
    seeds: tuple[int, ...],
) -> None:
    metric = str(spec["metric"])
    backend_offset = {"Heron": -0.16, "Nighthawk": 0.16}

    # Seed-paired backend comparisons within each condition.
    for key in CONDITION_ORDER:
        center = CONDITION_META[key]["x"]
        for seed in seeds:
            values = []
            x_values = []
            for backend in BACKENDS:
                row = frame.loc[
                    (frame["condition_key"] == key)
                    & (frame["backend"] == backend)
                    & (frame["seed"] == seed),
                    metric,
                ]
                values.append(float(row.iloc[0]))
                x_values.append(center + backend_offset[backend])
            ax.plot(
                x_values,
                values,
                color="#B8B8B8",
                linewidth=0.65,
                alpha=0.36,
                zorder=1,
            )

    for key in CONDITION_ORDER:
        center = CONDITION_META[key]["x"]
        for backend in BACKENDS:
            subset = frame.loc[
                (frame["condition_key"] == key)
                & (frame["backend"] == backend)
            ].sort_values("seed")
            values = subset[metric].astype(float).to_numpy()
            x = center + backend_offset[backend]

            # Deterministic tiny offsets preserve visibility without changing
            # the backend pairing structure.
            offsets = np.linspace(-0.045, 0.045, len(values))
            ax.scatter(
                np.full(len(values), x) + offsets,
                values,
                s=30,
                marker=BACKEND_MARKERS[backend],
                color=BACKEND_COLOURS[backend],
                edgecolor="white",
                linewidth=0.55,
                alpha=0.88,
                zorder=3,
            )

            mean, ci_low, ci_high = mean_ci(values)
            ax.errorbar(
                x,
                mean,
                yerr=[[mean - ci_low], [ci_high - mean]],
                fmt="D",
                markersize=6.0,
                markerfacecolor=BACKEND_COLOURS[backend],
                markeredgecolor="#111111",
                markeredgewidth=0.8,
                ecolor="#111111",
                elinewidth=1.0,
                capsize=2.8,
                zorder=4,
            )

    ax.axvline(1.8, color="#D5D5D5", linewidth=0.8, linestyle="--", zorder=0)
    ax.set_xticks(
        [CONDITION_META[key]["x"] for key in CONDITION_ORDER],
        [CONDITION_META[key]["plot_label"] for key in CONDITION_ORDER],
    )
    ax.set_xlim(-0.48, 4.08)
    ax.set_ylabel(str(spec["ylabel"]))
    ax.set_title(str(spec["title"]), loc="left", pad=8, fontweight="bold")
    ax.grid(axis="y", color="#E5E5E5", linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    all_values = frame[metric].astype(float).to_numpy()
    if bool(spec["accuracy"]):
        lower = max(
            0.0,
            np.floor((float(np.min(all_values)) - 0.08) * 10.0) / 10.0,
        )
        ax.set_ylim(lower, 1.025)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    else:
        minimum = float(np.min(all_values))
        maximum = float(np.max(all_values))
        spread = max(maximum - minimum, 0.02)
        lower = max(0.0, minimum - 0.18 * spread)
        upper = maximum + 0.22 * spread
        ax.set_ylim(lower, upper)

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


def descriptive_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in ANALYSIS_METRICS:
        for (backend, condition), subset in frame.groupby(
            ["backend", "condition_key"],
            sort=False,
        ):
            values = subset[metric].astype(float).to_numpy()
            mean, ci_low, ci_high = mean_ci(values)
            rows.append(
                {
                    "metric": metric,
                    "backend": backend,
                    "condition_key": condition,
                    "condition": CONDITION_META[str(condition)]["label"],
                    "n": int(len(values)),
                    "mean": mean,
                    "sd": float(np.std(values, ddof=1)),
                    "median": float(np.median(values)),
                    "ci95_low": ci_low,
                    "ci95_high": ci_high,
                    "minimum": float(np.min(values)),
                    "maximum": float(np.max(values)),
                }
            )
    return pd.DataFrame(rows)


def paired_stats(difference: pd.Series) -> dict[str, Any]:
    values = difference.dropna().astype(float).to_numpy()
    n = int(len(values))
    mean = float(np.mean(values))
    sd = float(np.std(values, ddof=1))
    se = float(stats.sem(values))
    ci_low, ci_high = stats.t.interval(
        0.95,
        df=n - 1,
        loc=mean,
        scale=se,
    )
    t_result = stats.ttest_1samp(values, popmean=0.0)
    try:
        w_result = stats.wilcoxon(
            values,
            zero_method="wilcox",
            correction=False,
            alternative="two-sided",
            method="auto",
        )
        w_statistic = float(w_result.statistic)
        w_p = float(w_result.pvalue)
    except ValueError:
        w_statistic = math.nan
        w_p = math.nan

    return {
        "n_pairs": n,
        "mean_difference": mean,
        "sd_difference": sd,
        "se_difference": se,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "t_statistic": float(t_result.statistic),
        "t_df": n - 1,
        "t_p": float(t_result.pvalue),
        "cohen_dz": float(mean / sd) if sd > 0 else math.nan,
        "wilcoxon_statistic": w_statistic,
        "wilcoxon_p": w_p,
        "n_nighthawk_better": int(np.sum(values < 0)),
        "n_heron_better": int(np.sum(values > 0)),
        "n_ties": int(np.sum(np.isclose(values, 0.0, atol=1e-12))),
    }


def paired_contrast_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for metric in ANALYSIS_METRICS:
        pivot = frame.pivot_table(
            index="seed",
            columns=["backend", "condition_key"],
            values=metric,
            aggfunc="first",
        )

        condition_differences: dict[str, pd.Series] = {}
        for condition in CONDITION_ORDER:
            difference = (
                pivot[("Nighthawk", condition)]
                - pivot[("Heron", condition)]
            )
            condition_differences[condition] = difference
            rows.append(
                {
                    "metric": metric,
                    "contrast": f"backend_at_{condition}",
                    "condition_key": condition,
                    "condition": CONDITION_META[condition]["label"],
                    "difference_definition": "Nighthawk minus Heron",
                    **paired_stats(difference),
                }
            )

        average_difference = pd.concat(
            condition_differences,
            axis=1,
        ).mean(axis=1)
        rows.append(
            {
                "metric": metric,
                "contrast": "backend_average_across_four_conditions",
                "condition_key": "average",
                "condition": "Average across four phantom conditions",
                "difference_definition": "Nighthawk minus Heron",
                **paired_stats(average_difference),
            }
        )

    result = pd.DataFrame(rows)
    result["t_p_holm_within_metric"] = math.nan
    result["wilcoxon_p_holm_within_metric"] = math.nan

    for metric, indices in result.groupby("metric").groups.items():
        indices = list(indices)
        _, adjusted_t, _, _ = multipletests(
            result.loc[indices, "t_p"].to_numpy(dtype=float),
            method="holm",
        )
        result.loc[indices, "t_p_holm_within_metric"] = adjusted_t

        raw_w = result.loc[indices, "wilcoxon_p"].to_numpy(dtype=float)
        finite = np.isfinite(raw_w)
        adjusted_w = np.full(len(indices), np.nan)
        if finite.any():
            _, adjusted_finite, _, _ = multipletests(
                raw_w[finite],
                method="holm",
            )
            adjusted_w[finite] = adjusted_finite
        result.loc[indices, "wilcoxon_p_holm_within_metric"] = adjusted_w

    return result


def repeated_measures_anova(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in ANALYSIS_METRICS:
        analysis = frame[
            [
                "seed",
                "backend",
                "crossing_angle",
                "orientation",
                metric,
            ]
        ].dropna()

        fitted = AnovaRM(
            analysis,
            depvar=metric,
            subject="seed",
            within=["backend", "crossing_angle", "orientation"],
        ).fit()

        table = fitted.anova_table.reset_index().rename(
            columns={"index": "effect"}
        )
        for _, row in table.iterrows():
            f_value = float(row["F Value"])
            num_df = float(row["Num DF"])
            den_df = float(row["Den DF"])
            partial_eta_squared = (
                f_value * num_df
                / (f_value * num_df + den_df)
            )
            rows.append(
                {
                    "metric": metric,
                    "effect": str(row["effect"]),
                    "f_value": f_value,
                    "numerator_df": num_df,
                    "denominator_df": den_df,
                    "p_value": float(row["Pr > F"]),
                    "partial_eta_squared": float(partial_eta_squared),
                }
            )

    result = pd.DataFrame(rows)
    result["p_holm_within_metric"] = math.nan
    for metric, indices in result.groupby("metric").groups.items():
        indices = list(indices)
        _, adjusted, _, _ = multipletests(
            result.loc[indices, "p_value"].to_numpy(dtype=float),
            method="holm",
        )
        result.loc[indices, "p_holm_within_metric"] = adjusted
    return result


def format_p(value: float) -> str:
    if not np.isfinite(value):
        return "NA"
    if value < 0.001:
        return f"{value:.2e}"
    return f"{value:.3f}"


def primary_table(
    descriptives: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for metric in (
        "mean_normalized_gap",
        "hardware_minus_ideal_expected_gap",
        "cvar10_normalized_gap",
        "best_agreement_with_exact",
    ):
        for condition in (*CONDITION_ORDER, "average"):
            if condition == "average":
                contrast_name = "backend_average_across_four_conditions"
                label = "Average"
                heron_mean = (
                    descriptives.loc[
                        (descriptives["metric"] == metric)
                        & (descriptives["backend"] == "Heron"),
                        "mean",
                    ].mean()
                )
                nighthawk_mean = (
                    descriptives.loc[
                        (descriptives["metric"] == metric)
                        & (descriptives["backend"] == "Nighthawk"),
                        "mean",
                    ].mean()
                )
            else:
                contrast_name = f"backend_at_{condition}"
                label = CONDITION_META[condition]["label"]
                heron_mean = float(
                    descriptives.loc[
                        (descriptives["metric"] == metric)
                        & (descriptives["backend"] == "Heron")
                        & (descriptives["condition_key"] == condition),
                        "mean",
                    ].iloc[0]
                )
                nighthawk_mean = float(
                    descriptives.loc[
                        (descriptives["metric"] == metric)
                        & (descriptives["backend"] == "Nighthawk")
                        & (descriptives["condition_key"] == condition),
                        "mean",
                    ].iloc[0]
                )

            contrast = contrasts.loc[
                (contrasts["metric"] == metric)
                & (contrasts["contrast"] == contrast_name)
            ].iloc[0]
            rows.append(
                {
                    "metric": metric,
                    "condition": label,
                    "heron_mean": float(heron_mean),
                    "nighthawk_mean": float(nighthawk_mean),
                    "nighthawk_minus_heron": float(
                        contrast["mean_difference"]
                    ),
                    "ci95_low": float(contrast["ci95_low"]),
                    "ci95_high": float(contrast["ci95_high"]),
                    "paired_t_p": float(contrast["t_p"]),
                    "paired_t_p_holm": float(
                        contrast["t_p_holm_within_metric"]
                    ),
                    "cohen_dz": float(contrast["cohen_dz"]),
                }
            )
    return pd.DataFrame(rows)


def result_text(
    anova: pd.DataFrame,
    contrasts: pd.DataFrame,
) -> str:
    def anova_row(metric: str, effect: str) -> pd.Series:
        return anova.loc[
            (anova["metric"] == metric)
            & (anova["effect"] == effect)
        ].iloc[0]

    def contrast_row(metric: str, contrast: str) -> pd.Series:
        return contrasts.loc[
            (contrasts["metric"] == metric)
            & (contrasts["contrast"] == contrast)
        ].iloc[0]

    mean_backend = anova_row("mean_normalized_gap", "backend")
    mean_average = contrast_row(
        "mean_normalized_gap",
        "backend_average_across_four_conditions",
    )
    penalty_backend = anova_row(
        "hardware_minus_ideal_expected_gap",
        "backend",
    )
    penalty_average = contrast_row(
        "hardware_minus_ideal_expected_gap",
        "backend_average_across_four_conditions",
    )
    cvar_backend = anova_row("cvar10_normalized_gap", "backend")
    agreement_backend = anova_row(
        "best_agreement_with_exact",
        "backend",
    )

    lines = [
        "# Heron-versus-Nighthawk phantom comparison",
        "",
        "This comparison evaluates the complete backend-tailored workflows. "
        "The Nighthawk-minus-Heron difference is paired by seed and phantom "
        "condition; negative gap differences favor Nighthawk, whereas positive "
        "agreement differences favor Nighthawk.",
        "",
        "## Primary end-to-end outcome",
        "",
        (
            "For mean normalized gap, the repeated-measures backend effect was "
            f"F({mean_backend['numerator_df']:.0f},"
            f"{mean_backend['denominator_df']:.0f})="
            f"{mean_backend['f_value']:.2f}, "
            f"raw p={format_p(float(mean_backend['p_value']))}, "
            f"Holm-adjusted p="
            f"{format_p(float(mean_backend['p_holm_within_metric']))}, "
            f"partial eta-squared="
            f"{mean_backend['partial_eta_squared']:.3f}. "
            "Averaged across the four conditions, the paired Nighthawk-minus-"
            f"Heron difference was {mean_average['mean_difference']:.5f} "
            f"(95% CI {mean_average['ci95_low']:.5f} to "
            f"{mean_average['ci95_high']:.5f}; "
            f"Holm-adjusted p="
            f"{format_p(float(mean_average['t_p_holm_within_metric']))}; "
            f"Cohen's dz={mean_average['cohen_dz']:.2f})."
        ),
        "",
        "## Hardware execution penalty",
        "",
        (
            "For hardware-minus-ideal expected gap, the backend effect was "
            f"F({penalty_backend['numerator_df']:.0f},"
            f"{penalty_backend['denominator_df']:.0f})="
            f"{penalty_backend['f_value']:.2f}, "
            f"raw p={format_p(float(penalty_backend['p_value']))}, "
            f"Holm-adjusted p="
            f"{format_p(float(penalty_backend['p_holm_within_metric']))}, "
            f"partial eta-squared="
            f"{penalty_backend['partial_eta_squared']:.3f}. "
            "The average paired Nighthawk-minus-Heron difference was "
            f"{penalty_average['mean_difference']:.5f} "
            f"(95% CI {penalty_average['ci95_low']:.5f} to "
            f"{penalty_average['ci95_high']:.5f})."
        ),
        "",
        "## Secondary outcomes",
        "",
        (
            "The backend effect for CVaR-10% normalized gap was "
            f"F({cvar_backend['numerator_df']:.0f},"
            f"{cvar_backend['denominator_df']:.0f})="
            f"{cvar_backend['f_value']:.2f}, "
            f"Holm-adjusted p="
            f"{format_p(float(cvar_backend['p_holm_within_metric']))}. "
            "For best-assignment agreement with the exact solution, the "
            "backend effect was "
            f"F({agreement_backend['numerator_df']:.0f},"
            f"{agreement_backend['denominator_df']:.0f})="
            f"{agreement_backend['f_value']:.2f}, "
            f"Holm-adjusted p="
            f"{format_p(float(agreement_backend['p_holm_within_metric']))}."
        ),
        "",
        "The full tables include backend-by-crossing-angle, backend-by-"
        "orientation and three-way interactions, condition-specific paired "
        "contrasts, Wilcoxon robustness tests and Holm-adjusted p-values.",
        "",
        "## Interpretation boundary",
        "",
        "Because each backend used its own frozen angle pair, native gate mode, "
        "calibration-aware layout and calibration epoch, these results compare "
        "the two optimized experimental workflows rather than estimating a "
        "processor-family effect in isolation.",
        "",
    ]
    return "\n".join(lines)


def caption_text(seeds: tuple[int, ...]) -> str:
    return (
        "Backend-tailored Heron-versus-Nighthawk comparison across controlled "
        "crossing-fiber phantoms. The original and globally 45-degree-rotated "
        "90-degree and 60-degree phantoms were evaluated using the same paired "
        f"seeds ({', '.join(map(str, seeds))}). Heron used its frozen "
        "hardware-calibrated p=1 angles and fractional-gate workflow; "
        "Nighthawk used its frozen backend-calibrated angles and standard-CZ "
        "workflow. Panels show (A) QPU mean normalized gap, (B) the additional "
        "hardware penalty relative to each backend's ideal p=1 expected gap, "
        "(C) QPU CVaR-10% normalized gap, and (D) agreement of the best "
        "measured assignment with the exact MILP solution. Small symbols are "
        "individual seeds, grey lines pair the same instance across backends, "
        "and diamonds with error bars show means and 95% confidence intervals. "
        "The comparison is an end-to-end workflow comparison rather than an "
        "isolated processor-architecture comparison."
    )


def main() -> None:
    args = parse_args()
    base.configure_matplotlib(args.latex)
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in args.seeds)

    heron = read_backend_conditions(
        backend="Heron",
        strategy=args.heron_strategy,
        original_paths=[
            args.heron_original_heldout,
            args.heron_original_replication,
        ],
        rotated90_paths=args.heron_rotated90,
        normal60_paths=args.heron_normal60,
        rotated60_paths=args.heron_rotated60,
        seeds=seeds,
    )
    nighthawk = read_backend_conditions(
        backend="Nighthawk",
        strategy=args.nighthawk_strategy,
        original_paths=args.nighthawk_original90,
        rotated90_paths=args.nighthawk_rotated90,
        normal60_paths=args.nighthawk_normal60,
        rotated60_paths=args.nighthawk_rotated60,
        seeds=seeds,
    )

    combined = pd.concat([heron, nighthawk], ignore_index=True)
    validate_balanced_design(combined, seeds)
    combined = combined.sort_values(
        ["seed", "condition_key", "backend"]
    ).reset_index(drop=True)

    export_columns = [
        "backend",
        "backend_strategy",
        "condition_key",
        "condition",
        "geometry_expected",
        "crossing_angle",
        "orientation",
        "seed",
        "exact_solution_label_accuracy",
        "ideal_expected_normalized_gap",
        "mean_normalized_gap",
        "hardware_minus_ideal_expected_gap",
        "cvar10_normalized_gap",
        "best_normalized_gap",
        "best_label_accuracy",
        "best_agreement_with_exact",
    ]
    combined[export_columns].to_csv(
        args.out / "backend_comparison_seed_values.csv",
        index=False,
    )

    descriptives = descriptive_table(combined)
    descriptives.to_csv(
        args.out / "backend_comparison_descriptives.csv",
        index=False,
    )

    contrasts = paired_contrast_table(combined)
    contrasts.to_csv(
        args.out / "backend_comparison_paired_contrasts.csv",
        index=False,
    )

    anova = repeated_measures_anova(combined)
    anova.to_csv(
        args.out / "backend_comparison_rm_anova.csv",
        index=False,
    )

    primary = primary_table(descriptives, contrasts)
    primary.to_csv(
        args.out / "backend_comparison_primary_table.csv",
        index=False,
    )
    (args.out / "backend_comparison_primary_table.tex").write_text(
        primary.to_latex(index=False, float_format="%.5f"),
        encoding="utf-8",
    )

    # Individual panels.
    for spec in PANEL_SPECS:
        figure, axis = plt.subplots(figsize=(4.8, 3.8))
        draw_backend_panel(axis, combined, dict(spec), seeds)
        save_figure(
            figure,
            args.out / str(spec["filename"]),
            args.formats,
            args.dpi,
        )
        plt.close(figure)

    # Combined figure uses the same dimensions and spacing language as the
    # established phantom summary figure.
    figure, axes = plt.subplots(2, 2, figsize=(10.2, 7.7))
    figure.subplots_adjust(
        left=0.085,
        right=0.985,
        bottom=0.145,
        top=0.900,
        wspace=0.260,
        hspace=0.300,
    )
    for axis, spec in zip(axes.flat, PANEL_SPECS):
        draw_backend_panel(axis, combined, dict(spec), seeds)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker=BACKEND_MARKERS["Heron"],
            linestyle="none",
            markerfacecolor=BACKEND_COLOURS["Heron"],
            markeredgecolor="white",
            markersize=6.5,
            label="Heron (backend-tailored)",
        ),
        Line2D(
            [0],
            [0],
            marker=BACKEND_MARKERS["Nighthawk"],
            linestyle="none",
            markerfacecolor=BACKEND_COLOURS["Nighthawk"],
            markeredgecolor="white",
            markersize=6.5,
            label="Nighthawk (backend-tailored)",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="-",
            color="#111111",
            markerfacecolor="#777777",
            markeredgecolor="#111111",
            markersize=6.3,
            label="Mean ± 95% CI",
        ),
        Line2D(
            [0],
            [0],
            color="#B8B8B8",
            linewidth=1.0,
            label="Same seed and condition",
        ),
    ]
    figure.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.025),
    )
    figure.suptitle(
        "Backend-tailored Heron versus Nighthawk performance",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(
        figure,
        args.out / "figure_heron_vs_nighthawk_phantoms",
        args.formats,
        args.dpi,
    )
    plt.close(figure)

    results_markdown = result_text(anova, contrasts)
    (args.out / "backend_comparison_results.md").write_text(
        results_markdown,
        encoding="utf-8",
    )
    caption = caption_text(seeds)
    (args.out / "figure_caption.txt").write_text(
        caption + "\n",
        encoding="utf-8",
    )

    metadata = {
        "design": {
            "type": "2x2x2 within-seed repeated measures",
            "factors": {
                "backend": list(BACKENDS),
                "crossing_angle": ["90", "60"],
                "orientation": ["original", "rotated"],
            },
            "seeds": list(seeds),
            "difference_definition": "Nighthawk minus Heron",
            "primary_outcome": "mean_normalized_gap",
            "hardware_penalty_outcome": (
                "mean_normalized_gap - ideal_expected_normalized_gap"
            ),
            "multiplicity_adjustment": "Holm within metric",
        },
        "workflow_definition": {
            "Heron": {
                "backend": "ibm_aachen",
                "strategy": args.heron_strategy,
                "gate_mode": "fractional",
            },
            "Nighthawk": {
                "backend": "ibm_berlin",
                "strategy": args.nighthawk_strategy,
                "gate_mode": "standard CZ",
            },
        },
        "interpretation_limit": (
            "Backend-tailored workflow comparison; not an isolated causal "
            "processor-family comparison."
        ),
        "outputs": {
            "figure": "figure_heron_vs_nighthawk_phantoms",
            "seed_values": "backend_comparison_seed_values.csv",
            "descriptives": "backend_comparison_descriptives.csv",
            "paired_contrasts": "backend_comparison_paired_contrasts.csv",
            "rm_anova": "backend_comparison_rm_anova.csv",
            "primary_table": "backend_comparison_primary_table.csv",
            "results_text": "backend_comparison_results.md",
            "caption": "figure_caption.txt",
        },
    }
    (args.out / "backend_comparison_summary.json").write_text(
        json.dumps(metadata, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print(results_markdown)
    print(f"\nWrote comparison package to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
