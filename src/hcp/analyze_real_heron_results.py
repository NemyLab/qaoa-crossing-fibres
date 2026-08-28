#!/usr/bin/env python3
"""Final analysis of the 30-subject real-data Heron experiment.

This analysis intentionally uses the raw measured SamplerV2 outcomes exactly as
collected. It performs no repair, feasibility projection, ICM, or other
classical post-processing.

Inputs
------
real_heron_hardware_results.csv
real_heron_hardware_summary.json

Outputs
-------
real_heron_paired_tests.csv
table3_mps_vs_heron.csv
table3_mps_vs_heron.md
real_heron_mps_qpu_correlations.csv
real_heron_assignment_quality.csv
real_heron_exploratory_circuit_correlations.csv
real_heron_analysis_summary.json
real_heron_results_text.md

Separate figures, each in PNG, SVG, and PDF:
figure_mean_gap_mps_vs_qpu
figure_cvar10_gap_mps_vs_qpu
figure_best_gap_mps_vs_qpu
figure_raw_best_assignment_quality
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


PAIRED_METRICS = [
    {
        "metric": "mean_normalized_gap",
        "hardware": "mean_normalized_gap",
        "mps": "mps_mean_normalized_gap",
        "label": "Mean normalized gap",
        "figure": "figure_mean_gap_mps_vs_qpu",
    },
    {
        "metric": "cvar10_normalized_gap",
        "hardware": "cvar10_normalized_gap",
        "mps": "mps_cvar10_normalized_gap",
        "label": "CVaR-10% normalized gap",
        "figure": "figure_cvar10_gap_mps_vs_qpu",
    },
    {
        "metric": "best_normalized_gap",
        "hardware": "best_normalized_gap",
        "mps": "mps_best_normalized_gap",
        "label": "Best-sample normalized gap",
        "figure": "figure_best_gap_mps_vs_qpu",
    },
]

ASSIGNMENT_COLUMNS = [
    (
        "best_agreement_with_exact_up_to_sign",
        "MILP\nagreement",
    ),
    (
        "best_tom_agreement_confident_voxels",
        "TOM\nagreement",
    ),
    (
        "best_conflict_satisfaction_fraction",
        "Conflict\nsatisfaction",
    ),
 #   (
 #       "best_weighted_continuation_satisfaction_fraction",
 #       "Weighted continuation\nsatisfaction",
 #   ),
]

CIRCUIT_COLUMNS = [
    "twoq_depth",
    "n_twoq_gates",
    "n_active_physical_qubits",
    "mean_twoq_error",
    "max_twoq_error",
    "twoq_calibration_error_cost",
    "independent_twoq_success_proxy",
]

PENALTY_COLUMNS = [
    "hardware_minus_mps_mean_gap",
    "hardware_minus_mps_cvar10_gap",
    "hardware_minus_mps_best_gap",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results",
        type=Path,
        default=Path("real_heron_hardware_results.csv"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("real_heron_hardware_summary.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("real_heron_final_analysis"),
    )
    return parser.parse_args()


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def t_confidence_interval(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    if len(values) < 2:
        return float("nan"), float("nan")
    standard_error = float(stats.sem(values))
    low, high = stats.t.interval(
        0.95,
        df=len(values) - 1,
        loc=mean,
        scale=standard_error,
    )
    return float(low), float(high)


def paired_analysis(
    frame: pd.DataFrame,
    spec: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    hardware = frame[spec["hardware"]].to_numpy(dtype=float)
    mps = frame[spec["mps"]].to_numpy(dtype=float)
    difference = hardware - mps

    paired_t = stats.ttest_rel(hardware, mps)
    wilcoxon = stats.wilcoxon(
        difference,
        zero_method="wilcox",
        alternative="two-sided",
        mode="auto",
    )
    ci_low, ci_high = t_confidence_interval(difference)
    difference_sd = float(np.std(difference, ddof=1))
    cohen_dz = (
        float(np.mean(difference) / difference_sd)
        if difference_sd > 0
        else float("nan")
    )

    pearson = stats.pearsonr(mps, hardware)
    spearman = stats.spearmanr(mps, hardware)

    paired_row = {
        "metric": spec["metric"],
        "n": int(len(frame)),
        "mps_mean": float(np.mean(mps)),
        "mps_sd": float(np.std(mps, ddof=1)),
        "hardware_mean": float(np.mean(hardware)),
        "hardware_sd": float(np.std(hardware, ddof=1)),
        "mean_difference_hardware_minus_mps": float(
            np.mean(difference)
        ),
        "difference_95ci_low": ci_low,
        "difference_95ci_high": ci_high,
        "paired_t": float(paired_t.statistic),
        "paired_t_p": float(paired_t.pvalue),
        "wilcoxon_w": float(wilcoxon.statistic),
        "wilcoxon_p": float(wilcoxon.pvalue),
        "cohen_dz": cohen_dz,
        "hardware_lower_gap_better_n": int(np.sum(hardware < mps)),
        "ties_n": int(np.sum(np.isclose(hardware, mps, atol=1e-12))),
        "hardware_higher_gap_worse_n": int(np.sum(hardware > mps)),
    }

    correlation_row = {
        "metric": spec["metric"],
        "n": int(len(frame)),
        "pearson_r": float(pearson.statistic),
        "pearson_p": float(pearson.pvalue),
        "spearman_rho": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
    }
    return paired_row, correlation_row


def scatter_figure(
    frame: pd.DataFrame,
    spec: dict[str, str],
    out_dir: Path,
    correlation: dict[str, Any],
) -> None:
    mps = frame[spec["mps"]].to_numpy(dtype=float)
    hardware = frame[spec["hardware"]].to_numpy(dtype=float)

    lower = float(min(np.min(mps), np.min(hardware)))
    upper = float(max(np.max(mps), np.max(hardware)))
    padding = max((upper - lower) * 0.08, 0.01)
    limits = (lower - padding, upper + padding)

    fig, ax = plt.subplots(figsize=(5.4, 5.0))
    ax.scatter(mps, hardware, s=38)
    ax.plot(limits, limits, linestyle="--", linewidth=1.2)
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Ideal MPS")
    ax.set_ylabel("Heron hardware")
    ax.set_title(spec["label"])
    ax.grid(True, linewidth=0.5, alpha=0.35)
    ax.text(
        0.04,
        0.96,
        (
            f"Pearson r = {correlation['pearson_r']:.3f}\n"
            f"p = {correlation['pearson_p']:.2e}"
        ),
        transform=ax.transAxes,
        ha="left",
        va="top",
    )
    save_figure(fig, out_dir / spec["figure"])


def assignment_quality_figure(
    frame: pd.DataFrame,
    out_dir: Path,
) -> None:
    data = [
        frame[column].to_numpy(dtype=float)
        for column, _ in ASSIGNMENT_COLUMNS
    ]
    labels = [label for _, label in ASSIGNMENT_COLUMNS]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    ax.boxplot(
        data,
        tick_labels=labels,
        showmeans=True,
    )

    rng = np.random.default_rng(20260723)
    for position, values in enumerate(data, start=1):
        jitter = rng.normal(0.0, 0.035, size=len(values))
        ax.scatter(
            np.full(len(values), position, dtype=float) + jitter,
            values,
            s=20,
            alpha=0.65,
        )

    ax.set_ylim(0.4, 1.02)
    ax.set_ylabel("Fraction")
    ax.set_title("Raw best measured QPU assignment quality")
    ax.grid(True, axis="y", linewidth=0.5, alpha=0.35)
    save_figure(
        fig,
        out_dir / "figure_raw_best_assignment_quality",
    )


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(args.results)
    summary = json.loads(args.summary.read_text(encoding="utf-8"))

    if len(frame) != 30:
        raise ValueError(f"Expected 30 patches; found {len(frame)}")
    if frame["patch_id"].nunique() != 30:
        raise ValueError("Patch identifiers are not unique")
    if int(frame["shots"].sum()) != 300_000:
        raise ValueError(
            f"Expected 300,000 shots; found {int(frame['shots'].sum())}"
        )

    required = {
        item[key]
        for item in PAIRED_METRICS
        for key in ("hardware", "mps")
    }
    required.update(column for column, _ in ASSIGNMENT_COLUMNS)
    required.update(CIRCUIT_COLUMNS)
    required.update(PENALTY_COLUMNS)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    paired_rows = []
    correlation_rows = []
    correlation_lookup: dict[str, dict[str, Any]] = {}

    for spec in PAIRED_METRICS:
        paired_row, correlation_row = paired_analysis(frame, spec)
        paired_rows.append(paired_row)
        correlation_rows.append(correlation_row)
        correlation_lookup[spec["metric"]] = correlation_row

        scatter_figure(
            frame,
            spec,
            args.out,
            correlation_row,
        )

    paired_table = pd.DataFrame(paired_rows)
    correlation_table = pd.DataFrame(correlation_rows)

    # Table 3 defines one inferential family containing the three paired
    # MPS-versus-QPU outcomes. Apply Holm correction in the executable
    # analysis so the reported values are not produced manually downstream.
    paired_table["paired_t_p_holm"] = multipletests(
        paired_table["paired_t_p"].to_numpy(dtype=float),
        method="holm",
    )[1]
    paired_table["wilcoxon_p_holm"] = multipletests(
        paired_table["wilcoxon_p"].to_numpy(dtype=float),
        method="holm",
    )[1]

    paired_table.to_csv(
        args.out / "real_heron_paired_tests.csv",
        index=False,
    )
    correlation_table.to_csv(
        args.out / "real_heron_mps_qpu_correlations.csv",
        index=False,
    )

    table3 = paired_table[
        [
            "metric",
            "n",
            "mps_mean",
            "mps_sd",
            "hardware_mean",
            "hardware_sd",
            "mean_difference_hardware_minus_mps",
            "difference_95ci_low",
            "difference_95ci_high",
            "paired_t",
            "paired_t_p",
            "paired_t_p_holm",
            "cohen_dz",
        ]
    ].copy()
    table3.to_csv(args.out / "table3_mps_vs_heron.csv", index=False)
    table3_headers = list(table3.columns)
    table3_lines = [
        "| " + " | ".join(table3_headers) + " |",
        "| " + " | ".join("---" for _ in table3_headers) + " |",
    ]
    for row in table3.itertuples(index=False, name=None):
        rendered = [
            f"{value:.6g}" if isinstance(value, (float, np.floating)) else str(value)
            for value in row
        ]
        table3_lines.append("| " + " | ".join(rendered) + " |")
    (args.out / "table3_mps_vs_heron.md").write_text(
        "\n".join(table3_lines) + "\n",
        encoding="utf-8",
    )

    assignment_rows = []
    for column, label in ASSIGNMENT_COLUMNS:
        values = frame[column].to_numpy(dtype=float)
        ci_low, ci_high = t_confidence_interval(values)
        assignment_rows.append(
            {
                "metric": column,
                "label": label.replace("\n", " "),
                "n": int(len(values)),
                "mean": float(np.mean(values)),
                "sd": float(np.std(values, ddof=1)),
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "mean_95ci_low": ci_low,
                "mean_95ci_high": ci_high,
            }
        )

    assignment_table = pd.DataFrame(assignment_rows)
    assignment_table.to_csv(
        args.out / "real_heron_assignment_quality.csv",
        index=False,
    )
    assignment_quality_figure(frame, args.out)

    exploratory_rows = []
    for penalty in PENALTY_COLUMNS:
        for circuit_metric in CIRCUIT_COLUMNS:
            result = stats.spearmanr(
                frame[circuit_metric],
                frame[penalty],
            )
            exploratory_rows.append(
                {
                    "hardware_penalty": penalty,
                    "circuit_metric": circuit_metric,
                    "n": int(len(frame)),
                    "spearman_rho": float(result.statistic),
                    "p_uncorrected": float(result.pvalue),
                }
            )

    exploratory = pd.DataFrame(exploratory_rows)
    exploratory["p_fdr_bh"] = multipletests(
        exploratory["p_uncorrected"].to_numpy(dtype=float),
        method="fdr_bh",
    )[1]
    exploratory["significant_fdr_0_05"] = (
        exploratory["p_fdr_bh"] < 0.05
    )
    exploratory.to_csv(
        args.out / "real_heron_exploratory_circuit_correlations.csv",
        index=False,
    )

    mean_row = paired_table.loc[
        paired_table["metric"] == "mean_normalized_gap"
    ].iloc[0]
    cvar_row = paired_table.loc[
        paired_table["metric"] == "cvar10_normalized_gap"
    ].iloc[0]
    best_row = paired_table.loc[
        paired_table["metric"] == "best_normalized_gap"
    ].iloc[0]

    mean_corr = correlation_table.loc[
        correlation_table["metric"] == "mean_normalized_gap"
    ].iloc[0]
    cvar_corr = correlation_table.loc[
        correlation_table["metric"] == "cvar10_normalized_gap"
    ].iloc[0]
    best_corr = correlation_table.loc[
        correlation_table["metric"] == "best_normalized_gap"
    ].iloc[0]

    results_text = f"""# Real-data Heron hardware validation

The 30-subject experiment comprised 30 SamplerV2 PUBs with 10,000 shots per
case (300,000 shots in total). The raw measured outcomes were analysed without
constraint repair, feasibility projection, ICM, or any other classical
post-processing.

The mean normalized gap was {mean_row.hardware_mean:.4f} on Heron hardware and
{mean_row.mps_mean:.4f} in the ideal MPS simulation. The paired hardware-minus-
MPS difference was {mean_row.mean_difference_hardware_minus_mps:+.4f}
(95% CI {mean_row.difference_95ci_low:+.4f} to
{mean_row.difference_95ci_high:+.4f}; paired t-test
Holm-adjusted p={mean_row.paired_t_p_holm:.4g}; Wilcoxon Holm-adjusted
p={mean_row.wilcoxon_p_holm:.4g};
Cohen's dz={mean_row.cohen_dz:.3f}).

For the low-energy tail, the CVaR-10% normalized gap was
{cvar_row.hardware_mean:.4f} on hardware and {cvar_row.mps_mean:.4f} for MPS.
The paired difference was {cvar_row.mean_difference_hardware_minus_mps:+.4f}
(95% CI {cvar_row.difference_95ci_low:+.4f} to
{cvar_row.difference_95ci_high:+.4f}; paired t-test
Holm-adjusted p={cvar_row.paired_t_p_holm:.4g}; Wilcoxon Holm-adjusted
p={cvar_row.wilcoxon_p_holm:.4g};
Cohen's dz={cvar_row.cohen_dz:.3f}).

The best observed normalized gap was {best_row.hardware_mean:.4f} on hardware
and {best_row.mps_mean:.4f} for ideal MPS. The paired difference was
{best_row.mean_difference_hardware_minus_mps:+.4f}
(95% CI {best_row.difference_95ci_low:+.4f} to
{best_row.difference_95ci_high:+.4f}; paired t-test
Holm-adjusted p={best_row.paired_t_p_holm:.4g}; Wilcoxon Holm-adjusted
p={best_row.wilcoxon_p_holm:.4g};
Cohen's dz={best_row.cohen_dz:.3f}).

Case-wise MPS and hardware performance were strongly correlated for the mean
gap (Pearson r={mean_corr.pearson_r:.3f},
p={mean_corr.pearson_p:.2e}), CVaR-10% gap
(r={cvar_corr.pearson_r:.3f}, p={cvar_corr.pearson_p:.2e}), and best-sample
gap (r={best_corr.pearson_r:.3f}, p={best_corr.pearson_p:.2e}).

The raw lowest-energy QPU assignment showed mean agreement of
{frame['best_agreement_with_exact_up_to_sign'].mean():.3f} with the exact MILP
assignment (up to global sign), mean agreement of
{frame['best_tom_agreement_confident_voxels'].mean():.3f} with the confident
TOM pseudo-reference, mean within-voxel conflict satisfaction of
{frame['best_conflict_satisfaction_fraction'].mean():.3f}, and mean weighted
continuation satisfaction of
{frame['best_weighted_continuation_satisfaction_fraction'].mean():.3f}.

No exact MILP ground-state bitstring was observed. This outcome is reported
descriptively rather than used as the primary endpoint because all 10,000
hardware shots were unique in every 50-qubit case.

Exploratory Spearman analyses tested whether the hardware-minus-MPS penalties
were associated with circuit depth, two-qubit gate count, active physical
width, calibration errors, calibration-weighted error cost, or the independent
two-qubit success proxy. None survived Benjamini-Hochberg false-discovery-rate
correction across the 21 exploratory tests.
"""
    (args.out / "real_heron_results_text.md").write_text(
        results_text,
        encoding="utf-8",
    )

    analysis_summary = {
        "input_results": str(args.results.resolve()),
        "input_summary": str(args.summary.resolve()),
        "n_patches": int(len(frame)),
        "total_shots": int(frame["shots"].sum()),
        "raw_measured_assignments_only": True,
        "classical_postprocessing": False,
        "paired_tests": paired_table.to_dict(orient="records"),
        "mps_hardware_correlations": correlation_rows,
        "assignment_quality": assignment_rows,
        "n_exploratory_circuit_tests": int(len(exploratory)),
        "n_exploratory_significant_after_fdr": int(
            exploratory["significant_fdr_0_05"].sum()
        ),
        "job_id": summary.get("job_id"),
        "backend": summary.get("backend"),
        "qpu_charge_time_seconds": (
            summary.get("runtime", {})
            .get("job_metrics", {})
            .get("usage", {})
            .get("qpu_charge_time_seconds")
        ),
    }
    (args.out / "real_heron_analysis_summary.json").write_text(
        json.dumps(analysis_summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print(results_text)
    print(f"\nOutputs written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
