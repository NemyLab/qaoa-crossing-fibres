#!/usr/bin/env python3
"""Paired factorial statistics for the four-condition phantom benchmark.

Primary design
--------------
The same 15 random seeds are evaluated under a fully paired 2x2 design:

    crossing angle: 90 degrees vs 60 degrees
    orientation:    original vs globally rotated 45 degrees

For each panel metric, the script calculates three orthogonal within-seed
contrasts:

    angle main effect:
        mean(60 original, 60 rotated) - mean(90 original, 90 rotated)

    rotation main effect:
        mean(90 rotated, 60 rotated) - mean(90 original, 60 original)

    angle x rotation interaction:
        (60 rotated - 60 original) - (90 rotated - 90 original)

Inference uses an exact two-sided sign-flip permutation test across seeds.
With 15 seeds, all 2^15 = 32,768 sign patterns are enumerated. The seed—not
the individual QPU shot—is the independent statistical unit.

The script also exports four paired simple contrasts per metric and reports
paired Cohen's dz, bootstrap confidence intervals, and Holm-adjusted p-values.

Run this script from crossing_fibres_step7 beside:
    make_phantom_difficulty_summary_figure_all15.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

try:
    import make_phantom_difficulty_summary_figure_all15 as figure_data
except ImportError as exc:
    raise SystemExit(
        "Could not import make_phantom_difficulty_summary_figure_all15.py. "
        "Place both scripts in crossing_fibres_step7 and run from that directory."
    ) from exc


METRICS = (
    {
        "panel": "A",
        "metric": "exact_solution_label_accuracy",
        "label": "Exact MILP label accuracy",
        "better_direction": "higher",
        "role": "encoding/ground-truth endpoint",
    },
    {
        "panel": "B",
        "metric": "ideal_expected_normalized_gap",
        "label": "Ideal QAOA expected normalized gap",
        "better_direction": "lower",
        "role": "ideal-algorithm endpoint",
    },
    {
        "panel": "C",
        "metric": "cvar10_normalized_gap",
        "label": "QPU CVaR10 normalized gap",
        "better_direction": "lower",
        "role": "primary hardware endpoint",
    },
    {
        "panel": "D",
        "metric": "best_agreement_with_exact",
        "label": "Best QPU assignment agreement with exact MILP",
        "better_direction": "higher",
        "role": "secondary shot-dependent endpoint",
    },
)

CONDITION_ORDER = (
    "90_original",
    "90_rotated",
    "60_original",
    "60_rotated",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original-heldout",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--original-replication",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rotated90",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--normal60",
        nargs="+",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--rotated60",
        nargs="+",
        type=Path,
        required=True,
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
        "--bootstrap-repetitions",
        type=int,
        default=100000,
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=20260718,
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("phantom_factorial_statistics_all15"),
    )
    return parser.parse_args()


def load_long_table(args: argparse.Namespace) -> pd.DataFrame:
    data_args = SimpleNamespace(
        original_heldout=args.original_heldout,
        original_replication=args.original_replication,
        rotated90=args.rotated90,
        normal60=args.normal60,
        rotated60=args.rotated60,
        strategy=args.strategy,
        seeds=args.seeds,
    )
    return figure_data.make_long_table(data_args)


def to_wide(long_table: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = long_table.pivot(
        index="seed",
        columns="condition_key",
        values=metric,
    ).sort_index()

    missing_columns = [column for column in CONDITION_ORDER if column not in wide.columns]
    if missing_columns:
        raise ValueError(f"{metric}: missing condition columns {missing_columns}")
    wide = wide.loc[:, CONDITION_ORDER]

    if wide.isna().any().any():
        bad = wide[wide.isna().any(axis=1)]
        raise ValueError(f"{metric}: missing paired values:\n{bad}")

    return wide.astype(float)


def exact_sign_flip_pvalue(differences: np.ndarray) -> float:
    """Exact two-sided sign-flip test of the mean paired contrast."""
    differences = np.asarray(differences, dtype=float)
    n = differences.size
    if n == 0:
        return float("nan")
    if n > 20:
        raise ValueError(
            f"Exact enumeration requested for n={n}; this script is intended "
            "for the 15-seed benchmark."
        )

    observed = abs(float(np.mean(differences)))
    indices = np.arange(1 << n, dtype=np.uint32)[:, None]
    bit_positions = np.arange(n, dtype=np.uint32)[None, :]
    signs = np.where(((indices >> bit_positions) & 1) == 1, 1.0, -1.0)
    permuted = np.abs((signs @ differences) / n)

    # Numerical tolerance prevents floating-point equality problems.
    tolerance = max(1e-15, observed * 1e-12)
    return float(np.mean(permuted >= observed - tolerance))


def bootstrap_mean_ci(
    differences: np.ndarray,
    repetitions: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    differences = np.asarray(differences, dtype=float)
    n = differences.size
    draw_indices = rng.integers(0, n, size=(repetitions, n))
    bootstrap_means = differences[draw_indices].mean(axis=1)
    low, high = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(low), float(high)


def paired_cohen_dz(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=float)
    sd = float(np.std(differences, ddof=1))
    if sd == 0.0:
        if float(np.mean(differences)) == 0.0:
            return 0.0
        return float(np.sign(np.mean(differences)) * np.inf)
    return float(np.mean(differences) / sd)


def holm_adjust(p_values: pd.Series) -> pd.Series:
    """Holm family-wise-error correction preserving the original index."""
    p = p_values.astype(float)
    valid = p.notna()
    values = p.loc[valid].to_numpy()
    m = values.size
    if m == 0:
        return pd.Series(np.nan, index=p.index, dtype=float)

    order = np.argsort(values)
    ordered = values[order]
    adjusted_ordered = np.maximum.accumulate((m - np.arange(m)) * ordered)
    adjusted_ordered = np.minimum(adjusted_ordered, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_ordered

    result = pd.Series(np.nan, index=p.index, dtype=float)
    result.loc[valid] = adjusted
    return result


def summarize_contrast(
    differences: np.ndarray,
    *,
    contrast_name: str,
    definition: str,
    metric_spec: dict,
    repetitions: int,
    rng: np.random.Generator,
) -> dict:
    differences = np.asarray(differences, dtype=float)
    ci_low, ci_high = bootstrap_mean_ci(differences, repetitions, rng)
    mean_difference = float(np.mean(differences))
    return {
        "panel": metric_spec["panel"],
        "metric": metric_spec["metric"],
        "metric_label": metric_spec["label"],
        "metric_role": metric_spec["role"],
        "better_direction": metric_spec["better_direction"],
        "contrast": contrast_name,
        "contrast_definition": definition,
        "n_seeds": int(differences.size),
        "mean_difference": mean_difference,
        "median_difference": float(np.median(differences)),
        "sd_difference": float(np.std(differences, ddof=1)),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "cohen_dz": paired_cohen_dz(differences),
        "exact_sign_flip_p": exact_sign_flip_pvalue(differences),
        "positive_seed_count": int(np.sum(differences > 0)),
        "negative_seed_count": int(np.sum(differences < 0)),
        "zero_seed_count": int(np.sum(differences == 0)),
        "seed_differences": json.dumps(
            {str(index): float(value) for index, value in enumerate(differences)}
        ),
    }


def factorial_contrasts(wide: pd.DataFrame) -> dict[str, np.ndarray]:
    y90o = wide["90_original"].to_numpy()
    y90r = wide["90_rotated"].to_numpy()
    y60o = wide["60_original"].to_numpy()
    y60r = wide["60_rotated"].to_numpy()

    return {
        "crossing_angle_main_effect": 0.5 * ((y60o + y60r) - (y90o + y90r)),
        "rotation_main_effect": 0.5 * ((y90r + y60r) - (y90o + y60o)),
        "angle_by_rotation_interaction": (y60r - y60o) - (y90r - y90o),
    }


def simple_contrasts(wide: pd.DataFrame) -> dict[str, np.ndarray]:
    y90o = wide["90_original"].to_numpy()
    y90r = wide["90_rotated"].to_numpy()
    y60o = wide["60_original"].to_numpy()
    y60r = wide["60_rotated"].to_numpy()

    return {
        "rotation_effect_at_90deg": y90r - y90o,
        "rotation_effect_at_60deg": y60r - y60o,
        "60_minus_90_in_original_orientation": y60o - y90o,
        "60_minus_90_in_rotated_orientation": y60r - y90r,
    }


FACTORIAL_DEFINITIONS = {
    "crossing_angle_main_effect":
        "average(60° original, 60° rotated) minus average(90° original, 90° rotated)",
    "rotation_main_effect":
        "average(rotated 90°, rotated 60°) minus average(original 90°, original 60°)",
    "angle_by_rotation_interaction":
        "(60° rotated minus 60° original) minus (90° rotated minus 90° original)",
}

SIMPLE_DEFINITIONS = {
    "rotation_effect_at_90deg":
        "90° rotated minus 90° original",
    "rotation_effect_at_60deg":
        "60° rotated minus 60° original",
    "60_minus_90_in_original_orientation":
        "60° original minus 90° original",
    "60_minus_90_in_rotated_orientation":
        "60° rotated minus 90° rotated",
}


def condition_descriptives(long_table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for metric_spec in METRICS:
        metric = metric_spec["metric"]
        for condition_key in CONDITION_ORDER:
            values = (
                long_table.loc[long_table["condition_key"] == condition_key, metric]
                .astype(float)
            )
            rows.append(
                {
                    "panel": metric_spec["panel"],
                    "metric": metric,
                    "metric_label": metric_spec["label"],
                    "condition_key": condition_key,
                    "n_seeds": int(values.size),
                    "mean": float(values.mean()),
                    "sd": float(values.std(ddof=1)),
                    "median": float(values.median()),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame(rows)


def add_multiple_testing_columns(
    factorial: pd.DataFrame,
    simple: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    factorial = factorial.copy()
    simple = simple.copy()

    # Preferred correction: three factorial terms within each metric.
    factorial["p_holm_within_metric"] = (
        factorial.groupby("metric", group_keys=False)["exact_sign_flip_p"]
        .apply(holm_adjust)
    )

    # Conservative global correction across all 12 factorial tests.
    factorial["p_holm_all_factorial_tests"] = holm_adjust(
        factorial["exact_sign_flip_p"]
    )

    # Four simple contrasts within each metric.
    simple["p_holm_within_metric"] = (
        simple.groupby("metric", group_keys=False)["exact_sign_flip_p"]
        .apply(holm_adjust)
    )
    simple["p_holm_all_simple_tests"] = holm_adjust(
        simple["exact_sign_flip_p"]
    )
    return factorial, simple


def format_p(value: float) -> str:
    if np.isnan(value):
        return "NA"
    if value < 0.001:
        return "<0.001"
    return f"{value:.3f}"


def make_reporting_notes(
    factorial: pd.DataFrame,
    simple: pd.DataFrame,
    args: argparse.Namespace,
) -> str:
    lines = [
        "Recommended statistical reporting",
        "=================================",
        "",
        f"Independent unit: randomized phantom seed (n={len(args.seeds)}).",
        "The 10,000 QPU shots per PUB were not treated as independent observations.",
        "All tests are paired because every seed was evaluated in all four conditions.",
        "",
        "Primary endpoint",
        "----------------",
        "Panel C: QPU CVaR10 normalized gap using the hardware-calibrated angle strategy.",
        "Report the 2x2 paired factorial contrasts: crossing-angle main effect,",
        "rotation main effect, and angle-by-rotation interaction.",
        "",
        "Inference",
        "---------",
        "Two-sided exact sign-flip permutation tests across seeds (2^15 permutations),",
        "with Holm correction across the three factorial terms within each metric.",
        "Effects are reported as mean paired differences with 95% bootstrap confidence",
        "intervals and paired Cohen's dz.",
        "",
        "Sign convention",
        "---------------",
        "All effects are expressed as 60° minus 90° and/or rotated minus original.",
        "For gap metrics (Panels B/C), positive effects indicate worse performance.",
        "For agreement/accuracy metrics (Panels A/D), negative effects indicate worse performance.",
        "",
        "Factorial results",
        "-----------------",
    ]

    for _, row in factorial.iterrows():
        lines.append(
            f"Panel {row['panel']} — {row['contrast']}: "
            f"mean difference={row['mean_difference']:.4f}, "
            f"95% bootstrap CI [{row['bootstrap_ci95_low']:.4f}, "
            f"{row['bootstrap_ci95_high']:.4f}], "
            f"dz={row['cohen_dz']:.3f}, "
            f"exact p={format_p(row['exact_sign_flip_p'])}, "
            f"Holm-adjusted p={format_p(row['p_holm_within_metric'])}."
        )

    lines.extend(
        [
            "",
            "Interpret simple contrasts only when scientifically useful, particularly",
            "when the interaction is non-negligible. Panel D should remain secondary",
            "because a best-observed bitstring is dependent on shot count and extreme",
            "sampling events.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    long_table = load_long_table(args)
    expected_rows = len(args.seeds) * len(CONDITION_ORDER)
    if len(long_table) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} selected rows, found {len(long_table)}"
        )

    rng = np.random.default_rng(args.random_seed)
    factorial_rows = []
    simple_rows = []

    for metric_spec in METRICS:
        wide = to_wide(long_table, metric_spec["metric"])

        for contrast_name, differences in factorial_contrasts(wide).items():
            row = summarize_contrast(
                differences,
                contrast_name=contrast_name,
                definition=FACTORIAL_DEFINITIONS[contrast_name],
                metric_spec=metric_spec,
                repetitions=args.bootstrap_repetitions,
                rng=rng,
            )
            row["seed_differences"] = json.dumps(
                {
                    str(int(seed)): float(value)
                    for seed, value in zip(wide.index, differences)
                }
            )
            factorial_rows.append(row)

        for contrast_name, differences in simple_contrasts(wide).items():
            row = summarize_contrast(
                differences,
                contrast_name=contrast_name,
                definition=SIMPLE_DEFINITIONS[contrast_name],
                metric_spec=metric_spec,
                repetitions=args.bootstrap_repetitions,
                rng=rng,
            )
            row["seed_differences"] = json.dumps(
                {
                    str(int(seed)): float(value)
                    for seed, value in zip(wide.index, differences)
                }
            )
            simple_rows.append(row)

    factorial = pd.DataFrame(factorial_rows)
    simple = pd.DataFrame(simple_rows)
    factorial, simple = add_multiple_testing_columns(factorial, simple)
    descriptives = condition_descriptives(long_table)

    factorial.to_csv(args.out / "factorial_effects_exact_permutation.csv", index=False)
    simple.to_csv(args.out / "simple_paired_contrasts_exact_permutation.csv", index=False)
    descriptives.to_csv(args.out / "condition_descriptives.csv", index=False)

    notes = make_reporting_notes(factorial, simple, args)
    (args.out / "reporting_notes.txt").write_text(notes + "\n", encoding="utf-8")

    metadata = {
        "strategy": args.strategy,
        "seeds": [int(seed) for seed in args.seeds],
        "independent_unit": "randomized phantom seed",
        "shots_are_independent_units": False,
        "factorial_design": {
            "crossing_angle": ["90_deg", "60_deg"],
            "orientation": ["original", "rotated_45_deg"],
        },
        "test": "two-sided exact sign-flip permutation test of paired mean contrast",
        "number_of_exact_sign_patterns": int(2 ** len(args.seeds)),
        "bootstrap_repetitions": int(args.bootstrap_repetitions),
        "bootstrap_random_seed": int(args.random_seed),
        "multiplicity": {
            "preferred": "Holm correction across three factorial effects within each metric",
            "also_exported": "Holm correction across all factorial tests",
        },
        "primary_endpoint": "Panel C: cvar10_normalized_gap",
        "secondary_endpoint_warning":
            "Panel D is shot-count-dependent because it uses the best observed bitstring.",
    }
    (args.out / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print("\nFactorial effects:")
    display_columns = [
        "panel",
        "contrast",
        "mean_difference",
        "bootstrap_ci95_low",
        "bootstrap_ci95_high",
        "cohen_dz",
        "exact_sign_flip_p",
        "p_holm_within_metric",
    ]
    print(factorial[display_columns].to_string(index=False))
    print(f"\nResults written to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
