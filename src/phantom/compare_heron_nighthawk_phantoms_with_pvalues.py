#!/usr/bin/env python3
"""Add condition-specific Heron-versus-Nighthawk p-values to the existing figure.

This is a drop-in wrapper around ``compare_heron_nighthawk_phantoms.py``.
It preserves the original data loading, statistics, plotting style, panel
layout, colours, markers, legends, file names, and exported tables. The only
plotting addition is a Heron-versus-Nighthawk bracket above each phantom
condition in every panel, labelled with a Holm-adjusted paired P value.

Run this script with exactly the same command-line arguments that you use for
``compare_heron_nighthawk_phantoms.py``. Keep both scripts in the same folder.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import compare_heron_nighthawk_phantoms as original


# Preserve references to the original functions before monkey-patching them.
_ORIGINAL_DRAW_BACKEND_PANEL = original.draw_backend_panel
_ORIGINAL_CAPTION_TEXT = original.caption_text


def _format_adjusted_p(value: float) -> str:
    """Format a Holm-adjusted P value for a Matplotlib annotation."""
    if not np.isfinite(value):
        return r"$P_{\mathrm{adj}}=\mathrm{NA}$"
    if value < 0.001:
        return r"$P_{\mathrm{adj}}<0.001$"
    return rf"$P_{{\mathrm{{adj}}}}={value:.3f}$"


def _condition_specific_adjusted_p_values(
    frame: pd.DataFrame,
    metric: str,
) -> dict[str, float]:
    """Calculate four paired backend tests and Holm-adjust them within panel.

    Each test compares Nighthawk with Heron for the same 15 seeds in one
    phantom condition. The difference is Nighthawk minus Heron. Only the four
    condition-specific tests displayed in the panel enter the Holm family;
    no pooled across-condition contrast is included.
    """
    pivot = frame.pivot_table(
        index="seed",
        columns=["backend", "condition_key"],
        values=metric,
        aggfunc="first",
    )

    raw_p_values: list[float] = []
    conditions: list[str] = []

    for condition in original.CONDITION_ORDER:
        difference = (
            pivot[("Nighthawk", condition)]
            - pivot[("Heron", condition)]
        ).dropna()

        result = original.stats.ttest_1samp(
            difference.to_numpy(dtype=float),
            popmean=0.0,
        )
        raw_p_values.append(float(result.pvalue))
        conditions.append(str(condition))

    _, adjusted, _, _ = original.multipletests(
        np.asarray(raw_p_values, dtype=float),
        method="holm",
    )
    return {
        condition: float(p_value)
        for condition, p_value in zip(conditions, adjusted)
    }


def _add_backend_p_value_brackets(
    ax: Any,
    frame: pd.DataFrame,
    metric: str,
) -> None:
    """Add one paired Heron-versus-Nighthawk bracket per condition."""
    adjusted_p = _condition_specific_adjusted_p_values(frame, metric)

    x_offset = {"Heron": -0.16, "Nighthawk": 0.16}
    y_min, y_max = ax.get_ylim()
    y_span = max(float(y_max - y_min), 1e-9)

    bracket_height = 0.018 * y_span
    data_padding = 0.045 * y_span
    text_padding = 0.012 * y_span

    annotations: list[tuple[float, float, float, str]] = []
    required_top = y_max

    for condition in original.CONDITION_ORDER:
        centre = float(original.CONDITION_META[condition]["x"])
        x_left = centre + x_offset["Heron"]
        x_right = centre + x_offset["Nighthawk"]

        condition_values = frame.loc[
            frame["condition_key"] == condition,
            metric,
        ].astype(float)
        local_maximum = float(condition_values.max())

        bracket_y = local_maximum + data_padding
        text_y = bracket_y + bracket_height + text_padding
        required_top = max(required_top, text_y + 0.035 * y_span)

        annotations.append(
            (
                x_left,
                x_right,
                bracket_y,
                _format_adjusted_p(adjusted_p[condition]),
            )
        )

    # Add only the headroom required for the annotations.
    if required_top > y_max:
        ax.set_ylim(y_min, required_top)

    for x_left, x_right, bracket_y, label in annotations:
        ax.plot(
            [x_left, x_left, x_right, x_right],
            [
                bracket_y,
                bracket_y + bracket_height,
                bracket_y + bracket_height,
                bracket_y,
            ],
            color="#333333",
            linewidth=0.8,
            clip_on=False,
            zorder=6,
        )
        ax.text(
            (x_left + x_right) / 2.0,
            bracket_y + bracket_height + text_padding,
            label,
            ha="center",
            va="bottom",
            fontsize=7.2,
            color="#222222",
            clip_on=False,
            zorder=7,
        )


def _set_latex_ylabel(ax: Any, spec: dict[str, Any]) -> None:
    """Replace only the y-axis label with compact manuscript notation."""
    metric = str(spec.get("metric", "")).strip().lower()
    current_label = str(ax.get_ylabel()).strip().lower()

    labels = {
        "qpu_mean_normalized_gap": (
            "QPU mean gap"
            r"$\widehat{G}_{\mathrm{QPU}}$"
        ),
        "qpu_execution_penalty": (
            "QPU execution penalty"
            r"$\Delta G_{\mathrm{exec}}$"
        ),
        "qpu_cvar10_normalized_gap": (
            "QPU CVaR gap"
            r"$\widehat{C}_{0.10,\mathrm{QPU}}$"
        ),
        "best_agreement_with_exact": (
            "Agreement with MILP "
            r"$A(\widehat{\mathbf{s}}_{\mathrm{best}},\mathbf{s}^{*})$"
        ),
        "best_assignment_agreement_with_milp": (
            "Agreement with MILP "
            r"$A(\widehat{\mathbf{s}}_{\mathrm{best}},\mathbf{s}^{*})$"
        ),
    }

    label = labels.get(metric)

    # Fallbacks support older versions of the original plotting script,
    # including the panel-B label "hardware minus ideal expected gap".
    if label is None:
        if (
            "execution" in metric
            or "execution penalty" in current_label
            or "minusideal" in current_label.replace(" ", "")
            or ("hardware" in current_label and "ideal" in current_label)
        ):
            label = (
                "QPU execution penalty "
                r"$\Delta G_{\mathrm{exec}}$"
            )
        elif "cvar" in metric or "cvar" in current_label:
            label = (
                "QPU CVaR norm. gap "
                r"$\widehat{C}_{0.10,\mathrm{QPU}}$"
            )
        elif "agreement" in metric or "agreement" in current_label:
            label = (
                "Agreement with MILP "
                r"$A(\widehat{\mathbf{s}}_{\mathrm{best}},\mathbf{s}^{*})$"
            )
        elif (
            "mean_normalized_gap" in metric
            or "mean normalized gap" in current_label
        ):
            label = (
                "QPU mean norm. gap "
                r"$\widehat{G}_{\mathrm{QPU}}$"
            )

    if label is not None:
        ax.set_ylabel(label, fontsize=8.5, labelpad=7)


def _remove_direction_annotations(ax: Any) -> None:
    """Remove only the 'higher/lower is better' helper text."""
    for artist in list(ax.texts):
        value = str(artist.get_text()).strip().lower()
        if "higher is better" in value or "lower is better" in value:
            artist.remove()

def draw_backend_panel_with_pvalues(
    ax: Any,
    frame: pd.DataFrame,
    spec: dict[str, Any],
    seeds: tuple[int, ...],
) -> None:
    """Draw the original panel, then add labels and statistical annotations."""
    _ORIGINAL_DRAW_BACKEND_PANEL(ax, frame, spec, seeds)
    _remove_direction_annotations(ax)
    _set_latex_ylabel(ax, spec)
    _add_backend_p_value_brackets(
        ax=ax,
        frame=frame,
        metric=str(spec["metric"]),
    )


def caption_text_with_pvalues(seeds: tuple[int, ...]) -> str:
    """Append the statistical-annotation definition to the original caption."""
    return (
        _ORIGINAL_CAPTION_TEXT(seeds)
        + " Brackets compare Heron and Nighthawk within each phantom "
        "condition. Displayed P values are from two-sided paired t-tests "
        "across matched seeds and are Holm-adjusted across the four "
        "condition-specific comparisons within each panel."
    )


# Patch only the panel renderer and caption. The original main function and all
# other analysis/export functions remain unchanged.
original.draw_backend_panel = draw_backend_panel_with_pvalues
original.caption_text = caption_text_with_pvalues


if __name__ == "__main__":
    original.main()
