#!/usr/bin/env python3
"""Generate the Nighthawk phantom 2x2 figure using the unchanged Heron plotter.

This wrapper imports:
    make_phantom_difficulty_summary_figure_all15.py

and executes its original plotting/export main() unchanged. Only the data-loading
function and command-line defaults are replaced so that the four Nighthawk result
CSVs are used with the frozen `nighthawk_calibrated` strategy.

Consequently, the following remain exactly those of the Heron figure:
    - 2x2 layout and dimensions
    - panel order and titles
    - fonts and font sizes
    - condition colors
    - seed markers and paired lines
    - mean diamonds and 95% CIs
    - axes, limits, spacing, legend, and suptitle
    - individual-panel exports
    - PNG/PDF/SVG export behavior and output filenames
"""
from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

try:
    import make_phantom_difficulty_summary_figure_all15 as heron_figure
except ImportError as exc:
    raise SystemExit(
        "Could not import make_phantom_difficulty_summary_figure_all15.py. "
        "Place this Nighthawk wrapper beside the original Heron script and "
        "run it from crossing_fibres_step7."
    ) from exc


EXPECTED_HELPERS = (
    "main",
    "read_results_many",
    "prepare_condition",
    "configure_matplotlib",
    "adjusted_panel_specs",
    "draw_metric_panel",
    "save_figure",
    "summary_table",
    "paired_rotation_table",
    "caption_text",
    "CONDITIONS",
)

missing_helpers = [
    name for name in EXPECTED_HELPERS
    if not hasattr(heron_figure, name)
]
if missing_helpers:
    raise SystemExit(
        "The imported Heron plotter is not the expected all15 version. "
        f"Missing: {missing_helpers}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the Nighthawk phantom difficulty/orientation figure using "
            "the original Heron plotting code unchanged."
        )
    )
    parser.add_argument(
        "--original90",
        nargs="+",
        type=Path,
        required=True,
        help=(
            "One or more non-overlapping original-90° Nighthawk hardware "
            "result CSVs. The file may also contain transferred_center rows; "
            "they are filtered out by --strategy."
        ),
    )
    parser.add_argument(
        "--rotated90",
        nargs="+",
        type=Path,
        required=True,
        help="One or more non-overlapping rotated-90° Nighthawk result CSVs.",
    )
    parser.add_argument(
        "--normal60",
        nargs="+",
        type=Path,
        required=True,
        help="One or more non-overlapping original-60° Nighthawk result CSVs.",
    )
    parser.add_argument(
        "--rotated60",
        nargs="+",
        type=Path,
        required=True,
        help="One or more non-overlapping rotated-60° Nighthawk result CSVs.",
    )
    parser.add_argument(
        "--strategy",
        default="nighthawk_calibrated",
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
        help=(
            "Keep the Heron default cvar10_normalized_gap for an identical "
            "Panel C definition."
        ),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("paper_figures_phantom_summary_nighthawk"),
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


def make_long_table_nighthawk(args: argparse.Namespace):
    """Nighthawk-only replacement for the original Heron input assembly."""
    source_frames = {
        "90_original": heron_figure.read_results_many(args.original90),
        "90_rotated": heron_figure.read_results_many(args.rotated90),
        "60_original": heron_figure.read_results_many(args.normal60),
        "60_rotated": heron_figure.read_results_many(args.rotated60),
    }

    seeds = tuple(int(seed) for seed in args.seeds)
    prepared = []
    for key, label, geometry, _x in heron_figure.CONDITIONS:
        prepared.append(
            heron_figure.prepare_condition(
                source_frames[key],
                key=key,
                label=label,
                geometry=geometry,
                strategy=args.strategy,
                seeds=seeds,
            )
        )
    return heron_figure.pd.concat(prepared, ignore_index=True)


def main() -> None:
    args = parse_args()

    # The original main() calls its module-level parse_args() and
    # make_long_table(). Replace only those two hooks.
    heron_figure.parse_args = lambda: SimpleNamespace(**vars(args))
    heron_figure.make_long_table = make_long_table_nighthawk

    print(
        "Using the unchanged visual and export code from:\n"
        f"  {Path(heron_figure.__file__).resolve()}\n"
        "Only Nighthawk data loading and defaults are replaced."
    )
    heron_figure.main()


if __name__ == "__main__":
    main()
