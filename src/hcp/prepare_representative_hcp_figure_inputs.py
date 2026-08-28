#!/usr/bin/env python3
"""Prepare one cohort PUB for the established v17 real-HCP figure script.

The original v17 plotter expects the legacy single-instance hardware filenames.
This adapter extracts one patch from the 30-PUB Heron collector outputs without
changing any bitstrings, assignments, energies, or reported metrics.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("real_heron_8n_hardware/real_heron_hardware_results.csv"),
    )
    parser.add_argument(
        "--counts",
        type=Path,
        default=Path("real_heron_8n_hardware/real_heron_hardware_counts.json.gz"),
    )
    parser.add_argument(
        "--best-assignments",
        type=Path,
        default=Path("real_heron_8n_hardware/real_heron_best_assignments.npz"),
    )
    parser.add_argument("--patch-id", default="103414_R_88_86_76")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("representative_103414_hardware_for_figure"),
    )
    return parser.parse_args()


def spin_bitstring_candidates(spins: np.ndarray) -> list[str]:
    bits = ((np.asarray(spins, dtype=np.int8) + 1) // 2).astype(np.int8)
    candidates: list[str] = []
    for values in (bits, 1 - bits):
        bitstring = "".join(str(int(value)) for value in values)
        candidates.extend((bitstring, bitstring[::-1]))
    return candidates


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.results_csv)
    selected = frame.loc[frame["patch_id"].astype(str) == args.patch_id]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one row for {args.patch_id}; found {len(selected)}"
        )
    row = selected.iloc[0].to_dict()

    with gzip.open(args.counts, "rt", encoding="utf-8") as handle:
        all_counts = json.load(handle)
    if args.patch_id not in all_counts:
        raise KeyError(f"Patch {args.patch_id} is absent from {args.counts}")
    patch_counts = {
        str(bitstring): int(count)
        for bitstring, count in all_counts[args.patch_id].items()
    }
    if sum(patch_counts.values()) != int(row["shots"]):
        raise ValueError("Count total does not match the result-row shot count")

    exact_key = f"{args.patch_id}__exact_spins_rcm"
    best_key = f"{args.patch_id}__best_spins_rcm"
    with np.load(args.best_assignments, allow_pickle=False) as archive:
        missing = [key for key in (exact_key, best_key) if key not in archive.files]
        if missing:
            raise KeyError(f"Missing assignment arrays: {missing}")
        exact_spins = np.asarray(archive[exact_key], dtype=np.int8)
        best_spins = np.asarray(archive[best_key], dtype=np.int8)

    measured_best_candidates = [
        bitstring
        for bitstring in spin_bitstring_candidates(best_spins)
        if bitstring in patch_counts
    ]
    if len(measured_best_candidates) != 1:
        raise RuntimeError(
            "Could not identify exactly one measured bitstring for the saved "
            f"best assignment; found {len(measured_best_candidates)}"
        )
    best_bitstring = measured_best_candidates[0]

    strategy = "hardware_calibrated"
    json_row = {
        key: (value.item() if isinstance(value, np.generic) else value)
        for key, value in row.items()
    }
    json_row["angle_strategy"] = strategy
    json_row["best_bitstring"] = best_bitstring
    json_row["best_assignment"] = {
        "voxel_cst_peak_agreement_with_tom": float(
            row["best_tom_agreement_all_voxels"]
        )
    }

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "real_fractional_hardware_results.json").write_text(
        json.dumps(
            {
                "source": "30-PUB eight-neighbour Heron cohort",
                "patch_id": args.patch_id,
                "strategies": [json_row],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    with gzip.open(
        args.out / "real_fractional_hardware_counts.json.gz",
        "wt",
        encoding="utf-8",
    ) as handle:
        json.dump({strategy: patch_counts}, handle)
    np.savez_compressed(
        args.out / "real_fractional_best_assignments.npz",
        hardware_calibrated_best_spins_rcm=best_spins,
        hardware_calibrated_exact_spins_rcm=exact_spins,
    )

    manifest = {
        "patch_id": args.patch_id,
        "strategy": strategy,
        "shots": int(row["shots"]),
        "n_unique_bitstrings": len(patch_counts),
        "best_bitstring": best_bitstring,
        "best_energy": float(row["best_energy"]),
        "best_normalized_gap": float(row["best_normalized_gap"]),
        "best_agreement_with_exact_up_to_sign": float(
            row["best_agreement_with_exact_up_to_sign"]
        ),
        "source_results_csv": str(args.results_csv),
        "source_counts": str(args.counts),
        "source_best_assignments": str(args.best_assignments),
    }
    (args.out / "figure_input_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
