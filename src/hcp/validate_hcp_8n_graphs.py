#!/usr/bin/env python3
"""Validate the representative and 30-subject eight-neighbour HCP graphs.

This validator is intentionally independent of Qiskit. Run it after rebuilding
the graphs and before ideal simulation or hardware preflight.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EXPECTED_GRID_DIRECTIONS = {
    "horizontal": 80,
    "vertical": 80,
    "diagonal_down_left": 64,
    "diagonal_down_right": 64,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--representative",
        type=Path,
        required=True,
        help="Representative graph_validation directory.",
    )
    parser.add_argument(
        "--replication-root",
        type=Path,
        required=True,
        help="Root containing 30 <patch_id>/graph_validation directories.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parents[2] / "data" / "hcp" / "hcp_patch_manifest.csv",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser.parse_args()


def load_manifest(path: Path) -> list[str]:
    if not path.is_file():
        raise FileNotFoundError(path)

    frame = pd.read_csv(path)

    if "patch_id" not in frame.columns:
        raise ValueError(f"{path} does not contain a 'patch_id' column")

    values = frame["patch_id"].astype(str).tolist()

    if len(values) != 30 or len(set(values)) != 30:
        raise ValueError(
            f"Expected 30 unique patch identifiers in {path}; found {len(values)}"
        )

    return values


def require_file(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def check_graph(label: str, graph_dir: Path) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    summary = json.loads(
        require_file(graph_dir / "real_graph_validation.json").read_text(
            encoding="utf-8"
        )
    )
    edges = pd.read_csv(require_file(graph_dir / "real_graph_edges_pre_rcm.csv"))
    candidates = pd.read_csv(
        require_file(graph_dir / "real_graph_all_continuation_candidates.csv")
    )

    graph = summary.get("graph", {})
    safety = summary.get("safety_checks", {})
    parameters = summary.get("parameters", {})

    expected_equal = {
        "voxel_neighbourhood": (graph.get("voxel_neighbourhood"), 8),
        "n_grid_adjacencies": (graph.get("n_grid_adjacencies"), 72),
        "n_peak_pair_candidates": (graph.get("n_peak_pair_candidates"), 288),
        "n_nodes": (graph.get("n_nodes"), 50),
        "n_conflict_edges": (graph.get("n_conflict_edges"), 25),
        "degree_cap": (graph.get("degree_cap"), 3),
        "parameter_voxel_neighbourhood": (
            parameters.get("voxel_neighbourhood"),
            8,
        ),
    }
    for name, (observed, expected) in expected_equal.items():
        if observed != expected:
            errors.append(f"{name}: expected {expected!r}, found {observed!r}")

    if graph.get("uses_peak_amplitude_weighting") is not True:
        errors.append("uses_peak_amplitude_weighting is not true")
    if parameters.get("uses_peak_amplitude_weighting") is not True:
        errors.append("parameters do not record amplitude weighting")
    if int(graph.get("maximum_degree", 999)) > 3:
        errors.append(f"maximum degree exceeds 3: {graph.get('maximum_degree')}")
    if int(graph.get("n_connected_components", 999)) != 1:
        errors.append(
            f"graph is not connected: {graph.get('n_connected_components')} components"
        )

    for name in (
        "uses_eight_neighbour_lattice",
        "n_grid_adjacencies_is_72",
        "n_peak_pair_candidates_is_288",
        "uses_peak_amplitude_weighting",
        "maximum_degree_within_cap",
        "n_nodes_is_50",
        "n_conflict_edges_is_25",
        "all_conflicts_satisfied_by_exact",
        "rcm_energy_invariant",
    ):
        if safety.get(name) is not True:
            errors.append(f"safety check failed or absent: {name}")

    if len(candidates) != 288:
        errors.append(f"candidate CSV has {len(candidates)} rows, expected 288")
    required_candidate_columns = {
        "grid_direction",
        "amplitude_i",
        "amplitude_j",
        "amplitude_term",
        "angular_compatibility",
        "weight",
        "above_minimum_weight",
        "accepted",
        "rejected_by_degree_cap",
    }
    missing = required_candidate_columns - set(candidates.columns)
    if missing:
        errors.append(f"candidate CSV is missing columns: {sorted(missing)}")
    else:
        direction_counts = candidates["grid_direction"].value_counts().to_dict()
        if direction_counts != EXPECTED_GRID_DIRECTIONS:
            errors.append(
                "candidate direction counts differ: "
                f"expected {EXPECTED_GRID_DIRECTIONS}, found {direction_counts}"
            )
        reconstructed = (
            candidates["amplitude_term"].to_numpy(dtype=float)
            * candidates["angular_compatibility"].to_numpy(dtype=float)
        )
        if not np.allclose(
            candidates["weight"].to_numpy(dtype=float),
            reconstructed,
            rtol=1e-10,
            atol=1e-12,
        ):
            errors.append("candidate weights do not equal amplitude × angular terms")
        accepted_below = candidates.loc[
            candidates["accepted"].astype(bool)
            & ~candidates["above_minimum_weight"].astype(bool)
        ]
        if len(accepted_below):
            errors.append(f"{len(accepted_below)} accepted candidates are below threshold")

    continuation = edges.loc[edges["edge_type"].astype(str) == "continuation"]
    conflicts = edges.loc[edges["edge_type"].astype(str) == "conflict"]
    if len(conflicts) != 25:
        errors.append(f"edge CSV has {len(conflicts)} conflicts, expected 25")
    if "accepted" in candidates and int(candidates["accepted"].astype(bool).sum()) != len(
        continuation
    ):
        errors.append("accepted candidate count differs from continuation-edge count")

    diagonal_candidates = int(
        candidates["grid_direction"].astype(str).str.startswith("diagonal").sum()
    )
    row = {
        "case": label,
        "graph_dir": str(graph_dir.resolve()),
        "n_nodes": int(graph.get("n_nodes", -1)),
        "n_conflict_edges": int(graph.get("n_conflict_edges", -1)),
        "n_continuation_edges": int(graph.get("n_continuation_edges", -1)),
        "n_diagonal_candidates": diagonal_candidates,
        "n_diagonal_continuation_edges": int(
            graph.get("n_diagonal_continuation_edges", -1)
        ),
        "n_voxel_pairs_with_multiple_accepted_continuations": int(
            graph.get("n_voxel_pairs_with_multiple_accepted_continuations", -1)
        ),
        "maximum_degree": int(graph.get("maximum_degree", -1)),
        "n_connected_components": int(graph.get("n_connected_components", -1)),
        "exact_energy": float(summary.get("exact_anchored", {}).get("energy", np.nan)),
        "tom_agreement_all_voxels": float(
            summary.get("exact_anchored", {}).get(
                "voxel_cst_peak_accuracy_vs_tom", np.nan
            )
        ),
        "tom_agreement_unanchored_voxels": float(
            summary.get("exact_anchored", {}).get(
                "unanchored_voxel_cst_peak_accuracy_vs_tom", np.nan
            )
        ),
        "validation_passed": not errors,
        "validation_errors": " | ".join(errors),
    }
    return row, errors


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    patch_ids = load_manifest(args.manifest)

    rows: list[dict[str, Any]] = []
    failures: dict[str, list[str]] = {}

    representative_row, representative_errors = check_graph(
        "103414_R_88_86_76", args.representative
    )
    rows.append(representative_row)
    if representative_errors:
        failures["103414_R_88_86_76"] = representative_errors

    observed_dirs = {
        path.parent.name
        for path in args.replication_root.glob("*/graph_validation")
        if path.is_dir()
    }
    expected_dirs = set(patch_ids)
    missing_dirs = sorted(expected_dirs - observed_dirs)
    unexpected_dirs = sorted(observed_dirs - expected_dirs)
    if missing_dirs:
        failures["cohort_missing"] = missing_dirs
    if unexpected_dirs:
        failures["cohort_unexpected"] = unexpected_dirs

    for patch_id in patch_ids:
        graph_dir = args.replication_root / patch_id / "graph_validation"
        try:
            row, errors = check_graph(patch_id, graph_dir)
        except Exception as exc:
            row = {
                "case": patch_id,
                "graph_dir": str(graph_dir.resolve()),
                "validation_passed": False,
                "validation_errors": f"{type(exc).__name__}: {exc}",
            }
            errors = [row["validation_errors"]]
        rows.append(row)
        if errors:
            failures[patch_id] = errors

    frame = pd.DataFrame(rows)
    frame.to_csv(args.out / "hcp_8n_graph_validation.csv", index=False)
    report = {
        "representative_cases": 1,
        "replication_cases_expected": 30,
        "cases_checked": int(len(frame)),
        "cases_passed": int(frame["validation_passed"].fillna(False).sum()),
        "all_passed": not failures,
        "cohort_manifest": patch_ids,
        "failures": failures,
    }
    (args.out / "hcp_8n_graph_validation.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )

    print(frame.to_string(index=False))
    print(
        f"\nPassed {report['cases_passed']}/{report['cases_checked']} graph checks."
    )
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
