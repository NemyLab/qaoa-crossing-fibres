#!/usr/bin/env python3
"""
Build the first 50-node real-dMRI crossing-fibre Ising graph and run the
classical/ideal-QAOA preflight.

Run from the established crossing_fibres_step7 project directory so the script
can import:
    step1_crossing_graph.py
    step5_curved_signal_graph.py
    step9_common.py
    step9_run_10000.py (preferred) or step6_qaoa_small_graph.py

Input:
    real_patch_arrays.npz from 05_extract_real_patch_v2.py

Model:
    - 25 voxels, two CSD peak candidates per voxel -> 50 spins
    - one antiferromagnetic conflict edge per voxel
    - all four peak-pair candidates for every 8-neighbour voxel pair
    - the same amplitude-weighted angular compatibility used for the phantoms
    - the same global greedy degree pruning used for the phantoms
    - four TOM-informed boundary anchors used only to orient the bundle labels

No QPU job is submitted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

from step1_crossing_graph import CrossingGraph
from step5_curved_signal_graph import exact_milp
from step9_common import get_rcm_order, normalized_gap, reorder_graph

try:
    from step9_run_10000 import expected_energy_lightcone_p1
    EXPECTATION_SOURCE = "step9_run_10000.expected_energy_lightcone_p1"
except ImportError:
    from step6_qaoa_small_graph import expected_energy_lightcone_p1
    EXPECTATION_SOURCE = "step6_qaoa_small_graph.expected_energy_lightcone_p1"


ANGLE_STRATEGIES = {
    "transferred_center": {
        "gamma": 0.7771731744054942,
        "beta": 1.1743489498936892,
    },
    "hardware_calibrated": {
        "gamma": 0.7103982796590439,
        "beta": 1.22628080530017,
    },
}

EDGE_COLUMNS = [
    "i",
    "j",
    "edge_type",
    "weight",
    "orientation_angle_deg",
    "displacement_angle_i_deg",
    "displacement_angle_j_deg",
    "ground_truth_same_bundle",
]

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build and validate a 50-node real-dMRI CrossingGraph."
    )
    parser.add_argument(
        "--patch",
        type=Path,
        required=True,
        help="Path to real_patch_arrays.npz.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Output directory.",
    )
    parser.add_argument("--degree-cap", type=int, default=3)
    parser.add_argument("--conflict-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--sigma-orientation-deg", type=float, default=18.0)
    parser.add_argument("--sigma-displacement-deg", type=float, default=22.0)
    parser.add_argument("--minimum-continuation-weight", type=float, default=0.01)
    parser.add_argument(
        "--anchor-mode",
        choices=("boundary4", "none"),
        default="boundary4",
        help="Use four TOM-informed boundary voxels or no unary anchors.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100307,
        help="Metadata seed; no stochastic graph construction is used.",
    )
    return parser.parse_args()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"Cannot serialise {type(value).__name__}")


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= EPS:
        raise ValueError(f"Invalid zero/non-finite vector: {vector}")
    return vector / norm


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Acute angle between antipodally equivalent fibre axes."""
    cosine = float(np.clip(abs(np.dot(unit(a), unit(b))), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def gaussian_compatibility(
    orientation_angle_deg: float,
    displacement_angle_i_deg: float,
    displacement_angle_j_deg: float,
    sigma_orientation_deg: float,
    sigma_displacement_deg: float,
) -> float:
    exponent = -0.5 * (
        (orientation_angle_deg / sigma_orientation_deg) ** 2
        + (displacement_angle_i_deg / sigma_displacement_deg) ** 2
        + (displacement_angle_j_deg / sigma_displacement_deg) ** 2
    )
    return float(np.exp(exponent))


def load_patch(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {name: np.asarray(archive[name]) for name in archive.files}

    required = {
        "voxel_ijk",
        "candidate_unit_vectors",
        "candidate_amplitudes",
        "tom_match_peak",
        "tom_match_angle_deg",
        "affine",
    }
    missing = required.difference(data)
    if missing:
        raise ValueError(f"Patch archive is missing: {sorted(missing)}")

    if data["voxel_ijk"].shape != (25, 3):
        raise ValueError(
            f"Expected voxel_ijk shape (25,3), found {data['voxel_ijk'].shape}"
        )
    if data["candidate_unit_vectors"].shape != (25, 2, 3):
        raise ValueError(
            "Expected candidate_unit_vectors shape (25,2,3), found "
            f"{data['candidate_unit_vectors'].shape}"
        )
    if data["candidate_amplitudes"].shape != (25, 2):
        raise ValueError(
            f"Expected candidate_amplitudes shape (25,2), found "
            f"{data['candidate_amplitudes'].shape}"
        )
    return data


def build_nodes_and_truth(data: dict[str, np.ndarray]) -> tuple[pd.DataFrame, np.ndarray]:
    directions = np.asarray(data["candidate_unit_vectors"], dtype=float)
    amplitudes = np.asarray(data["candidate_amplitudes"], dtype=float)
    matched_peak = np.asarray(data["tom_match_peak"], dtype=int)
    matched_angle = np.asarray(data["tom_match_angle_deg"], dtype=float)
    voxel_ijk = np.asarray(data["voxel_ijk"], dtype=int)

    if not np.all(np.isin(matched_peak, [0, 1])):
        raise ValueError("Every voxel must have a TOM-matched peak index of 0 or 1.")
    if not np.all(np.isfinite(directions)):
        raise ValueError("Candidate directions contain non-finite values.")

    rows: list[dict[str, Any]] = []
    truth = np.empty(50, dtype=np.int8)

    for voxel in range(25):
        row = voxel // 5
        col = voxel % 5
        amp_sum = float(np.sum(amplitudes[voxel]))
        fraction_peak0 = (
            float(amplitudes[voxel, 0] / amp_sum) if amp_sum > EPS else float("nan")
        )
        crossing_angle = axis_angle_deg(directions[voxel, 0], directions[voxel, 1])

        for peak in range(2):
            node = 2 * voxel + peak
            pseudo_spin = +1 if peak == int(matched_peak[voxel]) else -1
            truth[node] = pseudo_spin
            direction = directions[voxel, peak]
            rows.append(
                {
                    "node_id": node,
                    "row": row,
                    "col": col,
                    "peak_type": f"real_csd_peak_{peak}",
                    "dir_x": float(direction[0]),
                    "dir_y": float(direction[1]),
                    "dir_z": float(direction[2]),
                    "amplitude": float(amplitudes[voxel, peak]),
                    "true_spin": int(pseudo_spin),
                    "fraction_a": fraction_peak0,
                    "true_crossing_angle_deg": crossing_angle,
                    "nnls_residual_norm": float("nan"),
                    "voxel_i": int(voxel_ijk[voxel, 0]),
                    "voxel_j": int(voxel_ijk[voxel, 1]),
                    "voxel_k": int(voxel_ijk[voxel, 2]),
                    "tom_match_angle_deg": float(matched_angle[voxel]),
                    "tom_matched_peak": int(matched_peak[voxel]),
                }
            )

    return pd.DataFrame(rows), truth


def boundary_anchor_voxels(tom_angles: np.ndarray) -> list[dict[str, Any]]:
    """
    Choose one unique high-confidence voxel from each 5x5 boundary side.
    These anchors only fix the global bundle-label convention.
    """
    tom_angles = np.asarray(tom_angles, dtype=float)
    sides = {
        "top": [0 * 5 + col for col in range(5)],
        "bottom": [4 * 5 + col for col in range(5)],
        "left": [row * 5 + 0 for row in range(5)],
        "right": [row * 5 + 4 for row in range(5)],
    }

    selected: list[dict[str, Any]] = []
    used: set[int] = set()
    for side, voxels in sides.items():
        ordered = sorted(voxels, key=lambda voxel: (tom_angles[voxel], voxel))
        chosen = next((voxel for voxel in ordered if voxel not in used), ordered[0])
        used.add(chosen)
        selected.append(
            {
                "side": side,
                "voxel_index": int(chosen),
                "row": int(chosen // 5),
                "col": int(chosen % 5),
                "tom_match_angle_deg": float(tom_angles[chosen]),
            }
        )
    return selected


def voxel_world_centres(
    voxel_ijk: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray:
    homogeneous = np.column_stack(
        [np.asarray(voxel_ijk, dtype=float), np.ones(len(voxel_ijk), dtype=float)]
    )
    return (np.asarray(affine, dtype=float) @ homogeneous.T).T[:, :3]


def neighbouring_voxel_pairs() -> list[tuple[int, int, str]]:
    """Return each undirected edge/corner-sharing voxel pair exactly once."""
    pairs: list[tuple[int, int, str]] = []
    offsets = (
        (0, 1, "horizontal"),
        (1, -1, "diagonal_down_left"),
        (1, 0, "vertical"),
        (1, 1, "diagonal_down_right"),
    )
    for row in range(5):
        for col in range(5):
            voxel_a = row * 5 + col
            for delta_row, delta_col, direction in offsets:
                row_b = row + delta_row
                col_b = col + delta_col
                if not (0 <= row_b < 5 and 0 <= col_b < 5):
                    continue
                voxel_b = row_b * 5 + col_b
                pairs.append((voxel_a, voxel_b, direction))
    if len(pairs) != 72:
        raise RuntimeError(f"Expected 72 grid adjacencies, found {len(pairs)}")
    return pairs


def build_graph(
    data: dict[str, np.ndarray],
    *,
    degree_cap: int,
    conflict_weight: float,
    anchor_weight: float,
    sigma_orientation_deg: float,
    sigma_displacement_deg: float,
    minimum_continuation_weight: float,
    anchor_mode: str,
    seed: int,
) -> tuple[CrossingGraph, dict[str, Any], pd.DataFrame]:
    if degree_cap < 2:
        raise ValueError("degree_cap must be at least 2 because every node has a conflict edge.")

    nodes, truth = build_nodes_and_truth(data)
    directions = np.asarray(data["candidate_unit_vectors"], dtype=float)
    amplitudes = np.asarray(data["candidate_amplitudes"], dtype=float)
    voxel_ijk = np.asarray(data["voxel_ijk"], dtype=int)
    affine = np.asarray(data["affine"], dtype=float)
    centres = voxel_world_centres(voxel_ijk, affine)
    tom_angles = np.asarray(data["tom_match_angle_deg"], dtype=float)

    edge_rows: list[dict[str, Any]] = []

    # Exactly one conflict edge per voxel.
    for voxel in range(25):
        node0 = 2 * voxel
        node1 = node0 + 1
        edge_rows.append(
            {
                "i": node0,
                "j": node1,
                "edge_type": "conflict",
                "weight": float(conflict_weight),
                "orientation_angle_deg": axis_angle_deg(
                    directions[voxel, 0], directions[voxel, 1]
                ),
                "displacement_angle_i_deg": float("nan"),
                "displacement_angle_j_deg": float("nan"),
                "ground_truth_same_bundle": False,
            }
        )

    # Match the phantom Step-5 rule exactly for continuation-edge generation:
    # inspect all four peak-to-peak links for every edge- or corner-sharing
    # voxel pair, apply the amplitude-weighted angular compatibility, threshold
    # each link separately, and only then perform global degree pruning. More
    # than one peak-pair link may therefore survive for one voxel pair.
    continuation_candidates: list[dict[str, Any]] = []
    candidate_table_rows: list[dict[str, Any]] = []

    for voxel_a, voxel_b, grid_direction in neighbouring_voxel_pairs():
        displacement = unit(centres[voxel_b] - centres[voxel_a])

        for peak_a in range(2):
            for peak_b in range(2):
                node_a = 2 * voxel_a + peak_a
                node_b = 2 * voxel_b + peak_b
                orientation_angle = axis_angle_deg(
                    directions[voxel_a, peak_a],
                    directions[voxel_b, peak_b],
                )
                displacement_angle_a = axis_angle_deg(
                    directions[voxel_a, peak_a], displacement
                )
                displacement_angle_b = axis_angle_deg(
                    directions[voxel_b, peak_b], displacement
                )
                angular_compatibility = gaussian_compatibility(
                    orientation_angle,
                    displacement_angle_a,
                    displacement_angle_b,
                    sigma_orientation_deg,
                    sigma_displacement_deg,
                )
                amplitude_term = math.sqrt(
                    max(float(amplitudes[voxel_a, peak_a]), 0.0)
                    * max(float(amplitudes[voxel_b, peak_b]), 0.0)
                )
                weight = float(amplitude_term * angular_compatibility)
                record = {
                    "voxel_a": voxel_a,
                    "voxel_b": voxel_b,
                    "grid_direction": grid_direction,
                    "peak_a": peak_a,
                    "peak_b": peak_b,
                    "i": node_a,
                    "j": node_b,
                    "amplitude_i": float(amplitudes[voxel_a, peak_a]),
                    "amplitude_j": float(amplitudes[voxel_b, peak_b]),
                    "amplitude_term": float(amplitude_term),
                    "angular_compatibility": float(angular_compatibility),
                    "weight": weight,
                    "orientation_angle_deg": orientation_angle,
                    "displacement_angle_i_deg": displacement_angle_a,
                    "displacement_angle_j_deg": displacement_angle_b,
                    "ground_truth_same_bundle": bool(
                        truth[node_a] == truth[node_b]
                    ),
                    "above_minimum_weight": bool(
                        weight >= minimum_continuation_weight
                    ),
                }
                candidate_table_rows.append(record.copy())
                if weight >= minimum_continuation_weight:
                    continuation_candidates.append(record)

    # The conflict edge already contributes one to every node degree. Continue
    # adding the strongest spatial links while respecting the total degree cap.
    degree = np.ones(50, dtype=int)
    accepted: list[dict[str, Any]] = []
    rejected_degree: list[dict[str, Any]] = []

    for candidate in sorted(
        continuation_candidates,
        key=lambda item: (
            -float(item["weight"]),
            int(item["voxel_a"]),
            int(item["voxel_b"]),
            int(item["i"]),
            int(item["j"]),
        ),
    ):
        i = int(candidate["i"])
        j = int(candidate["j"])
        if degree[i] >= degree_cap or degree[j] >= degree_cap:
            rejected_degree.append(candidate)
            continue
        degree[i] += 1
        degree[j] += 1
        accepted.append(candidate)
        edge_rows.append(
            {
                "i": i,
                "j": j,
                "edge_type": "continuation",
                "weight": float(candidate["weight"]),
                "orientation_angle_deg": float(candidate["orientation_angle_deg"]),
                "displacement_angle_i_deg": float(
                    candidate["displacement_angle_i_deg"]
                ),
                "displacement_angle_j_deg": float(
                    candidate["displacement_angle_j_deg"]
                ),
                "ground_truth_same_bundle": bool(
                    candidate["ground_truth_same_bundle"]
                ),
            }
        )

    edges = pd.DataFrame(edge_rows, columns=EDGE_COLUMNS)

    fields = np.zeros(50, dtype=float)
    selected_anchors: list[dict[str, Any]] = []
    if anchor_mode == "boundary4":
        selected_anchors = boundary_anchor_voxels(tom_angles)
        for anchor in selected_anchors:
            voxel = int(anchor["voxel_index"])
            for peak in range(2):
                node = 2 * voxel + peak
                fields[node] = float(anchor_weight) * int(truth[node])
    elif anchor_mode != "none":
        raise ValueError(f"Unknown anchor mode: {anchor_mode}")

    graph = CrossingGraph(
        nodes=nodes,
        edges=edges,
        fields=fields,
        true_spins=truth,
        grid_size=5,
        seed=int(seed),
    )

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.n_nodes))
    for edge in edges.itertuples(index=False):
        nx_graph.add_edge(int(edge.i), int(edge.j))

    accepted_same = [
        bool(item["ground_truth_same_bundle"]) for item in accepted
    ]
    accepted_diagonal = [
        item for item in accepted
        if str(item["grid_direction"]).startswith("diagonal")
    ]
    accepted_by_voxel_pair: dict[tuple[int, int], int] = {}
    for item in accepted:
        key = (int(item["voxel_a"]), int(item["voxel_b"]))
        accepted_by_voxel_pair[key] = accepted_by_voxel_pair.get(key, 0) + 1
    diagnostics = {
        "n_nodes": int(graph.n_nodes),
        "n_edges": int(len(edges)),
        "n_conflict_edges": int(np.sum(edges["edge_type"].eq("conflict"))),
        "n_continuation_edges": int(np.sum(edges["edge_type"].eq("continuation"))),
        "voxel_neighbourhood": 8,
        "n_grid_adjacencies": 72,
        "n_peak_pair_candidates": int(len(candidate_table_rows)),
        "n_candidates_above_minimum_weight": int(len(continuation_candidates)),
        "n_continuations_rejected_by_degree_cap": int(len(rejected_degree)),
        "n_diagonal_continuation_edges": int(len(accepted_diagonal)),
        "n_voxel_pairs_with_multiple_accepted_continuations": int(
            sum(count > 1 for count in accepted_by_voxel_pair.values())
        ),
        "uses_peak_amplitude_weighting": True,
        "candidate_selection_rule": (
            "all peak-pair candidates above threshold before global degree pruning"
        ),
        "degree_cap": int(degree_cap),
        "maximum_degree": int(np.max(degree)),
        "mean_degree": float(np.mean(degree)),
        "n_connected_components": int(nx.number_connected_components(nx_graph)),
        "continuation_weight_min": (
            float(min(item["weight"] for item in accepted)) if accepted else float("nan")
        ),
        "continuation_weight_median": (
            float(np.median([item["weight"] for item in accepted]))
            if accepted
            else float("nan")
        ),
        "continuation_weight_max": (
            float(max(item["weight"] for item in accepted)) if accepted else float("nan")
        ),
        "tom_consistent_continuation_fraction": (
            float(np.mean(accepted_same)) if accepted_same else float("nan")
        ),
        "anchors": selected_anchors,
        "n_anchored_voxels": int(len(selected_anchors)),
        "n_nonzero_fields": int(np.count_nonzero(fields)),
        "graph_construction_note": (
            "TOM determines pseudo-reference spins and four boundary anchors only; "
            "continuation edges use the two CSD peak axes, peak amplitudes and "
            "voxel displacement over an eight-neighbour lattice."
        ),
    }

    candidate_table = pd.DataFrame(candidate_table_rows)
    accepted_keys = {
        (int(item["voxel_a"]), int(item["voxel_b"]), int(item["i"]), int(item["j"]))
        for item in accepted
    }
    candidate_table["accepted"] = [
        (
            int(row.voxel_a),
            int(row.voxel_b),
            int(row.i),
            int(row.j),
        )
        in accepted_keys
        for row in candidate_table.itertuples(index=False)
    ]
    rejected_degree_keys = {
        (int(item["voxel_a"]), int(item["voxel_b"]), int(item["i"]), int(item["j"]))
        for item in rejected_degree
    }
    candidate_table["rejected_by_degree_cap"] = [
        (
            int(row.voxel_a),
            int(row.voxel_b),
            int(row.i),
            int(row.j),
        )
        in rejected_degree_keys
        for row in candidate_table.itertuples(index=False)
    ]
    return graph, diagnostics, candidate_table


def align_to_reference(spins: np.ndarray, reference: np.ndarray) -> tuple[np.ndarray, int]:
    spins = np.asarray(spins, dtype=np.int8)
    reference = np.asarray(reference, dtype=np.int8)
    direct = int(np.sum(spins == reference))
    flipped = int(np.sum(-spins == reference))
    sign = +1 if direct >= flipped else -1
    return (sign * spins).astype(np.int8), sign


def assignment_metrics(
    graph: CrossingGraph,
    spins: np.ndarray,
    anchored_voxels: set[int],
) -> dict[str, Any]:
    spins = np.asarray(spins, dtype=np.int8)
    truth = np.asarray(graph.true_spins, dtype=np.int8)
    aligned, sign = align_to_reference(spins, truth)

    conflict_edges = graph.edges[graph.edges["edge_type"].eq("conflict")]
    continuation_edges = graph.edges[graph.edges["edge_type"].eq("continuation")]

    conflict_satisfied = [
        int(aligned[int(edge.i)] * aligned[int(edge.j)] == -1)
        for edge in conflict_edges.itertuples(index=False)
    ]
    continuation_satisfied = [
        int(aligned[int(edge.i)] * aligned[int(edge.j)] == +1)
        for edge in continuation_edges.itertuples(index=False)
    ]
    continuation_weights = continuation_edges["weight"].to_numpy(dtype=float)

    voxel_correct = []
    unanchored_correct = []
    selected_peak = []
    for voxel in range(25):
        node0, node1 = 2 * voxel, 2 * voxel + 1
        positive = [peak for peak, node in enumerate((node0, node1)) if aligned[node] == +1]
        chosen = positive[0] if len(positive) == 1 else -1
        expected = 0 if truth[node0] == +1 else 1
        correct = int(chosen == expected)
        selected_peak.append(chosen)
        voxel_correct.append(correct)
        if voxel not in anchored_voxels:
            unanchored_correct.append(correct)

    weighted_continuation = (
        float(
            np.dot(
                continuation_weights,
                np.asarray(continuation_satisfied, dtype=float),
            )
            / np.sum(continuation_weights)
        )
        if len(continuation_weights) and np.sum(continuation_weights) > 0
        else float("nan")
    )

    return {
        "global_alignment_sign": int(sign),
        "node_accuracy_vs_tom": float(np.mean(aligned == truth)),
        "voxel_cst_peak_accuracy_vs_tom": float(np.mean(voxel_correct)),
        "unanchored_voxel_cst_peak_accuracy_vs_tom": (
            float(np.mean(unanchored_correct)) if unanchored_correct else float("nan")
        ),
        "n_unanchored_voxels": int(len(unanchored_correct)),
        "conflict_satisfaction_fraction": float(np.mean(conflict_satisfied)),
        "continuation_satisfaction_fraction": (
            float(np.mean(continuation_satisfied))
            if continuation_satisfied
            else float("nan")
        ),
        "weighted_continuation_satisfaction_fraction": weighted_continuation,
        "selected_positive_peak_by_voxel": selected_peak,
        "aligned_spins": aligned,
    }


def graph_degree_array(graph: CrossingGraph) -> np.ndarray:
    degree = np.zeros(graph.n_nodes, dtype=int)
    for edge in graph.edges.itertuples(index=False):
        degree[int(edge.i)] += 1
        degree[int(edge.j)] += 1
    return degree


def make_assignment_figure(
    graph: CrossingGraph,
    exact_aligned: np.ndarray,
    output: Path,
) -> None:
    nodes = graph.nodes.sort_values("node_id")
    truth = np.asarray(graph.true_spins, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.6), constrained_layout=True)
    panels = [
        (axes[0], truth, "TractSeg TOM pseudo-reference"),
        (axes[1], exact_aligned, "Exact MILP assignment"),
    ]

    for ax, spins, title in panels:
        for row in range(5):
            for col in range(5):
                ax.add_patch(
                    plt.Rectangle(
                        (col - 0.48, 4 - row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        linewidth=0.5,
                    )
                )

        for node in nodes.itertuples(index=False):
            x = float(node.col)
            y = float(4 - node.row)
            dx = float(node.dir_x)
            dy = float(node.dir_z)
            projected_norm = math.hypot(dx, dy)
            if projected_norm <= EPS:
                continue
            dx /= projected_norm
            dy /= projected_norm
            half = 0.34
            linestyle = "-" if int(spins[int(node.node_id)]) == +1 else "--"
            ax.plot(
                [x - half * dx, x + half * dx],
                [y - half * dy, y + half * dy],
                linestyle=linestyle,
                linewidth=2.0,
            )

        ax.set_title(title)
        ax.set_xlim(-0.65, 4.65)
        ax.set_ylim(-0.65, 4.65)
        ax.set_aspect("equal")
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xlabel("Patch column")
        ax.set_ylabel("Patch row (inferior → superior)")
        ax.grid(False)

    fig.suptitle(
        "Real HCP 5×5 crossing-fibre patch: solid = CST-labelled bundle, dashed = competitor"
    )
    fig.savefig(output.with_suffix(".png"), dpi=220)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = load_patch(args.patch)

    raw_graph, graph_diagnostics, candidate_table = build_graph(
        data,
        degree_cap=args.degree_cap,
        conflict_weight=args.conflict_weight,
        anchor_weight=args.anchor_weight,
        sigma_orientation_deg=args.sigma_orientation_deg,
        sigma_displacement_deg=args.sigma_displacement_deg,
        minimum_continuation_weight=args.minimum_continuation_weight,
        anchor_mode=args.anchor_mode,
        seed=args.seed,
    )

    # No-anchor sensitivity graph: identical nodes and edges, zero fields.
    no_anchor_graph = CrossingGraph(
        nodes=raw_graph.nodes.copy(),
        edges=raw_graph.edges.copy(),
        fields=np.zeros_like(raw_graph.fields, dtype=float),
        true_spins=np.asarray(raw_graph.true_spins, dtype=np.int8).copy(),
        grid_size=raw_graph.grid_size,
        seed=raw_graph.seed,
    )

    rcm_order = np.asarray(get_rcm_order(raw_graph), dtype=int)
    graph = reorder_graph(raw_graph, rcm_order)
    no_anchor_rcm = reorder_graph(no_anchor_graph, rcm_order)

    exact_spins_rcm, exact_energy, solver_info = exact_milp(graph)
    exact_spins_rcm = np.asarray(exact_spins_rcm, dtype=np.int8)

    no_anchor_spins_rcm, no_anchor_energy, no_anchor_solver_info = exact_milp(
        no_anchor_rcm
    )
    no_anchor_spins_rcm = np.asarray(no_anchor_spins_rcm, dtype=np.int8)

    # Map RCM-order assignments back to the original voxel/peak order.
    exact_original = np.empty(50, dtype=np.int8)
    no_anchor_original = np.empty(50, dtype=np.int8)
    rcm_index_for_original = np.empty(50, dtype=int)
    for new_index, old_index in enumerate(rcm_order):
        exact_original[int(old_index)] = exact_spins_rcm[int(new_index)]
        no_anchor_original[int(old_index)] = no_anchor_spins_rcm[int(new_index)]
        rcm_index_for_original[int(old_index)] = int(new_index)

    anchored_voxels = {
        int(item["voxel_index"]) for item in graph_diagnostics["anchors"]
    }
    primary_metrics = assignment_metrics(raw_graph, exact_original, anchored_voxels)
    no_anchor_metrics = assignment_metrics(
        no_anchor_graph, no_anchor_original, set()
    )

    aligned_primary = np.asarray(primary_metrics.pop("aligned_spins"), dtype=np.int8)
    aligned_no_anchor = np.asarray(
        no_anchor_metrics.pop("aligned_spins"), dtype=np.int8
    )
    primary_vs_no_anchor = float(np.mean(aligned_primary == aligned_no_anchor))

    qaoa_rows: list[dict[str, Any]] = []
    for strategy, angles in ANGLE_STRATEGIES.items():
        gamma = float(angles["gamma"])
        beta = float(angles["beta"])
        expected_energy = float(
            expected_energy_lightcone_p1(graph, gamma, beta)
        )
        qaoa_rows.append(
            {
                "angle_strategy": strategy,
                "gamma": gamma,
                "beta": beta,
                "ideal_expected_energy": expected_energy,
                "ideal_expected_normalized_gap": float(
                    normalized_gap(expected_energy, exact_energy)
                ),
                "expectation_source": EXPECTATION_SOURCE,
            }
        )
    qaoa_table = pd.DataFrame(qaoa_rows)

    raw_graph.nodes.to_csv(args.out / "real_graph_nodes_pre_rcm.csv", index=False)
    raw_graph.edges.to_csv(args.out / "real_graph_edges_pre_rcm.csv", index=False)
    graph.nodes.to_csv(args.out / "real_graph_nodes_rcm.csv", index=False)
    graph.edges.to_csv(args.out / "real_graph_edges_rcm.csv", index=False)
    candidate_table.to_csv(
        args.out / "real_graph_all_continuation_candidates.csv", index=False
    )
    qaoa_table.to_csv(args.out / "real_graph_ideal_qaoa.csv", index=False)

    assignment = raw_graph.nodes.copy()
    assignment["rcm_index"] = rcm_index_for_original
    assignment["field"] = np.asarray(raw_graph.fields, dtype=float)
    assignment["is_anchor_voxel"] = [
        int(row // 2) in anchored_voxels for row in range(50)
    ]
    assignment["exact_spin_raw"] = exact_original
    assignment["exact_spin_aligned_to_tom"] = aligned_primary
    assignment["no_anchor_spin_aligned_to_tom"] = aligned_no_anchor
    assignment["agrees_with_tom"] = (
        aligned_primary == np.asarray(raw_graph.true_spins)
    )
    assignment.to_csv(args.out / "real_graph_exact_assignment.csv", index=False)

    np.savez_compressed(
        args.out / "real_graph_solution_arrays.npz",
        rcm_order=rcm_order,
        fields_pre_rcm=np.asarray(raw_graph.fields, dtype=np.float64),
        true_spins_pre_rcm=np.asarray(raw_graph.true_spins, dtype=np.int8),
        exact_spins_rcm=exact_spins_rcm,
        exact_spins_pre_rcm=exact_original,
        exact_spins_pre_rcm_aligned_to_tom=aligned_primary,
        no_anchor_spins_pre_rcm_aligned_to_tom=aligned_no_anchor,
    )

    make_assignment_figure(
        raw_graph,
        aligned_primary,
        args.out / "real_graph_assignment_comparison",
    )

    degree = graph_degree_array(raw_graph)
    summary = {
        "input_patch": str(args.patch.resolve()),
        "parameters": {
            "voxel_neighbourhood": 8,
            "continuation_candidate_rule": (
                "all four peak-pair candidates per edge/corner-sharing voxel pair"
            ),
            "uses_peak_amplitude_weighting": True,
            "degree_cap": int(args.degree_cap),
            "conflict_weight": float(args.conflict_weight),
            "anchor_weight": float(args.anchor_weight),
            "sigma_orientation_deg": float(args.sigma_orientation_deg),
            "sigma_displacement_deg": float(args.sigma_displacement_deg),
            "minimum_continuation_weight": float(
                args.minimum_continuation_weight
            ),
            "anchor_mode": args.anchor_mode,
            "angle_strategies": ANGLE_STRATEGIES,
        },
        "graph": graph_diagnostics,
        "degree_distribution": {
            str(value): int(np.sum(degree == value))
            for value in sorted(np.unique(degree))
        },
        "rcm": {
            "order": rcm_order.tolist(),
            "is_valid_permutation": bool(
                np.array_equal(np.sort(rcm_order), np.arange(50))
            ),
            "energy_invariant": bool(
                np.isclose(
                    raw_graph.energy(exact_original),
                    graph.energy(exact_spins_rcm),
                    atol=1e-9,
                )
            ),
        },
        "exact_anchored": {
            "energy": float(exact_energy),
            "solver_info": solver_info,
            **primary_metrics,
        },
        "exact_no_anchor_sensitivity": {
            "energy": float(no_anchor_energy),
            "solver_info": no_anchor_solver_info,
            **no_anchor_metrics,
            "assignment_agreement_with_anchored_after_tom_alignment": (
                primary_vs_no_anchor
            ),
        },
        "ideal_qaoa": qaoa_rows,
        "safety_checks": {
            "n_nodes_is_50": bool(raw_graph.n_nodes == 50),
            "n_conflict_edges_is_25": bool(
                graph_diagnostics["n_conflict_edges"] == 25
            ),
            "uses_eight_neighbour_lattice": bool(
                graph_diagnostics["voxel_neighbourhood"] == 8
            ),
            "n_grid_adjacencies_is_72": bool(
                graph_diagnostics["n_grid_adjacencies"] == 72
            ),
            "n_peak_pair_candidates_is_288": bool(
                graph_diagnostics["n_peak_pair_candidates"] == 288
            ),
            "uses_peak_amplitude_weighting": bool(
                graph_diagnostics["uses_peak_amplitude_weighting"]
            ),
            "maximum_degree_within_cap": bool(
                graph_diagnostics["maximum_degree"] <= args.degree_cap
            ),
            "all_conflicts_satisfied_by_exact": bool(
                np.isclose(
                    primary_metrics["conflict_satisfaction_fraction"], 1.0
                )
            ),
            "rcm_energy_invariant": bool(
                np.isclose(
                    raw_graph.energy(exact_original),
                    graph.energy(exact_spins_rcm),
                    atol=1e-9,
                )
            ),
            "uses_qpu": False,
            "uses_shot_sampling": False,
        },
        "interpretation": {
            "tom_role": (
                "TractSeg TOM is used as a pseudo-reference and for four boundary "
                "anchors. It is not treated as biological ground truth."
            ),
            "primary_real_data_test": (
                "The most informative first checks are unanchored-voxel TOM agreement, "
                "agreement between anchored and no-anchor optima, graph connectivity, "
                "and ideal frozen-angle QAOA normalized gap."
            ),
            "next_gate": (
                "Do not prepare a QPU circuit until the classical graph and ideal "
                "QAOA outputs have been reviewed."
            ),
        },
        "files": {
            "nodes_pre_rcm": "real_graph_nodes_pre_rcm.csv",
            "edges_pre_rcm": "real_graph_edges_pre_rcm.csv",
            "nodes_rcm": "real_graph_nodes_rcm.csv",
            "edges_rcm": "real_graph_edges_rcm.csv",
            "all_continuation_candidates": (
                "real_graph_all_continuation_candidates.csv"
            ),
            "exact_assignment": "real_graph_exact_assignment.csv",
            "ideal_qaoa": "real_graph_ideal_qaoa.csv",
            "solution_arrays": "real_graph_solution_arrays.npz",
            "assignment_figure_png": "real_graph_assignment_comparison.png",
            "assignment_figure_pdf": "real_graph_assignment_comparison.pdf",
        },
    }

    summary_path = args.out / "real_graph_validation.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print("\nREAL-DMRI GRAPH PREFLIGHT")
    print("=" * 72)
    print(
        f"Nodes: {raw_graph.n_nodes}; edges: {len(raw_graph.edges)} "
        f"({graph_diagnostics['n_continuation_edges']} continuation + "
        f"{graph_diagnostics['n_conflict_edges']} conflict)"
    )
    print(
        f"Components: {graph_diagnostics['n_connected_components']}; "
        f"maximum degree: {graph_diagnostics['maximum_degree']}"
    )
    print(f"Exact energy: {float(exact_energy):.12g}")
    print(
        "TOM agreement, all/unanchored voxels: "
        f"{primary_metrics['voxel_cst_peak_accuracy_vs_tom']:.3f} / "
        f"{primary_metrics['unanchored_voxel_cst_peak_accuracy_vs_tom']:.3f}"
    )
    print(
        "Anchored vs no-anchor assignment agreement: "
        f"{primary_vs_no_anchor:.3f}"
    )
    print("\nIdeal p=1 QAOA:")
    print(qaoa_table.to_string(index=False))
    print(f"\nWrote outputs to: {args.out.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
