#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from step1_crossing_graph import CrossingGraph, axis_angle_deg, exact_enumeration, unit
from step4_signal_derived_graph import (
    build_response_dictionary,
    extract_two_peaks,
    fibonacci_measurements,
    planar_orientation_dictionary,
    tensor_response,
)


GEOMETRIES = {
    "straight_90": {"base_a": 0.0, "base_b": 90.0, "curvature": 0.0},
    "curved_90": {"base_a": 0.0, "base_b": 90.0, "curvature": 20.0},
    "curved_60": {"base_a": 15.0, "base_b": 75.0, "curvature": 14.0},
    "curved_45": {"base_a": 22.5, "base_b": 67.5, "curvature": 10.0},
    "curved_90_rot45": {"base_a": 45.0, "base_b": 135.0, "curvature": 20.0},
    "curved_60_rot45": {"base_a": 60.0, "base_b": 120.0, "curvature": 14.0},
    "curved_45_rot45": {"base_a": 67.5, "base_b": 112.5, "curvature": 10.0},
}


def direction_from_angle(angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    return np.array([np.cos(angle), np.sin(angle), 0.0], dtype=float)


def true_directions(
    row: int,
    col: int,
    grid_size: int,
    geometry: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    if geometry not in GEOMETRIES:
        raise ValueError(f"Unknown geometry: {geometry}")
    spec = GEOMETRIES[geometry]
    center = (grid_size - 1) / 2.0
    scale = max(center, 1.0)
    x = (col - center) / scale
    y = (row - center) / scale

    # Smoothly curved crossing fields. Bundle A is predominantly left-right;
    # bundle B is predominantly top-bottom. Cross-terms prevent the geometry
    # from reducing to identical rows or columns.
    angle_a = spec["base_a"] + spec["curvature"] * (0.80 * x + 0.20 * y)
    angle_b = spec["base_b"] + spec["curvature"] * (-0.80 * y + 0.20 * x)
    a = direction_from_angle(angle_a)
    b = direction_from_angle(angle_b)
    crossing = axis_angle_deg(a, b)
    return a, b, crossing


def fraction_a_for_voxel(
    row: int,
    col: int,
    grid_size: int,
    fraction_mode: str,
    rng: np.random.Generator,
) -> float:
    center = (grid_size - 1) / 2.0
    scale = max(center, 1.0)
    x = (col - center) / scale
    y = (row - center) / scale
    if fraction_mode == "balanced":
        return 0.5
    if fraction_mode == "variable":
        value = 0.5 + 0.13 * x - 0.07 * y + rng.normal(0.0, 0.02)
        return float(np.clip(value, 0.30, 0.70))
    if fraction_mode == "imbalanced":
        dominant_a = (row + col) % 2 == 0
        mean = 0.72 if dominant_a else 0.28
        return float(np.clip(mean + rng.normal(0.0, 0.025), 0.20, 0.80))
    raise ValueError(f"Unknown fraction mode: {fraction_mode}")


def simulate_signal(
    gradients: np.ndarray,
    direction_a: np.ndarray,
    direction_b: np.ndarray,
    fraction_a: float,
    snr: float,
    b_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    clean = (
        fraction_a * tensor_response(gradients, direction_a, b_value=b_value)
        + (1.0 - fraction_a)
        * tensor_response(gradients, direction_b, b_value=b_value)
    )
    sigma = 1.0 / max(float(snr), 1e-8)
    real = rng.normal(0.0, sigma, size=len(gradients))
    imag = rng.normal(0.0, sigma, size=len(gradients))
    return np.sqrt(np.square(clean + real) + np.square(imag))


def match_peaks(
    estimated: np.ndarray,
    truth_a: np.ndarray,
    truth_b: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    truth = [truth_a, truth_b]
    best = None
    for perm in permutations((0, 1)):
        errors = np.array(
            [
                axis_angle_deg(estimated[perm[0]], truth[0]),
                axis_angle_deg(estimated[perm[1]], truth[1]),
            ]
        )
        total = float(errors.sum())
        if best is None or total < best[0]:
            best = (total, perm, errors)
    assert best is not None
    _, perm, errors = best
    spins = np.empty(2, dtype=int)
    spins[perm[0]] = +1
    spins[perm[1]] = -1
    return spins, float(errors.mean()), float(errors.max())


def exact_milp(graph: CrossingGraph) -> tuple[np.ndarray, float, dict]:
    """Exact sparse Ising solver using SciPy/HiGHS MILP linearization."""
    n = graph.n_nodes
    edge_records = list(graph.edges.itertuples(index=False))
    m = len(edge_records)
    n_variables = n + m
    c = np.zeros(n_variables, dtype=float)
    constant = float(np.sum(graph.fields))

    # -h_i s_i with s_i=2x_i-1.
    c[:n] += -2.0 * graph.fields

    rows = []
    cols = []
    data = []
    lower = []
    upper = []
    row_id = 0

    for e, edge in enumerate(edge_records):
        i, j = int(edge.i), int(edge.j)
        q = -float(edge.weight) if edge.edge_type == "continuation" else float(edge.weight)
        y = n + e
        # q s_i s_j = q(4y - 2x_i - 2x_j + 1)
        c[i] += -2.0 * q
        c[j] += -2.0 * q
        c[y] += 4.0 * q
        constant += q

        # y <= x_i
        rows += [row_id, row_id]
        cols += [y, i]
        data += [1.0, -1.0]
        lower.append(-np.inf)
        upper.append(0.0)
        row_id += 1
        # y <= x_j
        rows += [row_id, row_id]
        cols += [y, j]
        data += [1.0, -1.0]
        lower.append(-np.inf)
        upper.append(0.0)
        row_id += 1
        # y >= x_i + x_j - 1  => y-x_i-x_j >= -1
        rows += [row_id, row_id, row_id]
        cols += [y, i, j]
        data += [1.0, -1.0, -1.0]
        lower.append(-1.0)
        upper.append(np.inf)
        row_id += 1

    matrix = coo_matrix((data, (rows, cols)), shape=(row_id, n_variables)).tocsr()
    constraints = LinearConstraint(matrix, np.asarray(lower), np.asarray(upper))
    result = milp(
        c=c,
        integrality=np.ones(n_variables, dtype=int),
        bounds=Bounds(np.zeros(n_variables), np.ones(n_variables)),
        constraints=constraints,
        options={"time_limit": 120.0, "mip_rel_gap": 0.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    binary = np.rint(result.x[:n]).astype(int)
    spins = 2 * binary - 1
    direct_energy = graph.energy(spins)
    solver_energy = float(result.fun + constant)
    if not np.isclose(direct_energy, solver_energy, atol=1e-6):
        raise RuntimeError(
            f"MILP/direct energy mismatch: {solver_energy} vs {direct_energy}"
        )
    info = {
        "solver_status": int(result.status),
        "solver_message": str(result.message),
        "mip_gap": float(getattr(result, "mip_gap", np.nan)),
        "mip_node_count": int(getattr(result, "mip_node_count", -1)),
    }
    return spins, float(direct_energy), info


def build_curved_signal_graph(
    *,
    grid_size: int,
    n_gradients: int,
    snr: float,
    b_value: float,
    geometry: str,
    fraction_mode: str,
    degree_cap: int,
    conflict_weight: float,
    anchor_weight: float,
    sigma_orientation_deg: float,
    sigma_displacement_deg: float,
    minimum_continuation_weight: float,
    seed: int,
) -> tuple[CrossingGraph, dict]:
    if degree_cap not in (3, 4):
        raise ValueError("degree_cap must be 3 or 4")
    rng = np.random.default_rng(seed)
    gradients = fibonacci_measurements(n_gradients)
    axes = planar_orientation_dictionary(step_deg=2.0)
    dictionary = build_response_dictionary(gradients, axes, b_value=b_value)

    node_rows = []
    node_lookup = {}
    true_axes_by_voxel = {}
    peak_mean_errors = []
    peak_max_errors = []
    residuals = []
    crossings = []

    for row in range(grid_size):
        for col in range(grid_size):
            truth_a, truth_b, crossing = true_directions(
                row, col, grid_size, geometry
            )
            true_axes_by_voxel[(row, col)] = (truth_a, truth_b)
            crossings.append(crossing)
            fraction_a = fraction_a_for_voxel(
                row, col, grid_size, fraction_mode, rng
            )
            signal = simulate_signal(
                gradients,
                truth_a,
                truth_b,
                fraction_a,
                snr,
                b_value,
                rng,
            )
            estimated, amplitudes, residual = extract_two_peaks(
                signal,
                dictionary,
                axes,
                minimum_separation_deg=25.0,
            )
            true_spins, mean_error, max_error = match_peaks(
                estimated, truth_a, truth_b
            )
            peak_mean_errors.append(mean_error)
            peak_max_errors.append(max_error)
            residuals.append(residual)

            for peak_index in (0, 1):
                node_id = len(node_rows)
                direction = estimated[peak_index]
                node_rows.append(
                    {
                        "node_id": node_id,
                        "row": row,
                        "col": col,
                        "peak_type": f"estimated_peak_{peak_index}",
                        "dir_x": direction[0],
                        "dir_y": direction[1],
                        "dir_z": direction[2],
                        "amplitude": float(amplitudes[peak_index]),
                        "true_spin": int(true_spins[peak_index]),
                        "fraction_a": fraction_a,
                        "true_crossing_angle_deg": crossing,
                        "nnls_residual_norm": residual,
                    }
                )
                node_lookup[(row, col, peak_index)] = node_id

    nodes = pd.DataFrame(node_rows)
    directions = nodes[["dir_x", "dir_y", "dir_z"]].to_numpy(float)
    amplitudes = nodes["amplitude"].to_numpy(float)
    edges = []
    degree = np.zeros(len(nodes), dtype=int)
    used = set()

    # Mandatory within-voxel conflicts consume one degree per node.
    for row in range(grid_size):
        for col in range(grid_size):
            i = node_lookup[(row, col, 0)]
            j = node_lookup[(row, col, 1)]
            edges.append(
                {
                    "i": min(i, j),
                    "j": max(i, j),
                    "edge_type": "conflict",
                    "weight": float(conflict_weight),
                    "orientation_angle_deg": axis_angle_deg(directions[i], directions[j]),
                    "displacement_angle_i_deg": np.nan,
                    "displacement_angle_j_deg": np.nan,
                    "ground_truth_same_bundle": False,
                }
            )
            degree[i] += 1
            degree[j] += 1
            used.add((min(i, j), max(i, j), "conflict"))

    # All geometrically plausible edges over an 8-neighbour voxel lattice.
    candidates = []
    neighbour_offsets = [(0, 1), (1, -1), (1, 0), (1, 1)]
    for row in range(grid_size):
        for col in range(grid_size):
            for dr, dc in neighbour_offsets:
                row2, col2 = row + dr, col + dc
                if not (0 <= row2 < grid_size and 0 <= col2 < grid_size):
                    continue
                displacement = unit(np.array([float(dc), float(dr), 0.0]))
                for peak_1 in (0, 1):
                    i = node_lookup[(row, col, peak_1)]
                    for peak_2 in (0, 1):
                        j = node_lookup[(row2, col2, peak_2)]
                        theta = axis_angle_deg(directions[i], directions[j])
                        phi_i = axis_angle_deg(directions[i], displacement)
                        phi_j = axis_angle_deg(directions[j], displacement)
                        orientation_term = np.exp(-0.5 * (theta / sigma_orientation_deg) ** 2)
                        displacement_term = np.exp(
                            -0.5
                            * (
                                (phi_i / sigma_displacement_deg) ** 2
                                + (phi_j / sigma_displacement_deg) ** 2
                            )
                        )
                        amplitude_term = np.sqrt(
                            max(amplitudes[i], 0.0) * max(amplitudes[j], 0.0)
                        )
                        weight = float(amplitude_term * orientation_term * displacement_term)
                        if weight >= minimum_continuation_weight:
                            candidates.append((weight, i, j, theta, phi_i, phi_j))

    # Greedy maximum-weight b-matching approximation. Since conflict edges
    # already consume one degree, continuation degree is at most 2 or 3.
    candidates.sort(reverse=True, key=lambda item: item[0])
    for weight, i, j, theta, phi_i, phi_j in candidates:
        key = (min(i, j), max(i, j), "continuation")
        if key in used:
            continue
        if degree[i] >= degree_cap or degree[j] >= degree_cap:
            continue
        edges.append(
            {
                "i": min(i, j),
                "j": max(i, j),
                "edge_type": "continuation",
                "weight": weight,
                "orientation_angle_deg": theta,
                "displacement_angle_i_deg": phi_i,
                "displacement_angle_j_deg": phi_j,
                "ground_truth_same_bundle": bool(
                    nodes.iloc[i]["true_spin"] == nodes.iloc[j]["true_spin"]
                ),
            }
        )
        degree[i] += 1
        degree[j] += 1
        used.add(key)

    edges_df = pd.DataFrame(edges)
    fields = np.zeros(len(nodes), dtype=float)

    # Endpoint-style priors derived from known phantom entry regions, not
    # from the labels of individual estimated peaks.
    left_col = 0
    top_row = 0
    for row in range(grid_size):
        truth_a, _, _ = true_directions(row, left_col, grid_size, geometry)
        options = [node_lookup[(row, left_col, p)] for p in (0, 1)]
        chosen = min(options, key=lambda i: axis_angle_deg(directions[i], truth_a))
        fields[chosen] += anchor_weight
    for col in range(grid_size):
        _, truth_b, _ = true_directions(top_row, col, grid_size, geometry)
        options = [node_lookup[(top_row, col, p)] for p in (0, 1)]
        chosen = min(options, key=lambda i: axis_angle_deg(directions[i], truth_b))
        fields[chosen] -= anchor_weight

    graph = CrossingGraph(
        nodes=nodes,
        edges=edges_df,
        fields=fields,
        true_spins=nodes["true_spin"].to_numpy(dtype=int),
        grid_size=grid_size,
        seed=seed,
    )
    continuation = edges_df[edges_df["edge_type"] == "continuation"]
    diagnostics = {
        "mean_peak_error_deg": float(np.mean(peak_mean_errors)),
        "mean_maximum_peak_error_deg": float(np.mean(peak_max_errors)),
        "mean_nnls_residual_norm": float(np.mean(residuals)),
        "mean_true_crossing_angle_deg": float(np.mean(crossings)),
        "minimum_true_crossing_angle_deg": float(np.min(crossings)),
        "correct_continuation_fraction": (
            float(continuation["ground_truth_same_bundle"].mean())
            if len(continuation) else np.nan
        ),
        "n_connected_components_proxy": np.nan,
    }
    return graph, diagnostics


def plot_solution(graph: CrossingGraph, spins: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    for node in graph.nodes.itertuples(index=False):
        i = int(node.node_id)
        direction = np.array([node.dir_x, node.dir_y], dtype=float)
        direction /= max(np.linalg.norm(direction), 1e-12)
        dx, dy = 0.34 * direction
        style = "-" if spins[i] == +1 else "--"
        width = 1.0 + 2.0 * min(float(node.amplitude), 1.0)
        ax.plot(
            [node.col - dx, node.col + dx],
            [node.row - dy, node.row + dy],
            linestyle=style,
            linewidth=width,
        )
    ax.set_title(title)
    ax.set_xlabel("Voxel column")
    ax.set_ylabel("Voxel row")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def evaluate_case(args, geometry: str, snr: float, degree_cap: int, seed: int):
    graph, diagnostics = build_curved_signal_graph(
        grid_size=args.grid_size,
        n_gradients=args.n_gradients,
        snr=snr,
        b_value=args.b_value,
        geometry=geometry,
        fraction_mode=args.fraction_mode,
        degree_cap=degree_cap,
        conflict_weight=args.conflict_weight,
        anchor_weight=args.anchor_weight,
        sigma_orientation_deg=args.sigma_orientation_deg,
        sigma_displacement_deg=args.sigma_displacement_deg,
        minimum_continuation_weight=args.minimum_continuation_weight,
        seed=seed,
    )
    spins, energy, solver = exact_milp(graph)
    truth_energy = graph.energy(graph.true_spins)
    accuracy = float(np.mean(spins == graph.true_spins))
    degree = graph.degree_table()["degree"]
    return (
        {
            "seed": seed,
            "geometry": geometry,
            "snr": snr,
            "degree_cap": degree_cap,
            "fraction_mode": args.fraction_mode,
            "n_nodes_qubits": graph.n_nodes,
            "n_edges": len(graph.edges),
            "n_continuation_edges": int((graph.edges["edge_type"] == "continuation").sum()),
            "observed_maximum_degree": int(degree.max()),
            "mean_degree": float(degree.mean()),
            "label_accuracy": accuracy,
            "exact_recovery": bool(np.isclose(accuracy, 1.0)),
            "truth_energy": truth_energy,
            "exact_energy": energy,
            "truth_optimality_gap": float(truth_energy - energy),
            "truth_is_optimal": bool(np.isclose(truth_energy, energy, atol=1e-7)),
            **diagnostics,
            **solver,
        },
        graph,
        spins,
    )


def recommend(summary: pd.DataFrame, target: float) -> dict:
    result = {"target_exact_recovery_rate": target}
    for cap in sorted(summary["degree_cap"].unique()):
        table = summary.copy()
        table = table[table["degree_cap"] == cap]
        table = table[
            (table["mean_label_accuracy"] >= 0.80)
            & (table["exact_recovery_rate"] < 1.0)
        ].copy()
        if table.empty:
            result[f"degree_{int(cap)}"] = None
            continue
        table["distance"] = abs(table["exact_recovery_rate"] - target)
        row = table.sort_values(
            ["distance", "mean_label_accuracy"], ascending=[True, False]
        ).iloc[0]
        result[f"degree_{int(cap)}"] = {
            "geometry": str(row["geometry"]),
            "snr": float(row["snr"]),
            "fraction_mode": str(row["fraction_mode"]),
            "observed_exact_recovery_rate": float(row["exact_recovery_rate"]),
            "observed_mean_label_accuracy": float(row["mean_label_accuracy"]),
            "observed_correct_continuation_fraction": float(
                row["mean_correct_continuation_fraction"]
            ),
            "observed_maximum_degree": int(row["observed_maximum_degree"]),
            "n_nodes_qubits": int(row["n_nodes_qubits"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--n-gradients", type=int, default=60)
    parser.add_argument("--b-value", type=float, default=2000.0)
    parser.add_argument("--geometries", nargs="+", choices=list(GEOMETRIES), default=list(GEOMETRIES))
    parser.add_argument("--snr-values", nargs="+", type=float, default=[20.0, 15.0, 10.0, 8.0])
    parser.add_argument("--degree-caps", nargs="+", type=int, choices=[3, 4], default=[3, 4])
    parser.add_argument("--fraction-mode", choices=["balanced", "variable", "imbalanced"], default="imbalanced")
    parser.add_argument("--conflict-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--sigma-orientation-deg", type=float, default=18.0)
    parser.add_argument("--sigma-displacement-deg", type=float, default=22.0)
    parser.add_argument("--minimum-continuation-weight", type=float, default=0.01)
    parser.add_argument("--target-recovery", type=float, default=0.70)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    example_saved = False
    for geometry in args.geometries:
        for snr in args.snr_values:
            for cap in args.degree_caps:
                for seed in range(args.n_seeds):
                    record, graph, spins = evaluate_case(args, geometry, snr, cap, seed)
                    records.append(record)
                    if (
                        not example_saved
                        and geometry == "curved_60"
                        and snr == 10.0
                        and cap == 3
                    ):
                        graph.nodes.assign(field=graph.fields, predicted_spin=spins).to_csv(
                            output / "example_curved_nodes.csv", index=False
                        )
                        graph.edges.to_csv(output / "example_curved_edges.csv", index=False)
                        plot_solution(
                            graph,
                            spins,
                            output / "example_curved_solution.png",
                            "Curved signal-derived graph: solid A, dashed B",
                        )
                        example_saved = True

    results = pd.DataFrame(records)
    summary = (
        results.groupby(
            ["geometry", "snr", "degree_cap", "fraction_mode"], as_index=False
        )
        .agg(
            n_seeds=("seed", "size"),
            n_nodes_qubits=("n_nodes_qubits", "first"),
            mean_peak_error_deg=("mean_peak_error_deg", "mean"),
            mean_crossing_angle_deg=("mean_true_crossing_angle_deg", "mean"),
            minimum_crossing_angle_deg=("minimum_true_crossing_angle_deg", "min"),
            mean_correct_continuation_fraction=("correct_continuation_fraction", "mean"),
            mean_label_accuracy=("label_accuracy", "mean"),
            exact_recovery_rate=("exact_recovery", "mean"),
            truth_optimal_rate=("truth_is_optimal", "mean"),
            mean_truth_optimality_gap=("truth_optimality_gap", "mean"),
            mean_edges=("n_edges", "mean"),
            observed_maximum_degree=("observed_maximum_degree", "max"),
            mean_mip_nodes=("mip_node_count", "mean"),
        )
        .sort_values(["degree_cap", "geometry", "snr"], ascending=[True, True, False])
    )
    recommendations = recommend(summary, args.target_recovery)
    results.to_csv(output / "curved_signal_results.csv", index=False)
    summary.to_csv(output / "curved_signal_summary.csv", index=False)
    (output / "recommended_curved_scenarios.json").write_text(
        json.dumps(recommendations, indent=2)
    )

    fig, ax = plt.subplots(figsize=(10, 6))
    for (geometry, cap), group in summary.groupby(["geometry", "degree_cap"]):
        ordered = group.sort_values("snr")
        ax.plot(
            ordered["snr"],
            ordered["exact_recovery_rate"],
            marker="o",
            label=f"{geometry}, degree {cap}",
        )
    ax.set_xlabel("SNR")
    ax.set_ylabel("Exact recovery rate")
    ax.set_ylim(-0.02, 1.02)
    ax.legend(fontsize=7, ncol=2)
    ax.set_title("Curved signal-derived crossing graphs")
    fig.tight_layout()
    fig.savefig(output / "curved_exact_recovery.png", dpi=180)
    plt.close(fig)

    print(summary.to_string(index=False))
    print("\nRecommended scenarios:")
    print(json.dumps(recommendations, indent=2))


if __name__ == "__main__":
    main()
