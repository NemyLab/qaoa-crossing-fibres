#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from itertools import permutations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import nnls

from step1_crossing_graph import (
    CrossingGraph,
    axis_angle_deg,
    exact_enumeration,
    unit,
)


def fibonacci_measurements(n: int) -> np.ndarray:
    """Approximately uniform 3-D diffusion-encoding directions."""
    if n < 6:
        raise ValueError("At least six measurement directions are required")
    index = np.arange(n, dtype=float)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    y = 1.0 - 2.0 * (index + 0.5) / n
    radius = np.sqrt(np.maximum(0.0, 1.0 - y * y))
    theta = golden_angle * index
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return np.column_stack([x, y, z])


def planar_orientation_dictionary(step_deg: float = 2.0) -> np.ndarray:
    """
    Candidate fiber axes in the phantom plane.

    The phantom is deliberately planar in this bridge experiment. Real-data
    processing later replaces this dictionary by MRtrix FOD peaks in 3-D.
    """
    if not 0.5 <= step_deg <= 15.0:
        raise ValueError("dictionary step must be between 0.5 and 15 degrees")
    angles = np.arange(0.0, 180.0, step_deg)
    return np.column_stack(
        [
            np.cos(np.deg2rad(angles)),
            np.sin(np.deg2rad(angles)),
            np.zeros(len(angles)),
        ]
    )


def tensor_response(
    gradients: np.ndarray,
    direction: np.ndarray,
    *,
    b_value: float,
    d_parallel: float = 1.7e-3,
    d_perpendicular: float = 0.3e-3,
) -> np.ndarray:
    direction = unit(direction)
    projection_squared = np.square(gradients @ direction)
    adc = d_perpendicular + (
        d_parallel - d_perpendicular
    ) * projection_squared
    return np.exp(-b_value * adc)


def simulate_two_fiber_signal(
    gradients: np.ndarray,
    *,
    horizontal_fraction: float,
    snr: float,
    b_value: float,
    rng: np.random.Generator,
) -> np.ndarray:
    horizontal = np.array([1.0, 0.0, 0.0])
    vertical = np.array([0.0, 1.0, 0.0])
    vertical_fraction = 1.0 - horizontal_fraction

    clean = (
        horizontal_fraction
        * tensor_response(gradients, horizontal, b_value=b_value)
        + vertical_fraction
        * tensor_response(gradients, vertical, b_value=b_value)
    )

    # Rician magnitude noise with S0 normalized to one.
    sigma = 1.0 / max(float(snr), 1e-8)
    noise_real = rng.normal(0.0, sigma, size=len(gradients))
    noise_imag = rng.normal(0.0, sigma, size=len(gradients))
    return np.sqrt(np.square(clean + noise_real) + np.square(noise_imag))


def build_response_dictionary(
    gradients: np.ndarray,
    candidate_axes: np.ndarray,
    *,
    b_value: float,
) -> np.ndarray:
    atoms = [
        tensor_response(gradients, direction, b_value=b_value)
        for direction in candidate_axes
    ]
    # Include one isotropic response to absorb isotropic/noise-like signal.
    isotropic = np.full(
        len(gradients),
        np.exp(-b_value * 0.8e-3),
        dtype=float,
    )
    return np.column_stack([*atoms, isotropic])


def extract_two_peaks(
    signal: np.ndarray,
    response_dictionary: np.ndarray,
    candidate_axes: np.ndarray,
    *,
    minimum_separation_deg: float = 30.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate two peaks with NNLS followed by angular non-maximum suppression.

    This is a controlled bridge from measured signals to fixel graphs. It is
    not intended to replace CSD in the final experiment.
    """
    coefficients, residual_norm = nnls(
        response_dictionary,
        signal,
        maxiter=10_000,
    )
    fiber_coefficients = coefficients[:-1]
    order = np.argsort(-fiber_coefficients)
    selected: list[int] = []

    for index in order:
        if fiber_coefficients[index] <= 1e-8:
            continue
        if all(
            axis_angle_deg(candidate_axes[index], candidate_axes[previous])
            >= minimum_separation_deg
            for previous in selected
        ):
            selected.append(int(index))
        if len(selected) == 2:
            break

    # Extremely noisy signals can yield only one nonzero NNLS atom. In that
    # case select the strongest sufficiently separated dictionary atom as a
    # low-confidence second peak, so graph size remains fixed for exact tests.
    if len(selected) < 2:
        for index in order:
            if index in selected:
                continue
            if all(
                axis_angle_deg(candidate_axes[index], candidate_axes[previous])
                >= minimum_separation_deg
                for previous in selected
            ):
                selected.append(int(index))
            if len(selected) == 2:
                break

    if len(selected) != 2:
        raise RuntimeError("Could not extract two separated candidate peaks")

    directions = candidate_axes[selected].copy()
    amplitudes = fiber_coefficients[selected].copy()
    return directions, amplitudes, float(residual_norm)


def match_estimated_peaks_to_truth(
    directions: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Assign the two estimated peaks to horizontal/vertical ground truth."""
    truth = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )
    best_permutation = None
    best_total_error = np.inf
    best_errors = None

    for permutation in permutations((0, 1)):
        errors = np.array(
            [
                axis_angle_deg(directions[permutation[0]], truth[0]),
                axis_angle_deg(directions[permutation[1]], truth[1]),
            ]
        )
        if errors.sum() < best_total_error:
            best_total_error = float(errors.sum())
            best_permutation = permutation
            best_errors = errors

    assert best_permutation is not None
    assert best_errors is not None

    # Return labels in the original estimated-peak order.
    spins = np.empty(2, dtype=int)
    spins[best_permutation[0]] = +1
    spins[best_permutation[1]] = -1
    return spins, float(best_errors.mean()), float(best_errors.max())


def horizontal_fraction_for_voxel(
    row: int,
    col: int,
    grid_size: int,
    *,
    fraction_mode: str,
    rng: np.random.Generator,
) -> float:
    if fraction_mode == "balanced":
        return 0.5
    if fraction_mode == "variable":
        center = (grid_size - 1) / 2.0
        scale = max(center, 1.0)
        value = 0.5 + 0.15 * (col - center) / scale
        return float(np.clip(value + rng.normal(0.0, 0.025), 0.30, 0.70))
    if fraction_mode == "imbalanced":
        # Alternate which bundle is weaker so neither label has a global
        # amplitude advantage.
        dominant_horizontal = (row + col) % 2 == 0
        mean = 0.72 if dominant_horizontal else 0.28
        return float(np.clip(mean + rng.normal(0.0, 0.025), 0.20, 0.80))
    raise ValueError(f"Unknown fraction mode: {fraction_mode}")


def build_signal_derived_graph(
    *,
    grid_size: int,
    n_gradients: int,
    snr: float,
    b_value: float,
    fraction_mode: str,
    conflict_weight: float,
    anchor_weight: float,
    sigma_orientation_deg: float,
    sigma_displacement_deg: float,
    minimum_continuation_weight: float,
    seed: int,
) -> tuple[CrossingGraph, dict[str, float]]:
    rng = np.random.default_rng(seed)
    gradients = fibonacci_measurements(n_gradients)
    candidate_axes = planar_orientation_dictionary(step_deg=2.0)
    response_dictionary = build_response_dictionary(
        gradients,
        candidate_axes,
        b_value=b_value,
    )

    node_rows: list[dict] = []
    node_lookup: dict[tuple[int, int, int], int] = {}
    voxel_peak_mean_errors: list[float] = []
    voxel_peak_max_errors: list[float] = []
    voxel_residuals: list[float] = []

    for row in range(grid_size):
        for col in range(grid_size):
            fraction = horizontal_fraction_for_voxel(
                row,
                col,
                grid_size,
                fraction_mode=fraction_mode,
                rng=rng,
            )
            signal = simulate_two_fiber_signal(
                gradients,
                horizontal_fraction=fraction,
                snr=snr,
                b_value=b_value,
                rng=rng,
            )
            directions, amplitudes, residual_norm = extract_two_peaks(
                signal,
                response_dictionary,
                candidate_axes,
            )
            true_spins, mean_error, max_error = match_estimated_peaks_to_truth(
                directions
            )

            voxel_peak_mean_errors.append(mean_error)
            voxel_peak_max_errors.append(max_error)
            voxel_residuals.append(residual_norm)

            for peak_index in range(2):
                node_id = len(node_rows)
                direction = directions[peak_index]
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
                        "horizontal_fraction": fraction,
                        "nnls_residual_norm": residual_norm,
                    }
                )
                node_lookup[(row, col, peak_index)] = node_id

    nodes = pd.DataFrame(node_rows)
    directions = nodes[["dir_x", "dir_y", "dir_z"]].to_numpy(float)
    amplitudes = nodes["amplitude"].to_numpy(float)
    edge_rows: list[dict] = []

    # Same-voxel conflict edges.
    for row in range(grid_size):
        for col in range(grid_size):
            i = node_lookup[(row, col, 0)]
            j = node_lookup[(row, col, 1)]
            edge_rows.append(
                {
                    "i": min(i, j),
                    "j": max(i, j),
                    "edge_type": "conflict",
                    "weight": float(conflict_weight),
                    "orientation_angle_deg": axis_angle_deg(
                        directions[i], directions[j]
                    ),
                    "displacement_angle_i_deg": np.nan,
                    "displacement_angle_j_deg": np.nan,
                    "ground_truth_same_bundle": False,
                }
            )

    # Select one strongest continuation for each right/down voxel pair.
    for row in range(grid_size):
        for col in range(grid_size):
            for dr, dc in ((0, 1), (1, 0)):
                row2, col2 = row + dr, col + dc
                if row2 >= grid_size or col2 >= grid_size:
                    continue

                displacement = unit(np.array([float(dc), float(dr), 0.0]))
                candidates: list[
                    tuple[float, int, int, float, float, float]
                ] = []

                for peak_1 in (0, 1):
                    i = node_lookup[(row, col, peak_1)]
                    for peak_2 in (0, 1):
                        j = node_lookup[(row2, col2, peak_2)]
                        theta = axis_angle_deg(directions[i], directions[j])
                        phi_i = axis_angle_deg(directions[i], displacement)
                        phi_j = axis_angle_deg(directions[j], displacement)

                        orientation_term = np.exp(
                            -0.5 * (theta / sigma_orientation_deg) ** 2
                        )
                        displacement_term = np.exp(
                            -0.5
                            * (
                                (phi_i / sigma_displacement_deg) ** 2
                                + (phi_j / sigma_displacement_deg) ** 2
                            )
                        )
                        amplitude_term = np.sqrt(
                            max(amplitudes[i], 0.0)
                            * max(amplitudes[j], 0.0)
                        )
                        weight = float(
                            amplitude_term
                            * orientation_term
                            * displacement_term
                        )
                        candidates.append(
                            (weight, i, j, theta, phi_i, phi_j)
                        )

                candidates.sort(reverse=True, key=lambda item: item[0])
                weight, i, j, theta, phi_i, phi_j = candidates[0]
                if weight >= minimum_continuation_weight:
                    edge_rows.append(
                        {
                            "i": min(i, j),
                            "j": max(i, j),
                            "edge_type": "continuation",
                            "weight": weight,
                            "orientation_angle_deg": theta,
                            "displacement_angle_i_deg": phi_i,
                            "displacement_angle_j_deg": phi_j,
                            "ground_truth_same_bundle": bool(
                                nodes.iloc[i]["true_spin"]
                                == nodes.iloc[j]["true_spin"]
                            ),
                        }
                    )

    edges = pd.DataFrame(edge_rows)
    fields = np.zeros(len(nodes), dtype=float)

    # Boundary anchors are selected from estimated orientations only.
    horizontal_axis = np.array([1.0, 0.0, 0.0])
    vertical_axis = np.array([0.0, 1.0, 0.0])

    for row in range(grid_size):
        candidates = [
            node_lookup[(row, 0, peak_index)] for peak_index in (0, 1)
        ]
        selected = min(
            candidates,
            key=lambda node_id: axis_angle_deg(
                directions[node_id], horizontal_axis
            ),
        )
        fields[selected] += anchor_weight

    for col in range(grid_size):
        candidates = [
            node_lookup[(0, col, peak_index)] for peak_index in (0, 1)
        ]
        selected = min(
            candidates,
            key=lambda node_id: axis_angle_deg(
                directions[node_id], vertical_axis
            ),
        )
        fields[selected] -= anchor_weight

    graph = CrossingGraph(
        nodes=nodes,
        edges=edges,
        fields=fields,
        true_spins=nodes["true_spin"].to_numpy(dtype=int),
        grid_size=grid_size,
        seed=seed,
    )

    continuation = edges[edges["edge_type"] == "continuation"]
    diagnostics = {
        "mean_peak_error_deg": float(np.mean(voxel_peak_mean_errors)),
        "maximum_voxel_peak_error_deg": float(
            np.max(voxel_peak_max_errors)
        ),
        "mean_nnls_residual_norm": float(np.mean(voxel_residuals)),
        "continuation_edge_correct_fraction": (
            float(continuation["ground_truth_same_bundle"].mean())
            if len(continuation) > 0
            else np.nan
        ),
    }
    return graph, diagnostics


def evaluate_case(
    *,
    snr: float,
    fraction_mode: str,
    seed: int,
    args: argparse.Namespace,
) -> tuple[dict, CrossingGraph, np.ndarray]:
    graph, diagnostics = build_signal_derived_graph(
        grid_size=args.grid_size,
        n_gradients=args.n_gradients,
        snr=snr,
        b_value=args.b_value,
        fraction_mode=fraction_mode,
        conflict_weight=args.conflict_weight,
        anchor_weight=args.anchor_weight,
        sigma_orientation_deg=args.sigma_orientation_deg,
        sigma_displacement_deg=args.sigma_displacement_deg,
        minimum_continuation_weight=args.minimum_continuation_weight,
        seed=seed,
    )
    predicted, exact_energy = exact_enumeration(graph)
    accuracy = float(np.mean(predicted == graph.true_spins))
    truth_energy = graph.energy(graph.true_spins)
    degree = graph.degree_table()["degree"]

    record = {
        "seed": seed,
        "snr": snr,
        "fraction_mode": fraction_mode,
        "n_nodes": graph.n_nodes,
        "n_edges": len(graph.edges),
        "n_continuation_edges": int(
            (graph.edges["edge_type"] == "continuation").sum()
        ),
        "maximum_degree": int(degree.max()),
        "mean_degree": float(degree.mean()),
        "label_accuracy": accuracy,
        "exact_recovery": bool(np.isclose(accuracy, 1.0)),
        "truth_energy": truth_energy,
        "exact_energy": exact_energy,
        "truth_optimality_gap": float(truth_energy - exact_energy),
        "truth_is_optimal": bool(
            np.isclose(truth_energy, exact_energy, atol=1e-10)
        ),
        **diagnostics,
    }
    return record, graph, predicted


def plot_example(
    graph: CrossingGraph,
    predicted: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 6.5))
    for node in graph.nodes.itertuples(index=False):
        node_id = int(node.node_id)
        direction = np.array([node.dir_x, node.dir_y])
        direction /= max(np.linalg.norm(direction), 1e-12)
        dx, dy = 0.36 * direction
        linestyle = "-" if predicted[node_id] == +1 else "--"
        linewidth = 1.2 + 2.0 * min(float(node.amplitude), 1.0)
        ax.plot(
            [node.col - dx, node.col + dx],
            [node.row - dy, node.row + dy],
            linestyle=linestyle,
            linewidth=linewidth,
        )
    ax.set_title(title)
    ax.set_xlabel("Voxel column")
    ax.set_ylabel("Voxel row")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-seeds", type=int, default=20)
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--n-gradients", type=int, default=60)
    parser.add_argument("--b-value", type=float, default=2000.0)
    parser.add_argument(
        "--snr-values",
        type=float,
        nargs="+",
        default=[30.0, 20.0, 15.0, 10.0, 8.0],
    )
    parser.add_argument(
        "--fraction-modes",
        nargs="+",
        choices=["balanced", "variable", "imbalanced"],
        default=["balanced", "variable", "imbalanced"],
    )
    parser.add_argument("--conflict-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--sigma-orientation-deg", type=float, default=15.0)
    parser.add_argument("--sigma-displacement-deg", type=float, default=15.0)
    parser.add_argument(
        "--minimum-continuation-weight",
        type=float,
        default=0.02,
    )
    parser.add_argument("--target-recovery", type=float, default=0.70)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)

    n_nodes = 2 * args.grid_size * args.grid_size
    records: list[dict] = []
    example_saved = False

    for fraction_mode in args.fraction_modes:
        for snr in args.snr_values:
            for seed in range(args.n_seeds):
                record, graph, predicted = evaluate_case(
                    snr=snr,
                    fraction_mode=fraction_mode,
                    seed=seed,
                    args=args,
                )
                records.append(record)

                if not example_saved and fraction_mode == "variable" and snr == 10.0:
                    graph.nodes.assign(
                        field=graph.fields,
                        predicted_spin=predicted,
                    ).to_csv(output / "example_nodes.csv", index=False)
                    graph.edges.to_csv(output / "example_edges.csv", index=False)
                    plot_example(
                        graph,
                        predicted,
                        output / "example_signal_derived_solution.png",
                        title="Signal-derived peaks: solid +1, dashed -1",
                    )
                    example_saved = True

    results = pd.DataFrame(records)
    summary = (
        results.groupby(["fraction_mode", "snr"], as_index=False)
        .agg(
            n_seeds=("seed", "size"),
            mean_peak_error_deg=("mean_peak_error_deg", "mean"),
            mean_maximum_peak_error_deg=(
                "maximum_voxel_peak_error_deg",
                "mean",
            ),
            mean_nnls_residual_norm=("mean_nnls_residual_norm", "mean"),
            mean_correct_continuation_fraction=(
                "continuation_edge_correct_fraction",
                "mean",
            ),
            mean_label_accuracy=("label_accuracy", "mean"),
            exact_recovery_rate=("exact_recovery", "mean"),
            truth_optimal_rate=("truth_is_optimal", "mean"),
            mean_truth_optimality_gap=("truth_optimality_gap", "mean"),
            mean_edges=("n_edges", "mean"),
            observed_maximum_degree=("maximum_degree", "max"),
        )
        .sort_values(["fraction_mode", "snr"], ascending=[True, False])
    )

    candidates = summary.copy()
    candidates["distance_to_target"] = abs(
        candidates["exact_recovery_rate"] - args.target_recovery
    )
    recommended = candidates.sort_values(
        ["distance_to_target", "mean_label_accuracy"],
        ascending=[True, False],
    ).iloc[0]

    recommendation = {
        "target_exact_recovery_rate": args.target_recovery,
        "recommended_signal_scenario": {
            "fraction_mode": str(recommended["fraction_mode"]),
            "snr": float(recommended["snr"]),
            "observed_exact_recovery_rate": float(
                recommended["exact_recovery_rate"]
            ),
            "observed_mean_label_accuracy": float(
                recommended["mean_label_accuracy"]
            ),
            "observed_mean_peak_error_deg": float(
                recommended["mean_peak_error_deg"]
            ),
            "observed_correct_continuation_fraction": float(
                recommended["mean_correct_continuation_fraction"]
            ),
            "observed_maximum_degree": int(
                recommended["observed_maximum_degree"]
            ),
        },
    }

    results.to_csv(output / "signal_graph_results.csv", index=False)
    summary.to_csv(output / "signal_graph_summary.csv", index=False)
    (output / "recommended_signal_scenario.json").write_text(
        json.dumps(recommendation, indent=2)
    )

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for mode, group in summary.groupby("fraction_mode"):
        ordered = group.sort_values("snr")
        ax.plot(
            ordered["snr"],
            ordered["exact_recovery_rate"],
            marker="o",
            label=mode,
        )
    ax.set_xlabel("SNR")
    ax.set_ylabel("Exact partition recovery rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Recovery from signal-derived FOD-peak approximations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "recovery_by_snr.png", dpi=180)
    plt.close(fig)

    print("\nSignal-derived graph summary:")
    print(summary.to_string(index=False))
    print("\nRecommended signal scenario:")
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
