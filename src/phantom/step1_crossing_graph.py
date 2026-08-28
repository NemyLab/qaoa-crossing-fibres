#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def unit(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=float)
    norm = np.linalg.norm(vector)
    if norm <= 1e-12:
        raise ValueError("Direction vector cannot have zero norm")
    return vector / norm


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Antipodally symmetric angle between two axes."""
    value = float(np.clip(abs(np.dot(unit(a), unit(b))), 0.0, 1.0))
    return float(np.degrees(np.arccos(value)))


def rotate_2d(direction: np.ndarray, angle_deg: float) -> np.ndarray:
    angle = np.deg2rad(angle_deg)
    c, s = np.cos(angle), np.sin(angle)
    rotation = np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    return unit(rotation @ direction)


@dataclass(frozen=True)
class CrossingGraph:
    nodes: pd.DataFrame
    edges: pd.DataFrame
    fields: np.ndarray
    true_spins: np.ndarray
    grid_size: int
    seed: int

    @property
    def n_nodes(self) -> int:
        return int(len(self.nodes))

    def energy(self, spins: Iterable[int]) -> float:
        spins = np.asarray(tuple(spins), dtype=int)
        if spins.shape != (self.n_nodes,):
            raise ValueError("spins has the wrong shape")
        if not np.all(np.isin(spins, [-1, 1])):
            raise ValueError("spins must contain only -1 and +1")

        value = -float(np.dot(self.fields, spins))
        for edge in self.edges.itertuples(index=False):
            product_spin = spins[int(edge.i)] * spins[int(edge.j)]
            if edge.edge_type == "continuation":
                value -= float(edge.weight) * product_spin
            elif edge.edge_type == "conflict":
                value += float(edge.weight) * product_spin
            else:
                raise ValueError(f"Unknown edge type: {edge.edge_type}")
        return float(value)

    def degree_table(self) -> pd.DataFrame:
        degree = np.zeros(self.n_nodes, dtype=int)
        for edge in self.edges.itertuples(index=False):
            degree[int(edge.i)] += 1
            degree[int(edge.j)] += 1
        result = self.nodes[["node_id", "row", "col", "peak_type"]].copy()
        result["degree"] = degree
        return result


def build_graph(
    *,
    grid_size: int,
    angle_noise_deg: float,
    amplitude_noise_sd: float,
    sigma_orientation_deg: float,
    sigma_displacement_deg: float,
    minimum_continuation_weight: float,
    conflict_weight: float,
    anchor_weight: float,
    seed: int,
) -> CrossingGraph:
    """
    Build a two-layer crossing-fixel graph.

    Every voxel contains an approximately horizontal peak (+1) and an
    approximately vertical peak (-1). For every neighbouring voxel pair, only
    the strongest geometrically plausible peak-to-peak continuation is kept.
    """
    if grid_size < 2:
        raise ValueError("grid_size must be at least 2")

    rng = np.random.default_rng(seed)
    horizontal = np.array([1.0, 0.0, 0.0])
    vertical = np.array([0.0, 1.0, 0.0])

    node_rows: list[dict] = []
    node_lookup: dict[tuple[int, int, str], int] = {}

    for row in range(grid_size):
        for col in range(grid_size):
            for peak_type, base_direction, true_spin in (
                ("horizontal", horizontal, +1),
                ("vertical", vertical, -1),
            ):
                node_id = len(node_rows)
                direction = rotate_2d(
                    base_direction,
                    rng.normal(0.0, angle_noise_deg),
                )
                amplitude = float(
                    np.clip(
                        1.0 + rng.normal(0.0, amplitude_noise_sd),
                        0.25,
                        None,
                    )
                )
                node_rows.append(
                    {
                        "node_id": node_id,
                        "row": row,
                        "col": col,
                        "peak_type": peak_type,
                        "dir_x": direction[0],
                        "dir_y": direction[1],
                        "dir_z": direction[2],
                        "amplitude": amplitude,
                        "true_spin": true_spin,
                    }
                )
                node_lookup[(row, col, peak_type)] = node_id

    nodes = pd.DataFrame(node_rows)
    directions = nodes[["dir_x", "dir_y", "dir_z"]].to_numpy(float)
    amplitudes = nodes["amplitude"].to_numpy(float)

    edge_rows: list[dict] = []

    # Same-voxel conflict edges.
    for row in range(grid_size):
        for col in range(grid_size):
            i = node_lookup[(row, col, "horizontal")]
            j = node_lookup[(row, col, "vertical")]
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
                }
            )

    # Each voxel pair is visited once: right and down neighbours.
    for row in range(grid_size):
        for col in range(grid_size):
            for dr, dc in ((0, 1), (1, 0)):
                row2, col2 = row + dr, col + dc
                if row2 >= grid_size or col2 >= grid_size:
                    continue

                displacement = unit(np.array([float(dc), float(dr), 0.0]))
                candidates: list[tuple[float, int, int, float, float, float]] = []

                for type_1 in ("horizontal", "vertical"):
                    i = node_lookup[(row, col, type_1)]
                    for type_2 in ("horizontal", "vertical"):
                        j = node_lookup[(row2, col2, type_2)]

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
                        amplitude_term = np.sqrt(amplitudes[i] * amplitudes[j])
                        weight = float(
                            amplitude_term * orientation_term * displacement_term
                        )
                        candidates.append((weight, i, j, theta, phi_i, phi_j))

                # A single strongest continuation per neighbouring voxel pair.
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
                        }
                    )

    edges = pd.DataFrame(edge_rows)
    fields = np.zeros(len(nodes), dtype=float)

    # The graph partition has an arbitrary global sign without anchors.
    for row in range(grid_size):
        fields[node_lookup[(row, 0, "horizontal")]] += anchor_weight
    for col in range(grid_size):
        fields[node_lookup[(0, col, "vertical")]] -= anchor_weight

    return CrossingGraph(
        nodes=nodes,
        edges=edges,
        fields=fields,
        true_spins=nodes["true_spin"].to_numpy(dtype=int),
        grid_size=grid_size,
        seed=seed,
    )


def exact_enumeration(graph: CrossingGraph) -> tuple[np.ndarray, float]:
    """Exact vectorized solver for graphs with at most 24 spins."""
    n = graph.n_nodes
    if n > 24:
        raise ValueError("Enumeration is limited to 24 nodes; use Gurobi")

    edge_i = graph.edges["i"].to_numpy(dtype=int)
    edge_j = graph.edges["j"].to_numpy(dtype=int)
    edge_sign = np.where(
        graph.edges["edge_type"].to_numpy() == "continuation",
        -1.0,
        +1.0,
    )
    edge_coeff = edge_sign * graph.edges["weight"].to_numpy(dtype=float)

    best_index = None
    best_energy = np.inf
    total = 1 << n
    chunk_size = 65536
    bit_positions = np.arange(n, dtype=np.uint64)

    for first in range(0, total, chunk_size):
        last = min(first + chunk_size, total)
        indices = np.arange(first, last, dtype=np.uint64)
        bits = ((indices[:, None] >> bit_positions[None, :]) & 1).astype(np.int8)
        spins = 2.0 * bits - 1.0

        energies = -(spins @ graph.fields)
        for i, j, coeff in zip(edge_i, edge_j, edge_coeff):
            energies += coeff * spins[:, i] * spins[:, j]

        local = int(np.argmin(energies))
        if float(energies[local]) < best_energy:
            best_energy = float(energies[local])
            best_index = int(indices[local])

    assert best_index is not None
    best_bits = ((np.uint64(best_index) >> bit_positions) & 1).astype(np.int8)
    best_spins = (2 * best_bits - 1).astype(int)
    return best_spins, best_energy


def solve_gurobi(graph: CrossingGraph) -> tuple[np.ndarray, float]:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except ImportError as exc:
        raise RuntimeError("gurobipy is not installed") from exc

    model = gp.Model("crossing_fixel_signed_graph")
    model.Params.OutputFlag = 0
    x = model.addVars(graph.n_nodes, vtype=GRB.BINARY, name="x")
    objective = gp.QuadExpr()

    # s_i = 2*x_i - 1
    for i, field in enumerate(graph.fields):
        objective += -float(field) * (2.0 * x[i] - 1.0)

    for edge in graph.edges.itertuples(index=False):
        i, j = int(edge.i), int(edge.j)
        spin_product = 4.0 * x[i] * x[j] - 2.0 * x[i] - 2.0 * x[j] + 1.0
        if edge.edge_type == "continuation":
            objective += -float(edge.weight) * spin_product
        else:
            objective += float(edge.weight) * spin_product

    model.setObjective(objective, GRB.MINIMIZE)
    model.optimize()
    if model.Status != GRB.OPTIMAL:
        raise RuntimeError(f"Gurobi status: {model.Status}")

    spins = np.array(
        [+1 if float(x[i].X) >= 0.5 else -1 for i in range(graph.n_nodes)],
        dtype=int,
    )
    return spins, float(model.ObjVal)


def plot_graph(graph: CrossingGraph, path: Path) -> None:
    positions: dict[int, tuple[float, float]] = {}
    for node in graph.nodes.itertuples(index=False):
        offset = -0.12 if node.peak_type == "horizontal" else 0.12
        positions[int(node.node_id)] = (float(node.col), float(node.row) + offset)

    fig, ax = plt.subplots(figsize=(7, 7))
    for edge in graph.edges.itertuples(index=False):
        x1, y1 = positions[int(edge.i)]
        x2, y2 = positions[int(edge.j)]
        linestyle = "-" if edge.edge_type == "continuation" else "--"
        ax.plot(
            [x1, x2],
            [y1, y2],
            linestyle=linestyle,
            linewidth=0.8 + float(edge.weight),
        )
    for node in graph.nodes.itertuples(index=False):
        x, y = positions[int(node.node_id)]
        marker = "o" if node.peak_type == "horizontal" else "s"
        ax.scatter([x], [y], marker=marker, s=55)
        ax.text(x + 0.03, y + 0.03, str(node.node_id), fontsize=7)

    ax.set_title("Solid: continuation; dashed: same-voxel conflict")
    ax.set_xlabel("Voxel column")
    ax.set_ylabel("Voxel row")
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_solution(graph: CrossingGraph, spins: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 7))
    for node in graph.nodes.itertuples(index=False):
        node_id = int(node.node_id)
        direction = unit(np.array([node.dir_x, node.dir_y, node.dir_z]))[:2]
        x, y = float(node.col), float(node.row)
        dx, dy = 0.36 * direction
        linestyle = "-" if spins[node_id] == +1 else "--"
        ax.plot(
            [x - dx, x + dx],
            [y - dy, y + dy],
            linestyle=linestyle,
            linewidth=2.0,
        )

    ax.set_title("Exact assignment: solid +1; dashed -1")
    ax.set_xlabel("Voxel column")
    ax.set_ylabel("Voxel row")
    ax.set_xlim(-0.7, graph.grid_size - 0.3)
    ax.set_ylim(-0.7, graph.grid_size - 0.3)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid-size", type=int, default=3)
    parser.add_argument("--angle-noise-deg", type=float, default=5.0)
    parser.add_argument("--amplitude-noise-sd", type=float, default=0.05)
    parser.add_argument("--conflict-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--solver", choices=["auto", "enumeration", "gurobi"], default="auto")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    graph = build_graph(
        grid_size=args.grid_size,
        angle_noise_deg=args.angle_noise_deg,
        amplitude_noise_sd=args.amplitude_noise_sd,
        sigma_orientation_deg=15.0,
        sigma_displacement_deg=15.0,
        minimum_continuation_weight=0.15,
        conflict_weight=args.conflict_weight,
        anchor_weight=args.anchor_weight,
        seed=args.seed,
    )

    if args.solver == "enumeration" or (args.solver == "auto" and graph.n_nodes <= 24):
        predicted, exact_energy = exact_enumeration(graph)
        solver_used = "enumeration"
    else:
        predicted, exact_energy = solve_gurobi(graph)
        solver_used = "gurobi"

    degrees = graph.degree_table()
    accuracy = float(np.mean(predicted == graph.true_spins))
    ground_truth_energy = graph.energy(graph.true_spins)

    nodes = graph.nodes.copy()
    nodes["field"] = graph.fields
    nodes.to_csv(out / "nodes.csv", index=False)
    graph.edges.to_csv(out / "edges.csv", index=False)
    degrees.to_csv(out / "degrees.csv", index=False)

    solution = nodes[["node_id", "row", "col", "peak_type", "true_spin"]].copy()
    solution["predicted_spin"] = predicted
    solution["correct"] = solution["true_spin"] == solution["predicted_spin"]
    solution.to_csv(out / "solution.csv", index=False)

    summary = {
        "grid_size": args.grid_size,
        "seed": args.seed,
        "solver": solver_used,
        "n_nodes_qubits": graph.n_nodes,
        "n_edges_total": int(len(graph.edges)),
        "n_continuation_edges": int((graph.edges["edge_type"] == "continuation").sum()),
        "n_conflict_edges": int((graph.edges["edge_type"] == "conflict").sum()),
        "max_degree": int(degrees["degree"].max()),
        "mean_degree": float(degrees["degree"].mean()),
        "ground_truth_energy": ground_truth_energy,
        "exact_energy": exact_energy,
        "ground_truth_is_optimal": bool(np.isclose(ground_truth_energy, exact_energy, atol=1e-10)),
        "label_accuracy": accuracy,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))

    plot_graph(graph, out / "graph_structure.png")
    plot_solution(graph, predicted, out / "solution_map.png")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
