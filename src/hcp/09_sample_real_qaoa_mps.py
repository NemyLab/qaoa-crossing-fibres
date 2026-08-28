#!/usr/bin/env python3
"""
Sample the real 50-qubit HCP crossing-fibre QAOA circuit with Qiskit Aer MPS.

This is the final no-QPU validation before hardware transpilation:
  - reconstructs the saved RCM-labelled CrossingGraph exactly;
  - optionally applies a graph-preserving Fiedler ordering inside MPS only;
  - reorders mutually commuting RZZ terms into an MPS-efficient sweep;
  - maps sampled assignments back to the saved RCM labels before analysis;
  - validates the stored exact MILP solution;
  - runs 10,000 ideal MPS shots for both frozen p=1 angle strategies;
  - compares sampled mean energy against the exact light-cone expectation;
  - reports the best sampled energy and agreement with the exact MILP and
    TractSeg-derived CST reference.

Run from the crossing_fibres_step7 project directory.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from step1_crossing_graph import CrossingGraph
from step5_curved_signal_graph import exact_milp
from step9_common import (
    graph_coefficients,
    normalized_gap,
    qiskit_bitstring_to_spins,
    reorder_graph,
    spins_to_qiskit_bitstring,
)
from step9_run_10000 import (
    expected_energy_lightcone_p1,
)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ideal MPS shot sampling for the real 50-qubit graph."
    )
    parser.add_argument(
        "--graph-dir",
        type=Path,
        required=True,
        help="Directory produced by 07_build_real_crossing_graph.py.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--bond-dimension", type=int, default=512)
    parser.add_argument("--truncation-threshold", type=float, default=1e-12)
    parser.add_argument("--simulator-seed", type=int, default=100_307_009)
    parser.add_argument(
        "--mps-ordering",
        choices=("fiedler", "pre_rcm", "rcm"),
        default="fiedler",
        help=(
            "Qubit order used only inside the classical MPS simulator. "
            "Outputs are always mapped back to the saved RCM labels."
        ),
    )
    parser.add_argument(
        "--mps-swap-direction",
        choices=("mps_swap_left", "mps_swap_right"),
        default="mps_swap_right",
        help="Aer direction for internal swaps used by non-local MPS gates.",
    )
    parser.add_argument(
        "--mps-edge-order",
        choices=("right_endpoint", "original"),
        default="right_endpoint",
        help=(
            "Order of the mutually commuting RZZ terms. 'right_endpoint' "
            "performs a deterministic left-to-right sweep that limits "
            "intermediate MPS entanglement; 'original' preserves CSV row order."
        ),
    )
    return parser.parse_args()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return repr(value)


def load_saved_graph(graph_dir: Path) -> tuple[CrossingGraph, dict[str, np.ndarray], dict]:
    nodes_path = graph_dir / "real_graph_nodes_rcm.csv"
    edges_path = graph_dir / "real_graph_edges_rcm.csv"
    arrays_path = graph_dir / "real_graph_solution_arrays.npz"
    validation_path = graph_dir / "real_graph_validation.json"

    for path in (nodes_path, edges_path, arrays_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    nodes = pd.read_csv(nodes_path).sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(edges_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    validation = json.loads(validation_path.read_text(encoding="utf-8"))

    required_node_columns = {"node_id", "old_node_id", "true_spin"}
    missing = required_node_columns.difference(nodes.columns)
    if missing:
        raise ValueError(f"RCM node table is missing columns: {sorted(missing)}")

    fields_pre_rcm = np.asarray(arrays["fields_pre_rcm"], dtype=float)
    if fields_pre_rcm.shape != (50,):
        raise ValueError(
            f"Expected fields_pre_rcm shape (50,), found {fields_pre_rcm.shape}"
        )

    old_ids = nodes["old_node_id"].to_numpy(dtype=int)
    fields_rcm = fields_pre_rcm[old_ids]
    true_spins_rcm = nodes["true_spin"].to_numpy(dtype=np.int8)

    graph = CrossingGraph(
        nodes=nodes,
        edges=edges,
        fields=fields_rcm,
        true_spins=true_spins_rcm,
        grid_size=5,
        seed=0,
    )
    if graph.n_nodes != 50:
        raise ValueError(f"Expected 50 nodes, found {graph.n_nodes}")

    return graph, arrays, validation


def graph_adjacency(graph: CrossingGraph) -> list[set[int]]:
    adjacency = [set() for _ in range(graph.n_nodes)]
    for edge in graph.edges.itertuples(index=False):
        i, j = int(edge.i), int(edge.j)
        adjacency[i].add(j)
        adjacency[j].add(i)
    return adjacency


def connected_components(adjacency: list[set[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        stack = [start]
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            neighbours = sorted(adjacency[node].intersection(unseen), reverse=True)
            for neighbour in neighbours:
                unseen.remove(neighbour)
                stack.append(neighbour)
        components.append(sorted(component))
    return sorted(components, key=lambda values: (-len(values), min(values)))


def fiedler_order(graph: CrossingGraph) -> list[int]:
    """Return a deterministic component-wise Fiedler-vector ordering."""
    adjacency = graph_adjacency(graph)
    order: list[int] = []

    for component in connected_components(adjacency):
        if len(component) <= 2:
            order.extend(component)
            continue

        local_index = {node: index for index, node in enumerate(component)}
        matrix = np.zeros((len(component), len(component)), dtype=float)
        for node in component:
            i = local_index[node]
            for neighbour in adjacency[node]:
                if neighbour in local_index:
                    matrix[i, local_index[neighbour]] = 1.0
        laplacian = np.diag(matrix.sum(axis=1)) - matrix
        eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
        positive = np.flatnonzero(eigenvalues > 1e-10)
        if len(positive) == 0:
            order.extend(component)
            continue

        fiedler_value = float(eigenvalues[int(positive[0])])
        eigenspace_mask = np.isclose(
            eigenvalues,
            fiedler_value,
            rtol=1e-8,
            atol=1e-10,
        )
        eigenspace = np.asarray(eigenvectors[:, eigenspace_mask], dtype=float)
        probes = [np.linspace(-1.0, 1.0, len(component), dtype=float)]
        probes.extend(np.eye(len(component), dtype=float))
        vector = None
        for probe in probes:
            candidate = eigenspace @ (eigenspace.T @ probe)
            norm = float(np.linalg.norm(candidate))
            if norm > 1e-10:
                vector = candidate / norm
                break
        if vector is None:
            raise RuntimeError("Could not construct a deterministic Fiedler vector")
        pivot = int(np.argmax(np.abs(vector)))
        if vector[pivot] > 0:
            vector = -vector
        component_order = [
            node
            for _, node in sorted(
                (float(vector[local_index[node]]), node) for node in component
            )
        ]
        order.extend(component_order)

    if sorted(order) != list(range(graph.n_nodes)):
        raise RuntimeError("Fiedler ordering is not a complete node permutation")
    return order


def select_mps_order(graph: CrossingGraph, mode: str) -> list[int]:
    if mode == "rcm":
        return list(range(graph.n_nodes))
    if mode == "pre_rcm":
        if "old_node_id" not in graph.nodes.columns:
            raise ValueError("The saved RCM node table has no old_node_id column")
        ordered = graph.nodes.sort_values("old_node_id")
        return ordered["node_id"].to_numpy(dtype=int).tolist()
    if mode == "fiedler":
        return fiedler_order(graph)
    raise ValueError(f"Unsupported MPS ordering: {mode}")


def maximum_crossing_matching(
    left_nodes: set[int],
    crossing_edges: list[tuple[int, int]],
) -> int:
    adjacency: dict[int, list[int]] = {node: [] for node in left_nodes}
    for i, j in crossing_edges:
        if i in left_nodes:
            adjacency[i].append(j)
        else:
            adjacency[j].append(i)

    right_match: dict[int, int] = {}

    def augment(left: int, seen_right: set[int]) -> bool:
        for right in adjacency[left]:
            if right in seen_right:
                continue
            seen_right.add(right)
            if right not in right_match or augment(right_match[right], seen_right):
                right_match[right] = left
                return True
        return False

    return int(sum(augment(left, set()) for left in adjacency))


def ordering_diagnostics(graph: CrossingGraph, order: list[int]) -> dict[str, Any]:
    if sorted(order) != list(range(graph.n_nodes)):
        raise ValueError("MPS order must be a permutation of all graph nodes")

    position = np.empty(graph.n_nodes, dtype=int)
    for new, old in enumerate(order):
        position[old] = new
    edges = [
        (int(edge.i), int(edge.j))
        for edge in graph.edges.itertuples(index=False)
    ]

    cut_edges: list[int] = []
    left_boundary_sizes: list[int] = []
    right_boundary_sizes: list[int] = []
    matching_sizes: list[int] = []
    for cut in range(1, graph.n_nodes):
        left = set(order[:cut])
        crossing = [
            (i, j)
            for i, j in edges
            if (position[i] < cut) != (position[j] < cut)
        ]
        left_boundary = {
            i if i in left else j
            for i, j in crossing
        }
        right_boundary = {
            j if i in left else i
            for i, j in crossing
        }
        cut_edges.append(len(crossing))
        left_boundary_sizes.append(len(left_boundary))
        right_boundary_sizes.append(len(right_boundary))
        matching_sizes.append(maximum_crossing_matching(left, crossing))

    maximum_matching = max(matching_sizes, default=0)
    maximum_min_boundary = max(
        (
            min(left, right)
            for left, right in zip(left_boundary_sizes, right_boundary_sizes)
        ),
        default=0,
    )
    edge_spans = [abs(int(position[i]) - int(position[j])) for i, j in edges]

    return {
        "order_rcm_node_ids": [int(node) for node in order],
        "maximum_edge_span": int(max(edge_spans, default=0)),
        "total_edge_span": int(sum(edge_spans)),
        "maximum_crossing_edge_count": int(max(cut_edges, default=0)),
        "maximum_crossing_matching_size": int(maximum_matching),
        "final_state_rank_upper_bound_from_matching": int(2**maximum_matching),
        "maximum_smaller_boundary_size": int(maximum_min_boundary),
        "final_state_rank_upper_bound_from_boundary": int(2**maximum_min_boundary),
        "cut_edge_counts": cut_edges,
        "crossing_matching_sizes": matching_sizes,
        "left_boundary_sizes": left_boundary_sizes,
        "right_boundary_sizes": right_boundary_sizes,
    }


def ordered_zz_coefficients(
    graph: CrossingGraph,
    mode: str,
) -> list[tuple[int, int, float]]:
    """Return mathematically equivalent p=1 cost terms in an MPS-aware order."""
    _, zz = graph_coefficients(graph)
    if mode == "original":
        return list(zz)
    if mode == "right_endpoint":
        return sorted(
            zz,
            key=lambda item: (
                max(int(item[0]), int(item[1])),
                min(int(item[0]), int(item[1])),
            ),
        )
    raise ValueError(f"Unsupported MPS edge order: {mode}")


def build_mps_measured_qaoa_circuit(
    graph: CrossingGraph,
    gamma: float,
    beta: float,
    edge_order: str,
):
    """Build the same p=1 QAOA state with commuting RZZ terms MPS-ordered."""
    from qiskit import QuantumCircuit

    z, _ = graph_coefficients(graph)
    zz = ordered_zz_coefficients(graph, edge_order)
    circuit = QuantumCircuit(graph.n_nodes, graph.n_nodes)
    circuit.h(range(graph.n_nodes))
    for i, coefficient in enumerate(z):
        if abs(coefficient) > 1e-15:
            circuit.rz(2.0 * gamma * coefficient, i)
    for edge_index, (i, j, coefficient) in enumerate(zz):
        circuit.rzz(2.0 * gamma * coefficient, i, j)
        if edge_order == "right_endpoint":
            current_endpoint = max(int(i), int(j))
            next_endpoint = (
                max(int(zz[edge_index + 1][0]), int(zz[edge_index + 1][1]))
                if edge_index + 1 < len(zz)
                else None
            )
            if next_endpoint != current_endpoint:
                # Preserve the endpoint sweep through transpilation. Barriers
                # are simulation-only and do not alter the QAOA state.
                circuit.barrier()
    circuit.rx(2.0 * beta, range(graph.n_nodes))
    circuit.measure(range(graph.n_nodes), range(graph.n_nodes))
    return circuit


def dynamic_mps_routing_diagnostics(
    graph: CrossingGraph,
    edge_order: str,
    swap_direction: str,
) -> dict[str, Any]:
    """Bound Schmidt rank throughout Aer-style internal MPS routing.

    The p=1 RZZ terms commute.  At every routing step, the Schmidt rank across
    a chain cut is bounded by 2**min(n_left_boundary, n_right_boundary) for the
    already applied weighted graph-state edges.  This includes every internal
    adjacent swap left in place by Aer.
    """
    n_nodes = int(graph.n_nodes)
    edges = [
        (int(i), int(j))
        for i, j, _ in ordered_zz_coefficients(graph, edge_order)
    ]
    chain = list(range(n_nodes))
    applied: list[tuple[int, int]] = []
    maximum_boundary = 0
    maximum_event: dict[str, Any] | None = None
    n_internal_swaps = 0

    def current_maximum_boundary() -> tuple[int, int, int, int]:
        position = np.empty(n_nodes, dtype=int)
        for chain_position, node in enumerate(chain):
            position[node] = chain_position
        best = (0, 0, 0, 0)
        for cut in range(1, n_nodes):
            left_boundary: set[int] = set()
            right_boundary: set[int] = set()
            for i, j in applied:
                i_left = bool(position[i] < cut)
                j_left = bool(position[j] < cut)
                if i_left == j_left:
                    continue
                if i_left:
                    left_boundary.add(i)
                    right_boundary.add(j)
                else:
                    left_boundary.add(j)
                    right_boundary.add(i)
            value = min(len(left_boundary), len(right_boundary))
            if value > best[0]:
                best = (
                    int(value),
                    int(cut),
                    int(len(left_boundary)),
                    int(len(right_boundary)),
                )
        return best

    def record(event_type: str, edge_index: int, edge: tuple[int, int]) -> None:
        nonlocal maximum_boundary, maximum_event
        value, cut, n_left, n_right = current_maximum_boundary()
        if value > maximum_boundary:
            maximum_boundary = value
            maximum_event = {
                "event_type": event_type,
                "edge_index": int(edge_index),
                "edge": [int(edge[0]), int(edge[1])],
                "cut": int(cut),
                "left_boundary_size": int(n_left),
                "right_boundary_size": int(n_right),
            }

    for edge_index, (i, j) in enumerate(edges):
        position = {node: index for index, node in enumerate(chain)}
        lower, upper = sorted((position[i], position[j]))
        if swap_direction == "mps_swap_right":
            swaps = ((index, index + 1) for index in range(lower, upper - 1))
        elif swap_direction == "mps_swap_left":
            swaps = ((index - 1, index) for index in range(upper, lower + 1, -1))
        else:
            raise ValueError(f"Unsupported MPS swap direction: {swap_direction}")

        for first, second in swaps:
            chain[first], chain[second] = chain[second], chain[first]
            n_internal_swaps += 1
            record("internal_swap", edge_index, (i, j))

        applied.append((i, j))
        record("rzz", edge_index, (i, j))

    return {
        "edge_order": str(edge_order),
        "swap_direction": str(swap_direction),
        "n_rzz_terms": int(len(edges)),
        "predicted_internal_swaps": int(n_internal_swaps),
        "maximum_dynamic_smaller_boundary_size": int(maximum_boundary),
        "dynamic_schmidt_rank_upper_bound": int(2**maximum_boundary),
        "maximum_event": maximum_event,
        "final_chain_order": [int(node) for node in chain],
    }


def remap_counts_to_rcm(
    counts: dict[str, int],
    order: list[int],
    n_nodes: int,
) -> dict[str, int]:
    """Map MPS-chain bitstrings back to the saved RCM-labelled graph."""
    canonical: dict[str, int] = {}
    order_array = np.asarray(order, dtype=int)
    for bitstring, count in counts.items():
        spins_mps = qiskit_bitstring_to_spins(bitstring, n_nodes)
        spins_rcm = np.empty(n_nodes, dtype=np.int8)
        spins_rcm[order_array] = spins_mps
        canonical_bitstring = spins_to_qiskit_bitstring(spins_rcm)
        canonical[canonical_bitstring] = (
            canonical.get(canonical_bitstring, 0) + int(count)
        )
    return canonical


def aligned_accuracy(spins: np.ndarray, reference: np.ndarray) -> float:
    spins = np.asarray(spins, dtype=int)
    reference = np.asarray(reference, dtype=int)
    return float(max(np.mean(spins == reference), np.mean(-spins == reference)))


def cvar_energy(energy_counts: list[tuple[float, int]], alpha: float = 0.10) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0,1]")
    total = int(sum(count for _, count in energy_counts))
    if total <= 0:
        raise ValueError("No samples")
    target = alpha * total
    accumulated = 0.0
    weighted = 0.0
    for energy, count in sorted(energy_counts, key=lambda item: item[0]):
        take = min(float(count), target - accumulated)
        if take <= 0:
            break
        weighted += take * float(energy)
        accumulated += take
    if accumulated <= 0:
        raise RuntimeError("CVaR accumulation failed")
    return float(weighted / accumulated)


def analyse_counts(
    *,
    counts: dict[str, int],
    graph: CrossingGraph,
    exact_spins: np.ndarray,
    exact_energy: float,
    analytic_expected_energy: float,
) -> tuple[dict[str, Any], dict[str, Any]]:
    shots = int(sum(counts.values()))
    weighted_sum = 0.0
    weighted_square_sum = 0.0
    exact_hits = 0
    best_energy = np.inf
    best_bitstring = None
    best_spins = None
    energy_counts: list[tuple[float, int]] = []

    for bitstring, count in counts.items():
        spins = qiskit_bitstring_to_spins(bitstring, graph.n_nodes)
        energy = float(graph.energy(spins))
        count = int(count)
        weighted_sum += count * energy
        weighted_square_sum += count * energy * energy
        energy_counts.append((energy, count))

        if np.isclose(energy, exact_energy, atol=1e-8):
            exact_hits += count
        if energy < best_energy - 1e-12:
            best_energy = energy
            best_bitstring = str(bitstring)
            best_spins = np.asarray(spins, dtype=np.int8)

    mean_energy = float(weighted_sum / shots)
    variance = max(weighted_square_sum / shots - mean_energy * mean_energy, 0.0)
    mean_energy_se = float(np.sqrt(variance / shots))
    cvar10 = cvar_energy(energy_counts, alpha=0.10)

    if best_spins is None or best_bitstring is None:
        raise RuntimeError("No best sample was identified")

    mean_minus_analytic = float(mean_energy - analytic_expected_energy)
    mean_vs_analytic_z = (
        float(mean_minus_analytic / mean_energy_se)
        if mean_energy_se > 0
        else float("nan")
    )

    metrics = {
        "shots": shots,
        "n_unique_bitstrings": int(len(counts)),
        "mean_energy": mean_energy,
        "mean_energy_se": mean_energy_se,
        "analytic_expected_energy": float(analytic_expected_energy),
        "sampled_minus_analytic_energy": mean_minus_analytic,
        "sampled_vs_analytic_z": mean_vs_analytic_z,
        "sampled_expectation_within_5se": bool(
            np.isfinite(mean_vs_analytic_z) and abs(mean_vs_analytic_z) <= 5.0
        ),
        "mean_normalized_gap": float(normalized_gap(mean_energy, exact_energy)),
        "cvar10_energy": float(cvar10),
        "cvar10_normalized_gap": float(normalized_gap(cvar10, exact_energy)),
        "best_bitstring": best_bitstring,
        "best_energy": float(best_energy),
        "best_normalized_gap": float(normalized_gap(best_energy, exact_energy)),
        "best_agreement_with_exact_up_to_sign": aligned_accuracy(
            best_spins, exact_spins
        ),
        "best_agreement_with_tom_reference_up_to_sign": aligned_accuracy(
            best_spins, graph.true_spins
        ),
        "exact_ground_hits": int(exact_hits),
        "exact_ground_hit_fraction": float(exact_hits / shots),
    }
    best = {
        "bitstring": best_bitstring,
        "spins_rcm": best_spins,
        "energy": float(best_energy),
    }
    return metrics, best


def main() -> None:
    args = parse_args()
    if args.shots < 1:
        raise ValueError("--shots must be positive")
    if args.bond_dimension < 2:
        raise ValueError("--bond-dimension must be at least 2")

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    args.out.mkdir(parents=True, exist_ok=True)
    graph, stored_arrays, graph_validation = load_saved_graph(args.graph_dir)
    mps_order = select_mps_order(graph, args.mps_ordering)
    mps_ordering_diagnostics = ordering_diagnostics(graph, mps_order)
    simulation_graph = reorder_graph(graph, mps_order)
    mps_routing_diagnostics = dynamic_mps_routing_diagnostics(
        simulation_graph,
        str(args.mps_edge_order),
        str(args.mps_swap_direction),
    )

    exact_spins, exact_energy, solver_info = exact_milp(graph)
    exact_spins = np.asarray(exact_spins, dtype=np.int8)

    stored_exact = np.asarray(stored_arrays["exact_spins_rcm"], dtype=np.int8)
    stored_energy = float(graph_validation["exact_anchored"]["energy"])
    if not np.isclose(exact_energy, stored_energy, atol=1e-8):
        raise RuntimeError(
            f"Reconstructed exact energy {exact_energy} differs from stored {stored_energy}"
        )
    if not np.isclose(graph.energy(stored_exact), exact_energy, atol=1e-8):
        raise RuntimeError("Stored exact spin assignment is not exact for reconstructed graph")
    if not np.isclose(
        simulation_graph.energy(stored_exact[np.asarray(mps_order, dtype=int)]),
        exact_energy,
        atol=1e-8,
    ):
        raise RuntimeError("MPS-only node permutation changed the exact graph energy")

    all_counts: dict[str, dict[str, int]] = {}
    all_best: dict[str, dict[str, Any]] = {}
    all_aer_metadata: dict[str, Any] = {}
    result_rows: list[dict[str, Any]] = []

    for strategy_index, (strategy, angles) in enumerate(ANGLE_STRATEGIES.items()):
        gamma = float(angles["gamma"])
        beta = float(angles["beta"])
        analytic = float(expected_energy_lightcone_p1(graph, gamma, beta))
        analytic_permuted = float(
            expected_energy_lightcone_p1(simulation_graph, gamma, beta)
        )
        if not np.isclose(analytic, analytic_permuted, atol=1e-10):
            raise RuntimeError(
                "MPS-only node permutation changed the analytic p=1 expectation"
            )
        circuit = build_mps_measured_qaoa_circuit(
            simulation_graph,
            gamma,
            beta,
            str(args.mps_edge_order),
        )

        simulator = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=int(args.bond_dimension),
            matrix_product_state_truncation_threshold=float(
                args.truncation_threshold
            ),
            mps_sample_measure_algorithm="mps_apply_measure",
            mps_log_data=True,
            mps_swap_direction=str(args.mps_swap_direction),
        )
        compiled = transpile(circuit, simulator, optimization_level=0)

        start = time.perf_counter()
        result = simulator.run(
            compiled,
            shots=int(args.shots),
            seed_simulator=int(args.simulator_seed + strategy_index),
        ).result()
        elapsed = float(time.perf_counter() - start)
        raw_counts = {
            str(bitstring): int(count)
            for bitstring, count in result.get_counts(0).items()
        }
        counts = remap_counts_to_rcm(
            raw_counts,
            mps_order,
            graph.n_nodes,
        )
        all_aer_metadata[strategy] = json_safe(result.results[0].metadata)

        metrics, best = analyse_counts(
            counts=counts,
            graph=graph,
            exact_spins=exact_spins,
            exact_energy=float(exact_energy),
            analytic_expected_energy=analytic,
        )
        row = {
            "angle_strategy": strategy,
            "gamma": gamma,
            "beta": beta,
            "bond_dimension": int(args.bond_dimension),
            "truncation_threshold": float(args.truncation_threshold),
            "mps_ordering": str(args.mps_ordering),
            "mps_swap_direction": str(args.mps_swap_direction),
            "mps_edge_order": str(args.mps_edge_order),
            "mps_maximum_crossing_matching_size": int(
                mps_ordering_diagnostics["maximum_crossing_matching_size"]
            ),
            "mps_maximum_smaller_boundary_size": int(
                mps_ordering_diagnostics["maximum_smaller_boundary_size"]
            ),
            "mps_dynamic_maximum_smaller_boundary_size": int(
                mps_routing_diagnostics[
                    "maximum_dynamic_smaller_boundary_size"
                ]
            ),
            "mps_dynamic_schmidt_rank_upper_bound": int(
                mps_routing_diagnostics["dynamic_schmidt_rank_upper_bound"]
            ),
            "mps_predicted_internal_swaps": int(
                mps_routing_diagnostics["predicted_internal_swaps"]
            ),
            "simulator_seed": int(args.simulator_seed + strategy_index),
            "simulation_seconds": elapsed,
            "compiled_depth": int(compiled.depth()),
            "compiled_twoq_depth": int(
                compiled.depth(lambda instruction: len(instruction.qubits) == 2)
            ),
            **metrics,
        }
        result_rows.append(row)
        all_counts[strategy] = counts
        all_best[strategy] = best

        print(
            f"{strategy} [{args.mps_ordering}]: "
            f"mean gap={row['mean_normalized_gap']:.4f}, "
            f"best gap={row['best_normalized_gap']:.4f}, "
            f"best exact agreement={row['best_agreement_with_exact_up_to_sign']:.3f}, "
            f"exact hits={row['exact_ground_hits']}",
            flush=True,
        )

    frame = pd.DataFrame(result_rows)
    frame.to_csv(args.out / "real_mps_results.csv", index=False)

    with gzip.open(args.out / "real_mps_counts.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(all_counts, handle)

    with gzip.open(
        args.out / "real_mps_aer_metadata.json.gz", "wt", encoding="utf-8"
    ) as handle:
        json.dump(all_aer_metadata, handle, indent=2)

    np.savez_compressed(
        args.out / "real_mps_best_assignments.npz",
        exact_spins_rcm=exact_spins,
        transferred_center_spins_rcm=np.asarray(
            all_best["transferred_center"]["spins_rcm"], dtype=np.int8
        ),
        hardware_calibrated_spins_rcm=np.asarray(
            all_best["hardware_calibrated"]["spins_rcm"], dtype=np.int8
        ),
    )

    summary = {
        "graph_dir": str(args.graph_dir.resolve()),
        "n_nodes": int(graph.n_nodes),
        "n_edges": int(len(graph.edges)),
        "exact_energy": float(exact_energy),
        "exact_solver_info": solver_info,
        "shots_per_strategy": int(args.shots),
        "bond_dimension": int(args.bond_dimension),
        "truncation_threshold": float(args.truncation_threshold),
        "mps_ordering": str(args.mps_ordering),
        "mps_swap_direction": str(args.mps_swap_direction),
        "mps_edge_order": str(args.mps_edge_order),
        "mps_ordering_diagnostics": mps_ordering_diagnostics,
        "mps_routing_diagnostics": mps_routing_diagnostics,
        "strategies": result_rows,
        "validation": {
            "reconstructed_exact_energy_matches_saved": True,
            "mps_permutation_preserves_exact_energy": True,
            "mps_permutation_preserves_analytic_expectation": True,
            "sampled_bitstrings_mapped_back_to_rcm_labels": True,
            "commuting_rzz_reordering_preserves_qaoa_state": True,
            "bond_cap_covers_dynamic_rank_upper_bound": bool(
                int(args.bond_dimension)
                >= int(mps_routing_diagnostics["dynamic_schmidt_rank_upper_bound"])
            ),
            "both_sampled_means_within_5se_of_analytic": bool(
                frame["sampled_expectation_within_5se"].all()
            ),
            "uses_qpu": False,
            "uses_noise_model": False,
        },
        "next_gate": (
            "Review the best sampled gaps and analytic-vs-sampled agreement. "
            "Only then run hardware transpilation/calibration preflight."
        ),
        "files": {
            "results": "real_mps_results.csv",
            "counts": "real_mps_counts.json.gz",
            "aer_metadata": "real_mps_aer_metadata.json.gz",
            "best_assignments": "real_mps_best_assignments.npz",
        },
    }
    (args.out / "real_mps_summary.json").write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print("\nREAL HCP IDEAL MPS SUMMARY")
    print("=" * 78)
    print(
        "MPS ordering: "
        f"{args.mps_ordering}; maximum crossing matching="
        f"{mps_ordering_diagnostics['maximum_crossing_matching_size']}; "
        "maximum smaller boundary="
        f"{mps_ordering_diagnostics['maximum_smaller_boundary_size']}"
    )
    print(
        "MPS routing: "
        f"edge order={args.mps_edge_order}; "
        f"swap direction={args.mps_swap_direction}; "
        "maximum dynamic smaller boundary="
        f"{mps_routing_diagnostics['maximum_dynamic_smaller_boundary_size']}; "
        "Schmidt-rank upper bound="
        f"{mps_routing_diagnostics['dynamic_schmidt_rank_upper_bound']}; "
        "predicted internal swaps="
        f"{mps_routing_diagnostics['predicted_internal_swaps']}"
    )
    columns = [
        "angle_strategy",
        "analytic_expected_energy",
        "mean_energy",
        "mean_energy_se",
        "sampled_vs_analytic_z",
        "mean_normalized_gap",
        "cvar10_normalized_gap",
        "best_normalized_gap",
        "best_agreement_with_exact_up_to_sign",
        "best_agreement_with_tom_reference_up_to_sign",
        "exact_ground_hits",
        "simulation_seconds",
    ]
    print(frame[columns].to_string(index=False))
    print(f"\nWrote outputs to: {args.out.resolve()}")


if __name__ == "__main__":
    main()
