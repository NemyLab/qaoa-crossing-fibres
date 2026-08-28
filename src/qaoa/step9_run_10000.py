#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from step5_curved_signal_graph import build_curved_signal_graph, exact_milp
from step9_common import (
    get_rcm_order,
    graph_coefficients,
    label_accuracy,
    normalized_gap,
    qiskit_bitstring_to_spins,
    reorder_graph,
    save_counts,
)


def expected_energy_lightcone_p1(graph, gamma: float, beta: float) -> float:
    """Exact p=1 expectation from small reverse-light-cone circuits."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    z_coefficients, zz_coefficients = graph_coefficients(graph)
    incident: list[list[tuple[int, int, float]]] = [
        [] for _ in range(graph.n_nodes)
    ]
    for i, j, coefficient in zz_coefficients:
        incident[i].append((i, j, coefficient))
        incident[j].append((i, j, coefficient))

    def term_expectation(targets: tuple[int, ...]) -> float:
        relevant = set(targets)
        relevant_edges: dict[tuple[int, int], float] = {}
        for target in targets:
            for i, j, coefficient in incident[target]:
                relevant.add(i)
                relevant.add(j)
                relevant_edges[(min(i, j), max(i, j))] = coefficient
        ordered = sorted(relevant)
        local = {node: index for index, node in enumerate(ordered)}
        circuit = QuantumCircuit(len(ordered))
        circuit.h(range(len(ordered)))
        for node in targets:
            coefficient = z_coefficients[node]
            if abs(coefficient) > 1e-15:
                circuit.rz(2.0 * gamma * coefficient, local[node])
        for (i, j), coefficient in relevant_edges.items():
            circuit.rzz(2.0 * gamma * coefficient, local[i], local[j])
        for node in targets:
            circuit.rx(2.0 * beta, local[node])

        state = Statevector.from_instruction(circuit)
        probabilities = np.square(np.abs(np.asarray(state.data)))
        indices = np.arange(len(probabilities), dtype=np.uint64)
        eigenvalue = np.ones(len(probabilities), dtype=float)
        for node in targets:
            bit = (indices >> np.uint64(local[node])) & np.uint64(1)
            eigenvalue *= 1.0 - 2.0 * bit.astype(float)
        return float(probabilities @ eigenvalue)

    expected = 0.0
    for i, coefficient in enumerate(z_coefficients):
        if abs(coefficient) > 1e-15:
            expected += coefficient * term_expectation((i,))
    for i, j, coefficient in zz_coefficients:
        expected += coefficient * term_expectation((i, j))
    return float(expected)


def build_measured_qaoa_circuit(graph, gamma: float, beta: float):
    from qiskit import QuantumCircuit

    z, zz = graph_coefficients(graph)
    circuit = QuantumCircuit(graph.n_nodes, graph.n_nodes)
    circuit.h(range(graph.n_nodes))
    for i, coefficient in enumerate(z):
        if abs(coefficient) > 1e-15:
            circuit.rz(2.0 * gamma * coefficient, i)
    for i, j, coefficient in zz:
        circuit.rzz(2.0 * gamma * coefficient, i, j)
    circuit.rx(2.0 * beta, range(graph.n_nodes))
    circuit.measure(range(graph.n_nodes), range(graph.n_nodes))
    return circuit


def parse_bond_map(path: Path) -> dict[int, int]:
    payload = json.loads(path.read_text())
    return {int(seed): int(value) for seed, value in payload.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", type=Path, default=Path("frozen_shared_angles_p1.json"))
    parser.add_argument("--bond-map", type=Path, default=Path("bond_dimensions.json"))
    parser.add_argument("--seeds", nargs="+", type=int, default=[2000, 2001, 2002, 2003])
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--geometry", default="curved_90")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--fraction-mode", default="imbalanced")
    parser.add_argument("--degree-cap", type=int, default=3)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--n-gradients", type=int, default=60)
    parser.add_argument("--b-value", type=float, default=2000.0)
    parser.add_argument("--conflict-weight", type=float, default=1.0)
    parser.add_argument("--anchor-weight", type=float, default=0.35)
    parser.add_argument("--sigma-orientation-deg", type=float, default=18.0)
    parser.add_argument("--sigma-displacement-deg", type=float, default=22.0)
    parser.add_argument("--minimum-continuation-weight", type=float, default=0.01)
    parser.add_argument("--truncation-threshold", type=float, default=1e-12)
    parser.add_argument("--simulator-seed-base", type=int, default=930_000)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    from qiskit import transpile
    from qiskit_aer import AerSimulator

    angles = json.loads(args.angles.read_text())
    gamma = float(angles["gamma"][0])
    beta = float(angles["beta"][0])
    bond_map = parse_bond_map(args.bond_map)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "counts").mkdir(exist_ok=True)

    rows: list[dict] = []
    for seed in args.seeds:
        if seed not in bond_map:
            raise ValueError(f"No MPS bond dimension configured for seed {seed}")
        bond_dimension = bond_map[seed]
        print(f"\nSeed {seed}: building and RCM-reordering graph...", flush=True)
        start = time.perf_counter()
        graph, diagnostics = build_curved_signal_graph(
            grid_size=args.grid_size,
            n_gradients=args.n_gradients,
            snr=args.snr,
            b_value=args.b_value,
            geometry=args.geometry,
            fraction_mode=args.fraction_mode,
            degree_cap=args.degree_cap,
            conflict_weight=args.conflict_weight,
            anchor_weight=args.anchor_weight,
            sigma_orientation_deg=args.sigma_orientation_deg,
            sigma_displacement_deg=args.sigma_displacement_deg,
            minimum_continuation_weight=args.minimum_continuation_weight,
            seed=seed,
        )
        graph = reorder_graph(graph, get_rcm_order(graph))
        graph_seconds = time.perf_counter() - start

        print(f"Seed {seed}: solving exact MILP...", flush=True)
        start = time.perf_counter()
        exact_spins, exact_energy, solver_info = exact_milp(graph)
        milp_seconds = time.perf_counter() - start

        ideal_expected_energy = expected_energy_lightcone_p1(graph, gamma, beta)
        ideal_expected_gap = normalized_gap(ideal_expected_energy, exact_energy)
        circuit = build_measured_qaoa_circuit(graph, gamma, beta)
        simulator = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_max_bond_dimension=bond_dimension,
            matrix_product_state_truncation_threshold=args.truncation_threshold,
            mps_sample_measure_algorithm="mps_apply_measure",
            mps_log_data=True,
        )
        compiled = transpile(circuit, simulator, optimization_level=0)

        print(
            f"Seed {seed}: MPS sampling {args.shots} shots "
            f"with bond dimension {bond_dimension}...",
            flush=True,
        )
        start = time.perf_counter()
        result = simulator.run(
            compiled,
            shots=args.shots,
            seed_simulator=args.simulator_seed_base + seed,
        ).result()
        mps_seconds = time.perf_counter() - start
        counts = {str(key): int(value) for key, value in result.get_counts().items()}

        energies = []
        accuracies = []
        weights = []
        best_bitstring = None
        best_energy = np.inf
        best_accuracy = np.nan
        exact_hits = 0
        for bitstring, count in counts.items():
            spins = qiskit_bitstring_to_spins(bitstring, graph.n_nodes)
            energy = float(graph.energy(spins))
            accuracy = label_accuracy(spins, graph.true_spins)
            energies.append(energy)
            accuracies.append(accuracy)
            weights.append(count)
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring
                best_accuracy = accuracy
            if np.isclose(energy, exact_energy, atol=1e-8):
                exact_hits += count

        sampled_expected_energy = float(np.average(energies, weights=weights))
        sampled_expected_gap = normalized_gap(sampled_expected_energy, exact_energy)
        best_gap = normalized_gap(best_energy, exact_energy)
        metadata = {
            "seed": seed,
            "shots": args.shots,
            "bond_dimension": bond_dimension,
            "truncation_threshold": args.truncation_threshold,
            "gamma": gamma,
            "beta": beta,
            "geometry": args.geometry,
            "snr": args.snr,
            "fraction_mode": args.fraction_mode,
            "degree_cap": args.degree_cap,
            "exact_energy": exact_energy,
            "ideal_expected_energy": ideal_expected_energy,
            "ideal_expected_normalized_gap": ideal_expected_gap,
        }
        save_counts(
            args.out / "counts" / f"seed_{seed}_counts.json.gz",
            counts,
            metadata,
        )
        rows.append(
            {
                "seed": seed,
                "shots": args.shots,
                "bond_dimension": bond_dimension,
                "n_nodes_qubits": graph.n_nodes,
                "n_edges": len(graph.edges),
                "maximum_degree": int(graph.degree_table()["degree"].max()),
                "exact_energy": exact_energy,
                "exact_solution_label_accuracy": label_accuracy(exact_spins, graph.true_spins),
                "ideal_expected_energy": ideal_expected_energy,
                "ideal_expected_normalized_gap": ideal_expected_gap,
                "sampled_expected_energy": sampled_expected_energy,
                "sampled_expected_normalized_gap": sampled_expected_gap,
                "sampled_vs_ideal_gap_error": abs(sampled_expected_gap - ideal_expected_gap),
                "best_sampled_bitstring": best_bitstring,
                "best_sampled_energy": best_energy,
                "best_sampled_normalized_gap": best_gap,
                "best_sampled_label_accuracy": best_accuracy,
                "exact_ground_hits": exact_hits,
                "n_unique_bitstrings": len(counts),
                "graph_seconds": graph_seconds,
                "milp_seconds": milp_seconds,
                "mps_seconds": mps_seconds,
                "mip_node_count": int(solver_info.get("mip_node_count", -1)),
                **diagnostics,
            }
        )
        print(
            f"Seed {seed}: best_gap={best_gap:.4f}, "
            f"sampled_gap={sampled_expected_gap:.4f}, "
            f"ideal_gap={ideal_expected_gap:.4f}, "
            f"exact_hits={exact_hits}, MPS={mps_seconds:.2f}s",
            flush=True,
        )

    table = pd.DataFrame(rows)
    table.to_csv(args.out / "qaoa_10000_instance_metrics.csv", index=False)
    summary = {
        "n_seeds": int(len(table)),
        "shots_per_seed": int(args.shots),
        "mean_ideal_expected_normalized_gap": float(table["ideal_expected_normalized_gap"].mean()),
        "mean_sampled_expected_normalized_gap": float(table["sampled_expected_normalized_gap"].mean()),
        "mean_sampled_vs_ideal_gap_error": float(table["sampled_vs_ideal_gap_error"].mean()),
        "mean_best_sampled_normalized_gap": float(table["best_sampled_normalized_gap"].mean()),
        "mean_best_sampled_label_accuracy": float(table["best_sampled_label_accuracy"].mean()),
        "total_exact_ground_hits": int(table["exact_ground_hits"].sum()),
    }
    (args.out / "qaoa_10000_summary.json").write_text(json.dumps(summary, indent=2))
    print("\nSummary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
