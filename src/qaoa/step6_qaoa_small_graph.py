#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import qmc

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import Statevector

from step1_crossing_graph import CrossingGraph
from step4_signal_derived_graph import build_signal_derived_graph


class BasisEnergyCache:
    """All computational-basis spins and energies for <=24 nodes.

    Qiskit basis bit b_i is decoded as s_i = 1 - 2 b_i, so |0> means +1
    and |1> means -1. This makes the graph spin equal to the Z eigenvalue.
    """

    def __init__(self, n_nodes: int):
        if n_nodes > 24:
            raise ValueError("BasisEnergyCache is limited to 24 nodes")
        self.n_nodes = int(n_nodes)
        total = 1 << self.n_nodes
        indices = np.arange(total, dtype=np.uint32)
        positions = np.arange(self.n_nodes, dtype=np.uint32)
        bits = ((indices[:, None] >> positions[None, :]) & 1).astype(np.int8)
        self.spins = (1 - 2 * bits).astype(np.int8)

    def energies(self, graph: CrossingGraph) -> np.ndarray:
        if graph.n_nodes != self.n_nodes:
            raise ValueError("Graph node count differs from cache")
        values = -(self.spins @ graph.fields)
        for edge in graph.edges.itertuples(index=False):
            coefficient = (
                -float(edge.weight)
                if edge.edge_type == "continuation"
                else +float(edge.weight)
            )
            values += (
                coefficient
                * self.spins[:, int(edge.i)]
                * self.spins[:, int(edge.j)]
            )
        return values.astype(float)

    def exact(self, graph: CrossingGraph) -> tuple[np.ndarray, float, np.ndarray]:
        energies = self.energies(graph)
        minimum = float(energies.min())
        mask = np.isclose(energies, minimum, atol=1e-10)
        first = int(np.flatnonzero(mask)[0])
        return self.spins[first].astype(int).copy(), minimum, mask

    def index_for_spins(self, spins: np.ndarray) -> int:
        spins = np.asarray(spins, dtype=int)
        if spins.shape != (self.n_nodes,):
            raise ValueError("spins has wrong shape")
        bits = ((1 - spins) // 2).astype(np.uint64)
        powers = (1 << np.arange(self.n_nodes, dtype=np.uint64))
        return int(np.dot(bits, powers))


def graph_coefficients(graph: CrossingGraph) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Return Z and ZZ coefficients for H_C."""
    z = -np.asarray(graph.fields, dtype=float).copy()
    zz: list[tuple[int, int, float]] = []
    for edge in graph.edges.itertuples(index=False):
        coefficient = (
            -float(edge.weight)
            if edge.edge_type == "continuation"
            else +float(edge.weight)
        )
        zz.append((int(edge.i), int(edge.j), coefficient))
    return z, zz




def expected_energy_lightcone_p1(
    graph: CrossingGraph,
    gamma_value: float,
    beta_value: float,
) -> float:
    """Exact p=1 expectation from small reverse-lightcone circuits.

    For a degree-three graph, each Z or ZZ term requires at most the target
    nodes and their immediate neighbours. This avoids repeatedly simulating
    the complete 18-qubit state during angle training.
    """
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

        # Only fields on target nodes can affect the target observable at p=1.
        for node in targets:
            coefficient = z_coefficients[node]
            if abs(coefficient) > 1e-15:
                circuit.rz(
                    2.0 * gamma_value * coefficient,
                    local[node],
                )
        for (i, j), coefficient in relevant_edges.items():
            circuit.rzz(
                2.0 * gamma_value * coefficient,
                local[i],
                local[j],
            )
        for node in targets:
            circuit.rx(2.0 * beta_value, local[node])

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


def build_qaoa_circuit(graph: CrossingGraph, reps: int = 1):
    if reps < 1:
        raise ValueError("reps must be at least one")
    gamma = ParameterVector("gamma", reps)
    beta = ParameterVector("beta", reps)
    circuit = QuantumCircuit(graph.n_nodes)
    circuit.h(range(graph.n_nodes))
    z, zz = graph_coefficients(graph)

    for layer in range(reps):
        for qubit, coefficient in enumerate(z):
            if abs(coefficient) > 1e-15:
                circuit.rz(2.0 * gamma[layer] * coefficient, qubit)
        for i, j, coefficient in zz:
            circuit.rzz(2.0 * gamma[layer] * coefficient, i, j)
        for qubit in range(graph.n_nodes):
            circuit.rx(2.0 * beta[layer], qubit)
    return circuit, gamma, beta


def bind_angles(circuit, gamma, beta, theta: np.ndarray):
    theta = np.asarray(theta, dtype=float)
    reps = len(gamma)
    if theta.shape != (2 * reps,):
        raise ValueError("theta must contain reps gamma and reps beta values")
    mapping = {gamma[i]: float(theta[i]) for i in range(reps)}
    mapping.update({beta[i]: float(theta[reps + i]) for i in range(reps)})
    return circuit.assign_parameters(mapping, inplace=False)


class QAOAInstance:
    def __init__(self, graph: CrossingGraph, reps: int, cache: BasisEnergyCache):
        self.graph = graph
        self.reps = reps
        self.cache = cache
        self.circuit, self.gamma, self.beta = build_qaoa_circuit(graph, reps)
        self.energies = cache.energies(graph)
        self.exact_energy = float(self.energies.min())
        self.ground_mask = np.isclose(self.energies, self.exact_energy, atol=1e-10)
        self.truth_index = cache.index_for_spins(graph.true_spins)
        self.uniform_energy = float(self.energies.mean())
        self.denominator = max(self.uniform_energy - self.exact_energy, 1e-12)

    def probabilities(self, theta: np.ndarray) -> np.ndarray:
        bound = bind_angles(self.circuit, self.gamma, self.beta, theta)
        state = Statevector.from_instruction(bound)
        return np.square(np.abs(np.asarray(state.data)))

    def expected_normalized_gap(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if self.reps == 1:
            expectation = expected_energy_lightcone_p1(
                self.graph,
                gamma_value=float(theta[0]),
                beta_value=float(theta[1]),
            )
        else:
            probabilities = self.probabilities(theta)
            expectation = float(probabilities @ self.energies)
        return (expectation - self.exact_energy) / self.denominator

    def metrics(self, theta: np.ndarray) -> dict[str, float | int | bool]:
        probabilities = self.probabilities(theta)
        expected_energy = float(probabilities @ self.energies)
        p_ground = float(probabilities[self.ground_mask].sum())
        p_truth = float(probabilities[self.truth_index])
        threshold_1pct = self.exact_energy + 0.01 * self.denominator
        threshold_5pct = self.exact_energy + 0.05 * self.denominator
        p_within_1pct = float(
            probabilities[self.energies <= threshold_1pct + 1e-12].sum()
        )
        p_within_5pct = float(
            probabilities[self.energies <= threshold_5pct + 1e-12].sum()
        )
        mode_index = int(np.argmax(probabilities))
        mode_spins = self.cache.spins[mode_index]
        mode_energy = float(self.energies[mode_index])
        mode_accuracy = float(np.mean(mode_spins == self.graph.true_spins))
        top_indices = np.argpartition(probabilities, -10)[-10:]
        top10_best_energy = float(self.energies[top_indices].min())
        top10_contains_ground = bool(np.any(self.ground_mask[top_indices]))
        return {
            "seed": int(self.graph.seed),
            "n_nodes_qubits": int(self.graph.n_nodes),
            "n_edges": int(len(self.graph.edges)),
            "exact_energy": self.exact_energy,
            "uniform_energy": self.uniform_energy,
            "expected_energy": expected_energy,
            "expected_normalized_gap": (
                expected_energy - self.exact_energy
            ) / self.denominator,
            "p_ground": p_ground,
            "uniform_p_ground": float(self.ground_mask.mean()),
            "p_ground_enrichment": p_ground / max(float(self.ground_mask.mean()), 1e-15),
            "p_truth": p_truth,
            "p_within_1pct": p_within_1pct,
            "p_within_5pct": p_within_5pct,
            "expected_ground_hits_8192": 8192.0 * p_ground,
            "probability_at_least_one_ground_8192": float(
                1.0 - (1.0 - p_ground) ** 8192
            ),
            "mode_energy": mode_energy,
            "mode_is_ground": bool(np.isclose(mode_energy, self.exact_energy, atol=1e-10)),
            "mode_label_accuracy": mode_accuracy,
            "top10_best_energy": top10_best_energy,
            "top10_contains_ground": top10_contains_ground,
        }


def make_graph(seed: int, snr: float, fraction_mode: str) -> CrossingGraph:
    graph, _ = build_signal_derived_graph(
        grid_size=3,
        n_gradients=60,
        snr=snr,
        b_value=1000.0,
        fraction_mode=fraction_mode,
        conflict_weight=1.0,
        anchor_weight=0.35,
        sigma_orientation_deg=15.0,
        sigma_displacement_deg=15.0,
        minimum_continuation_weight=0.15,
        seed=seed,
    )
    return graph


def train_shared_angles(
    instances: list[QAOAInstance],
    reps: int,
    sobol_points: int,
    seed: int,
) -> dict:
    if not instances:
        raise ValueError("No training instances")
    dimension = 2 * reps
    sampler = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    exponent = int(np.ceil(np.log2(max(sobol_points, 2))))
    points = sampler.random_base2(exponent)[:sobol_points]
    lower = np.array([0.0] * reps + [0.0] * reps)
    upper = np.array([np.pi] * reps + [0.5 * np.pi] * reps)
    candidates = lower + points * (upper - lower)

    def objective(theta: np.ndarray) -> float:
        return float(
            np.mean(
                [instance.expected_normalized_gap(theta) for instance in instances]
            )
        )

    values = np.array([objective(theta) for theta in candidates])
    best = candidates[int(np.argmin(values))]
    local = minimize(
        objective,
        best,
        method="Nelder-Mead",
        options={"maxiter": 180, "xatol": 1e-4, "fatol": 1e-6},
    )
    theta = local.x if local.fun <= values.min() else best
    return {
        "reps": int(reps),
        "gamma": [float(value) for value in theta[:reps]],
        "beta": [float(value) for value in theta[reps:]],
        "training_objective_mean_normalized_gap": float(objective(theta)),
        "sobol_points": int(sobol_points),
        "optimizer_success": bool(local.success),
        "optimizer_message": str(local.message),
        "n_training_instances": int(len(instances)),
    }


def plot_metrics(data: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(data["uniform_p_ground"], data["p_ground"])
    upper = max(data["p_ground"].max(), data["uniform_p_ground"].max())
    ax.plot([0.0, upper], [0.0, upper], linestyle="--")
    ax.set_xlabel("Uniform ground-state probability")
    ax.set_ylabel("QAOA ground-state probability")
    ax.set_title("Transferred QAOA angles on held-out graphs")
    fig.tight_layout()
    fig.savefig(output / "ground_probability_vs_uniform.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(data["expected_normalized_gap"], bins=12)
    ax.set_xlabel("Expected normalized energy gap")
    ax.set_ylabel("Held-out graph count")
    ax.set_title("QAOA expected energy: 0 is exact, 1 is uniform")
    fig.tight_layout()
    fig.savefig(output / "normalized_gap_histogram.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--train-cases", type=int, default=6)
    parser.add_argument("--test-cases", type=int, default=20)
    parser.add_argument("--train-start", type=int, default=0)
    parser.add_argument("--test-start", type=int, default=1000)
    parser.add_argument("--snr", type=float, default=8.0)
    parser.add_argument("--fraction-mode", default="imbalanced")
    parser.add_argument("--sobol-points", type=int, default=64)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    cache = BasisEnergyCache(18)

    train_graphs = [
        make_graph(seed, args.snr, args.fraction_mode)
        for seed in range(args.train_start, args.train_start + args.train_cases)
    ]
    train_instances = [QAOAInstance(g, args.reps, cache) for g in train_graphs]
    angles = train_shared_angles(
        train_instances,
        reps=args.reps,
        sobol_points=args.sobol_points,
        seed=12345,
    )
    theta = np.array(angles["gamma"] + angles["beta"], dtype=float)
    (output / f"shared_angles_p{args.reps}.json").write_text(
        json.dumps(angles, indent=2)
    )

    rows = []
    for split, start, count in (
        ("train", args.train_start, args.train_cases),
        ("test", args.test_start, args.test_cases),
    ):
        for seed in range(start, start + count):
            graph = make_graph(seed, args.snr, args.fraction_mode)
            instance = QAOAInstance(graph, args.reps, cache)
            rows.append({"split": split, **instance.metrics(theta)})

    metrics = pd.DataFrame(rows)
    metrics.to_csv(output / "qaoa_instance_metrics.csv", index=False)
    summary = (
        metrics.groupby("split", as_index=False)
        .agg(
            n_graphs=("seed", "size"),
            mean_expected_normalized_gap=("expected_normalized_gap", "mean"),
            median_expected_normalized_gap=("expected_normalized_gap", "median"),
            mean_p_ground=("p_ground", "mean"),
            median_p_ground=("p_ground", "median"),
            mean_uniform_p_ground=("uniform_p_ground", "mean"),
            mean_p_ground_enrichment=("p_ground_enrichment", "mean"),
            mean_p_within_1pct=("p_within_1pct", "mean"),
            mean_p_within_5pct=("p_within_5pct", "mean"),
            mean_expected_ground_hits_8192=("expected_ground_hits_8192", "mean"),
            mean_probability_at_least_one_ground_8192=(
                "probability_at_least_one_ground_8192", "mean"
            ),
            mode_ground_rate=("mode_is_ground", "mean"),
            top10_ground_rate=("top10_contains_ground", "mean"),
            mean_mode_label_accuracy=("mode_label_accuracy", "mean"),
        )
    )
    summary.to_csv(output / "qaoa_summary.csv", index=False)
    plot_metrics(metrics[metrics["split"] == "test"], output)
    print(json.dumps(angles, indent=2))
    print("\nSummary:")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
