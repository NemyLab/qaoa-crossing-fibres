#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Iterable
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qiskit import transpile
from qiskit_aer import AerSimulator
import networkx as nx

from step1_crossing_graph import CrossingGraph
from step5_curved_signal_graph import build_curved_signal_graph, exact_milp
from step6_qaoa_small_graph import (
    bind_angles,
    build_qaoa_circuit,
    expected_energy_lightcone_p1,
)

def get_rcm_order(graph):
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.n_nodes))

    for edge in graph.edges.itertuples(index=False):
        nx_graph.add_edge(int(edge.i), int(edge.j))

    return list(nx.utils.reverse_cuthill_mckee_ordering(nx_graph))


def reorder_graph(graph, order):
    import numpy as np

    old_to_new = {
        old_index: new_index
        for new_index, old_index in enumerate(order)
    }

    nodes = graph.nodes.copy()
    nodes["old_node_id"] = nodes["node_id"]
    nodes["node_id"] = nodes["old_node_id"].map(old_to_new)
    nodes = nodes.sort_values("node_id").reset_index(drop=True)

    edges = graph.edges.copy()
    edges["i"] = edges["i"].map(old_to_new)
    edges["j"] = edges["j"].map(old_to_new)

    fields = np.zeros_like(graph.fields)
    true_spins = np.zeros_like(graph.true_spins)

    for old_index, new_index in old_to_new.items():
        fields[new_index] = graph.fields[old_index]
        true_spins[new_index] = graph.true_spins[old_index]

    return CrossingGraph(
        nodes=nodes,
        edges=edges,
        fields=fields,
        true_spins=true_spins,
        grid_size=graph.grid_size,
        seed=graph.seed,
    )


def decode_qiskit_bitstring(bitstring: str, n_qubits: int) -> np.ndarray:
    """Decode displayed Qiskit counts key into graph spins.

    Qiskit displays the highest-index classical bit on the left. Reversing the
    string returns qubit order 0..n-1. The project convention is s=1-2b, so
    computational |0> corresponds to spin +1 and |1> to spin -1.
    """
    clean = bitstring.replace(" ", "")
    if len(clean) != n_qubits:
        raise ValueError(
            f"Expected {n_qubits} measured bits, received {len(clean)}"
        )
    bits = np.fromiter((int(character) for character in clean[::-1]), dtype=int)
    return 1 - 2 * bits


def sample_metrics(
    graph: CrossingGraph,
    counts: dict[str, int],
    exact_energy: float,
    uniform_energy: float,
) -> tuple[dict, list[dict]]:
    total = int(sum(counts.values()))
    if total <= 0:
        raise ValueError("Counts are empty")
    denominator = max(uniform_energy - exact_energy, 1e-12)

    records: list[dict] = []
    energy_sum = 0.0
    ground_hits = 0
    within_1pct_hits = 0
    within_5pct_hits = 0
    best_energy = np.inf
    best_energy_spins: np.ndarray | None = None
    best_accuracy = -np.inf

    threshold_1 = exact_energy + 0.01 * denominator
    threshold_5 = exact_energy + 0.05 * denominator

    for bitstring, count in counts.items():
        spins = decode_qiskit_bitstring(bitstring, graph.n_nodes)
        energy = float(graph.energy(spins))
        accuracy = float(np.mean(spins == graph.true_spins))
        count = int(count)
        energy_sum += count * energy
        if np.isclose(energy, exact_energy, atol=1e-7):
            ground_hits += count
        if energy <= threshold_1 + 1e-12:
            within_1pct_hits += count
        if energy <= threshold_5 + 1e-12:
            within_5pct_hits += count
        if energy < best_energy - 1e-10:
            best_energy = energy
            best_energy_spins = spins.copy()
        best_accuracy = max(best_accuracy, accuracy)
        records.append(
            {
                "bitstring": bitstring,
                "count": count,
                "probability": count / total,
                "energy": energy,
                "normalized_gap": (energy - exact_energy) / denominator,
                "label_accuracy": accuracy,
            }
        )

    records.sort(key=lambda row: (-row["count"], row["energy"]))
    mode = records[0]
    top10_frequency = records[:10]
    top10_contains_ground = any(
        np.isclose(row["energy"], exact_energy, atol=1e-7)
        for row in top10_frequency
    )

    if best_energy_spins is None:
        raise RuntimeError("No sampled state was decoded")

    metrics = {
        "shots": total,
        "n_unique_sampled_states": len(records),
        "sampled_expected_energy": energy_sum / total,
        "sampled_expected_normalized_gap": (
            energy_sum / total - exact_energy
        ) / denominator,
        "best_sampled_energy": float(best_energy),
        "best_sampled_normalized_gap": float(
            (best_energy - exact_energy) / denominator
        ),
        "p_ground_sampled": ground_hits / total,
        "ground_hits": int(ground_hits),
        "p_within_1pct_sampled": within_1pct_hits / total,
        "p_within_5pct_sampled": within_5pct_hits / total,
        "best_energy_sample_label_accuracy": float(
            np.mean(best_energy_spins == graph.true_spins)
        ),
        "maximum_sampled_label_accuracy": float(best_accuracy),
        "mode_energy": float(mode["energy"]),
        "mode_normalized_gap": float(mode["normalized_gap"]),
        "mode_label_accuracy": float(mode["label_accuracy"]),
        "top10_frequency_contains_ground": bool(top10_contains_ground),
    }
    return metrics, records


def build_graph(
    *,
    seed: int,
    geometry: str,
    snr: float,
    degree_cap: int,
    fraction_mode: str,
) -> tuple[CrossingGraph, dict]:
    return build_curved_signal_graph(
        grid_size=5,
        n_gradients=60,
        snr=snr,
        b_value=2000.0,
        geometry=geometry,
        fraction_mode=fraction_mode,
        degree_cap=degree_cap,
        conflict_weight=1.0,
        anchor_weight=0.35,
        sigma_orientation_deg=18.0,
        sigma_displacement_deg=22.0,
        minimum_continuation_weight=0.01,
        seed=seed,
    )


def run_mps_counts(
    graph: CrossingGraph,
    gamma: float,
    beta: float,
    shots: int,
    simulator_seed: int,
    optimization_level: int,
) -> tuple[dict[str, int], dict]:
    circuit, gamma_parameters, beta_parameters = build_qaoa_circuit(
        graph,
        reps=1,
    )
    bound = bind_angles(
        circuit,
        gamma_parameters,
        beta_parameters,
        np.asarray([gamma, beta], dtype=float),
    )
    bound.measure_all()

    import os

    simulator = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_max_bond_dimension=256,
        matrix_product_state_truncation_threshold=1e-12,
        mps_sample_measure_algorithm="mps_apply_measure",
        mps_log_data=True,
    )
    #simulator.set_options(
    #    mps_sample_measure_algorithm="mps_probabilities",
    #)
    isa = transpile(
        bound,
        simulator,
        optimization_level=optimization_level,
        seed_transpiler=simulator_seed,
    )

    job = simulator.run(
        isa,
        shots=int(shots),
        seed_simulator=int(simulator_seed),
    )

    result = job.result()
    counts = result.get_counts(isa)
    metadata = {
        "simulator_method": "matrix_product_state",
        "transpiled_depth": int(isa.depth()),
        "transpiled_size": int(isa.size()),
        "transpiled_two_qubit_gates": int(
            sum(len(instruction.qubits) == 2 for instruction in isa.data)
        ),
        "simulator_seed": int(simulator_seed),
        "optimization_level": int(optimization_level),
    }
    return counts, metadata


def choose_representative_seeds(data: pd.DataFrame) -> dict:
    degree3 = data[data["degree_cap"] == 3].sort_values(
        "ideal_expected_normalized_gap"
    )
    if degree3.empty:
        return {}

    def row_payload(row: pd.Series) -> dict:
        return {
            "seed": int(row["seed"]),
            "degree_cap": int(row["degree_cap"]),
            "ideal_expected_normalized_gap": float(
                row["ideal_expected_normalized_gap"]
            ),
            "best_sampled_normalized_gap": float(
                row["best_sampled_normalized_gap"]
            ),
            "exact_solution_label_accuracy": float(
                row["exact_solution_label_accuracy"]
            ),
        }

    return {
        "easy": row_payload(degree3.iloc[0]),
        "median": row_payload(degree3.iloc[len(degree3) // 2]),
        "hard": row_payload(degree3.iloc[-1]),
    }


def plot_results(data: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5.5))
    positions = np.arange(len(data))
    width = 0.36
    ax.bar(
        positions - width / 2,
        data["ideal_expected_normalized_gap"],
        width=width,
        label="Exact p=1 expectation",
    )
    ax.bar(
        positions + width / 2,
        data["best_sampled_normalized_gap"],
        width=width,
        label="Best MPS sample",
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        [f"d{row.degree_cap}/s{row.seed}" for row in data.itertuples()],
        rotation=45,
        ha="right",
    )
    ax.axhline(1.0, linestyle="--", linewidth=1.0)
    ax.set_ylabel("Normalized energy gap")
    ax.set_title("Frozen p=1 angle transfer to 50-qubit graphs")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "qaoa_50q_normalized_gaps.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5.5))
    ax.scatter(
        data["ideal_expected_normalized_gap"],
        data["sampled_expected_normalized_gap"],
    )
    limits = [
        min(
            data["ideal_expected_normalized_gap"].min(),
            data["sampled_expected_normalized_gap"].min(),
        ),
        max(
            data["ideal_expected_normalized_gap"].max(),
            data["sampled_expected_normalized_gap"].max(),
        ),
    ]
    ax.plot(limits, limits, linestyle="--")
    ax.set_xlabel("Exact light-cone expectation gap")
    ax.set_ylabel("MPS sampled expectation gap")
    ax.set_title("MPS sampling validation")
    fig.tight_layout()
    fig.savefig(output / "mps_expectation_validation.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--angles",
        default="frozen_shared_angles_p1.json",
    )
    parser.add_argument("--n-seeds", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=2000)
    parser.add_argument("--geometry", default="curved_90")
    parser.add_argument("--snr", type=float, default=10.0)
    parser.add_argument("--fraction-mode", default="imbalanced")
    parser.add_argument(
        "--degree-caps",
        nargs="+",
        type=int,
        choices=[3, 4],
        default=[3],
    )
    parser.add_argument("--shots", type=int, default=2048)
    parser.add_argument("--optimization-level", type=int, default=1)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if args.shots <= 0:
        raise ValueError("shots must be positive")

    angles = json.loads(Path(args.angles).read_text())
    if int(angles["reps"]) != 1:
        raise ValueError("Step 7 currently accepts only p=1 angles")
    gamma = float(angles["gamma"][0])
    beta = float(angles["beta"][0])

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    counts_directory = output / "counts"
    counts_directory.mkdir(exist_ok=True)
    top_directory = output / "top_sampled_states"
    top_directory.mkdir(exist_ok=True)

    rows: list[dict] = []
    for degree_cap in args.degree_caps:
        for offset in range(args.n_seeds):
            seed = args.seed_start + offset

            print(f"\nSeed {seed}: building graph...", flush=True)
            start = time.perf_counter()

            graph, diagnostics = build_graph(
                seed=seed,
                geometry=args.geometry,
                snr=args.snr,
                degree_cap=degree_cap,
                fraction_mode=args.fraction_mode,
            )

            print(
                f"Seed {seed}: graph built in "
                f"{time.perf_counter() - start:.2f} s",
                flush=True,
            )

            rcm_order = get_rcm_order(graph)
            graph = reorder_graph(graph, rcm_order)

            print(f"Seed {seed}: solving MILP...", flush=True)
            start = time.perf_counter()

            exact_spins, exact_energy, solver = exact_milp(graph)

            print(
                f"Seed {seed}: MILP solved in "
                f"{time.perf_counter() - start:.2f} s",
                flush=True,
            )

            uniform_energy = 0.0
            denominator = max(uniform_energy - exact_energy, 1e-12)
            ideal_expected_energy = expected_energy_lightcone_p1(
                graph,
                gamma_value=gamma,
                beta_value=beta,
            )

            print(f"Seed {seed}: running MPS simulation...", flush=True)
            start = time.perf_counter()

            counts, simulation_metadata = run_mps_counts(
                graph,
                gamma,
                beta,
                shots=args.shots,
                simulator_seed=300_000 + seed + 1000 * degree_cap,
                optimization_level=args.optimization_level,
            )

            print(
                f"Seed {seed}: MPS finished in "
                f"{time.perf_counter() - start:.2f} s",
                flush=True,
            )

            sampled, top_records = sample_metrics(
                graph,
                counts,
                exact_energy,
                uniform_energy,
            )

            with gzip.open(
                counts_directory / f"degree{degree_cap}_seed{seed}.json.gz",
                "wt",
                encoding="utf-8",
            ) as handle:
                json.dump(counts, handle)
            pd.DataFrame(top_records[:100]).to_csv(
                top_directory / f"degree{degree_cap}_seed{seed}_top100.csv",
                index=False,
            )

            rows.append(
                {
                    "seed": seed,
                    "geometry": args.geometry,
                    "snr": args.snr,
                    "fraction_mode": args.fraction_mode,
                    "degree_cap": degree_cap,
                    "n_nodes_qubits": graph.n_nodes,
                    "n_edges": len(graph.edges),
                    "observed_maximum_degree": int(
                        graph.degree_table()["degree"].max()
                    ),
                    "gamma": gamma,
                    "beta": beta,
                    "exact_energy": exact_energy,
                    "uniform_energy": uniform_energy,
                    "ideal_expected_energy": ideal_expected_energy,
                    "ideal_expected_normalized_gap": (
                        ideal_expected_energy - exact_energy
                    ) / denominator,
                    "exact_solution_label_accuracy": float(
                        np.mean(exact_spins == graph.true_spins)
                    ),
                    "truth_energy": float(graph.energy(graph.true_spins)),
                    "truth_optimality_gap": float(
                        graph.energy(graph.true_spins) - exact_energy
                    ),
                    **diagnostics,
                    **solver,
                    **simulation_metadata,
                    **sampled,
                }
            )
            print(
                f"degree={degree_cap} seed={seed} "
                f"ideal_gap={rows[-1]['ideal_expected_normalized_gap']:.4f} "
                f"best_gap={rows[-1]['best_sampled_normalized_gap']:.4f} "
                f"ground_hits={rows[-1]['ground_hits']}"
            )

    data = pd.DataFrame(rows)
    data.to_csv(output / "qaoa_50q_instance_metrics.csv", index=False)
    summary = (
        data.groupby(
            ["geometry", "snr", "fraction_mode", "degree_cap"],
            as_index=False,
        )
        .agg(
            n_graphs=("seed", "size"),
            n_nodes_qubits=("n_nodes_qubits", "first"),
            mean_edges=("n_edges", "mean"),
            observed_maximum_degree=("observed_maximum_degree", "max"),
            mean_ideal_expected_normalized_gap=(
                "ideal_expected_normalized_gap",
                "mean",
            ),
            median_ideal_expected_normalized_gap=(
                "ideal_expected_normalized_gap",
                "median",
            ),
            mean_sampled_expected_normalized_gap=(
                "sampled_expected_normalized_gap",
                "mean",
            ),
            mean_absolute_expectation_sampling_error=(
                "sampled_expected_normalized_gap",
                lambda values: float(
                    np.mean(
                        np.abs(
                            values.to_numpy()
                            - data.loc[
                                values.index,
                                "ideal_expected_normalized_gap",
                            ].to_numpy()
                        )
                    )
                ),
            ),
            mean_best_sampled_normalized_gap=(
                "best_sampled_normalized_gap",
                "mean",
            ),
            exact_ground_hit_rate=("ground_hits", lambda x: float(np.mean(x > 0))),
            mean_ground_hits=("ground_hits", "mean"),
            mean_p_within_1pct=("p_within_1pct_sampled", "mean"),
            mean_p_within_5pct=("p_within_5pct_sampled", "mean"),
            mean_best_energy_sample_label_accuracy=(
                "best_energy_sample_label_accuracy",
                "mean",
            ),
            mean_exact_solution_label_accuracy=(
                "exact_solution_label_accuracy",
                "mean",
            ),
        )
    )
    summary.to_csv(output / "qaoa_50q_summary.csv", index=False)
    representatives = choose_representative_seeds(data)
    (output / "recommended_hardware_seeds.json").write_text(
        json.dumps(representatives, indent=2)
    )
    plot_results(data, output)

    run_metadata = {
        "angles_file": str(args.angles),
        "gamma": gamma,
        "beta": beta,
        "shots": args.shots,
        "seed_start": args.seed_start,
        "n_seeds": args.n_seeds,
        "geometry": args.geometry,
        "snr": args.snr,
        "fraction_mode": args.fraction_mode,
        "degree_caps": args.degree_caps,
    }
    (output / "run_metadata.json").write_text(
        json.dumps(run_metadata, indent=2)
    )
    print("\nSummary:")
    print(summary.to_string(index=False))
    print("\nRecommended degree-3 hardware seeds:")
    print(json.dumps(representatives, indent=2))


if __name__ == "__main__":
    main()
