#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

STRATEGIES = ("hardware_calibrated", "transferred_center")
DEFAULT_BUDGETS = (100, 256, 500, 1000, 2000, 5000)
METHODS = (
    "qpu_hardware_calibrated",
    "qpu_transferred_center",
    "uniform_random",
    "greedy_random_restart",
    "simulated_annealing",
)


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def load_counts(path: Path) -> dict[str, dict[str, int]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if "counts_by_pub_index" not in payload:
        raise ValueError(f"{path}: missing counts_by_pub_index")
    return {
        str(pub): {str(bit): int(count) for bit, count in counts.items()}
        for pub, counts in payload["counts_by_pub_index"].items()
    }


def read_cohort(directory: Path, prefix: str, cohort: str):
    results_path = directory / f"{prefix}_hardware_results.csv"
    counts_path = directory / f"{prefix}_hardware_counts.json.gz"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    if not counts_path.exists():
        raise FileNotFoundError(counts_path)
    frame = pd.read_csv(results_path)
    frame["cohort"] = cohort
    frame["counts_source"] = prefix
    return frame, load_counts(counts_path)


def graph_ising_coefficients(graph):
    """Return H(s)=sum_i b_i s_i + sum_(i,j) J_ij s_i s_j."""
    linear = -np.asarray(graph.fields, dtype=float).copy()
    edge_i: list[int] = []
    edge_j: list[int] = []
    edge_jij: list[float] = []
    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(graph.n_nodes)]
    for edge in graph.edges.itertuples(index=False):
        i = int(edge.i)
        j = int(edge.j)
        weight = float(edge.weight)
        if edge.edge_type == "continuation":
            coupling = -weight
        elif edge.edge_type == "conflict":
            coupling = +weight
        else:
            raise ValueError(f"Unknown edge type: {edge.edge_type}")
        edge_i.append(i)
        edge_j.append(j)
        edge_jij.append(coupling)
        adjacency[i].append((j, coupling))
        adjacency[j].append((i, coupling))
    return (
        linear,
        np.asarray(edge_i, dtype=np.int64),
        np.asarray(edge_j, dtype=np.int64),
        np.asarray(edge_jij, dtype=float),
        adjacency,
    )


def energies_batch(
    spins: np.ndarray,
    linear: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_jij: np.ndarray,
) -> np.ndarray:
    spins = np.asarray(spins, dtype=np.int8)
    if spins.ndim == 1:
        spins = spins[None, :]
    values = spins @ linear
    if edge_i.size:
        values = values + (spins[:, edge_i] * spins[:, edge_j]) @ edge_jij
    return np.asarray(values, dtype=float)


def normalized_gap(energy: float | np.ndarray, exact_energy: float):
    denominator = -float(exact_energy)
    if denominator <= 0:
        raise ValueError("Expected negative exact energy")
    return (np.asarray(energy, dtype=float) - exact_energy) / denominator


def label_accuracy(spins: np.ndarray, truth: np.ndarray) -> float:
    """Match the project-wide label-accuracy definition exactly."""
    return float(np.mean(np.asarray(spins) == np.asarray(truth)))


def qiskit_bitstring_to_spins(bitstring: str, n_nodes: int) -> np.ndarray:
    clean = bitstring.replace(" ", "").replace("_", "")
    if len(clean) != n_nodes or set(clean) - {"0", "1"}:
        raise ValueError(f"Invalid {n_nodes}-qubit bitstring: {bitstring!r}")
    bits = np.fromiter((int(c) for c in clean[::-1]), dtype=np.int8)
    return 1 - 2 * bits


@dataclass
class TrialResult:
    best_energy: float
    best_accuracy: float
    exact_hit: bool
    evaluations_used: int


def random_states(rng: np.random.Generator, n_states: int, n_nodes: int) -> np.ndarray:
    bits = rng.integers(0, 2, size=(n_states, n_nodes), dtype=np.int8)
    return 1 - 2 * bits


def uniform_trials(
    *,
    rng: np.random.Generator,
    repeats: int,
    budget: int,
    n_nodes: int,
    linear: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_jij: np.ndarray,
    truth: np.ndarray,
    exact_energy: float,
) -> list[TrialResult]:
    # Process all repetitions together while keeping memory bounded.
    results: list[TrialResult] = []
    chunk_repeats = max(1, min(repeats, max(1, 1_000_000 // budget)))
    done = 0
    while done < repeats:
        current = min(chunk_repeats, repeats - done)
        states = random_states(rng, current * budget, n_nodes)
        energies = energies_batch(states, linear, edge_i, edge_j, edge_jij).reshape(current, budget)
        states = states.reshape(current, budget, n_nodes)
        best_indices = np.argmin(energies, axis=1)
        for row in range(current):
            idx = int(best_indices[row])
            best_energy = float(energies[row, idx])
            best_spins = states[row, idx]
            results.append(
                TrialResult(
                    best_energy=best_energy,
                    best_accuracy=label_accuracy(best_spins, truth),
                    exact_hit=bool(np.isclose(best_energy, exact_energy, atol=1e-8)),
                    evaluations_used=budget,
                )
            )
        done += current
    return results


def initialise_local_field(spins: np.ndarray, linear: np.ndarray, adjacency):
    local = linear.copy()
    for i, neighbours in enumerate(adjacency):
        for j, coupling in neighbours:
            local[i] += coupling * spins[j]
    return local


def greedy_random_restart_trial(
    *,
    rng: np.random.Generator,
    budget: int,
    n_nodes: int,
    linear: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_jij: np.ndarray,
    adjacency,
    truth: np.ndarray,
    exact_energy: float,
) -> TrialResult:
    evaluations = 0
    best_energy = math.inf
    best_spins = None

    while evaluations < budget:
        spins = random_states(rng, 1, n_nodes)[0]
        energy = float(energies_batch(spins, linear, edge_i, edge_j, edge_jij)[0])
        evaluations += 1
        if energy < best_energy:
            best_energy = energy
            best_spins = spins.copy()
        if evaluations >= budget:
            break

        local_field = initialise_local_field(spins, linear, adjacency)
        while evaluations < budget:
            order = rng.permutation(n_nodes)
            n_check = min(n_nodes, budget - evaluations)
            candidates = order[:n_check]
            deltas = -2.0 * spins[candidates].astype(float) * local_field[candidates]
            evaluations += n_check
            best_pos = int(np.argmin(deltas))
            delta = float(deltas[best_pos])
            if delta >= -1e-12:
                # A complete neighbourhood with no improvement is a local optimum.
                # For a partial neighbourhood, use a new random restart as well; this
                # avoids silently spending queries without producing a new candidate.
                break
            index = int(candidates[best_pos])
            old_spin = int(spins[index])
            spins[index] = -old_spin
            energy += delta
            for neighbour, coupling in adjacency[index]:
                local_field[neighbour] += -2.0 * coupling * old_spin
            if energy < best_energy:
                best_energy = float(energy)
                best_spins = spins.copy()
            if np.isclose(best_energy, exact_energy, atol=1e-8):
                # Continue only if there is remaining budget? The best cannot improve,
                # so stopping preserves the stated maximum query budget.
                return TrialResult(
                    best_energy=float(best_energy),
                    best_accuracy=label_accuracy(best_spins, truth),
                    exact_hit=True,
                    evaluations_used=evaluations,
                )

    assert best_spins is not None
    direct = float(energies_batch(best_spins, linear, edge_i, edge_j, edge_jij)[0])
    if not np.isclose(direct, best_energy, atol=1e-8):
        raise RuntimeError("Greedy incremental energy mismatch")
    return TrialResult(
        best_energy=float(best_energy),
        best_accuracy=label_accuracy(best_spins, truth),
        exact_hit=bool(np.isclose(best_energy, exact_energy, atol=1e-8)),
        evaluations_used=evaluations,
    )


def flip_delta_scale(linear: np.ndarray, adjacency) -> float:
    bounds = []
    for i, neighbours in enumerate(adjacency):
        bounds.append(2.0 * (abs(linear[i]) + sum(abs(c) for _, c in neighbours)))
    scale = float(np.median(np.asarray(bounds, dtype=float)))
    return max(scale, 1e-6)


def simulated_annealing_trial(
    *,
    rng: np.random.Generator,
    budget: int,
    n_nodes: int,
    linear: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_jij: np.ndarray,
    adjacency,
    truth: np.ndarray,
    exact_energy: float,
    restarts: int,
) -> TrialResult:
    if restarts < 1:
        raise ValueError("SA restarts must be positive")
    allocations = np.full(restarts, budget // restarts, dtype=int)
    allocations[: budget % restarts] += 1
    allocations = allocations[allocations > 0]

    best_energy = math.inf
    best_spins = None
    evaluations = 0
    scale = flip_delta_scale(linear, adjacency)

    for chain_budget in allocations:
        spins = random_states(rng, 1, n_nodes)[0]
        energy = float(energies_batch(spins, linear, edge_i, edge_j, edge_jij)[0])
        evaluations += 1
        local_field = initialise_local_field(spins, linear, adjacency)
        if energy < best_energy:
            best_energy = energy
            best_spins = spins.copy()
        proposals = max(0, int(chain_budget) - 1)
        if proposals == 0:
            continue
        t_start = scale
        t_end = 0.01 * scale
        for step in range(proposals):
            fraction = step / max(1, proposals - 1)
            temperature = t_start * (t_end / t_start) ** fraction
            index = int(rng.integers(0, n_nodes))
            delta = float(-2.0 * spins[index] * local_field[index])
            evaluations += 1
            if delta <= 0.0 or rng.random() < math.exp(-delta / max(temperature, 1e-12)):
                old_spin = int(spins[index])
                spins[index] = -old_spin
                energy += delta
                for neighbour, coupling in adjacency[index]:
                    local_field[neighbour] += -2.0 * coupling * old_spin
                if energy < best_energy:
                    best_energy = float(energy)
                    best_spins = spins.copy()
                    if np.isclose(best_energy, exact_energy, atol=1e-8):
                        return TrialResult(
                            best_energy=float(best_energy),
                            best_accuracy=label_accuracy(best_spins, truth),
                            exact_hit=True,
                            evaluations_used=evaluations,
                        )

    assert best_spins is not None
    direct = float(energies_batch(best_spins, linear, edge_i, edge_j, edge_jij)[0])
    if not np.isclose(direct, best_energy, atol=1e-8):
        raise RuntimeError("SA incremental energy mismatch")
    return TrialResult(
        best_energy=float(best_energy),
        best_accuracy=label_accuracy(best_spins, truth),
        exact_hit=bool(np.isclose(best_energy, exact_energy, atol=1e-8)),
        evaluations_used=evaluations,
    )


def qpu_empirical_trials(
    *,
    rng: np.random.Generator,
    counts: dict[str, int],
    graph,
    exact_energy: float,
    repeats: int,
    budget: int,
) -> list[TrialResult]:
    bitstrings = list(counts)
    weights = np.asarray([counts[b] for b in bitstrings], dtype=float)
    probabilities = weights / weights.sum()
    spins = np.vstack([qiskit_bitstring_to_spins(b, graph.n_nodes) for b in bitstrings])
    energies = np.asarray([float(graph.energy(s)) for s in spins], dtype=float)
    accuracies = np.asarray(
        [label_accuracy(s, graph.true_spins) for s in spins], dtype=float
    )
    output: list[TrialResult] = []
    for _ in range(repeats):
        chosen = rng.choice(len(bitstrings), size=budget, replace=True, p=probabilities)
        local_energies = energies[chosen]
        best_local = int(np.argmin(local_energies))
        support_index = int(chosen[best_local])
        best_energy = float(energies[support_index])
        output.append(
            TrialResult(
                best_energy=best_energy,
                best_accuracy=float(accuracies[support_index]),
                exact_hit=bool(np.isclose(best_energy, exact_energy, atol=1e-8)),
                evaluations_used=budget,
            )
        )
    return output


def summarize_trials(
    *,
    seed: int,
    budget: int,
    method: str,
    trials: list[TrialResult],
    exact_energy: float,
) -> dict:
    energies = np.asarray([t.best_energy for t in trials], dtype=float)
    gaps = normalized_gap(energies, exact_energy)
    accuracies = np.asarray([t.best_accuracy for t in trials], dtype=float)
    exact_hits = np.asarray([t.exact_hit for t in trials], dtype=float)
    evaluations = np.asarray([t.evaluations_used for t in trials], dtype=float)
    return {
        "seed": int(seed),
        "budget": int(budget),
        "method": method,
        "n_trials": int(len(trials)),
        "mean_best_energy": float(energies.mean()),
        "sd_best_energy": float(energies.std(ddof=1)),
        "mean_best_gap": float(gaps.mean()),
        "sd_best_gap": float(gaps.std(ddof=1)),
        "median_best_gap": float(np.median(gaps)),
        "q10_best_gap": float(np.quantile(gaps, 0.10)),
        "q90_best_gap": float(np.quantile(gaps, 0.90)),
        "mean_best_label_accuracy": float(accuracies.mean()),
        "sd_best_label_accuracy": float(accuracies.std(ddof=1)),
        "exact_recovery_probability": float(exact_hits.mean()),
        "mean_evaluations_used": float(evaluations.mean()),
    }


def paired_test(differences: Iterable[float]) -> dict:
    x = np.asarray(list(differences), dtype=float)
    if x.size < 2:
        raise ValueError("Need at least two seed-level differences")
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    se = sd / math.sqrt(x.size)
    tcrit = float(stats.t.ppf(0.975, x.size - 1))
    nonzero = x[np.abs(x) > 1e-15]
    wilcoxon = (
        float(stats.wilcoxon(nonzero, alternative="less", method="auto").pvalue)
        if nonzero.size else float("nan")
    )
    return {
        "n_seeds": int(x.size),
        "mean_qpu_minus_classical_gap": mean,
        "sd_difference": sd,
        "se_difference": float(se),
        "difference_95ci_low": float(mean - tcrit * se),
        "difference_95ci_high": float(mean + tcrit * se),
        "one_sided_paired_t_pvalue": float(
            stats.ttest_1samp(x, 0.0, alternative="less").pvalue
        ),
        "one_sided_wilcoxon_pvalue": wilcoxon,
        "qpu_better_on_n_seeds": int(np.sum(x < 0)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Matched offline classical controls for the 15-seed QPU experiment."
    )
    parser.add_argument("--heldout-dir", type=Path, required=True)
    parser.add_argument("--replication-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=list(DEFAULT_BUDGETS))
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--sa-restarts", type=int, default=4)
    parser.add_argument("--random-seed", type=int, default=17012026)
    args = parser.parse_args()
    if args.repeats < 20:
        raise ValueError("Use at least 20 repetitions")
    if args.sa_restarts < 1:
        raise ValueError("--sa-restarts must be positive")
    budgets = sorted(set(int(b) for b in args.budgets))
    if min(budgets) < 2:
        raise ValueError("Budgets must be at least 2")
    args.out.mkdir(parents=True, exist_ok=True)

    held, held_counts = read_cohort(args.heldout_dir, "heldout", "initial_heldout")
    repl, repl_counts = read_cohort(args.replication_dir, "replication", "replication")
    frame = pd.concat([held, repl], ignore_index=True)
    seeds = sorted(frame["seed"].astype(int).unique().tolist())
    if seeds != list(range(2001, 2016)):
        raise ValueError(f"Expected seeds 2001-2015; found {seeds}")

    from step12_common import rebuild_graph

    rows: list[dict] = []
    start = time.perf_counter()

    for seed in seeds:
        graph, _ = rebuild_graph(seed)
        seed_rows = frame[frame["seed"].astype(int) == seed]
        if set(seed_rows["angle_strategy"]) != set(STRATEGIES):
            raise ValueError(f"Seed {seed}: missing QPU strategy")
        exact_energies = seed_rows["exact_energy"].astype(float).unique()
        if len(exact_energies) != 1:
            raise ValueError(f"Seed {seed}: inconsistent exact energies")
        exact_energy = float(exact_energies[0])
        linear, edge_i, edge_j, edge_jij, adjacency = graph_ising_coefficients(graph)

        for budget in budgets:
            # Separate deterministic RNG streams per seed, budget, and method.
            base = np.random.SeedSequence([args.random_seed, seed, budget])
            child = base.spawn(5)

            for strategy_index, strategy in enumerate(STRATEGIES):
                row = seed_rows[seed_rows["angle_strategy"] == strategy].iloc[0]
                source_counts = held_counts if row["counts_source"] == "heldout" else repl_counts
                counts = source_counts[str(int(row["pub_index"]))]
                method = f"qpu_{strategy}"
                qpu_trials = qpu_empirical_trials(
                    rng=np.random.default_rng(child[strategy_index]),
                    counts=counts,
                    graph=graph,
                    exact_energy=exact_energy,
                    repeats=args.repeats,
                    budget=budget,
                )
                rows.append(
                    summarize_trials(
                        seed=seed,
                        budget=budget,
                        method=method,
                        trials=qpu_trials,
                        exact_energy=exact_energy,
                    )
                )

            uniform = uniform_trials(
                rng=np.random.default_rng(child[2]),
                repeats=args.repeats,
                budget=budget,
                n_nodes=graph.n_nodes,
                linear=linear,
                edge_i=edge_i,
                edge_j=edge_j,
                edge_jij=edge_jij,
                truth=graph.true_spins,
                exact_energy=exact_energy,
            )
            rows.append(
                summarize_trials(
                    seed=seed,
                    budget=budget,
                    method="uniform_random",
                    trials=uniform,
                    exact_energy=exact_energy,
                )
            )

            greedy_rng = np.random.default_rng(child[3])
            greedy = [
                greedy_random_restart_trial(
                    rng=greedy_rng,
                    budget=budget,
                    n_nodes=graph.n_nodes,
                    linear=linear,
                    edge_i=edge_i,
                    edge_j=edge_j,
                    edge_jij=edge_jij,
                    adjacency=adjacency,
                    truth=graph.true_spins,
                    exact_energy=exact_energy,
                )
                for _ in range(args.repeats)
            ]
            rows.append(
                summarize_trials(
                    seed=seed,
                    budget=budget,
                    method="greedy_random_restart",
                    trials=greedy,
                    exact_energy=exact_energy,
                )
            )

            sa_rng = np.random.default_rng(child[4])
            annealed = [
                simulated_annealing_trial(
                    rng=sa_rng,
                    budget=budget,
                    n_nodes=graph.n_nodes,
                    linear=linear,
                    edge_i=edge_i,
                    edge_j=edge_j,
                    edge_jij=edge_jij,
                    adjacency=adjacency,
                    truth=graph.true_spins,
                    exact_energy=exact_energy,
                    restarts=args.sa_restarts,
                )
                for _ in range(args.repeats)
            ]
            rows.append(
                summarize_trials(
                    seed=seed,
                    budget=budget,
                    method="simulated_annealing",
                    trials=annealed,
                    exact_energy=exact_energy,
                )
            )

        print(f"Completed seed {seed}", flush=True)

    seed_results = pd.DataFrame(rows)
    seed_results.to_csv(args.out / "control_results_by_seed_budget.csv", index=False)

    aggregate_rows: list[dict] = []
    for (budget, method), group in seed_results.groupby(["budget", "method"], sort=True):
        aggregate_rows.append(
            {
                "budget": int(budget),
                "method": method,
                "n_seeds": int(group.shape[0]),
                "mean_best_gap_across_seeds": float(group["mean_best_gap"].mean()),
                "sd_best_gap_across_seeds": float(group["mean_best_gap"].std(ddof=1)),
                "mean_best_label_accuracy_across_seeds": float(
                    group["mean_best_label_accuracy"].mean()
                ),
                "mean_exact_recovery_probability_across_seeds": float(
                    group["exact_recovery_probability"].mean()
                ),
                "mean_evaluations_used": float(group["mean_evaluations_used"].mean()),
            }
        )
    aggregate = pd.DataFrame(aggregate_rows)
    aggregate.to_csv(args.out / "control_aggregate.csv", index=False)

    paired_rows: list[dict] = []
    qpu_method = "qpu_hardware_calibrated"
    for budget in budgets:
        wide = seed_results[seed_results["budget"] == budget].pivot(
            index="seed", columns="method", values="mean_best_gap"
        )
        for comparator in ("uniform_random", "greedy_random_restart", "simulated_annealing"):
            differences = wide[qpu_method] - wide[comparator]
            summary = paired_test(differences)
            paired_rows.append(
                {
                    "budget": int(budget),
                    "qpu_method": qpu_method,
                    "comparator": comparator,
                    **summary,
                }
            )
    paired = pd.DataFrame(paired_rows)
    paired.to_csv(args.out / "qpu_vs_classical_paired.csv", index=False)

    elapsed = time.perf_counter() - start
    payload = {
        "design": {
            "seeds": [int(s) for s in seeds],
            "n_seeds": int(len(seeds)),
            "budgets": budgets,
            "repeats_per_seed_budget_method": int(args.repeats),
            "sa_restarts": int(args.sa_restarts),
            "random_seed": int(args.random_seed),
            "budget_definition": (
                "One QPU shot or one classical objective/delta query counts as one "
                "algorithmic evaluation. This normalizes candidate-query budget, not "
                "wall-clock time, monetary cost, or physical resource use."
            ),
            "qpu_sampling": "bootstrap draws from observed hardware empirical distributions",
            "greedy": "budgeted random-restart best-improvement one-spin descent",
            "simulated_annealing": "fixed geometric schedule with graph-scaled temperature",
        },
        "elapsed_wall_seconds": float(elapsed),
        "main_output_files": [
            "control_results_by_seed_budget.csv",
            "control_aggregate.csv",
            "qpu_vs_classical_paired.csv",
        ],
    }
    (args.out / "classical_control_aggregate.json").write_text(
        json.dumps(payload, indent=2, default=json_default) + "\n"
    )

    # Separate figures, no subplots or custom styling.
    for metric, ylabel, filename in (
        ("mean_best_gap_across_seeds", "Mean best normalized gap", "best_gap_vs_budget.png"),
        (
            "mean_best_label_accuracy_across_seeds",
            "Mean best-state label accuracy",
            "best_accuracy_vs_budget.png",
        ),
        (
            "mean_exact_recovery_probability_across_seeds",
            "Mean exact-recovery probability",
            "exact_recovery_vs_budget.png",
        ),
    ):
        plt.figure(figsize=(8, 5))
        for method, group in aggregate.groupby("method", sort=False):
            group = group.sort_values("budget")
            plt.plot(group["budget"], group[metric], marker="o", label=method)
        plt.xscale("log")
        plt.xlabel("Objective-query / sampling budget")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(args.out / filename, dpi=180)
        plt.close()

    print(json.dumps(payload, indent=2, default=json_default))


if __name__ == "__main__":
    main()
