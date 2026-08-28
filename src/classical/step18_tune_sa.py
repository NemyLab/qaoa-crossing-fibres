#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from step12_common import rebuild_graph
from step17_classical_controls import graph_ising_coefficients, normalized_gap
from step18_common import simulated_annealing_tunable_trial
from step5_curved_signal_graph import exact_milp


def json_default(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune simulated annealing on the training graph (seed 2000) only."
    )
    parser.add_argument("--seed", type=int, default=2000)
    parser.add_argument("--budgets", type=int, nargs="+", default=[100, 256, 500, 1000, 2000, 5000])
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--restarts", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument(
        "--t-start-multipliers", type=float, nargs="+", default=[0.25, 0.5, 1.0, 2.0]
    )
    parser.add_argument(
        "--t-end-multipliers", type=float, nargs="+", default=[0.001, 0.01, 0.05]
    )
    parser.add_argument("--random-seed", type=int, default=18012026)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.seed != 2000:
        raise ValueError("Primary tuning must use training seed 2000 only")
    if args.repeats < 20:
        raise ValueError("Use at least 20 repeats")
    budgets = sorted(set(int(x) for x in args.budgets))
    if min(budgets) < 2:
        raise ValueError("Budgets must be at least 2")
    args.out.mkdir(parents=True, exist_ok=True)

    graph, diagnostics = rebuild_graph(args.seed)
    exact_start = time.perf_counter()
    _, exact_energy, solver_info = exact_milp(graph)
    exact_wall_seconds = time.perf_counter() - exact_start
    linear, edge_i, edge_j, edge_jij, adjacency = graph_ising_coefficients(graph)

    configs = []
    for restarts in sorted(set(args.restarts)):
        for t_start in sorted(set(args.t_start_multipliers)):
            for t_end in sorted(set(args.t_end_multipliers)):
                if restarts >= 1 and 0 < t_end < t_start:
                    configs.append((int(restarts), float(t_start), float(t_end)))
    if not configs:
        raise ValueError("No valid SA configurations")

    rows: list[dict] = []
    overall_start = time.perf_counter()
    for config_index, (restarts, t_start, t_end) in enumerate(configs):
        for budget in budgets:
            rng = np.random.default_rng(
                np.random.SeedSequence([args.random_seed, args.seed, config_index, budget])
            )
            start = time.perf_counter()
            trials = [
                simulated_annealing_tunable_trial(
                    rng=rng,
                    budget=budget,
                    n_nodes=graph.n_nodes,
                    linear=linear,
                    edge_i=edge_i,
                    edge_j=edge_j,
                    edge_jij=edge_jij,
                    adjacency=adjacency,
                    truth=graph.true_spins,
                    exact_energy=exact_energy,
                    restarts=restarts,
                    t_start_multiplier=t_start,
                    t_end_multiplier=t_end,
                )
                for _ in range(args.repeats)
            ]
            elapsed = time.perf_counter() - start
            energies = np.asarray([trial.best_energy for trial in trials], dtype=float)
            gaps = normalized_gap(energies, exact_energy)
            accuracies = np.asarray([trial.best_accuracy for trial in trials], dtype=float)
            rows.append(
                {
                    "config_index": config_index,
                    "restarts": restarts,
                    "t_start_multiplier": t_start,
                    "t_end_multiplier": t_end,
                    "budget": budget,
                    "repeats": args.repeats,
                    "mean_best_gap": float(gaps.mean()),
                    "sd_best_gap": float(gaps.std(ddof=1)),
                    "mean_best_label_accuracy": float(accuracies.mean()),
                    "exact_recovery_probability": float(
                        np.mean([trial.exact_hit for trial in trials])
                    ),
                    "mean_evaluations_used": float(
                        np.mean([trial.evaluations_used for trial in trials])
                    ),
                    "batch_wall_seconds": float(elapsed),
                    "mean_wall_seconds_per_trial": float(elapsed / args.repeats),
                }
            )
        print(f"Completed SA configuration {config_index + 1}/{len(configs)}", flush=True)

    results = pd.DataFrame(rows)
    results.to_csv(args.out / "sa_tuning_results.csv", index=False)

    score_rows = []
    for config_index, group in results.groupby("config_index", sort=True):
        low = group[group["budget"] <= 500]
        score_rows.append(
            {
                "config_index": int(config_index),
                "restarts": int(group["restarts"].iloc[0]),
                "t_start_multiplier": float(group["t_start_multiplier"].iloc[0]),
                "t_end_multiplier": float(group["t_end_multiplier"].iloc[0]),
                "primary_score_mean_gap_across_budgets": float(group["mean_best_gap"].mean()),
                "secondary_low_budget_mean_gap": float(low["mean_best_gap"].mean()),
                "mean_wall_seconds_per_trial": float(group["mean_wall_seconds_per_trial"].mean()),
            }
        )
    scores = pd.DataFrame(score_rows).sort_values(
        [
            "primary_score_mean_gap_across_budgets",
            "secondary_low_budget_mean_gap",
            "mean_wall_seconds_per_trial",
            "config_index",
        ],
        kind="mergesort",
    )
    scores.to_csv(args.out / "sa_tuning_scores.csv", index=False)
    winner = scores.iloc[0]

    selected = {
        "selection_scope": "training graph seed 2000 only",
        "selection_rule": (
            "minimum unweighted mean normalized best gap across the six predefined budgets; "
            "ties by mean gap at budgets <=500, then wall time"
        ),
        "training_seed": int(args.seed),
        "budgets": budgets,
        "repeats_per_budget_configuration": int(args.repeats),
        "n_configurations": int(len(configs)),
        "selected_config_index": int(winner["config_index"]),
        "selected_restarts": int(winner["restarts"]),
        "selected_t_start_multiplier": float(winner["t_start_multiplier"]),
        "selected_t_end_multiplier": float(winner["t_end_multiplier"]),
        "selected_primary_score": float(winner["primary_score_mean_gap_across_budgets"]),
        "selected_low_budget_score": float(winner["secondary_low_budget_mean_gap"]),
        "exact_energy": float(exact_energy),
        "exact_milp_wall_seconds": float(exact_wall_seconds),
        "graph_diagnostics": diagnostics,
        "solver_info": solver_info,
        "elapsed_wall_seconds": float(time.perf_counter() - overall_start),
        "warning": (
            "The selected SA configuration is frozen before evaluation on seeds 2001-2015. "
            "Do not retune using held-out results."
        ),
    }
    (args.out / "selected_sa_configuration.json").write_text(
        json.dumps(selected, indent=2, default=json_default), encoding="utf-8"
    )
    print(json.dumps(selected, indent=2, default=json_default))


if __name__ == "__main__":
    main()
