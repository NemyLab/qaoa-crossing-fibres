#!/usr/bin/env python3
"""Collect and analyse the nighthawk rotated-60-degree QAOA Runtime job."""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService

from step5_curved_signal_graph import build_curved_signal_graph, exact_milp
from step9_common import (
    get_rcm_order,
    normalized_gap,
    qiskit_bitstring_to_spins,
    reorder_graph,
)

GEOMETRY = "curved_60_rot45"


def load_qpy_list(path: Path):
    with path.open("rb") as handle:
        return list(qpy.load(handle))


def build_graph(seed: int):
    graph, diagnostics = build_curved_signal_graph(
        grid_size=5,
        n_gradients=60,
        snr=10.0,
        b_value=2000.0,
        geometry=GEOMETRY,
        fraction_mode="imbalanced",
        degree_cap=3,
        conflict_weight=1.0,
        anchor_weight=0.35,
        sigma_orientation_deg=18.0,
        sigma_displacement_deg=22.0,
        minimum_continuation_weight=0.01,
        seed=int(seed),
    )
    return reorder_graph(graph, get_rcm_order(graph)), diagnostics


def extract_counts(bit_array) -> dict[str, int]:
    shape = tuple(bit_array.shape)
    if shape == ():
        raw = bit_array.get_counts()
    elif shape == (1,):
        raw = bit_array.get_counts(0)
    else:
        raise ValueError(f"Expected scalar BitArray; got shape {shape}")
    return {str(key): int(value) for key, value in raw.items()}


def locate_bit_array(pub_result):
    data = pub_result.data
    if hasattr(data, "meas"):
        return data.meas
    for name in dir(data):
        if name.startswith("_"):
            continue
        value = getattr(data, name)
        if hasattr(value, "get_counts") and hasattr(value, "shape"):
            return value
    raise AttributeError("Could not locate measured BitArray in PUB result")


def aligned_accuracy(spins: np.ndarray, truth: np.ndarray) -> float:
    spins = np.asarray(spins)
    truth = np.asarray(truth)
    return float(max(np.mean(spins == truth), np.mean(-spins == truth)))


def cvar_energy(energy_counts: list[tuple[float, int]], alpha: float = 0.10) -> float:
    shots = int(sum(count for _, count in energy_counts))
    target = float(alpha * shots)
    used = 0.0
    total = 0.0
    for energy, count in sorted(energy_counts, key=lambda item: item[0]):
        take = min(float(count), target - used)
        if take <= 0:
            break
        total += take * float(energy)
        used += take
    if used <= 0:
        raise ValueError("No shots available for CVaR")
    return float(total / used)


def analyse_counts(
    counts: dict[str, int],
    graph,
    exact_spins: np.ndarray,
    exact_energy: float,
) -> dict[str, Any]:
    shots = int(sum(counts.values()))
    if shots <= 0:
        raise ValueError("Counts are empty")

    weighted_sum = 0.0
    weighted_square_sum = 0.0
    best_energy = np.inf
    best_bitstring = None
    best_truth_accuracy = np.nan
    best_exact_accuracy = np.nan
    exact_hits = 0
    energy_counts: list[tuple[float, int]] = []

    for bitstring, count in counts.items():
        spins = qiskit_bitstring_to_spins(bitstring, graph.n_nodes)
        energy = float(graph.energy(spins))
        weighted_sum += int(count) * energy
        weighted_square_sum += int(count) * energy * energy
        energy_counts.append((energy, int(count)))
        if np.isclose(energy, exact_energy, atol=1e-8):
            exact_hits += int(count)
        if energy < best_energy - 1e-12:
            best_energy = energy
            best_bitstring = bitstring
            best_truth_accuracy = aligned_accuracy(spins, graph.true_spins)
            best_exact_accuracy = aligned_accuracy(spins, exact_spins)

    mean_energy = weighted_sum / shots
    variance = max(weighted_square_sum / shots - mean_energy * mean_energy, 0.0)
    mean_energy_se = float(np.sqrt(variance / shots))
    cvar10 = cvar_energy(energy_counts, alpha=0.10)

    return {
        "shots": shots,
        "n_unique_bitstrings": int(len(counts)),
        "mean_energy": float(mean_energy),
        "mean_energy_se": mean_energy_se,
        "mean_normalized_gap": float(normalized_gap(mean_energy, exact_energy)),
        "mean_normalized_gap_se": float(mean_energy_se / (-exact_energy)),
        "cvar10_energy": float(cvar10),
        "cvar10_normalized_gap": float(normalized_gap(cvar10, exact_energy)),
        "best_bitstring": str(best_bitstring),
        "best_energy": float(best_energy),
        "best_normalized_gap": float(normalized_gap(best_energy, exact_energy)),
        "best_label_accuracy": float(best_truth_accuracy),
        "best_agreement_with_exact": float(best_exact_accuracy),
        "exact_ground_hits": int(exact_hits),
    }


def paired_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in frame.groupby("seed"):
        indexed = group.set_index("angle_strategy")
        expected = {"nighthawk_calibrated", "transferred_center"}
        if set(indexed.index) != expected:
            raise ValueError(f"Seed {seed}: missing angle strategy")
        calibrated = indexed.loc["nighthawk_calibrated"]
        transferred = indexed.loc["transferred_center"]
        energy_difference = float(calibrated.mean_energy - transferred.mean_energy)
        difference_se = float(
            np.sqrt(calibrated.mean_energy_se**2 + transferred.mean_energy_se**2)
        )
        rows.append(
            {
                "seed": int(seed),
                "calibrated_minus_transferred_mean_energy": energy_difference,
                "difference_se": difference_se,
                "difference_z": (
                    float(energy_difference / difference_se)
                    if difference_se > 0
                    else np.nan
                ),
                "calibrated_minus_transferred_mean_gap": float(
                    calibrated.mean_normalized_gap
                    - transferred.mean_normalized_gap
                ),
                "calibrated_minus_transferred_cvar10_gap": float(
                    calibrated.cvar10_normalized_gap
                    - transferred.cvar10_normalized_gap
                ),
                "calibrated_minus_transferred_best_gap": float(
                    calibrated.best_normalized_gap
                    - transferred.best_normalized_gap
                ),
                "calibrated_minus_transferred_best_label_accuracy": float(
                    calibrated.best_label_accuracy
                    - transferred.best_label_accuracy
                ),
                "calibrated_better_mean_energy": bool(energy_difference < 0),
            }
        )
    return pd.DataFrame(rows).sort_values("seed")


def job_metrics_safe(job):
    try:
        return job.metrics()
    except Exception as exc:
        return {"unavailable": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", type=Path, required=True)
    args = parser.parse_args()

    prepared = args.prepared
    submission = json.loads((prepared / "nighthawk_curved60rot45_submission.json").read_text())
    candidates = pd.read_csv(prepared / submission["files"]["candidates"])
    circuits = load_qpy_list(prepared / submission["files"]["qpy"])
    if len(circuits) != len(candidates):
        raise RuntimeError("Circuit/candidate count mismatch")

    service = QiskitRuntimeService()
    job = service.job(str(submission["job_id"]))
    result = job.result()
    if len(result) != len(candidates):
        raise RuntimeError(
            f"Runtime returned {len(result)} PUB results for {len(candidates)} candidates"
        )

    rows: list[dict[str, Any]] = []
    counts_by_pub: dict[str, dict[str, int]] = {}
    graph_cache = {}

    for candidate in candidates.itertuples(index=False):
        pub_index = int(candidate.pub_index)
        counts = extract_counts(locate_bit_array(result[pub_index]))
        counts_by_pub[str(pub_index)] = counts

        seed = int(candidate.seed)
        if seed not in graph_cache:
            graph, diagnostics = build_graph(seed)
            exact_spins, exact_energy, solver_info = exact_milp(graph)
            graph_cache[seed] = (
                graph,
                diagnostics,
                np.asarray(exact_spins, dtype=np.int8),
                float(exact_energy),
                solver_info,
            )
        graph, diagnostics, exact_spins, exact_energy, solver_info = graph_cache[seed]

        if not np.isclose(float(candidate.exact_energy), exact_energy, atol=1e-8):
            raise RuntimeError(
                f"Seed {seed}: candidate exact energy {candidate.exact_energy} "
                f"!= rebuilt value {exact_energy}"
            )

        metrics = analyse_counts(counts, graph, exact_spins, exact_energy)
        row = {column: getattr(candidate, column) for column in candidates.columns}
        row.update(metrics)
        row.update(
            {
                "hardware_minus_ideal_expected_gap": float(
                    metrics["mean_normalized_gap"]
                    - float(candidate.ideal_expected_normalized_gap)
                ),
                "mean_peak_error_deg": float(diagnostics["mean_peak_error_deg"]),
                "correct_continuation_fraction": float(
                    diagnostics["correct_continuation_fraction"]
                ),
                "mip_node_count": int(solver_info.get("mip_node_count", -1)),
            }
        )
        rows.append(row)

    frame = pd.DataFrame(rows).sort_values("pub_index")
    frame.to_csv(prepared / "nighthawk_curved60rot45_hardware_results.csv", index=False)
    with gzip.open(prepared / "nighthawk_curved60rot45_hardware_counts.json.gz", "wt") as handle:
        json.dump(counts_by_pub, handle)

    paired = paired_comparison(frame)
    paired.to_csv(prepared / "nighthawk_curved60rot45_paired_angle_comparison.csv", index=False)

    strategy_summary = (
        frame.groupby("angle_strategy", as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            mean_hardware_expected_gap=("mean_normalized_gap", "mean"),
            sd_hardware_expected_gap=("mean_normalized_gap", "std"),
            mean_hardware_cvar10_gap=("cvar10_normalized_gap", "mean"),
            mean_hardware_best_gap=("best_normalized_gap", "mean"),
            mean_hardware_best_label_accuracy=("best_label_accuracy", "mean"),
            mean_best_agreement_with_exact=("best_agreement_with_exact", "mean"),
            total_exact_ground_hits=("exact_ground_hits", "sum"),
            mean_ideal_expected_gap=("ideal_expected_normalized_gap", "mean"),
        )
        .sort_values("mean_hardware_expected_gap")
    )
    strategy_summary.to_csv(prepared / "nighthawk_curved60rot45_strategy_summary.csv", index=False)

    summary = {
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "job_id": str(submission["job_id"]),
        "backend": submission["backend_live"]["backend"],
        "geometry": GEOMETRY,
        "seeds": sorted(frame.seed.astype(int).unique().tolist()),
        "shots_per_pub": int(submission["shots_per_pub"]),
        "n_pubs": int(len(frame)),
        "dd": submission["dd"],
        "job_metrics": job_metrics_safe(job),
        "strategy_summary": strategy_summary.to_dict(orient="records"),
        "paired_angle_comparison": paired.to_dict(orient="records"),
        "files": {
            "results": "nighthawk_curved60rot45_hardware_results.csv",
            "counts": "nighthawk_curved60rot45_hardware_counts.json.gz",
            "paired": "nighthawk_curved60rot45_paired_angle_comparison.csv",
            "strategy_summary": "nighthawk_curved60rot45_strategy_summary.csv",
        },
    }
    (prepared / "nighthawk_curved60rot45_hardware_summary.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )

    display_columns = [
        "seed",
        "angle_strategy",
        "mean_normalized_gap",
        "cvar10_normalized_gap",
        "best_normalized_gap",
        "best_label_accuracy",
        "best_agreement_with_exact",
        "exact_ground_hits",
    ]
    print("\nNighthawk rotated-60-degree hardware results:")
    print(frame[display_columns].to_string(index=False))
    print("\nStrategy summary:")
    print(strategy_summary.to_string(index=False))
    print("\nPaired Nighthawk-calibrated minus transferred comparison:")
    print(paired.to_string(index=False))


if __name__ == "__main__":
    main()
