#!/usr/bin/env python3
"""Retrieve and analyse the final 30-PUB real-data Heron SamplerV2 job.

No QPU job is submitted.

For each patch, the script:
  * retrieves the measured bitstring counts;
  * reconstructs the saved 50-node graph;
  * solves the exact MILP reference;
  * calculates mean, CVaR-10%, CVaR-1%, and best-sample energies;
  * reports normalized gaps and exact-solution agreement;
  * reports confidence-restricted TOM agreement;
  * compares hardware outcomes with the 10,000-shot ideal MPS baseline.

The output is one aggregate row per patch plus compressed raw counts.
"""
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit_ibm_runtime import QiskitRuntimeService

from step1_crossing_graph import CrossingGraph
from step5_curved_signal_graph import exact_milp
from step9_common import normalized_gap, qiskit_bitstring_to_spins


EPS = 1e-12
TOM_CONFIDENCE_ANGLE_DEG = 25.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prepared",
        type=Path,
        default=Path("real_heron_submission"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("real_heron_hardware"),
    )
    return parser.parse_args()


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return repr(value)


def clean_bitstring(bitstring: str) -> str:
    return "".join(str(bitstring).split())


def available_data_fields(data_bin: Any) -> list[str]:
    fields: list[str] = []
    try:
        fields.extend(str(key) for key in data_bin.keys())
    except Exception:
        pass

    for name in dir(data_bin):
        if name.startswith("_"):
            continue
        try:
            value = getattr(data_bin, name)
        except Exception:
            continue
        if hasattr(value, "get_counts") and name not in fields:
            fields.append(name)
    return fields


def extract_counts(
    pub_result: Any,
    expected_bits: int = 50,
) -> tuple[dict[str, int], str]:
    data = pub_result.data
    candidates: list[tuple[str, Any]] = []

    for name in available_data_fields(data):
        try:
            value = getattr(data, name)
        except Exception:
            try:
                value = data[name]
            except Exception:
                continue
        if hasattr(value, "get_counts"):
            candidates.append((str(name), value))

    if not candidates:
        raise RuntimeError(
            "No SamplerV2 BitArray was found. "
            f"Available fields: {available_data_fields(data)}"
        )

    descriptions = []
    for name, bit_array in candidates:
        raw_counts = bit_array.get_counts()
        counts = {
            clean_bitstring(bitstring): int(count)
            for bitstring, count in raw_counts.items()
        }
        lengths = sorted({len(bitstring) for bitstring in counts})
        descriptions.append({"field": name, "bit_lengths": lengths})
        if lengths == [expected_bits]:
            return counts, name

    raise RuntimeError(
        f"No BitArray contained {expected_bits}-bit outcomes. "
        f"Candidates: {descriptions}"
    )


def load_graph(
    graph_dir: Path,
) -> tuple[CrossingGraph, pd.DataFrame, dict[str, Any]]:
    nodes_path = graph_dir / "real_graph_nodes_rcm.csv"
    edges_path = graph_dir / "real_graph_edges_rcm.csv"
    arrays_path = graph_dir / "real_graph_solution_arrays.npz"
    validation_path = graph_dir / "real_graph_validation.json"

    for path in (nodes_path, edges_path, arrays_path, validation_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    nodes = pd.read_csv(nodes_path).sort_values(
        "node_id"
    ).reset_index(drop=True)
    edges = pd.read_csv(edges_path)
    validation = json.loads(
        validation_path.read_text(encoding="utf-8")
    )

    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {
            name: np.asarray(archive[name])
            for name in archive.files
        }

    if "old_node_id" not in nodes.columns:
        raise ValueError("RCM node table is missing old_node_id")

    fields_pre_rcm = np.asarray(
        arrays["fields_pre_rcm"],
        dtype=float,
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
        raise ValueError(
            f"Expected 50 nodes; found {graph.n_nodes}"
        )
    return graph, nodes, validation


def aligned_accuracy(
    spins: np.ndarray,
    reference: np.ndarray,
) -> float:
    spins = np.asarray(spins, dtype=int)
    reference = np.asarray(reference, dtype=int)
    return float(
        max(
            np.mean(spins == reference),
            np.mean(-spins == reference),
        )
    )


def align_with_mask(
    spins: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, int]:
    spins = np.asarray(spins, dtype=np.int8)
    reference = np.asarray(reference, dtype=np.int8)
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        raise ValueError("No confident TOM nodes available")
    direct = int(np.sum(spins[mask] == reference[mask]))
    flipped = int(np.sum(-spins[mask] == reference[mask]))
    sign = +1 if direct >= flipped else -1
    return (sign * spins).astype(np.int8), sign


def cvar_energy(
    pairs: list[tuple[float, int]],
    alpha: float,
) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must lie in (0,1]")
    total = int(sum(count for _, count in pairs))
    target = float(alpha * total)
    accepted = 0.0
    weighted = 0.0

    for energy, count in sorted(pairs, key=lambda item: item[0]):
        take = min(float(count), target - accepted)
        if take <= 0:
            break
        weighted += take * float(energy)
        accepted += take

    if accepted <= 0:
        raise RuntimeError("CVaR received no samples")
    return float(weighted / accepted)


def assignment_metrics(
    graph: CrossingGraph,
    nodes: pd.DataFrame,
    spins_rcm: np.ndarray,
) -> dict[str, Any]:
    angles = nodes["tom_match_angle_deg"].to_numpy(dtype=float)
    confident_nodes = (
        np.isfinite(angles)
        & (angles <= TOM_CONFIDENCE_ANGLE_DEG)
    )

    aligned_tom, sign = align_with_mask(
        spins_rcm,
        graph.true_spins,
        confident_nodes,
    )

    conflict = graph.edges[
        graph.edges["edge_type"].eq("conflict")
    ]
    continuation = graph.edges[
        graph.edges["edge_type"].eq("continuation")
    ]

    conflict_satisfied = np.asarray(
        [
            aligned_tom[int(edge.i)]
            * aligned_tom[int(edge.j)]
            == -1
            for edge in conflict.itertuples(index=False)
        ],
        dtype=float,
    )
    continuation_satisfied = np.asarray(
        [
            aligned_tom[int(edge.i)]
            * aligned_tom[int(edge.j)]
            == +1
            for edge in continuation.itertuples(index=False)
        ],
        dtype=float,
    )
    continuation_weights = continuation[
        "weight"
    ].to_numpy(dtype=float)

    spins_pre_rcm = np.empty(50, dtype=np.int8)
    truth_pre_rcm = np.empty(50, dtype=np.int8)
    confident_pre_rcm = np.zeros(50, dtype=bool)

    for row_index, row in enumerate(
        graph.nodes.itertuples(index=False)
    ):
        new_node = int(row.node_id)
        old_node = int(row.old_node_id)
        spins_pre_rcm[old_node] = aligned_tom[new_node]
        truth_pre_rcm[old_node] = int(row.true_spin)
        confident_pre_rcm[old_node] = bool(
            confident_nodes[row_index]
        )

    all_voxel_correct: list[int] = []
    confident_voxel_correct: list[int] = []
    selected_peak_by_voxel: list[int] = []

    for voxel in range(25):
        node0 = 2 * voxel
        node1 = node0 + 1
        positive = [
            peak
            for peak, node in enumerate((node0, node1))
            if spins_pre_rcm[node] == +1
        ]
        selected_peak = positive[0] if len(positive) == 1 else -1
        tom_peak = 0 if truth_pre_rcm[node0] == +1 else 1
        correct = int(selected_peak == tom_peak)

        selected_peak_by_voxel.append(int(selected_peak))
        all_voxel_correct.append(correct)

        if (
            confident_pre_rcm[node0]
            and confident_pre_rcm[node1]
        ):
            confident_voxel_correct.append(correct)

    weighted_continuation = (
        float(
            np.dot(
                continuation_weights,
                continuation_satisfied,
            )
            / np.sum(continuation_weights)
        )
        if len(continuation_weights)
        and float(np.sum(continuation_weights)) > 0
        else float("nan")
    )

    return {
        "tom_alignment_sign": int(sign),
        "n_tom_confident_voxels": int(
            len(confident_voxel_correct)
        ),
        "node_agreement_with_tom_all": float(
            np.mean(aligned_tom == graph.true_spins)
        ),
        "node_agreement_with_tom_confident": float(
            np.mean(
                aligned_tom[confident_nodes]
                == graph.true_spins[confident_nodes]
            )
        ),
        "voxel_cst_peak_agreement_with_tom_all": float(
            np.mean(all_voxel_correct)
        ),
        "voxel_cst_peak_agreement_with_tom_confident": float(
            np.mean(confident_voxel_correct)
        ),
        "conflict_satisfaction_fraction": float(
            np.mean(conflict_satisfied)
        ),
        "continuation_satisfaction_fraction": float(
            np.mean(continuation_satisfied)
        ),
        "weighted_continuation_satisfaction_fraction": (
            weighted_continuation
        ),
        "selected_peak_by_voxel": selected_peak_by_voxel,
        "spins_rcm_aligned_to_tom": aligned_tom,
        "spins_pre_rcm_aligned_to_tom": spins_pre_rcm,
    }


def analyse_counts(
    *,
    counts: dict[str, int],
    graph: CrossingGraph,
    nodes: pd.DataFrame,
    exact_spins: np.ndarray,
    exact_energy: float,
) -> dict[str, Any]:
    shots = int(sum(counts.values()))
    weighted_energy = 0.0
    weighted_square = 0.0
    pairs: list[tuple[float, int]] = []
    exact_hits = 0

    best_energy = float("inf")
    best_count = -1
    best_bitstring = ""
    best_spins: np.ndarray | None = None

    for bitstring, count in counts.items():
        spins = np.asarray(
            qiskit_bitstring_to_spins(
                bitstring,
                graph.n_nodes,
            ),
            dtype=np.int8,
        )
        energy = float(graph.energy(spins))
        count = int(count)

        weighted_energy += count * energy
        weighted_square += count * energy * energy
        pairs.append((energy, count))

        if np.isclose(energy, exact_energy, atol=1e-8):
            exact_hits += count

        if (
            energy < best_energy - EPS
            or (
                np.isclose(energy, best_energy, atol=EPS)
                and count > best_count
            )
        ):
            best_energy = energy
            best_count = count
            best_bitstring = bitstring
            best_spins = spins.copy()

    if best_spins is None:
        raise RuntimeError("No best sample was found")

    mean_energy = float(weighted_energy / shots)
    variance = max(
        weighted_square / shots - mean_energy * mean_energy,
        0.0,
    )
    mean_energy_se = float(np.sqrt(variance / shots))
    cvar10 = cvar_energy(pairs, 0.10)
    cvar01 = cvar_energy(pairs, 0.01)
    assignment = assignment_metrics(graph, nodes, best_spins)

    return {
        "shots": shots,
        "n_unique_bitstrings": int(len(counts)),
        "mean_energy": mean_energy,
        "mean_energy_se": mean_energy_se,
        "mean_normalized_gap": float(
            normalized_gap(mean_energy, exact_energy)
        ),
        "cvar10_energy": cvar10,
        "cvar10_normalized_gap": float(
            normalized_gap(cvar10, exact_energy)
        ),
        "cvar01_energy": cvar01,
        "cvar01_normalized_gap": float(
            normalized_gap(cvar01, exact_energy)
        ),
        "best_bitstring": best_bitstring,
        "best_bitstring_count": int(best_count),
        "best_energy": float(best_energy),
        "best_normalized_gap": float(
            normalized_gap(best_energy, exact_energy)
        ),
        "best_agreement_with_exact_up_to_sign": aligned_accuracy(
            best_spins,
            exact_spins,
        ),
        "exact_ground_hits": int(exact_hits),
        "exact_ground_hit_fraction": float(exact_hits / shots),
        "assignment": assignment,
        "best_spins_rcm": best_spins,
    }


def safe_job_metrics(job: Any) -> Any:
    try:
        return job.metrics()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def safe_job_usage(job: Any) -> Any:
    try:
        return job.usage()
    except Exception as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}


def main() -> None:
    args = parse_args()
    args.prepared = args.prepared.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    args.out = args.out.resolve()

    submission_path = (
        args.prepared / "real_heron_submission.json"
    )
    candidates_path = (
        args.prepared / "real_heron_submission_candidates.csv"
    )
    for path in (submission_path, candidates_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    submission = json.loads(
        submission_path.read_text(encoding="utf-8")
    )
    candidates = pd.read_csv(candidates_path).sort_values(
        "pub_index"
    ).reset_index(drop=True)

    expected_order = submission["pub_order"]
    observed_order = candidates["patch_id"].astype(str).tolist()
    if observed_order != expected_order:
        raise RuntimeError(
            "Candidate order differs from the submitted PUB order"
        )

    service = QiskitRuntimeService()
    job_id = str(submission["job_id"])
    job = service.job(job_id)
    status = str(job.status())
    if "DONE" not in status.upper():
        raise RuntimeError(
            f"Job {job_id} is not complete; status is {status}"
        )

    result = job.result()
    if len(result) != len(candidates):
        raise RuntimeError(
            f"Expected {len(candidates)} PUB results; "
            f"found {len(result)}"
        )

    expected_shots = int(
        submission["protocol"]["shots_per_pub"]
    )
    all_counts: dict[str, dict[str, int]] = {}
    rows: list[dict[str, Any]] = []
    assignment_arrays: dict[str, np.ndarray] = {}

    for pub_index, candidate in candidates.iterrows():
        patch_id = str(candidate["patch_id"])
        graph_dir = Path(str(candidate["graph_dir"]))
        mps_path = Path(str(candidate["mps_results_csv"]))

        graph, nodes, graph_validation = load_graph(graph_dir)
        exact_spins, exact_energy, exact_solver_info = exact_milp(
            graph
        )
        exact_spins = np.asarray(exact_spins, dtype=np.int8)

        mps = pd.read_csv(mps_path)
        mps = mps.loc[
            mps["angle_strategy"].astype(str)
            == "hardware_calibrated"
        ]
        if len(mps) != 1:
            raise RuntimeError(
                f"{patch_id}: expected one hardware-calibrated MPS row"
            )
        mps_row = mps.iloc[0]

        counts, register_name = extract_counts(
            result[int(pub_index)]
        )
        if sum(counts.values()) != expected_shots:
            raise RuntimeError(
                f"{patch_id}: retrieved {sum(counts.values())} shots; "
                f"expected {expected_shots}"
            )

        metrics = analyse_counts(
            counts=counts,
            graph=graph,
            nodes=nodes,
            exact_spins=exact_spins,
            exact_energy=float(exact_energy),
        )
        assignment = metrics["assignment"]

        all_counts[patch_id] = counts
        assignment_arrays[
            f"{patch_id}__exact_spins_rcm"
        ] = exact_spins
        assignment_arrays[
            f"{patch_id}__best_spins_rcm"
        ] = np.asarray(
            metrics["best_spins_rcm"],
            dtype=np.int8,
        )
        assignment_arrays[
            f"{patch_id}__best_spins_rcm_aligned_to_tom"
        ] = np.asarray(
            assignment["spins_rcm_aligned_to_tom"],
            dtype=np.int8,
        )

        mean_gap = float(metrics["mean_normalized_gap"])
        mps_mean_gap = float(mps_row["mean_normalized_gap"])

        rows.append(
            {
                "job_id": job_id,
                "pub_index": int(pub_index),
                "patch_id": patch_id,
                "register_name": register_name,
                "gamma": float(candidate["gamma"]),
                "beta": float(candidate["beta"]),
                "shots": int(metrics["shots"]),
                "n_unique_bitstrings": int(
                    metrics["n_unique_bitstrings"]
                ),
                "exact_energy": float(exact_energy),
                "mean_energy": float(metrics["mean_energy"]),
                "mean_energy_se": float(
                    metrics["mean_energy_se"]
                ),
                "mean_normalized_gap": mean_gap,
                "cvar10_energy": float(metrics["cvar10_energy"]),
                "cvar10_normalized_gap": float(
                    metrics["cvar10_normalized_gap"]
                ),
                "cvar01_energy": float(metrics["cvar01_energy"]),
                "cvar01_normalized_gap": float(
                    metrics["cvar01_normalized_gap"]
                ),
                "best_energy": float(metrics["best_energy"]),
                "best_normalized_gap": float(
                    metrics["best_normalized_gap"]
                ),
                "best_agreement_with_exact_up_to_sign": float(
                    metrics[
                        "best_agreement_with_exact_up_to_sign"
                    ]
                ),
                "best_tom_agreement_all_voxels": float(
                    assignment[
                        "voxel_cst_peak_agreement_with_tom_all"
                    ]
                ),
                "best_tom_agreement_confident_voxels": float(
                    assignment[
                        "voxel_cst_peak_agreement_with_tom_confident"
                    ]
                ),
                "n_tom_confident_voxels": int(
                    assignment["n_tom_confident_voxels"]
                ),
                "best_conflict_satisfaction_fraction": float(
                    assignment[
                        "conflict_satisfaction_fraction"
                    ]
                ),
                "best_weighted_continuation_satisfaction_fraction": (
                    float(
                        assignment[
                            "weighted_continuation_satisfaction_fraction"
                        ]
                    )
                ),
                "exact_ground_hits": int(
                    metrics["exact_ground_hits"]
                ),
                "exact_ground_hit_fraction": float(
                    metrics["exact_ground_hit_fraction"]
                ),
                "mps_mean_normalized_gap": mps_mean_gap,
                "mps_cvar10_normalized_gap": float(
                    mps_row["cvar10_normalized_gap"]
                ),
                "mps_best_normalized_gap": float(
                    mps_row["best_normalized_gap"]
                ),
                "hardware_minus_mps_mean_gap": float(
                    mean_gap - mps_mean_gap
                ),
                "hardware_minus_mps_cvar10_gap": float(
                    metrics["cvar10_normalized_gap"]
                    - float(mps_row["cvar10_normalized_gap"])
                ),
                "hardware_minus_mps_best_gap": float(
                    metrics["best_normalized_gap"]
                    - float(mps_row["best_normalized_gap"])
                ),
                "transpiler_seed": int(
                    candidate["transpiler_seed"]
                ),
                "twoq_depth": int(candidate["twoq_depth"]),
                "n_twoq_gates": int(
                    candidate["n_twoq_gates"]
                ),
                "n_active_physical_qubits": int(
                    candidate["n_active_physical_qubits"]
                ),
                "mean_twoq_error": float(
                    candidate["mean_twoq_error"]
                ),
                "max_twoq_error": float(
                    candidate["max_twoq_error"]
                ),
                "twoq_calibration_error_cost": float(
                    candidate["twoq_calibration_error_cost"]
                ),
                "independent_twoq_success_proxy": float(
                    candidate["independent_twoq_success_proxy"]
                ),
            }
        )

        print(
            f"{patch_id}: mean gap={mean_gap:.4f}, "
            f"MPS={mps_mean_gap:.4f}, "
            f"penalty={mean_gap - mps_mean_gap:+.4f}, "
            f"best gap={metrics['best_normalized_gap']:.4f}",
            flush=True,
        )

    frame = pd.DataFrame(rows).sort_values(
        "pub_index"
    ).reset_index(drop=True)

    results_csv = args.out / "real_heron_hardware_results.csv"
    frame.to_csv(results_csv, index=False)

    counts_path = (
        args.out / "real_heron_hardware_counts.json.gz"
    )
    with gzip.open(counts_path, "wt", encoding="utf-8") as handle:
        json.dump(all_counts, handle)

    assignments_path = (
        args.out / "real_heron_best_assignments.npz"
    )
    np.savez_compressed(assignments_path, **assignment_arrays)

    numeric = [
        "mean_normalized_gap",
        "cvar10_normalized_gap",
        "cvar01_normalized_gap",
        "best_normalized_gap",
        "best_agreement_with_exact_up_to_sign",
        "best_tom_agreement_confident_voxels",
        "best_weighted_continuation_satisfaction_fraction",
        "exact_ground_hit_fraction",
        "mps_mean_normalized_gap",
        "hardware_minus_mps_mean_gap",
        "hardware_minus_mps_cvar10_gap",
        "hardware_minus_mps_best_gap",
    ]
    aggregate_rows = []
    for column in numeric:
        values = pd.to_numeric(
            frame[column],
            errors="coerce",
        ).dropna()
        aggregate_rows.append(
            {
                "metric": column,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)),
                "median": float(values.median()),
                "minimum": float(values.min()),
                "maximum": float(values.max()),
            }
        )

    aggregate_csv = (
        args.out / "real_heron_hardware_aggregate.csv"
    )
    pd.DataFrame(aggregate_rows).to_csv(
        aggregate_csv,
        index=False,
    )

    summary = {
        "job_id": job_id,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "backend": submission["backend_live"]["backend"],
        "protocol": submission["protocol"],
        "n_pubs": int(len(frame)),
        "total_retrieved_shots": int(frame["shots"].sum()),
        "total_exact_ground_hits": int(
            frame["exact_ground_hits"].sum()
        ),
        "mean_hardware_normalized_gap": float(
            frame["mean_normalized_gap"].mean()
        ),
        "mean_mps_normalized_gap": float(
            frame["mps_mean_normalized_gap"].mean()
        ),
        "mean_hardware_minus_mps_gap": float(
            frame["hardware_minus_mps_mean_gap"].mean()
        ),
        "runtime": {
            "job_metrics": safe_job_metrics(job),
            "job_usage": safe_job_usage(job),
            "result_metadata": getattr(result, "metadata", None),
        },
        "files": {
            "results_csv": results_csv.name,
            "aggregate_csv": aggregate_csv.name,
            "counts": counts_path.name,
            "best_assignments": assignments_path.name,
        },
    }

    summary_path = args.out / "real_heron_hardware_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print("\nREAL-DATA HERON HARDWARE SUMMARY")
    print("=" * 88)
    print(json.dumps(summary, indent=2, default=json_default))


if __name__ == "__main__":
    main()
