#!/usr/bin/env python3
"""Submit and collect the single-PUB Nighthawk real-dMRI QAOA experiment.

Run from crossing_fibres_step7.

Commands:
  python run_real_qpu_nighthawk_103414.py preview \
    --prepared real_data_8n_replication/103414_R_88_86_76/nighthawk_preflight

  python run_real_qpu_nighthawk_103414.py submit \
    --prepared real_data_8n_replication/103414_R_88_86_76/nighthawk_preflight

  python run_real_qpu_nighthawk_103414.py collect \
    --prepared real_data_8n_replication/103414_R_88_86_76/nighthawk_preflight

The script uses:
  * ibm_berlin, standard CZ gates
  * one frozen Nighthawk-calibrated circuit
  * 10,000 shots by default
  * XY4 dynamical decoupling
  * no gate twirling
  * no measurement twirling
"""
from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from step1_crossing_graph import CrossingGraph
from step5_curved_signal_graph import exact_milp
from step9_common import normalized_gap, qiskit_bitstring_to_spins


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def json_default(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return repr(value)


def backend_name(backend: Any) -> str:
    value = getattr(backend, "name", "unknown")
    return str(value() if callable(value) else value)


def load_one_qpy(path: Path):
    with path.open("rb") as handle:
        circuits = list(qpy.load(handle))
    if len(circuits) != 1:
        raise RuntimeError(f"Expected one circuit in {path}; found {len(circuits)}")
    return circuits[0]


def instruction_error(
    backend: Any,
    operation_name: str,
    qargs: tuple[int, ...],
) -> float | None:
    try:
        mapping = backend.target[operation_name]
    except Exception:
        return None

    props = None
    try:
        props = mapping.get(qargs)
    except Exception:
        pass
    if props is None and len(qargs) == 2:
        try:
            props = mapping.get(tuple(reversed(qargs)))
        except Exception:
            pass

    value = getattr(props, "error", None) if props is not None else None
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def validate_live_circuit(
    circuit,
    backend,
    *,
    maximum_twoq_error: float,
) -> dict[str, Any]:
    backend.check_faulty(circuit)
    supported = set(map(str, backend.operation_names))

    counts = {
        str(name): int(count)
        for name, count in circuit.count_ops().items()
    }
    unsupported = sorted(
        name
        for name in counts
        if name not in supported and name != "barrier"
    )
    if unsupported:
        raise RuntimeError(f"Circuit contains unsupported operations: {unsupported}")
    if counts.get("rzz", 0) or counts.get("cx", 0) or counts.get("ecr", 0):
        raise RuntimeError(
            "Expected a standard-CZ circuit, but non-CZ two-qubit gates remain"
        )

    errors: list[float] = []
    missing: list[dict[str, Any]] = []
    used_pairs: set[tuple[int, int]] = set()

    for item in circuit.data:
        operation = item.operation
        qargs = tuple(
            int(circuit.find_bit(qubit).index)
            for qubit in item.qubits
        )
        if len(qargs) != 2:
            continue
        error = instruction_error(backend, str(operation.name), qargs)
        if error is None:
            missing.append(
                {"operation": str(operation.name), "qargs": list(qargs)}
            )
        else:
            errors.append(error)
        used_pairs.add(tuple(sorted(qargs)))

    if missing:
        raise RuntimeError(
            f"Missing live error data for {len(missing)} two-qubit operations"
        )
    if not errors:
        raise RuntimeError("No two-qubit operations found")

    live_maximum = float(np.max(errors))
    if live_maximum > maximum_twoq_error:
        raise RuntimeError(
            f"Live maximum two-qubit error {live_maximum:.6f} exceeds "
            f"{maximum_twoq_error:.6f}"
        )

    return {
        "depth": int(circuit.depth()),
        "twoq_depth": int(
            circuit.depth(filter_function=lambda item: len(item.qubits) == 2)
        ),
        "operation_counts": counts,
        "n_twoq_gates": int(len(errors)),
        "n_unique_twoq_couplers": int(len(used_pairs)),
        "mean_twoq_error": float(np.mean(errors)),
        "median_twoq_error": float(np.median(errors)),
        "max_twoq_error": live_maximum,
        "independent_twoq_success_proxy": float(
            np.prod(1.0 - np.asarray(errors, dtype=float))
        ),
    }


def load_saved_graph(graph_dir: Path) -> CrossingGraph:
    nodes_path = graph_dir / "real_graph_nodes_rcm.csv"
    edges_path = graph_dir / "real_graph_edges_rcm.csv"
    arrays_path = graph_dir / "real_graph_solution_arrays.npz"

    for path in (nodes_path, edges_path, arrays_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    nodes = pd.read_csv(nodes_path).sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(edges_path)
    with np.load(arrays_path, allow_pickle=False) as archive:
        fields_pre_rcm = np.asarray(archive["fields_pre_rcm"], dtype=float)

    if "old_node_id" not in nodes.columns:
        raise ValueError("RCM node table does not contain old_node_id")

    fields_rcm = fields_pre_rcm[nodes["old_node_id"].to_numpy(dtype=int)]
    true_spins_rcm = nodes["true_spin"].to_numpy(dtype=np.int8)

    case_name = graph_dir.parent.name
    numeric_tokens = [token for token in case_name.split("_") if token.isdigit()]
    if not numeric_tokens:
        raise RuntimeError(
            f"Could not infer the HCP subject ID from case directory {case_name!r}"
        )
    subject_id = int(numeric_tokens[0])

    graph = CrossingGraph(
        nodes=nodes,
        edges=edges,
        fields=fields_rcm,
        true_spins=true_spins_rcm,
        grid_size=5,
        seed=subject_id,
    )
    if graph.n_nodes != 50:
        raise RuntimeError(f"Expected 50 graph nodes; found {graph.n_nodes}")
    return graph


def extract_counts(pub_result) -> tuple[dict[str, int], str]:
    data = pub_result.data
    preferred = ["meas", "c", "cr", "measure"]
    names = preferred + [
        name for name in dir(data) if not name.startswith("_")
    ]

    seen = set()
    for name in names:
        if name in seen or not hasattr(data, name):
            continue
        seen.add(name)
        candidate = getattr(data, name)
        if not hasattr(candidate, "get_counts"):
            continue

        shape = getattr(candidate, "shape", ())
        if shape == ():
            raw = candidate.get_counts()
        elif shape == (1,):
            raw = candidate.get_counts(0)
        else:
            continue

        counts = {str(bitstring): int(count) for bitstring, count in raw.items()}
        return counts, name

    raise RuntimeError("Could not locate measurement counts in Sampler result")


def cvar_energy(expanded_energies: np.ndarray, alpha: float = 0.10) -> float:
    n_take = max(1, int(math.ceil(alpha * len(expanded_energies))))
    return float(
        np.mean(np.partition(expanded_energies, n_take - 1)[:n_take])
    )


def summarize_counts(
    counts: dict[str, int],
    graph: CrossingGraph,
    exact_spins: np.ndarray,
    exact_energy: float,
) -> dict[str, Any]:
    shots = int(sum(counts.values()))
    energies = np.empty(shots, dtype=float)
    cursor = 0
    best_energy = math.inf
    best_bitstring = None
    best_spins = None
    exact_hits = 0

    for bitstring, count in counts.items():
        compact = bitstring.replace(" ", "")
        if len(compact) != graph.n_nodes:
            raise RuntimeError(
                f"Expected {graph.n_nodes}-bit outcomes; got {len(compact)} "
                f"for {bitstring!r}"
            )

        spins = np.asarray(
            qiskit_bitstring_to_spins(compact, graph.n_nodes),
            dtype=np.int8,
        )
        energy = float(graph.energy(spins))
        energies[cursor:cursor + count] = energy
        cursor += count

        if np.isclose(energy, exact_energy, atol=1e-9):
            exact_hits += int(count)

        if energy < best_energy:
            best_energy = energy
            best_bitstring = compact
            best_spins = spins

    if best_spins is None:
        raise RuntimeError("No measured outcomes")

    mean_energy = float(np.mean(energies))
    mean_energy_se = float(np.std(energies, ddof=0) / np.sqrt(shots))
    cvar10 = cvar_energy(energies, alpha=0.10)
    agreement = float(
        max(
            np.mean(best_spins == exact_spins),
            np.mean(-best_spins == exact_spins),
        )
    )

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
        "best_energy": float(best_energy),
        "best_normalized_gap": float(
            normalized_gap(best_energy, exact_energy)
        ),
        "best_bitstring": best_bitstring,
        "best_agreement_with_exact_up_to_sign": agreement,
        "exact_ground_hits": int(exact_hits),
        "exact_ground_hit_rate": float(exact_hits / shots),
    }


def load_context(prepared: Path) -> tuple[dict[str, Any], Path, Path]:
    preflight_path = prepared / "real_qpu_preflight.json"
    if not preflight_path.is_file():
        raise FileNotFoundError(preflight_path)

    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    qpy_path = prepared / preflight["files"]["qpy"]
    if not qpy_path.is_file():
        raise FileNotFoundError(qpy_path)

    graph_dir = Path(preflight["graph"]["directory"])
    return preflight, qpy_path, graph_dir


def preview_or_submit(args, *, submit: bool) -> None:
    prepared = args.prepared
    preflight, qpy_path, _ = load_context(prepared)

    if preflight["backend"]["name"] != "ibm_berlin":
        raise RuntimeError("Preflight backend is not ibm_berlin")
    if bool(preflight["backend"].get("use_fractional_gates", False)):
        raise RuntimeError("Preflight unexpectedly used fractional gates")
    if preflight["circuit"]["angle_strategy"] != "nighthawk_calibrated":
        raise RuntimeError("Preflight does not use nighthawk_calibrated angles")
    if not bool(
        preflight["safety"]["winning_max_twoq_error_within_threshold"]
    ):
        raise RuntimeError("Preflight maximum-error safety check failed")
    if bool(preflight["safety"]["twoq_depth_warning"]):
        raise RuntimeError("Preflight two-qubit depth warning is active")

    circuit = load_one_qpy(qpy_path)
    service = QiskitRuntimeService()
    backend = service.backend("ibm_berlin", use_fractional_gates=False)
    backend.refresh()
    status = backend.status()
    if not bool(status.operational):
        raise RuntimeError(f"ibm_berlin is not operational: {status}")

    live_metrics = validate_live_circuit(
        circuit,
        backend,
        maximum_twoq_error=args.max_twoq_error,
    )

    payload = {
        "validated_at_utc": utc_now(),
        "backend": "ibm_berlin",
        "backend_live": {
            "operational": bool(status.operational),
            "pending_jobs": int(status.pending_jobs),
            "status_message": str(status.status_msg),
        },
        "graph_directory": str(preflight["graph"]["directory"]),
        "angle_strategy": "nighthawk_calibrated",
        "gamma": float(preflight["circuit"]["gamma"]),
        "beta": float(preflight["circuit"]["beta"]),
        "gate_mode": "standard",
        "qpy": qpy_path.name,
        "n_pubs": 1,
        "shots_per_pub": int(args.shots),
        "total_requested_shots": int(args.shots),
        "dd": "XY4",
        "gate_twirling": False,
        "measurement_twirling": False,
        "live_circuit_metrics": live_metrics,
    }

    preview_path = prepared / "real_nighthawk_submission_preview.json"
    preview_path.write_text(
        json.dumps(payload, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    if not submit:
        print(json.dumps(payload, indent=2, default=json_default))
        print("\nPreview only. No QPU job was submitted.")
        return

    submission_path = prepared / "real_nighthawk_submission.json"
    if submission_path.exists():
        raise RuntimeError(
            f"{submission_path} already exists; refusing duplicate submission"
        )

    sampler = Sampler(mode=backend)
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = "XY4"
    sampler.options.environment.job_tags = [
        "DMRI_CROSSING_QAOA",
        "real_hcp",
        "ibm_berlin",
        "standard_cz",
        "nighthawk_calibrated",
        "untwirled",
        "dd_XY4",
    ]

    job = sampler.run([(circuit,)], shots=args.shots)
    submission = {
        **payload,
        "submitted_at_utc": utc_now(),
        "job_id": job.job_id(),
    }
    submission_path.write_text(
        json.dumps(submission, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(submission, indent=2, default=json_default))
    print(f"\nSubmitted Runtime job: {job.job_id()}")


def collect(args) -> None:
    prepared = args.prepared
    preflight, _, graph_dir = load_context(prepared)

    submission_path = prepared / "real_nighthawk_submission.json"
    if not submission_path.is_file():
        raise FileNotFoundError(submission_path)
    submission = json.loads(submission_path.read_text(encoding="utf-8"))

    service = QiskitRuntimeService()
    job = service.job(submission["job_id"])
    print(f"Job {job.job_id()} status: {job.status()}", flush=True)
    result = job.result()
    if len(result) != 1:
        raise RuntimeError(f"Expected one PUB result; found {len(result)}")

    counts, register_name = extract_counts(result[0])
    if sum(counts.values()) != int(submission["shots_per_pub"]):
        raise RuntimeError(
            f"Expected {submission['shots_per_pub']} shots; "
            f"found {sum(counts.values())}"
        )

    graph = load_saved_graph(graph_dir)
    exact_spins, exact_energy, solver_info = exact_milp(graph)
    exact_spins = np.asarray(exact_spins, dtype=np.int8)

    metrics = summarize_counts(
        counts,
        graph,
        exact_spins,
        float(exact_energy),
    )

    row = {
        "backend": "ibm_berlin",
        "case": graph_dir.parent.name,
        "angle_strategy": "nighthawk_calibrated",
        "gamma": float(preflight["circuit"]["gamma"]),
        "beta": float(preflight["circuit"]["beta"]),
        "register_name": register_name,
        "exact_energy": float(exact_energy),
        **metrics,
    }
    pd.DataFrame([row]).to_csv(
        prepared / "real_nighthawk_hardware_results.csv",
        index=False,
    )

    with gzip.open(
        prepared / "real_nighthawk_hardware_counts.json.gz",
        "wt",
        encoding="utf-8",
    ) as handle:
        json.dump(counts, handle)

    summary = {
        "collected_at_utc": utc_now(),
        "job_id": submission["job_id"],
        "backend": "ibm_berlin",
        "case": graph_dir.parent.name,
        "graph_directory": str(graph_dir),
        "angle_strategy": "nighthawk_calibrated",
        "gamma": float(preflight["circuit"]["gamma"]),
        "beta": float(preflight["circuit"]["beta"]),
        "shots": int(submission["shots_per_pub"]),
        "dd": "XY4",
        "gate_twirling": False,
        "measurement_twirling": False,
        "result": row,
        "exact_solver_info": solver_info,
        "job_metrics": job.metrics(),
        "files": {
            "results": "real_nighthawk_hardware_results.csv",
            "counts": "real_nighthawk_hardware_counts.json.gz",
        },
    }
    summary_path = prepared / "real_nighthawk_hardware_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, default=json_default) + "\n",
        encoding="utf-8",
    )

    print("\nREAL Nighthawk QPU RESULT")
    print("=" * 78)
    print(f"Mean normalized gap: {metrics['mean_normalized_gap']:.6f}")
    print(f"CVaR-10% normalized gap: {metrics['cvar10_normalized_gap']:.6f}")
    print(f"Best normalized gap: {metrics['best_normalized_gap']:.6f}")
    print(
        "Best agreement with exact up to sign: "
        f"{metrics['best_agreement_with_exact_up_to_sign']:.3f}"
    )
    print(f"Exact ground-state hits: {metrics['exact_ground_hits']}")
    print(f"Wrote: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("preview", "submit", "collect"))
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--shots", type=int, default=10_000)
    parser.add_argument("--max-twoq-error", type=float, default=0.03)
    args = parser.parse_args()

    if args.shots < 1:
        raise ValueError("--shots must be positive")
    if args.max_twoq_error <= 0:
        raise ValueError("--max-twoq-error must be positive")

    if args.command == "preview":
        preview_or_submit(args, submit=False)
    elif args.command == "submit":
        preview_or_submit(args, submit=True)
    else:
        collect(args)


if __name__ == "__main__":
    main()
