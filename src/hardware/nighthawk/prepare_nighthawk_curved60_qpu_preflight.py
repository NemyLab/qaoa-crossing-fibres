#!/usr/bin/env python3
"""Prepare and safety-check a nighthawk curved-60-degree QAOA hardware experiment.

This script does NOT submit a QPU job. It:
  1. builds curved_60 graphs for selected seeds,
  2. RCM-relabels each graph,
  3. computes the exact MILP reference and ideal p=1 expectations,
  4. searches multiple transpiler seeds on the requested IBM backend,
  5. selects a layout using calibration-aware safety criteria,
  6. binds the transferred and Nighthawk-calibrated angles,
  7. writes QPY circuits only if every selected seed passes the safety threshold.

Run from the project directory containing the existing step*.py modules.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveIdentityEquivalent
from qiskit_ibm_runtime import QiskitRuntimeService

from step5_curved_signal_graph import build_curved_signal_graph, exact_milp
from step9_common import get_rcm_order, normalized_gap, reorder_graph
from step9_run_10000 import expected_energy_lightcone_p1
from step12_common import build_parameterized_qaoa_circuit

GEOMETRY = "curved_60"
DEFAULT_SEEDS = tuple(range(2001, 2016))
ANGLE_STRATEGIES = {
    "nighthawk_calibrated": {
        "gamma": 0.7071731744054941,
        "beta": 1.1743489498936892,
    },
    "transferred_center": {
        "gamma": 0.7771731744054942,
        "beta": 1.1743489498936892,
    },
}


def json_default(value: Any):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def backend_name(backend) -> str:
    value = getattr(backend, "name", "unknown")
    return str(value() if callable(value) else value)


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
    graph = reorder_graph(graph, get_rcm_order(graph))
    if graph.n_nodes != 50:
        raise RuntimeError(f"Seed {seed}: expected 50 nodes, found {graph.n_nodes}")
    return graph, diagnostics


def aligned_accuracy(spins: np.ndarray, truth: np.ndarray) -> float:
    spins = np.asarray(spins)
    truth = np.asarray(truth)
    return float(max(np.mean(spins == truth), np.mean(-spins == truth)))


def two_qubit_depth(circuit) -> int:
    try:
        return int(circuit.depth(lambda item: len(item.qubits) == 2))
    except Exception:
        return -1


def active_physical_qubits(circuit) -> list[int]:
    active: set[int] = set()
    for item in circuit.data:
        if item.operation.name in {"barrier", "delay"}:
            continue
        for qubit in item.qubits:
            active.add(int(circuit.find_bit(qubit).index))
    return sorted(active)


def operation_counts(circuit) -> dict[str, int]:
    return {str(k): int(v) for k, v in circuit.count_ops().items()}


def target_instruction_error(backend, operation_name: str, qargs: tuple[int, ...]) -> float | None:
    try:
        mapping = backend.target[operation_name]
        properties = mapping.get(tuple(qargs))
        if properties is None and len(qargs) == 2:
            properties = mapping.get(tuple(reversed(qargs)))
        if properties is None or properties.error is None:
            return None
        value = float(properties.error)
        return value if np.isfinite(value) else None
    except Exception:
        return None


def calibration_metrics(backend, circuit) -> dict[str, Any]:
    twoq_errors: list[float] = []
    twoq_records: list[dict[str, Any]] = []
    readout_errors: list[float] = []

    for item in circuit.data:
        qargs = tuple(int(circuit.find_bit(q).index) for q in item.qubits)
        if len(qargs) == 2:
            error = target_instruction_error(backend, str(item.operation.name), qargs)
            if error is not None:
                twoq_errors.append(error)
                twoq_records.append({
                    "operation": str(item.operation.name),
                    "q0": qargs[0],
                    "q1": qargs[1],
                    "error": error,
                })

    for qubit in active_physical_qubits(circuit):
        error = target_instruction_error(backend, "measure", (qubit,))
        if error is not None:
            readout_errors.append(error)

    all_errors = twoq_errors + readout_errors
    error_proxy = float(sum(-math.log(max(1.0 - e, 1e-12)) for e in all_errors))
    return {
        "n_twoq_errors_available": int(len(twoq_errors)),
        "mean_twoq_error": float(np.mean(twoq_errors)) if twoq_errors else None,
        "max_twoq_error": float(np.max(twoq_errors)) if twoq_errors else None,
        "mean_readout_error": float(np.mean(readout_errors)) if readout_errors else None,
        "max_readout_error": float(np.max(readout_errors)) if readout_errors else None,
        "error_proxy_negative_log_success": error_proxy,
        "estimated_independent_success": float(math.exp(-error_proxy)),
        "worst_twoq_operations": sorted(
            twoq_records, key=lambda row: row["error"], reverse=True
        )[:10],
    }


def bind_standard(parameterized_isa, *, gamma: float, beta: float, backend):
    """Bind an already-routed standard-CZ ISA circuit without fractional folding."""
    parameters = {
        parameter.name: parameter
        for parameter in parameterized_isa.parameters
    }
    if set(parameters) != {"gamma", "beta"}:
        raise ValueError(
            f"Expected gamma and beta; found {sorted(parameters)}"
        )

    bound = parameterized_isa.assign_parameters(
        {
            parameters["gamma"]: float(gamma),
            parameters["beta"]: float(beta),
        },
        inplace=False,
    )
    if bound.parameters:
        raise RuntimeError(
            f"Unbound parameters remain: {sorted(p.name for p in bound.parameters)}"
        )

    backend.check_faulty(bound)
    supported = set(map(str, backend.operation_names))
    operations = {
        str(name): int(count)
        for name, count in bound.count_ops().items()
    }
    unsupported = sorted(
        name
        for name in operations
        if name not in supported and name not in {"barrier"}
    )
    if unsupported:
        raise RuntimeError(
            f"Bound standard circuit contains non-ISA operations: {unsupported}"
        )
    if int(operations.get("rzz", 0)) != 0:
        raise RuntimeError(
            "Standard Nighthawk circuit unexpectedly contains RZZ gates"
        )

    return bound, []


def trial_sort_key(record: dict[str, Any]):
    max_error = record.get("max_twoq_error")
    mean_error = record.get("mean_twoq_error")
    return (
        float("inf") if max_error is None else float(max_error),
        float("inf") if mean_error is None else float(mean_error),
        float(record["error_proxy_negative_log_success"]),
        int(record["twoq_count"]),
        int(record["twoq_depth"]),
        int(record["depth"]),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="ibm_berlin")
    parser.add_argument(
        "--gate-mode", choices=["standard", "fractional"], default="standard"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--optimization-level", type=int, default=3)
    parser.add_argument("--transpiler-seed-base", type=int, default=245_000_000)
    parser.add_argument("--max-twoq-error", type=float, default=0.03)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.trials < 1:
        raise ValueError("--trials must be positive")
    args.out.mkdir(parents=True, exist_ok=True)

    service = QiskitRuntimeService()
    backend = service.backend(
        args.backend,
        use_fractional_gates=(args.gate_mode == "fractional"),
    )
    status = backend.status()
    status_payload = {
        "backend": backend_name(backend),
        "operational": bool(status.operational),
        "pending_jobs": int(status.pending_jobs),
        "status_message": str(status.status_msg),
    }

    base_validation = {
        "experiment": "nighthawk_curved_60_degree_phantom",
        "geometry": GEOMETRY,
        "seeds": [int(s) for s in args.seeds],
        "angle_strategies": ANGLE_STRATEGIES,
        "backend_status": status_payload,
        "gate_mode": args.gate_mode,
        "optimization_level": int(args.optimization_level),
        "transpiler_trials_per_seed": int(args.trials),
        "max_twoq_error_threshold": float(args.max_twoq_error),
        "uses_qpu": False,
        "submission_allowed": False,
    }

    if not bool(status.operational):
        base_validation["failure"] = "Backend is not operational; no circuits were prepared."
        (args.out / "nighthawk_curved60_preflight_validation.json").write_text(
            json.dumps(base_validation, indent=2, default=json_default), encoding="utf-8"
        )
        print(json.dumps(base_validation, indent=2), flush=True)
        return

    operations = set(map(str, backend.operation_names))
    if args.gate_mode == "fractional" and "rzz" not in operations:
        raise RuntimeError(f"Backend {args.backend} fractional target does not expose rzz")

    trial_rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    circuits = []
    seed_pass: dict[str, bool] = {}

    for seed in map(int, args.seeds):
        print(f"Building and transpiling {GEOMETRY}, seed {seed}", flush=True)
        graph, diagnostics = build_graph(seed)
        exact_spins, exact_energy, solver_info = exact_milp(graph)
        exact_spins = np.asarray(exact_spins, dtype=np.int8)
        exact_accuracy = aligned_accuracy(exact_spins, np.asarray(graph.true_spins))
        logical = build_parameterized_qaoa_circuit(graph)

        seed_trials: list[dict[str, Any]] = []
        parameterized_by_trial: dict[int, Any] = {}
        for trial in range(args.trials):
            transpiler_seed = int(args.transpiler_seed_base + 1000 * seed + trial)
            manager = generate_preset_pass_manager(
                optimization_level=args.optimization_level,
                backend=backend,
                seed_transpiler=transpiler_seed,
            )
            isa = manager.run(logical)
            backend.check_faulty(isa)
            ops = operation_counts(isa)
            if args.gate_mode == "fractional" and ops.get("rzz", 0) != len(graph.edges):
                raise RuntimeError(
                    f"Seed {seed}, trial {trial}: expected {len(graph.edges)} RZZ gates; "
                    f"found {ops.get('rzz', 0)}"
                )

            # Bind the Nighthawk-calibrated point for calibration-aware scoring.
            scored, _ = bind_standard(
                isa,
                gamma=ANGLE_STRATEGIES["nighthawk_calibrated"]["gamma"],
                beta=ANGLE_STRATEGIES["nighthawk_calibrated"]["beta"],
                backend=backend,
            )
            scored_ops = operation_counts(scored)
            cal = calibration_metrics(backend, scored)
            record = {
                "geometry": GEOMETRY,
                "seed": seed,
                "trial": trial,
                "transpiler_seed": transpiler_seed,
                "n_nodes": int(graph.n_nodes),
                "n_edges": int(len(graph.edges)),
                "depth": int(scored.depth()),
                "twoq_depth": two_qubit_depth(scored),
                "twoq_count": int(sum(v for k, v in scored_ops.items() if k in {"rzz", "cz", "ecr", "cx"})),
                "rzz_count": int(scored_ops.get("rzz", 0)),
                "cz_count": int(scored_ops.get("cz", 0)),
                "active_qubits": active_physical_qubits(scored),
                **cal,
            }
            seed_trials.append(record)
            trial_rows.append(record)
            parameterized_by_trial[trial] = isa

        best = min(seed_trials, key=trial_sort_key)
        best_trial = int(best["trial"])
        best_parameterized = parameterized_by_trial[best_trial]
        max_error = best.get("max_twoq_error")
        passed = bool(max_error is not None and float(max_error) <= args.max_twoq_error)
        seed_pass[str(seed)] = passed
        best_rows.append({**best, "passes_max_twoq_error": passed})

        if not passed:
            print(
                f"SAFETY FAILURE seed {seed}: max_twoq_error={max_error} "
                f"> {args.max_twoq_error}",
                flush=True,
            )
            continue

        for strategy, angles in ANGLE_STRATEGIES.items():
            folded, rzz_angles = bind_standard(
                best_parameterized,
                gamma=float(angles["gamma"]),
                beta=float(angles["beta"]),
                backend=backend,
            )
            ops = operation_counts(folded)
            ideal_energy = float(
                expected_energy_lightcone_p1(
                    graph, float(angles["gamma"]), float(angles["beta"])
                )
            )
            candidate_rows.append({
                "pub_index": len(circuits),
                "geometry": GEOMETRY,
                "seed": seed,
                "angle_strategy": strategy,
                "gamma": float(angles["gamma"]),
                "beta": float(angles["beta"]),
                "exact_energy": float(exact_energy),
                "exact_solution_label_accuracy": float(exact_accuracy),
                "ideal_expected_energy": ideal_energy,
                "ideal_expected_normalized_gap": float(normalized_gap(ideal_energy, exact_energy)),
                "mean_peak_error_deg": float(diagnostics["mean_peak_error_deg"]),
                "correct_continuation_fraction": float(diagnostics["correct_continuation_fraction"]),
                "transpiler_seed": int(best["transpiler_seed"]),
                "depth": int(folded.depth()),
                "twoq_depth": two_qubit_depth(folded),
                "twoq_count": int(sum(v for k, v in ops.items() if k in {"rzz", "cz", "ecr", "cx"})),
                "rzz_count": int(ops.get("rzz", 0)),
                "cz_count": int(ops.get("cz", 0)),
                "minimum_rzz_angle": float(min(rzz_angles)) if rzz_angles else None,
                "maximum_rzz_angle": float(max(rzz_angles)) if rzz_angles else None,
                "max_twoq_error": max_error,
                "mean_twoq_error": best.get("mean_twoq_error"),
                "estimated_independent_success": best.get("estimated_independent_success"),
            })
            circuits.append(folded)

    trials_frame = pd.DataFrame(trial_rows)
    best_frame = pd.DataFrame(best_rows)
    candidates = pd.DataFrame(candidate_rows)
    trials_frame.to_csv(args.out / "nighthawk_curved60_transpile_trials.csv", index=False)
    best_frame.to_csv(args.out / "nighthawk_curved60_best_layouts.csv", index=False)
    candidates.to_csv(args.out / "nighthawk_curved60_candidates.csv", index=False)

    all_pass = bool(seed_pass and all(seed_pass.values()))
    qpy_name = None
    if all_pass and len(circuits) == len(args.seeds) * len(ANGLE_STRATEGIES):
        qpy_name = "nighthawk_curved60_bound_standard.qpy"
        with (args.out / qpy_name).open("wb") as handle:
            qpy.dump(circuits, handle)

    validation = {
        **base_validation,
        "seed_safety": seed_pass,
        "all_seeds_pass_calibration_safety": all_pass,
        "submission_allowed": all_pass,
        "n_prepared_pubs": int(len(circuits)) if all_pass else 0,
        "files": {
            "transpile_trials": "nighthawk_curved60_transpile_trials.csv",
            "best_layouts": "nighthawk_curved60_best_layouts.csv",
            "candidates": "nighthawk_curved60_candidates.csv",
            "qpy": qpy_name,
        },
        "warning": (
            "Preflight only. Do not submit unless submission_allowed is true and "
            "the validation has been reviewed."
        ),
    }
    (args.out / "nighthawk_curved60_preflight_validation.json").write_text(
        json.dumps(validation, indent=2, default=json_default), encoding="utf-8"
    )

    print("\nBest layout per seed:")
    if not best_frame.empty:
        columns = [
            "seed", "transpiler_seed", "depth", "twoq_depth", "twoq_count",
            "mean_twoq_error", "max_twoq_error", "estimated_independent_success",
            "passes_max_twoq_error",
        ]
        print(best_frame[columns].to_string(index=False))
    print("\nValidation:")
    print(json.dumps(validation, indent=2, default=json_default), flush=True)


if __name__ == "__main__":
    main()
