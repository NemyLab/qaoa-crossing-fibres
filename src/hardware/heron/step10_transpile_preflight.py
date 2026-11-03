#!/usr/bin/env python3
"""Transpilation helper metrics used by the Heron preflight/replication workflow.

Restored from the original step10_transpile_preflight.py (helper functions only;
the full preflight driver is archived with the project materials).
"""
from __future__ import annotations

import math

import numpy as np



def active_physical_qubits(circuit) -> list[int]:
    active: set[int] = set()
    for item in circuit.data:
        name = item.operation.name
        if name in {"barrier", "delay"}:
            continue
        for qubit in item.qubits:
            active.add(int(circuit.find_bit(qubit).index))
    return sorted(active)


def count_two_qubit_operations(circuit) -> tuple[int, dict[str, int]]:
    total = 0
    by_name: dict[str, int] = {}
    for item in circuit.data:
        if len(item.qubits) == 2:
            total += 1
            name = str(item.operation.name)
            by_name[name] = by_name.get(name, 0) + 1
    return total, by_name


def two_qubit_depth(circuit) -> int:
    try:
        return int(circuit.depth(lambda item: len(item.qubits) == 2))
    except Exception:
        return -1


def target_error_metrics(backend, circuit) -> dict[str, float]:
    twoq_errors: list[float] = []
    readout_errors: list[float] = []

    for item in circuit.data:
        name = str(item.operation.name)
        qidx = tuple(int(circuit.find_bit(q).index) for q in item.qubits)
        if len(qidx) == 2:
            try:
                props = backend.target[name].get(qidx)
                error = None if props is None else props.error
                if error is not None and np.isfinite(error):
                    twoq_errors.append(float(error))
            except Exception:
                pass
        elif name == "measure" and len(qidx) == 1:
            try:
                props = backend.target["measure"].get(qidx)
                error = None if props is None else props.error
                if error is not None and np.isfinite(error):
                    readout_errors.append(float(error))
            except Exception:
                pass

    all_errors = [*twoq_errors, *readout_errors]
    log_success = 0.0
    for error in all_errors:
        clipped = min(max(error, 0.0), 1.0 - 1e-15)
        log_success += math.log1p(-clipped)

    return {
        "mean_twoq_error": float(np.mean(twoq_errors)) if twoq_errors else np.nan,
        "max_twoq_error": float(np.max(twoq_errors)) if twoq_errors else np.nan,
        "mean_readout_error": (
            float(np.mean(readout_errors)) if readout_errors else np.nan
        ),
        "error_proxy_negative_log_success": float(-log_success),
        "estimated_independent_success": float(math.exp(log_success)),
        "n_twoq_errors_available": int(len(twoq_errors)),
        "n_readout_errors_available": int(len(readout_errors)),
    }