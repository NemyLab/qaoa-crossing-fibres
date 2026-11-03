from __future__ import annotations

import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def gate_mode_to_fractional(gate_mode: str) -> bool:
    if gate_mode == "fractional":
        return True
    if gate_mode == "standard":
        return False
    raise ValueError(f"Unknown gate mode: {gate_mode}")


def find_best_circuit(
    preflight_dir: Path,
    *,
    backend: str,
    gate_mode: str,
    graph_seed: int,
) -> tuple[Path, dict]:
    table_path = preflight_dir / "best_transpilation_per_seed.csv"
    if not table_path.exists():
        raise FileNotFoundError(table_path)
    table = pd.read_csv(table_path)
    subset = table[
        (table["backend"].astype(str) == backend)
        & (table["gate_mode"].astype(str) == gate_mode)
        & (table["graph_seed"].astype(int) == int(graph_seed))
    ]
    if len(subset) != 1:
        raise ValueError(
            f"Expected exactly one circuit for {backend}/{gate_mode}/seed {graph_seed}; "
            f"found {len(subset)}"
        )
    record = subset.iloc[0].to_dict()
    qpy_path = preflight_dir / str(record["qpy_file"])
    if not qpy_path.exists():
        raise FileNotFoundError(qpy_path)
    return qpy_path, record


def load_single_qpy(path: Path):
    from qiskit import qpy

    with path.open("rb") as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise ValueError(f"Expected one circuit in {path}; found {len(circuits)}")
    return circuits[0]


def get_counts_from_pub_result(pub_result, circuit) -> dict[str, int]:
    if not circuit.cregs:
        raise ValueError("Circuit has no classical register")
    if len(circuit.cregs) != 1:
        raise ValueError("This pilot expects exactly one classical register")
    register_name = circuit.cregs[0].name
    data = pub_result.data
    if not hasattr(data, register_name):
        available = [name for name in dir(data) if not name.startswith("_")]
        raise AttributeError(
            f"Result data has no register {register_name!r}; available fields: {available}"
        )
    bit_array = getattr(data, register_name)
    return {str(k): int(v) for k, v in bit_array.get_counts().items()}


def save_counts(path: Path, counts: dict[str, int], metadata: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"counts": counts, "metadata": metadata}, handle, indent=2)


def job_metrics_safe(job) -> dict:
    try:
        value = job.metrics()
        return value if isinstance(value, dict) else {"raw": str(value)}
    except Exception as exc:
        return {"metrics_error": repr(exc)}
