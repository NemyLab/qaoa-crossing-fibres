from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable

import networkx as nx
import numpy as np


def qiskit_bitstring_to_spins(bitstring: str, n_nodes: int) -> np.ndarray:
    clean = bitstring.replace(" ", "").replace("_", "")
    if len(clean) != n_nodes or set(clean) - {"0", "1"}:
        raise ValueError(f"Invalid {n_nodes}-qubit bitstring: {bitstring!r}")
    # Qiskit prints classical bits high-to-low; qubit 0 is rightmost.
    bits = np.fromiter((int(char) for char in clean[::-1]), dtype=np.int8)
    return 1 - 2 * bits


def spins_to_qiskit_bitstring(spins: Iterable[int]) -> str:
    spins = np.asarray(tuple(spins), dtype=int)
    if not np.all(np.isin(spins, (-1, +1))):
        raise ValueError("spins must contain only -1 and +1")
    bits = ((1 - spins) // 2).astype(int)
    return "".join(str(int(bit)) for bit in bits[::-1])


def normalized_gap(energy: float, exact_energy: float) -> float:
    denominator = -float(exact_energy)
    if denominator <= 0.0:
        raise ValueError("The exact energy must be negative for this normalization")
    return float((float(energy) - float(exact_energy)) / denominator)


def label_accuracy(spins: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.asarray(spins) == np.asarray(truth)))


def get_rcm_order(graph) -> list[int]:
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(graph.n_nodes))
    for edge in graph.edges.itertuples(index=False):
        nx_graph.add_edge(int(edge.i), int(edge.j))
    return list(nx.utils.reverse_cuthill_mckee_ordering(nx_graph))


def reorder_graph(graph, order: Iterable[int]):
    # Import here so the utility module remains testable without project files.
    from step1_crossing_graph import CrossingGraph

    order = list(map(int, order))
    if sorted(order) != list(range(graph.n_nodes)):
        raise ValueError("order must be a permutation of all graph nodes")
    old_to_new = {old: new for new, old in enumerate(order)}

    nodes = graph.nodes.copy()
    nodes["old_node_id"] = nodes["node_id"].astype(int)
    nodes["node_id"] = nodes["old_node_id"].map(old_to_new).astype(int)
    nodes = nodes.sort_values("node_id").reset_index(drop=True)

    edges = graph.edges.copy()
    edges["i"] = edges["i"].astype(int).map(old_to_new).astype(int)
    edges["j"] = edges["j"].astype(int).map(old_to_new).astype(int)

    fields = np.zeros_like(graph.fields, dtype=float)
    true_spins = np.zeros_like(graph.true_spins, dtype=int)
    for old, new in old_to_new.items():
        fields[new] = graph.fields[old]
        true_spins[new] = graph.true_spins[old]

    return CrossingGraph(
        nodes=nodes,
        edges=edges,
        fields=fields,
        true_spins=true_spins,
        grid_size=graph.grid_size,
        seed=graph.seed,
    )


def graph_coefficients(graph) -> tuple[np.ndarray, list[tuple[int, int, float]]]:
    """Return H = sum_i z_i Z_i + sum_(i,j) J_ij Z_i Z_j."""
    z = -np.asarray(graph.fields, dtype=float).copy()
    zz: list[tuple[int, int, float]] = []
    for edge in graph.edges.itertuples(index=False):
        if edge.edge_type == "continuation":
            coefficient = -float(edge.weight)
        elif edge.edge_type == "conflict":
            coefficient = +float(edge.weight)
        else:
            raise ValueError(f"Unknown edge type: {edge.edge_type}")
        zz.append((int(edge.i), int(edge.j), coefficient))
    return z, zz


def energy_batch(graph, spins: np.ndarray) -> np.ndarray:
    """Vectorized graph energy for shape (n_states, n_nodes)."""
    spins = np.asarray(spins, dtype=np.int8)
    if spins.ndim != 2 or spins.shape[1] != graph.n_nodes:
        raise ValueError("spins must have shape (n_states, n_nodes)")
    energy = -(spins.astype(float) @ np.asarray(graph.fields, dtype=float))
    for edge in graph.edges.itertuples(index=False):
        product = spins[:, int(edge.i)] * spins[:, int(edge.j)]
        if edge.edge_type == "continuation":
            energy += -float(edge.weight) * product
        elif edge.edge_type == "conflict":
            energy += +float(edge.weight) * product
        else:
            raise ValueError(f"Unknown edge type: {edge.edge_type}")
    return energy.astype(float)


def save_counts(path: Path, counts: dict[str, int], metadata: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"counts": counts}
    if metadata is not None:
        payload["metadata"] = metadata
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)


def load_counts(path: Path) -> tuple[dict[str, int], dict]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
    else:
        payload = json.loads(path.read_text())
    if isinstance(payload, dict) and "counts" in payload:
        return (
            {str(key): int(value) for key, value in payload["counts"].items()},
            dict(payload.get("metadata", {})),
        )
    if isinstance(payload, dict):
        return {str(key): int(value) for key, value in payload.items()}, {}
    raise ValueError(f"Unsupported counts payload in {path}")
