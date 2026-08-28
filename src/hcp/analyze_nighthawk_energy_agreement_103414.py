#!/usr/bin/env python3
"""Audit energy--assignment agreement for the final 103414 Nighthawk run.

The analysis is deliberately read-only with respect to the QPU results.  It
re-evaluates every saved hardware (and, by default, ideal-MPS) bitstring using
the same Qiskit bit ordering and the same saved RCM-labelled Ising graph used by
the collection script.

Run from the crossing_fibres_step7 project directory, for example:

    python analyze_nighthawk_energy_agreement_103414.py \
      --case-dir real_data_8n_replication/103414_R_88_86_76

Outputs are written to <case-dir>/nighthawk_energy_agreement_analysis unless
--out is supplied.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXPECTED_CASE = "103414_R_88_86_76"
EXPECTED_NODES = 50
EXPECTED_EDGES = 64
EXPECTED_EXACT_ENERGY = -36.1419628855128
TAIL_SIZES = (1, 10, 100, 1_000)
AGREEMENT_TARGETS = (0.60, 0.70, 0.80, 0.90, 1.00)
NEAR_BEST_GAP_DELTAS = (0.0, 0.005, 0.010, 0.020, 0.050, 0.100)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def clean_bitstring(value: Any, n_nodes: int) -> str:
    bitstring = str(value).replace(" ", "").replace("_", "")
    if len(bitstring) != n_nodes or set(bitstring) - {"0", "1"}:
        raise ValueError(f"Invalid {n_nodes}-qubit bitstring: {value!r}")
    return bitstring


def qiskit_bitstrings_to_spins(bitstrings: Iterable[str], n_nodes: int) -> np.ndarray:
    """Convert Qiskit's high-to-low classical-bit strings to RCM-node spins."""
    values = list(bitstrings)
    if not values:
        raise ValueError("No bitstrings were supplied")
    joined = "".join(values).encode("ascii")
    encoded = np.frombuffer(joined, dtype=np.uint8).reshape(len(values), n_nodes)
    bits = encoded[:, ::-1] - ord("0")
    if np.any((bits != 0) & (bits != 1)):
        raise ValueError("Bitstrings contain characters other than 0 and 1")
    return (1 - 2 * bits).astype(np.int8, copy=False)


def aligned_agreement(spins: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.int8)
    direct = np.mean(spins == reference[None, :], axis=1)
    reversed_sign = np.mean(-spins == reference[None, :], axis=1)
    return np.maximum(direct, reversed_sign)


def one_aligned_agreement(first: np.ndarray, second: np.ndarray) -> float:
    first = np.asarray(first, dtype=np.int8)
    second = np.asarray(second, dtype=np.int8)
    return float(max(np.mean(first == second), np.mean(-first == second)))


def normalized_gap(energy: np.ndarray | float, exact_energy: float):
    denominator = -float(exact_energy)
    if denominator <= 0:
        raise ValueError("The exact energy must be negative for this normalization")
    return (np.asarray(energy, dtype=float) - float(exact_energy)) / denominator


def read_json(path: Path) -> Any:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return json.load(handle)
    return json.loads(path.read_text(encoding="utf-8"))


def extract_counts(payload: Any, *, strategy: str | None = None) -> dict[str, int]:
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Counts payload is not a non-empty JSON object")

    if strategy is not None and strategy in payload:
        payload = payload[strategy]
    elif not all(isinstance(value, (int, float)) for value in payload.values()):
        if len(payload) == 1:
            payload = next(iter(payload.values()))
        else:
            raise ValueError(
                "Counts file contains multiple strategies; specify the intended one"
            )

    if not isinstance(payload, dict) or not payload:
        raise ValueError("Selected counts object is empty or invalid")

    counts: dict[str, int] = {}
    for bitstring, raw_count in payload.items():
        count = int(raw_count)
        if count <= 0 or float(raw_count) != count:
            raise ValueError(f"Invalid count for {bitstring!r}: {raw_count!r}")
        counts[str(bitstring)] = count
    return counts


def load_graph(graph_dir: Path) -> dict[str, Any]:
    nodes_path = graph_dir / "real_graph_nodes_rcm.csv"
    edges_path = graph_dir / "real_graph_edges_rcm.csv"
    arrays_path = graph_dir / "real_graph_solution_arrays.npz"
    validation_path = graph_dir / "real_graph_validation.json"
    for path in (nodes_path, edges_path, arrays_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    nodes = pd.read_csv(nodes_path).sort_values("node_id").reset_index(drop=True)
    edges = pd.read_csv(edges_path).reset_index(drop=True)
    required_node_columns = {"node_id", "old_node_id", "true_spin"}
    required_edge_columns = {"i", "j", "edge_type", "weight"}
    if missing := required_node_columns - set(nodes.columns):
        raise ValueError(f"Missing node columns: {sorted(missing)}")
    if missing := required_edge_columns - set(edges.columns):
        raise ValueError(f"Missing edge columns: {sorted(missing)}")
    if nodes["node_id"].tolist() != list(range(len(nodes))):
        raise ValueError("RCM node_id values are not contiguous from zero")

    edge_types = set(edges["edge_type"].astype(str))
    if edge_types != {"conflict", "continuation"}:
        raise ValueError(f"Unexpected edge types: {sorted(edge_types)}")

    with np.load(arrays_path, allow_pickle=False) as archive:
        fields_pre_rcm = np.asarray(archive["fields_pre_rcm"], dtype=float)
        exact_spins = np.asarray(archive["exact_spins_rcm"], dtype=np.int8)

    old_node_ids = nodes["old_node_id"].to_numpy(dtype=int)
    fields = fields_pre_rcm[old_node_ids]
    tom_spins = nodes["true_spin"].to_numpy(dtype=np.int8)

    if len(nodes) != EXPECTED_NODES:
        raise RuntimeError(f"Expected {EXPECTED_NODES} nodes; found {len(nodes)}")
    if len(edges) != EXPECTED_EDGES:
        raise RuntimeError(f"Expected {EXPECTED_EDGES} edges; found {len(edges)}")
    for name, values in (("exact", exact_spins), ("TOM", tom_spins)):
        if values.shape != (len(nodes),) or not np.all(np.isin(values, (-1, 1))):
            raise ValueError(f"Invalid {name} spin reference")

    validation = read_json(validation_path) if validation_path.is_file() else None
    return {
        "nodes": nodes,
        "edges": edges,
        "fields": fields,
        "exact_spins": exact_spins,
        "tom_spins": tom_spins,
        "validation": validation,
    }


def graph_energies(spins: np.ndarray, graph: dict[str, Any]) -> np.ndarray:
    fields = graph["fields"]
    edges = graph["edges"]
    energies = -(spins * fields[None, :]).sum(axis=1)

    left = edges["i"].to_numpy(dtype=int)
    right = edges["j"].to_numpy(dtype=int)
    weights = edges["weight"].to_numpy(dtype=float)
    types = edges["edge_type"].astype(str).to_numpy()
    coefficients = np.where(types == "conflict", weights, -weights)
    energies += (
        spins[:, left] * spins[:, right] * coefficients[None, :]
    ).sum(axis=1)
    return energies


def edge_diagnostics(spins: np.ndarray, graph: dict[str, Any]) -> dict[str, np.ndarray]:
    edges = graph["edges"]
    left = edges["i"].to_numpy(dtype=int)
    right = edges["j"].to_numpy(dtype=int)
    weights = edges["weight"].to_numpy(dtype=float)
    types = edges["edge_type"].astype(str).to_numpy()
    products = spins[:, left] * spins[:, right]
    conflict = types == "conflict"
    continuation = types == "continuation"

    conflict_satisfied = np.mean(products[:, conflict] == -1, axis=1)
    continuation_satisfied = np.mean(products[:, continuation] == 1, axis=1)
    continuation_weights = weights[continuation]
    weighted_continuation = (
        ((products[:, continuation] == 1) * continuation_weights[None, :]).sum(axis=1)
        / continuation_weights.sum()
    )

    fields = graph["fields"]
    anchored = np.flatnonzero(np.abs(fields) > 1e-15)
    if anchored.size:
        anchor_satisfied = np.mean(
            spins[:, anchored] * np.sign(fields[anchored])[None, :] > 0,
            axis=1,
        )
    else:
        anchor_satisfied = np.full(len(spins), np.nan)

    return {
        "conflict_satisfaction_fraction": conflict_satisfied,
        "continuation_satisfaction_fraction": continuation_satisfied,
        "weighted_continuation_satisfaction_fraction": weighted_continuation,
        "anchor_field_satisfaction_fraction": anchor_satisfied,
    }


def analyse_counts(
    source: str,
    counts: dict[str, int],
    graph: dict[str, Any],
    exact_energy: float,
) -> pd.DataFrame:
    n_nodes = len(graph["nodes"])
    compact_counts: dict[str, int] = {}
    first_seen: dict[str, int] = {}
    for input_order, (raw_bitstring, count) in enumerate(counts.items()):
        bitstring = clean_bitstring(raw_bitstring, n_nodes)
        compact_counts[bitstring] = compact_counts.get(bitstring, 0) + int(count)
        first_seen.setdefault(bitstring, input_order)

    bitstrings = list(compact_counts)
    spins = qiskit_bitstrings_to_spins(bitstrings, n_nodes)
    energies = graph_energies(spins, graph)
    exact_agreement = aligned_agreement(spins, graph["exact_spins"])
    tom_agreement = aligned_agreement(spins, graph["tom_spins"])
    diagnostics = edge_diagnostics(spins, graph)

    frame = pd.DataFrame(
        {
            "source": source,
            "bitstring": bitstrings,
            "count": [compact_counts[value] for value in bitstrings],
            "input_order": [first_seen[value] for value in bitstrings],
            "energy": energies,
            "normalized_gap": normalized_gap(energies, exact_energy),
            "exact_agreement_up_to_sign": exact_agreement,
            "exact_hamming_distance_up_to_sign": np.rint(
                n_nodes * (1.0 - exact_agreement)
            ).astype(int),
            "tom_agreement_up_to_sign": tom_agreement,
            "tom_hamming_distance_up_to_sign": np.rint(
                n_nodes * (1.0 - tom_agreement)
            ).astype(int),
            **diagnostics,
        }
    )
    frame = frame.sort_values(
        ["energy", "input_order"], kind="stable"
    ).reset_index(drop=True)
    frame["energy_order"] = np.arange(1, len(frame) + 1)
    frame["energy_rank_min"] = frame["energy"].rank(method="min").astype(int)
    frame["shot_rank_end"] = frame["count"].cumsum()
    frame["shot_rank_start"] = frame["shot_rank_end"] - frame["count"] + 1
    best_energy = float(frame["energy"].iloc[0])
    frame["is_minimum_energy_tie"] = np.isclose(
        frame["energy"], best_energy, rtol=0.0, atol=1e-10
    )
    return frame


def expand_shots(frame: pd.DataFrame) -> pd.DataFrame:
    repeated = frame.loc[frame.index.repeat(frame["count"].astype(int))].copy()
    repeated = repeated.reset_index(drop=True)
    repeated["shot_rank"] = np.arange(1, len(repeated) + 1)
    return repeated


def descriptive_spearman(first: np.ndarray, second: np.ndarray) -> float:
    first_rank = pd.Series(first).rank(method="average").to_numpy(dtype=float)
    second_rank = pd.Series(second).rank(method="average").to_numpy(dtype=float)
    if np.std(first_rank) == 0 or np.std(second_rank) == 0:
        return float("nan")
    return float(np.corrcoef(first_rank, second_rank)[0, 1])


def source_summary(
    source: str,
    frame: pd.DataFrame,
    shots: pd.DataFrame,
    exact_energy: float,
    reported_best: str | None,
) -> dict[str, Any]:
    cvar_n = max(1, int(math.ceil(0.10 * len(shots))))
    cvar = shots.iloc[:cvar_n]
    minimum = float(frame["energy"].min())
    ties = frame[np.isclose(frame["energy"], minimum, rtol=0.0, atol=1e-10)]

    if reported_best is not None:
        reported_best = clean_bitstring(reported_best, EXPECTED_NODES)
        match = frame[frame["bitstring"] == reported_best]
        if len(match) != 1:
            raise RuntimeError(
                f"Stored {source} best bitstring was not found exactly once in counts"
            )
        selected = match.iloc[0]
        if not np.isclose(float(selected["energy"]), minimum, atol=1e-9):
            raise RuntimeError(f"Stored {source} best bitstring is not minimum-energy")
    else:
        selected = ties.sort_values("input_order", kind="stable").iloc[0]

    max_exact = float(frame["exact_agreement_up_to_sign"].max())
    max_tom = float(frame["tom_agreement_up_to_sign"].max())
    best_max_exact = frame[
        np.isclose(frame["exact_agreement_up_to_sign"], max_exact, atol=1e-12)
    ].iloc[0]
    best_max_tom = frame[
        np.isclose(frame["tom_agreement_up_to_sign"], max_tom, atol=1e-12)
    ].iloc[0]

    mean_energy = float(shots["energy"].mean())
    mean_energy_se = float(shots["energy"].std(ddof=0) / np.sqrt(len(shots)))
    return {
        "source": source,
        "shots": int(len(shots)),
        "unique_bitstrings": int(len(frame)),
        "mean_energy": mean_energy,
        "mean_energy_se": mean_energy_se,
        "mean_normalized_gap": float(normalized_gap(mean_energy, exact_energy)),
        "cvar10_energy": float(cvar["energy"].mean()),
        "cvar10_normalized_gap": float(
            normalized_gap(float(cvar["energy"].mean()), exact_energy)
        ),
        "best_energy": minimum,
        "best_normalized_gap": float(normalized_gap(minimum, exact_energy)),
        "reported_best_bitstring": str(selected["bitstring"]),
        "reported_best_exact_agreement_up_to_sign": float(
            selected["exact_agreement_up_to_sign"]
        ),
        "reported_best_tom_agreement_up_to_sign": float(
            selected["tom_agreement_up_to_sign"]
        ),
        "minimum_energy_tie_count": int(len(ties)),
        "minimum_energy_tie_exact_agreement_min": float(
            ties["exact_agreement_up_to_sign"].min()
        ),
        "minimum_energy_tie_exact_agreement_max": float(
            ties["exact_agreement_up_to_sign"].max()
        ),
        "maximum_sampled_exact_agreement_up_to_sign": max_exact,
        "gap_at_best_energy_with_maximum_exact_agreement": float(
            best_max_exact["normalized_gap"]
        ),
        "energy_rank_of_best_maximum_exact_agreement": int(
            best_max_exact["energy_order"]
        ),
        "maximum_sampled_tom_agreement_up_to_sign": max_tom,
        "gap_at_best_energy_with_maximum_tom_agreement": float(
            best_max_tom["normalized_gap"]
        ),
        "energy_rank_of_best_maximum_tom_agreement": int(
            best_max_tom["energy_order"]
        ),
        "exact_ground_hits": int(
            np.sum(np.isclose(shots["energy"], exact_energy, atol=1e-9))
        ),
        "spearman_gap_vs_exact_agreement": descriptive_spearman(
            shots["normalized_gap"].to_numpy(),
            shots["exact_agreement_up_to_sign"].to_numpy(),
        ),
        "spearman_gap_vs_tom_agreement": descriptive_spearman(
            shots["normalized_gap"].to_numpy(),
            shots["tom_agreement_up_to_sign"].to_numpy(),
        ),
    }


def low_energy_tail_summary(
    source: str, shots: pd.DataFrame, sizes: Iterable[int]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for requested in sizes:
        n = min(int(requested), len(shots))
        tail = shots.iloc[:n]
        rows.append(
            {
                "source": source,
                "lowest_n_shots": n,
                "fraction_of_all_shots": n / len(shots),
                "maximum_normalized_gap_in_tail": float(tail["normalized_gap"].max()),
                "mean_normalized_gap_in_tail": float(tail["normalized_gap"].mean()),
                "exact_agreement_mean": float(
                    tail["exact_agreement_up_to_sign"].mean()
                ),
                "exact_agreement_median": float(
                    tail["exact_agreement_up_to_sign"].median()
                ),
                "exact_agreement_min": float(
                    tail["exact_agreement_up_to_sign"].min()
                ),
                "exact_agreement_max": float(
                    tail["exact_agreement_up_to_sign"].max()
                ),
                "tom_agreement_mean": float(tail["tom_agreement_up_to_sign"].mean()),
                "tom_agreement_median": float(
                    tail["tom_agreement_up_to_sign"].median()
                ),
                "tom_agreement_min": float(tail["tom_agreement_up_to_sign"].min()),
                "tom_agreement_max": float(tail["tom_agreement_up_to_sign"].max()),
                "conflict_satisfaction_mean": float(
                    tail["conflict_satisfaction_fraction"].mean()
                ),
                "continuation_satisfaction_mean": float(
                    tail["continuation_satisfaction_fraction"].mean()
                ),
                "weighted_continuation_satisfaction_mean": float(
                    tail["weighted_continuation_satisfaction_fraction"].mean()
                ),
            }
        )
    return pd.DataFrame(rows)


def agreement_target_summary(
    source: str,
    shots: pd.DataFrame,
    targets: Iterable[float],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    cvar_n = max(1, int(math.ceil(0.10 * len(shots))))
    for reference_name, column in (
        ("exact", "exact_agreement_up_to_sign"),
        ("tom", "tom_agreement_up_to_sign"),
    ):
        for target in targets:
            qualifying = shots[shots[column] >= float(target) - 1e-12]
            qualifying_cvar = shots.iloc[:cvar_n]
            qualifying_cvar = qualifying_cvar[
                qualifying_cvar[column] >= float(target) - 1e-12
            ]
            if qualifying.empty:
                best_gap = float("nan")
                best_rank = None
                agreement_at_best_gap = float("nan")
                bitstring = None
            else:
                best = qualifying.iloc[0]
                best_gap = float(best["normalized_gap"])
                best_rank = int(best["shot_rank"])
                agreement_at_best_gap = float(best[column])
                bitstring = str(best["bitstring"])
            rows.append(
                {
                    "source": source,
                    "reference": reference_name,
                    "agreement_target": float(target),
                    "qualifying_shots": int(len(qualifying)),
                    "qualifying_shot_fraction": float(len(qualifying) / len(shots)),
                    "qualifying_shots_in_lowest_10pct_energy": int(
                        len(qualifying_cvar)
                    ),
                    "fraction_of_lowest_10pct_meeting_target": float(
                        len(qualifying_cvar) / cvar_n
                    ),
                    "minimum_normalized_gap_meeting_target": best_gap,
                    "lowest_energy_shot_rank_meeting_target": best_rank,
                    "agreement_at_minimum_gap_meeting_target": agreement_at_best_gap,
                    "bitstring_at_minimum_gap_meeting_target": bitstring,
                }
            )
    return pd.DataFrame(rows)


def near_best_summary(source: str, shots: pd.DataFrame) -> pd.DataFrame:
    best_gap = float(shots["normalized_gap"].iloc[0])
    rows: list[dict[str, Any]] = []
    for delta in NEAR_BEST_GAP_DELTAS:
        subset = shots[
            shots["normalized_gap"] <= best_gap + float(delta) + 1e-12
        ]
        rows.append(
            {
                "source": source,
                "gap_delta_from_best": float(delta),
                "maximum_allowed_normalized_gap": best_gap + float(delta),
                "qualifying_shots": int(len(subset)),
                "qualifying_shot_fraction": float(len(subset) / len(shots)),
                "exact_agreement_median": float(
                    subset["exact_agreement_up_to_sign"].median()
                ),
                "exact_agreement_max": float(
                    subset["exact_agreement_up_to_sign"].max()
                ),
                "tom_agreement_median": float(
                    subset["tom_agreement_up_to_sign"].median()
                ),
                "tom_agreement_max": float(
                    subset["tom_agreement_up_to_sign"].max()
                ),
            }
        )
    return pd.DataFrame(rows)


def pareto_front(source: str, frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for reference_name, column in (
        ("exact", "exact_agreement_up_to_sign"),
        ("tom", "tom_agreement_up_to_sign"),
    ):
        best_seen = -np.inf
        ordered = frame.sort_values(
            ["normalized_gap", column], ascending=[True, False], kind="stable"
        )
        for _, item in ordered.iterrows():
            agreement = float(item[column])
            if agreement > best_seen + 1e-12:
                rows.append(
                    pd.Series(
                        {
                            "source": source,
                            "reference": reference_name,
                            "bitstring": item["bitstring"],
                            "count": int(item["count"]),
                            "energy_order": int(item["energy_order"]),
                            "normalized_gap": float(item["normalized_gap"]),
                            "agreement_up_to_sign": agreement,
                        }
                    )
                )
                best_seen = agreement
    return pd.DataFrame(rows)


def validate_stored_metrics(
    source: str,
    computed: dict[str, Any],
    stored: dict[str, Any] | None,
) -> str | None:
    if stored is None:
        return None
    stored_result = stored.get("result", stored)
    checks = {
        "mean_normalized_gap": "mean_normalized_gap",
        "cvar10_normalized_gap": "cvar10_normalized_gap",
        "best_normalized_gap": "best_normalized_gap",
        "best_agreement_with_exact_up_to_sign": (
            "reported_best_exact_agreement_up_to_sign"
        ),
    }
    for stored_key, computed_key in checks.items():
        if stored_key not in stored_result:
            continue
        if not np.isclose(
            float(stored_result[stored_key]),
            float(computed[computed_key]),
            rtol=1e-8,
            atol=5e-7,
        ):
            raise RuntimeError(
                f"{source} {stored_key} does not reproduce: "
                f"stored={stored_result[stored_key]!r}, "
                f"computed={computed[computed_key]!r}"
            )
    value = stored_result.get("best_bitstring")
    return str(value) if value is not None else None


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def make_figure(
    frames: dict[str, pd.DataFrame],
    shots: dict[str, pd.DataFrame],
    summaries: dict[str, dict[str, Any]],
    references_equivalent: bool,
    out: Path,
) -> None:
    colors = {"hardware": "#2C6EAA", "ideal_mps": "#E07A2D"}
    labels = {"hardware": "Nighthawk QPU", "ideal_mps": "Ideal MPS"}
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.5), constrained_layout=True)

    left, right = axes
    for source, frame in frames.items():
        color = colors.get(source)
        label = labels.get(source, source)
        left.scatter(
            frame["normalized_gap"],
            100.0 * frame["exact_agreement_up_to_sign"],
            s=10,
            alpha=0.20,
            linewidths=0,
            color=color,
            label=label,
            rasterized=True,
        )
        reported = summaries[source]["reported_best_bitstring"]
        winner = frame[frame["bitstring"] == reported].iloc[0]
        left.scatter(
            [winner["normalized_gap"]],
            [100.0 * winner["exact_agreement_up_to_sign"]],
            s=115,
            marker="*",
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=5,
        )

    left.axhline(50.0, color="0.55", linewidth=0.9, linestyle=":")
    left.set_xlabel("Normalized gap")
    left.set_ylabel("Agreement with exact assignment up to sign (%)")
    left.set_ylim(49.0, 101.0)
    left.set_title("a  Energy–assignment relationship", loc="left", fontweight="bold")
    left.grid(alpha=0.18, linewidth=0.7)
    left.legend(frameon=False)

    for source, shot_frame in shots.items():
        color = colors.get(source)
        label = labels.get(source, source)
        rank = shot_frame["shot_rank"].to_numpy()
        cumulative_exact = np.maximum.accumulate(
            shot_frame["exact_agreement_up_to_sign"].to_numpy()
        )
        right.plot(
            rank,
            100.0 * cumulative_exact,
            color=color,
            linewidth=2.0,
            label=f"{label}: exact",
        )
        if not references_equivalent:
            cumulative_tom = np.maximum.accumulate(
                shot_frame["tom_agreement_up_to_sign"].to_numpy()
            )
            right.plot(
                rank,
                100.0 * cumulative_tom,
                color=color,
                linewidth=1.5,
                linestyle="--",
                label=f"{label}: TOM",
            )

    for rank_value in (10, 100, 1_000):
        right.axvline(rank_value, color="0.82", linewidth=0.7, zorder=0)
    right.set_xscale("log")
    right.set_xlim(1, max(len(value) for value in shots.values()))
    right.set_ylim(49.0, 101.0)
    right.set_xlabel("Number of lowest-energy shots admitted")
    right.set_ylabel("Best agreement encountered (%)")
    right.set_title("b  Best attainable agreement in the tail", loc="left", fontweight="bold")
    right.grid(alpha=0.18, linewidth=0.7)
    right.legend(frameon=False)

    suffix = " (exact and TOM references coincide up to sign)" if references_equivalent else ""
    fig.suptitle("Nighthawk energy–agreement audit" + suffix, fontsize=12)
    for extension, kwargs in (
        ("png", {"dpi": 300}),
        ("pdf", {}),
        ("svg", {}),
    ):
        fig.savefig(out / f"nighthawk_energy_agreement_diagnostic.{extension}", **kwargs)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory (default: <case-dir>/nighthawk_energy_agreement_analysis)",
    )
    parser.add_argument(
        "--skip-ideal",
        action="store_true",
        help="Analyse hardware only if the ideal-MPS counts are unavailable",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    case_dir = args.case_dir.resolve()
    if case_dir.name != EXPECTED_CASE:
        raise RuntimeError(
            f"Expected final case {EXPECTED_CASE!r}; received {case_dir.name!r}"
        )

    graph_dir = case_dir / "graph_validation"
    hardware_dir = case_dir / "nighthawk_preflight"
    ideal_dir = case_dir / "mps_nighthawk_ideal"
    out = (args.out or case_dir / "nighthawk_energy_agreement_analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)

    graph = load_graph(graph_dir)
    exact_energy = float(
        graph_energies(graph["exact_spins"][None, :], graph)[0]
    )
    if not np.isclose(exact_energy, EXPECTED_EXACT_ENERGY, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"Exact energy mismatch: expected {EXPECTED_EXACT_ENERGY}, "
            f"reconstructed {exact_energy}"
        )
    if graph["validation"] is not None:
        stored_energy = float(graph["validation"]["exact_anchored"]["energy"])
        if not np.isclose(exact_energy, stored_energy, rtol=0.0, atol=1e-9):
            raise RuntimeError("Exact energy differs from graph-validation metadata")

    source_inputs: dict[str, dict[str, Any]] = {}
    hardware_counts_path = hardware_dir / "real_nighthawk_hardware_counts.json.gz"
    hardware_summary_path = hardware_dir / "real_nighthawk_hardware_summary.json"
    for path in (hardware_counts_path, hardware_summary_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    hardware_summary = read_json(hardware_summary_path)
    source_inputs["hardware"] = {
        "counts": extract_counts(read_json(hardware_counts_path)),
        "stored": hardware_summary,
    }

    if not args.skip_ideal:
        ideal_counts_path = ideal_dir / "real_mps_counts.json.gz"
        ideal_results_path = ideal_dir / "real_mps_results.csv"
        for path in (ideal_counts_path, ideal_results_path):
            if not path.is_file():
                raise FileNotFoundError(path)
        ideal_results = pd.read_csv(
            ideal_results_path, dtype={"best_bitstring": str}
        )
        selected = ideal_results[
            ideal_results["angle_strategy"].astype(str) == "nighthawk_calibrated"
        ]
        if len(selected) != 1:
            raise RuntimeError(
                "Expected one nighthawk_calibrated row in ideal MPS results"
            )
        source_inputs["ideal_mps"] = {
            "counts": extract_counts(
                read_json(ideal_counts_path), strategy="nighthawk_calibrated"
            ),
            "stored": selected.iloc[0].to_dict(),
        }

    frames: dict[str, pd.DataFrame] = {}
    shot_frames: dict[str, pd.DataFrame] = {}
    summaries: dict[str, dict[str, Any]] = {}
    tail_frames: list[pd.DataFrame] = []
    target_frames: list[pd.DataFrame] = []
    near_best_frames: list[pd.DataFrame] = []
    pareto_frames: list[pd.DataFrame] = []

    # First pass computes the stored winner independently from the raw counts.
    for source, values in source_inputs.items():
        frame = analyse_counts(source, values["counts"], graph, exact_energy)
        shots = expand_shots(frame)
        stored = values["stored"]
        stored_result = stored.get("result", stored)
        reported_best = stored_result.get("best_bitstring")
        summary = source_summary(
            source, frame, shots, exact_energy, reported_best
        )
        validate_stored_metrics(source, summary, stored)

        frame["is_reported_best"] = (
            frame["bitstring"] == summary["reported_best_bitstring"]
        )
        frames[source] = frame
        shot_frames[source] = shots
        summaries[source] = summary
        tail_frames.append(low_energy_tail_summary(source, shots, TAIL_SIZES))
        target_frames.append(
            agreement_target_summary(source, shots, AGREEMENT_TARGETS)
        )
        near_best_frames.append(near_best_summary(source, shots))
        pareto_frames.append(pareto_front(source, frame))

    outcomes = pd.concat(frames.values(), ignore_index=True)
    tails = pd.concat(tail_frames, ignore_index=True)
    targets = pd.concat(target_frames, ignore_index=True)
    near_best = pd.concat(near_best_frames, ignore_index=True)
    pareto = pd.concat(pareto_frames, ignore_index=True)
    source_table = pd.DataFrame(summaries.values())

    exact_tom_agreement = one_aligned_agreement(
        graph["exact_spins"], graph["tom_spins"]
    )
    references_equivalent = np.isclose(exact_tom_agreement, 1.0, atol=1e-12)

    outcomes.to_csv(
        out / "nighthawk_energy_agreement_outcomes.csv.gz",
        index=False,
        compression="gzip",
    )
    source_table.to_csv(out / "nighthawk_energy_agreement_source_summary.csv", index=False)
    tails.to_csv(out / "nighthawk_low_energy_tail_summary.csv", index=False)
    targets.to_csv(out / "nighthawk_agreement_target_summary.csv", index=False)
    near_best.to_csv(out / "nighthawk_near_best_summary.csv", index=False)
    pareto.to_csv(out / "nighthawk_energy_agreement_pareto_front.csv", index=False)

    summary_payload = {
        "created_at_utc": utc_now(),
        "case": case_dir.name,
        "graph_directory": str(graph_dir),
        "n_nodes": int(len(graph["nodes"])),
        "n_edges": int(len(graph["edges"])),
        "exact_energy": exact_energy,
        "exact_tom_reference_agreement_up_to_sign": exact_tom_agreement,
        "exact_and_tom_references_equivalent_up_to_sign": bool(
            references_equivalent
        ),
        "sources": summaries,
        "low_energy_tails": tails.to_dict(orient="records"),
        "agreement_targets": targets.to_dict(orient="records"),
        "near_best": near_best.to_dict(orient="records"),
        "files": {
            "outcomes": "nighthawk_energy_agreement_outcomes.csv.gz",
            "source_summary": "nighthawk_energy_agreement_source_summary.csv",
            "low_energy_tails": "nighthawk_low_energy_tail_summary.csv",
            "agreement_targets": "nighthawk_agreement_target_summary.csv",
            "near_best": "nighthawk_near_best_summary.csv",
            "pareto_front": "nighthawk_energy_agreement_pareto_front.csv",
            "figure_png": "nighthawk_energy_agreement_diagnostic.png",
            "figure_pdf": "nighthawk_energy_agreement_diagnostic.pdf",
            "figure_svg": "nighthawk_energy_agreement_diagnostic.svg",
        },
    }
    (out / "nighthawk_energy_agreement_summary.json").write_text(
        json.dumps(json_ready(summary_payload), indent=2) + "\n",
        encoding="utf-8",
    )
    make_figure(
        frames,
        shot_frames,
        summaries,
        bool(references_equivalent),
        out,
    )

    print("\nNIGHTHAWK ENERGY--AGREEMENT AUDIT")
    print("=" * 86)
    print(f"Case: {case_dir.name}")
    print(f"Exact energy: {exact_energy:.12f}")
    print(
        "Exact--TOM reference agreement up to sign: "
        f"{100.0 * exact_tom_agreement:.1f}%"
    )
    print("\nSOURCE SUMMARY")
    display_source = source_table[
        [
            "source",
            "shots",
            "unique_bitstrings",
            "mean_normalized_gap",
            "cvar10_normalized_gap",
            "best_normalized_gap",
            "reported_best_exact_agreement_up_to_sign",
            "maximum_sampled_exact_agreement_up_to_sign",
            "gap_at_best_energy_with_maximum_exact_agreement",
        ]
    ].copy()
    print(display_source.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nLOW-ENERGY TAIL")
    display_tails = tails[
        [
            "source",
            "lowest_n_shots",
            "maximum_normalized_gap_in_tail",
            "exact_agreement_median",
            "exact_agreement_max",
            "tom_agreement_median",
            "tom_agreement_max",
        ]
    ].copy()
    for column in (
        "exact_agreement_median",
        "exact_agreement_max",
        "tom_agreement_median",
        "tom_agreement_max",
    ):
        display_tails[column] *= 100.0
    print(display_tails.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\nLOWEST-ENERGY OUTCOME REACHING EACH EXACT-AGREEMENT TARGET")
    display_targets = targets[targets["reference"] == "exact"][
        [
            "source",
            "agreement_target",
            "qualifying_shots",
            "minimum_normalized_gap_meeting_target",
            "lowest_energy_shot_rank_meeting_target",
        ]
    ].copy()
    display_targets["agreement_target"] *= 100.0
    print(display_targets.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nWrote outputs to: {out}")


if __name__ == "__main__":
    main()
