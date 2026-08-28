from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from step5_curved_signal_graph import build_curved_signal_graph, exact_milp
from step9_common import (
    get_rcm_order,
    graph_coefficients,
    label_accuracy,
    normalized_gap,
    qiskit_bitstring_to_spins,
    reorder_graph,
)
from step9_run_10000 import expected_energy_lightcone_p1
from step11_common import gate_mode_to_fractional, job_metrics_safe


def rebuild_graph(seed: int = 2000):
    graph, diagnostics = build_curved_signal_graph(
        grid_size=5,
        n_gradients=60,
        snr=10.0,
        b_value=2000.0,
        geometry='curved_90',
        fraction_mode='imbalanced',
        degree_cap=3,
        conflict_weight=1.0,
        anchor_weight=0.35,
        sigma_orientation_deg=18.0,
        sigma_displacement_deg=22.0,
        minimum_continuation_weight=0.01,
        seed=seed,
    )
    return reorder_graph(graph, get_rcm_order(graph)), diagnostics


def build_parameterized_qaoa_circuit(graph):
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    gamma = Parameter('gamma')
    beta = Parameter('beta')
    z_coefficients, zz_coefficients = graph_coefficients(graph)

    circuit = QuantumCircuit(graph.n_nodes, graph.n_nodes)
    circuit.h(range(graph.n_nodes))
    for i, coefficient in enumerate(z_coefficients):
        if abs(coefficient) > 1e-15:
            circuit.rz(2.0 * gamma * coefficient, i)
    for i, j, coefficient in zz_coefficients:
        circuit.rzz(2.0 * gamma * coefficient, i, j)
    circuit.rx(2.0 * beta, range(graph.n_nodes))
    circuit.measure(range(graph.n_nodes), range(graph.n_nodes))
    return circuit


def build_angle_grid(
    *,
    gamma_center: float,
    beta_center: float,
    gamma_half_width: float,
    beta_half_width: float,
    points_per_axis: int,
) -> pd.DataFrame:
    if points_per_axis < 3 or points_per_axis % 2 == 0:
        raise ValueError('points_per_axis must be an odd integer of at least 3')
    gamma_values = np.linspace(
        gamma_center - gamma_half_width,
        gamma_center + gamma_half_width,
        points_per_axis,
    )
    beta_values = np.linspace(
        beta_center - beta_half_width,
        beta_center + beta_half_width,
        points_per_axis,
    )
    rows = []
    index = 0
    center_index = points_per_axis // 2
    for gamma_index, gamma in enumerate(gamma_values):
        for beta_index, beta in enumerate(beta_values):
            rows.append(
                {
                    'grid_index': index,
                    'gamma_index': gamma_index,
                    'beta_index': beta_index,
                    'gamma': float(gamma),
                    'beta': float(beta),
                    'is_transferred_center': bool(
                        gamma_index == center_index and beta_index == center_index
                    ),
                }
            )
            index += 1
    return pd.DataFrame(rows)


def parameter_values_for_circuit(circuit, grid: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    parameters = list(circuit.parameters)
    names = [str(parameter.name) for parameter in parameters]
    if set(names) != {'gamma', 'beta'} or len(names) != 2:
        raise ValueError(f'Expected parameters gamma and beta; found {names}')
    values = np.empty((len(grid), len(parameters)), dtype=float)
    for row_index, row in grid.reset_index(drop=True).iterrows():
        lookup = {'gamma': float(row['gamma']), 'beta': float(row['beta'])}
        values[row_index] = [lookup[name] for name in names]
    return values, names


def get_result_bit_array(pub_result, circuit):
    if len(circuit.cregs) != 1:
        raise ValueError('Expected exactly one classical register')
    name = circuit.cregs[0].name
    if not hasattr(pub_result.data, name):
        available = [item for item in dir(pub_result.data) if not item.startswith('_')]
        raise AttributeError(f'Result has no register {name!r}; available={available}')
    return getattr(pub_result.data, name)


def cvar_from_energy_counts(
    energy_counts: Iterable[tuple[float, int]],
    *,
    shots: int,
    alpha: float = 0.10,
) -> float:
    if not 0.0 < alpha <= 1.0:
        raise ValueError('alpha must be in (0, 1]')
    target = alpha * shots
    used = 0.0
    total = 0.0
    for energy, count in sorted(energy_counts, key=lambda item: item[0]):
        take = min(float(count), target - used)
        if take <= 0:
            break
        total += take * float(energy)
        used += take
    if used <= 0:
        raise ValueError('No shots available for CVaR')
    return float(total / used)


def analyze_counts(counts: dict[str, int], graph, exact_energy: float) -> dict:
    shots = int(sum(counts.values()))
    weighted_sum = 0.0
    weighted_square_sum = 0.0
    best_energy = np.inf
    best_bitstring = None
    best_accuracy = np.nan
    exact_hits = 0
    energy_counts: list[tuple[float, int]] = []

    for bitstring, count in counts.items():
        spins = qiskit_bitstring_to_spins(bitstring, graph.n_nodes)
        energy = float(graph.energy(spins))
        weighted_sum += count * energy
        weighted_square_sum += count * energy * energy
        energy_counts.append((energy, int(count)))
        if energy < best_energy:
            best_energy = energy
            best_bitstring = bitstring
            best_accuracy = label_accuracy(spins, graph.true_spins)
        if np.isclose(energy, exact_energy, atol=1e-8):
            exact_hits += int(count)

    mean_energy = weighted_sum / shots
    if shots > 1:
        sample_variance = max(
            0.0,
            (weighted_square_sum - shots * mean_energy * mean_energy) / (shots - 1),
        )
        mean_energy_se = float(np.sqrt(sample_variance / shots))
    else:
        mean_energy_se = np.nan
    cvar10_energy = cvar_from_energy_counts(energy_counts, shots=shots, alpha=0.10)

    return {
        'shots': shots,
        'n_unique_bitstrings': int(len(counts)),
        'mean_energy': float(mean_energy),
        'mean_energy_se': mean_energy_se,
        'mean_normalized_gap': normalized_gap(mean_energy, exact_energy),
        'mean_normalized_gap_se': float(mean_energy_se / (-exact_energy)),
        'cvar10_energy': cvar10_energy,
        'cvar10_normalized_gap': normalized_gap(cvar10_energy, exact_energy),
        'best_bitstring': best_bitstring,
        'best_energy': float(best_energy),
        'best_normalized_gap': normalized_gap(best_energy, exact_energy),
        'best_label_accuracy': float(best_accuracy),
        'exact_ground_hits': int(exact_hits),
    }


def save_grid_counts(path: Path, counts_by_index: dict[int, dict[str, int]], metadata: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        'counts_by_grid_index': {
            str(index): {str(key): int(value) for key, value in counts.items()}
            for index, counts in counts_by_index.items()
        },
        'metadata': metadata,
    }
    with gzip.open(path, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle)
