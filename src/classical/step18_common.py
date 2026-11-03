from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from step17_classical_controls import (
    TrialResult,
    energies_batch,
    flip_delta_scale,
    initialise_local_field,
    label_accuracy,
    random_states,
)


def simulated_annealing_tunable_trial(
    *,
    rng: np.random.Generator,
    budget: int,
    n_nodes: int,
    linear: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_jij: np.ndarray,
    adjacency,
    truth: np.ndarray,
    exact_energy: float,
    restarts: int,
    t_start_multiplier: float,
    t_end_multiplier: float,
) -> TrialResult:
    """Budgeted SA with a geometric schedule and explicit query accounting.

    One initial full-energy calculation and each proposed spin-flip delta count as
    one query. The routine never exceeds ``budget``.
    """
    if budget < 1:
        raise ValueError("budget must be positive")
    if restarts < 1:
        raise ValueError("restarts must be positive")
    if t_start_multiplier <= 0 or t_end_multiplier <= 0:
        raise ValueError("temperature multipliers must be positive")
    if t_end_multiplier >= t_start_multiplier:
        raise ValueError("t_end_multiplier must be smaller than t_start_multiplier")

    allocations = np.full(restarts, budget // restarts, dtype=int)
    allocations[: budget % restarts] += 1
    allocations = allocations[allocations > 0]

    best_energy = math.inf
    best_spins = None
    evaluations = 0
    scale = flip_delta_scale(linear, adjacency)
    t_start = max(scale * float(t_start_multiplier), 1e-12)
    t_end = max(scale * float(t_end_multiplier), 1e-12)

    for chain_budget in allocations:
        spins = random_states(rng, 1, n_nodes)[0]
        energy = float(energies_batch(spins, linear, edge_i, edge_j, edge_jij)[0])
        evaluations += 1
        local_field = initialise_local_field(spins, linear, adjacency)
        if energy < best_energy:
            best_energy = energy
            best_spins = spins.copy()
        proposals = max(0, int(chain_budget) - 1)
        for step in range(proposals):
            fraction = step / max(1, proposals - 1)
            temperature = t_start * (t_end / t_start) ** fraction
            index = int(rng.integers(0, n_nodes))
            delta = float(-2.0 * spins[index] * local_field[index])
            evaluations += 1
            if delta <= 0.0 or rng.random() < math.exp(-delta / temperature):
                old_spin = int(spins[index])
                spins[index] = -old_spin
                energy += delta
                for neighbour, coupling in adjacency[index]:
                    local_field[neighbour] += -2.0 * coupling * old_spin
                if energy < best_energy:
                    best_energy = float(energy)
                    best_spins = spins.copy()
                    if np.isclose(best_energy, exact_energy, atol=1e-8):
                        return TrialResult(
                            best_energy=float(best_energy),
                            best_accuracy=label_accuracy(best_spins, truth),
                            exact_hit=True,
                            evaluations_used=evaluations,
                        )

    assert best_spins is not None
    direct = float(energies_batch(best_spins, linear, edge_i, edge_j, edge_jij)[0])
    if not np.isclose(direct, best_energy, atol=1e-8):
        raise RuntimeError("SA incremental energy mismatch")
    return TrialResult(
        best_energy=float(best_energy),
        best_accuracy=label_accuracy(best_spins, truth),
        exact_hit=bool(np.isclose(best_energy, exact_energy, atol=1e-8)),
        evaluations_used=evaluations,
    )


def greedy_first_improvement_trial(
    *,
    rng: np.random.Generator,
    budget: int,
    n_nodes: int,
    linear: np.ndarray,
    edge_i: np.ndarray,
    edge_j: np.ndarray,
    edge_jij: np.ndarray,
    adjacency,
    truth: np.ndarray,
    exact_energy: float,
) -> TrialResult:
    """Random-restart, randomized first-improvement one-spin descent.

    Each initial full-energy calculation and each inspected spin-flip delta count
    as one query. The first improving flip in a randomized neighbourhood order is
    accepted immediately. A restart occurs after a complete non-improving scan.
    """
    if budget < 1:
        raise ValueError("budget must be positive")

    evaluations = 0
    best_energy = math.inf
    best_spins = None

    while evaluations < budget:
        spins = random_states(rng, 1, n_nodes)[0]
        energy = float(energies_batch(spins, linear, edge_i, edge_j, edge_jij)[0])
        evaluations += 1
        local_field = initialise_local_field(spins, linear, adjacency)
        if energy < best_energy:
            best_energy = energy
            best_spins = spins.copy()
            if np.isclose(best_energy, exact_energy, atol=1e-8):
                break

        while evaluations < budget:
            accepted = False
            completed_scan = True
            for index in rng.permutation(n_nodes):
                if evaluations >= budget:
                    completed_scan = False
                    break
                index = int(index)
                delta = float(-2.0 * spins[index] * local_field[index])
                evaluations += 1
                if delta < -1e-12:
                    old_spin = int(spins[index])
                    spins[index] = -old_spin
                    energy += delta
                    for neighbour, coupling in adjacency[index]:
                        local_field[neighbour] += -2.0 * coupling * old_spin
                    accepted = True
                    if energy < best_energy:
                        best_energy = float(energy)
                        best_spins = spins.copy()
                        if np.isclose(best_energy, exact_energy, atol=1e-8):
                            return TrialResult(
                                best_energy=float(best_energy),
                                best_accuracy=label_accuracy(best_spins, truth),
                                exact_hit=True,
                                evaluations_used=evaluations,
                            )
                    break
            if not accepted:
                # Full scan without improvement: local optimum. If the budget was
                # exhausted during a partial scan, the outer loop also terminates.
                break
            if not completed_scan and evaluations >= budget:
                break

    assert best_spins is not None
    direct = float(energies_batch(best_spins, linear, edge_i, edge_j, edge_jij)[0])
    if not np.isclose(direct, best_energy, atol=1e-8):
        raise RuntimeError("First-improvement incremental energy mismatch")
    return TrialResult(
        best_energy=float(best_energy),
        best_accuracy=label_accuracy(best_spins, truth),
        exact_hit=bool(np.isclose(best_energy, exact_energy, atol=1e-8)),
        evaluations_used=evaluations,
    )
