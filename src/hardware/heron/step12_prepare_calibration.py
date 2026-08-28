#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.transpiler import generate_preset_pass_manager
from qiskit_ibm_runtime import QiskitRuntimeService

from step11_common import find_best_circuit, gate_mode_to_fractional
from step12_common import (
    build_angle_grid,
    build_parameterized_qaoa_circuit,
    parameter_values_for_circuit,
    rebuild_graph,
)
from step9_common import normalized_gap
from step9_run_10000 import expected_energy_lightcone_p1
from step5_curved_signal_graph import exact_milp


def active_physical_qubits(circuit) -> list[int]:
    active = set()
    for item in circuit.data:
        if item.operation.name in {'barrier', 'delay'}:
            continue
        for qubit in item.qubits:
            active.add(int(circuit.find_bit(qubit).index))
    return sorted(active)


def two_qubit_depth(circuit) -> int:
    try:
        return int(circuit.depth(lambda item: len(item.qubits) == 2))
    except Exception:
        return -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--backend', default='ibm_aachen')
    parser.add_argument('--gate-mode', choices=['standard', 'fractional'], default='fractional')
    parser.add_argument('--graph-seed', type=int, default=2000)
    parser.add_argument('--gamma-center', type=float, default=0.7771731744054942)
    parser.add_argument('--beta-center', type=float, default=1.1743489498936892)
    parser.add_argument('--gamma-half-width', type=float, default=0.14)
    parser.add_argument('--beta-half-width', type=float, default=0.12)
    parser.add_argument('--points-per-axis', type=int, default=5)
    parser.add_argument('--optimization-level', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    _, preflight_record = find_best_circuit(
        args.preflight,
        backend=args.backend,
        gate_mode=args.gate_mode,
        graph_seed=args.graph_seed,
    )
    transpiler_seed = int(preflight_record['transpiler_seed'])

    service = QiskitRuntimeService()
    backend = service.backend(
        args.backend,
        use_fractional_gates=gate_mode_to_fractional(args.gate_mode),
    )
    status = backend.status()
    if not bool(status.operational):
        raise RuntimeError(f'Backend {args.backend} is not operational')

    graph, diagnostics = rebuild_graph(args.graph_seed)
    _, exact_energy, _ = exact_milp(graph)
    logical = build_parameterized_qaoa_circuit(graph)
    manager = generate_preset_pass_manager(
        optimization_level=args.optimization_level,
        backend=backend,
        seed_transpiler=transpiler_seed,
    )
    isa = manager.run(logical)
    backend.check_faulty(isa)

    operation_counts = {str(k): int(v) for k, v in isa.count_ops().items()}
    if args.gate_mode == 'fractional' and operation_counts.get('rzz', 0) != len(graph.edges):
        raise RuntimeError(
            'Parameterized fractional circuit did not preserve one rzz per graph edge: '
            f"rzz={operation_counts.get('rzz', 0)}, edges={len(graph.edges)}"
        )
    if isa.num_clbits != graph.n_nodes:
        raise RuntimeError('Classical-bit count does not match graph nodes')

    grid = build_angle_grid(
        gamma_center=args.gamma_center,
        beta_center=args.beta_center,
        gamma_half_width=args.gamma_half_width,
        beta_half_width=args.beta_half_width,
        points_per_axis=args.points_per_axis,
    )
    ideal_energies = []
    ideal_gaps = []
    for row in grid.itertuples(index=False):
        energy = expected_energy_lightcone_p1(graph, float(row.gamma), float(row.beta))
        ideal_energies.append(energy)
        ideal_gaps.append(normalized_gap(energy, exact_energy))
    grid['ideal_expected_energy'] = ideal_energies
    grid['ideal_expected_normalized_gap'] = ideal_gaps

    parameter_values, parameter_names = parameter_values_for_circuit(isa, grid)
    grid.to_csv(args.out / 'calibration_grid.csv', index=False)
    np.save(args.out / 'parameter_values.npy', parameter_values)
    with (args.out / 'parameterized_seed2000_isa.qpy').open('wb') as handle:
        qpy.dump(isa, handle)

    active = active_physical_qubits(isa)
    validation = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'backend': args.backend,
        'gate_mode': args.gate_mode,
        'graph_seed': args.graph_seed,
        'backend_operational': bool(status.operational),
        'pending_jobs': int(status.pending_jobs),
        'transpiler_seed': transpiler_seed,
        'points_per_axis': args.points_per_axis,
        'n_parameter_sets': int(len(grid)),
        'parameter_names_in_circuit_order': parameter_names,
        'gamma_center': args.gamma_center,
        'beta_center': args.beta_center,
        'gamma_half_width': args.gamma_half_width,
        'beta_half_width': args.beta_half_width,
        'exact_energy': exact_energy,
        'circuit_qubits': int(isa.num_qubits),
        'circuit_clbits': int(isa.num_clbits),
        'circuit_depth': int(isa.depth()),
        'twoq_depth': two_qubit_depth(isa),
        'operation_counts': operation_counts,
        'active_physical_qubits': int(len(active)),
        'active_physical_qubit_list': active,
        'preflight_record': preflight_record,
        'graph_diagnostics': diagnostics,
        'files': {
            'qpy': 'parameterized_seed2000_isa.qpy',
            'grid': 'calibration_grid.csv',
            'parameter_values': 'parameter_values.npy',
        },
    }
    (args.out / 'calibration_validation.json').write_text(
        json.dumps(validation, indent=2, default=str)
    )
    print(json.dumps(validation, indent=2, default=str))
    print('\nPreparation only. No QPU job was submitted.')


if __name__ == '__main__':
    main()
