#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.transpiler import PassManager, generate_preset_pass_manager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveIdentityEquivalent
from qiskit_ibm_runtime import QiskitRuntimeService
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

from step11_common import find_best_circuit, gate_mode_to_fractional
from step12_common import build_parameterized_qaoa_circuit, rebuild_graph
from step5_curved_signal_graph import exact_milp
from step9_common import normalized_gap
from step9_run_10000 import expected_energy_lightcone_p1

TRANSFERRED_GAMMA = 0.7771731744054942
TRANSFERRED_BETA = 1.1743489498936892


def two_qubit_depth(circuit) -> int:
    try:
        return int(circuit.depth(lambda item: len(item.qubits) == 2))
    except Exception:
        return -1


def active_physical_qubits(circuit) -> list[int]:
    active = set()
    for item in circuit.data:
        if item.operation.name in {'barrier', 'delay'}:
            continue
        for qubit in item.qubits:
            active.add(int(circuit.find_bit(qubit).index))
    return sorted(active)


def bind_and_fold(parameterized_isa, *, gamma: float, beta: float, backend):
    parameter_by_name = {parameter.name: parameter for parameter in parameterized_isa.parameters}
    if set(parameter_by_name) != {'gamma', 'beta'}:
        raise ValueError(f'Expected gamma and beta; found {sorted(parameter_by_name)}')
    bound = parameterized_isa.assign_parameters(
        {
            parameter_by_name['gamma']: float(gamma),
            parameter_by_name['beta']: float(beta),
        },
        inplace=False,
    )
    manager = PassManager([
        FoldRzzAngle(),
        Optimize1qGatesDecomposition(target=backend.target),
        RemoveIdentityEquivalent(target=backend.target),
    ])
    folded = manager.run(bound)
    backend.check_faulty(folded)

    angles = []
    for instruction in folded.data:
        if instruction.operation.name == 'rzz':
            value = float(instruction.operation.params[0])
            angles.append(value)
            if value < -1e-10 or value > np.pi / 2 + 1e-10:
                raise RuntimeError(f'Invalid folded RZZ angle {value}')
    return folded, angles


def strategy_table(angle_json: dict) -> pd.DataFrame:
    return pd.DataFrame([
        {
            'angle_strategy': 'hardware_calibrated',
            'gamma': float(angle_json['selected_gamma']),
            'beta': float(angle_json['selected_beta']),
        },
        {
            'angle_strategy': 'transferred_center',
            'gamma': TRANSFERRED_GAMMA,
            'beta': TRANSFERRED_BETA,
        },
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--preflight', type=Path, required=True)
    parser.add_argument('--angles', type=Path, required=True)
    parser.add_argument('--backend', default='ibm_aachen')
    parser.add_argument('--gate-mode', choices=['standard', 'fractional'], default='fractional')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2001, 2002, 2003])
    parser.add_argument('--optimization-level', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    if set(args.seeds) != {2001, 2002, 2003}:
        raise ValueError('For the preregistered held-out test, use exactly seeds 2001 2002 2003')

    final_angles = json.loads(args.angles.read_text())
    if int(final_angles['graph_seed_used_for_calibration']) != 2000:
        raise ValueError('Hardware angles must have been calibrated on seed 2000')
    strategies = strategy_table(final_angles)

    service = QiskitRuntimeService()
    backend = service.backend(
        args.backend,
        use_fractional_gates=gate_mode_to_fractional(args.gate_mode),
    )
    status = backend.status()
    if not bool(status.operational):
        raise RuntimeError(f'Backend {args.backend} is not operational')

    circuits = []
    candidate_rows = []
    seed_validation = []

    for seed in sorted(args.seeds):
        _, preflight_record = find_best_circuit(
            args.preflight,
            backend=args.backend,
            gate_mode=args.gate_mode,
            graph_seed=seed,
        )
        transpiler_seed = int(preflight_record['transpiler_seed'])
        graph, diagnostics = rebuild_graph(seed)
        exact_spins, exact_energy, solver_info = exact_milp(graph)

        logical = build_parameterized_qaoa_circuit(graph)
        manager = generate_preset_pass_manager(
            optimization_level=args.optimization_level,
            backend=backend,
            seed_transpiler=transpiler_seed,
        )
        parameterized_isa = manager.run(logical)
        backend.check_faulty(parameterized_isa)
        ops = {str(k): int(v) for k, v in parameterized_isa.count_ops().items()}
        if args.gate_mode == 'fractional' and ops.get('rzz', 0) != len(graph.edges):
            raise RuntimeError(
                f'Seed {seed}: expected {len(graph.edges)} rzz gates; found {ops.get("rzz", 0)}'
            )

        seed_validation.append({
            'seed': seed,
            'transpiler_seed': transpiler_seed,
            'exact_energy': float(exact_energy),
            'exact_solution_label_accuracy': float(np.mean(exact_spins == graph.true_spins)),
            'n_nodes': int(graph.n_nodes),
            'n_edges': int(len(graph.edges)),
            'parameterized_depth': int(parameterized_isa.depth()),
            'parameterized_twoq_depth': two_qubit_depth(parameterized_isa),
            'parameterized_operation_counts': ops,
            'active_physical_qubits': active_physical_qubits(parameterized_isa),
            'preflight_record': preflight_record,
            'graph_diagnostics': diagnostics,
            'solver_info': solver_info,
        })

        for strategy in strategies.itertuples(index=False):
            ideal_energy = expected_energy_lightcone_p1(
                graph, float(strategy.gamma), float(strategy.beta)
            )
            folded, rzz_angles = bind_and_fold(
                parameterized_isa,
                gamma=float(strategy.gamma),
                beta=float(strategy.beta),
                backend=backend,
            )
            folded_ops = {str(k): int(v) for k, v in folded.count_ops().items()}
            pub_index = len(circuits)
            circuits.append(folded)
            candidate_rows.append({
                'pub_index': pub_index,
                'seed': seed,
                'angle_strategy': str(strategy.angle_strategy),
                'gamma': float(strategy.gamma),
                'beta': float(strategy.beta),
                'exact_energy': float(exact_energy),
                'exact_solution_label_accuracy': float(np.mean(exact_spins == graph.true_spins)),
                'ideal_expected_energy': float(ideal_energy),
                'ideal_expected_normalized_gap': float(normalized_gap(ideal_energy, exact_energy)),
                'depth': int(folded.depth()),
                'twoq_depth': two_qubit_depth(folded),
                'twoq_count': int(folded.num_nonlocal_gates()),
                'rzz_count': int(folded_ops.get('rzz', 0)),
                'cz_count': int(folded_ops.get('cz', 0)),
                'minimum_rzz_angle': float(min(rzz_angles)) if rzz_angles else np.nan,
                'maximum_rzz_angle': float(max(rzz_angles)) if rzz_angles else np.nan,
            })

    candidates = pd.DataFrame(candidate_rows)
    candidates.to_csv(args.out / 'heldout_candidates.csv', index=False)
    with (args.out / 'heldout_bound_folded_isa.qpy').open('wb') as handle:
        qpy.dump(circuits, handle)

    validation = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'backend': args.backend,
        'gate_mode': args.gate_mode,
        'backend_operational': bool(status.operational),
        'pending_jobs': int(status.pending_jobs),
        'heldout_seeds': sorted(args.seeds),
        'training_seed_excluded': 2000,
        'n_angle_strategies': 2,
        'n_pubs': int(len(candidates)),
        'selection_rule_frozen_before_heldout_test': (
            'hardware_calibrated angles are the independent-confirmation winner from seed 2000; '
            'transferred_center is retained as the paired comparator'
        ),
        'primary_heldout_endpoint': (
            'mean normalized energy gap across seeds 2001-2003 at 10000 shots per seed and strategy'
        ),
        'all_rzz_angles_within_fractional_range': bool(
            (candidates['minimum_rzz_angle'] >= -1e-10).all()
            and (candidates['maximum_rzz_angle'] <= np.pi / 2 + 1e-10).all()
        ),
        'minimum_folded_rzz_angle': float(candidates['minimum_rzz_angle'].min()),
        'maximum_folded_rzz_angle': float(candidates['maximum_rzz_angle'].max()),
        'hardware_calibrated_angles': {
            'gamma': float(final_angles['selected_gamma']),
            'beta': float(final_angles['selected_beta']),
        },
        'transferred_center_angles': {
            'gamma': TRANSFERRED_GAMMA,
            'beta': TRANSFERRED_BETA,
        },
        'seed_validation': seed_validation,
        'files': {
            'candidates': 'heldout_candidates.csv',
            'qpy': 'heldout_bound_folded_isa.qpy',
        },
    }
    (args.out / 'heldout_validation.json').write_text(
        json.dumps(validation, indent=2, default=str)
    )
    print(json.dumps(validation, indent=2, default=str))
    print('\nPreparation only. No QPU job was submitted.')


if __name__ == '__main__':
    main()
