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

from step5_curved_signal_graph import exact_milp
from step9_common import normalized_gap
from step9_run_10000 import expected_energy_lightcone_p1
from step10_transpile_preflight import (
    active_physical_qubits,
    count_two_qubit_operations,
    target_error_metrics,
    two_qubit_depth,
)
from step11_common import gate_mode_to_fractional
from step12_common import build_parameterized_qaoa_circuit, rebuild_graph

TRANSFERRED_GAMMA = 0.7771731744054942
TRANSFERRED_BETA = 1.1743489498936892
REPLICATION_SEEDS = tuple(range(2004, 2016))


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


def bind_and_fold(parameterized_isa, *, gamma: float, beta: float, backend):
    parameters = {parameter.name: parameter for parameter in parameterized_isa.parameters}
    if set(parameters) != {'gamma', 'beta'}:
        raise ValueError(f'Expected gamma and beta; found {sorted(parameters)}')
    bound = parameterized_isa.assign_parameters(
        {parameters['gamma']: float(gamma), parameters['beta']: float(beta)},
        inplace=False,
    )
    folded = PassManager([
        FoldRzzAngle(),
        Optimize1qGatesDecomposition(target=backend.target),
        RemoveIdentityEquivalent(target=backend.target),
    ]).run(bound)
    backend.check_faulty(folded)

    angles: list[float] = []
    for item in folded.data:
        if item.operation.name == 'rzz':
            value = float(item.operation.params[0])
            angles.append(value)
            if value < -1e-10 or value > np.pi / 2 + 1e-10:
                raise RuntimeError(f'Invalid folded RZZ angle {value}')
    return folded, angles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--angles', type=Path, required=True)
    parser.add_argument('--backend', default='ibm_aachen')
    parser.add_argument('--gate-mode', choices=['standard', 'fractional'], default='fractional')
    parser.add_argument('--seeds', type=int, nargs='+', default=list(REPLICATION_SEEDS))
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--optimization-level', type=int, default=3)
    parser.add_argument('--transpiler-seed-base', type=int, default=151_500_000)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = tuple(sorted(set(int(seed) for seed in args.seeds)))
    if seeds != REPLICATION_SEEDS:
        raise ValueError(
            f'For the frozen replication cohort, use exactly seeds {list(REPLICATION_SEEDS)}'
        )
    if args.trials < 1:
        raise ValueError('--trials must be positive')

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
    if args.gate_mode == 'fractional' and 'rzz' not in set(map(str, backend.operation_names)):
        raise RuntimeError(f'Backend {args.backend} fractional target does not expose rzz')

    circuits = []
    candidate_rows: list[dict] = []
    trial_rows: list[dict] = []
    seed_validation: list[dict] = []

    for seed in seeds:
        print(f'Preparing replication seed {seed}', flush=True)
        graph, diagnostics = rebuild_graph(seed)
        exact_spins, exact_energy, solver_info = exact_milp(graph)
        logical = build_parameterized_qaoa_circuit(graph)

        best = None
        for trial in range(args.trials):
            transpiler_seed = args.transpiler_seed_base + 1000 * seed + trial
            try:
                manager = generate_preset_pass_manager(
                    optimization_level=args.optimization_level,
                    backend=backend,
                    seed_transpiler=transpiler_seed,
                )
                isa = manager.run(logical)
                backend.check_faulty(isa)
                ops = {str(k): int(v) for k, v in isa.count_ops().items()}
                if args.gate_mode == 'fractional' and ops.get('rzz', 0) != len(graph.edges):
                    raise RuntimeError(
                        f'Expected {len(graph.edges)} rzz gates; found {ops.get("rzz", 0)}'
                    )
                twoq_count, twoq_breakdown = count_two_qubit_operations(isa)
                errors = target_error_metrics(backend, isa)
                record = {
                    'seed': seed,
                    'trial': trial,
                    'transpiler_seed': transpiler_seed,
                    'success': True,
                    'error': '',
                    'depth': int(isa.depth()),
                    'twoq_depth': int(two_qubit_depth(isa)),
                    'twoq_count': int(twoq_count),
                    'twoq_gate_breakdown': json.dumps(twoq_breakdown, sort_keys=True),
                    'active_physical_qubits': len(active_physical_qubits(isa)),
                    'active_physical_qubit_list': ';'.join(map(str, active_physical_qubits(isa))),
                    **errors,
                }
                score = (
                    float(record['error_proxy_negative_log_success']),
                    int(record['twoq_count']),
                    int(record['twoq_depth']),
                    int(record['depth']),
                )
                if best is None or score < best[0]:
                    best = (score, isa, record)
            except Exception as exc:
                record = {
                    'seed': seed,
                    'trial': trial,
                    'transpiler_seed': transpiler_seed,
                    'success': False,
                    'error': repr(exc),
                }
            trial_rows.append(record)

        if best is None:
            raise RuntimeError(f'No successful transpilation for seed {seed}')
        _, parameterized_isa, selected_record = best

        seed_validation.append({
            'seed': seed,
            'selected_transpilation': selected_record,
            'exact_energy': float(exact_energy),
            'exact_solution_label_accuracy': float(np.mean(exact_spins == graph.true_spins)),
            'n_nodes': int(graph.n_nodes),
            'n_edges': int(len(graph.edges)),
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
            twoq_count, _ = count_two_qubit_operations(folded)
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
                'twoq_depth': int(two_qubit_depth(folded)),
                'twoq_count': int(twoq_count),
                'rzz_count': int(folded_ops.get('rzz', 0)),
                'cz_count': int(folded_ops.get('cz', 0)),
                'minimum_rzz_angle': float(min(rzz_angles)) if rzz_angles else np.nan,
                'maximum_rzz_angle': float(max(rzz_angles)) if rzz_angles else np.nan,
                'selected_error_proxy': float(selected_record['error_proxy_negative_log_success']),
                'selected_mean_twoq_error': float(selected_record['mean_twoq_error']),
                'selected_mean_readout_error': float(selected_record['mean_readout_error']),
            })

    trials = pd.DataFrame(trial_rows)
    candidates = pd.DataFrame(candidate_rows)
    trials.to_csv(args.out / 'replication_transpile_trials.csv', index=False)
    candidates.to_csv(args.out / 'replication_candidates.csv', index=False)
    with (args.out / 'replication_bound_folded_isa.qpy').open('wb') as handle:
        qpy.dump(circuits, handle)

    validation = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'backend': args.backend,
        'gate_mode': args.gate_mode,
        'backend_operational': bool(status.operational),
        'pending_jobs': int(status.pending_jobs),
        'replication_seeds': list(seeds),
        'previously_used_seeds_excluded': [2000, 2001, 2002, 2003],
        'n_replication_seeds': len(seeds),
        'n_angle_strategies': 2,
        'n_pubs': int(len(candidates)),
        'transpilation_trials_per_seed': int(args.trials),
        'selection_rule_frozen_before_replication': (
            'hardware_calibrated angles are fixed from seed 2000; transferred_center is the paired comparator'
        ),
        'primary_replication_endpoint': (
            'paired mean normalized energy-gap difference across new seeds 2004-2015; negative favors hardware calibration'
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
            'candidates': 'replication_candidates.csv',
            'qpy': 'replication_bound_folded_isa.qpy',
            'transpile_trials': 'replication_transpile_trials.csv',
        },
    }
    (args.out / 'replication_validation.json').write_text(
        json.dumps(validation, indent=2, default=str)
    )
    print(json.dumps(validation, indent=2, default=str))
    print('\nPreparation only. No QPU job was submitted.')


if __name__ == '__main__':
    main()
