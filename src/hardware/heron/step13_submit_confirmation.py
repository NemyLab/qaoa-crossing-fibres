#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveIdentityEquivalent
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

from step11_common import gate_mode_to_fractional
from step12_common import rebuild_graph
from step5_curved_signal_graph import exact_milp
from step9_common import normalized_gap
from step9_run_10000 import expected_energy_lightcone_p1


def load_single_qpy(path: Path):
    with path.open('rb') as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise ValueError(f'Expected one circuit; found {len(circuits)}')
    return circuits[0]


def fit_weighted_quadratic(results: pd.DataFrame) -> tuple[float, float, np.ndarray]:
    g = results['gamma'].to_numpy(float)
    b = results['beta'].to_numpy(float)
    y = results['mean_energy'].to_numpy(float)
    se = results['mean_energy_se'].to_numpy(float)
    x = np.column_stack([np.ones(len(results)), g, b, g*g, b*b, g*b])
    weights = 1.0 / np.square(se)
    xtwx = x.T @ (weights[:, None] * x)
    xtwy = x.T @ (weights * y)
    coefficients = np.linalg.solve(xtwx, xtwy)
    _, cg, cb, cgg, cbb, cgb = coefficients
    hessian = np.array([[2.0*cgg, cgb], [cgb, 2.0*cbb]], dtype=float)
    stationary = np.linalg.solve(hessian, -np.array([cg, cb], dtype=float))
    if np.any(np.linalg.eigvalsh(hessian) <= 0):
        raise ValueError('The fitted quadratic does not have a local minimum')
    return float(stationary[0]), float(stationary[1]), coefficients


def candidate_table(results: pd.DataFrame) -> pd.DataFrame:
    required = {3, 7, 8, 12, 13}
    available = set(results['grid_index'].astype(int))
    if not required.issubset(available):
        raise ValueError(f'Missing required Step 12 grid indices: {sorted(required-available)}')
    indexed = results.set_index(results['grid_index'].astype(int))
    fitted_gamma, fitted_beta, coefficients = fit_weighted_quadratic(results)

    candidates = [
        ('coarse_grid_winner', float(indexed.loc[3, 'gamma']), float(indexed.loc[3, 'beta']), 3),
        ('quadratic_fitted_optimum', fitted_gamma, fitted_beta, None),
        ('same_beta_middle_gamma', float(indexed.loc[8, 'gamma']), float(indexed.loc[8, 'beta']), 8),
        ('ideal_grid_best', float(indexed.loc[7, 'gamma']), float(indexed.loc[7, 'beta']), 7),
        ('coarse_grid_runner_up', float(indexed.loc[13, 'gamma']), float(indexed.loc[13, 'beta']), 13),
        ('transferred_center', float(indexed.loc[12, 'gamma']), float(indexed.loc[12, 'beta']), 12),
    ]
    rows = []
    seen = set()
    for label, gamma, beta, source_index in candidates:
        key = (round(gamma, 12), round(beta, 12))
        if key in seen:
            continue
        seen.add(key)
        rows.append({
            'candidate_index': len(rows),
            'candidate_label': label,
            'gamma': gamma,
            'beta': beta,
            'source_step12_grid_index': source_index,
        })
    table = pd.DataFrame(rows)
    table.attrs['quadratic_coefficients'] = coefficients.tolist()
    return table


def bind_and_fold(parameterized_isa, candidates: pd.DataFrame, backend):
    parameter_by_name = {parameter.name: parameter for parameter in parameterized_isa.parameters}
    if set(parameter_by_name) != {'gamma', 'beta'}:
        raise ValueError(f'Expected gamma and beta; found {sorted(parameter_by_name)}')
    manager = PassManager([
        FoldRzzAngle(),
        Optimize1qGatesDecomposition(target=backend.target),
        RemoveIdentityEquivalent(target=backend.target),
    ])
    circuits = []
    rows = []
    tolerance = 1e-10
    for row in candidates.itertuples(index=False):
        bound = parameterized_isa.assign_parameters({
            parameter_by_name['gamma']: float(row.gamma),
            parameter_by_name['beta']: float(row.beta),
        }, inplace=False)
        folded = manager.run(bound)
        backend.check_faulty(folded)
        angles = []
        for instruction in folded.data:
            if instruction.operation.name == 'rzz':
                angle = float(instruction.operation.params[0])
                angles.append(angle)
                if angle < -tolerance or angle > np.pi/2+tolerance:
                    raise RuntimeError(f'Invalid folded RZZ angle {angle}')
        ops = {str(k): int(v) for k, v in folded.count_ops().items()}
        rows.append({
            'candidate_index': int(row.candidate_index),
            'candidate_label': str(row.candidate_label),
            'gamma': float(row.gamma),
            'beta': float(row.beta),
            'depth': int(folded.depth()),
            'twoq_count': int(folded.num_nonlocal_gates()),
            'rzz_count': int(ops.get('rzz', 0)),
            'cz_count': int(ops.get('cz', 0)),
            'minimum_rzz_angle': float(min(angles)) if angles else np.nan,
            'maximum_rzz_angle': float(max(angles)) if angles else np.nan,
        })
        circuits.append(folded)
    return circuits, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--step12-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--shots-per-point', type=int, default=2000)
    parser.add_argument('--dd', choices=['none', 'XX', 'XpXm', 'XY4'], default='XY4')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    validation = json.loads((args.step12_dir/'calibration_validation.json').read_text())
    results = pd.read_csv(args.step12_dir/'hardware_calibration_results.csv')
    candidates = candidate_table(results)

    graph, _ = rebuild_graph(int(validation['graph_seed']))
    _, exact_energy, _ = exact_milp(graph)
    ideal_energies = []
    for row in candidates.itertuples(index=False):
        energy = expected_energy_lightcone_p1(graph, float(row.gamma), float(row.beta))
        ideal_energies.append(energy)
    candidates['ideal_expected_energy'] = ideal_energies
    candidates['ideal_expected_normalized_gap'] = [normalized_gap(e, exact_energy) for e in ideal_energies]
    candidates.to_csv(args.out/'confirmation_candidates.csv', index=False)

    parameterized = load_single_qpy(args.step12_dir/validation['files']['qpy'])
    service = QiskitRuntimeService()
    backend = service.backend(
        str(validation['backend']),
        use_fractional_gates=gate_mode_to_fractional(str(validation['gate_mode'])),
    )
    status = backend.status()
    if not bool(status.operational):
        raise RuntimeError(f"Backend {validation['backend']} is not operational")
    circuits, fold_summary = bind_and_fold(parameterized, candidates, backend)
    fold_summary.to_csv(args.out/'confirmation_fold_summary.csv', index=False)
    with (args.out/'confirmation_bound_folded_isa.qpy').open('wb') as handle:
        qpy.dump(circuits, handle)

    preview = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'backend': str(validation['backend']),
        'gate_mode': str(validation['gate_mode']),
        'graph_seed': int(validation['graph_seed']),
        'shots_per_point': int(args.shots_per_point),
        'n_candidates': int(len(candidates)),
        'total_requested_shots': int(args.shots_per_point*len(candidates)),
        'dd': args.dd,
        'current_pending_jobs': int(status.pending_jobs),
        'execution_mode': 'bound_folded_confirmation_circuits_in_one_runtime_job',
        'all_rzz_angles_within_fractional_range': bool(
            (fold_summary['minimum_rzz_angle'] >= -1e-10).all()
            and (fold_summary['maximum_rzz_angle'] <= np.pi/2+1e-10).all()
        ),
        'minimum_folded_rzz_angle': float(fold_summary['minimum_rzz_angle'].min()),
        'maximum_folded_rzz_angle': float(fold_summary['maximum_rzz_angle'].max()),
        'files': {
            'candidates': 'confirmation_candidates.csv',
            'qpy': 'confirmation_bound_folded_isa.qpy',
            'fold_summary': 'confirmation_fold_summary.csv',
        },
    }
    (args.out/'confirmation_submission_preview.json').write_text(json.dumps(preview, indent=2))
    print(json.dumps(preview, indent=2), flush=True)
    print('\nCandidates:')
    print(candidates.to_string(index=False), flush=True)
    if not args.submit:
        print('\nValidation only. Re-run with --submit to use QPU time.', flush=True)
        return

    sampler = Sampler(mode=backend)
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    if args.dd == 'none':
        sampler.options.dynamical_decoupling.enable = False
    else:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = args.dd
    sampler.options.environment.job_tags = [
        'DMRI_CROSSING_QAOA', 'hardware_angle_confirmation', 'seed_2000',
        str(validation['gate_mode']), f'dd_{args.dd}', 'bound_folded_rzz'
    ]
    job = sampler.run([(circuit,) for circuit in circuits], shots=int(args.shots_per_point))
    submission = {
        **preview,
        'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
        'job_id': job.job_id(),
        'initial_status': str(job.status()),
    }
    (args.out/'confirmation_submission.json').write_text(json.dumps(submission, indent=2))
    print('\nSubmitted confirmation job:')
    print(json.dumps(submission, indent=2), flush=True)


if __name__ == '__main__':
    main()
