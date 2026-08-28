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
from qiskit.transpiler.passes import (
    Optimize1qGatesDecomposition,
    RemoveIdentityEquivalent,
)
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle

from step11_common import gate_mode_to_fractional


def load_single_qpy(path: Path):
    with path.open('rb') as handle:
        circuits = qpy.load(handle)
    if len(circuits) != 1:
        raise ValueError(f'Expected one circuit; found {len(circuits)}')
    return circuits[0]


def bind_and_fold_grid(parameterized_isa, grid: pd.DataFrame, backend):
    """Bind each angle pair, then fold every numeric RZZ angle into [0, pi/2].

    The input circuit is already routed to the selected physical layout.  Folding
    after binding preserves that layout while satisfying the IBM fractional-RZZ
    constraint.  All 25 bound circuits are then submitted as separate PUBs in a
    single Runtime job.
    """
    parameter_by_name = {parameter.name: parameter for parameter in parameterized_isa.parameters}
    if set(parameter_by_name) != {'gamma', 'beta'}:
        raise ValueError(
            f"Expected parameters gamma and beta; found {sorted(parameter_by_name)}"
        )

    fold_manager = PassManager(
        [
            FoldRzzAngle(),
            Optimize1qGatesDecomposition(target=backend.target),
            RemoveIdentityEquivalent(target=backend.target),
        ]
    )

    folded_circuits = []
    summaries = []
    tolerance = 1e-10
    for row in grid.itertuples(index=False):
        bound = parameterized_isa.assign_parameters(
            {
                parameter_by_name['gamma']: float(row.gamma),
                parameter_by_name['beta']: float(row.beta),
            },
            inplace=False,
        )
        folded = fold_manager.run(bound)
        if folded.num_parameters != 0:
            raise RuntimeError(
                f'Grid point {row.grid_index} still contains unbound parameters'
            )
        backend.check_faulty(folded)

        rzz_angles = []
        for instruction in folded.data:
            if instruction.operation.name == 'rzz':
                angle = float(instruction.operation.params[0])
                rzz_angles.append(angle)
                if angle < -tolerance or angle > np.pi / 2 + tolerance:
                    raise RuntimeError(
                        'FoldRzzAngle failed at grid point '
                        f'{row.grid_index}: RZZ angle={angle}'
                    )

        counts = {str(key): int(value) for key, value in folded.count_ops().items()}
        summaries.append(
            {
                'grid_index': int(row.grid_index),
                'gamma': float(row.gamma),
                'beta': float(row.beta),
                'depth': int(folded.depth()),
                'twoq_count': int(folded.num_nonlocal_gates()),
                'rzz_count': int(counts.get('rzz', 0)),
                'cz_count': int(counts.get('cz', 0)),
                'x_count': int(counts.get('x', 0)),
                'minimum_rzz_angle': float(min(rzz_angles)) if rzz_angles else np.nan,
                'maximum_rzz_angle': float(max(rzz_angles)) if rzz_angles else np.nan,
            }
        )
        folded_circuits.append(folded)

    return folded_circuits, pd.DataFrame(summaries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--shots-per-point', type=int, default=500)
    parser.add_argument('--dd', choices=['none', 'XX', 'XpXm', 'XY4'], default='XY4')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    if args.shots_per_point < 1:
        raise ValueError('--shots-per-point must be positive')

    validation = json.loads((args.prepared / 'calibration_validation.json').read_text())
    backend_name = str(validation['backend'])
    gate_mode = str(validation['gate_mode'])
    parameterized_isa = load_single_qpy(args.prepared / validation['files']['qpy'])
    grid = pd.read_csv(args.prepared / validation['files']['grid'])

    service = QiskitRuntimeService()
    backend = service.backend(
        backend_name,
        use_fractional_gates=gate_mode_to_fractional(gate_mode),
    )
    status = backend.status()
    if not bool(status.operational):
        raise RuntimeError(f'Backend {backend_name} is not operational')
    backend.check_faulty(parameterized_isa)

    folded_circuits, fold_summary = bind_and_fold_grid(
        parameterized_isa,
        grid,
        backend,
    )
    bound_qpy_name = 'bound_folded_grid_isa.qpy'
    with (args.prepared / bound_qpy_name).open('wb') as handle:
        qpy.dump(folded_circuits, handle)
    fold_summary.to_csv(args.prepared / 'bound_folded_grid_summary.csv', index=False)

    submission_preview = {
        **validation,
        'validated_for_submission_at_utc': datetime.now(timezone.utc).isoformat(),
        'current_pending_jobs': int(status.pending_jobs),
        'shots_per_point': int(args.shots_per_point),
        'total_requested_shots': int(args.shots_per_point * len(grid)),
        'dd': args.dd,
        'execution_mode': '25_bound_folded_circuits_in_one_runtime_job',
        'n_pubs': int(len(folded_circuits)),
        'bound_qpy_file': bound_qpy_name,
        'fold_summary_file': 'bound_folded_grid_summary.csv',
        'minimum_folded_rzz_angle': float(fold_summary['minimum_rzz_angle'].min()),
        'maximum_folded_rzz_angle': float(fold_summary['maximum_rzz_angle'].max()),
        'all_rzz_angles_within_fractional_range': bool(
            (fold_summary['minimum_rzz_angle'] >= -1e-10).all()
            and (fold_summary['maximum_rzz_angle'] <= np.pi / 2 + 1e-10).all()
        ),
    }
    (args.prepared / 'calibration_submission_preview.json').write_text(
        json.dumps(submission_preview, indent=2, default=str)
    )
    print(json.dumps(submission_preview, indent=2, default=str), flush=True)
    if not args.submit:
        print(
            '\nValidation only. All angle pairs were bound and folded locally. '
            'Re-run with --submit to use QPU time.',
            flush=True,
        )
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
        'DMRI_CROSSING_QAOA',
        'hardware_angle_grid',
        'seed_2000',
        gate_mode,
        f'dd_{args.dd}',
        'bound_folded_rzz',
    ]

    # One Runtime job containing 25 separately bound/folded PUBs.
    job = sampler.run(
        [(circuit,) for circuit in folded_circuits],
        shots=int(args.shots_per_point),
    )
    submission = {
        **submission_preview,
        'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
        'job_id': job.job_id(),
        'initial_status': str(job.status()),
    }
    (args.prepared / 'calibration_submission.json').write_text(
        json.dumps(submission, indent=2, default=str)
    )
    print('\nSubmitted calibration scan:')
    print(json.dumps(submission, indent=2, default=str), flush=True)


if __name__ == '__main__':
    main()
