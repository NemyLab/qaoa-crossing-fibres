#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler

from step11_common import gate_mode_to_fractional
from step15_prepare_replication import REPLICATION_SEEDS


def load_qpy_list(path: Path):
    with path.open('rb') as handle:
        return list(qpy.load(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--shots-per-pub', type=int, default=5000)
    parser.add_argument('--dd', choices=['none', 'XX', 'XpXm', 'XY4'], default='XY4')
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()

    validation = json.loads((args.prepared / 'replication_validation.json').read_text())
    candidates = pd.read_csv(args.prepared / validation['files']['candidates'])
    circuits = load_qpy_list(args.prepared / validation['files']['qpy'])
    if len(circuits) != len(candidates):
        raise ValueError('Circuit/candidate count mismatch')
    if tuple(sorted(candidates['seed'].astype(int).unique())) != REPLICATION_SEEDS:
        raise ValueError('Replication candidates do not contain the frozen seed cohort')
    if set(candidates['angle_strategy'].astype(str)) != {
        'hardware_calibrated', 'transferred_center'
    }:
        raise ValueError('Expected exactly the two frozen angle strategies')

    service = QiskitRuntimeService()
    backend = service.backend(
        str(validation['backend']),
        use_fractional_gates=gate_mode_to_fractional(str(validation['gate_mode'])),
    )
    status = backend.status()
    if not bool(status.operational):
        raise RuntimeError(f"Backend {validation['backend']} is not operational")
    for circuit in circuits:
        backend.check_faulty(circuit)

    preview = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'backend': str(validation['backend']),
        'gate_mode': str(validation['gate_mode']),
        'replication_seeds': list(REPLICATION_SEEDS),
        'angle_strategies': ['hardware_calibrated', 'transferred_center'],
        'n_pubs': int(len(circuits)),
        'shots_per_pub': int(args.shots_per_pub),
        'total_requested_shots': int(args.shots_per_pub * len(circuits)),
        'dd': args.dd,
        'current_pending_jobs': int(status.pending_jobs),
        'execution_mode': '24_bound_folded_replication_circuits_in_one_runtime_job',
        'files': validation['files'],
    }
    (args.prepared / 'replication_submission_preview.json').write_text(
        json.dumps(preview, indent=2)
    )
    print(json.dumps(preview, indent=2), flush=True)
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
        'DMRI_CROSSING_QAOA',
        'independent_replication',
        'seeds_2004_2015',
        str(validation['gate_mode']),
        f'dd_{args.dd}',
        'bound_folded_rzz',
    ]
    job = sampler.run([(circuit,) for circuit in circuits], shots=int(args.shots_per_pub))
    submission = {
        **preview,
        'submitted_at_utc': datetime.now(timezone.utc).isoformat(),
        'job_id': job.job_id(),
        'initial_status': str(job.status()),
    }
    (args.prepared / 'replication_submission.json').write_text(
        json.dumps(submission, indent=2)
    )
    print('\nSubmitted replication job:')
    print(json.dumps(submission, indent=2), flush=True)


if __name__ == '__main__':
    main()
