#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService

from step11_common import job_metrics_safe
from step12_common import (
    analyze_counts,
    get_result_bit_array,
    rebuild_graph,
    save_grid_counts,
)
from step5_curved_signal_graph import exact_milp


def load_qpy_list(path: Path):
    with path.open('rb') as handle:
        return list(qpy.load(handle))


def counts_from_bound_pub(pub_result, circuit) -> dict[str, int]:
    bit_array = get_result_bit_array(pub_result, circuit)
    shape = tuple(bit_array.shape)
    if shape == ():
        raw = bit_array.get_counts()
    elif shape == (1,):
        raw = bit_array.get_counts(0)
    else:
        raise ValueError(
            f'Expected scalar BitArray for one bound circuit; got shape {shape}'
        )
    return {str(key): int(value) for key, value in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()

    submission = json.loads((args.prepared / 'calibration_submission.json').read_text())
    grid = pd.read_csv(args.prepared / submission['files']['grid'])
    qpy_name = submission.get('bound_qpy_file')
    if not qpy_name:
        raise ValueError(
            'This collector expects the patched bound/folded Step 12 submission.'
        )
    circuits = load_qpy_list(args.prepared / qpy_name)
    if len(circuits) != len(grid):
        raise ValueError(
            f'Expected {len(grid)} bound circuits; found {len(circuits)}'
        )

    service = QiskitRuntimeService()
    job = service.job(str(submission['job_id']))
    print(f"Job {submission['job_id']} status: {job.status()}", flush=True)
    primitive_result = job.result()
    if len(primitive_result) != len(grid):
        raise ValueError(
            f'Expected {len(grid)} PUB results; found {len(primitive_result)}'
        )

    graph, diagnostics = rebuild_graph(int(submission['graph_seed']))
    exact_spins, exact_energy, solver_info = exact_milp(graph)

    rows = []
    counts_by_index = {}
    for position, row in enumerate(grid.itertuples(index=False)):
        index = int(row.grid_index)
        counts = counts_from_bound_pub(primitive_result[position], circuits[position])
        counts_by_index[index] = counts
        metrics = analyze_counts(counts, graph, exact_energy)
        rows.append(
            {
                **row._asdict(),
                **metrics,
                'hardware_minus_ideal_mean_gap': (
                    metrics['mean_normalized_gap']
                    - float(row.ideal_expected_normalized_gap)
                ),
            }
        )

    results = pd.DataFrame(rows).sort_values('grid_index')
    primary = results.sort_values(
        ['mean_energy', 'cvar10_energy', 'best_energy'],
        ascending=True,
    ).iloc[0]
    center = results[results['is_transferred_center'].astype(bool)]
    if len(center) != 1:
        raise ValueError('Expected exactly one transferred-center grid point')
    center = center.iloc[0]

    selected = {
        'selection_rule': (
            'minimum hardware sampled mean energy; ties resolved by CVaR-10% '
            'energy and then best measured energy'
        ),
        'backend': str(submission['backend']),
        'gate_mode': str(submission['gate_mode']),
        'graph_seed_used_for_calibration': int(submission['graph_seed']),
        'shots_per_point': int(submission['shots_per_point']),
        'selected_grid_index': int(primary['grid_index']),
        'selected_gamma': float(primary['gamma']),
        'selected_beta': float(primary['beta']),
        'selected_mean_energy': float(primary['mean_energy']),
        'selected_mean_energy_se': float(primary['mean_energy_se']),
        'selected_mean_normalized_gap': float(primary['mean_normalized_gap']),
        'selected_cvar10_normalized_gap': float(primary['cvar10_normalized_gap']),
        'selected_best_normalized_gap': float(primary['best_normalized_gap']),
        'transferred_center_gamma': float(center['gamma']),
        'transferred_center_beta': float(center['beta']),
        'transferred_center_mean_normalized_gap': float(center['mean_normalized_gap']),
        'improvement_over_center_mean_gap': float(
            center['mean_normalized_gap'] - primary['mean_normalized_gap']
        ),
        'warning': (
            'This is a coarse 25-point calibration on the training instance only. '
            'Confirm leading candidates with higher-shot measurements before freezing '
            'hardware-calibrated angles for held-out seeds.'
        ),
    }

    metadata = {
        'collected_at_utc': datetime.now(timezone.utc).isoformat(),
        'submission': submission,
        'job_metrics': job_metrics_safe(job),
        'exact_energy': exact_energy,
        'exact_solution_label_accuracy': float(np.mean(exact_spins == graph.true_spins)),
        'solver_info': solver_info,
        'graph_diagnostics': diagnostics,
    }
    save_grid_counts(
        args.prepared / 'hardware_calibration_counts.json.gz',
        counts_by_index,
        metadata,
    )
    results.to_csv(args.prepared / 'hardware_calibration_results.csv', index=False)
    (args.prepared / 'selected_hardware_angles.json').write_text(
        json.dumps(selected, indent=2, default=str)
    )
    (args.prepared / 'calibration_job_metrics.json').write_text(
        json.dumps(job_metrics_safe(job), indent=2, default=str)
    )
    print('\nCalibration results ranked by mean energy:')
    columns = [
        'grid_index', 'gamma', 'beta', 'mean_energy', 'mean_energy_se',
        'mean_normalized_gap', 'cvar10_normalized_gap', 'best_normalized_gap',
        'ideal_expected_normalized_gap',
    ]
    print(results.sort_values('mean_energy')[columns].head(10).to_string(index=False))
    print('\nSelected coarse-grid angles:')
    print(json.dumps(selected, indent=2), flush=True)


if __name__ == '__main__':
    main()
