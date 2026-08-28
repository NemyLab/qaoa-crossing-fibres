#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gzip
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from qiskit import qpy
from qiskit_ibm_runtime import QiskitRuntimeService

from step11_common import job_metrics_safe
from step12_common import analyze_counts, get_result_bit_array, rebuild_graph
from step5_curved_signal_graph import exact_milp


def load_qpy_list(path: Path):
    with path.open('rb') as handle:
        return list(qpy.load(handle))


def counts_from_pub(pub_result, circuit) -> dict[str, int]:
    bit_array = get_result_bit_array(pub_result, circuit)
    shape = tuple(bit_array.shape)
    if shape == ():
        raw = bit_array.get_counts()
    elif shape == (1,):
        raw = bit_array.get_counts(0)
    else:
        raise ValueError(f'Expected scalar BitArray; got {shape}')
    return {str(k): int(v) for k, v in raw.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()

    submission = json.loads((args.out/'confirmation_submission.json').read_text())
    candidates = pd.read_csv(args.out/submission['files']['candidates'])
    circuits = load_qpy_list(args.out/submission['files']['qpy'])
    if len(circuits) != len(candidates):
        raise ValueError('Circuit/candidate count mismatch')

    service = QiskitRuntimeService()
    job = service.job(str(submission['job_id']))
    print(f"Job {submission['job_id']} status: {job.status()}", flush=True)
    result = job.result()
    if len(result) != len(candidates):
        raise ValueError('PUB result/candidate count mismatch')

    graph, diagnostics = rebuild_graph(int(submission['graph_seed']))
    exact_spins, exact_energy, solver_info = exact_milp(graph)
    rows = []
    counts_payload = {}
    for position, row in enumerate(candidates.itertuples(index=False)):
        counts = counts_from_pub(result[position], circuits[position])
        counts_payload[str(int(row.candidate_index))] = counts
        metrics = analyze_counts(counts, graph, exact_energy)
        rows.append({**row._asdict(), **metrics})
    frame = pd.DataFrame(rows).sort_values('candidate_index')

    winner = frame.sort_values(['mean_energy','cvar10_energy','best_energy']).iloc[0]
    center = frame[frame['candidate_label'] == 'transferred_center']
    if len(center) != 1:
        raise ValueError('Expected one transferred_center candidate')
    center = center.iloc[0]
    difference = float(center['mean_energy'] - winner['mean_energy'])
    difference_se = float(np.sqrt(center['mean_energy_se']**2 + winner['mean_energy_se']**2))
    ci_low = difference - 1.96*difference_se
    ci_high = difference + 1.96*difference_se

    final = {
        'selection_rule': 'lowest mean energy in independent higher-shot confirmation; ties by CVaR-10% then best energy',
        'backend': submission['backend'],
        'graph_seed_used_for_calibration': int(submission['graph_seed']),
        'shots_per_candidate': int(submission['shots_per_point']),
        'selected_candidate_index': int(winner['candidate_index']),
        'selected_candidate_label': str(winner['candidate_label']),
        'selected_gamma': float(winner['gamma']),
        'selected_beta': float(winner['beta']),
        'selected_mean_energy': float(winner['mean_energy']),
        'selected_mean_energy_se': float(winner['mean_energy_se']),
        'selected_mean_normalized_gap': float(winner['mean_normalized_gap']),
        'selected_cvar10_normalized_gap': float(winner['cvar10_normalized_gap']),
        'selected_best_normalized_gap': float(winner['best_normalized_gap']),
        'selected_best_label_accuracy': float(winner['best_label_accuracy']),
        'transferred_center_mean_normalized_gap': float(center['mean_normalized_gap']),
        'energy_improvement_over_center': difference,
        'energy_improvement_over_center_se': difference_se,
        'energy_improvement_over_center_95ci': [float(ci_low), float(ci_high)],
        'confirmation_supports_improvement_over_center_at_95pct': bool(ci_low > 0),
        'note': 'Seeds 2001-2003 remain held out and were not used for angle selection.',
    }

    frame.to_csv(args.out/'hardware_angle_confirmation_results.csv', index=False)
    (args.out/'final_hardware_calibrated_angles.json').write_text(json.dumps(final, indent=2))
    metrics = job_metrics_safe(job)
    (args.out/'confirmation_job_metrics.json').write_text(json.dumps(metrics, indent=2, default=str))
    with gzip.open(args.out/'hardware_angle_confirmation_counts.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump({
            'counts_by_candidate_index': counts_payload,
            'metadata': {
                'collected_at_utc': datetime.now(timezone.utc).isoformat(),
                'submission': submission,
                'job_metrics': metrics,
                'exact_energy': exact_energy,
                'exact_solution_label_accuracy': float(np.mean(exact_spins == graph.true_spins)),
                'solver_info': solver_info,
                'graph_diagnostics': diagnostics,
            },
        }, handle)

    print('\nConfirmation results ranked by mean energy:')
    cols = ['candidate_label','gamma','beta','mean_energy','mean_energy_se','mean_normalized_gap','cvar10_normalized_gap','best_normalized_gap','best_label_accuracy']
    print(frame.sort_values('mean_energy')[cols].to_string(index=False))
    print('\nFinal hardware-calibrated angles:')
    print(json.dumps(final, indent=2), flush=True)


if __name__ == '__main__':
    main()
