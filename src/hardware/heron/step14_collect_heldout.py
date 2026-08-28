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


def paired_comparison(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed, group in frame.groupby('seed'):
        indexed = group.set_index('angle_strategy')
        if set(indexed.index) != {'hardware_calibrated', 'transferred_center'}:
            raise ValueError(f'Seed {seed}: missing angle strategy')
        calibrated = indexed.loc['hardware_calibrated']
        transferred = indexed.loc['transferred_center']
        energy_difference = float(calibrated['mean_energy'] - transferred['mean_energy'])
        difference_se = float(np.sqrt(
            calibrated['mean_energy_se'] ** 2 + transferred['mean_energy_se'] ** 2
        ))
        rows.append({
            'seed': int(seed),
            'calibrated_minus_transferred_mean_energy': energy_difference,
            'difference_mean_energy_se': difference_se,
            'difference_mean_energy_95ci_low': energy_difference - 1.96 * difference_se,
            'difference_mean_energy_95ci_high': energy_difference + 1.96 * difference_se,
            'calibrated_minus_transferred_mean_gap': float(
                calibrated['mean_normalized_gap'] - transferred['mean_normalized_gap']
            ),
            'calibrated_minus_transferred_cvar10_gap': float(
                calibrated['cvar10_normalized_gap'] - transferred['cvar10_normalized_gap']
            ),
            'calibrated_minus_transferred_best_gap': float(
                calibrated['best_normalized_gap'] - transferred['best_normalized_gap']
            ),
            'calibrated_minus_transferred_best_label_accuracy': float(
                calibrated['best_label_accuracy'] - transferred['best_label_accuracy']
            ),
            'calibrated_better_mean_energy': bool(energy_difference < 0),
        })
    return pd.DataFrame(rows).sort_values('seed')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()

    submission = json.loads((args.prepared / 'heldout_submission.json').read_text())
    candidates = pd.read_csv(args.prepared / submission['files']['candidates'])
    circuits = load_qpy_list(args.prepared / submission['files']['qpy'])
    if len(circuits) != len(candidates):
        raise ValueError('Circuit/candidate count mismatch')

    service = QiskitRuntimeService()
    job = service.job(str(submission['job_id']))
    print(f"Job {submission['job_id']} status: {job.status()}", flush=True)
    result = job.result()
    if len(result) != len(candidates):
        raise ValueError('PUB result/candidate count mismatch')

    graphs = {}
    rows = []
    counts_payload = {}
    for position, row in enumerate(candidates.itertuples(index=False)):
        seed = int(row.seed)
        if seed not in graphs:
            graph, diagnostics = rebuild_graph(seed)
            graphs[seed] = (graph, diagnostics)
        graph, _ = graphs[seed]
        counts = counts_from_pub(result[position], circuits[position])
        counts_payload[str(int(row.pub_index))] = counts
        metrics = analyze_counts(counts, graph, float(row.exact_energy))
        rows.append({**row._asdict(), **metrics})

    frame = pd.DataFrame(rows).sort_values(['seed', 'angle_strategy'])
    paired = paired_comparison(frame)

    strategy_aggregate = (
        frame.groupby('angle_strategy', as_index=False)
        .agg(
            n_seeds=('seed', 'nunique'),
            mean_hardware_expected_gap=('mean_normalized_gap', 'mean'),
            sd_hardware_expected_gap=('mean_normalized_gap', 'std'),
            mean_hardware_cvar10_gap=('cvar10_normalized_gap', 'mean'),
            mean_hardware_best_gap=('best_normalized_gap', 'mean'),
            mean_hardware_best_label_accuracy=('best_label_accuracy', 'mean'),
            total_exact_ground_hits=('exact_ground_hits', 'sum'),
            mean_ideal_expected_gap=('ideal_expected_normalized_gap', 'mean'),
        )
        .sort_values('mean_hardware_expected_gap')
    )

    aggregate = {
        'primary_endpoint': 'mean normalized energy gap across held-out seeds 2001-2003',
        'lower_is_better': True,
        'n_heldout_seeds': 3,
        'shots_per_seed_and_strategy': int(submission['shots_per_pub']),
        'mean_calibrated_minus_transferred_gap': float(
            paired['calibrated_minus_transferred_mean_gap'].mean()
        ),
        'calibrated_better_on_n_of_3_seeds': int(paired['calibrated_better_mean_energy'].sum()),
        'strategy_results': strategy_aggregate.to_dict(orient='records'),
        'interpretation_rule': (
            'The hardware-calibrated strategy is supported only if it improves the preregistered '
            'mean held-out normalized gap; seed-level consistency is reported descriptively.'
        ),
    }

    frame.to_csv(args.prepared / 'heldout_hardware_results.csv', index=False)
    paired.to_csv(args.prepared / 'heldout_paired_comparison.csv', index=False)
    strategy_aggregate.to_csv(args.prepared / 'heldout_strategy_aggregate.csv', index=False)
    (args.prepared / 'heldout_aggregate.json').write_text(json.dumps(aggregate, indent=2))
    metrics = job_metrics_safe(job)
    (args.prepared / 'heldout_job_metrics.json').write_text(json.dumps(metrics, indent=2, default=str))
    with gzip.open(
        args.prepared / 'heldout_hardware_counts.json.gz', 'wt', encoding='utf-8'
    ) as handle:
        json.dump({
            'counts_by_pub_index': counts_payload,
            'metadata': {
                'collected_at_utc': datetime.now(timezone.utc).isoformat(),
                'submission': submission,
                'job_metrics': metrics,
            },
        }, handle)

    print('\nHeld-out hardware results:')
    columns = [
        'seed', 'angle_strategy', 'gamma', 'beta',
        'mean_normalized_gap', 'cvar10_normalized_gap',
        'best_normalized_gap', 'best_label_accuracy', 'exact_ground_hits'
    ]
    print(frame[columns].to_string(index=False))
    print('\nPaired calibrated-minus-transferred comparison (negative is better):')
    print(paired.to_string(index=False))
    print('\nAggregate:')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
