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
from scipy import stats

from step11_common import job_metrics_safe
from step12_common import analyze_counts, get_result_bit_array, rebuild_graph
from step15_prepare_replication import REPLICATION_SEEDS


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
        difference = float(calibrated['mean_energy'] - transferred['mean_energy'])
        difference_se = float(np.sqrt(
            calibrated['mean_energy_se'] ** 2 + transferred['mean_energy_se'] ** 2
        ))
        rows.append({
            'seed': int(seed),
            'calibrated_minus_transferred_mean_energy': difference,
            'difference_mean_energy_se': difference_se,
            'difference_mean_energy_95ci_low': difference - 1.96 * difference_se,
            'difference_mean_energy_95ci_high': difference + 1.96 * difference_se,
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
            'calibrated_better_mean_energy': bool(difference < 0),
        })
    return pd.DataFrame(rows).sort_values('seed')


def paired_inference(gap_differences: np.ndarray) -> dict:
    values = np.asarray(gap_differences, dtype=float)
    n = int(values.size)
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else float('nan')
    se = float(sd / np.sqrt(n)) if n > 1 else float('nan')
    if n > 1:
        tcrit = float(stats.t.ppf(0.975, df=n - 1))
        ci = [mean - tcrit * se, mean + tcrit * se]
        t_result = stats.ttest_1samp(values, popmean=0.0, alternative='less')
        t_p = float(t_result.pvalue)
    else:
        ci = [float('nan'), float('nan')]
        t_p = float('nan')

    n_better = int(np.sum(values < 0))
    sign_p = float(stats.binomtest(n_better, n=n, p=0.5, alternative='greater').pvalue)
    nonzero = values[np.abs(values) > 1e-15]
    if nonzero.size:
        try:
            wilcoxon_p = float(stats.wilcoxon(
                nonzero, alternative='less', method='auto'
            ).pvalue)
        except Exception:
            wilcoxon_p = float('nan')
    else:
        wilcoxon_p = float('nan')

    return {
        'n_seeds': n,
        'mean_calibrated_minus_transferred_gap': mean,
        'sd_paired_gap_difference': sd,
        'se_paired_gap_difference': se,
        'paired_gap_difference_95ci': ci,
        'one_sided_paired_t_pvalue': t_p,
        'one_sided_wilcoxon_pvalue': wilcoxon_p,
        'calibrated_better_on_n_seeds': n_better,
        'one_sided_exact_sign_pvalue': sign_p,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepared', type=Path, required=True)
    args = parser.parse_args()

    submission = json.loads((args.prepared / 'replication_submission.json').read_text())
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
            graphs[seed] = rebuild_graph(seed)[0]
        graph = graphs[seed]
        counts = counts_from_pub(result[position], circuits[position])
        counts_payload[str(int(row.pub_index))] = counts
        metrics = analyze_counts(counts, graph, float(row.exact_energy))
        rows.append({**row._asdict(), **metrics})

    frame = pd.DataFrame(rows).sort_values(['seed', 'angle_strategy'])
    if tuple(sorted(frame['seed'].astype(int).unique())) != REPLICATION_SEEDS:
        raise ValueError('Collected result does not match frozen replication seeds')
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

    inference = paired_inference(
        paired['calibrated_minus_transferred_mean_gap'].to_numpy()
    )
    aggregate = {
        'primary_endpoint': (
            'paired mean normalized energy-gap difference across independent replication seeds 2004-2015'
        ),
        'lower_is_better': True,
        'n_replication_seeds': len(REPLICATION_SEEDS),
        'shots_per_seed_and_strategy': int(submission['shots_per_pub']),
        **inference,
        'strategy_results': strategy_aggregate.to_dict(orient='records'),
        'interpretation_rule': (
            'Replication supports hardware calibration when the mean paired gap difference is negative; '
            'the confidence interval, paired t, Wilcoxon, and sign results are reported without changing the endpoint.'
        ),
    }

    frame.to_csv(args.prepared / 'replication_hardware_results.csv', index=False)
    paired.to_csv(args.prepared / 'replication_paired_comparison.csv', index=False)
    strategy_aggregate.to_csv(args.prepared / 'replication_strategy_aggregate.csv', index=False)
    (args.prepared / 'replication_aggregate.json').write_text(
        json.dumps(aggregate, indent=2)
    )
    metrics = job_metrics_safe(job)
    (args.prepared / 'replication_job_metrics.json').write_text(
        json.dumps(metrics, indent=2, default=str)
    )
    with gzip.open(
        args.prepared / 'replication_hardware_counts.json.gz', 'wt', encoding='utf-8'
    ) as handle:
        json.dump({
            'counts_by_pub_index': counts_payload,
            'metadata': {
                'collected_at_utc': datetime.now(timezone.utc).isoformat(),
                'submission': submission,
                'job_metrics': metrics,
            },
        }, handle)

    print('\nReplication hardware results:')
    print(frame[[
        'seed', 'angle_strategy', 'mean_normalized_gap', 'cvar10_normalized_gap',
        'best_normalized_gap', 'best_label_accuracy'
    ]].to_string(index=False))
    print('\nPaired comparison (negative favors hardware calibration):')
    print(paired.to_string(index=False))
    print('\nReplication aggregate:')
    print(json.dumps(aggregate, indent=2), flush=True)


if __name__ == '__main__':
    main()
