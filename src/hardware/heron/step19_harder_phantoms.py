#!/usr/bin/env python3
"""Heron run on higher-curvature phantoms, where spatial coupling is needed.

Purpose
-------
On the four standard phantoms a non-spatial orientation rule (axial two-means)
recovers the labels as well as the exact Ising/MILP solution. Only when curvature
makes the two orientation populations overlap does the spatially coupled
solution beat the non-spatial rule: by 24 percentage points for the 90 degree
centre with c = 50 and by 13 points for the 60 degree centre with c = 60
(analysis/classical_baselines). These are the two primary conditions here.
This script tests whether the Heron p = 1 QAOA samples keep their
problem-dependent signal in that regime.

Design (frozen before any data are collected; see PREREGISTRATION below)
------
* Backend ibm_aachen (Heron r3), fractional RZZ gates, SamplerV2, XY4 DD,
  no twirling, 10,000 shots per circuit, all circuits in ONE runtime job.
* Angles: the frozen hardware-calibrated Heron angles from seed 2000.
* Conditions: c90_k50 and c60_k60, seeds 2001-2015 (15 each), plus a
  same-job drift control of the standard curved_90 phantom, seeds 2001-2005.
  Total 35 circuits.
* Transpilation: 20 trials per circuit, lowest calibrated error proxy wins
  (same rule as the replication cohort).

Usage
-----
    export PYTHONPATH=src/phantom:src/qaoa:src/hardware/heron
    # 1) build circuits (needs IBM account; no QPU time)
    python src/hardware/heron/step19_harder_phantoms.py prepare --out runs/step19
    # 2) validate, then submit (only --submit spends QPU time)
    python src/hardware/heron/step19_harder_phantoms.py submit --out runs/step19
    python src/hardware/heron/step19_harder_phantoms.py submit --out runs/step19 --submit
    # 3) collect and analyse once the job is done
    python src/hardware/heron/step19_harder_phantoms.py collect --out runs/step19

Offline check without an IBM account: add ``--fake`` to ``prepare`` (uses the
fake_aachen snapshot) and ``simulate`` instead of ``submit``/``collect`` to run
the analysis on noiseless Aer samples.
"""
from __future__ import annotations

import argparse
import gzip
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import step5_curved_signal_graph as s5
from step5_curved_signal_graph import build_curved_signal_graph, exact_milp
from step9_common import get_rcm_order, normalized_gap, qiskit_bitstring_to_spins, reorder_graph
from step9_run_10000 import expected_energy_lightcone_p1

# Frozen hardware-calibrated Heron angles (calibrated on seed 2000; Methods).
GAMMA = 0.7103982796590439
BETA = 1.22628080530017

HARDER = {
    'c90_k50': {'base_a': 0.0, 'base_b': 90.0, 'curvature': 50.0},
    'c60_k60': {'base_a': 15.0, 'base_b': 75.0, 'curvature': 60.0},
}
s5.GEOMETRIES.update(HARDER)

DESIGN = (
    [('c90_k50', seed, 'primary') for seed in range(2001, 2016)]
    + [('c60_k60', seed, 'primary') for seed in range(2001, 2016)]
    + [('curved_90', seed, 'drift_control') for seed in range(2001, 2006)]
)

PREREGISTRATION = {
    'frozen_at': 'before submission; committed to the repository with this script',
    'unit': 'graph seed (15 per harder geometry), paired within seed',
    'primary_endpoints': [
        'E1: Heron mean normalized gap < 1 (uniform random sampling) on each harder geometry',
        'E2: Heron execution penalty = Heron mean gap - ideal light-cone p=1 gap, per geometry',
        'E3: sign-invariant label agreement of the lowest-energy Heron sample vs the non-spatial '
        'axial two-means rule, paired by seed',
    ],
    'secondary_endpoints': [
        'Heron CVaR-10% normalized gap',
        'lowest-energy Heron sample vs best of an equal number of uniform random samples (agreement and gap)',
        'drift control: curved_90 seeds 2001-2005 vs the published Heron values',
    ],
    'tests': 'exact two-sided paired sign-flip tests on seed-level differences; Holm over E3 '
             'and the random-sampling comparison within each geometry',
    'no_parameter_retuning': True,
}


# ----------------------------------------------------------------------------
# Graph helpers
# ----------------------------------------------------------------------------

def build_graph(geometry: str, seed: int):
    graph, diagnostics = build_curved_signal_graph(
        grid_size=5, n_gradients=60, snr=10.0, b_value=2000.0, geometry=geometry,
        fraction_mode='imbalanced', degree_cap=3, conflict_weight=1.0, anchor_weight=0.35,
        sigma_orientation_deg=18.0, sigma_displacement_deg=22.0,
        minimum_continuation_weight=0.01, seed=seed,
    )
    return reorder_graph(graph, get_rcm_order(graph)), diagnostics


def sign_invariant_agreement(spins, truth) -> float:
    match = float(np.mean(np.asarray(spins) == np.asarray(truth)))
    return max(match, 1.0 - match)


def two_means_labels(nodes: pd.DataFrame, n_starts: int = 720) -> np.ndarray:
    """Non-spatial axial two-means rule: one label per peak, opposite within a voxel.

    Peak orientations are mapped to the doubled-angle unit circle and clustered
    into two groups. Lloyd iterations are started from ``n_starts`` splits by a
    line through the origin and the lowest within-cluster cost is kept, so the
    result does not depend on node order or on a single initialisation.
    """
    angle = np.arctan2(nodes['dir_y'].to_numpy(), nodes['dir_x'].to_numpy()) % np.pi
    z = np.exp(2j * angle)
    best = None
    for theta in np.linspace(0.0, 2.0 * np.pi, n_starts, endpoint=False):
        first = np.real(z * np.exp(-1j * theta)) > 0
        if first.all() or (~first).all():
            continue
        for _ in range(50):
            c0, c1 = z[first].mean(), z[~first].mean()
            updated = np.abs(z - c0) < np.abs(z - c1)
            if updated.all() or (~updated).all() or (updated == first).all():
                break
            first = updated
        c0, c1 = z[first].mean(), z[~first].mean()
        cost = float((np.abs(z[first] - c0) ** 2).sum() + (np.abs(z[~first] - c1) ** 2).sum())
        if best is None or cost < best[0] - 1e-12:
            best = (cost, c0, c1)
    _, c0, c1 = best
    labels = np.empty(len(nodes), dtype=int)
    for _, index in nodes.groupby(['row', 'col']).groups.items():
        i, j = list(index)
        di = abs(z[i] - c0) - abs(z[i] - c1)
        dj = abs(z[j] - c0) - abs(z[j] - c1)
        labels[i], labels[j] = (1, -1) if di < dj else (-1, 1)
    return labels


def sample_metrics(counts: dict[str, int], graph, exact_energy: float) -> dict:
    energies, weights, best = [], [], (np.inf, None)
    for bitstring, count in counts.items():
        spins = qiskit_bitstring_to_spins(bitstring, graph.n_nodes)
        energy = float(graph.energy(spins))
        energies.append(energy)
        weights.append(int(count))
        if energy < best[0]:
            best = (energy, spins)
    energies = np.asarray(energies)
    weights = np.asarray(weights)
    shots = int(weights.sum())
    order = np.argsort(energies)
    tail = int(np.ceil(0.10 * shots))
    taken, acc = 0, 0.0
    for k in order:
        use = min(weights[k], tail - taken)
        acc += use * energies[k]
        taken += use
        if taken >= tail:
            break
    mean_energy = float(np.dot(energies, weights) / shots)
    return {
        'shots': shots,
        'mean_normalized_gap': normalized_gap(mean_energy, exact_energy),
        'cvar10_normalized_gap': normalized_gap(acc / tail, exact_energy),
        'best_normalized_gap': normalized_gap(best[0], exact_energy),
        'best_sample_agreement': sign_invariant_agreement(best[1], graph.true_spins),
        'exact_ground_hits': int(weights[np.isclose(energies, exact_energy, atol=1e-8)].sum()),
    }


def random_counts(n_qubits: int, shots: int, seed: int) -> dict[str, int]:
    rng = np.random.default_rng(seed)
    bits = rng.integers(0, 2, size=(shots, n_qubits))
    strings = [''.join(map(str, row)) for row in bits]
    return pd.Series(strings).value_counts().to_dict()


def exact_sign_flip_p(differences) -> float:
    d = np.asarray(differences, dtype=float)
    d = d[np.abs(d) > 1e-12]
    if d.size == 0:
        return 1.0
    observed = abs(d.mean())
    stats = [abs(np.mean(d * np.array(signs)))
             for signs in itertools.product([1, -1], repeat=d.size)]
    return float(np.mean(np.asarray(stats) >= observed - 1e-12))


def holm(pvalues: list[float]) -> list[float]:
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, k in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[k]))
        adjusted[k] = running
    return adjusted.tolist()


# ----------------------------------------------------------------------------
# prepare
# ----------------------------------------------------------------------------

def get_backend(args):
    if getattr(args, 'fake', False):
        from qiskit_ibm_runtime.fake_provider import FakeAachen
        return FakeAachen()
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService()
    return service.backend(args.backend, use_fractional_gates=(args.gate_mode == 'fractional'))


def cmd_prepare(args) -> None:
    from qiskit import qpy
    from qiskit.transpiler import PassManager, generate_preset_pass_manager
    from qiskit.transpiler.passes import Optimize1qGatesDecomposition, RemoveIdentityEquivalent
    from step10_transpile_preflight import count_two_qubit_operations, target_error_metrics
    from step12_common import build_parameterized_qaoa_circuit

    args.out.mkdir(parents=True, exist_ok=True)
    backend = get_backend(args)
    native = set(map(str, backend.operation_names))
    fractional = args.gate_mode == 'fractional' and 'rzz' in native
    if args.gate_mode == 'fractional' and not fractional:
        if not args.fake:
            raise RuntimeError('Backend target does not expose rzz; check use_fractional_gates')
        print('Fake backend has no fractional rzz; dry run uses standard gates.', flush=True)

    fold = []
    if fractional:
        from qiskit_ibm_runtime.transpiler.passes import FoldRzzAngle
        fold = [FoldRzzAngle()]
    post = PassManager(fold + [
        Optimize1qGatesDecomposition(target=backend.target),
        RemoveIdentityEquivalent(target=backend.target),
    ])

    circuits, rows, trials = [], [], []
    for pub_index, (geometry, seed, role) in enumerate(DESIGN):
        print(f'[{pub_index + 1}/{len(DESIGN)}] {geometry} seed {seed}', flush=True)
        graph, _ = build_graph(geometry, seed)
        exact_spins, exact_energy, _ = exact_milp(graph)
        logical = build_parameterized_qaoa_circuit(graph)
        bound_logical = logical.assign_parameters(
            {p: (GAMMA if p.name == 'gamma' else BETA) for p in logical.parameters})
        best = None
        for trial in range(args.trials):
            tseed = args.transpiler_seed_base + 1000 * seed + 100 * (pub_index // 15) + trial
            try:
                pm = generate_preset_pass_manager(
                    optimization_level=3, backend=backend, seed_transpiler=tseed)
                isa = post.run(pm.run(bound_logical))
                if fractional and isa.count_ops().get('rzz', 0) != len(graph.edges):
                    raise RuntimeError('unexpected rzz count')
                errors = target_error_metrics(backend, isa)
                twoq, _ = count_two_qubit_operations(isa)
                score = (float(errors['error_proxy_negative_log_success']), int(twoq))
                trials.append({'pub_index': pub_index, 'trial': trial, 'transpiler_seed': tseed,
                               'error_proxy': score[0], 'twoq_count': score[1]})
                if best is None or score < best[0]:
                    best = (score, isa, tseed, errors)
            except Exception as exc:  # keep going; record why
                trials.append({'pub_index': pub_index, 'trial': trial, 'transpiler_seed': tseed,
                               'error': repr(exc)})
        if best is None:
            raise RuntimeError(f'No successful transpilation for {geometry} seed {seed}')
        (_, twoq), isa, tseed, errors = best
        ideal = expected_energy_lightcone_p1(graph, GAMMA, BETA)
        tm = two_means_labels(graph.nodes)
        circuits.append(isa)
        rows.append({
            'pub_index': pub_index, 'geometry': geometry, 'seed': seed, 'role': role,
            'gamma': GAMMA, 'beta': BETA, 'n_edges': int(len(graph.edges)),
            'exact_energy': float(exact_energy),
            'milp_agreement': sign_invariant_agreement(exact_spins, graph.true_spins),
            'two_means_agreement': sign_invariant_agreement(tm, graph.true_spins),
            'ideal_normalized_gap': normalized_gap(ideal, exact_energy),
            'transpiler_seed': tseed, 'twoq_count': int(twoq), 'depth': int(isa.depth()),
            'error_proxy': float(errors['error_proxy_negative_log_success']),
            'mean_readout_error': float(errors.get('mean_readout_error', np.nan)),
        })

    candidates = pd.DataFrame(rows)
    candidates.to_csv(args.out / 'candidates.csv', index=False)
    pd.DataFrame(trials).to_csv(args.out / 'transpile_trials.csv', index=False)
    with (args.out / 'circuits.qpy').open('wb') as handle:
        qpy.dump(circuits, handle)
    meta = {
        'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'backend': backend.name, 'gate_mode': 'fractional' if fractional else 'standard',
        'fake_backend': bool(args.fake), 'n_pubs': len(circuits),
        'transpilation_trials': args.trials, 'gamma': GAMMA, 'beta': BETA,
        'preregistration': PREREGISTRATION,
    }
    (args.out / 'prepared.json').write_text(json.dumps(meta, indent=2))
    print(candidates.groupby('geometry')[
        ['milp_agreement', 'two_means_agreement', 'ideal_normalized_gap', 'twoq_count', 'depth']
    ].mean().round(3).to_string())
    print('\nPreparation only. No QPU job was submitted.')


# ----------------------------------------------------------------------------
# submit
# ----------------------------------------------------------------------------

def load_prepared(out: Path):
    from qiskit import qpy
    meta = json.loads((out / 'prepared.json').read_text())
    candidates = pd.read_csv(out / 'candidates.csv')
    with (out / 'circuits.qpy').open('rb') as handle:
        circuits = list(qpy.load(handle))
    if len(circuits) != len(candidates) or len(circuits) != len(DESIGN):
        raise ValueError('Prepared circuits do not match the frozen design')
    return meta, candidates, circuits


def cmd_submit(args) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2 as Sampler
    meta, candidates, circuits = load_prepared(args.out)
    if meta['fake_backend']:
        raise RuntimeError('These circuits were prepared on a fake backend; re-run prepare without --fake')
    service = QiskitRuntimeService()
    backend = service.backend(meta['backend'], use_fractional_gates=meta['gate_mode'] == 'fractional')
    status = backend.status()
    if not status.operational:
        raise RuntimeError(f"{meta['backend']} is not operational")
    for circuit in circuits:
        backend.check_faulty(circuit)
    preview = {
        'backend': meta['backend'], 'n_pubs': len(circuits), 'shots_per_pub': args.shots,
        'total_shots': args.shots * len(circuits), 'dd': 'XY4',
        'max_execution_time_s': args.max_execution_time, 'pending_jobs': int(status.pending_jobs),
    }
    print(json.dumps(preview, indent=2))
    if not args.submit:
        print('\nValidation only. Re-run with --submit to use QPU time.')
        return
    sampler = Sampler(mode=backend)
    sampler.options.twirling.enable_gates = False
    sampler.options.twirling.enable_measure = False
    sampler.options.dynamical_decoupling.enable = True
    sampler.options.dynamical_decoupling.sequence_type = 'XY4'
    sampler.options.max_execution_time = int(args.max_execution_time)
    sampler.options.environment.job_tags = ['DMRI_CROSSING_QAOA', 'step19_harder_phantoms',
                                            meta['gate_mode'], 'dd_XY4']
    job = sampler.run([(c,) for c in circuits], shots=int(args.shots))
    record = {**preview, 'job_id': job.job_id(),
              'submitted_at_utc': datetime.now(timezone.utc).isoformat()}
    (args.out / 'submission.json').write_text(json.dumps(record, indent=2))
    print(f"\nSubmitted job {job.job_id()}")


# ----------------------------------------------------------------------------
# collect / simulate / analyse
# ----------------------------------------------------------------------------

def counts_from_pub(pub_result, circuit) -> dict[str, int]:
    from step12_common import get_result_bit_array
    bit_array = get_result_bit_array(pub_result, circuit)
    raw = bit_array.get_counts() if tuple(bit_array.shape) == () else bit_array.get_counts(0)
    return {str(k): int(v) for k, v in raw.items()}


GRAPHS_FILE = 'graphs.json.gz'


def graph_to_record(graph) -> dict:
    return {
        'nodes': graph.nodes.to_dict(orient='list'),
        'edges': graph.edges.to_dict(orient='list'),
        'fields': np.asarray(graph.fields, dtype=float).tolist(),
        'true_spins': np.asarray(graph.true_spins, dtype=int).tolist(),
        'grid_size': int(graph.grid_size), 'seed': int(graph.seed),
    }


def graph_from_record(record: dict):
    from step1_crossing_graph import CrossingGraph
    return CrossingGraph(
        nodes=pd.DataFrame(record['nodes']), edges=pd.DataFrame(record['edges']),
        fields=np.asarray(record['fields'], dtype=float),
        true_spins=np.asarray(record['true_spins'], dtype=int),
        grid_size=int(record['grid_size']), seed=int(record['seed']),
    )


def load_graphs(out: Path) -> dict[int, object] | None:
    """Graphs exactly as they were built on the machine that prepared the job.

    Peak extraction is a fit on a 2-degree orientation dictionary. In a few voxels of the
    high-curvature geometries two candidate axes fit almost equally well, so floating-point
    differences between platforms (BLAS, NumPy/SciPy versions) can change a peak and hence
    an edge weight or a reference label. Re-analysis therefore uses the stored graphs when
    they are available instead of regenerating them.
    """
    path = out / GRAPHS_FILE
    if not path.exists():
        return None
    with gzip.open(path, 'rt', encoding='utf-8') as handle:
        payload = json.load(handle)
    return {int(k): graph_from_record(v) for k, v in payload['graphs_by_pub_index'].items()}


def cmd_export_graphs(args) -> None:
    """Write the graphs of the frozen design, as built on this machine, next to the run."""
    import platform
    import scipy
    candidates = pd.read_csv(args.out / 'candidates.csv') if (args.out / 'candidates.csv').exists() else None
    graphs, checks = {}, []
    for pub_index, (geometry, seed, _role) in enumerate(DESIGN):
        graph, _ = build_graph(geometry, seed)
        spins, energy, _ = exact_milp(graph)
        record = graph_to_record(graph)
        record['exact_spins'] = np.asarray(spins, dtype=int).tolist()
        record['exact_energy'] = float(energy)
        graphs[str(pub_index)] = record
        if candidates is not None:
            stored = float(candidates.loc[candidates['pub_index'] == pub_index, 'exact_energy'].iloc[0])
            checks.append(abs(stored - float(energy)))
    payload = {
        'exported_at_utc': datetime.now(timezone.utc).isoformat(),
        'platform': platform.platform(), 'python': platform.python_version(),
        'numpy': np.__version__, 'scipy': scipy.__version__, 'pandas': pd.__version__,
        'graphs_by_pub_index': graphs,
    }
    with gzip.open(args.out / GRAPHS_FILE, 'wt', encoding='utf-8') as handle:
        json.dump(payload, handle)
    if checks:
        worst = max(checks)
        print(f'max |exact energy - candidates.csv| = {worst:.2e}')
        if worst > 1e-6:
            raise RuntimeError('Graphs differ from the prepared run; export on the machine that ran prepare')
    print(f'Wrote {args.out / GRAPHS_FILE} ({len(graphs)} graphs)')


def analyse(out: Path, candidates: pd.DataFrame, counts_by_pub: dict[int, dict], label: str) -> None:
    stored = load_graphs(out)
    if stored is None:
        print('No stored graphs; regenerating them on this machine.', flush=True)
    rows = []
    for row in candidates.itertuples(index=False):
        if stored is not None:
            graph = stored[int(row.pub_index)]
        else:
            graph, _ = build_graph(row.geometry, int(row.seed))
        counts = counts_by_pub[int(row.pub_index)]
        qpu = sample_metrics(counts, graph, float(row.exact_energy))
        shots = qpu['shots']
        rnd = sample_metrics(random_counts(graph.n_nodes, shots, 900_000 + int(row.pub_index)),
                             graph, float(row.exact_energy))
        rows.append({**row._asdict(), **{f'qpu_{k}': v for k, v in qpu.items()},
                     'random_best_sample_agreement': rnd['best_sample_agreement'],
                     'random_best_normalized_gap': rnd['best_normalized_gap'],
                     'random_mean_normalized_gap': rnd['mean_normalized_gap']})
    frame = pd.DataFrame(rows)
    frame.to_csv(out / f'{label}_seed_level.csv', index=False)

    summary = []
    for geometry, g in frame.groupby('geometry', sort=False):
        penalty = g['qpu_mean_normalized_gap'] - g['ideal_normalized_gap']
        e1 = g['qpu_mean_normalized_gap'] - 1.0
        e3 = g['qpu_best_sample_agreement'] - g['two_means_agreement']
        vs_random = g['qpu_best_sample_agreement'] - g['random_best_sample_agreement']
        p_e3, p_rand = exact_sign_flip_p(e3), exact_sign_flip_p(vs_random)
        adj = holm([p_e3, p_rand])
        summary.append({
            'geometry': geometry, 'n_seeds': len(g),
            'milp_agreement': g['milp_agreement'].mean(),
            'two_means_agreement': g['two_means_agreement'].mean(),
            'ideal_gap': g['ideal_normalized_gap'].mean(),
            'qpu_mean_gap': g['qpu_mean_normalized_gap'].mean(),
            'qpu_mean_gap_sd': g['qpu_mean_normalized_gap'].std(ddof=1),
            'E1_p_gap_below_random': exact_sign_flip_p(e1),
            'E2_execution_penalty': penalty.mean(),
            'qpu_cvar10_gap': g['qpu_cvar10_normalized_gap'].mean(),
            'qpu_best_sample_agreement': g['qpu_best_sample_agreement'].mean(),
            'random_best_sample_agreement': g['random_best_sample_agreement'].mean(),
            'E3_qpu_best_minus_two_means': e3.mean(),
            'E3_p': p_e3, 'E3_p_holm': adj[0],
            'qpu_best_minus_random_best': vs_random.mean(),
            'vs_random_p': p_rand, 'vs_random_p_holm': adj[1],
        })
    summary = pd.DataFrame(summary)
    summary.to_csv(out / f'{label}_summary.csv', index=False)
    print(summary.round(4).T.to_string())


def cmd_collect(args) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from step11_common import job_metrics_safe
    meta, candidates, circuits = load_prepared(args.out)
    submission = json.loads((args.out / 'submission.json').read_text())
    job = QiskitRuntimeService().job(submission['job_id'])
    print(f"Job {submission['job_id']}: {job.status()}", flush=True)
    result = job.result()
    counts = {int(r.pub_index): counts_from_pub(result[k], circuits[k])
              for k, r in enumerate(candidates.itertuples(index=False))}
    with gzip.open(args.out / 'hardware_counts.json.gz', 'wt', encoding='utf-8') as handle:
        json.dump({'counts_by_pub_index': counts, 'submission': submission,
                   'job_metrics': job_metrics_safe(job)}, handle, default=str)
    analyse(args.out, candidates, counts, 'hardware')


def cmd_reanalyse(args) -> None:
    """Recompute all metrics from saved counts, without IBM access."""
    candidates = pd.read_csv(args.out / 'candidates.csv')
    with gzip.open(args.out / 'hardware_counts.json.gz', 'rt', encoding='utf-8') as handle:
        payload = json.load(handle)
    counts = {int(k): v for k, v in payload['counts_by_pub_index'].items()}
    analyse(args.out, candidates, counts, args.label)


def cmd_simulate(args) -> None:
    """Noiseless check of the analysis path using Aer MPS on the logical circuits."""
    from qiskit_aer import AerSimulator
    from step12_common import build_parameterized_qaoa_circuit
    candidates = pd.read_csv(args.out / 'candidates.csv') if (args.out / 'candidates.csv').exists() else None
    if candidates is None:
        raise FileNotFoundError('run prepare first (with --fake is fine)')
    sim = AerSimulator(method='matrix_product_state', seed_simulator=7)
    counts = {}
    for row in candidates.itertuples(index=False):
        graph, _ = build_graph(row.geometry, int(row.seed))
        circuit = build_parameterized_qaoa_circuit(graph)
        circuit = circuit.assign_parameters(
            {p: (GAMMA if p.name == 'gamma' else BETA) for p in circuit.parameters})
        counts[int(row.pub_index)] = sim.run(circuit, shots=args.shots).result().get_counts()
    analyse(args.out, candidates, counts, 'noiseless_mps')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='cmd', required=True)
    p = sub.add_parser('prepare')
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--backend', default='ibm_aachen')
    p.add_argument('--gate-mode', choices=['fractional', 'standard'], default='fractional')
    p.add_argument('--trials', type=int, default=20)
    p.add_argument('--transpiler-seed-base', type=int, default=191_900_000)
    p.add_argument('--fake', action='store_true')
    s = sub.add_parser('submit')
    s.add_argument('--out', type=Path, required=True)
    s.add_argument('--shots', type=int, default=10_000)
    s.add_argument('--max-execution-time', type=int, default=900,
                   help='hard cap on QPU seconds for the job')
    s.add_argument('--submit', action='store_true')
    c = sub.add_parser('collect')
    c.add_argument('--out', type=Path, required=True)
    e = sub.add_parser('export-graphs')
    e.add_argument('--out', type=Path, required=True)
    r = sub.add_parser('reanalyse')
    r.add_argument('--out', type=Path, required=True)
    r.add_argument('--label', default='hardware_reanalysed')
    m = sub.add_parser('simulate')
    m.add_argument('--out', type=Path, required=True)
    m.add_argument('--shots', type=int, default=10_000)
    args = parser.parse_args()
    {'prepare': cmd_prepare, 'submit': cmd_submit, 'collect': cmd_collect,
     'simulate': cmd_simulate, 'export-graphs': cmd_export_graphs,
     'reanalyse': cmd_reanalyse}[args.cmd](args)


if __name__ == '__main__':
    main()
