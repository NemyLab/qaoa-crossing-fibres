"""Fast checks that the public code imports and reproduces published phantom values."""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
for sub in ["src/phantom", "src/qaoa", "src/classical", "src/hardware/heron"]:
    sys.path.insert(0, str(ROOT / sub))

OFFLINE_MODULES = [
    "step1_crossing_graph",
    "step4_signal_derived_graph",
    "step5_curved_signal_graph",
    "step9_common",
    "step9_run_10000",
    "step11_common",
    "step10_transpile_preflight",
    "step17_classical_controls",
    "step18_common",
]


@pytest.mark.parametrize("name", OFFLINE_MODULES)
def test_module_imports(name):
    importlib.import_module(name)


def _graph(seed: int, geometry: str = "curved_90"):
    from step5_curved_signal_graph import build_curved_signal_graph

    graph, _ = build_curved_signal_graph(
        grid_size=5, n_gradients=60, snr=10.0, b_value=2000.0, geometry=geometry,
        fraction_mode="imbalanced", degree_cap=3, conflict_weight=1.0, anchor_weight=0.35,
        sigma_orientation_deg=18.0, sigma_displacement_deg=22.0,
        minimum_continuation_weight=0.01, seed=seed,
    )
    return graph


def test_milp_reproduces_published_phantom_label_agreement():
    from step5_curved_signal_graph import exact_milp

    published = pd.read_csv(ROOT / "data/phantom/phantom_study_level_results.csv")
    row = published[(published.platform == "Heron") & (published.condition_key == "90_original")
                    & (published.unit_id == 2001)].iloc[0]
    graph = _graph(2001)
    spins, _, _ = exact_milp(graph)
    agreement = np.mean(np.asarray(spins) == graph.nodes["true_spin"].to_numpy())
    agreement = max(agreement, 1 - agreement)
    assert agreement == pytest.approx(row["best_agreement_with_anatomical_reference"], abs=1e-9)


def test_lightcone_ideal_gap_matches_published_value():
    from step5_curved_signal_graph import exact_milp
    from step9_common import get_rcm_order, reorder_graph
    from step9_run_10000 import expected_energy_lightcone_p1

    published = pd.read_csv(ROOT / "data/phantom/phantom_study_level_results.csv")
    row = published[(published.platform == "Heron") & (published.condition_key == "90_original")
                    & (published.unit_id == 2001)].iloc[0]
    graph = _graph(2001)
    _, exact_energy, _ = exact_milp(graph)
    rcm = reorder_graph(graph, get_rcm_order(graph))
    value = expected_energy_lightcone_p1(rcm, 0.7103982796590439, 1.22628080530017)
    gap = (value - exact_energy) / abs(exact_energy)
    assert gap == pytest.approx(row["ideal_mean_normalized_gap"], abs=1e-6)
