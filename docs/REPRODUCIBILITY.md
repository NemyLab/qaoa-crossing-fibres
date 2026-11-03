# Reproducibility map

## Optimization model and phantom generation

Start with `src/phantom/step1_crossing_graph.py`, `step4_signal_derived_graph.py` and `step5_curved_signal_graph.py`. The held-out manuscript phantom seeds are 2001–2015; seed 2000 is reserved for calibration/tuning.

## QAOA simulation and sampling

Core p=1 utilities and the 50-qubit transfer workflow are in `src/qaoa/`. Heron and Nighthawk hardware preparation/submission/collection scripts are separated under `src/hardware/`. Hardware reruns require IBM Quantum credentials and may compile differently because device calibration changes with time.

## Phantom statistics / processor comparison

The public source tables are in `data/phantom/`. `analyze_phantom_factorial_statistics.py` reproduces the factorial angle/rotation tests, and `compare_heron_nighthawk_phantoms_with_pvalues.py` reproduces the paired processor-family comparisons.

## Classical controls

`src/classical/step17_classical_controls.py` evaluates uniform sampling, first-improvement greedy, best-improvement greedy and simulated annealing at the manuscript budgets. `step18_tune_sa.py` documents the seed-2000-only SA tuning; the frozen selected configuration is in `data/classical/selected_sa_configuration.json`. Per-seed/per-budget outputs are included.

## HCP-YA

The validated eight-neighbour graph builder/validator and HCP analysis scripts are in `src/hcp/`. See `HCP_REPRODUCTION.md` for the data-access boundary. HCP participant-level derived outputs are not part of the public GitHub bundle.

## Module search path

Scripts import sibling modules across `src/` subfolders (for example, the hardware workflows import `step5_curved_signal_graph` and `step9_common`). Run them with all source folders on the path:

```bash
export PYTHONPATH="$PWD/src/phantom:$PWD/src/qaoa:$PWD/src/classical:$PWD/src/hardware/heron:$PWD/src/hcp"
```
