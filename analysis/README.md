# Audit and classical-only supplementary analyses

These scripts use only the released phantom generator and graph builder. They need no IBM Quantum access and no HCP data.

| Script | What it does | Output |
|---|---|---|
| `verify_phantom_tables.py` | Recomputes the Table 1/Table 2 descriptives, exact sign-flip contrasts (Holm over the angle and rotation effects) and the Heron-vs-Nighthawk paired tests from `data/phantom/phantom_study_level_results.csv`. | `verify_phantom_tables_output.txt` |
| `classical_baselines/nonspatial_baseline_standard_phantoms.py` | Regenerates the 60 held-out phantom graphs (seeds 2001–2015, four conditions), solves them exactly and compares the MILP labels with non-spatial baselines (orientation two-means, peak amplitude, random within-voxel-feasible). | `nonspatial_baseline_standard_phantoms.csv` |
| `classical_baselines/higher_curvature_phantoms.py` | Same comparison for phantoms with stronger curvature (20–60°), in which the orientation ranges of the two populations overlap. | `higher_curvature_phantoms.csv` |
| `classical_baselines/nonspatial.py` | Shared orientation two-means baseline. It restarts Lloyd iterations from 720 splits and keeps the lowest within-cluster cost, so it does not depend on node order. (An earlier single-start version gave order-dependent results on the c = 40° phantoms.) | — |
| `classical_baselines/higher_curvature_ideal_qaoa.py` | Exact light-cone p = 1 expected normalized gap with the frozen Heron parameters on standard and higher-curvature 90° phantoms. | `higher_curvature_ideal_qaoa.csv` |

Run from the repository root:

```bash
python analysis/verify_phantom_tables.py data/phantom/phantom_study_level_results.csv
python analysis/classical_baselines/nonspatial_baseline_standard_phantoms.py src/phantom analysis/classical_baselines/nonspatial_baseline_standard_phantoms.csv
python analysis/classical_baselines/higher_curvature_phantoms.py src/phantom analysis/classical_baselines/higher_curvature_phantoms.csv
python analysis/classical_baselines/higher_curvature_ideal_qaoa.py src analysis/classical_baselines/higher_curvature_ideal_qaoa.csv
```

The regenerated standard-phantom graphs reproduce the published MILP–phantom label agreement and the Heron ideal expected normalized gap (see `tests/test_smoke.py`).

The HCP counterpart, `src/hcp/baseline_nonspatial_tom.py`, has to be run locally on the HCP-derived graph outputs, which are not redistributed here.
