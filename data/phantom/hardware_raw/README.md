# Raw phantom hardware outputs (IBM Heron and Nighthawk)

Unmodified outputs of the phantom calibration and evaluation jobs, copied from the original working
directory with the same relative paths, so that the command below runs unchanged from this folder.

| Folder | Processor | Content |
|---|---|---|
| `results/step12_aachen_seed2000_grid/` | Heron, ibm_aachen | 5 × 5 calibration grid on the seed-2000 phantom (500 shots per pair) |
| `results/step13_aachen_seed2000_confirmation/` | Heron | 2,000-shot confirmation and the frozen Heron angles |
| `results/step14_aachen_heldout/`, `results/step15_aachen_replication/` | Heron | 90°, original orientation, seeds 2001–2015 |
| `rotated90_qpu_{preflight,remaining}/` | Heron | 90°, rotated by 45° |
| `curved60_qpu_{preflight,remaining}/` | Heron | 60°, original orientation |
| `curved60rot45_qpu_{preflight,remaining}/` | Heron | 60°, rotated by 45° |
| `results/nighthawk_standard_untwirled/seed2000_calibration/`, `.../seed2000_confirmation/` | Nighthawk, ibm_berlin | calibration grid, confirmation and the frozen Nighthawk angles |
| `results/nighthawk_standard_untwirled/nighthawk_{curved90,rotated90_single,curved60_single,curved60rot45_single}_preflight/` | Nighthawk | the four phantom conditions, seeds 2001–2015 |

File types in each folder:

* `*_counts.json.gz` - raw SamplerV2 bitstring counts (10,000 shots per evaluation circuit).
* `*.qpy` - the bound, hardware-transpiled (ISA) circuits that were submitted.
* `*_transpile_trials.csv`, `*_best_layouts.csv` - all transpilation trials with depth, two-qubit counts and the
  contemporaneous per-gate and readout calibration errors used for circuit selection.
* `*_submission.json`, `*_job_metrics.json` - job IDs, backend, shots, dynamical decoupling and timing.
* `*_hardware_results.csv`, `*_summary.json`, `*_paired_*comparison.csv` - per-circuit metrics derived from the counts.

Some folders also contain circuits for a second, noiseless-optimized angle strategy that was executed in the same
jobs; the manuscript uses the hardware-calibrated strategy (`hardware_calibrated` for Heron,
`nighthawk_calibrated` for Nighthawk).

Full-processor calibration snapshots summarized in Supplementary Table 1 are not included.

## Reproducing the Heron-versus-Nighthawk comparison (Figure 2 and Tables 1-2 inputs)

From this folder, with the repository `src/` subfolders on `PYTHONPATH`:

```bash
export PYTHONPATH=$(ls -d ../../../src/*/ ../../../src/hardware/*/ | tr '\n' ':')
python ../../../src/phantom/compare_heron_nighthawk_phantoms_with_pvalues.py \
  --heron-original-heldout results/step14_aachen_heldout/heldout_hardware_results.csv \
  --heron-original-replication results/step15_aachen_replication/replication_hardware_results.csv \
  --heron-rotated90 rotated90_qpu_preflight/rotated90_hardware_results.csv rotated90_qpu_remaining/rotated90_hardware_results.csv \
  --heron-normal60 curved60_qpu_preflight/curved60_hardware_results.csv curved60_qpu_remaining/curved60_hardware_results.csv \
  --heron-rotated60 curved60rot45_qpu_preflight/curved60rot45_hardware_results.csv curved60rot45_qpu_remaining/curved60rot45_hardware_results.csv \
  --heron-strategy hardware_calibrated \
  --nighthawk-original90 results/nighthawk_standard_untwirled/nighthawk_curved90_preflight/nighthawk_curved90_hardware_results.csv \
  --nighthawk-rotated90 results/nighthawk_standard_untwirled/nighthawk_rotated90_single_preflight/nighthawk_rotated90_hardware_results.csv \
  --nighthawk-normal60 results/nighthawk_standard_untwirled/nighthawk_curved60_single_preflight/nighthawk_curved60_hardware_results.csv \
  --nighthawk-rotated60 results/nighthawk_standard_untwirled/nighthawk_curved60rot45_single_preflight/nighthawk_curved60rot45_hardware_results.csv \
  --nighthawk-strategy nighthawk_calibrated \
  --seeds 2001 2002 2003 2004 2005 2006 2007 2008 2009 2010 2011 2012 2013 2014 2015 \
  --out comparison_out
```

The seed values, descriptives and paired contrasts produced this way match the archived manuscript outputs to
within 1e-14 (checked on 2026-11-25).
