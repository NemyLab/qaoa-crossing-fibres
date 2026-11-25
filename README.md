# Hardware-aware QAOA for crossing-fibre reconstruction in diffusion MRI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24536852.svg)](https://doi.org/10.5281/zenodo.24536852)

Code and public reproducibility materials accompanying the manuscript by Milan Nemy and Daniel Ferreira.

## Scope

This repository contains the code used to construct the crossing-fibre Ising problems, run the depth-one QAOA simulations/hardware workflows, reproduce the controlled phantom experiments, perform the finite-budget classical controls, and reproduce the analysis/figure-generation steps. It also contains the public numerical outputs for the synthetic phantom and classical-control experiments.

**HCP-YA imaging data are not redistributed in this repository.** A small set of numerical results and 2D renderings derived from HCP-YA Open Access data is included in `data/hcp/` and is redistributed under the WU-Minn HCP Open Access Data Use Terms (see [`data/hcp/HCP_DATA_USE_TERMS.md`](data/hcp/HCP_DATA_USE_TERMS.md)). Users who wish to reproduce the human-data analyses from the images must obtain HCP-YA data through ConnectomeDB and accept the same terms. See `docs/HCP_REPRODUCTION.md`.

## Repository layout

- `src/phantom/` – synthetic crossing-fibre graph generation, validation and phantom analyses.
- `src/qaoa/` – common noiseless QAOA and sampling utilities.
- `src/hardware/heron/` – Heron calibration, held-out and replication workflows.
- `src/hardware/nighthawk/` – Nighthawk phantom workflows.
- `src/classical/` – finite-budget classical controls and simulated-annealing tuning.
- `src/hcp/` – HCP graph construction, validation, MPS/QPU analysis and figure scripts. These scripts require locally obtained HCP data.
- `data/phantom/` – public synthetic-phantom numerical outputs underlying the manuscript analyses.
- `data/classical/` – per-seed/per-budget classical-control outputs and frozen SA configuration.
- `docs/` – reproduction notes.
- `analysis/` – table verification and classical-only supplementary analyses (non-spatial baselines, higher-curvature phantoms); see `analysis/README.md`.
- `tests/` – fast smoke tests (`pytest -q`), including a check that the regenerated phantoms reproduce published values.

## Software environment

The principal quantum experiments used Qiskit 2.4.2, Qiskit Aer 0.17.2 and qiskit-ibm-runtime 0.48.0. The included `requirements.txt` captures the local scientific-Python/Qiskit requirements used by the core workflow; IBM Runtime access is required only for hardware execution.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install qiskit-ibm-runtime==0.48.0
```

## Phantom design

The manuscript evaluation uses four paired phantom conditions (90° and 60° crossing angles, each in original and globally 45°-rotated orientation). Seed 2000 is reserved for calibration/tuning; seeds 2001–2015 are held out for evaluation. The generators in `src/phantom/` reconstruct the deterministic instance geometry and noise realizations described in the Methods.

## Quick check

```bash
pip install -r requirements.txt
pytest -q
```

The tests import the offline modules and confirm that phantom seed 2001 (90°, original orientation) reproduces the published MILP–phantom label agreement and the Heron ideal expected normalized gap.

## Hardware results

The repository preserves scripts for the IBM Heron (`ibm_aachen`) and Nighthawk (`ibm_berlin`) workflows. Re-running on current hardware will not reproduce historical calibration states exactly. Hardware job outputs should therefore be interpreted together with the archived v1.0.0 release (https://doi.org/10.5281/zenodo.24536852) rather than as deterministic unit tests. No IBM credentials are stored in this repository.

Raw phantom outputs: the bitstring counts, submitted transpiled circuits (QPY), transpilation trials with the contemporaneous calibration errors used for circuit selection, and job metadata of all phantom calibration and evaluation runs on both processors are in [`data/phantom/hardware_raw/`](data/phantom/hardware_raw) (see its README, which also gives the command that reproduces the Heron-versus-Nighthawk comparison). The higher-curvature Heron run is in [`data/phantom/step19_hardware/`](data/phantom/step19_hardware). Full-processor calibration snapshots summarized in Supplementary Table 1 are not included. `src/hardware/heron/step10_transpile_preflight.py` contains only the helper metrics imported by the replication workflow.

## Human Connectome Project data

The human-data workflow uses HCP-YA data obtained under the WU-Minn HCP Open Access Data Use Terms. The exact HCP-YA subjects and diffusion-grid patch centres used in the 30-subject analysis are listed in [`data/hcp/hcp_patch_manifest.csv`](data/hcp/hcp_patch_manifest.csv). The representative case used in the manuscript is subject 103414, with the 5×5 patch centred at diffusion-grid voxel (88, 86, 76).

This repository does not redistribute HCP imaging data or the per-patch graphs of the 30-subject analysis. As derived data under the WU-Minn HCP Open Access Data Use Terms, it includes the per-participant summary metrics of the 30-subject analysis ([`data/hcp/hcp_subject_level_results.csv`](data/hcp/hcp_subject_level_results.csv)) and, for the representative subject 103414, the 50-node graph, its exact solution, the 10,000 Heron bitstrings and the 2D renderings used in Figures 1 and 3 ([`data/hcp/representative_103414/`](data/hcp/representative_103414)). See [`docs/HCP_REPRODUCTION.md`](docs/HCP_REPRODUCTION.md) for reproduction instructions.

## Citation

If you use this code, synthetic data, or associated reproducibility materials, please cite the accompanying manuscript:

Nemy, M. & Ferreira, D. Hardware-aware QAOA for crossing-fibre reconstruction in diffusion MRI. Manuscript in preparation.

The version used in the manuscript is archived at Zenodo:

Nemy, M. & Ferreira, D. Code and data for "Hardware-aware QAOA for crossing-fibre reconstruction in diffusion MRI" (v1.0.0). Zenodo https://doi.org/10.5281/zenodo.24536852 (2026).

## License

The source code in this repository is released under the MIT License.

The repository contains only publicly redistributable code, synthetic phantom data, and derived numerical results. Human Connectome Project Young Adult (HCP-YA) imaging data are not redistributed here. The HCP-derived files in `data/hcp/` are not covered by the MIT License; they are redistributed under the WU–Minn HCP Open Access Data Use Terms, which also govern access to the original HCP-YA data.
