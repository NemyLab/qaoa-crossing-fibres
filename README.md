# Hardware-aware QAOA for crossing-fibre reconstruction in diffusion MRI

Code and public reproducibility materials accompanying the manuscript by Milan Nemy, Daniel Ferreira and Josef Sivic.

## Scope

This repository contains the code used to construct the crossing-fibre Ising problems, run the depth-one QAOA simulations/hardware workflows, reproduce the controlled phantom experiments, perform the finite-budget classical controls, and reproduce the analysis/figure-generation steps. It also contains the public numerical outputs for the synthetic phantom and classical-control experiments.

**HCP-YA subject-level data and HCP-derived subject-level files are intentionally not redistributed in this public repository.** Users who wish to reproduce the human-data analyses must obtain HCP-YA data through ConnectomeDB and accept the WU-Minn HCP Open Access Data Use Terms. See `docs/HCP_REPRODUCTION.md`.

## Repository layout

- `src/phantom/` – synthetic crossing-fibre graph generation, validation and phantom analyses.
- `src/qaoa/` – common noiseless QAOA and sampling utilities.
- `src/hardware/heron/` – Heron calibration, held-out and replication workflows.
- `src/hardware/nighthawk/` – Nighthawk phantom workflows.
- `src/classical/` – finite-budget classical controls and simulated-annealing tuning.
- `src/hcp/` – HCP graph construction, validation, MPS/QPU analysis and figure scripts. These scripts require locally obtained HCP data.
- `data/phantom/` – public synthetic-phantom numerical outputs underlying the manuscript analyses.
- `data/classical/` – per-seed/per-budget classical-control outputs and frozen SA configuration.
- `docs/` – reproduction notes, curation notes and source provenance.

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

## Hardware results

The repository preserves scripts for the IBM Heron (`ibm_aachen`) and Nighthawk (`ibm_berlin`) workflows. Re-running on current hardware will not reproduce historical calibration states exactly. Hardware job outputs should therefore be interpreted together with the archived manuscript results/DOI release rather than as deterministic unit tests. No IBM credentials are stored in this repository.

## Human Connectome Project data

The human-data workflow uses HCP-YA data obtained under the WU-Minn HCP Open Access Data Use Terms. This public repository includes code only for the HCP workflow, not subject-level HCP-derived graph tables, bitstrings or participant-level outputs. See `docs/HCP_REPRODUCTION.md`.

## Citation

If you use this code, synthetic data, or associated reproducibility materials, please cite the accompanying manuscript:

Nemy, M., Ferreira, D., & Sivic, J. Hardware-aware QAOA for crossing-fibre reconstruction in diffusion MRI. Manuscript in preparation.

## License

The source code in this repository is released under the MIT License.

The repository contains only publicly redistributable code, synthetic phantom data, and derived numerical results. Human Connectome Project Young Adult (HCP-YA) participant-level data and subject-derived files are not redistributed here. Access to the original HCP-YA data is subject to the WU–Minn HCP Data Use Terms.
