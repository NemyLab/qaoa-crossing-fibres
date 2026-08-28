# Synthetic phantom data

`phantom_instance_manifest.csv` enumerates the 60 held-out manuscript phantom instances: seeds 2001–2015 crossed with four paired conditions (90°/60° crossing angle × original/45° global rotation). Seed 2000 is reserved for QAOA parameter calibration and simulated-annealing tuning and is not part of the held-out manifest.

The actual graph instances are deterministic outputs of the generators in `src/phantom/` under the manuscript parameters (5×5 voxels, 60 diffusion directions, b=2000 s/mm², SNR=10, two candidate orientations per voxel). The numerical result tables in this folder are the curated outputs used for the manuscript phantom statistics and Heron–Nighthawk comparison.
