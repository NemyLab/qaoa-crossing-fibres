# Reproducing the HCP-YA analysis

The manuscript uses the Human Connectome Project Young Adult (HCP-YA) 2025 release. Access requires a ConnectomeDB account and acceptance of the WU-Minn HCP Open Access Data Use Terms. Everyone who works with shared HCP Open Access data must review and agree to those terms.

Official information: https://www.humanconnectome.org/study/hcp-young-adult/data-use-terms

For that reason this public GitHub package does **not** redistribute the HCP MRI volumes or subject-level HCP-derived graph/result files. The code required to reconstruct the analysis is in `src/hcp/`.

The manuscript workflow uses 5×5 coronal patches, two retained candidate orientations per voxel, eight-neighbour voxel connectivity, global degree pruning to maximum node degree 3, exact MILP optimization, noiseless MPS sampling and raw SamplerV2 hardware sampling. The representative manuscript example is HCP subject 103414 at diffusion-grid centre (88, 86, 76).

