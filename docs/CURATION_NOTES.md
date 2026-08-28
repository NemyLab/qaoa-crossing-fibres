# Curation notes

This repository was assembled from the manuscript project directory on 2026-08-28.

Key curation decisions:

1. The validated **eight-neighbour** HCP graph-construction branch is used. Earlier four-neighbour/exploratory real-data outputs are excluded.
2. The current 30-subject Heron run is job `d9toeplca1vs73a5jnfg`; its cohort mean normalized gaps are 0.523721 (QPU mean), 0.323176 (QPU CVaR-10%) and 0.139514 (QPU best), matching the current manuscript.
3. The current representative HCP case is subject 103414, patch `103414_R_88_86_76`. Older material using subject 102513 as the representative is excluded from the curated public workflow.
4. Public numerical data are restricted to synthetic-phantom/classical-control results. HCP subject-level derived outputs are kept in a separate private companion archive because HCP access/redistribution is governed by the WU-Minn HCP Open Access Data Use Terms.
5. Machine-specific absolute paths and an embedded local Google Drive account string found in selected JSON provenance fields were sanitized in the private archive. Scientific result values were not changed.
6. Historical exploratory figure/script versions are omitted where a clearly superseding manuscript version exists.
