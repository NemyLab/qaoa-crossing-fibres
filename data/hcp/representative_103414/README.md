# Representative HCP patch 103414_R_88_86_76 (Figures 1 and 3)

* `real_graph_*` - 50-node eight-neighbour Ising graph (RCM order), exact MILP solution and validation summary.
* `real_fractional_hardware_*` - Heron (ibm_aachen, fractional RZZ) results for the hardware-calibrated angles:
  summary JSON, all 10,000 measured bitstrings (gzip JSON) and the best measured assignment.
* `mrview_fod_glyphs_coronal.png` - MRtrix mrview rendering of the 5 x 5 FOD patch (radiological view), used in Figure 3a.
* `mrview_fod_glyphs_coronal_2048.png` - the same rendering at 2048 px, used in Figure 1b.
* `anatomy_coronal_patch_localization.png` - coronal T1 slice with the TractSeg right CST (orange), the JHU right
  superior corona radiata transformed to subject space (blue) and the predefined patch (red); radiological view; Figure 1a.

Derived values and 2D renderings only; no HCP image volumes are included.
