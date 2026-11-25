import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]+'/phantom'); sys.path.insert(0, sys.argv[1]+'/qaoa')
import step5_curved_signal_graph as s5
from step9_run_10000 import expected_energy_lightcone_p1
from step9_common import get_rcm_order, reorder_graph
G,B=0.7103982796590439,1.22628080530017
rows=[]
for name,spec in [('curved_90',None),('c90_k40',{'base_a':0,'base_b':90,'curvature':40.0}),('c90_k50',{'base_a':0,'base_b':90,'curvature':50.0}),('c60_k60',{'base_a':15,'base_b':75,'curvature':60.0})]:
    if spec: s5.GEOMETRIES[name]=spec
    for seed in range(2001,2016):
        gr,_=s5.build_curved_signal_graph(grid_size=5,n_gradients=60,snr=10.0,b_value=2000.0,geometry=name,fraction_mode='imbalanced',degree_cap=3,conflict_weight=1.0,anchor_weight=0.35,sigma_orientation_deg=18.0,sigma_displacement_deg=22.0,minimum_continuation_weight=0.01,seed=seed)
        s,e,_=s5.exact_milp(gr); g2=reorder_graph(gr,get_rcm_order(gr))
        ev=expected_energy_lightcone_p1(g2,G,B)
        rows.append(dict(geom=name,seed=seed,ideal_gap=(ev-e)/abs(e)))
r=pd.DataFrame(rows); print(r.groupby('geom').ideal_gap.agg(['mean','std']).round(3)); r.to_csv(sys.argv[2],index=False)
