import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, __import__('os').path.dirname(__file__))
from nonspatial import two_means_labels
from step5_curved_signal_graph import build_curved_signal_graph, exact_milp
def sinv(a,b): m=(a==b).mean(); return max(m,1-m)
rows=[]
for geom in ['curved_90','curved_90_rot45','curved_60','curved_60_rot45']:
    for seed in range(2001,2016):
        g,diag=build_curved_signal_graph(grid_size=5,n_gradients=60,snr=10.0,b_value=2000.0,geometry=geom,fraction_mode='imbalanced',degree_cap=3,conflict_weight=1.0,anchor_weight=0.35,sigma_orientation_deg=18.0,sigma_displacement_deg=22.0,minimum_continuation_weight=0.01,seed=seed)
        s,e,info=exact_milp(g)
        nodes=g.nodes if hasattr(g,'nodes') else None
        df=nodes if isinstance(nodes,pd.DataFrame) else pd.DataFrame(nodes)
        truth=df['true_spin'].to_numpy()
        triv=two_means_labels(df)
        amp=np.empty(len(df),int)
        for (r,cc),idx in df.groupby(['row','col']).groups.items():
            i,j=list(idx); amp[i],amp[j]=(1,-1) if df.amplitude[i]>=df.amplitude[j] else (-1,1)
        rng=np.random.default_rng(seed); rnd=np.empty(len(df),int)
        for (r,cc),idx in df.groupby(['row','col']).groups.items():
            i,j=list(idx); v=rng.choice([-1,1]); rnd[i],rnd[j]=v,-v
        rows.append(dict(geom=geom,seed=seed,milp=sinv(np.asarray(s),truth),orient2means=sinv(triv,truth),amplitude=sinv(amp,truth),random_feasible=sinv(rnd,truth),cross_min=df.true_crossing_angle_deg.min(),cross_mean=df.true_crossing_angle_deg.mean(),cross_max=df.true_crossing_angle_deg.max()))
r=pd.DataFrame(rows); print(r.groupby('geom')[['milp','orient2means','amplitude','random_feasible','cross_min','cross_mean','cross_max']].agg(['mean']).round(3).to_string())
r.to_csv(sys.argv[2],index=False)
