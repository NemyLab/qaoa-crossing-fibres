import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1]); sys.path.insert(0, __import__('os').path.dirname(__file__))
from nonspatial import two_means_labels
import step5_curved_signal_graph as s5
def sinv(a,b): m=(a==b).mean(); return max(m,1-m)
two_means=two_means_labels
rows=[]
for curv in [20,30,40,50,60]:
    for base,name in [((0,90),'90'),((15,75),'60')]:
        g=f'c{name}_k{curv}'; s5.GEOMETRIES[g]={'base_a':base[0],'base_b':base[1],'curvature':float(curv)}
        for seed in range(2001,2016):
            gr,_=s5.build_curved_signal_graph(grid_size=5,n_gradients=60,snr=10.0,b_value=2000.0,geometry=g,fraction_mode='imbalanced',degree_cap=3,conflict_weight=1.0,anchor_weight=0.35,sigma_orientation_deg=18.0,sigma_displacement_deg=22.0,minimum_continuation_weight=0.01,seed=seed)
            s,e,_=s5.exact_milp(gr); df=gr.nodes; t=df.true_spin.to_numpy()
            rows.append(dict(centre=name,curv=curv,seed=seed,milp=sinv(np.asarray(s),t),two_means=sinv(two_means(df),t),cmin=df.true_crossing_angle_deg.min()))
r=pd.DataFrame(rows); print(r.groupby(['centre','curv'])[['milp','two_means','cmin']].mean().round(3).to_string()); r.to_csv(sys.argv[2],index=False)
