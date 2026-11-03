import sys, numpy as np, pandas as pd
sys.path.insert(0, sys.argv[1])
import step5_curved_signal_graph as s5
def sinv(a,b): m=(a==b).mean(); return max(m,1-m)
def two_means(df):
    ang=np.arctan2(df.dir_y,df.dir_x)%np.pi; z=np.exp(2j*ang); c=[z[0],-z[0]]
    for _ in range(50):
        lab=np.abs(z-c[0])<np.abs(z-c[1]); c=[z[lab].mean() if lab.any() else c[0], z[~lab].mean() if (~lab).any() else c[1]]
    out=np.empty(len(df),int)
    for _,idx in df.groupby(['row','col']).groups.items():
        i,j=list(idx); di=abs(z[i]-c[0])-abs(z[i]-c[1]); dj=abs(z[j]-c[0])-abs(z[j]-c[1])
        out[i],out[j]=(1,-1) if di<dj else (-1,1)
    return out
rows=[]
for curv in [20,30,40,50,60]:
    for base,name in [((0,90),'90'),((15,75),'60')]:
        g=f'c{name}_k{curv}'; s5.GEOMETRIES[g]={'base_a':base[0],'base_b':base[1],'curvature':float(curv)}
        for seed in range(2001,2016):
            gr,_=s5.build_curved_signal_graph(grid_size=5,n_gradients=60,snr=10.0,b_value=2000.0,geometry=g,fraction_mode='imbalanced',degree_cap=3,conflict_weight=1.0,anchor_weight=0.35,sigma_orientation_deg=18.0,sigma_displacement_deg=22.0,minimum_continuation_weight=0.01,seed=seed)
            s,e,_=s5.exact_milp(gr); df=gr.nodes; t=df.true_spin.to_numpy()
            rows.append(dict(centre=name,curv=curv,seed=seed,milp=sinv(np.asarray(s),t),two_means=sinv(two_means(df),t),cmin=df.true_crossing_angle_deg.min()))
r=pd.DataFrame(rows); print(r.groupby(['centre','curv'])[['milp','two_means','cmin']].mean().round(3).to_string()); r.to_csv(sys.argv[2],index=False)
