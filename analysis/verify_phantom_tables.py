import itertools, sys, numpy as np, pandas as pd
from scipy import stats
d=pd.read_csv(sys.argv[1])
print(d.groupby(['platform','condition_key']).size())
mets=['best_agreement_with_anatomical_reference','ideal_mean_normalized_gap','qpu_mean_normalized_gap','qpu_cvar10_normalized_gap','best_agreement_with_exact']
signs=np.array(list(itertools.product([-1,1],repeat=15)))
def signflip(x):
    obs=abs(x.mean()); null=np.abs((signs*x).mean(1)); return (null>=obs-1e-12).mean()
def holm(p):
    p=np.asarray(p); o=np.argsort(p); m=len(p); adj=np.empty(m); run=0
    for r,i in enumerate(o): run=max(run,(m-r)*p[i]); adj[i]=min(1,run)
    return adj
for plat,g in d.groupby('platform'):
    print('\n==',plat)
    for m in mets:
        w=g.pivot(index='unit_id',columns='condition_key',values=m)
        desc=' '.join(f"{c}:{w[c].mean():.3f}({w[c].std(ddof=1):.3f})" for c in ['90_original','90_rotated','60_original','60_rotated'])
        ang=(w['60_original']+w['60_rotated'])/2-(w['90_original']+w['90_rotated'])/2
        rot=(w['90_rotated']+w['60_rotated'])/2-(w['90_original']+w['60_original'])/2
        inter=(w['60_rotated']-w['60_original'])-(w['90_rotated']-w['90_original'])
        p=[signflip(ang.values),signflip(rot.values),signflip(inter.values)]
        print(m,desc,f"| ang {ang.mean():.3f} raw={p[0]:.5f} holm2={holm(p[:2])[0]:.4f} holm3={holm(p)[0]:.4f} | rot {rot.mean():.3f} raw={p[1]:.4f} holm2={holm(p[:2])[1]:.3f} holm3={holm(p)[1]:.3f} | inter raw={p[2]:.3f}")
# backend comparison
print('\n== Heron vs Nighthawk paired t, Holm over 4 conditions')
for m in ['qpu_mean_normalized_gap','qpu_cvar10_normalized_gap','best_agreement_with_exact']:
    ps=[];out=[]
    for c in ['90_original','90_rotated','60_original','60_rotated']:
        h=d[(d.platform=='Heron')&(d.condition_key==c)].set_index('unit_id')[m]
        n=d[(d.platform=='Nighthawk')&(d.condition_key==c)].set_index('unit_id')[m]
        t=stats.ttest_rel(n,h.loc[n.index]); ps.append(t.pvalue); out.append((c,(n-h).mean()))
    print(m,[(c,round(x,3),round(p,3)) for (c,x),p in zip(out,holm(ps))])
m='penalty'
d['penalty']=d.qpu_mean_normalized_gap-d.ideal_mean_normalized_gap
ps=[];out=[]
for c in ['90_original','90_rotated','60_original','60_rotated']:
    h=d[(d.platform=='Heron')&(d.condition_key==c)].set_index('unit_id')[m]; n=d[(d.platform=='Nighthawk')&(d.condition_key==c)].set_index('unit_id')[m]
    t=stats.ttest_rel(n,h.loc[n.index]); ps.append(t.pvalue); out.append((c,(n-h).mean()))
print(m,[(c,round(x,3),round(p,5)) for (c,x),p in zip(out,holm(ps))])
print('Heron penalty by cond', d[d.platform=='Heron'].groupby('condition_key').penalty.agg(['mean','min','max']).round(3).to_dict())
print('Nighthawk penalty by cond', d[d.platform=='Nighthawk'].groupby('condition_key').penalty.agg(['mean','min','max']).round(3).to_dict())
