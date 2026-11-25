import sys, json, numpy as np, pandas as pd, networkx as nx
from scipy.optimize import minimize
from pathlib import Path
R=str(Path(__file__).resolve().parents[2] / 'src')
sys.path[:0]=[R+'/qaoa',R+'/phantom',R+'/hardware/heron',R]
import step19_harder_phantoms as st
from step9_common import graph_coefficients, normalized_gap
G0,B0=st.GAMMA,st.BETA
def prep(gr,p):
    z,zz=graph_coefficients(gr); G=nx.Graph(); G.add_nodes_from(range(gr.n_nodes)); G.add_edges_from((i,j) for i,j,_ in zz)
    terms=[([i],c) for i,c in enumerate(z) if abs(c)>1e-15]+[([i,j],c) for i,j,c in zz]
    jobs=[]
    for q,c in terms:
        ball=set()
        for v in q: ball|=set(nx.single_source_shortest_path_length(G,v,p))
        ball=sorted(ball); idx={v:k for k,v in enumerate(ball)}; k=len(ball)
        bits=((np.arange(2**k)[:,None]>>np.arange(k)[None,:])&1); spins=1-2*bits
        diag=sum(z[v]*spins[:,idx[v]] for v in ball)
        for i,j,w in zz:
            if i in idx and j in idx: diag=diag+w*spins[:,idx[i]]*spins[:,idx[j]]
        obs=c*np.prod([spins[:,idx[v]] for v in q],axis=0)
        jobs.append((k,diag,obs))
    return jobs
def mixer(psi,k,b):
    c,s=np.cos(b),-1j*np.sin(b); psi=psi.reshape([2]*k)
    for a in range(k):
        psi=np.moveaxis(psi,a,0); p0,p1=psi[0].copy(),psi[1].copy(); psi[0]=c*p0+s*p1; psi[1]=s*p0+c*p1; psi=np.moveaxis(psi,0,a)
    return psi.reshape(-1)
def energy(jobs,gs,bs):
    E=0.0
    for k,diag,obs in jobs:
        psi=np.full(2**k,2**(-k/2),complex)
        for ga,be in zip(gs,bs): psi=mixer(psi*np.exp(-1j*ga*diag),k,be)
        E+=float(np.real(np.vdot(psi,obs*psi)))
    return E
design=[(g,s) for g,s,r in st.DESIGN if g in ('c90_k50','c60_k60')]
D={}
for g,s in design:
    gr,_=st.build_graph(g,s); _,E,_=st.exact_milp(gr); D[(g,s)]=(E,prep(gr,1),prep(gr,2))
k0=design[0]; print('p1 check',normalized_gap(energy(D[k0][1],[G0],[B0]),D[k0][0]),'(expect 0.2523)',flush=True)
train=[k for k in design if k[1] in (2001,2002,2003)]
obj=lambda x: np.mean([normalized_gap(energy(D[k][2],x[:2],x[2:]),D[k][0]) for k in train])
best=None
for x0 in ([0.6*G0,1.1*G0,1.1*B0,0.6*B0],[0.5*G0,G0,B0,0.5*B0],[0.8*G0,0.8*G0,B0,0.8*B0],[G0,G0,B0,B0]):
    r=minimize(obj,x0,method='Nelder-Mead',options={'maxiter':400,'xatol':1e-3,'fatol':1e-5})
    print('start',np.round(x0,3),'->',np.round(r.x,4),round(r.fun,4),flush=True)
    if best is None or r.fun<best.fun: best=r
x=best.x; json.dump({'gammas':x[:2].tolist(),'betas':x[2:].tolist()},open(sys.argv[1]+'/p2_angles.json','w'))
rows=[dict(geometry=k[0],seed=k[1],train=k in train,gap_p1=normalized_gap(energy(D[k][1],[G0],[B0]),D[k][0]),
           gap_p2=normalized_gap(energy(D[k][2],x[:2],x[2:]),D[k][0])) for k in design]
d=pd.DataFrame(rows); d.to_csv(sys.argv[1]+'/p2_seed_level.csv',index=False)
print(d.groupby('geometry')[['gap_p1','gap_p2']].mean().round(4).to_string())
print(d[~d.train].groupby('geometry')[['gap_p1','gap_p2']].mean().round(4).to_string())
