import time, sys, numpy as np
from pathlib import Path
from gsebm import plasim_koopman_single as k, plasim_koopman_irregularity as ir
root=Path("data/Plasim"); mu=sys.argv[1]
s,e=ir.ANALYSIS_WINDOWS[mu]; t=time.time()
f=k.load_fields(root,s-1,700.0,mu,end_year=e); print("load",time.time()-t,f["years"][[0,-1]],f["state"].shape)
sp,rates,rank,_=k.fit_koopman(f,5,5e-5,10000,0,1e-5); print("fit",time.time()-t,rank)
li,p1,hi,p2,sc=k.leading_and_harmonic_eigenfunctions(sp,rates,f); w=rates[li].imag; print("period",2*np.pi/w, time.time()-t)
sm=k.state_mode_comparison(sp,f,rates,li,p1,hi,p2,sc,lag=5,maximum_training_snapshots=10000,seed=0)
for name,key,scale in (("surface","surface_anomaly",1),("ocean","ocean_anomaly",1e3)):
    r=ir.cycle_residuals(scale*f[key],p1,p2,sm[name]["direct"]); print(name,r["fractions"])
print(ir.irregularity_summary(p1,f["years"],{}))
ev=sp.evaluate_eigenfunctions(f["state"],batch_size=256); print("eval",ev.shape,time.time()-t)
idx=ir.slow_modes(rates,li); print(rates[idx])
