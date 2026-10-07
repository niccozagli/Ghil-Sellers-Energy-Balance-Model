import numpy as np, sys
sys.path.insert(0,"tests")
from test_plasim_koopman_irregularity import noisy_clock
from gsebm import plasim_koopman_irregularity as ir
for n in (1500,4000):
    rng=np.random.default_rng(10); psi=noisy_clock(rng,years=n,period=40.0,phase_noise=0.02); ph=ir.wrapped_phase(psi)
    pcs=np.zeros((n,2))
    for t in range(1,n): pcs[t]=np.array([0.85,0.4])*pcs[t-1]+rng.standard_normal(2)
    for lag in (1,):
        k=ir.fit_residual_koopman(pcs,ph,lag=lag); print(n,"rank",k["rank"])
        for i in ir.transverse_modes(k,count=6): print(f"  {k['rates'][i]:.4f} clock {k['clock_share'][i]:.2f} resid {k['residual_share'][i]:.2f}")
