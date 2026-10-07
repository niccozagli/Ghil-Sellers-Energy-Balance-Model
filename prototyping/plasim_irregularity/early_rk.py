import numpy as np, time, pickle
from pathlib import Path
from gsebm import plasim_koopman_irregularity as ir
root=Path("data/Plasim")
for mu in ("1240","1232p5"):
    r=ir.irregularity_analysis(root,mu)
    pickle.dump({k:r[k] for k in ("psi1","psi2","phase","omega","period","blocks","band_residuals","eofs","floquet","rates","slow_indices","fields","leading_index","state_modes")}, open(f"{Path(__file__).parent}/r_{mu}.pkl","wb"))
    for lag in (1,5):
        t=time.time(); k=ir.fit_residual_koopman(r["eofs"]["pcs"],r["phase"],lag=lag)
        tr=ir.transverse_modes(k)
        print(mu,"lag",lag,"rank",k["rank"],f"{time.time()-t:.0f}s")
        for i in tr: print(f"   λ={k['rates'][i].real:+.4f}{k['rates'][i].imag:+.4f}i  tau={-1/k['rates'][i].real:6.1f}  clock={k['clock_share'][i]:.2f} resid={k['residual_share'][i]:.2f}")
        cl=np.flatnonzero((k["clock_share"]>0.8)&(k["rates"].imag>1e-3))
        cl=cl[np.argsort(-k["rates"].real[cl])][:3]
        print("   clock:",[f"{k['rates'][i].real:+.4f}{k['rates'][i].imag:+.4f}i" for i in cl], "omega",round(r["omega"],4))
    print("floquet rates", (r["floquet"]["rate"][:4]).round(4))
