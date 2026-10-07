import numpy as np, pickle, sys
from gsebm import plasim_koopman_irregularity as ir
S=sys.argv[1]
for mu in ("1240","1232p5"):
    r=pickle.load(open(f"{S}/r_{mu}.pkl","rb")); psi=r["psi1"]; y=r["fields"]["years"]
    print("==",mu)
    edges=ir.cycle_crossings(psi,y); L=np.diff(edges)
    print(" cycle lengths:",np.round(L).astype(int).tolist())
    amp=np.abs(psi); 
    print(" |psi| 200-yr means:",np.round([amp[i:i+200].mean() for i in range(0,4000,200)],2).tolist())
    th=np.unwrap(np.angle(psi)); adv=np.diff(th)
    print(" backward phase steps:",int((adv<0).sum()), " years with |psi|<0.3:",int((amp<0.3).sum()))
    for a,b in ((0,2000),(2000,4000)):
        print(f"  half {a}-{b}: D={ir.phase_diffusion(psi[a:b])['diffusion']:.4f}  CV={np.std(ir.cycle_lengths(psi[a:b],y[a:b]))/np.mean(ir.cycle_lengths(psi[a:b],y[a:b])):.3f}")
