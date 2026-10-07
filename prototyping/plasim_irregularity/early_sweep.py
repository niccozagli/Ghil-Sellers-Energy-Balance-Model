import sys, pickle, numpy as np, matplotlib; matplotlib.use("Agg")
from pathlib import Path
from gsebm import plasim_koopman_irregularity as ir
out=Path(sys.argv[1]); sw={}
for mu in ("1240","1232p5"):
    sw[mu]=ir.robustness_sweep(Path("data/Plasim"),mu,progress=lambda m:print(m,flush=True))
pickle.dump(sw,open(out/"sweep.pkl","wb"))
ir.make_robustness_figure(sw).savefig(out/"robust.png",dpi=70)
metrics=list(sw["1240"]["baseline"])
for m in metrics:
    print(f"\n{m}")
    for v in sw["1240"]:
        a,b=sw["1240"][v][m],sw["1232p5"][v][m]
        print(f"  {v:22s} {a:9.4g} {b:9.4g}  ratio {b/a if a else np.nan:6.2f}")
