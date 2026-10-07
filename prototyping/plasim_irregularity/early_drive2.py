import sys, numpy as np, matplotlib; matplotlib.use("Agg")
from pathlib import Path
from gsebm import plasim_koopman_irregularity as ir
root=Path("data/Plasim"); out=Path(sys.argv[1])
res={m:ir.irregularity_analysis(root,m) for m in ("1240","1232p5")}
ir.make_residual_koopman_figure(res).savefig(out/"rk.png",dpi=80)
ir.make_coupling_figure(res).savefig(out/"coupling.png",dpi=80)
ir.make_comparison_figure({m:{"full":r["summary"],"blocks":r["block_summaries"]} for m,r in res.items()}).savefig(out/"comparison.png",dpi=60)
for k in res["1240"]["summary"]:
    if "Floquet" in k or "residual KDMD" in k: print(k,[round(res[m]["summary"].get(k,np.nan),4) for m in res])
