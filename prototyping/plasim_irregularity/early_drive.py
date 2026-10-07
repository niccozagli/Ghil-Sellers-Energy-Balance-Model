import sys, time, json, numpy as np, matplotlib
matplotlib.use("Agg")
from pathlib import Path
from gsebm import plasim_koopman_irregularity as ir
from gsebm.plasim_koopman_single import load_map_fields
root=Path("data/Plasim"); out=Path(sys.argv[1]); res={}
for mu in ("1240","1232p5"):
    t=time.time(); r=ir.irregularity_analysis(root,mu); res[mu]=r; print(mu,"analysis",time.time()-t)
    y=r["fields"]["years"]
    ir.make_clock_figure(r["psi1"],y,r["omega"],mu=mu).savefig(out/f"{mu}_clock.png",dpi=80)
    ir.make_state_residual_figure(r["blocks"],r["phase"],mu=mu).savefig(out/f"{mu}_state.png",dpi=80)
    ir.make_band_residual_figure(r["band_residuals"],r["phase"],y,mu=mu).savefig(out/f"{mu}_bands.png",dpi=80)
    maps=load_map_fields(root,mu,y)
    fl=ir.ice_flip_phases(ir.load_absolute_map(root,mu,y),r["psi1"])
    ir.make_flip_figure(fl,maps,period=r["period"],mu=mu).savefig(out/f"{mu}_flips.png",dpi=80)
    ir.make_map_residual_figure(maps,r["psi1"],mu=mu).savefig(out/f"{mu}_maps.png",dpi=70)
    print(mu,"figs",time.time()-t)
    print({n:round(v["correlation"],2) for n,v in r["next_cycle"].items()})
    print("slow",r["rates"][r["slow_indices"]].round(4), r["slow_projection"]["per_mode"].round(3))
    print("eof",r["eofs"]["fractions"].sum().round(2),"floq",r["floquet"]["per_cycle"][:4].round(3), (-1/r["floquet"]["rate"][:3]).round(1))
    del maps
ir.make_comparison_figure({m:{"full":r["summary"],"blocks":r["block_summaries"]} for m,r in res.items()}).savefig(out/"comparison.png",dpi=70)
ir.make_floquet_figure({m:r["floquet"] for m,r in res.items()}).savefig(out/"floquet.png",dpi=80)
for k in res["1240"]["summary"]: print(f"{k:45s}", " ".join(f"{res[m]['summary'][k]:10.4g}" for m in res))
