import numpy as np, pickle, sys
from scipy.signal import welch
from gsebm import plasim_koopman_irregularity as ir
S=sys.argv[1]
for mu in ("1240","1232p5"):
    r=pickle.load(open(f"{S}/r_{mu}.pkl","rb")); f=r["fields"]; ph=r["phase"]; P=r["period"]; psi=r["psi1"]; y=f["years"]
    print(f"\n===== {mu}  P={P:.1f}")
    # ratio residual std / cycle std per row
    for b,key,sc in (("surface","surface_anomaly",1),("ocean","ocean_anomaly",1e3)):
        G=sc*f[key]; idx=ir.phase_bin_index(ph,36)
        C=ir._group_means(G,idx,36); cyc=C[idx].std(0); res=r["blocks"][b]["residuals"]["amplitude"].std(0)
        lat=r["blocks"][b]["lat"]
        order=np.argsort(-res/cyc)[:4]
        print(b,"rows max res/cycle:",[(round(lat[i],1),round(res[i]/cyc[i],2),round(res[i],2),round(cyc[i],2)) for i in order])
        st=ir.phase_conditioned_stats(r["blocks"][b]["residuals"]["amplitude"],ph,12)
        w=r["blocks"][b]["weights"]; v=st["variance"]@w
        print(b,"var by phase (12 bins, /mean):",np.round(v/v.mean(),2))
        # mean cycle of that block at band with max residual
    # ice composite vs phase
    st=ir.phase_conditioned_stats(f["ice_anomaly"],ph,12); print("ice mean cycle:",np.round(st["mean"],3))
    st=ir.phase_conditioned_stats(r["band_residuals"]["SA ice area"],ph,12); print("ice resid var /mean:",np.round(st["variance"]/st["variance"].mean(),2))
    # residual spectra partition
    for n,s in r["band_residuals"].items():
        fr,pw=welch(s,nperseg=1024,detrend="linear"); per=1/np.maximum(fr,1e-9)
        tot=pw[1:].sum()
        print(f"  {n:22s} slow(>1.5P) {pw[(per>1.5*P)].sum()/tot:.2f}  nearP {pw[(per>0.7*P)&(per<=1.5*P)].sum()/tot:.2f}  fast(<P/2) {pw[(per<P/2)&(fr>0)].sum()/tot:.2f}")
    # lagged xcorr reservoir theta residual -> ice residual
    a=r["band_residuals"]["θ 0–700 m 22–27°S"]; b=r["band_residuals"]["SA ice area"]; e=r["band_residuals"]["edge Ts 28–45°S"]
    lags=range(-30,31,5)
    print("corr(θres(t), ice res(t+L)) L=-30..30:",[round(np.corrcoef(a[max(0,-L):len(a)-max(0,L)],b[max(0,L):len(b)-max(0,-L)])[0,1],2) for L in lags])
    # cycle amplitude vs length
    edges=ir.cycle_crossings(psi,y); L=np.diff(edges); amp=[np.abs(psi[(y>=s0)&(y<s1)]).mean() for s0,s1 in zip(edges[:-1],edges[1:])]
    res_mean=[a[(y>=s0)&(y<s1)].mean() for s0,s1 in zip(edges[:-1],edges[1:])]
    print("corr(cycle amp, length)",round(np.corrcoef(amp,L)[0,1],2)," corr(reservoir resid mean, length)",round(np.corrcoef(res_mean,L)[0,1],2), " corr(prev-cycle res, next length)", round(np.corrcoef(res_mean[:-1],L[1:])[0,1],2))
    # leading residual EOF pattern
    e0=r["eofs"]; print("EOF fractions",np.round(e0["fractions"][:4],2))
    for k in range(2):
        ps,po=e0["patterns"][0][k],e0["patterns"][1][k]
        print(f" EOF{k+1} Ts peak lat {r['blocks']['surface']['lat'][np.argmax(abs(ps))]:.0f}, θ peak lat {r['blocks']['ocean']['lat'][np.argmax(abs(po))]:.0f}; θ pattern",np.round(po[::3],0))
    # slow mode pattern
    print("ocean lats",np.round(r["blocks"]["ocean"]["lat"][::3],0))
