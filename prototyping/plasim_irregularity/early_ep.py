import numpy as np, pickle, sys
from gsebm import plasim_koopman_irregularity as ir
S=sys.argv[1]
for mu in ("1240","1232p5"):
    r=pickle.load(open(f"{S}/r_{mu}.pkl","rb")); psi=r["psi1"]; y=r["fields"]["years"]; f=r["fields"]
    amp=np.convolve(np.abs(psi),np.ones(101)/101,mode="same")
    ep=amp<0.75; 
    # episodes
    d=np.diff(np.r_[0,ep.astype(int),0]); st=np.flatnonzero(d==1); en=np.flatnonzero(d==-1)
    print("==",mu,"episode years",[(int(y[a]),int(y[b-1])) for a,b in zip(st,en)], "fraction",ep.mean().round(2))
    lat=r["blocks"]["ocean"]["lat"]; w=f["ocean_weights"]
    def band(lo,hi): m=(lat>=lo)&(lat<=hi); return 1e3*f["ocean_anomaly"][:,m]@w[m]/w[m].sum()
    res=r["band_residuals"]["θ 0–700 m 22–27°S"]; ice=r["band_residuals"]["SA ice area"]
    if ep.any():
        print("  in episodes: ice anomaly %.3f (std %.3f)  θ22-27 %.0f mK  θ33-42 %.0f mK  edgeTs %.2f K" % (f["ice_anomaly"][ep].mean(), f["ice_anomaly"].std(), band(-27,-22)[ep].mean(), band(-42,-33)[ep].mean(), np.average(f["surface_anomaly"][:, (f["surface_lat"]>=-45)&(f["surface_lat"]<=-28)],axis=1)[ep].mean()))
    q=~ep; lags=np.arange(-15,16)
    c=ir.lagged_correlation(res[q],ice[q],lags) if q.all() else None
    # outside episodes: contiguous not guaranteed; use masked series by zeroing? compute on longest calm segment
    d=np.diff(np.r_[0,q.astype(int),0]); st=np.flatnonzero(d==1); en=np.flatnonzero(d==-1); k=np.argmax(en-st); a,b=st[k],en[k]
    c=ir.lagged_correlation(res[a:b],ice[a:b],lags)
    print(f"  longest calm segment {y[a]}-{y[b-1]}: coupling min {c.min():.2f} at {lags[np.argmin(c)]}, D={ir.phase_diffusion(psi[a:b])['diffusion']:.4f}, lag1 ice {np.sum(ice[a+1:b]*ice[a:b-1])/np.sum(ice[a:b-1]**2):.2f}, lag1 res {np.sum(res[a+1:b]*res[a:b-1])/np.sum(res[a:b-1]**2):.2f}")
    print(f"  full: lag1 res {np.sum(res[1:]*res[:-1])/np.sum(res[:-1]**2):.2f}")
