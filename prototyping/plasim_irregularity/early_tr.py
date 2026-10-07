import numpy as np, pickle, sys
from gsebm import plasim_koopman_irregularity as ir
S=sys.argv[1]
for mu in ("1240","1232p5"):
    r=pickle.load(open(f"{S}/r_{mu}.pkl","rb")); ph=r["phase"]
    k=ir.fit_residual_koopman(r["eofs"]["pcs"],ph,lag=1); i=ir.transverse_modes(k)[0]
    phi=k["eigenfunctions"][:,i]; phi=(phi-phi.mean()); phi=phi/np.sqrt(np.mean(abs(phi)**2))
    print(mu, "rate",k["rates"][i].round(4))
    for b in ("surface","ocean"):
        R=r["blocks"][b]["residuals"]["amplitude"]; reg=(np.conj(phi)[:,None]*R).mean(0)
        lat=r["blocks"][b]["lat"]; v=np.real(reg)
        print(" ",b,"pattern (lat:value)",[(round(lat[j]),round(v[j],2)) for j in range(0,len(lat),3)])
    # where along cycle is phi variance / its increments
    st=ir.phase_conditioned_stats(np.abs(phi)**2,ph,12); print("  |phi|² by phase",np.round(st["mean"],2))
    inc=np.real(phi[1:]*0+phi[1:]-np.exp(k['rates'][i])*phi[:-1])
    st=ir.phase_conditioned_stats(inc**2,ph[:-1],12); print("  forcing var by phase",np.round(st["mean"]/st["mean"].mean(),2))
    fl=r["floquet"]; print("  floquet local growth max at phase",round(fl["phase_grid"][np.argmax(fl["local_growth"])]/2/np.pi,2), np.round(fl["local_growth"][::3],2))
    # cycle amplitude
    f=r["fields"]; G=1e3*f["ocean_anomaly"]; idx=ir.phase_bin_index(ph,36); C=ir._group_means(G,idx,36)
    lat=r["blocks"]["ocean"]["lat"]; band=(lat>=-27)&(lat<=-22); print("  θ 22-27 cycle std",round(C[idx][:,band].mean(1).std(),1)," resid std",round(r['band_residuals']['θ 0–700 m 22–27°S'].std(),1))
    band=(lat>=-42)&(lat<=-33); print("  θ 33-42 cycle std",round(C[idx][:,band].mean(1).std(),1)," resid std",round(r['band_residuals']['θ 0–700 m 33–42°S'].std(),1))
