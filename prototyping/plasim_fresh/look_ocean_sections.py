"""Plain look: window-mean Southern Ocean sections (10-yr blocks averaged over each window)."""

import matplotlib.pyplot as plt
import numpy as np

from common import equivalent_edge, load
from gsebm.plasim_global import RUNS

O10 = "/Users/niccolo/.claude/jobs/3fe717ba/tmp/ocean10/"
OUT = "../../figures/plasim_fresh"
TF = 271.25
labs = ["1312", "1265", "1245", "1235", "1232p5", "1230"]
fig, axes = plt.subplots(3, len(labs), figsize=(22, 11), sharey="row")
for j, lab in enumerate(labs):
    z = np.load(O10 + lab + ".npz"); w0, w1 = RUNS[lab].window
    k = (z["block_start"] >= w0) & (z["block_start"] + 9 <= w1)
    lat, vlat = z["lsg_lat"], z["lsg_vector_lat"]
    zc = z["depth_bounds"].mean(axis=1); zi = z["depth_bounds"][:, 1]
    d = load(lab); kk = (d["years"] >= w0) & (d["years"] <= w1); E = np.nanmean(equivalent_edge(d)[kk])
    s = lat < -10; sv = vlat < -10
    th = np.nanmean(z["potential_temperature"][k], axis=0) - TF
    cv = np.nanmean(z["convective_adjustment"][k], axis=0)
    psi = np.nancumsum(np.nanmean(z["meridional_volume_transport"][k], axis=0), axis=1) / 1e6
    im0 = axes[0, j].pcolormesh(lat[s], zc, th[s].T, vmin=0, vmax=6, cmap="viridis", shading="nearest")
    im1 = axes[1, j].pcolormesh(lat[s], zc, cv[s].T, vmin=0, vmax=0.3, cmap="magma", shading="nearest")
    im2 = axes[2, j].pcolormesh(vlat[sv], zi, psi[sv].T, vmin=-20, vmax=20, cmap="RdBu_r", shading="nearest")
    axes[2, j].contour(vlat[sv], zi, psi[sv].T, levels=np.arange(-30, 31, 5), colors="k", linewidths=0.4)
    for i in range(3):
        axes[i, j].axvline(-E, color="w" if i < 2 else "k", ls="--", lw=1.2)
        axes[i, j].set_ylim(3000, 0)
    axes[0, j].set_title(f"μ = {d['mu']:g} (dashed: ice edge {E:.1f}°S)", fontsize=9)
    axes[2, j].set_xlabel("latitude")
axes[0, 0].set_ylabel("θ − 271.25 K, depth (m)"); axes[1, 0].set_ylabel("convection frequency, depth (m)")
axes[2, 0].set_ylabel("overturning ψ (Sv), depth (m)")
fig.colorbar(im0, ax=axes[0, :], shrink=0.8, label="K above freezing")
fig.colorbar(im1, ax=axes[1, :], shrink=0.8, label="fraction of steps with convective adjustment")
fig.colorbar(im2, ax=axes[2, :], shrink=0.8, label="Sv (positive = clockwise)")
fig.suptitle("Plain look: Southern Ocean window means (zonal means over all longitudes)")
fig.savefig(f"{OUT}/look_ocean_sections.png", dpi=95); plt.close(fig)
print("ok")
