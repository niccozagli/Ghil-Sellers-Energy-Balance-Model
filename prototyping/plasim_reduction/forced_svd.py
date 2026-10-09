"""PlaSim forced response along the warm branch: SVD of window means + physical regressions.

Window means (plain time means over each run's analysis window, annual values) of:
  * zonal surface temperature, all longitudes, 32 Gaussian rows (weight = Gaussian weight);
  * zonal upper-ocean potential temperature 0-700 m, all longitudes (`zonal_potential_temperature`,
    wet-volume-weighted over levels whose bottom is <= 700 m; weight = wet surface area of the row).
SVD: run means centred over runs; each feature scaled by sqrt(area weight); each block divided by its
total across-run standard deviation so the blocks count equally ("Ts+ocean"); also Ts alone.
Amplitude a is rescaled so that da = 1 corresponds to 1 K of global-mean Ts.
Regressions across runs (least squares on a) give per-row "change per K of the mode".
"""
import numpy as np, h5py, json
from pathlib import Path
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.plasim_global import RUNS, run_mu, run_archive, read_global_series, time_mean, ice_edge_latitude

ROOT = Path('data/Plasim')
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
OUT = Path('figures/plasim_mechanism'); OUT.mkdir(parents=True, exist_ok=True)

def ocean_upper(label):
    with h5py.File(run_archive(ROOT, label), 'r') as f:
        years = f['year'][:]; a, b = RUNS[label].window
        idx = np.flatnonzero((years >= a) & (years <= b))
        bounds = f['depth_bounds'][:]; vol = f['wet_volume'][:]; area = f['wet_surface_area'][:]; lat = f['lsg_lat'][:]
        levels = np.flatnonzero(bounds[:, 1] <= 700.0)
        acc = np.zeros(f['zonal_potential_temperature'].shape[1:]); n = 0
        for k in range(idx[0], idx[-1] + 1, 200):
            blk = f['zonal_potential_temperature'][k:min(k + 200, idx[-1] + 1)]
            acc += blk.sum(0); n += blk.shape[0]
    mean = acc / n
    w = vol[:, levels]
    theta = np.where(w.sum(1) > 0, (mean[:, levels] * w).sum(1) / np.maximum(w.sum(1), 1e-30), np.nan)
    return lat, area, theta

rows = {}
for lab in LABELS:
    s = read_global_series(ROOT, lab)
    lat_o, area_o, theta = ocean_upper(lab)
    rows[lab] = dict(mu=run_mu(lab), lat=s.lat, w=s.weight, Ts=time_mean(s.surface_temperature),
                     ice=time_mean(s.ocean_ice), asr=time_mean(s.absorbed_shortwave), ins=time_mean(s.insolation),
                     olr=time_mean(s.outgoing_longwave), lsg_lat=s.lsg_lat, lsg_area=s.lsg_row_area,
                     uptake=time_mean(s.ocean_heat_uptake), theta=theta, area_o=area_o, lat_o=lat_o)
    print(lab, 'read', flush=True)

lat = rows[LABELS[0]]['lat']; w = rows[LABELS[0]]['w']
mu = np.array([rows[l]['mu'] for l in LABELS])
Ts = np.array([rows[l]['Ts'] for l in LABELS])
theta = np.array([rows[l]['theta'] for l in LABELS]); lat_o = rows[LABELS[0]]['lat_o']; area_o = rows[LABELS[0]]['area_o']
wet = np.isfinite(theta).all(0) & (area_o > 0)
theta, lat_o, area_o = theta[:, wet], lat_o[wet], area_o[wet] / area_o[wet].sum()
Tg = Ts @ w

def svd(blocks):
    cols = []
    for X, wt in blocks:
        Z = (X - X.mean(0)) * np.sqrt(wt)
        cols.append(Z / np.sqrt((Z**2).sum()))
    M = np.column_stack(cols)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    amp = U[:, 0] * S[0]
    k = np.polyfit(amp, Tg, 1)[0]   # rescale: 1 unit = 1 K global-mean Ts
    return S**2 / (S**2).sum(), amp * k

res = {}
for name, blocks in (('Ts', [(Ts, w)]), ('Ts+ocean', [(Ts, w), (theta, area_o)])):
    frac, a = svd(blocks)
    res[name] = a
    print(f'\nSVD {name}: variance fractions {np.round(frac[:4], 3)}')
    print('  amplitude (K of global Ts) by run:', ', '.join(f'{l}: {v:+.2f}' for l, v in zip(LABELS, a - a.mean())))
a = res['Ts+ocean']; a = a - a.mean()
print(f"  corr(amplitude Ts-only, Ts+ocean) = {np.corrcoef(res['Ts'], res['Ts+ocean'])[0,1]:.4f}; corr(amplitude, mu) = {np.corrcoef(a, mu)[0,1]:.4f}")

def reg(Y):
    Y = np.asarray(Y); A = np.column_stack([np.ones_like(a), a])
    coef, resid, *_ = np.linalg.lstsq(A, Y.reshape(len(a), -1), rcond=None)
    fit = A @ coef; r2 = 1 - ((Y.reshape(len(a), -1) - fit)**2).sum(0) / ((Y.reshape(len(a), -1) - Y.reshape(len(a), -1).mean(0))**2).sum(0)
    return coef[1].reshape(Y.shape[1:]), r2.reshape(Y.shape[1:])

ice = np.array([rows[l]['ice'] for l in LABELS]); asr = np.array([rows[l]['asr'] for l in LABELS])
ins = np.array([rows[l]['ins'] for l in LABELS]); olr = np.array([rows[l]['olr'] for l in LABELS])
alb = 1 - asr / ins
upt = np.array([rows[l]['uptake'] for l in LABELS]); lsg_lat = rows[LABELS[0]]['lsg_lat']; lsg_area = rows[LABELS[0]]['lsg_area']
dTs, r2Ts = reg(Ts); dth, _ = reg(theta); dice, _ = reg(np.nan_to_num(ice))
dalb, _ = reg(alb); dins, _ = reg(ins); dolr, _ = reg(olr); dasr, _ = reg(asr); dupt, _ = reg(upt)
albedo_term = -ins.mean(0) * dalb          # feedback part of dASR
forcing_term = (1 - alb.mean(0)) * dins     # direct insolation (mu) part
dN = dasr - dolr                            # TOA net per row, = convergence of total transport (steady)
G = lambda f: float(f @ w)
print('\nPer K of global-mean Ts along the forced direction (window means across runs):')
print(f'  OLR {G(dolr):+.2f}  albedo term {G(albedo_term):+.2f}  insolation term {G(forcing_term):+.2f}  '
      f'ASR {G(dasr):+.2f}  (check: albedo+insolation {G(albedo_term + forcing_term):+.2f})  W m-2 K-1')
print(f'  net feedback lambda = dOLR - albedo term = {G(dolr) - G(albedo_term):+.2f} W m-2 K-1')
edgeS = [float(np.squeeze(ice_edge_latitude(ice[i][None], lat, south=True))) for i in range(len(LABELS))]
edgeN = [float(np.squeeze(ice_edge_latitude(ice[i][None], lat, south=False))) for i in range(len(LABELS))]
print(f'  S edge {np.polyfit(a, edgeS, 1)[0]:+.2f} deg per K, N edge {np.polyfit(a, edgeN, 1)[0]:+.2f} deg per K (positive = poleward when warmer)')
S, N = lat < 0, lat > 0
print(f'  Ts change per K global: South mean {dTs[S] @ w[S] / w[S].sum():.2f}, North mean {dTs[N] @ w[N] / w[N].sum():.2f}; '
      f'max {dTs.max():.2f} K at {lat[np.argmax(dTs)]:.1f}, min {dTs.min():.2f} K at {lat[np.argmin(dTs)]:.1f}')
print(f'  Ts row fit R2: min {np.nanmin(r2Ts):.2f}, median {np.nanmedian(r2Ts):.2f}')
print('  ocean uptake change at S rows (W m-2 per K):', ', '.join(f'{la:.0f}:{v:+.1f}' for la, v in zip(lsg_lat, dupt) if -60 < la < -20 and lsg_area[np.argmin(abs(lsg_lat-la))] > 0)[:400])
json.dump(dict(labels=LABELS, mu=mu.tolist(), a=a.tolist()), open('/Users/niccolo/.claude/jobs/96e03936/tmp/forced_amp.json', 'w'))

fig, ax = plt.subplots(2, 2, figsize=(12, 8))
ax[0, 0].plot(mu, a, 'o', color='#2a6fb0')
for m_, a_, l in zip(mu, a, LABELS): ax[0, 0].annotate(l.replace('p', '.'), (m_, a_), fontsize=7, xytext=(3, 3), textcoords='offset points')
ax[0, 0].set(xlabel='μ (W m⁻²)', ylabel='mode amplitude (K of global-mean Ts)', title='Forced-response amplitude of each run (window means)')
ax[0, 1].plot(lat, dTs, '-o', ms=3, color='#2a6fb0', label='surface temperature (K per K)')
ax[0, 1].plot(lat_o, dth, '-', color='#c0612b', label='upper-ocean θ 0–700 m (K per K)')
ax[0, 1].plot(lat, -dice * 10, '--', color='#6b6b6b', label='ice concentration × (−10) (per K)')
ax[0, 1].axhline(0, color='k', lw=0.5); ax[0, 1].set(xlabel='latitude', title='Pattern per K of global-mean Ts'); ax[0, 1].legend(fontsize=8)
ax[1, 0].plot(lat, albedo_term, color='#2a6fb0', label='albedo term −I·dα (feedback)')
ax[1, 0].plot(lat, forcing_term, color='#7a7a7a', label='insolation term (1−α)·dI (μ forcing)')
ax[1, 0].plot(lat, -dolr, color='#c0612b', label='−dOLR')
ax[1, 0].plot(lat, dN, color='k', lw=1, label='net TOA = −div(total transport)')
ax[1, 0].axhline(0, color='k', lw=0.5); ax[1, 0].set(xlabel='latitude', ylabel='W m⁻² per K', title='TOA energy changes per K along the forced response'); ax[1, 0].legend(fontsize=8)
ax[1, 1].plot(lsg_lat, dupt, color='#2a6fb0'); ax[1, 1].axhline(0, color='k', lw=0.5)
ax[1, 1].set(xlabel='latitude (LSG rows)', ylabel='W m⁻² per K (into ocean)', title='Ocean heat uptake change per K')
fig.tight_layout(); fig.savefig(OUT / 'forced_response_svd.png', dpi=120)
