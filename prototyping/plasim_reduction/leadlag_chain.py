"""Lead-lag chain at the Southern edge, annual anomalies, run segments as in step 3.

Southern quantities (Gaussian weights over Southern T21 rows; LSG rows weighted by wet area):
  R    = ocean heat release at the edge = -(ocean uptake), mean over LSG rows within +-10 deg of the segment-mean S edge
  I    = ice loss = -(sum_j w_j * zonal ocean ice concentration_j) over S rows
  A    = albedo term = -sum_j w_j * mean(I_j) * (alpha_j - mean alpha_j) over S rows (W m-2 of S hemisphere)
  T    = S-mean surface temperature; O = S-mean OLR; U = ocean uptake over all S LSG rows (W m-2 of S ocean)
Atlantic sector: theta = SA upper-ocean 0-700 m (wet-area weighted), SI = -(SA ice area), ST = SA surface temperature.
corr(Y(t+k), X(t)): k > 0 means Y follows X. Peak = lag of max |corr| within +-10.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gsebm.plasim_global import ice_edge_latitude
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/states/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1233p75', '1232p5', '1235_new_IC']
SEGMENT = {'1245': (0, None), '1242p5': (500, None), '1240': (0, None), '1237p5': (0, None), '1235': (0, None),
           '1235_new_IC': (2499, None), '1233p75': (0, None), '1232p5': (0, 3000)}
K = 10; ks = np.arange(-K, K + 1)
def lagcorr(Y, X):
    Y = Y - Y.mean(); X = X - X.mean(); out = []
    for k in ks:
        a, b = (Y[k:], X[:len(X) - k]) if k >= 0 else (Y[:k], X[-k:])
        out.append(np.corrcoef(a, b)[0, 1])
    return np.array(out)
PAIRS = [('R', 'I', 'edge ocean release → S ice loss'), ('I', 'A', 'S ice loss → S albedo term'),
         ('A', 'T', 'S albedo term → S surface T'), ('I', 'T', 'S ice loss → S surface T'),
         ('T', 'O', 'S surface T → S OLR'), ('T', 'U', 'S surface T → S ocean uptake'),
         ('theta', 'SI', 'SA θ 0–700 → SA ice loss'), ('SI', 'ST', 'SA ice loss → SA surface T')]
fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True); axes = axes.ravel()
summary = {p[2]: [] for p in PAIRS}
for lab in LABELS:
    d = dict(np.load(TMP + lab + '.npz')); n = len(d['years']); s0, s1 = SEGMENT[lab]; g = slice(s0, s1 if s1 else n)
    lat, w = d['lat'], d['weight']; S = lat < 0; wS = w[S] / w[S].sum()
    ice = np.nan_to_num(d['ice'][g])
    edge = float(np.squeeze(ice_edge_latitude(ice.mean(0)[None], lat, south=True)))
    ll, area = d['lsg_lat'], d['lsg_area']
    band = (ll < 0) & (np.abs(np.abs(ll) - edge) <= 10) & (area > 0)
    Sall = (ll < 0) & (area > 0)
    up = d['uptake'][g]
    asr, ins = d['asr'][g], d['ins'][g]; alb = 1 - asr / ins
    V = dict(R=-(up[:, band] @ area[band]) / area[band].sum(), I=-(ice[:, S] @ wS),
             A=-(ins.mean(0)[S] * (alb[:, S] - alb[:, S].mean(0))) @ wS, T=d['Ts'][g][:, S] @ wS,
             O=d['olr'][g][:, S] @ wS, U=(up[:, Sall] @ area[Sall]) / area[Sall].sum(),
             theta=d['sa_ocean'][g] @ d['sa_ocean_w'], SI=-d['sa_ice'][g], ST=d['sa_surface'][g] @ d['sa_surface_w'])
    for ax, (x, y, name) in zip(axes, PAIRS):
        c = lagcorr(V[y], V[x]); h = len(V[x]) // 2
        c1 = lagcorr(V[y][:h], V[x][:h]); c2 = lagcorr(V[y][h:], V[x][h:])
        k = ks[np.argmax(np.abs(c))]
        summary[name].append((lab, int(k), float(c[ks == k][0]), int(ks[np.argmax(np.abs(c1))]), int(ks[np.argmax(np.abs(c2))]), float(c[K])))
        ax.plot(ks, c, lw=1, label=lab.replace('p', '.'))
    print(f'{lab:12s} S edge {edge:.1f}°, edge band rows {band.sum()}')
for ax, (_, _, name) in zip(axes, PAIRS):
    ax.axhline(0, color='k', lw=0.5); ax.axvline(0, color='k', lw=0.5); ax.set_title(name, fontsize=9)
axes[0].legend(fontsize=7); [a.set_xlabel('lag k (yr); k > 0: second follows first') for a in axes[4:]]
fig.suptitle('Lead–lag chain at the Southern edge (annual anomalies, no smoothing)'); fig.tight_layout()
fig.savefig('figures/plasim_mechanism/leadlag_chain.png', dpi=110)
for name, rows in summary.items():
    print(f'\n{name}: (run: peak lag, r at peak | peak lag in halves | r at lag 0)')
    print('   ' + '; '.join(f'{l}: {k:+d}, {r:+.2f} | {k1:+d}/{k2:+d} | {r0:+.2f}' for l, k, r, k1, k2, r0 in rows))
