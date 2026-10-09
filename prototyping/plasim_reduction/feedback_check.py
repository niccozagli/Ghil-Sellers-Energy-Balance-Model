"""Model-free check: are natural fluctuations radiatively damped? Annual values, run segments as in step 3.

Global means use Gaussian weights. Albedo term = -sum_j w_j * mean(I_j) * (alpha_j - mean alpha_j), alpha = 1 - ASR/I.
N = ASR - OLR (net down at TOA). Ocean uptake F_o = sum(row uptake * wet area) / sphere (into ocean, W m-2 of globe).
Ocean heat content H = global LSG OHC / sphere; dH/dt as the centred difference (H[t+1]-H[t-1])/2 per year.
Slopes are least squares on annual anomalies at lag 0. Lead-lag: corr(X(t+k), Tg(t)); k > 0 means X follows Tg.
"""
import numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/states/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1233p75', '1232p5', '1235_new_IC']
SEGMENT = {'1245': (0, None), '1242p5': (500, None), '1240': (0, None), '1237p5': (0, None), '1235': (0, None),
           '1235_new_IC': (2499, None), '1233p75': (0, None), '1232p5': (0, 3000)}
SPHERE = 4 * np.pi * 6.371e6**2; YEAR_S = 360 * 86400.0
def sl(y, x):
    x = x - x.mean(); return float((y - y.mean()) @ x / (x @ x))
def lagcorr(X, T, K=10):
    out = []
    for k in range(-K, K + 1):
        a, b = (X[k:], T[:len(T) - k]) if k >= 0 else (X[:k], T[-k:])
        out.append(np.corrcoef(a, b)[0, 1])
    return np.array(out)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2)); ks = np.arange(-10, 11)
print(f"{'run':12s} | per K of global Ts (W m-2 K-1): {'OLR':>5s} {'albedo':>6s} {'ASR':>5s} {'N':>6s} {'ocean uptake':>12s} {'dH/dt':>6s} | corr(N,Tg) lag0  N lag of extreme | corr(Fo,Tg) lag0 | S hemisphere per K of T_S: OLR albedo")
for lab in LABELS:
    d = dict(np.load(TMP + lab + '.npz')); n = len(d['years']); s0, s1 = SEGMENT[lab]; g = slice(s0, s1 if s1 else n)
    w = d['weight']; lat = d['lat']; S = lat < 0
    Ts, asr, ins, olr = d['Ts'][g], d['asr'][g], d['ins'][g], d['olr'][g]
    alb = 1 - asr / ins; albterm = -(ins.mean(0) * (alb - alb.mean(0)))
    Tg = Ts @ w; N = (asr - olr) @ w
    Fo = d['uptake'][g] @ d['lsg_area'] / SPHERE
    H = d['ohc_global'][g] / SPHERE; dH = np.r_[np.nan, (H[2:] - H[:-2]) / 2, np.nan] / YEAR_S
    ok = np.isfinite(dH)
    wS = w[S] / w[S].sum(); TS = Ts[:, S] @ wS
    r = dict(olr=sl(olr @ w, Tg), alb=sl(albterm @ w, Tg), asr=sl(asr @ w, Tg), N=sl(N, Tg), Fo=sl(Fo, Tg), dH=sl(dH[ok], Tg[ok]))
    cN = lagcorr(N - N.mean(), Tg - Tg.mean()); cF = lagcorr(Fo - Fo.mean(), Tg - Tg.mean())
    kN = ks[np.argmax(np.abs(cN))]
    print(f"{lab:12s} | {r['olr']:5.2f} {r['alb']:6.2f} {r['asr']:5.2f} {r['N']:+6.2f} {r['Fo']:+12.2f} {r['dH']:+6.2f} | {cN[10]:+.2f}  k={kN:+d} (r={cN[ks==kN][0]:+.2f}) | {cF[10]:+.2f} | {sl(olr[:, S] @ wS, TS):.2f} {sl(albterm[:, S] @ wS, TS):.2f}")
    ax[0].plot(ks, cN, label=lab.replace('p', '.')); ax[1].plot(ks, cF)
    ax[2].scatter(Tg - Tg.mean(), N - N.mean(), s=1, alpha=0.15)
ax[0].axhline(0, color='k', lw=0.5); ax[0].axvline(0, color='k', lw=0.5)
ax[0].set(xlabel='lag k (yr); k > 0: N follows Ts', ylabel='corr(N(t+k), global Ts(t))', title='Net TOA flux vs global Ts'); ax[0].legend(fontsize=7)
ax[1].axhline(0, color='k', lw=0.5); ax[1].axvline(0, color='k', lw=0.5)
ax[1].set(xlabel='lag k (yr); k > 0: uptake follows Ts', ylabel='corr(F_o(t+k), global Ts(t))', title='Ocean heat uptake vs global Ts')
ax[2].set(xlabel='global Ts anomaly (K)', ylabel='N anomaly (W m⁻²)', title='Annual N vs Ts, all runs')
fig.tight_layout(); fig.savefig('figures/plasim_mechanism/fluctuation_feedbacks.png', dpi=120)
