"""Test 3: natural mu-like forcing. F(t) = absorbed shortwave (rst) anomaly over the ice-free rows |lat| < 30,
expressed as a global-mean equivalent (W m^-2 of the globe): sum_{|lat|<30} w ASR / sum_all w (Gaussian weights).
Annual values, analysis segments, no filtering.

1. Is F forcing-like? Autocorrelation of F; corr(F(t+k), Ts_global(t)) at k = -5..5 (k > 0: F follows Ts).
2. Response function: R_x(k) = cov(x(t+k), F(t)) / var(F) for the edge (zonal, Pacific), TsS, U1, M1.
   Cumulative S_x(K) = sum_{k=0..K} R_x(k) = response to a sustained 1 W m^-2 of F after K yr (valid if F is
   white and exogenous). Null: same statistic with F circularly shifted by random offsets >= 500 yr (200 shifts),
   5-95% range.
3. Comparison: transient step response per W m^-2 of global absorbed forcing, dF = dS0 (1 - albedo_1240) / 4,
   block means of annual values of the steps from 1240 (as in part 4).
"""
import numpy as np
from gsebm.plasim_global import RUNS
src = open('prototyping/plasim_reduction/legs.py').read().split("print('NATURAL")[0]; exec(src)
w = dict(np.load(TMP + 'states/1240.npz'))['weight']; w = w / w.sum()

def series(lab):
    d = dict(np.load(TMP + f'full/{lab}.npz')); lat = d['lat']; trop = np.abs(lat) < 30
    yrs, X = load(lab)
    X['F'] = d['asr'][:, trop] @ w[trop]
    X['Tg'] = d['Ts'] @ w
    X['TsS'] = d['Ts'][:, lat < 0] @ w[lat < 0] / w[lat < 0].sum()
    X['alb'] = 1 - (d['asr'] @ w) / (d['ins'] @ w)
    return yrs, X

def R(x, F, k):
    f = F - F.mean(); x = x - x.mean(); n = len(f) - k
    return float(np.dot(x[k:k + n], f[:n]) / n / f.var())

KS = [0, 1, 2, 5, 10, 20, 50, 100, 200]
CUM = [10, 30, 100, 300, 1000]
OBS = ['e', 'e_pac', 'TsS', 'U1', 'M1']
rng = np.random.default_rng(3)
pool = {x: [] for x in OBS}
for lab in LABELS:
    S = seg(lab, *series(lab)); F = S['F']; n = len(F)
    acf = [np.corrcoef(F[:-k], F[k:])[0, 1] for k in (1, 2, 5, 10)]
    lead = [np.corrcoef(F[k:], S['Tg'][:n - k])[0, 1] if k >= 0 else np.corrcoef(F[:k], S['Tg'][-k:])[0, 1] for k in range(-5, 6)]
    print(f'\n=== {lab}: F sd {F.std():.3f} W m-2 | acf 1,2,5,10: ' + ' '.join(f'{a:+.2f}' for a in acf))
    print('   corr(F(t+k), Tg(t)), k=-5..5: ' + ' '.join(f'{c:+.2f}' for c in lead))
    for x in OBS:
        r = [R(S[x], F, k) for k in range(1001)]
        cum = np.cumsum(r)
        null = []
        for _ in range(200):
            Fs = np.roll(F, rng.integers(500, n - 500)); rs = [R(S[x], Fs, k) for k in range(1001)]; null.append(np.cumsum(rs)[CUM])
        lo, hi = np.percentile(null, [5, 95], axis=0)
        pool[x].append(cum[CUM])
        print(f'   {x:5s} R(k) ' + ' '.join(f'{r[k]:+.3f}' for k in KS) + ' | cum ' +
              ' '.join(f'{K}:{cum[K]:+.2f}[{l:+.2f},{h:+.2f}]' for K, l, h in zip(CUM, lo, hi)))

# transients per W m^-2 of global absorbed forcing
y0, X0 = series('1240'); S0 = seg('1240', y0, X0); a40 = float(np.mean(S0['alb'])); x40 = {x: np.mean(S0[x]) for x in OBS}
print(f'\nTRANSIENTS from 1240, per W m^-2 of global absorbed forcing (dS0 (1 - {a40:.3f}) / 4); block means of annual values')
for lab in ['1242p5', '1237p5', '1235_new_IC', '1233p75', '1232p5']:
    yrs, X = series(lab); t = yrs - 14999; dF = (run_mu(lab) - 1240.0) * (1 - a40) / 4
    out = []
    for x in OBS:
        b = [np.nanmean(X[x][(t >= K - 5) & (t < K + 5)]) - x40[x] if K < 1000 else np.nanmean(X[x][(t >= 800) & (t < 1200)]) - x40[x] for K in CUM]
        out.append(f'{x}: ' + ' '.join(f'{v / dF:+.2f}' for v in b))
    print(f'{lab:12s} ' + ' | '.join(out))
print('\nPOOLED natural cumulative response (mean over the 8 runs) at K =', CUM)
for x in OBS: print(f'   {x:5s} ' + ' '.join(f'{v:+.2f}' for v in np.mean(pool[x], 0)))
