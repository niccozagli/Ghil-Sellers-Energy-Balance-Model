"""Multi-lag tracking of the slow mode (and of the gyre cycle as a check), LIM and KDMD.

For tau = 1..40 yr: fit the operator at lag tau; follow a mode from lag to lag by pattern continuity (largest |cos|
with the mode's pattern at the previous lag; start at tau = 1 with the real mode best aligned with the forced response
r, or with the slowest complex mode for the cycle). Rates from the lambda(tau) sequence: Re s = slope of log|lambda|
vs tau (least squares, with intercept) over tau = 1-40 and over sub-ranges; Im s = slope of the unwrapped phase.
A clean Koopman eigenvalue gives a straight line through the origin.
State: as in slow_mode.py (variant 'full' with the 700-2000 m layer, 'nomid' without it).
"""
import sys, json, numpy as np
import importlib.util
variant = sys.argv[1]
MODES = sys.argv[2] if len(sys.argv) > 2 else 'slow,cycle'
src = open('/Users/niccolo/.claude/jobs/96e03936/tmp/slow_mode.py').read().split('results = {}')[0]
if variant == 'nomid':
    src = src.replace("names = list(BLOCKS)", "names = [b for b in BLOCKS if b != 'mid']")
exec(src)
from gsebm.linear_modes import eof_basis, fit_lim, fit_kdmd, pattern_cosine
TAUS = np.arange(1, 41)
PERIOD = {'1245': 46, '1242p5': 49, '1240': 51, '1237p5': 56, '1235': 61, '1235_new_IC': 61, '1233p75': 65, '1232p5': 67}
RUN = None
def lim_all(X, tau, b): res = fit_lim(b.pcs, tau, YEAR, b.patterns); return res.eigenvalues, res.field_patterns
def kdmd_all(X, tau):
    kd = fit_kdmd(X, ones, int(tau), YEAR, X.shape[0] - int(tau), seed=0, rel_threshold=1e-3)
    return np.exp(kd.rates * YEAR * tau), kd.field_patterns
def track(fitter, X, start):
    lam, pat = [], []
    prev = None
    for tau in TAUS:
        ev, P = fitter(tau)
        cos = np.array([pattern_cosine(p, r if prev is None and start == 'slow' else (prev if prev is not None else r), ones) for p in P])
        if prev is None:
            if start == 'slow':
                real = np.abs(np.imag(ev)) < 1e-9; i = int(np.argmax(np.where(real, cos, -1)))
            else:
                rates = np.log(ev.astype(complex)); cand = np.flatnonzero(rates.imag > 1e-9)
                i = cand[np.argmin(np.abs(rates.imag[cand] - 2 * np.pi / PERIOD[RUN]))]   # known cycle period (state A, step 3)
        else:
            i = int(np.argmax(cos))
        lam.append(complex(ev[i])); pat.append(P[i]); prev = P[i]
    lam = np.array(lam)
    logabs = np.log(np.abs(lam)); ph = np.unwrap(np.angle(lam))
    fit = lambda y, sl: np.polyfit(TAUS[sl], y[sl], 1)
    out = dict(re_all=fit(logabs, slice(0, 40))[0], icpt=fit(logabs, slice(0, 40))[1], re_1_10=fit(logabs, slice(0, 10))[0],
               re_10_20=fit(logabs, slice(9, 20))[0], re_20_40=fit(logabs, slice(19, 40))[0], im_all=fit(ph, slice(0, 40))[0],
               cos_r_1=pattern_cosine(pat[0], r, ones), cos_r_40=pattern_cosine(pat[-1], r, ones),
               cont_min=float(min(pattern_cosine(pat[k], pat[k + 1], ones) for k in range(len(pat) - 1))))
    return out, lam
res = {}
print(f'variant {variant}: rates in 1/yr; decay time = -1/Re s')
for l in LABELS:
    RUN = l
    d = D[l]; n = len(d['years']); s0, s1 = SEGMENT[l]; seg = slice(s0, s1 if s1 else n); X = build(d, seg)
    b = eof_basis(X, ones, 20)
    lim_f = lambda tau: lim_all(X, tau, b)
    kd_f = lambda tau: kdmd_all(X, tau)
    R = {}
    for name, f in (('LIM', lim_f), ('KDMD', kd_f)):
        for start in (('slow', 'cycle') if 'slow' in MODES else ('cycle',)):
            R[f'{name}_{start}'], lam = track(f, X, start)
    h = X.shape[0] // 2; bh = [eof_basis(X[:h], ones, 20), eof_basis(X[h:], ones, 20)]
    if 'slow' not in MODES:
        res[l] = R; c, cl = R['KDMD_cycle'], R['LIM_cycle']
        print(f"{l:12s} cycle: KDMD period {2*np.pi/c['im_all']:5.1f} yr, Re s {c['re_all']:+.4f}, cont {c['cont_min']:.2f} | LIM period {2*np.pi/cl['im_all']:5.1f} yr, Re s {cl['re_all']:+.4f}, cont {cl['cont_min']:.2f} | expected {PERIOD[l]} yr", flush=True)
        continue
    R['LIM_slow_halves'] = [track(lambda tau, Xh=Xh, bb=bb: lim_all(Xh, tau, bb), Xh, 'slow')[0]['re_all'] for Xh, bb in ((X[:h], bh[0]), (X[h:], bh[1]))]
    res[l] = R
    for k in ('LIM_slow', 'KDMD_slow'):
        o = R[k]
        print(f"{l:12s} {k:9s}: Re s {o['re_all']:+.4f} (τ 1–10 {o['re_1_10']:+.4f}, 10–20 {o['re_10_20']:+.4f}, 20–40 {o['re_20_40']:+.4f}) intercept {o['icpt']:+.3f} | Im s {o['im_all']:+.4f} | cos r {o['cos_r_1']:.2f}→{o['cos_r_40']:.2f}, min continuity {o['cont_min']:.2f}")
    c = R['KDMD_cycle']; cl = R['LIM_cycle']
    print(f"{'':12s} cycle    : KDMD period {2*np.pi/c['im_all']:5.1f} yr, Re s {c['re_all']:+.4f} | LIM period {2*np.pi/cl['im_all']:5.1f} yr, Re s {cl['re_all']:+.4f} | LIM slow halves {R['LIM_slow_halves'][0]:+.4f} / {R['LIM_slow_halves'][1]:+.4f}", flush=True)
json.dump(res, open(TMP + f'multilag_{variant}.json', 'w'), default=lambda x: float(x) if np.isscalar(x) else str(x))
