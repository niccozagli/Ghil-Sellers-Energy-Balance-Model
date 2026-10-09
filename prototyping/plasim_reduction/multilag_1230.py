"""Multi-lag tracking of the slow mode in 1230 (cold branch; stationary window, full 4000 yr).
Same state blocks and scales as slow_mode.py, except the ice block = Southern rows with 15 < |lat| < 50 (covers both the
warm-branch edge and 1230's edge at 25 S). References: (a) warm-branch forced response (slope of window means on the
step-2 amplitude), (b) the jump direction = window mean of 1230 minus that of 1232.5. Selection at tau = 1: real mode with
the largest |cos| with the reference; then tracked by pattern continuity to tau = 40. Variants with and without the
700-2000 m layer."""
import sys, json, numpy as np
variant = sys.argv[1]
src = open('/Users/niccolo/.claude/jobs/96e03936/tmp/slow_mode.py').read().split('results = {}')[0]
src = src.replace("ice_rows = S & (np.nanstd(np.nan_to_num(D['1240']['ice']), 0) > 1e-3)", "ice_rows = S & (np.abs(d0['lat']) > 15) & (np.abs(d0['lat']) < 50)")
if variant == 'nomid': src = src.replace("names = list(BLOCKS)", "names = [b for b in BLOCKS if b != 'mid']")
src = src.replace("LABELS = ['1245',", "LABELS = ['1230', '1245',").replace("SEGMENT = {'1245'", "SEGMENT = {'1230': (0, None), '1245'")
src = src.replace("amp = json.load", "LABELS_W = [l for l in LABELS if l != '1230']\namp = json.load").replace(
    "a = np.array([dict(zip(amp['labels'], amp['a']))[l] for l in LABELS])", "a = np.array([dict(zip(amp['labels'], amp['a']))[l] for l in LABELS_W])").replace(
    "M = np.array([means(D[l]) for l in LABELS])", "M = np.array([means(D[l]) for l in LABELS_W])")
exec(src)
from gsebm.linear_modes import eof_basis, fit_lim, fit_kdmd, pattern_cosine
refs = {'warm r': r, 'jump': means(D['1230']) - means(D['1232p5'])}
print('reference cosine (warm r vs jump):', round(pattern_cosine(refs['warm r'], refs['jump'], ones), 2))
TAUS = np.arange(1, 41)
d = D['1230']; X = build(d)
def fitters(Xs):
    b = eof_basis(Xs, ones, 20)
    def lim(tau): res = fit_lim(b.pcs, tau, YEAR, b.patterns); return res.eigenvalues, res.field_patterns
    def kd(tau):
        k = fit_kdmd(Xs, ones, int(tau), YEAR, Xs.shape[0] - int(tau), seed=0, rel_threshold=1e-3); return np.exp(k.rates * YEAR * tau), k.field_patterns
    return {'LIM': lim, 'KDMD': kd}
def track(f, ref):
    lam, pat, prev = [], [], None
    for tau in TAUS:
        ev, P = f(tau)
        if prev is None:
            cos = np.array([pattern_cosine(p, ref, ones) for p in P]); real = np.abs(np.imag(ev)) < 1e-9; i = int(np.argmax(np.where(real, cos, -1)))
        else:
            i = int(np.argmax([pattern_cosine(p, prev, ones) for p in P]))
        lam.append(complex(ev[i])); pat.append(P[i]); prev = P[i]
    la = np.log(np.abs(np.array(lam))); fit = lambda sl: np.polyfit(TAUS[sl], la[sl], 1)
    return dict(re=fit(slice(0, 40))[0], icpt=fit(slice(0, 40))[1], r1_10=fit(slice(0, 10))[0], r10_20=fit(slice(9, 20))[0], r20_40=fit(slice(19, 40))[0],
                im=float(np.polyfit(TAUS, np.unwrap(np.angle(lam)), 1)[0]), cos1=pattern_cosine(pat[0], ref, ones), cos40=pattern_cosine(pat[-1], ref, ones),
                cont=float(min(pattern_cosine(pat[k], pat[k + 1], ones) for k in range(39))), pattern=pat[0])
h = X.shape[0] // 2
out = {}
for rname, ref in refs.items():
    for meth in ('LIM', 'KDMD'):
        o = track(fitters(X)[meth], ref)
        line = f"1230 {variant:5s} ref {rname:6s} {meth:4s}: Re s {o['re']:+.4f} (τ1–10 {o['r1_10']:+.4f}, 10–20 {o['r10_20']:+.4f}, 20–40 {o['r20_40']:+.4f}) icpt {o['icpt']:+.3f} Im {o['im']:+.4f} | cos ref {o['cos1']:.2f}→{o['cos40']:.2f} cont {o['cont']:.2f}"
        if meth == 'LIM':
            hv = [track(fitters(Xh)['LIM'], ref)['re'] for Xh in (X[:h], X[h:])]; line += f" | halves {hv[0]:+.4f}/{hv[1]:+.4f}"
        print(line, flush=True)
        out[f'{rname}_{meth}'] = {k: v for k, v in o.items() if k != 'pattern'}
json.dump(out, open(TMP + f'multilag_1230_{variant}.json', 'w'), default=float)
