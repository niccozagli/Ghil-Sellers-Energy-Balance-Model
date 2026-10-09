"""Test 3 checks. (a) Distributed-lag regression x(t) = c + sum_{k=0..K} h_k F(t-k) + noise (least squares, K = 30),
which accounts for F's own persistence; report sum_{k<=10} h_k and sum_{k<=30} h_k with 5-95% from 100-yr block
bootstrap. (b) Does F lead or follow the slow ocean? corr(F(t+k), M1(t)) for k = -100..100 (k > 0: F follows M1)."""
import numpy as np
src = open('prototyping/plasim_reduction/test3.py').read().split("KS = [")[0]; exec(src)
K = 30; rng = np.random.default_rng(4)
def design(F):
    n = len(F); return np.column_stack([np.ones(n - K)] + [F[K - k:n - k] for k in range(K + 1)])
def fir(x, F):
    A = design(F); return np.linalg.lstsq(A, x[K:], rcond=None)[0][1:]
print('distributed-lag response, cumulative to 10 yr / 30 yr [5-95% block bootstrap]')
for lab in LABELS:
    S = seg(lab, *series(lab)); F = S['F']; n = len(F); out = []
    for x in ('e_pac', 'TsS', 'U1', 'M1'):
        h = fir(S[x], F); bs = []
        for _ in range(100):
            starts = rng.integers(0, n - 100, n // 100)
            hb = np.mean([0], 0)
            # bootstrap: fit on concatenated blocks (junction effects ignored, K << block length)
            idx = np.concatenate([np.arange(s, s + 100) for s in starts]); hb = fir(S[x][idx], F[idx]); bs.append([hb[:11].sum(), hb.sum()])
        lo, hi = np.percentile(bs, [5, 95], axis=0)
        out.append(f'{x}: {h[:11].sum():+.2f} [{lo[0]:+.2f},{hi[0]:+.2f}] / {h.sum():+.2f} [{lo[1]:+.2f},{hi[1]:+.2f}]')
    print(f'{lab:12s} ' + ' | '.join(out))
ks = [-100, -50, -20, -5, 0, 5, 20, 50, 100]
print('\ncorr(F(t+k), M1(t)), k =', ks)
for lab in LABELS:
    S = seg(lab, *series(lab)); F, M = S['F'], S['M1']; n = len(F)
    c = [np.corrcoef(F[k:], M[:n - k])[0, 1] if k >= 0 else np.corrcoef(F[:k], M[-k:])[0, 1] for k in ks]
    print(f'{lab:12s} ' + ' '.join(f'{v:+.2f}' for v in c))
