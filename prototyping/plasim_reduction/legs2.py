"""Test 2 continued: does the upper ocean follow the mid-depth ocean in natural variability as it does in the forced
slow stage? Natural: lag-0 slope of U1, U2 on M1 (K per K) per run segment, and lagged correlations of the Pacific
edge with U1 and M1 (corr(e(t+k), X(t)), k > 0 edge follows). Forced: (U_eq - U_100)/(M1_eq - M1_100)."""
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('legs', 'prototyping/plasim_reduction/legs.py')
src = open('prototyping/plasim_reduction/legs.py').read().split("print('NATURAL")[0]; exec(src)
def cc(y, x, k):
    a, b = (x[:len(x) - k], y[k:]) if k >= 0 else (x[-k:], y[:k]); return np.corrcoef(a, b)[0, 1]
ks = [-50, -20, -10, 0, 10, 20, 50, 100, 200]
print('NATURAL slope of U on M1 (K/K) [halves]; then corr(e_pac(t+k), X(t)) for k =', ks)
for lab in LABELS:
    S = seg(lab, *load(lab)); h = len(S['e']) // 2
    s = [f"{u}/M1 {slope(S[u], S['M1']):+.2f} [{slope(S[u][:h], S['M1'][:h]):+.2f} {slope(S[u][h:], S['M1'][h:]):+.2f}] r={np.corrcoef(S[u], S['M1'])[0,1]:+.2f}" for u in ('U1', 'U2')]
    print(f'{lab:12s} ' + ' | '.join(s))
    for X in ('U1', 'M1'):
        print(f'{"":12s} e_pac~{X}: ' + ' '.join(f'{cc(S["e_pac"], S[X], k):+.2f}' for k in ks))
print('\nFORCED slow stage (U_eq - U_100)/(M1_eq - M1_100)')
for lab in ['1242p5', '1237p5', '1235_new_IC', '1233p75', '1232p5']:
    yrs, X = load(lab); t = yrs - 14999; S = seg(lab, yrs, X); k = (t >= 80) & (t < 120)
    d = {b: np.nanmean(S[b]) - np.nanmean(X[b][k]) for b in BOX}
    print(f'{lab:12s} U1/M1 {d["U1"]/d["M1"]:+.2f}  U2/M1 {d["U2"]/d["M1"]:+.2f}')
