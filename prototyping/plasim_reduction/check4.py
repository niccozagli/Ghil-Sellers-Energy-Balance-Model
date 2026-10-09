"""Check 4: margin uncertainties (window halves, the 1235 pair) and the T21 row-crossing test.

For each part (full window, first half, second half): edge from the time-mean zonal ocean ice; cap |lat| >= 20 S.
Per step between runs a -> b: insolation term = sum dI (1 - mean alpha) area, albedo = -sum mean(I) d alpha area,
OLR = -sum dOLR area (zonal rows); margin = insolation term / edge change (PW per degree; > 0, smaller = more sensitive).
"""
import numpy as np, json
from gsebm.plasim_global import ice_edge_latitude, run_mu
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'; R = 6.371e6
SEQ = ['1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1233p75', '1232p5']
def z(lab, part):
    m = dict(np.load(TMP + f'maps/{lab}.npz')); g = lambda k: np.nanmean(m[f'{k}__{part}'], axis=1)
    lat = m['t21_lat']; w = m['t21_gaussian_weight']; ocean = m['lsm'] < 0.5
    sic = m[f'sea_ice_concentration__{part}']
    ice = np.where(ocean.sum(1) > 0, (np.nan_to_num(sic) * ocean).sum(1) / np.maximum(ocean.sum(1), 1), np.nan)
    I = g('rst') - g('rsut'); asr = g('rst'); olr = -g('rlut')
    return dict(lat=lat, area=w / w.sum() * 4 * np.pi * R**2, I=I, alpha=1 - asr / I, olr=olr,
                edge=float(ice_edge_latitude(ice[None], lat, south=True)[0]))
def step(a, b, part):
    A, B = z(a, part), z(b, part); cap = A['lat'] <= -20; ar = A['area'][cap]
    ab, Ib = 0.5 * (A['alpha'] + B['alpha'])[cap], 0.5 * (A['I'] + B['I'])[cap]
    ins = ((B['I'] - A['I'])[cap] * (1 - ab) * ar).sum() / 1e15
    alb = (-Ib * (B['alpha'] - A['alpha'])[cap] * ar).sum() / 1e15
    olr = -((B['olr'] - A['olr'])[cap] * ar).sum() / 1e15
    de = B['edge'] - A['edge']
    return dict(dedge=de, margin=ins / de, albedo=alb / de, olr=olr / de, edge_a=A['edge'], edge_b=B['edge'])
lat = dict(np.load(TMP + 'maps/1240.npz'))['t21_lat']; rowsS = np.sort(np.abs(lat[lat < 0]))
out = {}
for name, seq in (('main', SEQ), ('pair', [l if l != '1235' else '1235_new_IC' for l in SEQ])):
    for part in ('full', 'h1', 'h2'):
        segs = {'early 1265→1245': ('1265', '1245'), 'mid 1245→1235': ('1245', seq[6]), 'late 1235→1232.5': (seq[6], '1232p5')}
        res = {k: step(a, b, part) for k, (a, b) in segs.items()}
        steps = [step(a, b, part) for a, b in zip(seq[:-1], seq[1:])]
        out[f'{name}_{part}'] = dict(segments=res, steps=steps)
print('Segment margins (PW per degree) and per-degree albedo / OLR, for full window and halves:')
for key in ('main_full', 'main_h1', 'main_h2', 'pair_full', 'pair_h1', 'pair_h2'):
    s = out[key]['segments']
    print(f"  {key:10s} " + ' | '.join(f"{k.split()[0]}: m {v['margin']:.3f} alb {v['albedo']:.3f} olr {v['olr']:.3f}" for k, v in s.items()))
print('\nPer step (main sequence): edge a→b, crosses a T21 row centre?, margin full / h1 / h2')
for i, (a, b) in enumerate(zip(SEQ[:-1], SEQ[1:])):
    f, h1, h2 = (out[f'main_{p}']['steps'][i] for p in ('full', 'h1', 'h2'))
    lo, hi = sorted((f['edge_a'], f['edge_b'])); cross = [r for r in rowsS if lo < r < hi]
    print(f"  {a:>6s}→{b:<6s} {f['edge_a']:5.2f}→{f['edge_b']:5.2f} cross {str(cross) if cross else '-':<8} margin {f['margin']:.3f} / {h1['margin']:.3f} / {h2['margin']:.3f}  albedo {f['albedo']:.3f} olr {f['olr']:.3f}")
print('S T21 row centres:', np.round(rowsS[:10], 1))
json.dump(out, open(TMP + 'check4.json', 'w'), default=float)
