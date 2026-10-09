"""Check 3: surface (ice) vs atmosphere (cloud) part of the albedo term per degree of edge shift; OLR attribution.

Single-layer shortwave model (Donohoe & Battisti 2011) per T21 cell on window-mean maps (time-mean fluxes; an
approximation). I = rst - rsut, reflected fraction A = (I - rst)/I, surface downwelling D = rss/(1-as)/I, alpha = as:
  rho = (A - alpha D^2)/(1 - alpha^2 D^2), tau = D (1 - rho alpha), P = rho + tau^2 alpha/(1 - rho alpha).
Surface part of dP between two runs = mean over both atmospheres of P(alpha2) - P(alpha1); atmosphere part = rest.
Albedo term = -mean(I) dP, integrated over the cap |lat| >= 20 S (cell areas) in PW, relative to 1245.
OLR attribution: cap-mean Ts change per degree of edge, and cap OLR change per K of cap-mean Ts, early vs late.
"""
import numpy as np, json
from gsebm.plasim_global import ice_edge_latitude, run_mu
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
R = 6.371e6; REF = '1245'
RUNS_ = ['1312', '1288', '1265', '1250', '1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5', '1230', '1228p5']
def load(lab, part='full'):
    m = dict(np.load(TMP + f'maps/{lab}.npz')); g = lambda k: m[f'{k}__{part}']
    I = g('rst') - g('rsut'); rst = g('rst'); a_s = np.clip(g('as'), 0, 0.95)
    A = np.clip((I - rst) / I, 0, 1); D = np.clip(g('rss') / (1 - a_s) / I, 0, 1)
    rho = (A - a_s * D**2) / (1 - a_s**2 * D**2); tau = D * (1 - rho * a_s)
    lat = m['t21_lat']; w = m['t21_gaussian_weight']; nlon = m['t21_lon'].size
    area = (w / w.sum() * 4 * np.pi * R**2 / nlon)[:, None] * np.ones((1, nlon))
    ocean = m['lsm'] < 0.5; sic = g('sea_ice_concentration')
    zonal_ice = np.where(ocean.sum(1) > 0, (np.nan_to_num(sic) * ocean).sum(1) / np.maximum(ocean.sum(1), 1), np.nan)
    edge = float(ice_edge_latitude(zonal_ice[None], lat, south=True)[0])
    return dict(I=I, A=A, a=a_s, rho=rho, tau=tau, lat=lat, area=area, olr=-g('rlut'), ts=g('surface_temperature'), edge=edge, ocean=ocean, sic=sic)
P = lambda rho, tau, a: rho + tau**2 * a / (1 - rho * a)
ref = load(REF); cap = (ref['lat'] <= -20)[:, None] * np.ones_like(ref['I'], dtype=bool)
rows = []
print(f"{'run':12s} {'edge':>5s} | albedo term vs 1245 (PW): total  surface  atmosphere | closure check (P model vs A) | cap Ts (K)  cap OLR (PW)")
for lab in RUNS_:
    e = load(lab)
    Pref, Pe = P(ref['rho'], ref['tau'], ref['a']), P(e['rho'], e['tau'], e['a'])
    surf = 0.5 * ((P(ref['rho'], ref['tau'], e['a']) - Pref) + (Pe - P(e['rho'], e['tau'], ref['a'])))
    dP = Pe - Pref; Ib = 0.5 * (ref['I'] + e['I'])
    tot = float((-Ib * dP * ref['area'])[cap].sum() / 1e15); su = float((-Ib * surf * ref['area'])[cap].sum() / 1e15)
    clos = float(np.nanmax(np.abs(Pe - e['A'])[cap]))
    capTs = float((e['ts'] * ref['area'])[cap].sum() / ref['area'][cap].sum()); capOLR = float((e['olr'] * ref['area'])[cap].sum() / 1e15)
    rows.append(dict(run=lab, mu=run_mu(lab), edge=e['edge'], total=tot, surface=su, atmosphere=tot - su, capTs=capTs, capOLR=capOLR))
    print(f"{lab:12s} {e['edge']:5.1f} | {tot:+7.2f} {su:+7.2f} {tot - su:+7.2f} | {clos:.1e} | {capTs:7.2f} {capOLR:7.2f}")
by = {r['run']: r for r in rows}
def seg(a, b):
    ra, rb = by[a], by[b]; de = rb['edge'] - ra['edge']
    return {k: (rb[k] - ra[k]) / de for k in ('total', 'surface', 'atmosphere', 'capTs', 'capOLR')} | {'olr_per_K': (rb['capOLR'] - ra['capOLR']) / (rb['capTs'] - ra['capTs'])}
print('\nPer degree of edge shift (PW per degree; positive values = increase when the edge moves poleward):')
for a, b in (('1265', '1245'), ('1245', '1235'), ('1235', '1232p5'), ('1245', '1232p5'), ('1232p5', '1230')):
    s = seg(a, b)
    print(f"  {a:>6s}→{b:<6s}: albedo {s['total']:+.3f} = surface {s['surface']:+.3f} + atmosphere {s['atmosphere']:+.3f} (surface share {s['surface']/s['total']*100:.0f}%) | cap Ts {s['capTs']:+.3f} K/deg, cap OLR {s['capOLR']:+.3f} PW/deg, OLR per K of cap Ts {s['olr_per_K']:.3f} PW/K")
json.dump(rows, open(TMP + 'check3.json', 'w'))
