"""Check 2: year-to-year edge test (annual values, run segments).

Annual Southern edge e(t) (zonal ocean ice, 0.5 threshold, |lat|). Release centroid: phi_R(t) = sum |lat| R+ area / sum R+ area
over S LSG rows with 20 < |lat| < 55, R = -(uptake). Cap = |lat| >= 20 S. Cap terms (PW, anomalies):
  albedo = sum_rows -mean(I)(alpha - mean alpha) * area;  olr = -(OLR anomaly) * area;  ocean = cap release anomaly;
  residual = -(albedo + olr + ocean) = atmospheric import - storage (atmosphere + mixed layer + ice).
Regression of term(t+k) on e(t), k = 0..10; reported per degree of EQUATORWARD edge shift (= -slope).
"""
import numpy as np, json
from gsebm.plasim_global import ice_edge_latitude, EARTH_RADIUS_M
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1235_new_IC', '1233p75', '1232p5']
SEGMENT = {'1245': (0, None), '1242p5': (500, None), '1240': (0, None), '1237p5': (0, None), '1235': (0, None),
           '1235_new_IC': (2499, None), '1233p75': (0, None), '1232p5': (0, 3000)}
A = 4 * np.pi * EARTH_RADIUS_M**2; PHIC = 20.0
def slope_lag(y, x, k):
    a, b = x[:len(x) - k], y[k:]; a = a - a.mean(); return float((b - b.mean()) @ a / (a @ a))
out = {}
print(f"{'run':12s} {'edge':>5s} {'sd':>4s} | phi_R slope | per degree equatorward at lag 0 (PW): albedo  -OLR  ocean  resid | ocean at lags 1,3,5,10 | resid lags 1,5")
for lab in LABELS:
    d = dict(np.load(TMP + f'states/{lab}.npz')); n = len(d['years']); s0, s1 = SEGMENT[lab]; g = slice(s0, s1 if s1 else n)
    lat, w = d['lat'], d['weight']; ice = np.nan_to_num(d['ice'][g])
    e = ice_edge_latitude(ice, lat, south=True)
    ll, area = d['lsg_lat'], d['lsg_area']; R = -d['uptake'][g]
    band = (ll < -20) & (ll > -55) & (area > 0)
    Rp = np.clip(R[:, band], 0, None) * area[band]
    phiR = (Rp * np.abs(ll[band])).sum(1) / Rp.sum(1)
    capT = (lat <= -PHIC); aT = A * w / w.sum()
    asr, ins, olr = d['asr'][g], d['ins'][g], d['olr'][g]; alb = 1 - asr / ins
    albedo = (-(ins.mean(0) * (alb - alb.mean(0)))[:, capT] * aT[capT]).sum(1) / 1e15
    olrt = (-(olr - olr.mean(0))[:, capT] * aT[capT]).sum(1) / 1e15
    capO = (ll <= -PHIC) & (area > 0)
    ocean = (R[:, capO] * area[capO]).sum(1) / 1e15; ocean = ocean - ocean.mean()
    resid = -(albedo + olrt + ocean)
    terms = dict(albedo=albedo, olr=olrt, ocean=ocean, resid=resid)
    lag = {k: {t: -slope_lag(v, e, k) for t, v in terms.items()} for k in range(0, 11)}
    h = len(e) // 2
    halves = [{t: -slope_lag(v[sl], e[sl], 0) for t, v in terms.items()} for sl in (slice(0, h), slice(h, None))]
    sR = slope_lag(phiR, e, 0)
    out[lab] = dict(edge=float(e.mean()), edge_sd=float(e.std()), phiR_slope=sR, phiR_r=float(np.corrcoef(phiR, e)[0, 1]), lag=lag, halves=halves)
    L0 = lag[0]
    print(f"{lab:12s} {e.mean():5.1f} {e.std():4.2f} | {sR:5.2f} (r {np.corrcoef(phiR, e)[0,1]:.2f}) | {L0['albedo']:+.3f} {L0['olr']:+.3f} {L0['ocean']:+.3f} {L0['resid']:+.3f}"
          f" | {lag[1]['ocean']:+.3f} {lag[3]['ocean']:+.3f} {lag[5]['ocean']:+.3f} {lag[10]['ocean']:+.3f} | {lag[1]['resid']:+.3f} {lag[5]['resid']:+.3f}"
          f" | halves albedo {halves[0]['albedo']:+.3f}/{halves[1]['albedo']:+.3f} ocean {halves[0]['ocean']:+.3f}/{halves[1]['ocean']:+.3f}")
json.dump(out, open(TMP + 'check2.json', 'w'))
