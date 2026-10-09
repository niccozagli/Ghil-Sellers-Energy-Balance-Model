"""Follow-up: is there a slow coupled edge-ocean mode? Cross-correlation of the annual Southern edge e(t) with
Southern upper-ocean theta 0-700 m (all longitudes, wet-area weighted) and with global LSG heat content, at lags up to
200 yr; corr(e(t+k), X(t)), k > 0: edge follows ocean. Annual values, analysis segments, no smoothing."""
import numpy as np
from gsebm.plasim_global import ice_edge_latitude
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'
LABELS = ['1245', '1242p5', '1240', '1237p5', '1235', '1233p75', '1232p5']
SEGMENT = {'1245': (0, None), '1242p5': (500, None), '1240': (0, None), '1237p5': (0, None), '1235': (0, None), '1233p75': (0, None), '1232p5': (0, 3000)}
ks = np.array([-100, -50, -20, -10, 0, 10, 20, 50, 100, 200])
def cc(y, x, k):
    a, b = (x[:len(x) - k], y[k:]) if k >= 0 else (x[-k:], y[:k])
    return np.corrcoef(a, b)[0, 1]
print('corr(edge(t+k), X(t)); edge in degrees from the equator (larger = more poleward)')
print(f"{'run':8s} X     " + ' '.join(f'{k:>6d}' for k in ks))
for lab in LABELS:
    d = dict(np.load(TMP + f'states/{lab}.npz')); n = len(d['years']); s0, s1 = SEGMENT[lab]; g = slice(s0, s1 if s1 else n)
    e = ice_edge_latitude(np.nan_to_num(d['ice'][g]), d['lat'], south=True)
    so = (d['lsg_lat'] < 0) & np.isfinite(d['theta_up_rows']).all(0) & (d['lsg_area'] > 0)
    th = d['theta_up_rows'][g][:, so] @ (d['lsg_area'][so] / d['lsg_area'][so].sum())
    H = d['ohc_global'][g].astype(float)
    for name, X in (('Sθ', th), ('OHC', H)):
        print(f"{lab:8s} {name:5s} " + ' '.join(f'{cc(e, X, k):+6.2f}' for k in ks))
