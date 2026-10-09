"""Follow-up to check 5: surface vs atmosphere albedo parts per degree of edge shift, South vs North, 1245 -> 1232.5,
and the ocean-only vs land part of the Northern surface term (same single-layer model as check 3)."""
import numpy as np, importlib.util
spec = importlib.util.spec_from_file_location('c3', 'prototyping/plasim_reduction/check3_lib.py')
c3 = importlib.util.module_from_spec(spec); spec.loader.exec_module(c3)
from gsebm.plasim_global import ice_edge_latitude
A, B = c3.load('1245'), c3.load('1232p5')
P = c3.P
surf = 0.5 * ((P(A['rho'], A['tau'], B['a']) - P(A['rho'], A['tau'], A['a'])) + (P(B['rho'], B['tau'], B['a']) - P(B['rho'], B['tau'], A['a'])))
dP = P(B['rho'], B['tau'], B['a']) - P(A['rho'], A['tau'], A['a']); Ib = 0.5 * (A['I'] + B['I'])
for hemi, sgn in (('S', -1), ('N', 1)):
    cap = ((np.sign(A['lat']) == sgn) & (np.abs(A['lat']) >= 20))[:, None] * np.ones_like(A['I'], dtype=bool)
    ocean = A['ocean']
    zi = lambda e: np.where(e['ocean'].sum(1) > 0, (np.nan_to_num(e['sic']) * e['ocean']).sum(1) / np.maximum(e['ocean'].sum(1), 1), np.nan)
    de = float(ice_edge_latitude(zi(B)[None], A['lat'], south=sgn < 0)[0] - ice_edge_latitude(zi(A)[None], A['lat'], south=sgn < 0)[0])
    f = lambda x, m: float((-Ib * x * A['area'])[m].sum() / 1e15 / de)
    print(f"{hemi}: Δedge {de:+.2f}° | per degree: total {f(dP, cap):.3f}, surface {f(surf, cap):.3f} (ocean cells {f(surf, cap & ocean):.3f}, land {f(surf, cap & ~ocean):.3f}), atmosphere {f(dP - surf, cap):.3f}")
