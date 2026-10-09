"""Check 5: South vs North, and sectors. Window-mean maps; segment 1245 -> 1232.5 (and 1265 -> 1245).
Per degree of edge shift: albedo term (-mean I d alpha, planetary), ice area change (sic * cell area over ocean),
ocean area per degree of latitude at the mean edge (ocean fraction of the edge row * 2 pi R^2 cos(phi) pi/180),
and the Southern albedo term by sector (Atlantic 65W-20E, Indian 20-115E, Pacific 115E-65W)."""
import numpy as np, json
from gsebm.plasim_global import ice_edge_latitude
from gsebm.plasim_transitions import SECTORS, sector_mask
TMP = '/Users/niccolo/.claude/jobs/96e03936/tmp/'; R = 6.371e6
def load(lab):
    m = dict(np.load(TMP + f'maps/{lab}.npz')); g = lambda k: m[f'{k}__full']
    lat, lon, w = m['t21_lat'], m['t21_lon'], m['t21_gaussian_weight']; ocean = m['lsm'] < 0.5
    area = (w / w.sum() * 4 * np.pi * R**2 / lon.size)[:, None] * np.ones((1, lon.size))
    I = g('rst') - g('rsut'); sic = np.nan_to_num(g('sea_ice_concentration'))
    ice = np.where(ocean.sum(1) > 0, (sic * ocean).sum(1) / np.maximum(ocean.sum(1), 1), np.nan)
    return dict(lat=lat, lon=lon, ocean=ocean, area=area, I=I, alpha=1 - g('rst') / I, sic=sic,
                edgeS=float(ice_edge_latitude(ice[None], lat, south=True)[0]), edgeN=float(ice_edge_latitude(ice[None], lat, south=False)[0]))
out = {}
for a, b in (('1265', '1245'), ('1245', '1232p5')):
    A, B = load(a), load(b); Ib = 0.5 * (A['I'] + B['I']); alb = -Ib * (B['alpha'] - A['alpha']) * A['area'] / 1e15
    iceA = (B['sic'] - A['sic']) * A['ocean'] * A['area'] / 1e12
    res = {}
    for hemi, sgn, ek in (('S', -1, 'edgeS'), ('N', 1, 'edgeN')):
        cap = (np.sign(A['lat']) == sgn) & (np.abs(A['lat']) >= 20)
        de = B[ek] - A[ek]; emid = 0.5 * (A[ek] + B[ek])
        rows = np.flatnonzero(np.sign(A['lat']) == sgn); j = rows[np.argmin(np.abs(np.abs(A['lat'][rows]) - emid))]
        ofrac = A['ocean'][j].mean(); oarea = ofrac * 2 * np.pi * R**2 * np.cos(np.radians(emid)) * np.pi / 180 / 1e12
        r = dict(dedge=de, albedo=float(alb[cap].sum() / de), ice=float(iceA[cap].sum() / de), ocean_frac=float(ofrac), ocean_area_per_deg=float(oarea))
        r['albedo_per_ice'] = r['albedo'] / r['ice']
        if hemi == 'S':
            r['sectors'] = {s: float(alb[cap[:, None] & sector_mask(A['lon'], s)[None, :]].sum() / de) for s in SECTORS}
            r['sector_ocean_frac'] = {s: float((A['ocean'][j] & sector_mask(A['lon'], s)).sum() / A['lon'].size) for s in SECTORS}
        res[hemi] = r
    out[f'{a}->{b}'] = res
    S, N = res['S'], res['N']
    print(f'{a}→{b}: per degree of edge shift')
    for h, r in res.items():
        print(f"  {h}: Δedge {r['dedge']:+.2f}° | albedo {r['albedo']:.3f} PW | ice area {r['ice']:.2f}e12 m² | ocean fraction of edge row {r['ocean_frac']:.2f}, ocean area per degree {r['ocean_area_per_deg']:.2f}e12 m² | albedo per ice area {r['albedo_per_ice']*1e3:.1f} W m⁻² equiv (PW/1e12m²×1e3)")
    print(f"  S/N ratios: albedo {S['albedo']/N['albedo']:.2f}, ice area {S['ice']/N['ice']:.2f}, ocean area per degree {S['ocean_area_per_deg']/N['ocean_area_per_deg']:.2f}")
    tot = sum(S['sectors'].values())
    print('  S sectors: ' + ', '.join(f"{s} {v:.3f} PW ({v/tot*100:.0f}%; ocean share of edge row {S['sector_ocean_frac'][s]:.2f})" for s, v in S['sectors'].items()))
json.dump(out, open(TMP + 'check5.json', 'w'))
