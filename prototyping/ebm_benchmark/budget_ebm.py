"""EBM: slow-mode decay rate from the global energy budget, vs the Jacobian.

Area weight of grid point j: control width in x times cos(pi x / 2) (x = 2 phi / pi).
Global heat content H = sum_j width_j * thermal_capacity_j * T_j  (thermal capacity includes cos).
Global net radiation N = sum_j width_j * source_j(T)  (source = (ASR - OLR) * cos); transport sums to zero.
Mode amplitude a(t): projection of anomalies on (i) the adjoint slow eigenvector, (ii) the data-fitted LIM
left eigenvector of the mode aligned with dT*/dmu. Budget rate s = (dN/da) / (dH/da), slopes by least squares.
"""
import numpy as np, xarray as xr
from gsebm.ebm_stability import stability, operator_at, tendency_jacobian
from gsebm.linear_modes import eof_basis, fit_lim, pattern_cosine
from gsebm.physics import outgoing_longwave_radiation, absorbed_shortwave_radiation
from gsebm.time import YEAR

def slope(y, a):
    a = a - a.mean(); return float((y - y.mean()) @ a / (a @ a))

guess = np.full(205, 300.0)
for tag, mu, reals in (('1', 1.0, (1, 2)), ('0p98', 0.98, (1, 2)), ('0p972', 0.972, (1, 2)), ('0p97', 0.97, (1, 7))):
    st = stability(mu, guess); guess = st.temperature
    op = operator_at(mu); x = op.x; cosx = op.geometric_weight
    area = op.control_widths * cosx; area_w = area / area.sum()
    width = op.control_widths; capacity = op.thermal_capacity
    J = tendency_jacobian(op, st.temperature)
    ev, L = np.linalg.eig(J.T); u1 = L[:, np.argmax(ev.real)].real
    v1 = st.modes[0].real
    # exact global-budget (Galerkin) estimate using the true slow pattern v1
    eps = 1e-3
    dN = (width @ op.source_term(st.temperature + eps * v1) - width @ op.source_term(st.temperature)) / eps
    s_galerkin = dN / (width @ (capacity * v1)) * YEAR
    print(f'\nmu={mu}: Jacobian s1={st.rates[0].real:.4f}/yr; global-budget estimate with true pattern {s_galerkin:.4f}/yr')
    fields = op.empirical_fields
    for real in reals:
        with xr.open_dataset(f'data/new_stochastic_warm_mu{tag}_{{{real}}}.nc', engine='scipy') as d:
            t = d.time.values; T = d.temperature.values
        dt = float(np.median(np.diff(t))); T = T[t >= 500 * YEAR]
        H = T @ (width * capacity); N = op.source_term(T) @ width
        albedo = op.local_albedo(T)
        asr = (absorbed_shortwave_radiation(fields.solar_irradiance, albedo, op.params) * cosx) @ width
        olr = (outgoing_longwave_radiation(T, op.params) * cosx) @ width
        Tg = T @ area_w
        b = eof_basis(T, area, 10)
        lim = fit_lim(b.pcs, 12, dt, b.patterns)
        cos = [pattern_cosine(p, st.forced_response, area) for p in lim.field_patterns]
        k = int(np.argmax(cos))
        # left eigenvector of A in PC space gives the eigenfunction phi = w^T pcs
        A_pc = np.linalg.lstsq(b.pcs[:-12], b.pcs[12:], rcond=None)[0].T
        lev, Lv = np.linalg.eig(A_pc.T)
        j = np.argmin(abs(np.log(lev.astype(complex)) / (12 * dt) * YEAR - lim.rates[k] * YEAR))
        phi_lim = (b.pcs @ Lv[:, j]).real
        out = []
        for name, amp in (('adjoint', (T - T.mean(0)) @ u1), ('LIM eigenfunction', phi_lim)):
            sN, sH = slope(N, amp), slope(H, amp)
            sT = slope(Tg, amp)
            out.append(f'{name}: s_budget={sN / sH * YEAR:.4f}/yr '
                       f'[per K of global T: dN/dTg={sN/sT:.3e}, dASR/dTg={slope(asr, amp)/sT:.3e}, dOLR/dTg={slope(olr, amp)/sT:.3e}, dH/dTg={sH/sT:.3e}]')
        print(f'  realization {real}: LIM rate {lim.rates[k].real*YEAR:.4f} (cos {cos[k]:.3f})')
        for o in out: print('    ' + o)
