# PlaSim–LSG cold branch: mechanism check and Koopman path to snowball — implementation plan

Audience: Codex (implementation agent). Follow `AGENTS.md` throughout: Python 3.12 + uv, library code in
`src/gsebm/`, Typer CLIs in `scripts/` run with `PYTHONPATH=src`, marimo apps in `analysis/`, outputs in
`data/` and `figures/` (git-ignored), xarray `engine="scipy"`, never overwrite existing outputs, small
well-tested changes, run `PYTHONPATH=src uv run python -m unittest discover -s tests` after every task.

## 0. Scientific context (read first, do not re-derive)

Coupled PlaSim (T21L10) + LSG ocean, solar constant μ scanned on the cold branch. Findings so far:

- At μ = 1240 there is a coherent ~51-yr Southern Hemisphere ice–ocean oscillation (energy closes:
  dOHC/dt vs TOA N, r = 0.92, amplitude ~0.07 W m⁻²). NH zonal-T ACF is monotone (78% τ≈1.2 yr, 22% τ≈48 yr).
- Along the cold branch (μ 1245 → 1240 → 1235 → 1230 → 1225) the SH cycle period lengthens
  (43 → 51 → 58–60 → ~84 yr) and then disappears; at 1225 a lag-consistent real mode of 63–69 yr appears
  and the run collapses. Snowball threshold lies between μ = 1216 and 1225.
- Working hypothesis: a **delayed ice–albedo–ocean oscillator** (Bhattacharya, Ghil & Vulis 1982, P ≈ 2τ).
  Ocean heat convergence (subtropical 200–750 m water → wind-driven cells → convection at the ice margin)
  anchors the SH ice edge (Rose & Marshall 2009; Ferreira et al. 2011; Rose 2015). The delay grows as μ
  decreases (τ ≈ 21/25/30/40 yr), the oscillation loses coherence, and the final runaway is a loss of the
  ocean anchor (saddle-node).
- Expected Ruelle–Pollicott picture (Tantet et al. 2020, Part II): noisy limit cycle → parabola of
  resonances λ_{l,n}; noisy focus → triangular array; approach to saddle-node → one real resonance
  moving toward 0. Real parts always negative; the bifurcation is seen as a change of spectral pattern
  plus Re λ → 0 of the leading real mode.

Known model caveats to test, not to fix: 9 m sea-ice thickness cap with non-local heat compensation
(icemod `getiflx`, `xcfluxra`), no sea-ice dynamics, no snow on sea-ice in albedo, LSG diffusive numerics.

Two goals:
1. **Mechanism**: confirm the oscillation and the collapse are physical and follow the delayed
   ice–albedo–ocean chain, not a numerical artefact.
2. **Koopman**: show the path to the bifurcation as a change of reduced RP resonances vs μ, using zonal
   surface temperature augmented with a minimal set of physically motivated slow variables.

No changes to existing simulations. New μ values on the cold branch may be added later (Section 6).

Framing question (added): can the transition be fingerprinted from the **NH alone**? Working answer: the
NH should carry the *global* saddle-node mode (global-T-like, peaking at the ice line), but not the SH
oscillator that holds the memory. NH-only operators are expected to be non-Markovian (lag-inconsistent)
unless memory is added via time delays. Sections 3.0, 2.8 and 5.8 test this.

## 1. Inputs

Per experiment `CONTROL_360ppm_T21L10_10000Y_MU_<μ>`, raw output at
`<experiments_root>/<EXP>/output/<state>/` and `<experiments_root>/<EXP>/diag/<state>/` (state `spinup`;
default root `/Volumes/Nicco/Plasim/experiments`, must be a CLI option). Files: `*_PLA.<y0>-<y1>.nc`,
`*_LSG.<y0>-<y1>.nc`, `*_DIAG.<y0>-<y1>.txt`, and ICE output if present. Already-extracted files in
`data/Plasim/<EXP>/<EXP>_spinup_{diagnostics,zonal_temperatures,ocean_diagnostics}.nc`.

Priority μ: 1245, 1240, 1235, 1230, 1225, 1216 (plus warm reference 1265). Transients to discard:
reuse the `transients` dict from `analysis/checking_Plasim.py`; 1230 and 1225 are non-stationary — treat
them with windowed analysis (Section 5.5) and flag them in every output.

**Task 0 (inventory, no code changes):** write `scripts/inventory_plasim_outputs.py` that lists, per
experiment, available years, the variable names/codes in one PLA, one LSG and one DIAG/ICE file, and
which of the fields in Section 2 are present. Save `data/Plasim/inventory.json`. Stop and report if a
required field is missing — do not guess codes.

## 2. New extraction (library functions + tests + CLI)

Extend `src/gsebm/plasim_diagnostics.py` (or a new `src/gsebm/plasim_ocean_dynamics.py` if the file
becomes unwieldy). Each function gets a unit test on a small synthetic file in `tests/`.

2.1 **DIAG parser extension.** Parse all LSG DIAG lines, not only ATL max: `ATL min (AABW)`, `PAC max`,
`PAC min`, barotropic streamfunction max/min, up/downwelling transports, `Conv.adjustm. events`.
Output annual time series. Test with a hand-written DIAG snippet copied from a real file.

2.2 **Sea-ice cap diagnostics.** From PLA/ICE output: fraction of ocean cells with `sit` ≥ 8.9 m per
hemisphere; area-weighted non-local correction flux (`xcfluxra` or the variable that holds it, identify
in Task 0) in W m⁻² global mean. Annual series.

2.3 **Ice-line series.** Add the sic-based edge: latitude where the zonal-mean annual `sic` (code 210)
crosses 0.5, per hemisphere, linear interpolation between Gaussian latitudes (smooth, albedo-relevant).
Keep the existing persistent 5 cm edge (`*_persistent_sea_ice_edge_latitude`). Also zonal-mean `sit`.

2.4 **Surface energy fluxes and implied OHT.** From PLA: net SW and LW at surface (176/177),
sensible and latent heat fluxes, and sea-ice/snow-melt terms if present; zonal means on the 32
Gaussian latitudes. Implied ocean + ice heat transport
`OHT(φ) = -2π a² ∫_{-π/2}^{φ} (F_sfc_net − dH_ice/dt) cos φ' dφ'`, with the global mean imbalance removed
(document the choice); OHT convergence `−(1/(2π a² cos φ)) dOHT/dφ` in W m⁻².

2.5 **LSG zonal-mean ocean fields.** Annual zonal means of T, S, potential density (use LSG's own
equation of state from `lsgmod.f90`, `dens`; test against a few hand-computed values), meridional
velocity; meridional overturning streamfunction Ψ(φ, z) = −∫∫ v a cos φ dλ dz′ (global and, if basin
masks are available, Atlantic/Pacific); direct OHT = ∫∫ ρ c_p v θ dλ dz on the LSG grid. Check direct
OHT against implied OHT (2.4) in time mean: acceptance < 20% discrepancy away from the poles, otherwise
report.

2.6 **Convection and mixed layer.** Zonal-mean convective adjustment count by depth (if in LSG output),
convection depth (deepest level with nonzero events, Börner-style), and a density-based mixed-layer depth
(Δσ = 0.03 kg m⁻³ from the top level) in the 30–70°S and 30–70°N bands.

2.7 **Buoyancy flux split south of 30°S.** B_H = g α F_heat /(ρ c_p), B_FW = g β S (E − P − runoff +
ice melt/freeze), area-weighted over ocean south of 30°S and in the 5° band poleward of the ice edge.

2.8 **NH land snow.** From PLA: land snow cover fraction and snow depth (identify codes in Task 0),
land surface albedo, zonal means over land only, and area of snow-covered land per hemisphere. Needed
to test whether land snow albedo (PlaSim: 0.4–0.8, temperature-dependent) is an NH-specific feedback
near the threshold.

2.9 **Atmospheric heat transport.** Implied AHT(φ) from TOA net minus surface net flux (same integral and
global-mean removal as 2.4), and cross-equatorial AHT and OHT as annual series.

CLI: `scripts/extract_plasim_cold_branch_diagnostics.py --experiment-dir ... --state spinup --workers 4
--output-dir data/Plasim/<EXP>`, writes `<EXP>_spinup_cold_branch_diagnostics.nc` (1D series) and
`<EXP>_spinup_cold_branch_zonal.nc` (φ×t and φ×z×t fields). Refuse to overwrite.

## 3. Goal 1a — artefact checks (analysis module `src/gsebm/plasim_cycle.py`)

3.0 **Hemispheric order and coupling (do first, cheap).**
- *Who goes first at collapse:* for 1225 (and 1230 if it collapses), NH and SH ice-edge latitudes
  (sic 0.5 and persistent 5 cm), hemispheric ice area, hemispheric-mean Ts, and NH land snow area vs
  time. Detect the onset of runaway per series (e.g. first year the 50-yr running slope exceeds 3× its
  pre-collapse standard deviation; also report a changepoint estimate). Report onset years, their
  difference and a bootstrap interval. Also note whether the NH edge moves before or after the SH edge
  passes its minimum-OHT-convergence latitude (4.2).
- *Interhemispheric coupling in stationary runs (1245, 1240, 1235):* lagged cross-correlation (±150 yr)
  of NH vs SH ice-edge latitude, NH vs SH mean Ts, and each vs cross-equatorial AHT/OHT (2.9). Also
  NH-edge and NH-T phase composites on the SH cycle phase (3.1). Report sign, lag and coherence at the
  cycle frequency. Hypothesis to test: the NH sees the SH cycle weakly, damped and possibly with opposite
  sign because atmospheric heat transport compensates.
- *NH land snow:* composite land snow area by SH cycle phase, and its time-mean vs μ.

3.1 **Cycle phase.** Define the SH cycle phase from the Hilbert transform of a band-passed (20–150 yr,
zero-phase Butterworth) index: SH ice-edge latitude (sic 0.5). Validate with a second index (SH-mean
zonal T); phases must agree (circular correlation > 0.8). Provide `cycle_phase(series, band, dt)` and
`phase_bins(phase, n=8)`. Test on a synthetic noisy sinusoid with known phase.

3.2 **Energy closure by phase.** Composite, per phase bin: TOA N, dOHC/dt (full depth), d(ice latent
heat)/dt, residual. Pass criterion: residual not phase-locked (its composite amplitude < 20% of the N
composite amplitude).

3.3 **Cap artefact.** Composite cap fraction and cap correction flux by phase. Pass: correction flux
composite amplitude < 10% of the cycle energy amplitude (~0.07 W m⁻² at 1240). If failed, report — this
would make the cycle suspect.

3.4 **Window robustness.** Split each stationary run into 3 equal windows; period, ACF decay and cycle
amplitude must agree within bootstrap uncertainty.

## 4. Goal 1b — delivery chain and delayed-oscillator quantification

4.1 **Phase composites of the chain** (each as φ×phase or φ×z×phase panels, anomalies w.r.t. the
time mean): subtropical 200–750 m T and S (20–45°S); SH overturning cell strength and latitude of its
extremum; convection depth / MLD at 40–65°S; subsurface T at 30–45°S, 200–750 m; sic, sit and ice-edge
latitude; surface net flux, albedo and absorbed SW at the edge; TOA N. Produce the lag (in phase units
and years) of each link's maximum relative to the ice edge. Expected order (to be confirmed or refuted):
subsurface warming → convection/MLD → ice retreat → ASR increase → surface warming → ...

4.2 **OHT anchor vs μ.** For each μ, time-mean OHT convergence profile and ice-edge latitude. Report
(a) offset between edge and convergence maximum, (b) convergence value at the edge. Hypothesis: edge sits
just poleward of the maximum, and edge convergence falls toward ~25 W m⁻² before collapse.

4.3 **Delay and its spread.** Lagged regression of edge latitude on OHT convergence at the edge band
(and on subsurface 200–750 m T at 30–45°S), lags 0–80 yr. Fit a gamma kernel to the lag-response:
mean delay τ, spread σ. Check P ≈ 2τ across μ and whether σ/τ grows as μ decreases.

4.4 **Buoyancy.** Composite B_H and B_FW by phase; report which controls convection onset.

## 5. Goal 2 — Koopman / reduced RP spectra vs μ

Reuse the KDMD code in `external/Koopman-Dynamical-Response` (`koopman_response`); wrap, do not fork.
New module `src/gsebm/plasim_koopman.py` with tests on a synthetic stochastic Hopf / noisy focus
(Tantet et al. 2020 Part II) whose resonances are known analytically.

5.1 **Observables (fixed across μ).**
- Block A: global zonal-mean annual Ts on 32 Gaussian latitudes, sqrt(cos φ) weights.
- Block B: slow ocean variables — layer means of zonal-mean T in 200–750 m and 750–2000 m, per
  hemisphere, on 20–60° bands (4 scalars); variant: leading 2–3 EOFs of the zonal-mean 200–2000 m T
  field computed from the pooled 1240 run and **frozen** for all μ.
- Block C (optional): SH and NH sic-0.5 ice-edge latitudes.
All inputs: remove the post-transient mean only (no detrending for stationary runs; for 1230/1225
remove a linear trend per window and **recentre**).

5.2 **Kernel.** Block Gaussian kernel k(x,y) = exp(−Σ_b w_b ‖x_b − y_b‖² / m_b), m_b = median pairwise
squared distance of block b on the training set, frozen and reused for test data and for all μ (use the
1240 values as reference; also report per-μ m_b as sensitivity). Scan w_B ∈ {0, 0.5, 1, 2}, w_C ∈ {0, 1}.

5.3 **Protocol.** Annual data; lags τ ∈ {1, 2, 5, 10, 15} yr; equal-length windows across μ (length set
by the shortest stationary run); split-half and block bootstrap (block 200 yr, 200 resamples) for
uncertainty. Output eigenvalues converted to λ = log(μ_k)/τ (yr⁻¹).

5.4 **Diagnostics per μ.**
- Implied-timescale plot (−1/Re λ and 2π/Im λ vs lag) — a mode counts as resolved only if lag-consistent
  within bootstrap spread for τ ≥ 5 yr.
- Resonances in the complex plane with bootstrap clouds.
- Eigenfunction projections: correlation of leading eigenfunctions with global-mean T, SH ice edge,
  subsurface layer means, cycle phase (e^{iθ}) and amplitude.
- Comparison T-only (w_B = w_C = 0) vs augmented: the claim to test is that the ~40–80 yr mode appears
  and is lag-consistent only with Block B.

5.5 **Non-stationary runs (1230, 1225).** Sliding windows (e.g. 1500 yr, step 250 yr) with frozen m_b;
track the leading real resonance and the oscillatory pair vs window; flag windows with drift in
global T > 1 K.

5.6 **Independent validation.**
- ACF fits of SH zonal T and SH edge: C(t) = A e^{−γt} cos ωt + B e^{−t/τ}; compare γ, ω, τ with KDMD.
- Phase-conditioned analysis (anomalies w.r.t. the limit cycle): subtract the phase-bin mean
  (Section 3.1) from the fields, recompute ACF/KDMD; the amplitude (Floquet) relaxation rate should match
  the l = 1 resonance (32/38/51 yr at 1245/1240/1235 from previous analysis).

5.8 **NH-only fingerprint.** Same kernel/protocol, NH observables only (NH zonal Ts, 16 lats).
- (a) *Model-free:* fit the NH Ts (and NH-mean Ts) ACF with C(t) = A e^{−t/τ_f} + B e^{−t/τ_s}
  (+ optional damped cosine); report τ_s and B/(A+B) vs μ with bootstrap intervals.
- (b) *Delay-embedded KDMD:* augment NH Ts with its lagged copies x(t−kΔ), Δ ∈ {5, 10} yr,
  k = 0…K with K·Δ up to ~60 yr (or project delays onto leading EOFs to limit dimension). Use lag
  consistency (5.4) to decide whether a slow real mode is resolved and how many delays are needed.
- (c) *Projection:* correlate NH-only (with and without delays) leading eigenfunctions with the leading
  real mode and oscillatory pair from the full augmented analysis (5.4), and with global-mean Ts.
- Pass criterion for "NH fingerprints the transition": a lag-consistent real resonance whose −1/Re λ
  grows toward 1225 and whose eigenfunction correlates > 0.7 with the full-analysis leading real mode.
  Report explicitly whether the SH oscillatory pair is recoverable from NH data (expected: no, or weak).

5.7 **Summary figures (the paper story).**
- Fig K1: resonances in complex plane for all μ (one panel per μ, same axes).
- Fig K2: period 2π/Im λ, coherence −1/Re λ of the oscillatory pair, and −1/Re λ of the leading real
  mode vs μ, with ACF-fit and amplitude-relaxation estimates overlaid. Expected: parabola-like array at
  1245–1235, loss of the pair at ~1230, leading real mode slowing toward the collapse at 1225.
- Fig K3: leading eigenfunctions projected on (SH edge, subsurface T) plane and phase composites.
- Fig K4: NH-only fingerprint — τ_s and tail weight vs μ; NH delay-embedded resonances vs μ overlaid on
  the full-analysis leading real mode; NH vs SH ice-edge time series through the 1225 collapse with
  onset markers.

## 6. Optional: conceptual model and new μ runs

6.1 **Conceptual delayed EBM** (`src/gsebm/delayed_ebm.py`): existing Ghil–Sellers EBM (`gsebm.physics`)
+ one deep-ocean box per hemisphere + OHT convergence at the edge given by a gamma-distributed delay of
the edge-temperature anomaly with parameters (a, τ, σ, k) fitted from Section 4. Show it reproduces P(μ)
and the loss of oscillation when σ/τ exceeds the critical value. Tests: linear stability of the delayed
system vs a known analytical case (fixed delay, P → 2τ limit).

6.2 **New μ (user runs them, not Codex).** Branch from the same settled 1240 restart: 1242.5, 1237.5,
1232.5, 1228, 1226, 1224, 1222, 1220, 1218; ≥ 3000 yr after transient; save the fields of Section 2.
Codex only needs to make every script accept these experiments without code changes.

## 7. Deliverables and acceptance

- New tests all pass; existing tests unchanged.
- `data/Plasim/<EXP>/..._cold_branch_*.nc` for the priority μ.
- `analysis/plasim_cold_branch_mechanism.py` (marimo): Sections 3–4 figures.
- `analysis/plasim_cold_branch_koopman.py` (marimo): Section 5 figures.
- `figures/plasim_cold_branch/*.png|pdf`.
- `Plasim_Cold_Branch_Results.md`: one table per section with the hemispheric onset order and coupling
  lags (3.0), pass/fail for 3.2–3.4, the chain lags
  (4.1), τ, σ, P per μ (4.3), resonances with bootstrap intervals per μ (5.4), and the NH-only verdict (5.8).
- Also fix `Plasim_Diagnostics.md`: replace the legacy 26°N/27°S margins at 1240 with the per-cell 5 cm
  50%-occupancy edge (35.9°N / −34.0°) and sic-0.5 edge (38.2°N / −36.3°); state that Börner's thesis
  (Ch. 6 §6.1.2) defines the border as annual-mean thickness ≥ 5 cm, and that the zonal/occupancy
  reduction is ours.

## 8. Order of work

Task 0 → 2.1, 2.3, 2.8, 2.9 (cheap) → 3.0 (hemispheric order/coupling) → 3.1–3.4 (go/no-go on
physicality) → 2.4–2.7 → 4.x → 5.1–5.7 → 5.8 (NH-only fingerprint) → 6.1.
If 3.3 fails, stop after Section 3 and report before continuing.
