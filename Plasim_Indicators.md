# PlaSim-LSG indicators: definitions

This document defines the annual quantities that
`scripts/slurm/compute_plasim_indicators.py` computes from PlaSim-LSG output,
and how global, hemispheric and depth-band indicators are obtained from them.
The code in `src/gsebm/plasim_grids.py` and `src/gsebm/plasim_indicators.py`
follows it; each function names the section it implements.

Every rule below was checked against the model source (PlaSim 1.0,
`plasim/src/*.f90` and `plasim/bld/lsgmod.f90`) and against one decade of
output. Source routines are named in brackets.

## 1. Input

- One file per component and decade, `<run>_<PLA|LSG|ICE|OCE>.<first>-<last>.nc`,
  in `<run>/output/<phase>/`. The phase is `spinup` for most runs.
- Every field is an **annual mean**, written by the postprocessing
  (`cdo yearmean`). There is no seasonal information.
- A decade is used only when all four files exist and each holds ten annual
  records with consecutive model years (the time check of
  `gsebm.plasim_run_inventory.output_block`). Labels must be contiguous.
- `year` is the label from the file name. `model_year` is the year in the
  file's `time` coordinate. Their difference is constant within a run and is
  stored.
- Annual means of linear quantities (fluxes, velocities, streamfunctions,
  areas, volumes) are exact. A maximum or a threshold applied to an annual
  mean is not the annual mean of the instantaneous quantity.

## 2. Grids and geometry

### 2.1 Atmosphere, sea ice and slab ocean (PLA, ICE, OCE): T21 Gaussian grid

- 32 latitudes × 64 longitudes. The latitudes are the Gaussian latitudes
  φ_j = arcsin(x_j), with x_j and the weights w_j from Gauss–Legendre
  quadrature of order 32 (Σ w_j = 2), ordered north to south (`inigau`).
  There is no latitude on the equator.
- Planet radius R_P = 6 371 220 m (`p_earth.f90`, `plarad`).
- Area of latitude band j: A_j = 2π R_P² w_j. Each of its 64 cells has
  A_j / 64.
- Ocean points: `lsm` < 0.5 (`lsm` is 0 or 1 and constant in time). The
  ocean fraction of latitude j is f_j = (number of ocean points in row j) / 64.

### 2.2 Ocean (LSG): semi-staggered E grid

- 76 rows, 72 points per row. Row latitude φ_j = 2.5° × (38 − j + 0.5) for
  j = 1 … 76 (`inigr`); rows with |φ| > 90° are ghost rows and are never
  wet. Scalar points (`t`, `s`, `zeta`, `psi`) and vector points (`utot`,
  `vtot`) alternate along each row, and alternate rows are shifted by 2.5°.
- LSG radius R_L = 6 371 000 m. Grid step `dphi` = R_L × 5° (in radians),
  zonal step `dl(j)` = `dphi` cos φ_j (`inigr`).
- Horizontal area of one point: a_j = ½ `dl(j)` `dphi` = ½ (R_L 5°)² cos φ_j.
  The ½ is because scalar and vector points share the 5° × 5° boxes.
- Layer centres `du(k)` (`inipar`, 22 levels):
  25, 75, 125, 175, 225, 275, 350, 450, 550, 650, 750, 850, 950, 1100,
  1300, 1500, 1800, 2250, 2750, 3500, 4500, 5500 m, and `du(23)` = 6500 m.
- Interfaces (`inigr`): `dw(k)` = (`du(k)` + `du(k+1)`)/2, i.e. 50, 100, 150,
  200, 250, 312.5, 400, 500, 600, 700, 800, 900, 1025, 1200, 1400, 1650,
  2025, 2500, 3125, 4000, 5000, 6000 m; `dw(0)` = 0. Nominal thickness
  `ddw(k)` = `dw(k)` − `dw(k−1)`.
- **Thickness of layer k at a point of depth D**: the rule for `ddz` at
  scalar points (D = `depp`) and for `delta` at vector points (D = `depv`)
  ("Compute values for wet and ddz" and "Computation of layer thickness
  delta" in `lsgmod.f90`):
  - 0 if D ≤ `du(k)` (the layer is dry);
  - D − `dw(k−1)` if `du(k)` < D < `du(k+1)`;
  - `ddw(k)` otherwise.

  So the deepest wet layer runs from its top interface down to the sea floor,
  which can lie below the layer's nominal bottom interface.
- Wet masks: scalar layer k is wet where `depp` > `du(k)` (the file's
  `wet`); vector layer k is wet where `delta` > 0.1 m (the file's `wetvec`).
  The code checks that both rules reproduce the files.
- Wet volume of a scalar cell: V = a_j × `ddz(k)` × wet. The free surface
  `zeta` is left out of V, except in the salt content (§3.4).
- Constants (`lsgmod.f90`, "values changed according to PlaSim"):
  ρ₀ = 1030 kg m⁻³, c_p = 4180 J kg⁻¹ K⁻¹.

### 2.3 Masks

- **Atlantic** (scalar points; used for the vector point with the same index,
  as LSG does for its own overturning, `psimer`):
  1. basin codes 1 (Atlantic) and 4 (Arctic) from LSG's basin file `beklas`,
     as read (format `(1x,72i1)`, rows 3 … 74);
  2. `beklas` leaves 20 wet surface cells unlabelled. Eight lie on the
     Atlantic side and are added: Gulf of Mexico (18.75 °N, 267.5 °E),
     Labrador Sea and southern Greenland coast (61.25–66.25 °N, 307.5–317.5 °E,
     six cells), northern Baffin Bay (78.75 °N, 287.5 °E). The other twelve
     (Canadian Arctic, Bering Strait, Mediterranean, Indonesian seas, off
     Mozambique) are not. Both lists are written out in
     `plasim_grids.py`, and an unlabelled wet cell in neither list stops the
     computation;
  3. removed: south of 34 °S; the Mediterranean (0–40 °E and 355–360 °E,
     20–40 °N); Hudson Bay (265–283 °E, 50–64 °N); north of 80 °N; the Arctic
     Ocean proper (20–285 °E, 65–90 °N).

  These cuts follow Börner et al. (2026, `regions.Atlantic` in their code).
  LSG's own printed "ATL" values use a different basin, which reaches south
  to Antarctica and contains a few Indo-Pacific cells.
- **South American land points** (for the ACC): surface-dry scalar points at
  20–50 °S, 285–315 °E.

## 3. Stored quantities

Two data files per run. All variables are annual (`year` dimension), in
float32.

### 3.1 Zonal file, T21 fields (dimension `t21_lat`, 32)

Zonal mean over all 64 points of a latitude:

| Variable | Field | Units |
|---|---|---|
| `tas_zonal` | `tas`: lowest-level temperature brought to the surface along a dry adiabat, accumulated over every step (`fluxmod.f90`, `ntsa` = 2) | K |
| `ts_zonal` | `ts`: surface (skin) temperature, sampled at output times | K |
| `rst_zonal` | `rst`: net solar flux at the top, positive down | W m⁻² |
| `rlut_zonal` | `rlut`: net thermal flux at the top, negative (upward) | W m⁻² |
| `rsut_zonal` | `rsut`: upward solar flux at the top, negative | W m⁻² |
| `surface_net_flux_zonal` | `rss` + `rls` + `hfss` + `hfls`, positive down | W m⁻² |

Zonal mean over the ocean points of a latitude (NaN where there are none):

| Variable | Field | Units |
|---|---|---|
| `sst_ocean_zonal` | OCE `sst`: the 50 m slab temperature | K |
| `sic_ocean_zonal` | `sic`: instantaneous cover is 0 or 1, so the annual mean is the fraction of the year ice-covered | 1 |
| `sit_ocean_zonal` | `sit`: ice thickness, 0 when ice-free | m |
| `fldoa_ocean_zonal` | OCE `fldoa`: heat flux from LSG to the slab | W m⁻² |
| `cfluxra_ocean_zonal` | ICE `cfluxra`: heat flux used to hold ice at the 9 m cap | W m⁻² |

Static: `t21_gaussian_weight` (w_j), `t21_band_area` (A_j, m²),
`t21_ocean_fraction` (f_j).

### 3.2 Zonal file, LSG surface fields (dimension `lsg_row`, 76)

Area-weighted mean over the wet surface points of a row (NaN where there
are none):

| Variable | Field | Units |
|---|---|---|
| `zeta_zonal` | `zeta`: sea level | m |
| `flukhea_zonal` | `flukhea`: heat flux into LSG's top layer from the coupling | W m⁻² |

For the check of §5, per row (not zonal means):

| Variable | Definition | Units |
|---|---|---|
| `net_meridional_transport` | Σ_k T(k, j) over all wet vector points (§3.5), northward positive | Sv |
| `freshwater_flux_north_of` | Σ over rows 1 … j of Σ_i `fluwat` a_j over wet points | Sv |

Static: `lsg_row_lat`, `lsg_row_wet_area` (Σ a_j over wet surface points, m²).

### 3.3 Zonal file, scalars (dimension `year` only)

| Variable | Definition | Units |
|---|---|---|
| `salt_content` | §3.4 | ‰ m³ |
| `zeta_abs_max` | largest \|`zeta`\| over wet points (LSG clips `zeta` at ±25 m) | m |
| `acc_even_rows`, `acc_odd_rows` | mean of `psi` over the South American land points of even (odd) rows, rows numbered 1 … 76 | Sv |
| `acc` | mean of the two | Sv |

`psi` is LSG's barotropic streamfunction, integrated northward from
Antarctica, where it is 0, in steps of two rows (`zetapsi`). It is in Sv
(the factor 10⁻⁶ is in the source), although the file says m³ s⁻¹. On a
continent it equals the net eastward transport between Antarctica and that
continent; on South America that is the Drake Passage transport. The two
row parities are integrated separately, so they give two values.

### 3.4 Salt content

M_S = Σ `s` a_j h_k over wet scalar cells, with h_k = `ddz(k)` for k > 1 and
h_1 = `ddz(1)` + `zeta` − `sice` for the top layer (LSG's own salt inventory,
`Salt inv`). LSG uses a real freshwater flux, so salt is conserved and
volume is not: M_S should stay constant, and the global mean salinity moves
with sea ice and sea level.

### 3.5 Sections file (dimensions `lsg_row` 76 × `lsg_level` 22)

| Variable | Definition | Units |
|---|---|---|
| `t_section` | Σ_i `t` V / Σ_i V over the wet cells of the row and level | K |
| `s_section` | the same for `s` | ‰ |
| `s_atlantic_section` | the same over Atlantic cells only | ‰ |
| `psi_atlantic` | overturning streamfunction, Atlantic (below) | Sv |
| `psi_global` | the same over all wet vector points | Sv |

Static: `wet_volume` (Σ_i V, m³), `wet_volume_atlantic`,
`lsg_level_depth` (`du`), `lsg_level_top` (`dw(k−1)`), `lsg_level_bottom` (`dw(k)`).

**Overturning streamfunction** (`psimer` in `lsgmod.f90`; Börner et al.
2026, eq. 2.1):

ψ(j, k) = − Σ_{k' ≥ k} Σ_i `vtot`(i, j, k') `delta`(i, j, k') `dl(j)` × 10⁻⁶,

summed over the vector points i of row j in the basin, from the sea floor up
to and including level k. So ψ(j, k) is the streamfunction at the top
interface of level k, depth `dw(k−1)`. The minus sign makes the NADW cell
(northward near the surface, southward at depth) positive, as in LSG's
printed values and in Börner et al. ψ is linear in `vtot`, so the annual
mean is exact.

ψ at the surface (k = 1) is minus the net volume transport across the row,
which is not zero. LSG has a real freshwater flux, so in a steady state the
net southward volume transport across a row equals the net surface
freshwater flux (`fluwat`) north of it: a few tenths of a Sv. In one sample
decade the two agree to within about 0.03 Sv (correlation 0.996 over rows).

## 4. Indicators computed from the stored quantities

All of these are linear in the stored zonal means, so computing them from
the zonal means is exact.

- **Area mean** of a T21 field over a region R (globe, NH = rows with φ > 0,
  SH = rows with φ < 0): Σ_{j∈R} w_j z̄_j / Σ_{j∈R} w_j. For ocean-only fields
  the weight is w_j f_j.
- **Net TOA flux** N = mean of `rst_zonal` + `rlut_zonal` (positive into the
  climate system). Absorbed solar = `rst`, outgoing longwave = −`rlut`,
  incoming solar S = `rst` − `rsut`. It should equal
  GSOL0/4 × (1 − e²)^(−½), with e the run's orbital eccentricity.
- **Planetary albedo** = −mean(`rsut`) / mean(S), a ratio of area means.
- **Sea-ice area** in R = Σ_{j∈R} A_j f_j `sic_ocean_zonal`_j.
  **Ice volume** = the same with `sit`. **Cap flux** (W) = the same with
  `cfluxra`.
- **Depth bands** (whole LSG layers): upper 0–700 m = levels 1–10;
  intermediate 700–2025 m = levels 11–17; deep 2025 m–bottom = levels 18–22.
  The bottom cells of a band's last layer can reach slightly below its
  nominal bottom.
- **Ocean volume mean** of `t` or `s` over rows R and a band B:
  Σ_{j∈R, k∈B} section × `wet_volume` / Σ `wet_volume`.
- **Ocean heat content** = ρ₀ c_p Σ `t_section` × `wet_volume` (relative to
  0 K; only its changes matter).
- **Global mean sea level** = Σ_j `zeta_zonal` × `lsg_row_wet_area` /
  Σ `lsg_row_wet_area`.
- **AMOC strength** (Börner et al. 2026): maximum of `psi_atlantic` over all
  depths and over the rows in 46–66 °N; also the maximum over depth at the
  row nearest 26 °N. In a collapsed state the maximum can sit at depth and
  measure a weak deep cell instead.

## 5. Checks made while computing

- File latitudes equal the Gaussian latitudes; `wet` and `wetvec` equal the
  rules of §2.2; the unlabelled wet cells in `beklas` are the expected ones.
- Incoming solar is the same in every year of a run.
- Salt content changes by less than 10⁻⁶ of its value within a run;
  otherwise a warning.
- Freshwater balance: in the time mean over the run, `net_meridional_transport`
  + `freshwater_flux_north_of` should be close to 0 in every row (within
  single years it is not, because sea level stores water). The largest
  absolute value over rows is reported (0.047 Sv in the sample decade). It
  tests the vector-point geometry.

## References

- Börner R, Mehling O, von Hardenberg J, Lucarini V (2026). Global stability
  of the Atlantic overturning circulation: edge state, long transients and
  boundary crisis under CO2 forcing. Phil. Trans. R. Soc. A 384: 20250087.
  doi:10.1098/rsta.2025.0087. Code and data: doi:10.5281/zenodo.17053348.
- PlaSim 1.0 source: https://github.com/jhardenberg/PLASIM (tag `1.0`).
