# PLASIM diagnostics

Use `scripts/extract_plasim_diagnostics.py` to process a long PLASIM-LSG
integration. The script reads the original annual atmospheric NetCDF files and
the accompanying PLASIM `DIAG` text files, then writes scalar diagnostics and
a separate zonal-field dataset. `analysis/checking_Plasim.py` is the
interactive plotting companion and reads those saved datasets.

## Command-line workflow

For an experiment directory containing `output/spinup` and `diag/spinup`, run:

```bash
PYTHONPATH=src uv run python scripts/extract_plasim_diagnostics.py \
  --experiment-dir /Volumes/Nicco/Plasim/experiments/CONTROL_360ppm_T21L10_10000Y_MU_1312
```

Use `--state` to select another state subdirectory, `--workers` to set the
number of worker processes, and `--output-dir` to override the destination.
By default the script writes, without overwriting existing files:

```text
data/Plasim/<experiment>/<experiment>_<state>_diagnostics.nc
data/Plasim/<experiment>/<experiment>_<state>_zonal_temperatures.nc
```

The diagnostics dataset contains global, regional, hemispheric, TOA, AMOC,
and sea-ice diagnostics. The zonal dataset contains the `time × lat` variables
`zonal_surface_temperature` (`ts`), `zonal_2m_temperature` (`tas`),
`zonal_sea_ice_concentration` (`sic`),
`zonal_persistent_sea_ice_fraction` (local annual `sic >= 0.5`),
`zonal_sea_ice_thickness` (`sit`), `zonal_surface_albedo` (`as`), and
`zonal_toa_energy_imbalance` (`rst + rlut`).
Temperature and albedo use an unweighted longitude mean. Sea-ice variables
exclude land while retaining ice-free ocean as zero.

The calculations are implemented in `src/gsebm/plasim_diagnostics.py`. Each
worker opens one input file, computes a small in-memory dataset, and closes the
source file before the results are concatenated and sorted by model year.

## Spatial and temporal averaging

The atmospheric output uses a Gaussian latitude grid. A global or regional
mean is evaluated in two stages:

1. take an unweighted arithmetic mean over longitude;
2. take a latitude mean using the Gaussian quadrature weights returned by
   `numpy.polynomial.legendre.leggauss`.

The latitude coordinate is checked against the descending Gaussian grid before
any diagnostic is evaluated. This prevents an ordinary latitude grid from
silently being combined with inappropriate weights. Regional means normalize
the retained Gaussian weights within the selected latitude band.

The NetCDF fields have already been averaged by PLASIM over each output year;
the diagnostic code does not average them over time again. PLASIM encodes time
as `YYYYMMDD.fraction`, so the integer model year is

```text
year = floor(time / 10000)
```

The `DIAG` file can contain several ocean diagnostics in one year. These are
grouped by the year in the preceding `LSG timestep ... date` line and averaged
arithmetically. Atmospheric and ocean diagnostics are then matched by integer
model year. The analysis raises an error if an atmospheric year has no matching
LSG diagnostic.

## Temperature diagnostics

All temperatures use `ts`, PLASIM code 139, which is the surface temperature
in kelvin. This is distinct from `tas`, the two-metre air temperature.

| Output variable | Evaluation |
| --- | --- |
| `global_temperature` | Gaussian-area-weighted mean over the full globe |
| `tropical_temperature` | weighted mean from 30 degrees S through 30 degrees N |
| `northern_polar_temperature` | weighted mean from 30 through 90 degrees N |
| `southern_polar_temperature` | weighted mean from 30 through 90 degrees S |
| `northern_hemisphere_surface_temperature` | weighted mean over the full Northern Hemisphere from `ts` |
| `southern_hemisphere_surface_temperature` | weighted mean over the full Southern Hemisphere from `ts` |
| `northern_hemisphere_2m_temperature` | weighted mean over the full Northern Hemisphere from `tas` |
| `southern_hemisphere_2m_temperature` | weighted mean over the full Southern Hemisphere from `tas` |
| `northern_polar_temperature_gradient` | tropical minus northern-polar temperature |
| `southern_polar_temperature_gradient` | tropical minus southern-polar temperature |

The gradients are positive when the tropical band is warmer than the
corresponding polar band.

## Top-of-atmosphere energy budget

PLASIM's native radiation variables use signed net fluxes. In the files used
here, upward shortwave and longwave radiation are negative. The relevant
variables are:

| PLASIM variable | Code | Meaning in the native output |
| --- | ---: | --- |
| `rst` | 178 | net TOA shortwave radiation, positive downward |
| `rlut` | 179 | net TOA longwave radiation, negative for outgoing radiation |
| `rsut` | 203 | TOA upward shortwave radiation, stored as a negative flux |

This interpretation follows the local PLASIM sources:

- `PLASIM_INFO/PLASIM-1.0/plasim/src/plasimmod.f90` describes `dswfl` and
  `dlwfl` as net solar and net thermal radiation;
- `PLASIM_INFO/PLASIM-1.0/plasim/src/radmod.f90` constructs those net fluxes
  by adding their signed upward and downward components;
- `PLASIM_INFO/PLASIM-1.0/plasim/src/outmod.f90` writes the top-level net
  fields as codes 178 and 179 and the upward solar field as code 203;
- `PLASIM_INFO/PLASIM-1.0/scripts/burn.sh` identifies `rst` as net shortwave
  and converts the negative native upward fields when producing conventional
  positive-outward variables.

The global diagnostics are therefore:

| Output variable | Formula | Units and positive direction |
| --- | --- | --- |
| `global_toa_incoming_shortwave` | `global_mean(rst - rsut)` | W m^-2, downward |
| `global_toa_reflected_shortwave` | `global_mean(-rsut)` | W m^-2, outward |
| `global_toa_absorbed_shortwave` | `global_mean(rst)` | W m^-2, absorbed/downward |
| `global_toa_outgoing_longwave` | `global_mean(-rlut)` | W m^-2, outward |
| `global_toa_energy_imbalance` | `global_mean(rst + rlut)` | W m^-2, into the climate system |
| `global_planetary_albedo` | reflected shortwave / incoming shortwave | dimensionless |

Because spatial averaging is linear, the flux sums can equivalently be formed
before or after taking their global means. Planetary albedo is deliberately
the ratio of the globally averaged reflected and incoming fluxes. It is not an
area-weighted average of grid-cell ratios.

A positive `global_toa_energy_imbalance` means the coupled climate system is
receiving net radiative energy at the top of the atmosphere. A negative value
means it is losing net radiative energy. Values near zero are an important
spin-up indicator, but should be interpreted together with temperature, ocean,
and sea-ice drift rather than as proof that every component is equilibrated.

The zonal-field product also stores
`zonal_toa_energy_imbalance = longitude_mean(rst + rlut)`. Its stationary
latitude profile can be compared with a temperature Koopman mode: away from
the poles, a zero crossing of this net TOA balance is an extremum of the
implied total meridional energy transport.

## AMOC diagnostics

The AMOC quantities come from lines beginning with `ATL max (NADW)` in the
PLASIM-LSG `DIAG` output. All occurrences within a model year are averaged,
and `amoc_sample_count` records how many entries contributed.

| Output variable | DIAG value | Units |
| --- | --- | --- |
| `amoc_strength` | first column; Atlantic overturning maximum at 46--66 degrees N below approximately 700 m | Sv |
| `amoc_strength_16_44n` | second column; Atlantic NADW overturning strength at 16--44 degrees N | Sv |
| `amoc_nadw_export_30s` | third column; Atlantic NADW export near 30 degrees S | Sv |
| `amoc_sample_count` | number of `ATL max (NADW)` entries in the annual mean | count |

The first selected depth is approximately 750 m on the 22-level LSG grid.
These diagnostics reproduce the values reported by PLASIM rather than
reconstructing an overturning streamfunction from gridded ocean output.

## Interim sea-ice diagnostics

Native gridded LSG ice output is not currently available. The analysis therefore
uses two complementary interim sources:

- annual `sit` and `lsm` fields from the atmospheric PLASIM NetCDF files for
  the location of the effective ice margin;
- bulk sea-ice values printed by LSG in the `DIAG` text files for area, volume,
  and mean thickness.

These sources should not be expected to agree exactly. The atmospheric fields
have been exchanged onto the Gaussian atmospheric grid, whereas the reported
bulk values are evaluated by LSG on its own ocean grid.

### Persistent annual sea-ice edge

PLASIM thresholds prognostic sea-ice compactness at 0.5 to form the binary
ice mask used by the surface-albedo calculation. The saved annual `sic` field
is the time mean of that model mask. A local annual value `sic >= 0.5`
therefore identifies an ocean grid cell that experienced the sea-ice surface
state for at least half of the year.

The diagnostic preserves that local classification order. It first thresholds
annual `sic` independently at every ocean grid cell and only then averages the
binary result over ocean longitudes. `zonal_persistent_sea_ice_fraction` is
the resulting fraction at each latitude. The zonal ice edge is its equatorward
0.5 crossing, corresponding to at least half the ocean longitudes being
persistently ice-covered. Crossings are linearly interpolated between Gaussian
latitudes. A missing crossing receives `NaN`; a threshold already exceeded at
the ocean-bearing latitude nearest the equator is reported as 0 degrees.
Southern Hemisphere edge latitudes are negative.

| Output variable | Meaning |
| --- | --- |
| `northern_persistent_sea_ice_edge_latitude` | Northern annual persistent-ice edge |
| `southern_persistent_sea_ice_edge_latitude` | Southern annual persistent-ice edge |

This is an annual persistence diagnostic, not a seasonal maximum or minimum
ice edge. PLASIM applies the compactness threshold instantaneously; the second
annual `sic >= 0.5` test is part of the offline diagnostic.

### Bulk values from DIAG

The line beginning `Icevol. m**3` reports total volume, ice-covered area, and
their ratio as mean thickness. The `Iceareas:` line contains, in order,
Northern area, Southern area, Northern volume, and Southern volume. Inspection
of `PLASIM_INFO/PLASIM-1.0/lsg/src/lsgmod.f90` shows that LSG includes cells
whose thickness exceeds 0.01 m and prints the four hemispheric values after
multiplication by `1e-12`. The parser multiplies them by `1e12` to restore SI
units.

| Output variable | Units |
| --- | --- |
| `lsg_sea_ice_area` | m2 |
| `lsg_sea_ice_volume` | m3 |
| `lsg_sea_ice_mean_thickness` | m |
| `lsg_northern_sea_ice_area` | m2 |
| `lsg_southern_sea_ice_area` | m2 |
| `lsg_northern_sea_ice_volume` | m3 |
| `lsg_southern_sea_ice_volume` | m3 |
| `lsg_sea_ice_sample_count` | count |

Every reported value in a model year is averaged arithmetically. The annual
mean thickness is the mean of LSG's reported instantaneous ratios, not a new
ratio formed from the annual mean volume and area.

## Future native LSG ice diagnostics

Once gridded LSG output is retained, the native ocean grid should become the
primary source for sea-ice diagnostics. The next workflow should:

1. save ice thickness and, if available, concentration at sufficient cadence
   to resolve the seasonal cycle, together with the native grid-cell areas;
2. calculate cell-area-weighted global and hemispheric ice area, extent, and
   volume directly on the LSG grid;
3. calculate seasonal minima and maxima instead of inferring them from annual
   atmospheric means;
4. derive longitude-dependent ice-edge locations and documented hemispheric
   summaries, retaining explicit thickness or concentration thresholds;
5. compare the native calculations with the printed `DIAG` totals and with
   atmospheric-grid `sic` and `sit` before replacing any interim series.

During validation, native-grid, DIAG, and atmospheric-grid quantities should
remain separately named so coupling, remapping, temporal sampling, and
threshold differences are visible.

## Observed dynamical regimes in the current 360 ppm integrations

The following are exploratory observations from the currently saved diagnostic
datasets. They describe the simulated trajectories, not verified physical
mechanisms or predictions for the real climate system. The integrations use
different parts of state space as `mu` changes, so their fluctuations should
not be treated as samples from one common stationary process.

| `mu` | Observed trajectory after the listed transient | Interpretation to test, not an established mechanism |
| ---: | --- | --- |
| 1240 | A transition near years 5000--6000 leads to a cold, weak-AMOC state. Thereafter the temperature, AMOC, and ice diagnostics fluctuate weakly. The Northern persistent annual sea-ice edge lies near 38 degrees N, leaving an ice-free tropical belt rather than a full snowball state. | A highly ice-covered, weak-AMOC equilibrium candidate. |
| 1265 | The cold, ice-rich Southern Hemisphere shows a persistent approximately 110-year ice--temperature oscillation after year 4000. | A centennial Southern ice--temperature mode. |
| 1288 | The Southern ice area, volume, margin, and Southern temperature have a persistent, strongly anti-phased mode with an approximate 730-year period after year 4000. | A limit-cycle-like or quasi-periodic Southern ice--temperature mode. |
| 1312 | The AMOC weakens from about 22 Sv to about 2 Sv during the initial ice-growth phase around years 3000--3300. Later, Southern ice retreats and recovers on multi-century to millennial intervals while the AMOC remains weak. | A weak-AMOC regime with recurrent Southern ice adjustments. |
| 1367 | A Southern sea-ice retreat and warming transition develops over roughly years 3500--3530; the later part of the integration is comparatively steady. | A delayed adjustment between two ice-cover regimes. |

These labels are deliberately descriptive. In particular, a periodic-looking
series is not sufficient to establish a mathematical limit cycle, and a
transition does not alone demonstrate a bifurcation or a faulty restart.
Repeated perturbation/branch experiments from saved restart states are needed
to determine whether a trajectory returns to a fixed point, converges to a
cycle, or enters a different attractor.

Autocorrelation functions should be used only within a settled regime and for
like-for-like comparisons. They mix transient drift, periodicity, and local
relaxation when applied across these different `mu` values. For the oscillatory
runs, report spectral peaks, amplitude, phase relations, and recurrence in
addition to any integrated correlation time.

## Scope and reproducibility

The diagnostic workflow is read-only with respect to simulation results. It
does not modify PLASIM, restart files, annual NetCDF output, or `DIAG` files.
Variable attributes in the returned `xarray.Dataset` record units, spatial
averaging, formulas, and sign conventions so that saved or plotted values can
be interpreted without relying only on this document.
