"""Climate diagnostics for PLASIM NetCDF and text output."""

from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Sequence

import numpy as np
import xarray as xr


_LSG_DATE_PATTERN = re.compile(
    r"\*\s*LSG timestep\s+(\d+)\s+date:\s+\d{1,2}-[A-Za-z]{3}-(\d+|\*{4})\s*\*"
)
_OUTPUT_YEAR_RANGE_PATTERN = re.compile(
    r"_(?:PLA|DIAG|LSG)\.(\d+)-(\d+)\.(?:nc|txt)$"
)
_ATMOSPHERIC_OUTPUT_PATTERN = re.compile(r".+_PLA\.\d+-\d+\.nc$")
_DIAGNOSTIC_OUTPUT_PATTERN = re.compile(r".+_DIAG\.\d+-\d+\.txt$")
_LSG_OUTPUT_PATTERN = re.compile(r".+_LSG\.\d+-\d+\.nc$")
_LSG_STEPS_PER_YEAR = 36
_FLOAT_PATTERN = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[EeDd][-+]?\d+)?"
_AMOC_PATTERN = re.compile(
    rf"ATL max \(NADW\)\s*:\s*({_FLOAT_PATTERN})\s+"
    rf"({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})"
)
_ICE_TOTAL_PATTERN = re.compile(
    rf"Icevol\. m\*\*3\s+({_FLOAT_PATTERN})\s+"
    rf"icecov\.area m\*\*2\s+({_FLOAT_PATTERN})\s+"
    rf"Av\.thickness in m\s+({_FLOAT_PATTERN})"
)
_ICE_HEMISPHERE_PATTERN = re.compile(
    rf"Iceareas:\s+({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s+"
    rf"({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})"
)

SEA_ICE_COVER_THRESHOLD = 0.50
SEA_ICE_ZONAL_OCCUPANCY_THRESHOLD = 0.50
_DIAG_SCALE_TO_SI = 1.0e12

LSG_EARTH_RADIUS_M = 6_371_000.0
LSG_HORIZONTAL_GRID_SPACING_DEGREES = 5.0
LSG_REFERENCE_DENSITY_KG_M3 = 1030.0
LSG_SPECIFIC_HEAT_J_KG_K = 4180.0
LSG_REFERENCE_TEMPERATURE_K = 273.16
LSG_OCEAN_REGIONS = (
    ("global", -90.0, 90.0),
    ("northern_midlatitudes", 20.0, 60.0),
    ("southern_midlatitudes", -60.0, -30.0),
)
LSG_OCEAN_DEPTH_BANDS_M = (
    ("0_200m", 0.0, 200.0),
    ("200_750m", 200.0, 750.0),
    ("750_2000m", 750.0, 2000.0),
)


@dataclass(frozen=True)
class PlasimDiagnosticDatasets:
    """Derived datasets and input counts for one PLASIM experiment state."""

    diagnostics: xr.Dataset
    zonal_temperatures: xr.Dataset
    atmospheric_file_count: int
    diagnostic_file_count: int
    worker_count: int


def find_lsg_ocean_files(experiment_dir: str | Path, state: str) -> list[Path]:
    """Return annual-mean LSG NetCDF files for one experiment state."""
    output_dir = Path(experiment_dir) / "output" / state
    if not output_dir.is_dir():
        raise FileNotFoundError(f"PLASIM output directory does not exist: {output_dir}.")

    paths = sorted(
        (
            path
            for path in output_dir.iterdir()
            if path.is_file()
            and not path.name.startswith("._")
            and _LSG_OUTPUT_PATTERN.fullmatch(path.name) is not None
        ),
        key=lambda path: int(
            _OUTPUT_YEAR_RANGE_PATTERN.search(path.name).group(1)  # type: ignore[union-attr]
        ),
    )
    if not paths:
        raise FileNotFoundError(f"No LSG NetCDF files found in {output_dir}.")
    return paths


def find_plasim_input_files(
    experiment_dir: str | Path, state: str
) -> tuple[list[Path], list[Path]]:
    """Return valid atmospheric and LSG diagnostic files for one experiment state."""
    experiment_path = Path(experiment_dir)
    atmospheric_dir = experiment_path / "output" / state
    diagnostic_dir = experiment_path / "diag" / state
    if not atmospheric_dir.is_dir():
        raise FileNotFoundError(f"Atmospheric output directory does not exist: {atmospheric_dir}.")
    if not diagnostic_dir.is_dir():
        raise FileNotFoundError(f"LSG diagnostic directory does not exist: {diagnostic_dir}.")

    atmospheric_paths = sorted(
        path
        for path in atmospheric_dir.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and _ATMOSPHERIC_OUTPUT_PATTERN.fullmatch(path.name) is not None
    )
    diagnostic_paths = sorted(
        path
        for path in diagnostic_dir.iterdir()
        if path.is_file()
        and not path.name.startswith("._")
        and _DIAGNOSTIC_OUTPUT_PATTERN.fullmatch(path.name) is not None
    )
    if not atmospheric_paths:
        raise FileNotFoundError(f"No PLASIM atmospheric NetCDF files found in {atmospheric_dir}.")
    if not diagnostic_paths:
        raise FileNotFoundError(f"No PLASIM LSG diagnostic files found in {diagnostic_dir}.")
    return atmospheric_paths, diagnostic_paths


def _float_groups(match: re.Match[str]) -> tuple[float, ...]:
    """Return matched Fortran-formatted numbers as Python floats."""
    return tuple(
        float(value.replace("D", "E").replace("d", "e"))
        for value in match.groups()
    )


def _year_from_filename(path: str | Path, timestep: int) -> int:
    """Return the output-model year for an LSG timestep from its file name."""
    filename_match = _OUTPUT_YEAR_RANGE_PATTERN.search(Path(path).name)
    if filename_match is None:
        raise ValueError(
            "Cannot recover an output-model year: expected a PLASIM output filename "
            f"ending in '_PLA.<start>-<end>.nc' or '_DIAG.<start>-<end>.txt', got {path}."
        )

    start_year, end_year = (int(value) for value in filename_match.groups())
    if end_year < start_year:
        raise ValueError(f"Invalid diagnostic year range in {path}.")

    year = start_year + (timestep - 1) // _LSG_STEPS_PER_YEAR
    if not start_year <= year <= end_year:
        raise ValueError(
            f"LSG timestep {timestep} resolves to output-model year {year}, outside the "
            f"{start_year}-{end_year} range declared by {path}."
        )
    return year


def _filename_years(path: str | Path, count: int) -> np.ndarray | None:
    """Return consecutive output-model years declared by a PLASIM filename."""
    filename_match = _OUTPUT_YEAR_RANGE_PATTERN.search(Path(path).name)
    if filename_match is None:
        return None

    start_year, end_year = (int(value) for value in filename_match.groups())
    years = np.arange(start_year, end_year + 1, dtype=int)
    if years.size != count:
        raise ValueError(
            f"{path} declares {start_year}-{end_year} in its filename, but contains "
            f"{count} annual records."
        )
    return years


def _parse_lsg_diagnostic_file(
    path: str | Path,
) -> tuple[
    dict[int, list[tuple[float, float, float]]],
    dict[int, list[tuple[float, float, float]]],
    dict[int, list[tuple[float, float, float, float]]],
]:
    """Parse AMOC and sea-ice records from one PLASIM DIAG file."""
    amoc_by_year: dict[int, list[tuple[float, float, float]]] = {}
    ice_totals_by_year: dict[int, list[tuple[float, float, float]]] = {}
    ice_hemispheres_by_year: dict[
        int, list[tuple[float, float, float, float]]
    ] = {}
    current_year: int | None = None

    with Path(path).open(encoding="utf-8", errors="replace") as diagnostic_file:
        for line in diagnostic_file:
            date_match = _LSG_DATE_PATTERN.search(line)
            if date_match is not None:
                timestep = int(date_match.group(1))
                displayed_year = date_match.group(2)
                # PLASIM's printed LSG calendar can be offset from the model
                # output calendar and overflows at year 10000.  The output
                # filename is consequently the canonical model-year source
                # whenever it declares a year range.
                current_year = (
                    _year_from_filename(path, timestep)
                    if _OUTPUT_YEAR_RANGE_PATTERN.search(Path(path).name) is not None
                    else int(displayed_year)
                )
                continue

            matches = (
                ("AMOC", _AMOC_PATTERN.search(line), amoc_by_year),
                ("sea-ice total", _ICE_TOTAL_PATTERN.search(line), ice_totals_by_year),
                (
                    "hemispheric sea-ice",
                    _ICE_HEMISPHERE_PATTERN.search(line),
                    ice_hemispheres_by_year,
                ),
            )
            for diagnostic_name, match, values_by_year in matches:
                if match is None:
                    continue
                if current_year is None:
                    continue
                values_by_year.setdefault(current_year, []).append(
                    _float_groups(match)
                )

    return amoc_by_year, ice_totals_by_year, ice_hemispheres_by_year


def _amoc_dataset(
    values_by_year: dict[int, list[tuple[float, float, float]]],
    path: str | Path,
) -> xr.Dataset:
    """Build annual AMOC diagnostics from parsed values."""
    if not values_by_year:
        raise ValueError(f"No 'ATL max (NADW)' diagnostics found in {path}.")

    years = np.asarray(sorted(values_by_year), dtype=int)
    annual_means = np.asarray(
        [np.mean(values_by_year[year], axis=0) for year in years], dtype=float
    )
    sample_counts = np.asarray(
        [len(values_by_year[year]) for year in years], dtype=int
    )

    return xr.Dataset(
        {
            "amoc_strength": (
                "year",
                annual_means[:, 0],
                {
                    "units": "Sv",
                    "long_name": "annual mean Atlantic overturning strength",
                    "latitude_range": "46 to 66 degrees N",
                    "depth_range": "below approximately 700 m; first selected level is 750 m on the 22-level grid",
                    "source": "PLASIM DIAG ATL max (NADW), first column",
                },
            ),
            "amoc_strength_16_44n": (
                "year",
                annual_means[:, 1],
                {
                    "units": "Sv",
                    "long_name": "annual mean Atlantic NADW overturning strength at 16 to 44 degrees N",
                },
            ),
            "amoc_nadw_export_30s": (
                "year",
                annual_means[:, 2],
                {
                    "units": "Sv",
                    "long_name": "annual mean Atlantic NADW export near 30 degrees S",
                },
            ),
            "amoc_sample_count": (
                "year",
                sample_counts,
                {"long_name": "number of PLASIM AMOC diagnostics in annual mean"},
            ),
        },
        coords={"year": years},
        attrs={
            "amoc_aggregation": "arithmetic mean of all ATL max (NADW) entries in each model year",
        },
    )


def amoc_diagnostics_for_file(path: str | Path) -> xr.Dataset:
    """Return annual AMOC diagnostics parsed from one PLASIM DIAG file."""
    amoc_by_year, _, _ = _parse_lsg_diagnostic_file(path)
    return _amoc_dataset(amoc_by_year, path)


def lsg_diagnostics_for_file(path: str | Path) -> xr.Dataset:
    """Return annual AMOC and sea-ice diagnostics from one PLASIM DIAG file."""
    amoc_by_year, ice_totals_by_year, ice_hemispheres_by_year = (
        _parse_lsg_diagnostic_file(path)
    )
    amoc = _amoc_dataset(amoc_by_year, path)

    if not ice_totals_by_year or not ice_hemispheres_by_year:
        raise ValueError(f"No complete LSG sea-ice diagnostics found in {path}.")
    years = sorted(ice_totals_by_year)
    if years != sorted(ice_hemispheres_by_year):
        raise ValueError(f"LSG sea-ice diagnostic years do not match in {path}.")
    for year in years:
        if len(ice_totals_by_year[year]) != len(ice_hemispheres_by_year[year]):
            raise ValueError(
                f"LSG sea-ice diagnostic sample counts do not match for year {year} in {path}."
            )

    year_values = np.asarray(years, dtype=int)
    total_means = np.asarray(
        [np.mean(ice_totals_by_year[year], axis=0) for year in years], dtype=float
    )
    hemisphere_means = np.asarray(
        [np.mean(ice_hemispheres_by_year[year], axis=0) for year in years],
        dtype=float,
    )
    sample_counts = np.asarray(
        [len(ice_totals_by_year[year]) for year in years], dtype=int
    )
    ice = xr.Dataset(
        {
            "lsg_sea_ice_volume": (
                "year",
                total_means[:, 0],
                {
                    "long_name": "annual mean LSG total sea-ice volume",
                    "units": "m3",
                    "source": "PLASIM DIAG Icevol. m**3",
                },
            ),
            "lsg_sea_ice_area": (
                "year",
                total_means[:, 1],
                {
                    "long_name": "annual mean LSG ice-covered area",
                    "units": "m2",
                    "source": "PLASIM DIAG icecov.area m**2",
                },
            ),
            "lsg_sea_ice_mean_thickness": (
                "year",
                total_means[:, 2],
                {
                    "long_name": "annual mean LSG reported sea-ice thickness",
                    "units": "m",
                    "source": "PLASIM DIAG Av.thickness in m",
                },
            ),
            "lsg_northern_sea_ice_area": (
                "year",
                hemisphere_means[:, 0] * _DIAG_SCALE_TO_SI,
                {
                    "long_name": "annual mean LSG Northern Hemisphere ice-covered area",
                    "units": "m2",
                    "source": "PLASIM DIAG Iceareas, first value",
                },
            ),
            "lsg_southern_sea_ice_area": (
                "year",
                hemisphere_means[:, 1] * _DIAG_SCALE_TO_SI,
                {
                    "long_name": "annual mean LSG Southern Hemisphere ice-covered area",
                    "units": "m2",
                    "source": "PLASIM DIAG Iceareas, second value",
                },
            ),
            "lsg_northern_sea_ice_volume": (
                "year",
                hemisphere_means[:, 2] * _DIAG_SCALE_TO_SI,
                {
                    "long_name": "annual mean LSG Northern Hemisphere sea-ice volume",
                    "units": "m3",
                    "source": "PLASIM DIAG Iceareas, third value",
                },
            ),
            "lsg_southern_sea_ice_volume": (
                "year",
                hemisphere_means[:, 3] * _DIAG_SCALE_TO_SI,
                {
                    "long_name": "annual mean LSG Southern Hemisphere sea-ice volume",
                    "units": "m3",
                    "source": "PLASIM DIAG Iceareas, fourth value",
                },
            ),
            "lsg_sea_ice_sample_count": (
                "year",
                sample_counts,
                {"long_name": "number of LSG sea-ice reports in annual mean"},
            ),
        },
        coords={"year": year_values},
        attrs={
            "lsg_sea_ice_aggregation": (
                "arithmetic mean of the reported LSG sea-ice diagnostics in each model year"
            ),
            "lsg_iceareas_scale": (
                "printed hemispheric areas and volumes multiplied by 1e12 to recover SI units"
            ),
        },
    )
    missing_amoc_years = sorted(set(years) - set(amoc["year"].values.tolist()))
    if missing_amoc_years:
        raise ValueError(f"Missing AMOC diagnostics for model years {missing_amoc_years} in {path}.")
    return xr.merge([amoc, ice], combine_attrs="no_conflicts")


def _sea_ice_margin_latitude(
    latitude: np.ndarray,
    zonal_profile: np.ndarray,
    hemisphere_sign: int,
    threshold: float,
) -> float:
    """Return the equatorward threshold crossing in one hemisphere."""
    latitude_magnitude = hemisphere_sign * latitude
    valid = (latitude_magnitude > 0.0) & np.isfinite(zonal_profile)
    if not np.any(valid):
        return np.nan

    magnitudes = latitude_magnitude[valid]
    profile = zonal_profile[valid]
    order = np.argsort(magnitudes)
    magnitudes = magnitudes[order]
    profile = profile[order]
    qualifying = np.flatnonzero(profile >= threshold)
    if qualifying.size == 0:
        return np.nan

    poleward_index = int(qualifying[0])
    if poleward_index == 0:
        return 0.0
    equatorward_index = poleward_index - 1
    equatorward_value = profile[equatorward_index]
    poleward_value = profile[poleward_index]
    fraction = (threshold - equatorward_value) / (
        poleward_value - equatorward_value
    )
    margin_magnitude = magnitudes[equatorward_index] + fraction * (
        magnitudes[poleward_index] - magnitudes[equatorward_index]
    )
    return float(hemisphere_sign * margin_magnitude)


def atmospheric_diagnostics_for_file(path: str | Path) -> tuple[xr.Dataset, xr.Dataset]:
    """Return scalar diagnostics and zonal temperatures from one PLASIM file."""
    with xr.open_dataset(path) as dataset:
        latitude = np.asarray(dataset["lat"].values, dtype=float)
        gaussian_nodes, gaussian_weights = np.polynomial.legendre.leggauss(latitude.size)
        expected_latitude = np.degrees(np.arcsin(gaussian_nodes))[::-1]
        if not np.allclose(latitude, expected_latitude):
            raise ValueError(
                f"{path} does not use the expected descending Gaussian latitude grid."
            )

        latitude_weights = xr.DataArray(
            gaussian_weights[::-1],
            dims=("lat",),
            coords={"lat": dataset["lat"]},
        )
        zonal_temperature = dataset["ts"].mean("lon")

        def latitude_band_mean(
            values: xr.DataArray, mask: np.ndarray
        ) -> xr.DataArray:
            """Return the Gaussian-area-weighted mean over a latitude mask."""
            return (
                values.isel(lat=mask)
                .weighted(latitude_weights.isel(lat=mask))
                .mean("lat")
            )

        def global_mean(name: str) -> xr.DataArray:
            """Return a global mean from one PLASIM gridded field."""
            zonal_values = dataset[name].mean("lon")
            return zonal_values.weighted(latitude_weights).mean("lat")

        tropical_temperature = latitude_band_mean(
            zonal_temperature, np.abs(latitude) <= 30.0
        )
        northern_polar_temperature = latitude_band_mean(
            zonal_temperature, latitude >= 30.0
        )
        southern_polar_temperature = latitude_band_mean(
            zonal_temperature, latitude <= -30.0
        )
        global_temperature = zonal_temperature.weighted(latitude_weights).mean("lat")
        zonal_2m_temperature = dataset["tas"].mean("lon")
        global_2m_temperature = zonal_2m_temperature.weighted(latitude_weights).mean(
            "lat"
        )
        tropical_2m_temperature = latitude_band_mean(
            zonal_2m_temperature, np.abs(latitude) <= 30.0
        )
        northern_polar_2m_temperature = latitude_band_mean(
            zonal_2m_temperature, latitude >= 30.0
        )
        southern_polar_2m_temperature = latitude_band_mean(
            zonal_2m_temperature, latitude <= -30.0
        )
        northern_hemisphere_surface_temperature = latitude_band_mean(
            zonal_temperature, latitude > 0.0
        )
        southern_hemisphere_surface_temperature = latitude_band_mean(
            zonal_temperature, latitude < 0.0
        )
        northern_hemisphere_2m_temperature = latitude_band_mean(
            zonal_2m_temperature, latitude > 0.0
        )
        southern_hemisphere_2m_temperature = latitude_band_mean(
            zonal_2m_temperature, latitude < 0.0
        )

        # PLASIM code 178 (rst) is net TOA shortwave, while codes 203
        # (rsut) and 179 (rlut) are negative for upward fluxes. Therefore
        # incident shortwave is rst - rsut and the net TOA balance is
        # rst + rlut. See PLASIM_INFO/PLASIM-1.0/scripts/burn.sh.
        net_shortwave = global_mean("rst")
        upward_shortwave = global_mean("rsut")
        net_longwave = global_mean("rlut")
        incoming_shortwave = net_shortwave - upward_shortwave
        reflected_shortwave = -upward_shortwave
        outgoing_longwave = -net_longwave
        energy_imbalance = net_shortwave + net_longwave
        planetary_albedo = reflected_shortwave / incoming_shortwave
        zonal_toa_energy_imbalance = (dataset["rst"] + dataset["rlut"]).mean(
            "lon"
        )

        land_sea_mask = np.asarray(
            dataset["lsm"].transpose("time", "lat", "lon").values
        )
        if not np.all(land_sea_mask == land_sea_mask[0]):
            raise ValueError(f"{path} does not have a time-invariant land-sea mask.")
        if not np.all((land_sea_mask == 0) | (land_sea_mask == 1)):
            raise ValueError(f"{path} does not have a binary land-sea mask.")
        ocean_mask = xr.DataArray(
            land_sea_mask[0] == 0,
            dims=("lat", "lon"),
            coords={"lat": dataset["lat"], "lon": dataset["lon"]},
        )
        ocean_zonal_mean_thickness = dataset["sit"].where(ocean_mask).mean("lon")
        zonal_persistent_sea_ice_fraction = (
            (dataset["sic"] >= SEA_ICE_COVER_THRESHOLD)
            .where(ocean_mask)
            .mean("lon")
        )

        def ice_edges(
            zonal_profile: xr.DataArray,
            hemisphere_sign: int,
            threshold: float,
        ) -> xr.DataArray:
            """Return one threshold-crossing latitude for every annual record."""
            return xr.DataArray(
                np.asarray(
                    [
                        _sea_ice_margin_latitude(
                            latitude,
                            profile,
                            hemisphere_sign,
                            threshold,
                        )
                        for profile in np.asarray(zonal_profile.values)
                    ]
                ),
                dims=("time",),
                coords={"time": dataset["time"]},
            )

        northern_persistent_ice_edge = ice_edges(
            zonal_persistent_sea_ice_fraction,
            1,
            SEA_ICE_ZONAL_OCCUPANCY_THRESHOLD,
        )
        southern_persistent_ice_edge = ice_edges(
            zonal_persistent_sea_ice_fraction,
            -1,
            SEA_ICE_ZONAL_OCCUPANCY_THRESHOLD,
        )

        diagnostics = xr.Dataset(
            {
                "global_temperature": global_temperature,
                "tropical_temperature": tropical_temperature,
                "northern_polar_temperature": northern_polar_temperature,
                "southern_polar_temperature": southern_polar_temperature,
                "northern_polar_temperature_gradient": (
                    tropical_temperature - northern_polar_temperature
                ),
                "southern_polar_temperature_gradient": (
                    tropical_temperature - southern_polar_temperature
                ),
                "global_2m_temperature": global_2m_temperature,
                "tropical_2m_temperature": tropical_2m_temperature,
                "northern_polar_2m_temperature": northern_polar_2m_temperature,
                "southern_polar_2m_temperature": southern_polar_2m_temperature,
                "northern_polar_2m_temperature_gradient": (
                    tropical_2m_temperature - northern_polar_2m_temperature
                ),
                "southern_polar_2m_temperature_gradient": (
                    tropical_2m_temperature - southern_polar_2m_temperature
                ),
                "northern_hemisphere_surface_temperature": (
                    northern_hemisphere_surface_temperature
                ),
                "southern_hemisphere_surface_temperature": (
                    southern_hemisphere_surface_temperature
                ),
                "northern_hemisphere_2m_temperature": (
                    northern_hemisphere_2m_temperature
                ),
                "southern_hemisphere_2m_temperature": (
                    southern_hemisphere_2m_temperature
                ),
                "global_toa_incoming_shortwave": incoming_shortwave,
                "global_toa_reflected_shortwave": reflected_shortwave,
                "global_toa_absorbed_shortwave": net_shortwave,
                "global_toa_outgoing_longwave": outgoing_longwave,
                "global_toa_energy_imbalance": energy_imbalance,
                "global_planetary_albedo": planetary_albedo,
                "northern_persistent_sea_ice_edge_latitude": (
                    northern_persistent_ice_edge
                ),
                "southern_persistent_sea_ice_edge_latitude": (
                    southern_persistent_ice_edge
                ),
            }
        ).assign_attrs(
            temperature_variable="ts",
            latitude_weighting="Gaussian quadrature weights",
            longitude_weighting="unweighted mean over longitude",
            tropical_band="30 degrees S to 30 degrees N",
            polar_bands="30 to 90 degrees in each hemisphere",
            temperature_gradient="tropical temperature minus polar temperature",
            toa_flux_convention=(
                "rst is net downward shortwave; rsut and rlut are negative "
                "for upward fluxes"
            ),
        )

        flux_metadata = {
            "global_toa_incoming_shortwave": (
                "global mean incoming top-of-atmosphere shortwave radiation",
                "rst - rsut; positive downward",
            ),
            "global_toa_reflected_shortwave": (
                "global mean reflected top-of-atmosphere shortwave radiation",
                "-rsut; positive outward",
            ),
            "global_toa_absorbed_shortwave": (
                "global mean absorbed top-of-atmosphere shortwave radiation",
                "rst; positive downward",
            ),
            "global_toa_outgoing_longwave": (
                "global mean outgoing top-of-atmosphere longwave radiation",
                "-rlut; positive outward",
            ),
            "global_toa_energy_imbalance": (
                "global mean net top-of-atmosphere energy imbalance",
                "rst + rlut; positive into the climate system",
            ),
        }
        for variable_name, (long_name, convention) in flux_metadata.items():
            diagnostics[variable_name].attrs = {
                "long_name": long_name,
                "units": "W m-2",
                "spatial_averaging": (
                    "unweighted longitude mean and Gaussian-area-weighted "
                    "latitude mean"
                ),
                "flux_sign_convention": convention,
            }
        diagnostics["global_planetary_albedo"].attrs = {
            "long_name": "global planetary albedo",
            "units": "1",
            "calculation": (
                "global_toa_reflected_shortwave divided by "
                "global_toa_incoming_shortwave"
            ),
        }
        for variable_name, long_name, source_variable in (
            (
                "northern_hemisphere_surface_temperature",
                "Northern Hemisphere mean surface temperature",
                "ts",
            ),
            (
                "southern_hemisphere_surface_temperature",
                "Southern Hemisphere mean surface temperature",
                "ts",
            ),
            (
                "northern_hemisphere_2m_temperature",
                "Northern Hemisphere mean temperature at 2 m",
                "tas",
            ),
            (
                "southern_hemisphere_2m_temperature",
                "Southern Hemisphere mean temperature at 2 m",
                "tas",
            ),
        ):
            diagnostics[variable_name].attrs = {
                "long_name": long_name,
                "units": "K",
                "source_variable": source_variable,
                "spatial_averaging": (
                    "unweighted longitude mean and Gaussian-area-weighted "
                    "latitude mean over the hemisphere"
                ),
            }
        persistent_ice_calculation = (
            "local annual sic is first classified at 0.5; the fraction of "
            "ocean longitudes classified as persistently ice-covered is then "
            "computed at each latitude, and its 0.5 equatorward crossing is "
            "linearly interpolated between Gaussian latitudes"
        )
        for variable_name, hemisphere in (
            ("northern_persistent_sea_ice_edge_latitude", "Northern Hemisphere"),
            ("southern_persistent_sea_ice_edge_latitude", "Southern Hemisphere"),
        ):
            diagnostics[variable_name].attrs = {
                "long_name": f"annual {hemisphere} persistent sea-ice edge latitude",
                "units": "degrees_north",
                "source_variables": "sic, lsm",
                "local_annual_sea_ice_cover_threshold": SEA_ICE_COVER_THRESHOLD,
                "ocean_longitude_occupancy_threshold": (
                    SEA_ICE_ZONAL_OCCUPANCY_THRESHOLD
                ),
                "calculation": persistent_ice_calculation,
            }

        # PLASIM stores dates as YYYYMMDD.fraction (for example,
        # 29100630.5). Keep that source coordinate and add an integer model
        # year that is suitable for sorting and plotting.
        encoded_time = np.asarray(dataset["time"].values, dtype=float)
        encoded_year = np.floor(encoded_time / 10_000.0).astype(int)
        model_year = _filename_years(path, encoded_year.size)
        if model_year is None:
            model_year = encoded_year
        diagnostics = diagnostics.assign_coords(year=("time", model_year))
        zonal_temperatures = xr.Dataset(
            {
                "zonal_surface_temperature": zonal_temperature,
                "zonal_2m_temperature": zonal_2m_temperature,
                "zonal_sea_ice_concentration": dataset["sic"]
                .where(ocean_mask)
                .mean("lon"),
                "zonal_persistent_sea_ice_fraction": (
                    zonal_persistent_sea_ice_fraction
                ),
                "zonal_sea_ice_thickness": ocean_zonal_mean_thickness,
                "zonal_surface_albedo": dataset["as"].mean("lon"),
                "zonal_toa_energy_imbalance": zonal_toa_energy_imbalance,
            }
        ).assign_attrs(
            longitude_averaging="unweighted arithmetic mean over longitude",
            source_variables=(
                "ts (surface temperature), tas (temperature at 2 m), "
                "sic (sea-ice concentration and local persistent-ice "
                "classification), sit (sea-ice thickness), "
                "as (surface albedo), rst and rlut (net TOA radiation), "
                "lsm (land-sea mask)"
            ),
        )
        zonal_temperatures["zonal_surface_temperature"].attrs = {
            "long_name": "zonal-mean surface temperature",
            "units": "K",
            "source_variable": "ts",
        }
        zonal_temperatures["zonal_2m_temperature"].attrs = {
            "long_name": "zonal-mean temperature at 2 m",
            "units": "K",
            "source_variable": "tas",
        }
        zonal_temperatures["zonal_sea_ice_concentration"].attrs = {
            "long_name": "ocean-only zonal-mean sea-ice concentration",
            "units": "1",
            "source_variable": "sic",
            "masking": "land excluded with lsm; ice-free ocean retained as zero",
        }
        zonal_temperatures["zonal_persistent_sea_ice_fraction"].attrs = {
            "long_name": (
                "fraction of ocean longitudes with annual sea-ice cover "
                "at least 0.5"
            ),
            "units": "1",
            "source_variables": "sic, lsm",
            "local_annual_sea_ice_cover_threshold": SEA_ICE_COVER_THRESHOLD,
            "calculation_order": (
                "threshold local sic first, then average the binary ocean mask "
                "over longitude"
            ),
        }
        zonal_temperatures["zonal_sea_ice_thickness"].attrs = {
            "long_name": "ocean-only zonal-mean sea-ice thickness",
            "units": "m",
            "source_variable": "sit",
            "masking": "land excluded with lsm; ice-free ocean retained as zero",
        }
        zonal_temperatures["zonal_surface_albedo"].attrs = {
            "long_name": "zonal-mean surface albedo",
            "units": "1",
            "source_variable": "as",
            "spatial_coverage": "land and ocean",
        }
        zonal_temperatures["zonal_toa_energy_imbalance"].attrs = {
            "long_name": "zonal-mean net top-of-atmosphere energy imbalance",
            "units": "W m-2",
            "source_variables": "rst, rlut",
            "calculation": "longitude mean of rst + rlut",
            "flux_sign_convention": "positive into the climate system",
        }
        zonal_temperatures = zonal_temperatures.assign_coords(
            year=("time", model_year)
        )

        # Materialize both results before the source NetCDF file is closed.
        return diagnostics.load(), zonal_temperatures.load()


def temperature_diagnostics_for_file(path: str | Path) -> xr.Dataset:
    """Return scalar temperature, TOA, and ice-margin diagnostics from one file."""
    diagnostics, _ = atmospheric_diagnostics_for_file(path)
    return diagnostics


def zonal_temperature_for_file(path: str | Path) -> xr.Dataset:
    """Return zonal-mean surface and 2-metre temperatures from one PLASIM file."""
    _, zonal_temperatures = atmospheric_diagnostics_for_file(path)
    return zonal_temperatures


def _lsg_ocean_cell_volumes(dataset: xr.Dataset, path: str | Path) -> np.ndarray:
    """Return static cell volumes for each configured region and depth band."""
    required = {"t", "wet", "depp", "lat", "lon", "depth", "depth_2"}
    missing = sorted(required - set(dataset.variables))
    if missing:
        raise ValueError(f"Missing required LSG variables in {path}: {missing}.")

    temperature = dataset["t"]
    expected_dimensions = ("time", "depth", "south_north", "west_east")
    if temperature.dims != expected_dimensions:
        raise ValueError(
            f"LSG potential temperature in {path} has dimensions {temperature.dims}; "
            f"expected {expected_dimensions}."
        )
    if temperature.attrs.get("units") != "K":
        raise ValueError(f"LSG potential temperature in {path} must have units 'K'.")

    latitude = np.asarray(dataset["lat"].values, dtype=float)
    longitude = np.asarray(dataset["lon"].values, dtype=float)
    horizontal_shape = temperature.shape[-2:]
    if latitude.shape != horizontal_shape or longitude.shape != horizontal_shape:
        raise ValueError(f"LSG latitude/longitude coordinates have invalid shapes in {path}.")

    depth = np.asarray(dataset["depth"].values, dtype=float)
    lower_interfaces = np.concatenate(([0.0], np.asarray(dataset["depth_2"].values[:-1])))
    upper_interfaces = np.asarray(dataset["depth_2"].values, dtype=float)
    if depth.ndim != 1 or upper_interfaces.shape != depth.shape:
        raise ValueError(f"LSG vertical coordinates are incompatible in {path}.")
    if not (
        np.all(np.diff(depth) > 0.0)
        and np.all(upper_interfaces > lower_interfaces)
        and np.all((depth >= lower_interfaces) & (depth <= upper_interfaces))
    ):
        raise ValueError(f"LSG vertical coordinates are not strictly ordered in {path}.")

    wet_variable = dataset["wet"].transpose(*expected_dimensions)
    wet = np.asarray(wet_variable.isel(time=0).values, dtype=float)
    if not np.all((wet == 0.0) | (wet == 1.0)):
        raise ValueError(f"LSG wet mask is not binary in {path}.")
    if wet_variable.sizes["time"] > 1 and not np.array_equal(
        wet, np.asarray(wet_variable.isel(time=-1).values)
    ):
        raise ValueError(f"LSG wet mask changes with time in {path}.")
    if np.any(wet[:, np.abs(latitude) > 90.0] != 0.0):
        raise ValueError(f"LSG wet cells have latitudes outside [-90, 90] in {path}.")

    bathymetry_variable = dataset["depp"]
    if "lev" in bathymetry_variable.dims:
        bathymetry_variable = bathymetry_variable.isel(lev=0)
    bathymetry_variable = bathymetry_variable.transpose(
        "time", "south_north", "west_east"
    )
    bathymetry = np.asarray(bathymetry_variable.isel(time=0).values, dtype=float)
    if bathymetry_variable.sizes["time"] > 1 and not np.allclose(
        bathymetry,
        np.asarray(bathymetry_variable.isel(time=-1).values),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(f"LSG bathymetry changes with time in {path}.")

    grid_step_radians = np.deg2rad(LSG_HORIZONTAL_GRID_SPACING_DEGREES)
    meridional_grid_size = LSG_EARTH_RADIUS_M * grid_step_radians
    horizontal_area = (
        0.5
        * meridional_grid_size**2
        * np.cos(np.deg2rad(latitude))
    )
    horizontal_area = np.where(np.abs(latitude) <= 90.0, horizontal_area, 0.0)

    cell_volumes = np.empty(
        (
            len(LSG_OCEAN_REGIONS),
            len(LSG_OCEAN_DEPTH_BANDS_M),
            depth.size,
            *horizontal_shape,
        ),
        dtype=float,
    )
    for region_index, (_, latitude_min, latitude_max) in enumerate(
        LSG_OCEAN_REGIONS
    ):
        region_mask = (latitude >= latitude_min) & (latitude <= latitude_max)
        for band_index, (_, depth_min, depth_max) in enumerate(
            LSG_OCEAN_DEPTH_BANDS_M
        ):
            layer_thickness = np.maximum(
                0.0,
                np.minimum(
                    np.minimum(bathymetry[None, :, :], upper_interfaces[:, None, None]),
                    depth_max,
                )
                - np.maximum(lower_interfaces[:, None, None], depth_min),
            )
            volumes = (
                wet
                * layer_thickness
                * horizontal_area[None, :, :]
                * region_mask[None, :, :]
            )
            if not np.any(volumes > 0.0):
                raise ValueError(
                    f"LSG region {LSG_OCEAN_REGIONS[region_index][0]!r} and depth "
                    f"band {LSG_OCEAN_DEPTH_BANDS_M[band_index][0]!r} contain no "
                    f"wet volume in {path}."
                )
            cell_volumes[region_index, band_index] = volumes
    return cell_volumes


def lsg_ocean_diagnostics_for_file(path: str | Path) -> xr.Dataset:
    """Return regional, depth-resolved ocean heat diagnostics from one LSG file."""
    with xr.open_dataset(path) as dataset:
        cell_volumes = _lsg_ocean_cell_volumes(dataset, path)
        temperature = np.asarray(
            dataset["t"]
            .transpose("time", "depth", "south_north", "west_east")
            .values,
            dtype=float,
        )
        wet = np.asarray(dataset["wet"].isel(time=0).values, dtype=bool)
        if not np.isfinite(temperature[:, wet]).all():
            raise ValueError(f"LSG wet-cell potential temperature is non-finite in {path}.")
        years = _filename_years(path, temperature.shape[0])
        if years is None:
            raise ValueError(f"Cannot recover model years from LSG filename {path}.")

    ocean_volume = cell_volumes.sum(axis=(-3, -2, -1))
    temperature_integral = np.einsum(
        "tzyx,rbzyx->trb", temperature, cell_volumes, optimize=True
    )
    mean_temperature = temperature_integral / ocean_volume[None, :, :]
    heat_content = LSG_REFERENCE_DENSITY_KG_M3 * LSG_SPECIFIC_HEAT_J_KG_K * (
        temperature_integral
        - LSG_REFERENCE_TEMPERATURE_K * ocean_volume[None, :, :]
    )

    region_names = [region[0] for region in LSG_OCEAN_REGIONS]
    depth_band_names = [band[0] for band in LSG_OCEAN_DEPTH_BANDS_M]
    result = xr.Dataset(
        {
            "ocean_heat_content": (
                ("year", "ocean_region", "depth_band"),
                heat_content,
                {
                    "long_name": "regional ocean heat content relative to 273.16 K",
                    "units": "J",
                    "reference_temperature_K": LSG_REFERENCE_TEMPERATURE_K,
                },
            ),
            "ocean_mean_potential_temperature": (
                ("year", "ocean_region", "depth_band"),
                mean_temperature,
                {
                    "long_name": "volume-mean ocean potential temperature",
                    "units": "K",
                },
            ),
            "ocean_volume": (
                ("ocean_region", "depth_band"),
                ocean_volume,
                {"long_name": "static ocean volume used for weighting", "units": "m3"},
            ),
        },
        coords={
            "year": years,
            "ocean_region": region_names,
            "depth_band": depth_band_names,
            "region_latitude_min": (
                "ocean_region", [region[1] for region in LSG_OCEAN_REGIONS]
            ),
            "region_latitude_max": (
                "ocean_region", [region[2] for region in LSG_OCEAN_REGIONS]
            ),
            "depth_min": (
                "depth_band", [band[1] for band in LSG_OCEAN_DEPTH_BANDS_M]
            ),
            "depth_max": (
                "depth_band", [band[2] for band in LSG_OCEAN_DEPTH_BANDS_M]
            ),
        },
        attrs={
            "source_variable": "LSG potential temperature t",
            "horizontal_weighting": (
                "native LSG E-grid metric 0.5 * (R * 5 degrees)^2 * cos(latitude)"
            ),
            "vertical_weighting": (
                "partial bottom cells reconstructed from depp; fractional overlap "
                "at diagnostic depth boundaries"
            ),
            "reference_density_kg_m3": LSG_REFERENCE_DENSITY_KG_M3,
            "specific_heat_J_kg_K": LSG_SPECIFIC_HEAT_J_KG_K,
        },
    )
    return result


def build_lsg_ocean_diagnostics(
    lsg_paths: Sequence[str | Path], *, workers: int = 4
) -> xr.Dataset:
    """Build one aligned ocean-diagnostic dataset from annual LSG files."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if not lsg_paths:
        raise ValueError("At least one LSG NetCDF file is required.")

    worker_count = min(workers, len(lsg_paths))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(lsg_ocean_diagnostics_for_file, lsg_paths))

    reference_volume = results[0]["ocean_volume"]
    for result in results[1:]:
        if not np.allclose(
            result["ocean_volume"], reference_volume, rtol=1.0e-12, atol=0.0
        ):
            raise ValueError("LSG ocean grid or wet volume changes between files.")

    dynamic_results = [result.drop_vars("ocean_volume") for result in results]
    combined = xr.concat(
        dynamic_results,
        dim="year",
        data_vars="minimal",
        coords="minimal",
        compat="equals",
        combine_attrs="identical",
    ).sortby("year")
    years = np.asarray(combined["year"].values, dtype=int)
    if np.unique(years).size != years.size:
        raise ValueError("LSG ocean diagnostic years overlap between files.")
    if years.size > 1 and not np.array_equal(np.diff(years), np.ones(years.size - 1)):
        raise ValueError("LSG ocean diagnostic years are not contiguous.")
    combined["ocean_volume"] = reference_volume
    combined.attrs.update(
        lsg_file_count=len(lsg_paths),
        worker_count=worker_count,
    )
    return combined


def build_plasim_diagnostic_datasets(
    atmospheric_paths: Sequence[str | Path],
    diagnostic_paths: Sequence[str | Path],
    *,
    workers: int = 4,
) -> PlasimDiagnosticDatasets:
    """Build aligned scalar and zonal datasets from PLASIM and LSG output files."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if not atmospheric_paths:
        raise ValueError("At least one PLASIM atmospheric file is required.")
    if not diagnostic_paths:
        raise ValueError("At least one LSG diagnostic file is required.")

    worker_count = min(workers, max(len(atmospheric_paths), len(diagnostic_paths)))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        atmospheric_results = list(
            executor.map(atmospheric_diagnostics_for_file, atmospheric_paths)
        )
        lsg_file_diagnostics = list(
            executor.map(lsg_diagnostics_for_file, diagnostic_paths)
        )

    scalar_file_diagnostics, zonal_file_temperatures = zip(*atmospheric_results)
    diagnostics = xr.concat(scalar_file_diagnostics, dim="time").sortby("year")
    zonal_temperatures = xr.concat(zonal_file_temperatures, dim="time").sortby(
        "year"
    )
    lsg_diagnostics = xr.concat(lsg_file_diagnostics, dim="year").sortby("year")
    temperature_years = diagnostics["year"].values
    missing_lsg_years = sorted(
        set(temperature_years.tolist()) - set(lsg_diagnostics["year"].values.tolist())
    )
    if missing_lsg_years:
        raise ValueError(f"Missing LSG diagnostics for model years {missing_lsg_years}.")

    aligned_lsg = lsg_diagnostics.reindex(year=temperature_years)
    for variable_name, variable in aligned_lsg.data_vars.items():
        diagnostics[variable_name] = xr.DataArray(
            variable.values,
            dims=("time",),
            coords={"time": diagnostics["time"]},
            attrs=variable.attrs,
        )
    diagnostics.attrs.update(lsg_diagnostics.attrs)
    return PlasimDiagnosticDatasets(
        diagnostics=diagnostics,
        zonal_temperatures=zonal_temperatures,
        atmospheric_file_count=len(atmospheric_paths),
        diagnostic_file_count=len(diagnostic_paths),
        worker_count=worker_count,
    )
