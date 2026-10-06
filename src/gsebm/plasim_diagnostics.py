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
    r"_(?:PLA|DIAG|LSG|ICE)\.(\d+)-(\d+)\.(?:nc|txt)$"
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
_MOC_LINE_PATTERNS = {
    "atl_min_aabw": re.compile(
        rf"ATL min \(AABW\)\s*:\s*({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})"
    ),
    "pac_max_outflow": re.compile(
        rf"PAC max \(outflow\)\s*:\s*({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})"
    ),
    "pac_min_inflow": re.compile(
        rf"PAC min \(inflow\)\s*:\s*({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})\s+({_FLOAT_PATTERN})"
    ),
}
_BAROTROPIC_MAX_PATTERN = re.compile(
    rf"Maximum of barotropic streamfunction in Sv\s+({_FLOAT_PATTERN})\s+"
    rf"at lon=\s*({_FLOAT_PATTERN})\s*lat=\s*({_FLOAT_PATTERN})"
)
_UPWELLING_PATTERN = re.compile(
    rf"Upw\.transports in Sv\s+" + r"\s+".join([rf"({_FLOAT_PATTERN})"] * 6)
)
_CONVECTIVE_EVENTS_PATTERN = re.compile(
    r"Conv\.adjustm\. events:\s+" + r"\s+".join([r"(\d+)"] * 6)
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
LSG_LAYER_MAP_BANDS_M = (
    ("0_100m", 0.0, 100.0),
    ("150_300m", 150.0, 300.0),
    ("300_600m", 300.0, 600.0),
)
LSG_LAYER_MAP_VELOCITY_BANDS_M = LSG_LAYER_MAP_BANDS_M[:2]
PLASIM_EARTH_RADIUS_M = 6_371_000.0
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


@dataclass(frozen=True)
class PlasimColdBranchCheapDatasets:
    """Cheap annual diagnostics needed before full ocean extraction."""

    diagnostics: xr.Dataset
    zonal: xr.Dataset
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


def find_mechanism_field_files(
    experiment_dir: str | Path, state: str
) -> tuple[list[Path], list[Path]]:
    """Return matched LSG-ocean and PlaSim-atmosphere mechanism-field files."""
    output_dir = Path(experiment_dir) / "output" / state
    if not output_dir.is_dir():
        raise FileNotFoundError(f"PLASIM output directory does not exist: {output_dir}.")

    def sorted_matches(pattern: re.Pattern[str]) -> list[Path]:
        return sorted(
            (
                path
                for path in output_dir.iterdir()
                if path.is_file()
                and not path.name.startswith("._")
                and pattern.fullmatch(path.name) is not None
            ),
            key=lambda path: int(
                _OUTPUT_YEAR_RANGE_PATTERN.search(path.name).group(1)  # type: ignore[union-attr]
            ),
        )

    lsg_paths = sorted_matches(_LSG_OUTPUT_PATTERN)
    atmospheric_paths = sorted_matches(_ATMOSPHERIC_OUTPUT_PATTERN)
    if not lsg_paths:
        raise FileNotFoundError(f"No LSG NetCDF files found in {output_dir}.")
    if not atmospheric_paths:
        raise FileNotFoundError(f"No PLASIM atmospheric NetCDF files found in {output_dir}.")
    return lsg_paths, atmospheric_paths


def find_ice_files(experiment_dir: str | Path, state: str) -> list[Path]:
    """Return annual-mean ICE NetCDF files for one experiment state."""
    output_dir = Path(experiment_dir) / "output" / state
    if not output_dir.is_dir():
        raise FileNotFoundError(f"PLASIM output directory does not exist: {output_dir}.")
    paths = sorted(
        (
            path for path in output_dir.iterdir()
            if path.is_file() and not path.name.startswith("._")
            and re.fullmatch(r".+_ICE\.\d+-\d+\.nc", path.name)
        ),
        key=lambda path: int(_OUTPUT_YEAR_RANGE_PATTERN.search(path.name).group(1)),  # type: ignore[union-attr]
    )
    if not paths:
        raise FileNotFoundError(f"No ICE NetCDF files found in {output_dir}.")
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


def extended_lsg_diagnostics_for_file(path: str | Path) -> xr.Dataset:
    """Return every annual scalar diagnostic printed by the available LSG text."""
    result = lsg_diagnostics_for_file(path)
    records: dict[str, dict[int, list[tuple[float, ...]]]] = {
        name: {} for name in (*_MOC_LINE_PATTERNS, "barotropic_max", "upwelling", "convective_events")
    }
    current_year: int | None = None
    with Path(path).open(encoding="utf-8", errors="replace") as diagnostic_file:
        for line in diagnostic_file:
            date_match = _LSG_DATE_PATTERN.search(line)
            if date_match is not None:
                current_year = _year_from_filename(path, int(date_match.group(1)))
                continue
            if current_year is None:
                continue
            patterns = {
                **_MOC_LINE_PATTERNS,
                "barotropic_max": _BAROTROPIC_MAX_PATTERN,
                "upwelling": _UPWELLING_PATTERN,
                "convective_events": _CONVECTIVE_EVENTS_PATTERN,
            }
            for name, pattern in patterns.items():
                match = pattern.search(line)
                if match is not None:
                    records[name].setdefault(current_year, []).append(_float_groups(match))

    years = np.asarray(result["year"].values, dtype=int)

    def annual(name: str, width: int) -> np.ndarray:
        values = records[name]
        missing = [int(year) for year in years if int(year) not in values]
        if missing:
            raise ValueError(f"Missing {name} diagnostics for model years {missing} in {path}.")
        means = np.asarray([np.mean(values[int(year)], axis=0) for year in years])
        if means.shape != (years.size, width):
            raise ValueError(f"Unexpected {name} diagnostic width in {path}.")
        return means

    moc_names = {
        "atl_min_aabw": "atl_aabw_min",
        "pac_max_outflow": "pac_outflow_max",
        "pac_min_inflow": "pac_inflow_min",
    }
    for record_name, variable_prefix in moc_names.items():
        values = annual(record_name, 3)
        for index, suffix in enumerate(("46_66n", "16_44n", "30s")):
            result[f"{variable_prefix}_{suffix}"] = (
                "year",
                values[:, index],
                {"units": "Sv", "source": f"PLASIM DIAG {record_name}"},
            )

    barotropic = annual("barotropic_max", 3)
    for index, (name, units) in enumerate(
        (("barotropic_streamfunction_max", "Sv"), ("barotropic_max_longitude", "degrees_east"), ("barotropic_max_latitude", "degrees_north"))
    ):
        result[name] = ("year", barotropic[:, index], {"units": units})

    upwelling = annual("upwelling", 6)
    convection = annual("convective_events", 6)
    result["upwelling_transport"] = (
        ("year", "diagnostic_layer"),
        upwelling,
        {"units": "Sv", "source": "PLASIM DIAG Upw.transports in Sv"},
    )
    result["convective_adjustment_event_count"] = (
        ("year", "diagnostic_layer"),
        convection,
        {"units": "1", "source": "PLASIM DIAG Conv.adjustm. events"},
    )
    result = result.assign_coords(
        diagnostic_layer=np.arange(6),
        diagnostic_layer_depth=(
            "diagnostic_layer",
            np.asarray([75.0, 275.0, 650.0, 1100.0, 2250.0, 5500.0]),
        ),
    )
    result.attrs.update(
        unavailable_printed_diagnostics=(
            "The source DIAG format has no barotropic minimum or separate downwelling line; "
            "those quantities must be derived from gridded LSG output."
        )
    )
    return result


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


def _transport_from_zonal_source(
    source: np.ndarray, latitude: np.ndarray, gaussian_weights: np.ndarray
) -> np.ndarray:
    """Return northward transport from a zonal source with its global mean removed."""
    values = np.asarray(source, dtype=float)
    weights = np.asarray(gaussian_weights, dtype=float)
    anomaly = values - np.sum(values * weights[None, :], axis=1)[:, None] / np.sum(weights)
    order = np.argsort(latitude)
    ordered = anomaly[:, order]
    ordered_weights = weights[order]
    # Place transport at latitude-cell centres using half of the local cell.
    cumulative = np.cumsum(ordered * ordered_weights[None, :], axis=1)
    cumulative -= 0.5 * ordered * ordered_weights[None, :]
    transport_ordered = -2.0 * np.pi * PLASIM_EARTH_RADIUS_M**2 * cumulative
    inverse = np.argsort(order)
    return transport_ordered[:, inverse]


def cold_branch_atmospheric_diagnostics_for_file(
    path: str | Path,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Extract inexpensive ice-edge, snow, flux, and transport diagnostics."""
    with xr.open_dataset(path) as dataset:
        required = {
            "ts", "sic", "sit", "lsm", "snd", "as", "rss", "rls",
            "hfss", "hfls", "rst", "rlut",
        }
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ValueError(f"Missing cold-branch PLA variables in {path}: {missing}.")
        latitude = np.asarray(dataset["lat"].values, dtype=float)
        _, weights_ascending = np.polynomial.legendre.leggauss(latitude.size)
        weights = weights_ascending[::-1]
        latitude_weights = xr.DataArray(weights, dims="lat", coords={"lat": dataset["lat"]})
        land = dataset["lsm"].isel(time=0) > 0.5
        ocean = ~land
        if not np.array_equal(land.values, (dataset["lsm"].isel(time=-1) > 0.5).values):
            raise ValueError(f"{path} does not have a time-invariant land mask.")

        zonal_sic = dataset["sic"].where(ocean).mean("lon")
        zonal_sit = dataset["sit"].where(ocean).mean("lon")
        zonal_ts = dataset["ts"].mean("lon")
        zonal_surface_net = (
            dataset["rss"] + dataset["rls"] + dataset["hfss"] + dataset["hfls"]
        ).mean("lon")
        zonal_toa_net = (dataset["rst"] + dataset["rlut"]).mean("lon")
        zonal_atmospheric_source = zonal_toa_net - zonal_surface_net
        aht = _transport_from_zonal_source(
            zonal_atmospheric_source.values, latitude, weights
        )
        implied_oht = _transport_from_zonal_source(
            zonal_surface_net.values, latitude, weights
        )

        def edges(sign: int) -> np.ndarray:
            return np.asarray(
                [
                    _sea_ice_margin_latitude(latitude, profile, sign, 0.5)
                    for profile in zonal_sic.values
                ]
            )

        cell_area = xr.DataArray(
            2.0 * np.pi * PLASIM_EARTH_RADIUS_M**2 * weights / dataset.sizes["lon"],
            dims="lat",
            coords={"lat": dataset["lat"]},
        )
        snow_covered = (dataset["snd"] > 0.0) & land
        snow_area = snow_covered.astype(float) * cell_area
        nh_snow_area = snow_area.where(dataset["lat"] > 0.0).sum(("lat", "lon"))
        sh_snow_area = snow_area.where(dataset["lat"] < 0.0).sum(("lat", "lon"))
        land_count = land.sum("lon")
        zonal_land_snow_cover = snow_covered.sum("lon") / land_count.where(land_count > 0)
        zonal_land_snow_depth = dataset["snd"].where(land).mean("lon")
        zonal_land_albedo = dataset["as"].where(land).mean("lon")

        encoded_time = np.asarray(dataset["time"].values, dtype=float)
        years = _filename_years(path, encoded_time.size)
        if years is None:
            years = np.floor(encoded_time / 10_000.0).astype(int)
        scalar = xr.Dataset(
            {
                "northern_sic50_sea_ice_edge_latitude": ("time", edges(1)),
                "southern_sic50_sea_ice_edge_latitude": ("time", edges(-1)),
                "northern_land_snow_covered_area": nh_snow_area,
                "southern_land_snow_covered_area": sh_snow_area,
                "cross_equatorial_atmospheric_heat_transport": (
                    "time", np.asarray([np.interp(0.0, latitude[::-1], row[::-1]) for row in aht])
                ),
                "cross_equatorial_implied_ocean_heat_transport": (
                    "time", np.asarray([np.interp(0.0, latitude[::-1], row[::-1]) for row in implied_oht])
                ),
            },
            coords={"time": dataset["time"], "year": ("time", years)},
        )
        zonal = xr.Dataset(
            {
                "zonal_surface_temperature": zonal_ts,
                "zonal_sea_ice_concentration": zonal_sic,
                "zonal_sea_ice_thickness": zonal_sit,
                "zonal_land_snow_cover_fraction": zonal_land_snow_cover,
                "zonal_land_snow_depth": zonal_land_snow_depth,
                "zonal_land_surface_albedo": zonal_land_albedo,
                "zonal_surface_net_heat_flux": zonal_surface_net,
                "zonal_toa_net_radiation": zonal_toa_net,
                "zonal_atmospheric_heat_transport": (("time", "lat"), aht),
                "zonal_implied_ocean_heat_transport_preliminary": (
                    ("time", "lat"), implied_oht
                ),
            },
            coords={"time": dataset["time"], "lat": dataset["lat"], "year": ("time", years)},
        )

        for name in ("northern_sic50_sea_ice_edge_latitude", "southern_sic50_sea_ice_edge_latitude"):
            scalar[name].attrs.update(
                units="degrees_north",
                threshold=0.5,
                calculation="linear interpolation of the ocean-only zonal-mean sic crossing",
            )
        for name in ("northern_land_snow_covered_area", "southern_land_snow_covered_area"):
            scalar[name].attrs.update(
                units="m2",
                cover_definition="land grid cells with annual-mean snd > 0 m",
            )
        for name in ("cross_equatorial_atmospheric_heat_transport", "cross_equatorial_implied_ocean_heat_transport"):
            scalar[name].attrs.update(units="W", sign_convention="positive northward")
        flux_attrs = {
            "units": "W m-2",
            "sign_convention": "positive downward into the surface/climate system",
        }
        zonal["zonal_surface_net_heat_flux"].attrs.update(
            **flux_attrs, calculation="rss + rls + hfss + hfls"
        )
        zonal["zonal_toa_net_radiation"].attrs.update(
            **flux_attrs, calculation="rst + rlut"
        )
        zonal["zonal_atmospheric_heat_transport"].attrs.update(
            units="W", sign_convention="positive northward",
            source="TOA net minus surface net, global mean removed before integration",
        )
        zonal["zonal_implied_ocean_heat_transport_preliminary"].attrs.update(
            units="W", sign_convention="positive northward",
            caveat="ice latent heat tendency is not yet removed; final diagnostic is Plan 2.4",
        )
        return scalar.load(), zonal.load()


def ice_cap_diagnostics_for_file(path: str | Path) -> xr.Dataset:
    """Return annual 9 m sea-ice cap coverage and correction-flux diagnostics."""
    with xr.open_dataset(path) as dataset:
        required = {"iced", "cfluxra", "ls", "lat", "lon"}
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ValueError(f"Missing ICE cap variables in {path}: {missing}.")
        latitude = np.asarray(dataset["lat"].values, dtype=float)
        _, weights_ascending = np.polynomial.legendre.leggauss(latitude.size)
        weights = xr.DataArray(
            weights_ascending[::-1], dims="lat", coords={"lat": dataset["lat"]}
        )
        ocean = dataset["ls"].isel(time=0) < 0.5
        if not np.array_equal(ocean.values, (dataset["ls"].isel(time=-1) < 0.5).values):
            raise ValueError(f"{path} does not have a time-invariant ICE land mask.")
        capped = (dataset["iced"] >= 8.9) & ocean

        def ocean_area_fraction(hemisphere: np.ndarray) -> xr.DataArray:
            numerator = capped.where(hemisphere).astype(float).weighted(weights).sum(("lat", "lon"))
            denominator = ocean.where(hemisphere).astype(float).weighted(weights).sum(("lat", "lon"))
            return numerator / denominator

        global_flux = dataset["cfluxra"].mean("lon").weighted(weights).mean("lat")
        years = _filename_years(path, dataset.sizes["time"])
        if years is None:
            raise ValueError(f"Cannot recover model years from ICE filename {path}.")
        result = xr.Dataset(
            {
                "northern_ice_cap_ocean_area_fraction": ocean_area_fraction(dataset["lat"] > 0),
                "southern_ice_cap_ocean_area_fraction": ocean_area_fraction(dataset["lat"] < 0),
                "global_ice_cap_correction_flux": global_flux,
            },
            coords={"time": dataset["time"], "year": ("time", years)},
            attrs={"ice_thickness_cap_threshold_m": 8.9, "source_variables": "ICE iced, cfluxra, ls"},
        )
        for name in ("northern_ice_cap_ocean_area_fraction", "southern_ice_cap_ocean_area_fraction"):
            result[name].attrs.update(
                units="1", weighting="Gaussian grid-cell area over ocean cells in the hemisphere"
            )
        result["global_ice_cap_correction_flux"].attrs.update(
            units="W m-2",
            source="ICE cfluxra code 713",
            sign_convention="native model sign; positive values retained as written",
            weighting="global Gaussian grid-cell area including zero land values",
        )
        return result.load()


def build_ice_cap_diagnostics(
    ice_paths: Sequence[str | Path], *, workers: int = 4
) -> xr.Dataset:
    """Build a contiguous annual ICE cap diagnostic dataset."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if not ice_paths:
        raise ValueError("At least one ICE file is required.")
    with ProcessPoolExecutor(max_workers=min(workers, len(ice_paths))) as executor:
        results = list(executor.map(ice_cap_diagnostics_for_file, ice_paths))
    combined = xr.concat(results, dim="time").sortby("year")
    years = np.asarray(combined["year"].values, dtype=int)
    if years.size > 1 and not np.array_equal(np.diff(years), np.ones(years.size - 1)):
        raise ValueError("ICE cap diagnostic years are not contiguous.")
    return combined


def _lsg_ocean_geometry(
    dataset: xr.Dataset, path: str | Path
) -> tuple[np.ndarray, ...]:
    """Return validated static LSG scalar-grid geometry arrays."""
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

    return (
        latitude,
        longitude,
        depth,
        lower_interfaces,
        upper_interfaces,
        bathymetry,
        wet,
        horizontal_area,
    )


def _lsg_wet_cell_volumes(dataset: xr.Dataset, path: str | Path) -> np.ndarray:
    """Return full per-cell wet volume on the native LSG scalar grid."""
    (
        _,
        _,
        _,
        lower_interfaces,
        upper_interfaces,
        bathymetry,
        wet,
        horizontal_area,
    ) = _lsg_ocean_geometry(dataset, path)
    layer_thickness = np.maximum(
        0.0,
        np.minimum(bathymetry[None, :, :], upper_interfaces[:, None, None])
        - lower_interfaces[:, None, None],
    )
    return wet * layer_thickness * horizontal_area[None, :, :]


def _lsg_ocean_cell_volumes(dataset: xr.Dataset, path: str | Path) -> np.ndarray:
    """Return static cell volumes for each configured region and depth band."""
    (
        latitude,
        _,
        depth,
        lower_interfaces,
        upper_interfaces,
        bathymetry,
        wet,
        horizontal_area,
    ) = _lsg_ocean_geometry(dataset, path)
    horizontal_shape = latitude.shape
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


def lsg_total_heat_content_for_file(path: str | Path) -> xr.Dataset:
    """Return full-depth global ocean heat content from one annual LSG file."""
    with xr.open_dataset(path) as dataset:
        required = {"t", "wet", "depp", "lat", "depth", "depth_2"}
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ValueError(f"Missing full-depth LSG variables in {path}: {missing}.")
        temperature = np.asarray(
            dataset["t"].transpose("time", "depth", "south_north", "west_east"),
            dtype=float,
        )
        wet = np.asarray(dataset["wet"].isel(time=0), dtype=float)
        latitude = np.asarray(dataset["lat"], dtype=float)
        depth = np.asarray(dataset["depth"], dtype=float)
        upper = np.asarray(dataset["depth_2"], dtype=float)
        lower = np.concatenate(([0.0], upper[:-1]))
        if depth.shape != upper.shape:
            raise ValueError(f"Incompatible vertical coordinates in {path}.")
        bathymetry_variable = dataset["depp"]
        if "lev" in bathymetry_variable.dims:
            bathymetry_variable = bathymetry_variable.isel(lev=0)
        bathymetry = np.asarray(bathymetry_variable.isel(time=0), dtype=float)
        grid_step = np.deg2rad(LSG_HORIZONTAL_GRID_SPACING_DEGREES)
        area = 0.5 * (LSG_EARTH_RADIUS_M * grid_step) ** 2 * np.cos(np.deg2rad(latitude))
        area = np.where(np.abs(latitude) <= 90.0, area, 0.0)
        thickness = np.maximum(
            0.0,
            np.minimum(bathymetry[None, :, :], upper[:, None, None]) - lower[:, None, None],
        )
        volume = wet * thickness * area[None, :, :]
        if not np.isfinite(temperature[:, volume > 0]).all():
            raise ValueError(f"Non-finite wet-cell LSG temperature in {path}.")
        integral = np.einsum("tzyx,zyx->t", temperature, volume, optimize=True)
        total_volume = float(np.sum(volume))
        heat_content = LSG_REFERENCE_DENSITY_KG_M3 * LSG_SPECIFIC_HEAT_J_KG_K * (
            integral - LSG_REFERENCE_TEMPERATURE_K * total_volume
        )
        years = _filename_years(path, temperature.shape[0])
        if years is None:
            raise ValueError(f"Cannot recover model years from LSG filename {path}.")
        return xr.Dataset(
            {
                "full_depth_ocean_heat_content": (
                    "year", heat_content,
                    {"units": "J", "reference_temperature_K": LSG_REFERENCE_TEMPERATURE_K},
                ),
                "full_depth_ocean_volume": (
                    (), total_volume, {"units": "m3"}
                ),
            },
            coords={"year": years},
            attrs={
                "source_variable": "LSG potential temperature t",
                "horizontal_weighting": "native LSG E-grid metric",
                "vertical_weighting": "partial bottom cells from depp",
            },
        )


def build_lsg_total_heat_content(
    lsg_paths: Sequence[str | Path], *, workers: int = 4
) -> xr.Dataset:
    """Build contiguous full-depth global ocean heat content."""
    if workers < 1 or not lsg_paths:
        raise ValueError("workers must be positive and at least one LSG path is required")
    with ProcessPoolExecutor(max_workers=min(workers, len(lsg_paths))) as executor:
        results = list(executor.map(lsg_total_heat_content_for_file, lsg_paths))
    reference_volume = results[0]["full_depth_ocean_volume"]
    if any(not np.isclose(result["full_depth_ocean_volume"], reference_volume) for result in results[1:]):
        raise ValueError("Full-depth LSG ocean volume changes between files.")
    combined = xr.concat(
        [result.drop_vars("full_depth_ocean_volume") for result in results], dim="year"
    ).sortby("year")
    combined["full_depth_ocean_volume"] = reference_volume
    years = np.asarray(combined["year"], dtype=int)
    if years.size > 1 and not np.array_equal(np.diff(years), np.ones(years.size - 1)):
        raise ValueError("Full-depth heat-content years are not contiguous.")
    return combined


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


_MECHANISM_OPTIONAL_LSG_VARIABLES = {
    "s": "zonal_salinity",
    "w": "zonal_vertical_velocity",
    "convad": "zonal_convective_adjustment",
    "fluxhea": "zonal_ocean_heat_flux",
    "flukhea": "zonal_newtonian_coupling_heat_flux",
    "fluwat": "zonal_fresh_water_flux",
    "flukwat": "zonal_newtonian_coupling_fresh_water_flux",
    "fldsst": "zonal_sst_mismatch",
    "fldice": "zonal_ice_mismatch",
    "zeta": "zonal_sea_surface_height",
    "tbound": "zonal_boundary_temperature",
    "sice": "zonal_lsg_ice_thickness",
    "taux": "zonal_zonal_wind_stress",
}
_MECHANISM_OPTIONAL_PLA_VARIABLES = {
    "prsn": "zonal_snowfall",
    "evap": "zonal_evaporation",
}


def _lsg_latitude_rows(
    latitude: np.ndarray, wet: np.ndarray, path: str | Path
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return sorted wet scalar-point latitude rows and their masks."""
    rounded = np.round(np.asarray(latitude, dtype=float), decimals=6)
    for row in rounded:
        if np.unique(row).size != 1:
            raise ValueError(f"LSG horizontal row contains multiple latitudes in {path}.")
    wet_horizontal = np.any(wet > 0.0, axis=0)
    rows = np.unique(rounded[wet_horizontal])
    masks = [(rounded == row) & wet_horizontal for row in rows]
    if rows.size == 0 or any(not np.any(mask) for mask in masks):
        raise ValueError(f"LSG grid contains no wet latitude rows in {path}.")
    if not np.all(np.diff(rows) > 0.0):
        raise ValueError(f"LSG wet latitude rows are not strictly monotone in {path}.")
    return rows.astype(float), masks


def _row_weighted_mean(
    values: np.ndarray,
    weights: np.ndarray,
    row_masks: Sequence[np.ndarray],
) -> np.ndarray:
    """Reduce the final two horizontal axes with row-specific weights."""
    output_shape = values.shape[:-2] + (len(row_masks),)
    output = np.full(output_shape, np.nan, dtype=float)
    for row_index, mask in enumerate(row_masks):
        selected_weights = weights[..., mask]
        denominator = np.sum(selected_weights, axis=-1)
        numerator = np.sum(values[..., mask] * selected_weights, axis=-1)
        np.divide(
            numerator,
            denominator,
            out=output[..., row_index],
            where=denominator > 0.0,
        )
    return output


def _surface_lsg_field(variable: xr.DataArray, path: str | Path) -> np.ndarray:
    """Return one LSG surface field as a time-by-horizontal array."""
    field = variable
    if "lev" in field.dims:
        if field.sizes["lev"] != 1:
            raise ValueError(f"Surface LSG field {variable.name!r} has multiple levels in {path}.")
        field = field.isel(lev=0)
    expected = ("time", "south_north", "west_east")
    if field.dims != expected:
        raise ValueError(
            f"Surface LSG field {variable.name!r} has dimensions {field.dims} in {path}; "
            f"expected {expected}."
        )
    return np.asarray(field.values, dtype=float)


def _source_attrs(variable: xr.DataArray, **extra: object) -> dict[str, object]:
    """Return portable source metadata plus extraction-specific attributes."""
    attrs: dict[str, object] = {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in variable.attrs.items()
        if key in {"long_name", "units", "code", "cell_methods"}
    }
    attrs.update(source_variable=str(variable.name), **extra)
    return attrs


def _layer_overlap_thickness(
    bathymetry: np.ndarray,
    lower_interfaces: np.ndarray,
    upper_interfaces: np.ndarray,
    band_lower: float,
    band_upper: float,
) -> np.ndarray:
    """Return layer thickness inside one depth band and above bathymetry."""
    return np.maximum(
        0.0,
        np.minimum.reduce(
            (
                np.broadcast_to(bathymetry, (lower_interfaces.size, *bathymetry.shape)),
                np.broadcast_to(
                    upper_interfaces[:, None, None],
                    (lower_interfaces.size, *bathymetry.shape),
                ),
                np.full(
                    (lower_interfaces.size, *bathymetry.shape), band_upper, dtype=float
                ),
            )
        )
        - np.maximum(lower_interfaces[:, None, None], band_lower),
    )


def lsg_ocean_layer_maps_for_file(path: str | Path) -> xr.Dataset:
    """Extract annual native-grid ocean layer maps from one LSG file."""
    with xr.open_dataset(path) as dataset:
        (
            latitude,
            longitude,
            depth,
            lower_interfaces,
            upper_interfaces,
            bathymetry,
            wet,
            _,
        ) = _lsg_ocean_geometry(dataset, path)
        required = {"psi", "utot", "vtot", "wetvec", "depv", "lat_2", "lon_2"}
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ValueError(f"Missing required LSG layer-map variables in {path}: {missing}.")

        years = _filename_years(path, dataset.sizes["time"])
        if years is None:
            raise ValueError(f"Cannot recover model years from LSG filename {path}.")
        temperature = np.asarray(
            dataset["t"].transpose("time", "depth", "south_north", "west_east"),
            dtype=float,
        )
        if not np.isfinite(temperature[:, wet > 0.0]).all():
            raise ValueError(f"LSG wet-cell potential temperature is non-finite in {path}.")

        vector_expected = ("time", "depth", "south_north", "west_east")
        for name in ("utot", "vtot", "wetvec"):
            if dataset[name].dims != vector_expected:
                raise ValueError(
                    f"LSG {name} has dimensions {dataset[name].dims} in {path}; "
                    f"expected {vector_expected}."
                )
        for name in ("utot", "vtot"):
            if dataset[name].attrs.get("units") != "m/s":
                raise ValueError(f"LSG {name} in {path} must have units 'm/s'.")

        vector_wet_variable = dataset["wetvec"]
        vector_wet = np.asarray(vector_wet_variable.isel(time=0), dtype=float)
        if not np.all((vector_wet == 0.0) | (vector_wet == 1.0)):
            raise ValueError(f"LSG vector wet mask is not binary in {path}.")
        if not np.array_equal(
            vector_wet, np.asarray(vector_wet_variable.isel(time=-1), dtype=float)
        ):
            raise ValueError(f"LSG vector wet mask changes with time in {path}.")

        vector_bathymetry_variable = dataset["depv"]
        if "lev" in vector_bathymetry_variable.dims:
            if vector_bathymetry_variable.sizes["lev"] != 1:
                raise ValueError(f"LSG depv has multiple lev values in {path}.")
            vector_bathymetry_variable = vector_bathymetry_variable.isel(lev=0)
        vector_bathymetry_variable = vector_bathymetry_variable.transpose(
            "time", "south_north", "west_east"
        )
        vector_bathymetry = np.asarray(
            vector_bathymetry_variable.isel(time=0), dtype=float
        )
        if not np.array_equal(
            vector_bathymetry,
            np.asarray(vector_bathymetry_variable.isel(time=-1), dtype=float),
        ):
            raise ValueError(f"LSG vector bathymetry changes with time in {path}.")

        vector_latitude = np.asarray(dataset["lat_2"], dtype=float)
        vector_longitude = np.asarray(dataset["lon_2"], dtype=float)
        if vector_latitude.shape != latitude.shape or vector_longitude.shape != latitude.shape:
            raise ValueError(f"LSG vector coordinates have invalid shapes in {path}.")

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
        for label, band_lower, band_upper in LSG_LAYER_MAP_BANDS_M:
            thickness = _layer_overlap_thickness(
                bathymetry,
                lower_interfaces,
                upper_interfaces,
                band_lower,
                band_upper,
            ) * wet
            denominator = np.sum(thickness, axis=0)
            layer_mean = np.full(
                (dataset.sizes["time"], *latitude.shape), np.nan, dtype=float
            )
            np.divide(
                np.sum(temperature * thickness[None, :, :, :], axis=1),
                denominator[None, :, :],
                out=layer_mean,
                where=denominator[None, :, :] > 0.0,
            )
            data_vars[f"theta_layer_{label}"] = (
                ("year", "south_north", "west_east"),
                layer_mean.astype("float32"),
                {
                    "long_name": f"potential temperature averaged over {band_lower:g}-{band_upper:g} m",
                    "units": "K",
                    "source_variable": "t",
                    "vertical_weighting": "wet thickness including partial layer and bottom-cell overlap",
                    "depth_range_m": f"{band_lower:g}-{band_upper:g}",
                },
            )

        psi = dataset["psi"]
        if psi.dims != ("time", "lev", "south_north", "west_east") or psi.sizes["lev"] != 1:
            raise ValueError(f"LSG psi has unexpected dimensions in {path}: {psi.dims}.")
        psi_values = np.asarray(psi.isel(lev=0), dtype=float)
        if not np.isfinite(psi_values[:, wet[0] > 0.0]).all():
            raise ValueError(f"LSG wet-cell barotropic streamfunction is non-finite in {path}.")
        psi_values[:, wet[0] <= 0.0] = np.nan
        data_vars["barotropic_streamfunction"] = (
            ("year", "south_north", "west_east"),
            psi_values.astype("float32"),
            {
                "long_name": "horizontal barotropic streamfunction",
                "units": "Sv",
                "source_variable": "psi",
                "source_code": 27,
                "source_units_note": (
                    "raw NetCDF declares m3/s, but lsgmod.f90 computes psi with a "
                    "1e-6 factor; native numerical values are retained and labelled Sv"
                ),
                "sign_convention": (
                    "native LSG sign; no transformation; positive psi increments "
                    "northward for positive eastward depth-mean zonal flow"
                ),
            },
        )

        grid_step = np.deg2rad(LSG_HORIZONTAL_GRID_SPACING_DEGREES)
        dlh = LSG_EARTH_RADIUS_M * grid_step * np.cos(np.deg2rad(vector_latitude))
        dlh = np.where(np.abs(vector_latitude) <= 90.0, dlh, 0.0)
        for label, band_lower, band_upper in LSG_LAYER_MAP_BANDS_M:
            thickness = _layer_overlap_thickness(
                vector_bathymetry,
                lower_interfaces,
                upper_interfaces,
                band_lower,
                band_upper,
            ) * vector_wet
            weights = thickness * dlh[None, :, :]
            denominator = np.sum(weights, axis=0)
            for source_name in ("utot", "vtot"):
                values = np.asarray(dataset[source_name], dtype=float)
                if not np.isfinite(values[:, vector_wet > 0.0]).all():
                    raise ValueError(
                        f"LSG wet-point velocity {source_name!r} is non-finite in {path}."
                    )
                annual_mean = np.full(
                    (dataset.sizes["time"], *latitude.shape), np.nan, dtype=float
                )
                np.divide(
                    np.sum(values * weights[None, :, :, :], axis=1),
                    denominator[None, :, :],
                    out=annual_mean,
                    where=denominator[None, :, :] > 0.0,
                )
                component = "u" if source_name == "utot" else "v"
                data_vars[f"{component}_layer_{label}"] = (
                    ("year", "south_north", "west_east"),
                    annual_mean.astype("float32"),
                    {
                        "units": "m s-1",
                        "source_variable": source_name,
                        "weighting": "wet vector-face cross-sectional area within the depth band",
                        "sign_convention": "native LSG sign; not transformed",
                    },
                )
                data_vars[f"_{component}_sum_{label}"] = (
                    ("south_north", "west_east"),
                    np.nansum(annual_mean, axis=0),
                    {"annual_sample_count": int(dataset.sizes["time"])},
                )

        data_vars.update(
            {
                "wet": (
                    ("lsg_depth", "south_north", "west_east"),
                    wet.astype("int8"),
                    {"long_name": "static scalar-point wet mask", "units": "1"},
                ),
                "wetvec": (
                    ("lsg_depth", "south_north", "west_east"),
                    vector_wet.astype("int8"),
                    {"long_name": "static vector-point wet mask", "units": "1"},
                ),
                "depth_bounds": (
                    ("lsg_depth", "bounds"),
                    np.column_stack((lower_interfaces, upper_interfaces)),
                    {"long_name": "LSG layer depth bounds", "units": "m"},
                ),
            }
        )
        return xr.Dataset(
            data_vars,
            coords={
                "year": np.asarray(years, dtype=int),
                "lsg_depth": ("lsg_depth", depth, {"units": "m", "positive": "down"}),
                "south_north": np.arange(latitude.shape[0], dtype=int),
                "west_east": np.arange(latitude.shape[1], dtype=int),
                "bounds": np.asarray([0, 1], dtype=int),
                "lat": (("south_north", "west_east"), latitude, dict(dataset["lat"].attrs)),
                "lon": (("south_north", "west_east"), longitude, dict(dataset["lon"].attrs)),
                "lat_2": (
                    ("south_north", "west_east"),
                    vector_latitude,
                    dict(dataset["lat_2"].attrs),
                ),
                "lon_2": (
                    ("south_north", "west_east"),
                    vector_longitude,
                    dict(dataset["lon_2"].attrs),
                ),
            },
            attrs={"velocity_annual_sample_count": int(dataset.sizes["time"])},
        ).load()


def _lsg_meridional_transport_fields(
    dataset: xr.Dataset,
    path: str | Path,
    temperature: np.ndarray,
    depth: np.ndarray,
    lower_interfaces: np.ndarray,
    upper_interfaces: np.ndarray,
) -> xr.Dataset:
    """Return native E-grid meridional volume and temperature transports."""
    required = {"vtot", "wetvec", "depv", "lat_2"}
    missing = sorted(required - set(dataset.variables))
    if missing:
        raise ValueError(
            f"Missing required LSG vector-grid variables in {path}: {missing}."
        )

    expected = ("time", "depth", "south_north", "west_east")
    velocity = dataset["vtot"]
    vector_wet_variable = dataset["wetvec"]
    if velocity.dims != expected or vector_wet_variable.dims != expected:
        raise ValueError(
            f"LSG vtot/wetvec dimensions in {path} must both be {expected}."
        )
    if velocity.attrs.get("units") != "m/s":
        raise ValueError(f"LSG vtot in {path} must have units 'm/s'.")

    vector_latitude = np.asarray(dataset["lat_2"], dtype=float)
    if vector_latitude.shape != velocity.shape[-2:]:
        raise ValueError(f"LSG vector latitude has an invalid shape in {path}.")
    vector_wet = np.asarray(vector_wet_variable.isel(time=0), dtype=float)
    if not np.all((vector_wet == 0.0) | (vector_wet == 1.0)):
        raise ValueError(f"LSG vector wet mask is not binary in {path}.")
    if vector_wet_variable.sizes["time"] > 1 and not np.array_equal(
        vector_wet, np.asarray(vector_wet_variable.isel(time=-1), dtype=float)
    ):
        raise ValueError(f"LSG vector wet mask changes with time in {path}.")

    bathymetry_variable = dataset["depv"]
    if "lev" in bathymetry_variable.dims:
        if bathymetry_variable.sizes["lev"] != 1:
            raise ValueError(f"LSG vector bathymetry has multiple lev values in {path}.")
        bathymetry_variable = bathymetry_variable.isel(lev=0)
    bathymetry_variable = bathymetry_variable.transpose(
        "time", "south_north", "west_east"
    )
    vector_bathymetry = np.asarray(
        bathymetry_variable.isel(time=0), dtype=float
    )
    if bathymetry_variable.sizes["time"] > 1 and not np.allclose(
        vector_bathymetry,
        np.asarray(bathymetry_variable.isel(time=-1), dtype=float),
        rtol=0.0,
        atol=0.0,
    ):
        raise ValueError(f"LSG vector bathymetry changes with time in {path}.")

    thickness = np.maximum(
        0.0,
        np.minimum(vector_bathymetry[None, :, :], upper_interfaces[:, None, None])
        - lower_interfaces[:, None, None],
    )
    thickness *= vector_wet
    face_width = (
        LSG_EARTH_RADIUS_M
        * np.deg2rad(LSG_HORIZONTAL_GRID_SPACING_DEGREES)
        * np.cos(np.deg2rad(vector_latitude))
    )
    face_width = np.where(np.abs(vector_latitude) <= 90.0, face_width, 0.0)
    cross_section = thickness * face_width[None, :, :]

    # In adv_quick, vector point (q, r) is the northern corner of scalar
    # cell (q + 1, r + 1). Its other adjacent scalar cell is (q, r - 1).
    # The final vector row is outside this two-cell stencil and must be dry
    # in the native grid. At the first row adv_quick's max(j - 2, 1)
    # boundary rule selects the first scalar row as the northern neighbour.
    if np.any(vector_wet[:, -1, :] > 0.0):
        raise ValueError(
            f"LSG wet vector points occupy the final polar row without a complete "
            f"adv_quick temperature stencil in {path}."
        )
    face_temperature = np.full_like(temperature, np.nan, dtype=float)
    northern_temperature = np.concatenate(
        (temperature[:, :, :1, :], temperature[:, :, :-2, :]), axis=2
    )
    face_temperature[:, :, :-1, :] = 0.5 * (
        np.roll(temperature[:, :, 1:, :], shift=-1, axis=-1)
        + northern_temperature
    )

    vector_rows, vector_masks = _lsg_latitude_rows(
        vector_latitude, vector_wet, path
    )
    velocity_values = np.asarray(velocity, dtype=float)
    if not np.isfinite(velocity_values[:, vector_wet > 0.0]).all():
        raise ValueError(f"LSG wet-point meridional velocity is non-finite in {path}.")
    if not np.isfinite(face_temperature[:, vector_wet > 0.0]).all():
        raise ValueError(
            f"LSG wet-point face potential temperature is non-finite in {path}."
        )

    row_area = np.stack(
        [np.sum(cross_section[..., mask], axis=-1) for mask in vector_masks]
    )
    volume_transport = np.stack(
        [
            np.sum(
                velocity_values[..., mask] * cross_section[..., mask], axis=-1
            )
            for mask in vector_masks
        ],
        axis=1,
    )
    temperature_transport = np.stack(
        [
            np.sum(
                velocity_values[..., mask]
                * face_temperature[..., mask]
                * cross_section[..., mask],
                axis=-1,
            )
            for mask in vector_masks
        ],
        axis=1,
    )
    face_temperature_mean = _row_weighted_mean(
        face_temperature, cross_section, vector_masks
    ).transpose(0, 2, 1)

    years = _filename_years(path, dataset.sizes["time"])
    if years is None:
        raise ValueError(f"Cannot recover model years from LSG filename {path}.")
    return xr.Dataset(
        {
            "zonal_meridional_volume_transport": (
                ("year", "lsg_vector_lat", "lsg_depth"),
                volume_transport.astype("float32"),
                {
                    "long_name": "zonal integral of meridional volume transport",
                    "units": "m3 s-1",
                    "source_variable": "vtot",
                    "calculation": "sum(vtot * dlh * delta) on native LSG vector points",
                    "sign_convention": "native LSG sign; not transformed",
                },
            ),
            "zonal_face_potential_temperature": (
                ("year", "lsg_vector_lat", "lsg_depth"),
                face_temperature_mean.astype("float32"),
                {
                    "long_name": "face-area-weighted potential temperature on meridional faces",
                    "units": "K",
                    "source_variable": "t",
                    "interpolation": (
                        "native adv_quick E-grid arithmetic mean of the two adjacent "
                        "scalar-cell annual temperatures"
                    ),
                    "weighting": "static wet vector cross-sectional area",
                },
            ),
            "zonal_meridional_temperature_transport_proxy": (
                ("year", "lsg_vector_lat", "lsg_depth"),
                temperature_transport.astype("float32"),
                {
                    "long_name": "annual-mean-flow meridional temperature transport proxy",
                    "units": "K m3 s-1",
                    "source_variables": "vtot, t",
                    "calculation": "sum(vtot_bar * theta_face_bar * dlh * delta)",
                    "interpretation": (
                        "vbar*thetabar proxy; excludes sub-annual velocity-temperature "
                        "covariance and non-advective transport"
                    ),
                    "sign_convention": "native LSG sign; not transformed",
                },
            ),
            "zonal_net_meridional_volume_transport": (
                ("year", "lsg_vector_lat"),
                np.sum(volume_transport, axis=-1).astype("float32"),
                {
                    "long_name": "full-depth zonal integral of meridional volume transport",
                    "units": "m3 s-1",
                    "source_variable": "vtot",
                    "calculation": "sum over longitude and depth of vtot * dlh * delta",
                    "sign_convention": "native LSG sign; not transformed",
                },
            ),
            "wet_vector_cross_section_area": (
                ("lsg_vector_lat", "lsg_depth"),
                row_area.astype("float64"),
                {
                    "long_name": "static wet vector cross-sectional area",
                    "units": "m2",
                    "source_variables": "wetvec, depv",
                    "calculation": "sum(dlh * delta) over each vector latitude row",
                },
            ),
        },
        coords={
            "year": np.asarray(years, dtype=int),
            "lsg_vector_lat": (
                "lsg_vector_lat",
                vector_rows.astype("float64"),
                {
                    "long_name": "LSG vector-grid latitude row",
                    "units": "degrees_north",
                },
            ),
            "lsg_depth": depth.astype("float64"),
        },
    )


def lsg_zonal_fields_for_file(path: str | Path) -> xr.Dataset:
    """Extract annual zonal mechanism fields from one LSG ocean file."""
    with xr.open_dataset(path) as dataset:
        wet_volume_native = _lsg_wet_cell_volumes(dataset, path)
        (
            latitude,
            _,
            depth,
            lower_interfaces,
            upper_interfaces,
            _,
            wet,
            horizontal_area,
        ) = _lsg_ocean_geometry(dataset, path)
        row_latitudes, row_masks = _lsg_latitude_rows(latitude, wet, path)
        years = _filename_years(path, dataset.sizes["time"])
        if years is None:
            raise ValueError(f"Cannot recover model years from LSG filename {path}.")

        temperature = np.asarray(
            dataset["t"].transpose("time", "depth", "south_north", "west_east"),
            dtype=float,
        )
        if not np.isfinite(temperature[:, wet_volume_native > 0.0]).all():
            raise ValueError(f"LSG wet-cell potential temperature is non-finite in {path}.")

        vector_transport = _lsg_meridional_transport_fields(
            dataset,
            path,
            temperature,
            depth,
            lower_interfaces,
            upper_interfaces,
        )

        row_volume = np.stack(
            [np.sum(wet_volume_native[..., mask], axis=-1) for mask in row_masks]
        )
        row_cell_count = np.stack(
            [
                np.sum(wet_volume_native[..., mask] > 0.0, axis=-1)
                for mask in row_masks
            ]
        )
        surface_weights = wet[0] * horizontal_area
        row_surface_area = np.asarray(
            [np.sum(surface_weights[mask]) for mask in row_masks], dtype=float
        )
        temperature_mean = _row_weighted_mean(
            temperature, wet_volume_native, row_masks
        ).transpose(0, 2, 1)
        temperature_integral = np.stack(
            [
                np.sum(
                    temperature[..., mask] * wet_volume_native[..., mask],
                    axis=(-2, -1),
                )
                for mask in row_masks
            ],
            axis=1,
        )
        row_heat_content = (
            LSG_REFERENCE_DENSITY_KG_M3
            * LSG_SPECIFIC_HEAT_J_KG_K
            * (
                temperature_integral
                - LSG_REFERENCE_TEMPERATURE_K * np.sum(row_volume, axis=1)[None, :]
            )
        )

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {
            "zonal_potential_temperature": (
                ("year", "lsg_lat", "lsg_depth"),
                temperature_mean.astype("float32"),
                _source_attrs(
                    dataset["t"],
                    weighting="wet-volume-weighted within each latitude row and level",
                ),
            ),
            "zonal_ocean_heat_content": (
                ("year", "lsg_lat"),
                row_heat_content.astype("float32"),
                {
                    "long_name": "full-depth ocean heat content by LSG latitude row",
                    "units": "J",
                    "source_variable": "t",
                    "reference_temperature_K": LSG_REFERENCE_TEMPERATURE_K,
                },
            ),
            "wet_volume": (
                ("lsg_lat", "lsg_depth"),
                row_volume.astype("float64"),
                {"long_name": "static wet ocean volume", "units": "m3"},
            ),
            "wet_surface_area": (
                ("lsg_lat",),
                row_surface_area.astype("float64"),
                {"long_name": "static wet ocean surface area", "units": "m2"},
            ),
            "wet_cell_count": (
                ("lsg_lat", "lsg_depth"),
                row_cell_count.astype("float64"),
                {"long_name": "number of wet scalar cells", "units": "1"},
            ),
            "depth_bounds": (
                ("lsg_depth", "bounds"),
                np.column_stack((lower_interfaces, upper_interfaces)).astype("float64"),
                {"long_name": "LSG layer depth bounds", "units": "m"},
            ),
        }
        for name, variable in vector_transport.data_vars.items():
            data_vars[name] = (variable.dims, np.asarray(variable), dict(variable.attrs))
        skipped: list[str] = []

        if "s" in dataset:
            salinity = np.asarray(
                dataset["s"].transpose("time", "depth", "south_north", "west_east"),
                dtype=float,
            )
            salinity_mean = _row_weighted_mean(
                salinity, wet_volume_native, row_masks
            ).transpose(0, 2, 1)
            data_vars["zonal_salinity"] = (
                ("year", "lsg_lat", "lsg_depth"),
                salinity_mean.astype("float32"),
                _source_attrs(
                    dataset["s"],
                    weighting="wet-volume-weighted within each latitude row and level",
                ),
            )
        else:
            skipped.append("s")

        if "w" in dataset:
            vertical_velocity = dataset["w"]
            expected_w_dims = ("time", "depth_2", "south_north", "west_east")
            if vertical_velocity.dims != expected_w_dims:
                raise ValueError(
                    f"LSG w has dimensions {vertical_velocity.dims} in {path}; "
                    f"expected {expected_w_dims}."
                )
            interface_area = np.zeros_like(wet, dtype=float)
            interface_area[:-1] = wet[1:] * horizontal_area[None, :, :]
            w_mean = _row_weighted_mean(
                np.asarray(vertical_velocity, dtype=float), interface_area, row_masks
            ).transpose(0, 2, 1)
            data_vars["zonal_vertical_velocity"] = (
                ("year", "lsg_lat", "lsg_depth_interface"),
                w_mean.astype("float32"),
                _source_attrs(
                    vertical_velocity,
                    vertical_location="lower interface of each potential-temperature layer",
                    sign_convention="native LSG sign; not transformed",
                    weighting="wet horizontal area; an interface is wet when the layer below is wet",
                ),
            )
        else:
            skipped.append("w")

        if "convad" in dataset:
            convective = dataset["convad"]
            expected_convad_dims = (
                "time",
                "depth_3",
                "south_north",
                "west_east",
            )
            if convective.dims != expected_convad_dims or not np.allclose(
                dataset["depth_3"], dataset["depth"].isel(depth=slice(1, None))
            ):
                raise ValueError(f"LSG convad vertical grid is not aligned with depth[1:] in {path}.")
            convective_weights = wet[1:] * horizontal_area[None, :, :]
            convective_mean = _row_weighted_mean(
                np.asarray(convective, dtype=float), convective_weights, row_masks
            ).transpose(0, 2, 1)
            padded = np.full(
                (dataset.sizes["time"], row_latitudes.size, depth.size),
                np.nan,
                dtype="float32",
            )
            padded[:, :, 1:] = convective_mean.astype("float32")
            data_vars["zonal_convective_adjustment"] = (
                ("year", "lsg_lat", "lsg_depth"),
                padded,
                _source_attrs(
                    convective,
                    weighting="wet-area-weighted within each latitude row and level",
                    interpretation=(
                        "annual time mean of the source convective-adjustment event field; "
                        "the 25 m level is unavailable"
                    ),
                ),
            )
        else:
            skipped.append("convad")

        surface_outputs = {
            "fluxhea": "zonal_ocean_heat_flux",
            "flukhea": "zonal_newtonian_coupling_heat_flux",
            "fluwat": "zonal_fresh_water_flux",
            "flukwat": "zonal_newtonian_coupling_fresh_water_flux",
            "fldsst": "zonal_sst_mismatch",
            "fldice": "zonal_ice_mismatch",
            "zeta": "zonal_sea_surface_height",
            "tbound": "zonal_boundary_temperature",
        }
        for source_name, output_name in surface_outputs.items():
            if source_name not in dataset:
                skipped.append(source_name)
                continue
            row_mean = _row_weighted_mean(
                _surface_lsg_field(dataset[source_name], path),
                surface_weights,
                row_masks,
            )
            data_vars[output_name] = (
                ("year", "lsg_lat"),
                row_mean.astype("float32"),
                _source_attrs(
                    dataset[source_name],
                    weighting="wet-surface-area-weighted within each latitude row",
                    sign_convention="native LSG sign; not transformed",
                ),
            )
            if source_name == "zeta":
                global_mean = (
                    np.sum(row_mean * row_surface_area[None, :], axis=1)
                    / np.sum(row_surface_area)
                )
                data_vars["global_sea_surface_height"] = (
                    ("year",),
                    global_mean.astype("float32"),
                    _source_attrs(
                        dataset[source_name],
                        long_name="global wet-area-weighted mean sea-surface height",
                        weighting="wet-surface-area-weighted global mean",
                        sign_convention="native LSG sign; not transformed",
                    ),
                )

        if "sice" in dataset:
            sea_ice = _surface_lsg_field(dataset["sice"], path)
            ice_mean = _row_weighted_mean(sea_ice, surface_weights, row_masks)
            ice_fraction = _row_weighted_mean(
                (sea_ice >= 0.05).astype(float), surface_weights, row_masks
            )
            data_vars["zonal_lsg_ice_thickness"] = (
                ("year", "lsg_lat"),
                ice_mean.astype("float32"),
                _source_attrs(
                    dataset["sice"],
                    weighting="wet-surface-area-weighted within each latitude row",
                ),
            )
            data_vars["zonal_lsg_ice_covered_fraction"] = (
                ("year", "lsg_lat"),
                ice_fraction.astype("float32"),
                {
                    "long_name": "fraction of wet surface area with LSG ice thickness at least 0.05 m",
                    "units": "1",
                    "source_variable": "sice",
                    "ice_thickness_threshold_m": 0.05,
                },
            )
            global_ice_volume = np.sum(
                sea_ice * surface_weights[None, :, :], axis=(1, 2)
            )
            data_vars["global_lsg_ice_volume"] = (
                ("year",),
                global_ice_volume.astype("float32"),
                {
                    "long_name": "global LSG sea-ice volume including snow",
                    "units": "m3",
                    "source_variable": "sice",
                    "calculation": "sum of sice times native wet surface area",
                    "interpretation": "water-equivalent ice thickness including snow",
                },
            )
        else:
            skipped.append("sice")

        if "taux" in dataset and "wetvec" in dataset and "lat_2" in dataset:
            vector_latitude = np.asarray(dataset["lat_2"], dtype=float)
            vector_wet_variable = dataset["wetvec"].transpose(
                "time", "depth", "south_north", "west_east"
            )
            vector_wet = np.asarray(vector_wet_variable.isel(time=0), dtype=float)
            if not np.array_equal(
                vector_wet, np.asarray(vector_wet_variable.isel(time=-1), dtype=float)
            ):
                raise ValueError(f"LSG vector wet mask changes with time in {path}.")
            vector_rows, vector_masks = _lsg_latitude_rows(
                vector_latitude, vector_wet, path
            )
            vector_area = np.where(
                np.abs(vector_latitude) <= 90.0,
                0.5
                * (LSG_EARTH_RADIUS_M * np.deg2rad(LSG_HORIZONTAL_GRID_SPACING_DEGREES))
                ** 2
                * np.cos(np.deg2rad(vector_latitude)),
                0.0,
            )
            vector_surface_weights = vector_wet[0] * vector_area
            taux_mean = _row_weighted_mean(
                _surface_lsg_field(dataset["taux"], path),
                vector_surface_weights,
                vector_masks,
            )
            data_vars["zonal_zonal_wind_stress"] = (
                ("year", "lsg_vector_lat"),
                taux_mean.astype("float32"),
                _source_attrs(
                    dataset["taux"],
                    weighting="wet-vector-surface-area-weighted within each vector latitude row",
                    sign_convention="native LSG sign; not transformed",
                ),
            )
        else:
            skipped.append("taux or its vector-grid metadata")

        result = xr.Dataset(
            data_vars,
            coords={
                "year": np.asarray(years, dtype=int),
                "lsg_lat": (
                    "lsg_lat",
                    row_latitudes.astype("float64"),
                    {"long_name": "LSG scalar-grid latitude row", "units": "degrees_north"},
                ),
                "lsg_depth": (
                    "lsg_depth",
                    depth.astype("float64"),
                    {"long_name": "LSG layer-centre depth", "units": "m", "positive": "down"},
                ),
                "lsg_depth_interface": (
                    "lsg_depth_interface",
                    upper_interfaces.astype("float64"),
                    {"long_name": "LSG lower-layer-interface depth", "units": "m", "positive": "down"},
                ),
                "lsg_vector_lat": vector_transport.coords["lsg_vector_lat"],
                "bounds": np.asarray([0, 1], dtype=int),
            },
            attrs={
                "skipped_optional_lsg_variables": ", ".join(sorted(skipped)),
                "lsg_latitude_row_count": int(row_latitudes.size),
                "lsg_depth_level_count": int(depth.size),
                "lsg_depth_interface_count": int(upper_interfaces.size),
                "lsg_vector_latitude_row_count": int(
                    vector_transport.sizes["lsg_vector_lat"]
                ),
                "horizontal_weighting": (
                    "native LSG E-grid metric 0.5 * (R * 5 degrees)^2 * cos(latitude)"
                ),
                "vertical_weighting": "partial bottom cells reconstructed from depp",
            },
        )
        return result.load()


def plasim_surface_maps_for_file(path: str | Path) -> xr.Dataset:
    """Extract annual PlaSim surface maps and optional ocean-only zonal fluxes."""
    with xr.open_dataset(path) as dataset:
        required = {"sic", "sit", "ts", "lsm", "lat", "lon"}
        missing = sorted(required - set(dataset.variables))
        if missing:
            raise ValueError(f"Missing required PlaSim mechanism variables in {path}: {missing}.")
        expected = ("time", "lat", "lon")
        for name in ("sic", "sit", "ts", "lsm"):
            if dataset[name].dims != expected:
                raise ValueError(
                    f"PlaSim field {name!r} has dimensions {dataset[name].dims} in {path}; "
                    f"expected {expected}."
                )
        land_sea_mask = np.asarray(dataset["lsm"], dtype=float)
        if not np.allclose(land_sea_mask, land_sea_mask[0], rtol=0.0, atol=0.0):
            raise ValueError(f"PlaSim land-sea mask changes with time in {path}.")
        if not np.all((land_sea_mask[0] == 0.0) | (land_sea_mask[0] == 1.0)):
            raise ValueError(f"PlaSim land-sea mask is not binary in {path}.")
        years = _filename_years(path, dataset.sizes["time"])
        if years is None:
            raise ValueError(f"Cannot recover model years from PlaSim filename {path}.")

        data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {}
        for source_name, output_name in (
            ("sic", "sea_ice_concentration"),
            ("sit", "sea_ice_thickness"),
            ("ts", "surface_temperature"),
        ):
            data_vars[output_name] = (
                ("year", "t21_lat", "t21_lon"),
                np.asarray(dataset[source_name], dtype="float32"),
                _source_attrs(dataset[source_name]),
            )
        data_vars["lsm"] = (
            ("t21_lat", "t21_lon"),
            land_sea_mask[0].astype("float64"),
            _source_attrs(dataset["lsm"], static_field="time-invariant source mask"),
        )

        ocean = land_sea_mask[0] == 0.0
        ocean_count = np.sum(ocean, axis=1)
        skipped: list[str] = []
        for source_name, output_name in _MECHANISM_OPTIONAL_PLA_VARIABLES.items():
            if source_name not in dataset:
                skipped.append(source_name)
                continue
            values = np.asarray(dataset[source_name], dtype=float)
            numerator = np.sum(np.where(ocean[None, :, :], values, 0.0), axis=2)
            zonal = np.full(numerator.shape, np.nan, dtype=float)
            np.divide(
                numerator,
                ocean_count[None, :],
                out=zonal,
                where=ocean_count[None, :] > 0,
            )
            data_vars[output_name] = (
                ("year", "t21_lat"),
                zonal.astype("float32"),
                _source_attrs(
                    dataset[source_name],
                    weighting="unweighted mean over ocean longitudes; land excluded with lsm",
                ),
            )

        return xr.Dataset(
            data_vars,
            coords={
                "year": np.asarray(years, dtype=int),
                "t21_lat": (
                    "t21_lat",
                    np.asarray(dataset["lat"], dtype="float64"),
                    dict(dataset["lat"].attrs),
                ),
                "t21_lon": (
                    "t21_lon",
                    np.asarray(dataset["lon"], dtype="float64"),
                    dict(dataset["lon"].attrs),
                ),
            },
            attrs={
                "skipped_optional_plasim_variables": ", ".join(sorted(skipped)),
            },
        ).load()


def _assert_contiguous_unique_years(years: np.ndarray, product: str) -> None:
    """Validate a sorted annual coordinate."""
    if np.unique(years).size != years.size:
        raise ValueError(f"{product} years overlap.")
    if years.size > 1 and not np.array_equal(np.diff(years), np.ones(years.size - 1)):
        raise ValueError(f"{product} years are not contiguous.")


def _path_year_count(path: str | Path) -> int:
    """Return the inclusive number of years declared by an output filename."""
    match = _OUTPUT_YEAR_RANGE_PATTERN.search(Path(path).name)
    if match is None:
        raise ValueError(f"Cannot recover an output year range from {path}.")
    first, last = (int(value) for value in match.groups())
    if last < first:
        raise ValueError(f"Output filename has a reversed year range: {path}.")
    return last - first + 1


def _collect_mechanism_results(
    results: object,
    *,
    expected_year_count: int,
    static_names: Sequence[str],
    required_dynamic_names: set[str],
    optional_dynamic_names: set[str],
    product: str,
    work_dir: Path | None = None,
) -> tuple[xr.Dataset, xr.Dataset, set[str]]:
    """Incrementally assemble per-file datasets into preallocated annual arrays."""
    iterator = iter(results)  # type: ignore[arg-type]
    try:
        reference = next(iterator)
    except StopIteration as error:
        raise ValueError(f"No {product} per-file results were produced.") from error

    present_required = required_dynamic_names - set(reference.data_vars)
    if present_required:
        raise ValueError(f"{product} result is missing required fields: {sorted(present_required)}")
    common_optional = set(reference.data_vars) & optional_dynamic_names
    dynamic_names = required_dynamic_names | common_optional
    arrays: dict[str, np.ndarray] = {}
    dimensions: dict[str, tuple[str, ...]] = {}
    attributes: dict[str, dict[str, object]] = {}
    for name in dynamic_names:
        variable = reference[name]
        if not variable.dims or variable.dims[0] != "year":
            raise ValueError(f"Dynamic {product} field {name!r} does not begin with year.")
        shape = (expected_year_count, *variable.shape[1:])
        if work_dir is None:
            arrays[name] = np.empty(shape, dtype=variable.dtype)
        else:
            safe_product = re.sub(r"[^A-Za-z0-9]+", "_", product).strip("_").lower()
            arrays[name] = np.memmap(
                work_dir / f"{safe_product}_{name}.dat",
                mode="w+",
                dtype=variable.dtype,
                shape=shape,
            )
        dimensions[name] = variable.dims
        attributes[name] = dict(variable.attrs)
    years = np.empty(expected_year_count, dtype=int)
    cursor = 0

    def store(result: xr.Dataset) -> None:
        nonlocal cursor, common_optional
        count = result.sizes["year"]
        next_cursor = cursor + count
        if next_cursor > expected_year_count:
            raise ValueError(f"{product} results contain more years than their filenames declare.")
        for name in static_names:
            if not result[name].identical(reference[name]):
                raise ValueError(f"Static {product} field {name!r} changes between files.")
        for name in list(common_optional):
            if name not in result:
                common_optional.remove(name)
                arrays.pop(name)
                dimensions.pop(name)
                attributes.pop(name)
        for name in required_dynamic_names | common_optional:
            variable = result[name]
            if variable.dims != dimensions[name] or variable.shape[1:] != arrays[name].shape[1:]:
                raise ValueError(f"Dynamic {product} field {name!r} changes shape between files.")
            arrays[name][cursor:next_cursor] = np.asarray(variable.values)
        years[cursor:next_cursor] = np.asarray(result["year"], dtype=int)
        cursor = next_cursor

    store(reference)
    for result in iterator:
        store(result)
    if cursor != expected_year_count:
        raise ValueError(
            f"{product} results contain {cursor} years; filenames declare {expected_year_count}."
        )
    order = np.argsort(years)
    years = years[order]
    _assert_contiguous_unique_years(years, product)
    data_vars = {
        name: (dimensions[name], values[order], attributes[name])
        for name, values in arrays.items()
    }
    coordinate_names = {
        dimension
        for dims in dimensions.values()
        for dimension in dims
        if dimension != "year"
    }
    coordinates: dict[str, object] = {"year": years}
    for name in coordinate_names:
        coordinates[name] = reference.coords[name]
    dynamic = xr.Dataset(data_vars, coords=coordinates)
    return dynamic, reference, common_optional


def build_ocean_layer_maps(
    lsg_paths: Sequence[str | Path],
    *,
    workers: int = 2,
    work_dir: str | Path | None = None,
) -> xr.Dataset:
    """Build contiguous annual native-grid LSG layer maps."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if not lsg_paths:
        raise ValueError("At least one LSG NetCDF file is required.")
    layer_work_dir = Path(work_dir) if work_dir is not None else None
    if layer_work_dir is not None:
        layer_work_dir.mkdir(parents=True, exist_ok=True)

    dynamic_names = tuple(
        [f"theta_layer_{label}" for label, _, _ in LSG_LAYER_MAP_BANDS_M]
        + ["barotropic_streamfunction"]
    )
    velocity_accumulator_names = tuple(
        f"_{component}_sum_{label}"
        for label, _, _ in LSG_LAYER_MAP_VELOCITY_BANDS_M
        for component in ("u", "v")
    )
    static_names = ("wet", "wetvec", "depth_bounds")
    expected_year_count = sum(_path_year_count(path) for path in lsg_paths)
    worker_count = min(workers, len(lsg_paths))

    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        iterator = iter(executor.map(lsg_ocean_layer_maps_for_file, lsg_paths))
        try:
            reference = next(iterator)
        except StopIteration as error:
            raise ValueError("No LSG ocean-layer-map results were produced.") from error

        arrays: dict[str, np.ndarray] = {}
        for name in dynamic_names:
            variable = reference[name]
            shape = (expected_year_count, *variable.shape[1:])
            if layer_work_dir is None:
                arrays[name] = np.empty(shape, dtype="float32")
            else:
                arrays[name] = np.memmap(
                    layer_work_dir / f"{name}.dat",
                    mode="w+",
                    dtype="float32",
                    shape=shape,
                )
        velocity_sums = {
            name: np.zeros(reference[name].shape, dtype=float)
            for name in velocity_accumulator_names
        }
        years = np.empty(expected_year_count, dtype=int)
        cursor = 0
        velocity_sample_count = 0

        def store(result: xr.Dataset) -> None:
            nonlocal cursor, velocity_sample_count
            count = result.sizes["year"]
            next_cursor = cursor + count
            if next_cursor > expected_year_count:
                raise ValueError("LSG layer-map results exceed filename year counts.")
            for name in static_names:
                if not result[name].identical(reference[name]):
                    raise ValueError(f"Static LSG layer-map field {name!r} changes between files.")
            for name in ("lat", "lon", "lat_2", "lon_2", "lsg_depth"):
                if not result.coords[name].identical(reference.coords[name]):
                    raise ValueError(f"LSG layer-map coordinate {name!r} changes between files.")
            for name in dynamic_names:
                arrays[name][cursor:next_cursor] = np.asarray(result[name], dtype="float32")
            for name in velocity_accumulator_names:
                velocity_sums[name] += np.asarray(result[name], dtype=float)
            sample_count = int(result.attrs["velocity_annual_sample_count"])
            if sample_count != count:
                raise ValueError("LSG velocity and layer-map annual sample counts differ.")
            velocity_sample_count += sample_count
            years[cursor:next_cursor] = np.asarray(result["year"], dtype=int)
            cursor = next_cursor

        store(reference)
        for result in iterator:
            store(result)

    if cursor != expected_year_count:
        raise ValueError(
            f"LSG layer-map results contain {cursor} years; filenames declare "
            f"{expected_year_count}."
        )
    _assert_contiguous_unique_years(years, "LSG ocean-layer-map")
    if not np.all(np.diff(years) == 1):
        raise ValueError("LSG layer-map files were not supplied in chronological order.")

    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray, dict[str, object]]] = {
        name: (reference[name].dims, arrays[name], dict(reference[name].attrs))
        for name in dynamic_names
    }
    wetvec = np.asarray(reference["wetvec"], dtype=bool)
    depth_bounds = np.asarray(reference["depth_bounds"], dtype=float)
    valid_latitude = np.abs(np.asarray(reference["lat_2"], dtype=float)) <= 90.0
    for label, band_lower, band_upper in LSG_LAYER_MAP_VELOCITY_BANDS_M:
        nominal_overlap = np.maximum(
            0.0,
            np.minimum(depth_bounds[:, 1], band_upper)
            - np.maximum(depth_bounds[:, 0], band_lower),
        )
        valid = np.any(wetvec & (nominal_overlap[:, None, None] > 0.0), axis=0)
        valid &= valid_latitude
        for component, source_name in (("u", "utot"), ("v", "vtot")):
            mean = np.full(reference["lat_2"].shape, np.nan, dtype="float32")
            mean[valid] = (
                velocity_sums[f"_{component}_sum_{label}"][valid]
                / velocity_sample_count
            ).astype("float32")
            data_vars[f"{component}_mean_{label}"] = (
                ("south_north", "west_east"),
                mean,
                {
                    "long_name": (
                        f"time-mean {source_name} averaged over "
                        f"{band_lower:g}-{band_upper:g} m"
                    ),
                    "units": "m s-1",
                    "source_variable": source_name,
                    "time_weighting": "arithmetic mean of annual source records",
                    "vertical_weighting": (
                        "native vector-point dlh * delta with partial depth-band "
                        "and bottom-cell overlap"
                    ),
                    "sign_convention": "native LSG sign; not transformed",
                },
            )

    output = xr.Dataset(
        data_vars,
        coords={
            **{
                name: reference.coords[name]
                for name in reference.coords
                if name != "year"
            },
            "year": years,
        },
        attrs={
            "lsg_file_count": len(lsg_paths),
            "worker_count": worker_count,
            "velocity_annual_sample_count": velocity_sample_count,
            "grid": "native LSG E-grid; no regridding",
            "temporal_processing": "annual source values; no smoothing or filtering",
        },
    )
    for name in static_names:
        output[name] = reference[name]
    return output


def build_mechanism_fields(
    lsg_paths: Sequence[str | Path],
    atmospheric_paths: Sequence[str | Path],
    *,
    workers: int = 4,
    work_dir: str | Path | None = None,
) -> xr.Dataset:
    """Build aligned zonal-ocean and PlaSim surface mechanism fields.

    ``work_dir`` selects disk-backed arrays for large extractions. The caller
    must keep that directory alive until the returned dataset has been written.
    """
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if not lsg_paths or not atmospheric_paths:
        raise ValueError("At least one LSG and one PlaSim atmospheric file are required.")
    mechanism_work_dir = Path(work_dir) if work_dir is not None else None
    if mechanism_work_dir is not None:
        mechanism_work_dir.mkdir(parents=True, exist_ok=True)
    worker_count = min(workers, len(lsg_paths) + len(atmospheric_paths))
    lsg_static_names = (
        "wet_volume",
        "wet_surface_area",
        "wet_cell_count",
        "depth_bounds",
        "wet_vector_cross_section_area",
    )
    lsg_optional_outputs = set(_MECHANISM_OPTIONAL_LSG_VARIABLES.values()) | {
        "zonal_lsg_ice_covered_fraction",
        "global_sea_surface_height",
        "global_lsg_ice_volume",
    }
    lsg_dynamic_required = {
        "zonal_potential_temperature",
        "zonal_ocean_heat_content",
        "zonal_meridional_volume_transport",
        "zonal_face_potential_temperature",
        "zonal_meridional_temperature_transport_proxy",
        "zonal_net_meridional_volume_transport",
    }
    pla_optional_outputs = set(_MECHANISM_OPTIONAL_PLA_VARIABLES.values())
    pla_dynamic_required = {
        "sea_ice_concentration",
        "sea_ice_thickness",
        "surface_temperature",
    }
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        lsg_dynamic, lsg_reference, common_lsg_optional = _collect_mechanism_results(
            executor.map(lsg_zonal_fields_for_file, lsg_paths),
            expected_year_count=sum(_path_year_count(path) for path in lsg_paths),
            static_names=lsg_static_names,
            required_dynamic_names=lsg_dynamic_required,
            optional_dynamic_names=lsg_optional_outputs,
            product="LSG mechanism-field",
            work_dir=mechanism_work_dir,
        )
        (
            atmospheric_dynamic,
            atmospheric_reference,
            common_pla_optional,
        ) = _collect_mechanism_results(
            executor.map(plasim_surface_maps_for_file, atmospheric_paths),
            expected_year_count=sum(
                _path_year_count(path) for path in atmospheric_paths
            ),
            static_names=("lsm",),
            required_dynamic_names=pla_dynamic_required,
            optional_dynamic_names=pla_optional_outputs,
            product="PlaSim mechanism-field",
            work_dir=mechanism_work_dir,
        )
    lsg_years = np.asarray(lsg_dynamic["year"], dtype=int)
    atmospheric_years = np.asarray(atmospheric_dynamic["year"], dtype=int)
    if not np.array_equal(lsg_years, atmospheric_years):
        raise ValueError(
            "LSG and PLA years differ: "
            f"LSG {lsg_years[0]}-{lsg_years[-1]} ({lsg_years.size} years), "
            f"PLA {atmospheric_years[0]}-{atmospheric_years[-1]} "
            f"({atmospheric_years.size} years)."
        )

    combined = xr.merge(
        [lsg_dynamic, atmospheric_dynamic], compat="equals", join="exact"
    )
    for name in lsg_static_names:
        combined[name] = lsg_reference[name]
    combined["lsm"] = atmospheric_reference["lsm"]
    missing_lsg_optional = sorted(
        source_name
        for source_name, output_name in _MECHANISM_OPTIONAL_LSG_VARIABLES.items()
        if output_name not in common_lsg_optional
    )
    missing_pla_optional = sorted(
        source_name
        for source_name, output_name in _MECHANISM_OPTIONAL_PLA_VARIABLES.items()
        if output_name not in common_pla_optional
    )
    combined.attrs.update(
        lsg_file_count=len(lsg_paths),
        atmospheric_file_count=len(atmospheric_paths),
        worker_count=worker_count,
        skipped_optional_variables=", ".join(
            missing_lsg_optional + missing_pla_optional
        ),
        lsg_latitude_row_count=lsg_reference.sizes["lsg_lat"],
        lsg_depth_level_count=lsg_reference.sizes["lsg_depth"],
        lsg_depth_interface_count=lsg_reference.sizes["lsg_depth_interface"],
        lsg_vector_latitude_row_count=lsg_reference.sizes["lsg_vector_lat"],
        horizontal_weighting=lsg_reference.attrs["horizontal_weighting"],
        vertical_weighting=lsg_reference.attrs["vertical_weighting"],
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


def build_cold_branch_cheap_datasets(
    atmospheric_paths: Sequence[str | Path],
    diagnostic_paths: Sequence[str | Path],
    *,
    workers: int = 4,
) -> PlasimColdBranchCheapDatasets:
    """Build aligned cheap cold-branch atmospheric and printed LSG diagnostics."""
    if workers < 1:
        raise ValueError("workers must be at least 1.")
    if not atmospheric_paths or not diagnostic_paths:
        raise ValueError("Atmospheric and DIAG paths are both required.")
    worker_count = min(workers, max(len(atmospheric_paths), len(diagnostic_paths)))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        atmospheric_results = list(
            executor.map(cold_branch_atmospheric_diagnostics_for_file, atmospheric_paths)
        )
        printed_results = list(
            executor.map(extended_lsg_diagnostics_for_file, diagnostic_paths)
        )
    scalar_parts, zonal_parts = zip(*atmospheric_results)
    scalar = xr.concat(scalar_parts, dim="time").sortby("year")
    zonal = xr.concat(zonal_parts, dim="time").sortby("year")
    printed = xr.concat(printed_results, dim="year").sortby("year")
    years = np.asarray(scalar["year"].values, dtype=int)
    if not np.array_equal(years, np.asarray(printed["year"].values, dtype=int)):
        raise ValueError("PLA and DIAG years do not align exactly.")
    for name, variable in printed.data_vars.items():
        if variable.dims == ("year",):
            scalar[name] = xr.DataArray(
                variable.values, dims="time", coords={"time": scalar["time"]}, attrs=variable.attrs
            )
        elif variable.dims == ("year", "diagnostic_layer"):
            scalar[name] = xr.DataArray(
                variable.values,
                dims=("time", "diagnostic_layer"),
                coords={"time": scalar["time"], "diagnostic_layer": printed["diagnostic_layer"]},
                attrs=variable.attrs,
            )
    scalar = scalar.assign_coords(
        diagnostic_layer_depth=printed["diagnostic_layer_depth"]
    )
    scalar.attrs.update(printed.attrs)
    return PlasimColdBranchCheapDatasets(
        diagnostics=scalar,
        zonal=zonal,
        atmospheric_file_count=len(atmospheric_paths),
        diagnostic_file_count=len(diagnostic_paths),
        worker_count=worker_count,
    )
