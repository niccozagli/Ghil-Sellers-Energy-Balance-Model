"""Tests for diagnostics reconstructed from PLASIM output."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from gsebm.plasim_diagnostics import (
    LSG_EARTH_RADIUS_M,
    LSG_HORIZONTAL_GRID_SPACING_DEGREES,
    LSG_REFERENCE_DENSITY_KG_M3,
    LSG_REFERENCE_TEMPERATURE_K,
    LSG_SPECIFIC_HEAT_J_KG_K,
    SEA_ICE_COVER_THRESHOLD,
    SEA_ICE_ZONAL_OCCUPANCY_THRESHOLD,
    amoc_diagnostics_for_file,
    atmospheric_diagnostics_for_file,
    build_lsg_ocean_diagnostics,
    lsg_ocean_diagnostics_for_file,
    lsg_diagnostics_for_file,
    temperature_diagnostics_for_file,
    zonal_temperature_for_file,
)


def lsg_ocean_dataset(years: tuple[int, ...] = (2000, 2001)) -> xr.Dataset:
    """Return a compact annual LSG-like ocean dataset."""
    latitude = np.array([[30.0, 30.0], [-45.0, -45.0]])
    longitude = np.array([[0.0, 5.0], [0.0, 5.0]])
    depth = np.array([25.0, 75.0, 125.0, 175.0, 225.0, 275.0, 350.0,
                      450.0, 550.0, 650.0, 750.0, 850.0, 950.0, 1100.0,
                      1300.0, 1500.0, 1800.0, 2250.0])
    depth_interfaces = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 312.5,
                                 400.0, 500.0, 600.0, 700.0, 800.0, 900.0,
                                 1025.0, 1200.0, 1400.0, 1650.0, 2025.0,
                                 2500.0])
    shape = (len(years), depth.size, 2, 2)
    temperature = np.empty(shape)
    for time_index in range(len(years)):
        temperature[time_index] = 280.0 + time_index
    wet = np.ones(shape)
    wet[:, :, 1, 1] = 0.0
    bathymetry = np.array([[2500.0, 750.0], [2000.0, 0.0]])
    return xr.Dataset(
        {
            "t": (("time", "depth", "south_north", "west_east"), temperature,
                  {"units": "K"}),
            "wet": (("time", "depth", "south_north", "west_east"), wet),
            "depp": (("time", "lev", "south_north", "west_east"),
                     np.broadcast_to(bathymetry, (len(years), 1, 2, 2))),
        },
        coords={
            "time": np.arange(len(years), dtype=float),
            "depth": depth,
            "depth_2": depth_interfaces,
            "lev": [1.0],
            "lat": (("south_north", "west_east"), latitude),
            "lon": (("south_north", "west_east"), longitude),
        },
    )


def plasim_dataset() -> xr.Dataset:
    """Return a compact annual PLASIM-like dataset on a Gaussian grid."""
    nodes, _ = np.polynomial.legendre.leggauss(4)
    latitude = np.degrees(np.arcsin(nodes))[::-1]
    time = np.array([20000101.5, 20010101.5])
    longitude = np.array([0.0, 180.0])
    shape = (time.size, latitude.size, longitude.size)
    pattern = np.arange(np.prod(shape), dtype=float).reshape(shape)
    land_sea_mask = np.zeros(shape)
    land_sea_mask[:, 0, 1] = 1.0
    sea_ice_thickness = np.zeros(shape)
    sea_ice_thickness[0, 0, 0] = 0.1
    sea_ice_thickness[0, 0, 1] = 9.0
    sea_ice_thickness[0, 3, :] = 0.2
    sea_ice_thickness[1, 1, :] = 0.06
    sea_ice_concentration = np.clip(5.0 * sea_ice_thickness, 0.0, 1.0)
    surface_albedo = 0.1 + 0.01 * pattern

    return xr.Dataset(
        {
            "ts": (("time", "lat", "lon"), 260.0 + pattern),
            "tas": (("time", "lat", "lon"), 265.0 + pattern),
            "rst": (("time", "lat", "lon"), 210.0 + 0.5 * pattern),
            "rsut": (("time", "lat", "lon"), -(90.0 + 0.25 * pattern)),
            "rlut": (("time", "lat", "lon"), -(205.0 + 0.4 * pattern)),
            "lsm": (("time", "lat", "lon"), land_sea_mask),
            "as": (("time", "lat", "lon"), surface_albedo),
            "sic": (("time", "lat", "lon"), sea_ice_concentration),
            "sit": (("time", "lat", "lon"), sea_ice_thickness),
        },
        coords={"time": time, "lat": latitude, "lon": longitude},
    )


def gaussian_global_mean(dataset: xr.Dataset, name: str) -> np.ndarray:
    """Evaluate the expected Gaussian-weighted global mean."""
    _, weights = np.polynomial.legendre.leggauss(dataset.sizes["lat"])
    zonal_mean = dataset[name].mean("lon").values
    return np.average(zonal_mean, axis=1, weights=weights[::-1])


def gaussian_hemisphere_mean(
    dataset: xr.Dataset, name: str, hemisphere_sign: int
) -> np.ndarray:
    """Evaluate the expected Gaussian-weighted hemispheric mean."""
    _, weights = np.polynomial.legendre.leggauss(dataset.sizes["lat"])
    latitude = dataset["lat"].values
    hemisphere = hemisphere_sign * latitude > 0.0
    zonal_mean = dataset[name].mean("lon").values[:, hemisphere]
    return np.average(zonal_mean, axis=1, weights=weights[::-1][hemisphere])


def gaussian_band_mean(dataset: xr.Dataset, name: str, mask: np.ndarray) -> np.ndarray:
    """Evaluate the expected Gaussian-weighted mean over a latitude mask."""
    _, weights = np.polynomial.legendre.leggauss(dataset.sizes["lat"])
    zonal_mean = dataset[name].mean("lon").values[:, mask]
    return np.average(zonal_mean, axis=1, weights=weights[::-1][mask])


class PlasimDiagnosticsTest(unittest.TestCase):
    def test_temperature_and_toa_diagnostics_use_documented_formulas(self) -> None:
        source = plasim_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plasim.nc"
            source.to_netcdf(path, engine="scipy")
            diagnostics = temperature_diagnostics_for_file(path)

        temperature = gaussian_global_mean(source, "ts")
        rst = gaussian_global_mean(source, "rst")
        rsut = gaussian_global_mean(source, "rsut")
        rlut = gaussian_global_mean(source, "rlut")

        np.testing.assert_allclose(diagnostics["global_temperature"], temperature)
        np.testing.assert_allclose(
            diagnostics["global_toa_incoming_shortwave"], rst - rsut
        )
        np.testing.assert_allclose(
            diagnostics["global_toa_reflected_shortwave"], -rsut
        )
        np.testing.assert_allclose(
            diagnostics["global_toa_absorbed_shortwave"], rst
        )
        np.testing.assert_allclose(
            diagnostics["global_toa_outgoing_longwave"], -rlut
        )
        np.testing.assert_allclose(
            diagnostics["global_toa_energy_imbalance"], rst + rlut
        )
        np.testing.assert_allclose(
            diagnostics["global_planetary_albedo"], -rsut / (rst - rsut)
        )
        np.testing.assert_array_equal(diagnostics["year"], [2000, 2001])
        np.testing.assert_allclose(
            diagnostics["northern_hemisphere_surface_temperature"],
            gaussian_hemisphere_mean(source, "ts", 1),
        )
        np.testing.assert_allclose(
            diagnostics["southern_hemisphere_surface_temperature"],
            gaussian_hemisphere_mean(source, "ts", -1),
        )
        np.testing.assert_allclose(
            diagnostics["northern_hemisphere_2m_temperature"],
            gaussian_hemisphere_mean(source, "tas", 1),
        )
        np.testing.assert_allclose(
            diagnostics["southern_hemisphere_2m_temperature"],
            gaussian_hemisphere_mean(source, "tas", -1),
        )
        expected_tropical_2m = gaussian_band_mean(
            source, "tas", np.abs(source["lat"].values) <= 30.0
        )
        expected_northern_polar_2m = gaussian_band_mean(
            source, "tas", source["lat"].values >= 30.0
        )
        expected_southern_polar_2m = gaussian_band_mean(
            source, "tas", source["lat"].values <= -30.0
        )
        np.testing.assert_allclose(
            diagnostics["global_2m_temperature"], gaussian_global_mean(source, "tas")
        )
        np.testing.assert_allclose(
            diagnostics["northern_polar_2m_temperature_gradient"],
            expected_tropical_2m - expected_northern_polar_2m,
        )
        np.testing.assert_allclose(
            diagnostics["southern_polar_2m_temperature_gradient"],
            expected_tropical_2m - expected_southern_polar_2m,
        )
        self.assertEqual(
            diagnostics["global_toa_energy_imbalance"].attrs["units"],
            "W m-2",
        )
        self.assertIn(
            "positive into",
            diagnostics["global_toa_energy_imbalance"].attrs[
                "flux_sign_convention"
            ],
        )
        self.assertEqual(diagnostics["global_planetary_albedo"].attrs["units"], "1")

    def test_sea_ice_edges_threshold_local_cover_before_zonal_reduction(
        self,
    ) -> None:
        source = plasim_dataset()
        source["sic"][1, 1, :] = SEA_ICE_COVER_THRESHOLD
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plasim.nc"
            source.to_netcdf(path, engine="scipy")
            diagnostics = temperature_diagnostics_for_file(path)

        polar_latitude = float(source["lat"].isel(lat=0))
        equatorward_latitude = float(source["lat"].isel(lat=1))
        expected_margin = (equatorward_latitude + polar_latitude) / 2.0
        self.assertAlmostEqual(
            float(
                diagnostics["northern_persistent_sea_ice_edge_latitude"].isel(
                    time=0
                )
            ),
            expected_margin,
        )
        self.assertAlmostEqual(
            float(
                diagnostics["southern_persistent_sea_ice_edge_latitude"].isel(
                    time=0
                )
            ),
            -expected_margin,
        )
        self.assertEqual(
            float(
                diagnostics["northern_persistent_sea_ice_edge_latitude"].isel(
                    time=1
                )
            ),
            0.0,
        )
        self.assertTrue(
            np.isnan(
                diagnostics["southern_persistent_sea_ice_edge_latitude"].isel(
                    time=1
                )
            )
        )
        self.assertEqual(
            diagnostics["northern_persistent_sea_ice_edge_latitude"].attrs[
                "local_annual_sea_ice_cover_threshold"
            ],
            SEA_ICE_COVER_THRESHOLD,
        )
        self.assertEqual(
            diagnostics["northern_persistent_sea_ice_edge_latitude"].attrs[
                "ocean_longitude_occupancy_threshold"
            ],
            SEA_ICE_ZONAL_OCCUPANCY_THRESHOLD,
        )

    def test_average_cover_cannot_bypass_local_cover_threshold(self) -> None:
        source = plasim_dataset().isel(lon=[0, 0, 0, 0]).assign_coords(
            lon=[0.0, 90.0, 180.0, 270.0]
        )
        source["lsm"][:] = 0.0
        source["sic"][:] = 0.0
        source["sic"][0, 0, :] = [1.0, 0.49, 0.49, 0.49]
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plasim.nc"
            source.to_netcdf(path, engine="scipy")
            diagnostics, zonal = atmospheric_diagnostics_for_file(path)

        self.assertGreater(
            float(zonal["zonal_sea_ice_concentration"].isel(time=0, lat=0)),
            SEA_ICE_COVER_THRESHOLD,
        )
        self.assertEqual(
            float(zonal["zonal_persistent_sea_ice_fraction"].isel(time=0, lat=0)),
            0.25,
        )
        self.assertTrue(
            np.isnan(
                diagnostics["northern_persistent_sea_ice_edge_latitude"].isel(
                    time=0
                )
            )
        )

    def test_zonal_fields_use_documented_longitude_means_and_masks(self) -> None:
        source = plasim_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plasim.nc"
            source.to_netcdf(path, engine="scipy")
            zonal_temperature = zonal_temperature_for_file(path)

        np.testing.assert_allclose(
            zonal_temperature["zonal_surface_temperature"], source["ts"].mean("lon")
        )
        np.testing.assert_allclose(
            zonal_temperature["zonal_2m_temperature"], source["tas"].mean("lon")
        )
        ocean_mask = source["lsm"].isel(time=0) == 0
        np.testing.assert_allclose(
            zonal_temperature["zonal_sea_ice_concentration"],
            source["sic"].where(ocean_mask).mean("lon"),
        )
        np.testing.assert_allclose(
            zonal_temperature["zonal_persistent_sea_ice_fraction"],
            (source["sic"] >= SEA_ICE_COVER_THRESHOLD)
            .where(ocean_mask)
            .mean("lon"),
        )
        np.testing.assert_allclose(
            zonal_temperature["zonal_sea_ice_thickness"],
            source["sit"].where(ocean_mask).mean("lon"),
        )
        np.testing.assert_allclose(
            zonal_temperature["zonal_surface_albedo"], source["as"].mean("lon")
        )
        np.testing.assert_allclose(
            zonal_temperature["zonal_toa_energy_imbalance"],
            (source["rst"] + source["rlut"]).mean("lon"),
        )
        self.assertEqual(
            zonal_temperature["zonal_toa_energy_imbalance"].attrs[
                "flux_sign_convention"
            ],
            "positive into the climate system",
        )
        np.testing.assert_array_equal(zonal_temperature["year"], [2000, 2001])
        self.assertEqual(
            zonal_temperature.attrs["longitude_averaging"],
            "unweighted arithmetic mean over longitude",
        )

    def test_rejects_a_non_gaussian_latitude_grid(self) -> None:
        source = plasim_dataset().assign_coords(lat=[60.0, 20.0, -20.0, -60.0])
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plasim.nc"
            source.to_netcdf(path, engine="scipy")
            with self.assertRaisesRegex(ValueError, "Gaussian latitude grid"):
                temperature_diagnostics_for_file(path)

    def test_margin_ignores_latitude_belts_without_ocean_cells(self) -> None:
        source = plasim_dataset()
        source["lsm"][:, 0, :] = 1.0
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "plasim.nc"
            source.to_netcdf(path, engine="scipy")
            diagnostics = temperature_diagnostics_for_file(path)

        self.assertTrue(
            np.isnan(
                diagnostics["northern_persistent_sea_ice_edge_latitude"].isel(
                    time=0
                )
            )
        )

    def test_amoc_entries_are_averaged_by_model_year(self) -> None:
        diagnostic_text = """\
* LSG timestep 1 date: 1-Jan-2000 *
ATL max (NADW)   : 10.0 20.0 30.0
Icevol. m**3  0.100D+15 icecov.area m**2  0.200E+14 Av.thickness in m  5.0
Iceareas: 18.0 2.0 90.0 10.0
ATL max (NADW)   : 14.0 24.0 34.0
Icevol. m**3  0.200E+15 icecov.area m**2  0.400E+14 Av.thickness in m  5.0
Iceareas: 36.0 4.0 180.0 20.0
* LSG timestep 2 date: 1-Jan-2001 *
ATL max (NADW)   : 18.0 28.0 38.0
Icevol. m**3  0.120E+15 icecov.area m**2  0.300E+14 Av.thickness in m  4.0
Iceareas: 25.0 5.0 95.0 25.0
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "DIAG.txt"
            path.write_text(diagnostic_text, encoding="utf-8")
            diagnostics = lsg_diagnostics_for_file(path)
            amoc_only = amoc_diagnostics_for_file(path)

        np.testing.assert_array_equal(diagnostics["year"], [2000, 2001])
        np.testing.assert_allclose(diagnostics["amoc_strength"], [12.0, 18.0])
        np.testing.assert_allclose(
            diagnostics["amoc_strength_16_44n"], [22.0, 28.0]
        )
        np.testing.assert_allclose(
            diagnostics["amoc_nadw_export_30s"], [32.0, 38.0]
        )
        np.testing.assert_array_equal(diagnostics["amoc_sample_count"], [2, 1])
        np.testing.assert_allclose(amoc_only["amoc_strength"], [12.0, 18.0])
        np.testing.assert_allclose(
            diagnostics["lsg_sea_ice_volume"], [1.5e14, 1.2e14]
        )
        np.testing.assert_allclose(
            diagnostics["lsg_sea_ice_area"], [3.0e13, 3.0e13]
        )
        np.testing.assert_allclose(
            diagnostics["lsg_sea_ice_mean_thickness"], [5.0, 4.0]
        )
        np.testing.assert_allclose(
            diagnostics["lsg_northern_sea_ice_area"], [2.7e13, 2.5e13]
        )
        np.testing.assert_allclose(
            diagnostics["lsg_southern_sea_ice_area"], [3.0e12, 5.0e12]
        )
        np.testing.assert_allclose(
            diagnostics["lsg_northern_sea_ice_volume"], [1.35e14, 9.5e13]
        )
        np.testing.assert_allclose(
            diagnostics["lsg_southern_sea_ice_volume"], [1.5e13, 2.5e13]
        )
        np.testing.assert_array_equal(
            diagnostics["lsg_sea_ice_sample_count"], [2, 1]
        )

    def test_amoc_years_are_recovered_when_lsg_date_year_overflows(self) -> None:
        diagnostic_text = """\
* LSG timestep 1 date: 10-Jan-**** *
ATL max (NADW)   : 1.0 2.0 3.0
* LSG timestep 36 date: 30-Dec-**** *
ATL max (NADW)   : 4.0 5.0 6.0
* LSG timestep 37 date: 10-Jan-**** *
ATL max (NADW)   : 7.0 8.0 9.0
* LSG timestep 360 date: 30-Dec-**** *
ATL max (NADW)   : 10.0 11.0 12.0
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = (
                Path(temporary_directory)
                / "CONTROL_360ppm_T21L10_10000Y_MU_1312_DIAG.10000-10009.txt"
            )
            path.write_text(diagnostic_text, encoding="utf-8")
            diagnostics = amoc_diagnostics_for_file(path)

        np.testing.assert_array_equal(diagnostics["year"], [10000, 10001, 10009])
        np.testing.assert_allclose(
            diagnostics["amoc_strength"], [2.5, 7.0, 10.0]
        )

    def test_filename_years_override_offset_lsg_calendar(self) -> None:
        """PLASIM file ranges are canonical when the printed calendar is offset."""
        diagnostic_text = """\
* LSG timestep 1 date: 10-Jan-9500 *
ATL max (NADW)   : 1.0 2.0 3.0
* LSG timestep 37 date: 10-Jan-9501 *
ATL max (NADW)   : 4.0 5.0 6.0
"""
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = (
                Path(temporary_directory)
                / "CONTROL_360ppm_T21L10_10000Y_MU_1240_DIAG.10000-10009.txt"
            )
            path.write_text(diagnostic_text, encoding="utf-8")
            diagnostics = amoc_diagnostics_for_file(path)

        np.testing.assert_array_equal(diagnostics["year"], [10000, 10001])

    def test_ocean_heat_content_uses_model_volume_and_partial_depths(self) -> None:
        source = lsg_ocean_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = (
                Path(temporary_directory)
                / "CONTROL_360ppm_T21L10_10000Y_MU_1240_LSG.2000-2001.nc"
            )
            source.to_netcdf(path)
            diagnostics = lsg_ocean_diagnostics_for_file(path)

        np.testing.assert_array_equal(diagnostics["year"], [2000, 2001])
        np.testing.assert_allclose(
            diagnostics["ocean_mean_potential_temperature"],
            np.broadcast_to(np.array([280.0, 281.0])[:, None, None], (2, 3, 3)),
        )
        grid_size = LSG_EARTH_RADIUS_M * np.deg2rad(
            LSG_HORIZONTAL_GRID_SPACING_DEGREES
        )
        northern_cell_area = 0.5 * grid_size**2 * np.cos(np.deg2rad(30.0))
        expected_northern_deep_volume = northern_cell_area * (2000.0 - 750.0)
        self.assertAlmostEqual(
            float(
                diagnostics["ocean_volume"].sel(
                    ocean_region="northern_midlatitudes",
                    depth_band="750_2000m",
                )
            ),
            expected_northern_deep_volume,
        )
        volume = diagnostics["ocean_volume"].values
        expected_heat_content = (
            LSG_REFERENCE_DENSITY_KG_M3
            * LSG_SPECIFIC_HEAT_J_KG_K
            * (280.0 - LSG_REFERENCE_TEMPERATURE_K)
            * volume
        )
        np.testing.assert_allclose(
            diagnostics["ocean_heat_content"].isel(year=0), expected_heat_content
        )

    def test_build_ocean_diagnostics_sorts_files_and_rejects_year_gaps(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            paths = []
            for year in (2001, 2000):
                path = root / f"EXPERIMENT_LSG.{year}-{year}.nc"
                lsg_ocean_dataset((year,)).to_netcdf(path)
                paths.append(path)
            diagnostics = build_lsg_ocean_diagnostics(paths, workers=1)
            np.testing.assert_array_equal(diagnostics["year"], [2000, 2001])

            gap_path = root / "EXPERIMENT_LSG.2003-2003.nc"
            lsg_ocean_dataset((2003,)).to_netcdf(gap_path)
            with self.assertRaisesRegex(ValueError, "not contiguous"):
                build_lsg_ocean_diagnostics([paths[1], gap_path], workers=1)


if __name__ == "__main__":
    unittest.main()
