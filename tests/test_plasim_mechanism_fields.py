"""Tests for the PlaSim-LSG mechanism-field helpers used by raw-map extraction."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr

from gsebm.plasim_diagnostics import (
    LSG_EARTH_RADIUS_M,
    LSG_HORIZONTAL_GRID_SPACING_DEGREES,
    build_mechanism_fields,
    build_ocean_layer_maps,
    lsg_ocean_layer_maps_for_file,
    lsg_zonal_fields_for_file,
    plasim_surface_maps_for_file,
)



def mechanism_lsg_dataset(year_count: int = 2) -> xr.Dataset:
    """Return a compact LSG dataset with partial cells and optional fields."""
    latitude = np.array([[30.0, 30.0], [-45.0, -45.0]])
    longitude = np.array([[0.0, 5.0], [0.0, 5.0]])
    depth = np.array([25.0, 75.0, 125.0])
    interfaces = np.array([50.0, 100.0, 150.0])
    bathymetry = np.array([[150.0, 75.0], [125.0, 0.0]])
    wet_static = np.array(
        [
            [[1.0, 1.0], [1.0, 0.0]],
            [[1.0, 1.0], [1.0, 0.0]],
            [[1.0, 0.0], [1.0, 0.0]],
        ]
    )
    wet = np.broadcast_to(wet_static, (year_count, *wet_static.shape)).copy()
    vector_wet_static = np.array(
        [
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0]],
            [[0.0, 0.0], [0.0, 0.0]],
        ]
    )
    vector_wet = np.broadcast_to(
        vector_wet_static, (year_count, *vector_wet_static.shape)
    ).copy()
    temperature = np.empty_like(wet)
    temperature[:, :, 0, 0] = 280.0
    temperature[:, :, 0, 1] = 300.0
    temperature[:, :, 1, 0] = 270.0
    temperature[:, :, 1, 1] = 0.0
    temperature[1:] += 1.0
    salinity = np.where(wet > 0.0, 34.0 + temperature / 100.0, 0.0)
    vertical_velocity = np.broadcast_to(
        np.array(
            [
                [[1.0, 3.0], [5.0, 0.0]],
                [[2.0, 8.0], [6.0, 0.0]],
                [[9.0, 9.0], [9.0, 0.0]],
            ]
        ),
        (year_count, 3, 2, 2),
    ).copy()
    convective = np.broadcast_to(
        np.array(
            [
                [[1.0, 3.0], [5.0, 0.0]],
                [[2.0, 8.0], [6.0, 0.0]],
            ]
        ),
        (year_count, 2, 2, 2),
    ).copy()
    surface_shape = (year_count, 1, 2, 2)
    surface_pattern = np.broadcast_to(
        np.array([[[[1.0, 3.0], [5.0, 0.0]]]]), surface_shape
    ).copy()
    return xr.Dataset(
        {
            "t": (
                ("time", "depth", "south_north", "west_east"),
                temperature,
                {"long_name": "potential temperature", "units": "K"},
            ),
            "s": (
                ("time", "depth", "south_north", "west_east"),
                salinity,
                {"long_name": "salinity", "units": "0/00"},
            ),
            "w": (
                ("time", "depth_2", "south_north", "west_east"),
                vertical_velocity,
                {"long_name": "vertical velocity component", "units": "m/s"},
            ),
            "convad": (
                ("time", "depth_3", "south_north", "west_east"),
                convective,
                {"long_name": "convective adjustment events", "units": "1"},
            ),
            "wet": (("time", "depth", "south_north", "west_east"), wet),
            "vtot": (
                ("time", "depth", "south_north", "west_east"),
                np.broadcast_to(
                    np.array(
                        [
                            [[1.0, -0.5], [0.0, 0.0]],
                            [[2.0, 1.0], [0.0, 0.0]],
                            [[-1.0, 0.0], [0.0, 0.0]],
                        ]
                    ),
                    (year_count, 3, 2, 2),
                ).copy(),
                {"long_name": "meridional velocity", "units": "m/s"},
            ),
            "wetvec": (
                ("time", "depth", "south_north", "west_east"),
                vector_wet,
            ),
            "depp": (
                ("time", "lev", "south_north", "west_east"),
                np.broadcast_to(bathymetry, (year_count, 1, 2, 2)),
            ),
            "depv": (
                ("time", "lev", "south_north", "west_east"),
                np.broadcast_to(
                    np.array([[0.0, 75.0], [0.0, 0.0]]),
                    (year_count, 1, 2, 2),
                ),
            ),
            "sice": (
                ("time", "lev", "south_north", "west_east"),
                0.1 * surface_pattern,
                {"long_name": "ice thickness", "units": "m"},
            ),
            "zeta": (
                ("time", "lev", "south_north", "west_east"),
                0.01 * surface_pattern,
                {"long_name": "surface elevation", "units": "m"},
            ),
            "tbound": (
                ("time", "lev", "south_north", "west_east"),
                270.0 + surface_pattern,
                {"long_name": "boundary value of t", "units": "K"},
            ),
            "fluxhea": (
                ("time", "lev", "south_north", "west_east"),
                surface_pattern,
                {"long_name": "heat flux", "units": "W/m2"},
            ),
        },
        coords={
            "time": np.arange(year_count, dtype=float),
            "depth": depth,
            "depth_2": interfaces,
            "depth_3": depth[1:],
            "lev": [1.0],
            "lat": (("south_north", "west_east"), latitude),
            "lon": (("south_north", "west_east"), longitude),
            "lat_2": (
                ("south_north", "west_east"),
                np.array([[20.0, 20.0], [92.5, 92.5]]),
            ),
        },
    )


def mechanism_plasim_dataset(year_count: int = 2) -> xr.Dataset:
    """Return a compact PlaSim surface dataset."""
    latitude = np.array([45.0, -45.0])
    longitude = np.array([0.0, 180.0])
    shape = (year_count, 2, 2)
    pattern = np.arange(np.prod(shape), dtype=float).reshape(shape)
    land = np.zeros(shape)
    land[:, 0, 1] = 1.0
    return xr.Dataset(
        {
            "sic": (("time", "lat", "lon"), 0.1 * pattern),
            "sit": (("time", "lat", "lon"), 0.2 * pattern),
            "ts": (("time", "lat", "lon"), 270.0 + pattern),
            "lsm": (("time", "lat", "lon"), land),
            "prsn": (("time", "lat", "lon"), 1.0 + pattern),
            "evap": (("time", "lat", "lon"), -2.0 - pattern),
        },
        coords={"time": np.arange(year_count), "lat": latitude, "lon": longitude},
    )


def layer_map_lsg_dataset(year_count: int = 2) -> xr.Dataset:
    """Return a compact native-grid LSG dataset for layer-map tests."""
    depth = np.array([25.0, 75.0, 125.0, 175.0, 250.0, 450.0])
    interfaces = np.array([50.0, 100.0, 150.0, 200.0, 300.0, 600.0])
    latitude = np.array([[30.0, 30.0], [-30.0, -30.0]])
    longitude = np.array([[0.0, 5.0], [2.5, 7.5]])
    vector_latitude = latitude - 2.5
    vector_longitude = longitude + 2.5
    bathymetry = np.array([[550.0, 225.0], [0.0, 600.0]])
    lower = np.concatenate(([0.0], interfaces[:-1]))
    wet_static = (bathymetry[None, :, :] > lower[:, None, None]).astype(float)
    wet = np.broadcast_to(wet_static, (year_count, *wet_static.shape)).copy()
    temperature = np.empty_like(wet)
    zonal_velocity = np.empty_like(wet)
    for time in range(year_count):
        for level in range(depth.size):
            temperature[time, level] = 280.0 + 2.0 * level + time
            zonal_velocity[time, level] = 1.0 + level + 10.0 * time
    temperature = np.where(wet > 0.0, temperature, 0.0)
    zonal_velocity = np.where(wet > 0.0, zonal_velocity, 0.0)
    psi = np.broadcast_to(
        np.array([[[[10.0, 20.0], [30.0, 40.0]]]]),
        (year_count, 1, 2, 2),
    ).copy()
    psi[1:] += 1.0
    return xr.Dataset(
        {
            "t": (
                ("time", "depth", "south_north", "west_east"),
                temperature,
                {"units": "K", "cell_methods": "time: mean"},
            ),
            "wet": (("time", "depth", "south_north", "west_east"), wet),
            "depp": (
                ("time", "lev", "south_north", "west_east"),
                np.broadcast_to(bathymetry, (year_count, 1, 2, 2)),
            ),
            "psi": (
                ("time", "lev", "south_north", "west_east"),
                psi,
                {"units": "m3/s", "code": 27, "cell_methods": "time: mean"},
            ),
            "utot": (
                ("time", "depth", "south_north", "west_east"),
                zonal_velocity,
                {"units": "m/s", "cell_methods": "time: mean"},
            ),
            "vtot": (
                ("time", "depth", "south_north", "west_east"),
                -zonal_velocity,
                {"units": "m/s", "cell_methods": "time: mean"},
            ),
            "wetvec": (("time", "depth", "south_north", "west_east"), wet),
            "depv": (
                ("time", "lev", "south_north", "west_east"),
                np.broadcast_to(bathymetry, (year_count, 1, 2, 2)),
            ),
        },
        coords={
            "time": np.arange(year_count, dtype=float),
            "depth": depth,
            "depth_2": interfaces,
            "lev": [1.0],
            "lat": (("south_north", "west_east"), latitude),
            "lon": (("south_north", "west_east"), longitude),
            "lat_2": (("south_north", "west_east"), vector_latitude),
            "lon_2": (("south_north", "west_east"), vector_longitude),
        },
    )


class PlasimMechanismFieldsTest(unittest.TestCase):
    def test_ocean_layer_maps_use_partial_overlap_and_native_psi_sv(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "EXPERIMENT_LSG.2000-2001.nc"
            layer_map_lsg_dataset().to_netcdf(path, engine="scipy")
            fields = lsg_ocean_layer_maps_for_file(path)

        np.testing.assert_allclose(fields["theta_layer_0_100m"][:, 0, 0], [281.0, 282.0])
        np.testing.assert_allclose(
            fields["theta_layer_150_300m"][:, 0, 0],
            [(286.0 * 50.0 + 288.0 * 100.0) / 150.0,
             (287.0 * 50.0 + 289.0 * 100.0) / 150.0],
        )
        np.testing.assert_allclose(fields["theta_layer_300_600m"][:, 0, 0], [290.0, 291.0])
        self.assertTrue(np.isnan(fields["theta_layer_0_100m"][:, 1, 0]).all())
        np.testing.assert_allclose(fields["barotropic_streamfunction"][:, 0, 0], [10.0, 11.0])
        self.assertTrue(np.isnan(fields["barotropic_streamfunction"][:, 1, 0]).all())
        self.assertEqual(fields["barotropic_streamfunction"].attrs["units"], "Sv")
        np.testing.assert_array_equal(fields["wet"], fields["wetvec"])

    def test_ocean_layer_map_builder_time_means_vector_velocity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            first = root / "EXPERIMENT_LSG.2000-2001.nc"
            second = root / "EXPERIMENT_LSG.2002-2003.nc"
            layer_map_lsg_dataset().to_netcdf(first, engine="scipy")
            layer_map_lsg_dataset().to_netcdf(second, engine="scipy")
            fields = build_ocean_layer_maps([first, second], workers=1)

        np.testing.assert_array_equal(fields["year"], [2000, 2001, 2002, 2003])
        np.testing.assert_allclose(fields["u_mean_0_100m"][0, 0], 6.5)
        np.testing.assert_allclose(fields["v_mean_0_100m"][0, 0], -6.5)
        np.testing.assert_allclose(
            fields["u_mean_150_300m"][0, 0],
            (4.0 * 50.0 + 5.0 * 100.0) / 150.0 + 5.0,
        )
        self.assertTrue(np.isnan(fields["u_mean_0_100m"][1, 0]))
        self.assertEqual(fields.attrs["velocity_annual_sample_count"], 4)

    def test_meridional_transport_matches_direct_adv_quick_loop(self) -> None:
        source = mechanism_lsg_dataset()
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "EXPERIMENT_LSG.2000-2001.nc"
            source.to_netcdf(path, engine="scipy")
            fields = lsg_zonal_fields_for_file(path)

        temperature = np.asarray(source["t"], dtype=float)
        velocity = np.asarray(source["vtot"], dtype=float)
        wetvec = np.asarray(source["wetvec"].isel(time=0), dtype=float)
        depv = np.asarray(source["depv"].isel(time=0, lev=0), dtype=float)
        lower = np.array([0.0, 50.0, 100.0])
        upper = np.array([50.0, 100.0, 150.0])
        delta = np.maximum(
            0.0,
            np.minimum(depv[None, :, :], upper[:, None, None])
            - lower[:, None, None],
        ) * wetvec
        dlh = (
            LSG_EARTH_RADIUS_M
            * np.deg2rad(LSG_HORIZONTAL_GRID_SPACING_DEGREES)
            * np.cos(np.deg2rad(np.asarray(source["lat_2"], dtype=float)))
        )

        expected_volume = np.zeros((2, 3))
        expected_temperature = np.zeros((2, 3))
        expected_area = np.zeros(3)
        # Literal transcription for scalar row j=2 (Fortran indexing), whose
        # northern vector row is jm1=1 and northern scalar neighbour is jm2=1.
        j = 1
        jm1 = 0
        jm2 = 0
        for time in range(2):
            for k in range(3):
                for i in range(2):
                    im1 = (i - 1) % 2
                    area = dlh[jm1, im1] * delta[k, jm1, im1]
                    total_velocity = velocity[time, k, jm1, im1] * area
                    theta_face = 0.5 * (
                        temperature[time, k, j, i]
                        + temperature[time, k, jm2, im1]
                    )
                    expected_volume[time, k] += total_velocity
                    expected_temperature[time, k] += total_velocity * theta_face
                    if time == 0:
                        expected_area[k] += area

        transport = fields.sel(lsg_vector_lat=20.0)
        np.testing.assert_allclose(
            transport["zonal_meridional_volume_transport"], expected_volume
        )
        np.testing.assert_allclose(
            transport["zonal_meridional_temperature_transport_proxy"],
            expected_temperature,
        )
        np.testing.assert_allclose(
            transport["wet_vector_cross_section_area"], expected_area
        )
        np.testing.assert_allclose(
            transport["zonal_net_meridional_volume_transport"],
            np.sum(expected_volume, axis=1),
        )
        np.testing.assert_allclose(
            transport["zonal_face_potential_temperature"],
            np.array([[285.0, 285.0, np.nan], [286.0, 286.0, np.nan]]),
            equal_nan=True,
        )

    def test_lsg_zonal_means_use_partial_cell_volume_and_interface_depths(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "EXPERIMENT_LSG.2000-2001.nc"
            mechanism_lsg_dataset().to_netcdf(path, engine="scipy")
            fields = lsg_zonal_fields_for_file(path)

        np.testing.assert_array_equal(fields["lsg_lat"], [-45.0, 30.0])
        np.testing.assert_array_equal(fields["lsg_depth_interface"], [50.0, 100.0, 150.0])
        north = fields.sel(lsg_lat=30.0)
        np.testing.assert_allclose(
            north["zonal_potential_temperature"].isel(year=0),
            [290.0, (280.0 * 50.0 + 300.0 * 25.0) / 75.0, 280.0],
        )
        np.testing.assert_allclose(
            north["zonal_vertical_velocity"].isel(year=0),
            [2.0, 2.0, np.nan],
            equal_nan=True,
        )
        self.assertTrue(np.isnan(float(north["zonal_convective_adjustment"].isel(year=0, lsg_depth=0))))
        np.testing.assert_allclose(
            north["zonal_convective_adjustment"].isel(year=0, lsg_depth=slice(1, None)),
            [2.0, 2.0],
        )
        grid_size = LSG_EARTH_RADIUS_M * np.deg2rad(
            LSG_HORIZONTAL_GRID_SPACING_DEGREES
        )
        cell_area = 0.5 * grid_size**2 * np.cos(np.deg2rad(30.0))
        np.testing.assert_allclose(
            north["wet_volume"],
            [100.0 * cell_area, 75.0 * cell_area, 50.0 * cell_area],
        )
        self.assertEqual(fields["zonal_potential_temperature"].dtype, np.dtype("float32"))
        self.assertEqual(fields["wet_volume"].dtype, np.dtype("float64"))
        np.testing.assert_allclose(
            north["zonal_sea_surface_height"], [0.02, 0.02]
        )
        np.testing.assert_allclose(
            north["zonal_boundary_temperature"], [272.0, 272.0]
        )
        northern_area = 2.0 * cell_area
        southern_area = 0.5 * grid_size**2 * np.cos(np.deg2rad(45.0))
        expected_global_zeta = (
            0.02 * northern_area + 0.05 * southern_area
        ) / (northern_area + southern_area)
        np.testing.assert_allclose(
            fields["global_sea_surface_height"],
            [expected_global_zeta, expected_global_zeta],
        )
        expected_ice_volume = 0.2 * northern_area + 0.5 * southern_area
        np.testing.assert_allclose(
            fields["global_lsg_ice_volume"],
            [expected_ice_volume, expected_ice_volume],
        )

    def test_optional_variables_are_skipped_and_listed(self) -> None:
        source = mechanism_lsg_dataset().drop_vars(
            ["s", "w", "convad", "sice", "fluxhea"]
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "EXPERIMENT_LSG.2000-2001.nc"
            source.to_netcdf(path, engine="scipy")
            fields = lsg_zonal_fields_for_file(path)

        self.assertNotIn("zonal_salinity", fields)
        self.assertNotIn("zonal_vertical_velocity", fields)
        self.assertIn("s", fields.attrs["skipped_optional_lsg_variables"])
        self.assertIn("w", fields.attrs["skipped_optional_lsg_variables"])

    def test_plasim_surface_maps_keep_maps_and_ocean_zonal_means(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "EXPERIMENT_PLA.2000-2001.nc"
            source = mechanism_plasim_dataset()
            source.to_netcdf(path, engine="scipy")
            fields = plasim_surface_maps_for_file(path)

        self.assertEqual(fields["surface_temperature"].dims, ("year", "t21_lat", "t21_lon"))
        self.assertEqual(fields["lsm"].dims, ("t21_lat", "t21_lon"))
        np.testing.assert_allclose(
            fields["zonal_snowfall"].isel(year=0),
            [float(source["prsn"].isel(time=0, lat=0, lon=0)), 3.5],
        )

    def test_builder_merges_grids_and_rejects_year_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            lsg_path = root / "EXPERIMENT_LSG.2000-2001.nc"
            pla_path = root / "EXPERIMENT_PLA.2000-2001.nc"
            mechanism_lsg_dataset().to_netcdf(lsg_path, engine="scipy")
            mechanism_plasim_dataset().to_netcdf(pla_path, engine="scipy")
            fields = build_mechanism_fields([lsg_path], [pla_path], workers=1)

            self.assertIn("lsg_lat", fields.dims)
            self.assertIn("lsg_depth", fields.dims)
            self.assertIn("lsg_depth_interface", fields.dims)
            self.assertIn("t21_lat", fields.dims)
            self.assertIn("t21_lon", fields.dims)
            np.testing.assert_array_equal(fields["year"], [2000, 2001])

            mismatched_path = root / "EXPERIMENT_PLA.2001-2002.nc"
            mechanism_plasim_dataset().to_netcdf(mismatched_path, engine="scipy")
            with self.assertRaisesRegex(ValueError, "LSG and PLA years differ"):
                build_mechanism_fields([lsg_path], [mismatched_path], workers=1)


if __name__ == "__main__":
    unittest.main()
