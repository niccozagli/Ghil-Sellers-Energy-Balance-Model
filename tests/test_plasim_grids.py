"""Tests for PlaSim-LSG grid geometry (Plasim_Indicators.md §2)."""

from pathlib import Path
import tempfile
import unittest

import numpy as np
import xarray as xr

from gsebm.paths import get_repo_root
from gsebm.plasim_grids import (
    ATLANTIC_SIDE_UNLABELLED,
    LSG_LAYER_CENTRES_M,
    LSG_RADIUS_M,
    PLASIM_RADIUS_M,
    atlantic_mask,
    check_lsg_wet_masks,
    check_t21_latitudes,
    lsg_grid_from_lsg,
    lsg_interfaces,
    lsg_layer_thickness,
    lsg_point_area,
    lsg_row_latitudes,
    read_beklas,
    t21_band_area,
    t21_gaussian_grid,
)

ROOT = get_repo_root()
SAMPLE_LSG = ROOT / "data/Plasim/CONTROL_360ppm_T21L10_10000Y_MU_1232p5_LSG.27490-27499.nc"
SOURCE_BEKLAS = ROOT / "PLASIM_INFO/PLASIM-1.0/plasim/run/beklas"


def full_lsg_coordinates() -> tuple[np.ndarray, np.ndarray]:
    """Scalar-point latitude and longitude of the 76 x 72 grid, as in the output files."""
    latitude = np.repeat(lsg_row_latitudes()[:, None], 72, axis=1)
    first_longitude = np.where(np.arange(76) % 2 == 0, 2.5, 0.0)  # file rows alternate 2.5, 0.0
    longitude = first_longitude[:, None] + 5.0 * np.arange(72)[None, :]
    return latitude, longitude


class LayerGeometryTests(unittest.TestCase):
    def test_interfaces_follow_inigr(self):
        bottom, nominal = lsg_interfaces()
        expected = [50, 100, 150, 200, 250, 312.5, 400, 500, 600, 700, 800, 900, 1025,
                    1200, 1400, 1650, 2025, 2500, 3125, 4000, 5000, 6000]
        np.testing.assert_allclose(bottom, expected)
        self.assertEqual(nominal[0], 50.0)
        self.assertAlmostEqual(nominal.sum(), 6000.0)

    def test_deepest_wet_layer_reaches_the_floor(self):
        thickness = lsg_layer_thickness(np.array([25.0, 30.0, 100.0, 2100.0, 6200.0]))
        self.assertEqual(thickness.shape, (22, 5))
        np.testing.assert_allclose(thickness[:, 0], 0.0)  # D = du(1): dry
        self.assertEqual(thickness[0, 1], 30.0)  # du(1) < D < du(2): from 0 to D
        np.testing.assert_allclose(thickness[:3, 2], [50.0, 50.0, 0.0])
        # 2100 m lies below dw(17) = 2025 m but above du(18) = 2250 m: layer 17
        # (1650-2025 m nominal) reaches the floor and layer 18 is dry.
        self.assertEqual(thickness[16, 3], 2100.0 - 1650.0)
        self.assertEqual(thickness[17, 3], 0.0)
        # 6200 m: below dw(22) = 6000 m, so layer 22 is thicker than nominal.
        self.assertEqual(thickness[21, 4], 6200.0 - 5000.0)
        self.assertAlmostEqual(thickness[:, 4].sum(), 6200.0)

    def test_thickness_sums_to_depth_in_every_column(self):
        depth = np.linspace(26.0, 6400.0, 500)
        np.testing.assert_allclose(lsg_layer_thickness(depth).sum(axis=0), depth)

    def test_wet_mask_check_accepts_rules_and_rejects_others(self):
        depp = np.array([[30.0, 100.0, 4000.0]])
        depv = np.array([[80.0, 2000.0, 0.0]])
        wet = depp[None] > LSG_LAYER_CENTRES_M[:-1, None, None]
        wetvec = lsg_layer_thickness(depv) > 0.1
        check_lsg_wet_masks(wet, wetvec, depp, depv)
        wet[1, 0, 1] = False
        with self.assertRaises(ValueError):
            check_lsg_wet_masks(wet, wetvec, depp, depv)


class HorizontalGeometryTests(unittest.TestCase):
    def test_lsg_points_cover_the_sphere(self):
        latitude, _ = full_lsg_coordinates()
        total = lsg_point_area(latitude).sum()
        self.assertAlmostEqual(total / (4 * np.pi * LSG_RADIUS_M**2), 1.0, places=3)

    def test_lsg_ghost_rows_have_no_area(self):
        latitude = lsg_row_latitudes()
        self.assertEqual(latitude[0], 93.75)
        self.assertTrue(np.all(lsg_point_area(latitude[np.abs(latitude) > 90]) == 0.0))

    def test_t21_bands_cover_the_sphere(self):
        latitude, weight = t21_gaussian_grid()
        self.assertTrue(np.all(np.diff(latitude) < 0))
        self.assertAlmostEqual(weight.sum(), 2.0)
        np.testing.assert_allclose(latitude, -latitude[::-1])
        self.assertAlmostEqual(t21_band_area(weight).sum() / (4 * np.pi * PLASIM_RADIUS_M**2), 1.0)

    def test_t21_latitude_check_rejects_a_regular_grid(self):
        latitude, _ = t21_gaussian_grid()
        check_t21_latitudes(latitude)
        with self.assertRaises(ValueError):
            check_t21_latitudes(np.linspace(87.1875, -87.1875, 32))


class BasinTests(unittest.TestCase):
    def test_read_beklas(self):
        rows = ["0" + "".join(str((i + j) % 5) for i in range(72)) for j in range(72)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "beklas"
            path.write_text("\n".join(rows) + "\n")
            codes = read_beklas(path)
        self.assertEqual(codes.shape, (76, 72))
        self.assertTrue(np.all(codes[:2] == 0) and np.all(codes[-2:] == 0))
        self.assertEqual(codes[2, 3], 3)

    def test_atlantic_mask_cuts_and_unlabelled_cells(self):
        latitude, longitude = full_lsg_coordinates()
        wet = np.abs(latitude) <= 90
        codes = np.where(wet, 1, 0)
        listed = np.zeros_like(wet)
        for point_lat, point_lon in ATLANTIC_SIDE_UNLABELLED:
            listed |= np.isclose(latitude, point_lat) & np.isclose(longitude, point_lon)
        codes[listed] = 0
        mask = atlantic_mask(codes, latitude, longitude, wet)

        def at(lat, lon):
            return bool(mask[np.isclose(latitude, lat) & np.isclose(longitude, lon)][0])

        self.assertTrue(at(18.75, 267.5))  # listed unlabelled cell, added
        self.assertTrue(at(-33.75, 330.0))  # north of 34 S
        self.assertFalse(at(-36.25, 332.5))  # south of 34 S
        self.assertFalse(at(33.75, 22.5))  # Mediterranean
        self.assertFalse(at(56.25, 270.0))  # Hudson Bay
        self.assertFalse(at(81.25, 5.0))  # north of 80 N
        self.assertFalse(at(71.25, 100.0))  # Arctic Ocean proper
        self.assertTrue(at(71.25, 5.0))  # Nordic Seas

        codes[np.isclose(latitude, 1.25) & np.isclose(longitude, 330.0)] = 0  # not in either list
        with self.assertRaises(ValueError):
            atlantic_mask(codes, latitude, longitude, wet)


@unittest.skipUnless(SAMPLE_LSG.is_file() and SOURCE_BEKLAS.is_file(), "sample LSG decade not present")
class SampleDecadeGeometryTests(unittest.TestCase):
    def test_volumes_and_masks_of_the_sample(self):
        with xr.open_dataset(SAMPLE_LSG, decode_times=False, engine="h5netcdf") as lsg:
            grid = lsg_grid_from_lsg(lsg, SOURCE_BEKLAS)
        volume = grid.volume
        self.assertAlmostEqual(volume.sum() / 1.4282e18, 1.0, places=4)
        fractions = [volume[a:b].sum() / volume.sum() for a, b in ((0, 10), (10, 17), (17, 22))]
        np.testing.assert_allclose(fractions, [0.175, 0.322, 0.503], atol=5e-4)
        self.assertEqual(int(grid.atlantic.sum()), 583)
        self.assertEqual(int(grid.south_american_land.sum()), 40)


if __name__ == "__main__":
    unittest.main()
