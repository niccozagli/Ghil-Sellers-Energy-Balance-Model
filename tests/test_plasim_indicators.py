"""Tests for PlaSim-LSG indicators (Plasim_Indicators.md §3-§5)."""

from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import h5netcdf
import numpy as np
import xarray as xr

from gsebm.paths import get_repo_root
from gsebm.plasim_grids import t21_gaussian_grid
from gsebm.plasim_indicators import (
    acc_transport,
    area_mean,
    area_total,
    find_decades,
    layer_transport,
    lsg_volume_section,
    overturning_streamfunction,
    product_stem,
    product_status,
    run_indicators,
    run_status,
    salt_content,
    t21_ocean_zonal_mean,
    volume_mean,
    write_indicator_files,
)

ROOT = get_repo_root()
SAMPLE = ROOT / "data/Plasim"
SAMPLE_RUN = "CONTROL_360ppm_T21L10_10000Y_MU_1232p5"
SAMPLE_LABELS = "27490-27499"
SOURCE_BEKLAS = ROOT / "PLASIM_INFO/PLASIM-1.0/plasim/run/beklas"
EXP = "CONTROL_TEST"


def write_time_only_block(path: Path, model_years: list[int]) -> None:
    """A minimal output file holding only ``time``, as read by ``output_block``."""
    with h5netcdf.File(path, "w") as file:
        file.dimensions["time"] = len(model_years)
        time = file.create_variable("time", ("time",), "f8")
        time[:] = np.array(model_years) * 1e4 + 630.5
        time.attrs["units"] = "day as %Y%m%d.%f"


def make_experiment(root: Path, blocks: list[tuple[int, int]], streams=("PLA", "OCE", "ICE", "LSG"),
                    offset: int = 500) -> Path:
    """An experiment folder with time-only output blocks."""
    experiment = root / EXP
    output = experiment / "output" / "spinup"
    output.mkdir(parents=True, exist_ok=True)
    for first, last in blocks:
        for stream in streams:
            write_time_only_block(output / f"{EXP}_{stream}.{first}-{last}.nc",
                                  list(range(first - offset, last - offset + 1)))
    return experiment


class QuantityTests(unittest.TestCase):
    def test_ocean_zonal_mean_skips_land_and_empty_rows(self):
        field = np.array([[[1.0, 2.0, 9.0], [5.0, 5.0, 5.0]]])  # (time, lat, lon)
        ocean = np.array([[True, True, False], [False, False, False]])
        result = t21_ocean_zonal_mean(field, ocean)
        self.assertEqual(result[0, 0], 1.5)
        self.assertTrue(np.isnan(result[0, 1]))

    def test_volume_section_weights_by_volume(self):
        field = np.array([[[[1.0, 3.0, 100.0]]]])  # (time, level, row, point)
        volume = np.array([[[1.0, 3.0, 0.0]]])  # third cell dry
        self.assertEqual(lsg_volume_section(field, volume)[0, 0, 0], (1 * 1 + 3 * 3) / 4)

    def test_overturning_is_bottom_up_with_nadw_positive(self):
        # 2 Sv northward in the top level, 2 Sv southward in the bottom level.
        transport = np.array([[[2e6], [0.0], [-2e6]]])  # (time, level, row)
        psi = overturning_streamfunction(transport)
        np.testing.assert_allclose(psi[0, :, 0], [0.0, 2.0, 2.0])

    def test_layer_transport_ignores_dry_vector_points(self):
        vtot = np.array([[[[0.1, np.nan]]]])
        delta = np.array([[[10.0, 0.0]]])
        zonal_step = np.array([1000.0])
        np.testing.assert_allclose(layer_transport(vtot, delta, zonal_step)[0, 0, 0], 0.1 * 10 * 1000)
        mask = np.array([[False, True]])
        self.assertEqual(layer_transport(vtot, delta, zonal_step, mask)[0, 0, 0], 0.0)

    def test_salt_content_uses_the_free_surface_top_layer(self):
        s = np.full((1, 2, 1, 1), 35.0)
        ddz = np.array([[[50.0]], [[50.0]]])
        wet = np.ones((2, 1, 1), dtype=bool)
        area = np.array([[2.0]])
        result = salt_content(s, ddz, zeta=np.array([[[1.0]]]), sice=np.array([[[3.0]]]), area=area, wet=wet)
        self.assertEqual(result[0], 35.0 * 2.0 * ((50 + 1 - 3) + 50))

    def test_acc_uses_fortran_row_parity(self):
        psi = np.zeros((1, 4, 2))
        psi[0, 1] = 10.0  # Fortran row 2: even
        psi[0, 2] = 12.0  # Fortran row 3: odd
        land = np.zeros((4, 2), dtype=bool)
        land[1:3, 0] = True
        even, odd = acc_transport(psi, land)
        self.assertEqual((even[0], odd[0]), (10.0, 12.0))


class LaptopMeanTests(unittest.TestCase):
    def setUp(self):
        latitude, weight = t21_gaussian_grid()
        self.zonal = xr.Dataset(
            {"field": (("year", "t21_lat"), np.where(latitude > 0, 1.0, 3.0)[None, :]),
             "t21_gaussian_weight": ("t21_lat", weight),
             "t21_ocean_fraction": ("t21_lat", np.where(latitude > 0, 0.25, 0.75)),
             "t21_band_area": ("t21_lat", weight * 10.0)},
            coords={"year": [1], "t21_lat": latitude})

    def test_area_mean_by_region(self):
        self.assertAlmostEqual(float(area_mean(self.zonal.field, self.zonal, "nh")[0]), 1.0)
        self.assertAlmostEqual(float(area_mean(self.zonal.field, self.zonal, "sh")[0]), 3.0)
        self.assertAlmostEqual(float(area_mean(self.zonal.field, self.zonal)[0]), 2.0)
        # ocean-only: weights 0.25 (NH) and 0.75 (SH)
        self.assertAlmostEqual(float(area_mean(self.zonal.field, self.zonal, ocean_only=True)[0]),
                               (0.25 * 1 + 0.75 * 3) / 1.0)

    def test_area_total_uses_ocean_area(self):
        # sum_j A_j f_j z_j with A_j = 10 w_j: NH 10 * 1 * 0.25 * 1, SH 10 * 1 * 0.75 * 3
        self.assertAlmostEqual(float(area_total(self.zonal.field, self.zonal, "nh")[0]), 2.5)
        self.assertAlmostEqual(float(area_total(self.zonal.field, self.zonal, "sh")[0]), 22.5)

    def test_volume_mean_by_levels_and_region(self):
        sections = xr.Dataset(
            {"t": (("year", "lsg_level", "lsg_lat"), np.array([[[1.0, 2.0], [3.0, np.nan]]])),
             "wet_volume": (("lsg_level", "lsg_lat"), np.array([[1.0, 1.0], [2.0, 0.0]]))},
            coords={"year": [1], "lsg_level": [25.0, 75.0], "lsg_lat": [10.0, -10.0]})
        self.assertAlmostEqual(float(volume_mean(sections.t, sections)[0]), (1 + 2 + 3 * 2) / 4)
        self.assertAlmostEqual(float(volume_mean(sections.t, sections, slice(0, 1))[0]), 1.5)
        self.assertAlmostEqual(float(volume_mean(sections.t, sections, region="sh")[0]), 2.0)


class DecadeTests(unittest.TestCase):
    def test_complete_decades_and_a_trailing_incomplete_one(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = make_experiment(Path(directory), [(100, 109), (110, 119)])
            make_experiment(Path(directory), [(120, 129)], streams=("PLA", "LSG"))
            decades, left_out = find_decades(experiment, "spinup")
        self.assertEqual([(d.first, d.last) for d in decades], [(100, 109), (110, 119)])
        self.assertEqual(decades[0].label_minus_model_year, 500)
        self.assertEqual(left_out[0]["first"], 120)
        self.assertIn("missing streams", left_out[0]["reason"])

    def test_a_gap_inside_the_record_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = make_experiment(Path(directory), [(100, 109), (120, 129)])
            with self.assertRaises(ValueError):
                find_decades(experiment, "spinup")

    def test_an_incomplete_block_before_a_complete_one_raises(self):
        with tempfile.TemporaryDirectory() as directory:
            experiment = make_experiment(Path(directory), [(110, 119)])
            make_experiment(Path(directory), [(100, 109)], streams=("PLA",))
            with self.assertRaises(ValueError):
                find_decades(experiment, "spinup")


class StatusTests(unittest.TestCase):
    def test_product_status_values(self):
        self.assertEqual(product_status(None, []), "missing")
        self.assertEqual(product_status({"decades_used": [[100, 109]]}, []), "up to date")
        self.assertEqual(product_status({"decades_used": [[100, 109]]}, [(110, 119)]), "new decades")

    def test_run_status_follows_the_output_folder(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = make_experiment(root, [(100, 109)])
            products = root / "products"
            self.assertEqual(run_status(experiment, "spinup", products)["status"], "missing")

            products.mkdir()
            manifest = {"decades_used": [[100, 109]]}
            (products / f"{product_stem(EXP, 'spinup')}_manifest.json").write_text(json.dumps(manifest))
            self.assertEqual(run_status(experiment, "spinup", products)["status"], "up to date")

            make_experiment(root, [(110, 119)], streams=("PLA", "OCE"))  # still being written
            self.assertEqual(run_status(experiment, "spinup", products)["status"], "up to date")

            make_experiment(root, [(110, 119)])
            status = run_status(experiment, "spinup", products)
            self.assertEqual((status["status"], status["new_decades"]), ("new decades", 1))


def load_script():
    """Import scripts/slurm/compute_plasim_indicators.py as a module."""
    spec = importlib.util.spec_from_file_location(
        "compute_plasim_indicators", ROOT / "scripts/slurm/compute_plasim_indicators.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_script(arguments: list[str]) -> tuple[int, str]:
    """Run the script's main() with the given arguments; return (exit code, stdout)."""
    output = io.StringIO()
    with patch.object(sys, "argv", ["compute_plasim_indicators.py", *arguments]), redirect_stdout(output):
        code = load_script().main()
    return code, output.getvalue()


class StatusCommandTests(unittest.TestCase):
    def test_status_lists_missing_and_stale_runs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment = make_experiment(root / "experiments", [(100, 109)])
            code, text = run_script(["status", "--experiments-root", str(root / "experiments"),
                                     "--output-root", str(root / "products")])
            self.assertEqual(code, 0)
            self.assertIn("missing", text)
            code, text = run_script(["status", "--stale", "--experiments-root", str(root / "experiments"),
                                     "--output-root", str(root / "products")])
            self.assertEqual(text.strip(), f"{experiment} spinup")


SAMPLE_FILES = [SAMPLE / f"{SAMPLE_RUN}_{stream}.{SAMPLE_LABELS}.nc" for stream in ("PLA", "OCE", "ICE", "LSG")]


@unittest.skipUnless(all(path.is_file() for path in SAMPLE_FILES) and SOURCE_BEKLAS.is_file(),
                     "sample decade not present")
class SampleDecadeTests(unittest.TestCase):
    """The sample decade must reproduce the numbers checked by hand (PLASIM_INFO/INDICATORS.md)."""

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        experiment = Path(cls.directory.name) / SAMPLE_RUN
        (experiment / "output" / "spinup").mkdir(parents=True)
        for path in SAMPLE_FILES:
            (experiment / "output" / "spinup" / path.name).symlink_to(path)
        cls.experiment = experiment
        cls.zonal, cls.sections, cls.record = run_indicators(experiment, "spinup", SOURCE_BEKLAS)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_energy_and_temperature(self):
        zonal = self.zonal
        incoming = area_mean(zonal.rst_zonal - zonal.rsut_zonal, zonal)
        np.testing.assert_allclose(incoming, 308.163, atol=1e-3)
        net = {region: float(area_mean(zonal.rst_zonal + zonal.rlut_zonal, zonal, region).mean())
               for region in ("global", "nh", "sh")}
        self.assertAlmostEqual(net["global"], -0.270, places=3)
        self.assertAlmostEqual(net["nh"], 2.434, places=3)
        self.assertAlmostEqual(net["sh"], -2.974, places=3)
        self.assertAlmostEqual(float(area_mean(zonal.tas_zonal, zonal).mean()), 260.00, places=2)

    def test_zonal_means_give_the_same_global_mean_as_the_map(self):
        with xr.open_dataset(SAMPLE_FILES[0], decode_times=False, engine="h5netcdf") as pla:
            tas = pla["tas"].values
        _, weight = t21_gaussian_grid()
        direct = (tas.mean(axis=2) * weight).sum(axis=1) / weight.sum()
        np.testing.assert_allclose(area_mean(self.zonal.tas_zonal, self.zonal), direct, rtol=1e-6)

    def test_sea_ice_salt_and_acc(self):
        zonal = self.zonal
        self.assertAlmostEqual(float(area_total(zonal.sic_ocean_zonal, zonal, "nh")[0]) / 1e12, 47.69, places=2)
        self.assertAlmostEqual(float(area_total(zonal.sic_ocean_zonal, zonal, "sh")[0]) / 1e12, 100.02, places=2)
        self.assertAlmostEqual(float(zonal.salt_content[0]) / 4.998625e19, 1.0, places=6)
        self.assertLess(self.record["checks"]["salt_content_relative_range"], 1e-6)
        self.assertAlmostEqual(float(zonal.acc_even_rows.mean()), 14.27, places=2)
        self.assertAlmostEqual(float(zonal.acc_odd_rows.mean()), 13.48, places=2)

    def test_ocean_bands_and_freshwater_balance(self):
        bands = [float(volume_mean(self.sections.t_section, self.sections, levels).isel(year=0)) - 273.15
                 for levels in (slice(0, 10), slice(10, 17), slice(17, 22))]
        np.testing.assert_allclose(bands, [1.484, -0.125, -0.481], atol=1e-3)
        self.assertLess(self.record["checks"]["freshwater_balance_max_abs_sv"], 0.1)

    def test_files_and_status(self):
        output = Path(self.directory.name) / "products"
        written = write_indicator_files(self.zonal, self.sections, self.record, output, "test")
        self.assertEqual([path.suffix for path in written], [".nc", ".nc", ".json"])
        self.assertFalse(any(output.glob("*.partial")))
        self.assertEqual(run_status(self.experiment, "spinup", output)["status"], "up to date")

    def test_run_command_writes_the_product(self):
        root = Path(self.directory.name) / "cli_products"
        code, text = run_script(["run", "--output-root", str(root), "--beklas", str(SOURCE_BEKLAS),
                                 str(self.experiment)])
        self.assertEqual(code, 0)
        summary = json.loads(text.strip().splitlines()[-1])
        self.assertEqual((summary["first_label"], summary["last_label"], summary["decades"]), (27490, 27499, 1))
        stem = product_stem(SAMPLE_RUN, "spinup")
        zonal = xr.load_dataset(root / SAMPLE_RUN / f"{stem}_zonal.nc", engine="h5netcdf")
        self.assertEqual(zonal.attrs["schema"], "plasim-indicators/1")
        self.assertEqual(int(zonal.model_year[0]), 27490 - 500)


if __name__ == "__main__":
    unittest.main()
