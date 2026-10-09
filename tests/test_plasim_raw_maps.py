"""Focused checks for the annual PlaSim map archive."""

from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import json
import unittest

import numpy as np
import xarray as xr
from typer.testing import CliRunner

from gsebm.plasim_raw_maps import (
    _make_basin_masks,
    inventory_experiment,
)


class PlaSimRawMapTest(unittest.TestCase):
    def test_inventory_requires_matching_contiguous_source_blocks(self) -> None:
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "CONTROL_360ppm_T21L10_10000Y_MU_1240"
            source = experiment / "output" / "spinup"
            source.mkdir(parents=True)
            for kind in ("LSG", "PLA"):
                for first in (5000, 5010):
                    (source / f"run_{kind}.{first}-{first + 9}.nc").touch()
            inventory = inventory_experiment(experiment)
            self.assertEqual((inventory.first_year, inventory.last_year), (5000, 5019))
            (source / "run_PLA.5010-5019.nc").rename(source / "run_PLA.5020-5029.nc")
            with self.assertRaisesRegex(ValueError, "year coverage differs"):
                inventory_experiment(experiment)

    def test_basin_mask_respects_wrapped_longitude_and_wet_points(self) -> None:
        with TemporaryDirectory() as temporary:
            geometry = {
                "lat": (("south_north", "west_east"), np.array([[-30.0, -30.0], [30.0, -30.0]])),
                "lon": (("south_north", "west_east"), np.array([[300.0, 30.0], [300.0, 350.0]])),
                "lat_2": (("south_north", "west_east"), np.array([[-30.0, -30.0], [30.0, -30.0]])),
                "lon_2": (("south_north", "west_east"), np.array([[300.0, 30.0], [300.0, 350.0]])),
                "wet": (("lsg_depth", "south_north", "west_east"), np.array([[[1, 1], [1, 0]]], dtype=np.int8)),
                "wetvec": (("lsg_depth", "south_north", "west_east"), np.array([[[1, 0], [1, 1]]], dtype=np.int8)),
                "t21_lat": (("t21_lat",), np.array([-30.0, 30.0])),
                "t21_lon": (("t21_lon",), np.array([300.0, 350.0])),
                "lsm": (("t21_lat", "t21_lon"), np.array([[0.0, 1.0], [0.0, 0.0]])),
            }
            path = Path(temporary) / "masks.nc"
            _make_basin_masks(path, geometry, "synthetic")
            with xr.open_dataset(path, engine="h5netcdf") as result:
                np.testing.assert_array_equal(
                    result["lsg_scalar_south_atlantic"], [[1, 0], [0, 0]]
                )
                np.testing.assert_array_equal(
                    result["lsg_vector_south_atlantic"], [[1, 0], [0, 1]]
                )
                np.testing.assert_array_equal(
                    result["t21_south_atlantic"], [[1, 0], [0, 0]]
                )

    def test_cli_writes_report_to_requested_path(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "extract_plasim_raw_maps",
            Path(__file__).resolve().parents[1] / "scripts" / "extract_plasim_raw_maps.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "CONTROL_360ppm_T21L10_10000Y_MU_1240"
            experiment.mkdir()
            report = Path(temporary) / "reports" / "mu1240.json"
            result = CliRunner().invoke(module.app, [
                "--experiment-dir", str(experiment),
                "--output-root", str(Path(temporary) / "archives"),
                "--report-path", str(report),
            ])
            self.assertEqual(result.exit_code, 0, result.output)
            entries = json.loads(report.read_text())
            self.assertEqual(entries[0]["action"], "skip_missing_source")
            self.assertFalse((Path(temporary) / "archives" / "raw_map_extraction_inventory.json").exists())

if __name__ == "__main__":
    unittest.main()
