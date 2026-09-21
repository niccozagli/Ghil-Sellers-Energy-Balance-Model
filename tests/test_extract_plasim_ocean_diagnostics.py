"""Tests for the PLASIM-LSG ocean diagnostics extraction command."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from typer.testing import CliRunner


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "extract_plasim_ocean_diagnostics.py"
)
SPEC = importlib.util.spec_from_file_location(
    "extract_plasim_ocean_diagnostics", SCRIPT_PATH
)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def ocean_dataset() -> xr.Dataset:
    """Return a small LSG-like dataset covering all configured boxes."""
    latitude = np.array([[30.0, 30.0], [-45.0, -45.0]])
    longitude = np.array([[0.0, 5.0], [0.0, 5.0]])
    depth = np.array([25.0, 75.0, 125.0, 175.0, 225.0, 275.0, 350.0,
                      450.0, 550.0, 650.0, 750.0, 850.0, 950.0, 1100.0,
                      1300.0, 1500.0, 1800.0, 2250.0])
    interfaces = np.array([50.0, 100.0, 150.0, 200.0, 250.0, 312.5, 400.0,
                           500.0, 600.0, 700.0, 800.0, 900.0, 1025.0, 1200.0,
                           1400.0, 1650.0, 2025.0, 2500.0])
    shape = (1, depth.size, 2, 2)
    return xr.Dataset(
        {
            "t": (("time", "depth", "south_north", "west_east"),
                  np.full(shape, 280.0), {"units": "K"}),
            "wet": (("time", "depth", "south_north", "west_east"),
                    np.ones(shape)),
            "depp": (("time", "lev", "south_north", "west_east"),
                     np.full((1, 1, 2, 2), 2500.0)),
        },
        coords={
            "time": [0.0],
            "depth": depth,
            "depth_2": interfaces,
            "lev": [1.0],
            "lat": (("south_north", "west_east"), latitude),
            "lon": (("south_north", "west_east"), longitude),
        },
    )


class ExtractPlasimOceanDiagnosticsTest(unittest.TestCase):
    def test_writes_ocean_product_and_ignores_appledouble_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = root / "CONTROL_360ppm_T21L10_10000Y_MU_1240"
            source_dir = experiment / "output" / "spinup"
            source_dir.mkdir(parents=True)
            source_path = source_dir / f"{experiment.name}_LSG.2000-2000.nc"
            ocean_dataset().to_netcdf(source_path)
            (source_dir / f"._{source_path.name}").write_text("metadata")
            output_dir = root / "derived"

            result = CliRunner().invoke(
                extractor.app,
                [
                    "--experiment-dir", str(experiment),
                    "--output-dir", str(output_dir),
                    "--workers", "1",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            output_path = output_dir / f"{experiment.name}_spinup_ocean_diagnostics.nc"
            self.assertTrue(output_path.exists())
            with xr.open_dataset(output_path) as diagnostics:
                self.assertEqual(diagnostics.attrs["mu"], "1240")
                self.assertIn("ocean_heat_content", diagnostics)
                np.testing.assert_array_equal(diagnostics["year"], [2000])

            second_result = CliRunner().invoke(
                extractor.app,
                ["--experiment-dir", str(experiment), "--output-dir", str(output_dir)],
            )
            self.assertNotEqual(second_result.exit_code, 0)
            self.assertIn("Refusing to overwrite", str(second_result.exception))


if __name__ == "__main__":
    unittest.main()
