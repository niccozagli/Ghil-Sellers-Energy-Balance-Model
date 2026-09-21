"""Tests for the PLASIM diagnostics extraction command."""

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from typer.testing import CliRunner


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "extract_plasim_diagnostics.py"
SPEC = importlib.util.spec_from_file_location("extract_plasim_diagnostics", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def atmospheric_dataset(year: int) -> xr.Dataset:
    """Return one compact annual PLASIM-like atmospheric output dataset."""
    nodes, _ = np.polynomial.legendre.leggauss(2)
    latitude = np.degrees(np.arcsin(nodes))[::-1]
    longitude = np.array([0.0, 180.0])
    values = np.arange(4, dtype=float).reshape(1, 2, 2)
    return xr.Dataset(
        {
            "ts": (("time", "lat", "lon"), 260.0 + values),
            "tas": (("time", "lat", "lon"), 265.0 + values),
            "rst": (("time", "lat", "lon"), 210.0 + values),
            "rsut": (("time", "lat", "lon"), -(90.0 + values)),
            "rlut": (("time", "lat", "lon"), -(205.0 + values)),
            "lsm": (("time", "lat", "lon"), np.zeros_like(values)),
            "as": (("time", "lat", "lon"), 0.1 + 0.01 * values),
            "sic": (("time", "lat", "lon"), np.zeros_like(values)),
            "sit": (("time", "lat", "lon"), np.zeros_like(values)),
        },
        coords={"time": [year * 10_000 + 101.5], "lat": latitude, "lon": longitude},
    )


def diagnostic_text(year: int) -> str:
    """Return one complete annual LSG diagnostic record."""
    return f"""\
* LSG timestep 1 date: 10-Jan-{year} *
ATL max (NADW) : 10.0 20.0 30.0
Icevol. m**3 0.100E+15 icecov.area m**2 0.200E+14 Av.thickness in m 5.0
Iceareas: 18.0 2.0 90.0 10.0
"""


class ExtractPlasimDiagnosticsTest(unittest.TestCase):
    def test_writes_both_datasets_and_ignores_appledouble_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = root / "CONTROL_360ppm_T21L10_10000Y_MU_1312"
            atmospheric_dir = experiment / "output" / "spinup"
            diagnostic_dir = experiment / "diag" / "spinup"
            atmospheric_dir.mkdir(parents=True)
            diagnostic_dir.mkdir(parents=True)
            atmospheric_path = atmospheric_dir / (
                "CONTROL_360ppm_T21L10_10000Y_MU_1312_PLA.2000-2000.nc"
            )
            atmospheric_dataset(2000).to_netcdf(atmospheric_path, engine="scipy")
            (atmospheric_dir / f"._{atmospheric_path.name}").write_text("metadata")
            diagnostic_path = diagnostic_dir / (
                "CONTROL_360ppm_T21L10_10000Y_MU_1312_DIAG.2000-2000.txt"
            )
            diagnostic_path.write_text(diagnostic_text(2000), encoding="utf-8")
            (diagnostic_dir / f"._{diagnostic_path.name}").write_text("metadata")
            output_dir = root / "derived"

            result = CliRunner().invoke(
                extractor.app,
                [
                    "--experiment-dir",
                    str(experiment),
                    "--output-dir",
                    str(output_dir),
                    "--workers",
                    "1",
                ],
            )

            self.assertEqual(result.exit_code, 0, result.output)
            diagnostics_path = output_dir / (
                "CONTROL_360ppm_T21L10_10000Y_MU_1312_spinup_diagnostics.nc"
            )
            zonal_path = output_dir / (
                "CONTROL_360ppm_T21L10_10000Y_MU_1312_spinup_zonal_temperatures.nc"
            )
            self.assertTrue(diagnostics_path.exists())
            self.assertTrue(zonal_path.exists())
            with xr.open_dataset(diagnostics_path) as diagnostics:
                np.testing.assert_array_equal(diagnostics["year"], [2000])
                self.assertIn("amoc_strength", diagnostics)
                self.assertEqual(diagnostics.attrs["mu"], "1312")
            with xr.open_dataset(zonal_path) as zonal_temperatures:
                self.assertEqual(
                    set(zonal_temperatures.data_vars),
                    {
                        "zonal_surface_temperature",
                        "zonal_2m_temperature",
                        "zonal_sea_ice_concentration",
                        "zonal_persistent_sea_ice_fraction",
                        "zonal_sea_ice_thickness",
                        "zonal_surface_albedo",
                        "zonal_toa_energy_imbalance",
                    },
                )

    def test_refuses_to_overwrite_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            experiment = root / "CONTROL_360ppm_T21L10_10000Y_MU_1312"
            atmospheric_dir = experiment / "output" / "spinup"
            diagnostic_dir = experiment / "diag" / "spinup"
            atmospheric_dir.mkdir(parents=True)
            diagnostic_dir.mkdir(parents=True)
            atmospheric_dataset(2000).to_netcdf(
                atmospheric_dir / "CONTROL_360ppm_T21L10_10000Y_MU_1312_PLA.2000-2000.nc",
                engine="scipy",
            )
            (diagnostic_dir / "CONTROL_360ppm_T21L10_10000Y_MU_1312_DIAG.2000-2000.txt").write_text(
                diagnostic_text(2000), encoding="utf-8"
            )
            output_dir = root / "derived"
            output_dir.mkdir()
            existing = output_dir / (
                "CONTROL_360ppm_T21L10_10000Y_MU_1312_spinup_diagnostics.nc"
            )
            existing.touch()

            result = CliRunner().invoke(
                extractor.app,
                ["--experiment-dir", str(experiment), "--output-dir", str(output_dir)],
            )

            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("Refusing to overwrite", str(result.exception))


if __name__ == "__main__":
    unittest.main()
