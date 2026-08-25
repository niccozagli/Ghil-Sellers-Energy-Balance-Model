"""Tests for PLASIM zonal climate-diagnostics extraction."""

import importlib.util
import sys
import tempfile
import unittest

from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "cluster" / "extract_plasim_zonal_temperature.py"
SPEC = importlib.util.spec_from_file_location("extract_plasim_zonal_temperature", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
extractor = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extractor
SPEC.loader.exec_module(extractor)


def source_dataset(offset: float = 0.0, *, mask: np.ndarray | None = None) -> xr.Dataset:
    """Return a compact monthly PLASIM-like source dataset."""

    time = np.array([10.0 + offset, 20.0 + offset])
    lat = np.array([-45.0, 45.0])
    lon = np.array([0.0, 120.0, 240.0])
    base = np.arange(12, dtype=float).reshape(2, 2, 3) + offset
    if mask is None:
        mask = np.broadcast_to(np.array([[0, 1, 0], [1, 1, 1]]), base.shape).copy()
    fields = {
        "tas": base + 270,
        "ts": base + 260,
        "lsm": mask,
        "as": base / 100,
        "sic": base / 10,
        "sit": base / 20,
        "rst": base + 100,
        "rsut": -(base + 20),
        "rlut": -(base + 50),
    }
    dataset = xr.Dataset(
        {name: (("time", "lat", "lon"), values) for name, values in fields.items()},
        coords={"time": time, "lat": lat, "lon": lon},
        attrs={"title": "synthetic PLASIM"},
    )
    dataset["tas"].attrs["units"] = "K"
    dataset["rst"].attrs["units"] = "W m-2"
    dataset["time"].attrs["units"] = "days since 2000-01-01"
    return dataset


class ZonalClimateExtractorTest(unittest.TestCase):
    def _result(self, path: Path, *, reference: bool = True):
        return extractor._extract_zonal_climate(
            extractor.InputFile(path, 2000, 12, is_reference=reference)
        )

    def test_extracts_aligned_means_fluxes_and_provenance(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "CONTROL_360ppm_PLA.2000.12.nc"
            source_dataset().to_netcdf(path, engine="h5netcdf")
            output = extractor._output_dataset([self._result(path)], 360.0, 1365.0)

        np.testing.assert_allclose(output["zonal_T"], [[271, 274], [277, 280]])
        np.testing.assert_allclose(output["zonal_surface_temperature"], [[261, 264], [267, 270]])
        np.testing.assert_allclose(output["ocean_zonal_T"].isel(lat=0), [271, 277])
        self.assertTrue(np.isnan(output["ocean_zonal_T"].isel(lat=1)).all())
        np.testing.assert_allclose(output["ocean_zonal_sea_ice_cover"].isel(lat=0), [0.1, 0.7])
        np.testing.assert_allclose(output["ocean_zonal_sea_ice_thickness"].isel(lat=0), [0.05, 0.35])
        np.testing.assert_allclose(output["zonal_surface_albedo"], [[0.01, 0.04], [0.07, 0.10]])
        np.testing.assert_allclose(output["zonal_toa_net_shortwave"], 80.0)
        np.testing.assert_allclose(output["zonal_toa_outgoing_longwave"], [[51, 54], [57, 60]])
        np.testing.assert_allclose(output["zonal_toa_net_radiation"], [[29, 26], [23, 20]])
        np.testing.assert_allclose(output["ocean_fraction"], [2 / 3, 0])
        np.testing.assert_array_equal(output["source_year"], [2000, 2000])
        np.testing.assert_array_equal(output["source_month"], [12, 12])
        np.testing.assert_array_equal(output["source_file"], ["CONTROL_360ppm_PLA.2000.12.nc"] * 2)
        self.assertEqual(output["zonal_T"].attrs["units"], "K")
        self.assertIn("flux_sign_convention", output["zonal_toa_net_radiation"].attrs)
        self.assertEqual(
            set(output.data_vars),
            {
                "zonal_T", "zonal_surface_temperature", "ocean_zonal_T",
                "ocean_zonal_sea_ice_cover", "ocean_zonal_sea_ice_thickness",
                "zonal_surface_albedo", "zonal_toa_net_shortwave",
                "zonal_toa_outgoing_longwave", "zonal_toa_net_radiation", "ocean_fraction",
            },
        )

    def test_rejects_missing_variable_and_time_varying_mask(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            missing_path = directory / "CONTROL_360ppm_PLA.2000.12.nc"
            source_dataset().drop_vars("sit").to_netcdf(missing_path, engine="h5netcdf")
            with self.assertRaisesRegex(ValueError, "missing required variable 'sit'"):
                self._result(missing_path)

            mask = np.broadcast_to(np.array([[0, 1, 0], [1, 1, 1]]), (2, 2, 3)).copy()
            mask[1, 0, 0] = 1
            mask_path = directory / "CONTROL_360ppm_PLA.2001.01.nc"
            source_dataset(mask=mask).to_netcdf(mask_path, engine="h5netcdf")
            with self.assertRaisesRegex(ValueError, "time-invariant"):
                self._result(mask_path)

    def test_rejects_inconsistent_monthly_masks_and_grids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            directory = Path(temporary_directory)
            first = directory / "CONTROL_360ppm_PLA.2000.12.nc"
            second = directory / "CONTROL_360ppm_PLA.2001.01.nc"
            source_dataset().to_netcdf(first, engine="h5netcdf")
            changed_mask = np.broadcast_to(np.array([[1, 1, 0], [1, 1, 1]]), (2, 2, 3)).copy()
            source_dataset(mask=changed_mask).to_netcdf(second, engine="h5netcdf")
            with self.assertRaisesRegex(ValueError, "lsm differs"):
                extractor._output_dataset([self._result(first), self._result(second, reference=False)], 360, 1365)

            shifted_grid = source_dataset().assign_coords(lon=[1.0, 121.0, 241.0])
            shifted_grid.to_netcdf(second, engine="h5netcdf")
            with self.assertRaisesRegex(ValueError, "longitude grid differs"):
                extractor._output_dataset([self._result(first), self._result(second, reference=False)], 360, 1365)
