"""Focused checks for the annual PlaSim map archive."""

from pathlib import Path
from tempfile import TemporaryDirectory
import importlib.util
import json
import os
import shutil
import unittest
from unittest.mock import patch

import numpy as np
import xarray as xr
from typer.testing import CliRunner

from gsebm.plasim_raw_maps import (
    DIRECT_FIELDS,
    _declared_years,
    _direct_fields,
    _source_time,
    _make_basin_masks,
    inventory_experiment,
    plan_extraction,
    extract_raw_maps,
)
import h5netcdf
import h5py


class PlaSimRawMapTest(unittest.TestCase):
    def _synthetic_direct_sources(self, root: Path):
        years = np.array([600, 601], dtype=np.int32)
        times = np.array([1000630.5, 1010630.5])
        bounds = np.array([[1000101., 1001230.], [1010101., 1011230.]])
        geometry = {
            "t21_lat": (("t21_lat",), np.array([-30., 30.])),
            "t21_lon": (("t21_lon",), np.array([0., 180., 270.])),
            "lsm": (("t21_lat", "t21_lon"), np.zeros((2, 3))),
            "wet": (("lsg_depth", "south_north", "west_east"), np.ones((22, 3, 4), dtype=np.int8)),
            "wetvec": (("lsg_depth", "south_north", "west_east"), np.ones((22, 3, 4), dtype=np.int8)),
            "bathymetry": (("south_north", "west_east"), np.full((3, 4), 6000.)),
            "vector_bathymetry": (("south_north", "west_east"), np.full((3, 4), 6000.)),
            "lsg_interface_wet_area": (("lsg_depth_interface", "south_north", "west_east"), np.ones((22, 3, 4))),
            "lsg_depth": (("lsg_depth",), np.arange(22, dtype=float)),
            "lsg_depth_interface": (("lsg_depth_interface",), np.arange(22, dtype=float)),
            "lsg_convection_depth": (("lsg_convection_depth",), np.arange(21, dtype=float)),
            "lat": (("south_north", "west_east"), np.zeros((3, 4))),
            "lon": (("south_north", "west_east"), np.zeros((3, 4))),
            "lat_2": (("south_north", "west_east"), np.zeros((3, 4))),
            "lon_2": (("south_north", "west_east"), np.zeros((3, 4))),
            "pla_model_level": (("pla_model_level",), np.arange(10, dtype=float)),
            "pla_hyam": (("pla_model_level",), np.arange(10, dtype=float)),
            "pla_hybm": (("pla_model_level",), np.arange(10, dtype=float)),
            "pla_hyai": (("pla_model_interface",), np.arange(11, dtype=float)),
            "pla_hybi": (("pla_model_interface",), np.arange(11, dtype=float)),
        }
        geometry["wet"][1][:, -1, :] = 0
        geometry["wetvec"][1][:, -1, :] = 0
        geometry["lsg_interface_wet_area"][1][:, -1, :] = 0
        paths = []
        for component in ("lsg", "pla", "ice", "oce"):
            variables = {"time_bnds": (("time", "bnds"), bounds)}
            if component == "lsg":
                variables.update(wet=(("time", "depth", "south_north", "west_east"), np.broadcast_to(geometry["wet"][1], (2, 22, 3, 4))),
                                 wetvec=(("time", "depth", "south_north", "west_east"), np.broadcast_to(geometry["wetvec"][1], (2, 22, 3, 4))),
                                 depp=(("time", "lev", "south_north", "west_east"), np.broadcast_to(geometry["bathymetry"][1], (2, 1, 3, 4))),
                                 depv=(("time", "lev", "south_north", "west_east"), np.broadcast_to(geometry["vector_bathymetry"][1], (2, 1, 3, 4))))
                for name in ("lat", "lon", "lat_2", "lon_2"):
                    variables[name] = (geometry[name][0], geometry[name][1])
            if component in ("ice", "oce"):
                variables["ls"] = (("time", "lat", "lon"), np.zeros((2, 2, 3), dtype="float32"))
            for name in DIRECT_FIELDS[component]:
                if component == "lsg":
                    if name in ("t", "s", "utot", "vtot"):
                        dims, shape = ("time", "depth", "south_north", "west_east"), (2, 22, 3, 4)
                    elif name == "w":
                        dims, shape = ("time", "depth_2", "south_north", "west_east"), (2, 22, 3, 4)
                    elif name == "convad":
                        dims, shape = ("time", "depth_3", "south_north", "west_east"), (2, 21, 3, 4)
                    else:
                        dims, shape = ("time", "lev", "south_north", "west_east"), (2, 1, 3, 4)
                elif name in ("cl", "clw"):
                    dims, shape = ("time", "sfc", "lat", "lon"), (2, 10, 2, 3)
                else:
                    dims, shape = ("time", "lat", "lon"), (2, 2, 3)
                values = np.full(shape, -2.5 if name in ("heata", "ofluxa") else 1.25, dtype="float32")
                if component == "lsg":
                    values[..., -1, :] = np.nan
                variables[name] = (dims, values, {"units": "m2/s" if name in ("dssta", "qhda") else "1"})
            coords = {"time": ("time", times, {"units": "day as %Y%m%d.%f", "bounds": "time_bnds"})}
            if component == "lsg":
                coords.update(depth=("depth", geometry["lsg_depth"][1]),
                              depth_2=("depth_2", geometry["lsg_depth_interface"][1]),
                              depth_3=("depth_3", geometry["lsg_convection_depth"][1]))
            if component == "pla":
                coords.update(sfc=("sfc", geometry["pla_model_level"][1]),
                              hyam=("sfc", geometry["pla_hyam"][1]),
                              hybm=("sfc", geometry["pla_hybm"][1]),
                              hyai=("nhyi", geometry["pla_hyai"][1]),
                              hybi=("nhyi", geometry["pla_hybi"][1]))
            if component in ("pla", "ice", "oce"):
                coords.update(lat=("lat", geometry["t21_lat"][1]), lon=("lon", geometry["t21_lon"][1]))
            dataset = xr.Dataset(variables, coords=coords)
            path = root / f"run_{component.upper()}.600-601.nc"
            dataset.to_netcdf(path, engine="h5netcdf")
            paths.append(path)
        return tuple(paths), years, geometry

    def test_direct_fields_copy_full_depth_units_and_dry_values(self) -> None:
        with TemporaryDirectory() as temporary:
            paths, years, geometry = self._synthetic_direct_sources(Path(temporary))
            native = {name: np.asarray(xr.open_dataset(paths[0], decode_times=False)[name[4:]]) for name in ("lsg_t", "lsg_s")}
            block = _direct_fields(paths, years, geometry, native)
            self.assertEqual(sum(len(names) for names in DIRECT_FIELDS.values()), 72)
            self.assertEqual(block["lsg_t"][1].shape, (2, 22, 3, 4))
            self.assertEqual(block["lsg_w"][1].shape, (2, 22, 3, 4))
            self.assertEqual(block["lsg_convad"][1].shape, (2, 21, 3, 4))
            self.assertTrue(np.isnan(block["lsg_t"][1][0, 0, -1, 0]))
            self.assertEqual(block["oce_heata"][1][0, 0, 0], -2.5)
            self.assertEqual(block["oce_dssta"][2]["source_units"], "m2/s")
            self.assertEqual(block["oce_dssta"][2]["units"], "W m-2")
            self.assertEqual(block["ice_stoia"][2]["units"], "m s-1")
            self.assertNotIn("units", block["lsg_fldice"][2])
            with h5py.File(paths[0], "r+") as file:
                file["utot"][0, 0, 0, 0] = np.nan
            with self.assertRaisesRegex(ValueError, "Non-finite physical lsg_utot"):
                _direct_fields(paths, years, geometry, native)
    def test_inventory_requires_matching_contiguous_source_blocks(self) -> None:
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "CONTROL_360ppm_T21L10_10000Y_MU_1240"
            source = experiment / "output" / "spinup"
            source.mkdir(parents=True)
            for kind in ("LSG", "PLA", "ICE", "OCE"):
                for first in (5000, 5010):
                    (source / f"{experiment.name}_{kind}.{first}-{first + 9}.nc").touch()
            inventory = inventory_experiment(experiment)
            self.assertEqual((inventory.first_year, inventory.last_year), (5000, 5019))
            self.assertEqual(len(inventory.ice_paths), 2)
            (source / f"{experiment.name}_PLA.5010-5019.nc").rename(source / f"{experiment.name}_PLA.5020-5029.nc")
            with self.assertRaisesRegex(ValueError, "year coverage differs"):
                inventory_experiment(experiment)

    def test_inventory_rejects_partial_duplicate_and_gap(self) -> None:
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "run"
            source = experiment / "output" / "spinup"
            source.mkdir(parents=True)
            for component in ("LSG", "PLA", "ICE"):
                (source / f"run_{component}.10-19.nc").touch()
            with self.assertRaisesRegex(ValueError, "Missing OCE"):
                inventory_experiment(experiment)
            (source / "run_OCE.10-19.nc").touch()
            self.assertEqual(inventory_experiment(experiment).year_count, 10)
            (source / "run_LSG.010-019.nc").touch()
            with self.assertRaisesRegex(ValueError, "coverage differs"):
                inventory_experiment(experiment)
            (source / "run_LSG.010-019.nc").unlink()
            (source / "run_LSG.10-20.nc").touch()
            with self.assertRaisesRegex(ValueError, "coverage differs"):
                inventory_experiment(experiment)
            (source / "run_LSG.10-20.nc").unlink()
            for component in ("LSG", "PLA", "ICE", "OCE"):
                (source / f"run_{component}.21-29.nc").touch()
            with self.assertRaisesRegex(ValueError, "Missing or overlapping"):
                inventory_experiment(experiment)
            with self.assertRaisesRegex(ValueError, "Reversed"):
                _declared_years(source / "run_LSG.20-10.nc")

    def test_source_time_retains_declared_offset_and_bounds(self) -> None:
        years = np.arange(27490, 27492, dtype=np.int32)
        values = np.array([269900630.5, 269910630.5])
        bounds = np.array([[269900101., 269901230.], [269910101., 269911230.]])
        source = xr.Dataset({"time_bnds": (("time", "bnds"), bounds)},
                            coords={"time": ("time", values, {"units": "day as %Y%m%d.%f", "calendar": "proleptic_gregorian", "bounds": "time_bnds"})})
        result = _source_time(source, "ice", years)
        np.testing.assert_array_equal(result["source_internal_year_ice"][1], [26990, 26991])
        np.testing.assert_array_equal(result["source_declared_minus_internal_year_ice"][1], [500, 500])
        np.testing.assert_array_equal(result["source_time_bounds_ice"][1], bounds)
        source.time.attrs["units"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            _source_time(source, "ice", years)

    def test_v2_archive_requires_explicit_migration(self) -> None:
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "run"
            source = experiment / "output" / "spinup"
            source.mkdir(parents=True)
            for component in ("LSG", "PLA", "ICE", "OCE"):
                (source / f"run_{component}.10-19.nc").touch()
            inventory = inventory_experiment(experiment)
            archive = Path(temporary) / "run_spinup_raw_maps.nc"
            mask = Path(temporary) / "run_spinup_basin_masks.nc"
            with h5netcdf.File(archive, "w") as file:
                file.attrs["schema_version"] = "2"
            with h5netcdf.File(mask, "w") as file:
                file.attrs["schema_version"] = "2"
            with self.assertRaisesRegex(ValueError, "incompatible schema"):
                plan_extraction(inventory, archive, mask)
            self.assertEqual(plan_extraction(inventory, archive, mask, refresh=True).action, "rebuild")

    def test_writer_build_append_skip_and_dirty_tail_recovery(self) -> None:
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "run"
            source = experiment / "output" / "spinup"
            source.mkdir(parents=True)
            first_paths, _, geometry = self._synthetic_direct_sources(source)
            geometry["upper_depth"] = (("upper_depth",), np.arange(13.))
            geometry["lsg_lat"] = (("lsg_lat",), np.arange(3.))
            geometry["lsg_vector_lat"] = (("lsg_vector_lat",), np.arange(3.))
            for component in ("ice", "oce"):
                geometry[f"{component}_ls"] = (("t21_lat", "t21_lon"), np.zeros((2, 3), dtype="float32"))

            def block(lsg, pla, ice, oce, geometry):
                years = _declared_years(lsg)
                fields = {"surface_temperature": (("year", "t21_lat", "t21_lon"),
                                                  np.full((len(years), 2, 3), years[0], dtype="float32"), {})}
                for band in ("0_100m", "150_300m"):
                    for component in ("u", "v"):
                        fields[f"{component}_layer_{band}"] = (("year", "south_north", "west_east"),
                                                                np.ones((len(years), 3, 4), dtype="float32"), {})
                for kind in ("shortwave", "longwave"):
                    present = int(years[0] >= 602)
                    fields[f"toa_clear_sky_{kind}"] = (("year", "t21_lat", "t21_lon"),
                                                       np.full((len(years), 2, 3), 0.0 if present else np.nan), {})
                    fields[f"toa_clear_sky_{kind}_available"] = (("year",), np.full(len(years), present, dtype="int8"), {})
                for component in ("lsg", "pla", "ice", "oce"):
                    fields[f"source_time_{component}"] = (("year",), years.astype("float64"),
                                                          {"source_units": "day as %Y%m%d.%f", "source_calendar": "", "source_bounds_variable": "time_bnds"})
                return fields, {}

            destination = Path(temporary) / "archives" / "run"
            with patch("gsebm.plasim_raw_maps._static_geometry", return_value=geometry), patch("gsebm.plasim_raw_maps._annual_block", side_effect=block):
                first = inventory_experiment(experiment)
                archive, mask = extract_raw_maps(first, destination, workers=1)
                self.assertTrue(mask.exists())
                with h5netcdf.File(archive) as file:
                    self.assertEqual(file.attrs["committed_blocks"], 1)
                    self.assertEqual(file.dimensions["year"].size, 2)
                for path in first_paths:
                    shutil.copyfile(path, source / path.name.replace("600-601", "602-603"))
                inventory = inventory_experiment(experiment)
                self.assertEqual(plan_extraction(inventory, archive, mask).action, "append")
                extract_raw_maps(inventory, destination, workers=1)
                self.assertEqual(plan_extraction(inventory, archive, mask).action, "skip")
                with h5netcdf.File(archive) as file:
                    self.assertTrue(np.isnan(file.variables["toa_clear_sky_shortwave"][0]).all())
                    self.assertEqual(file.variables["toa_clear_sky_shortwave_available"][2], 1)
                    self.assertTrue(np.all(file.variables["toa_clear_sky_shortwave"][2] == 0))
                with h5netcdf.File(archive, "a") as file:
                    file.resize_dimension("year", 5)
                    file.resize_dimension("source_block", 3)
                    file.attrs["missing_optional_pla_maps"] = "[]"
                self.assertEqual(plan_extraction(inventory, archive, mask).action, "recover")
                extract_raw_maps(inventory, destination, workers=1)
                with h5netcdf.File(archive) as file:
                    self.assertEqual(file.dimensions["year"].size, 4)
                    self.assertEqual(file.dimensions["source_block"].size, 2)
                    self.assertEqual(file.attrs["committed_last_year"], 603)
                    self.assertEqual(json.loads(file.attrs["missing_optional_pla_maps"]),
                                     ["toa_clear_sky_shortwave", "toa_clear_sky_longwave"])
                    self.assertEqual(file.variables["u_mean_0_100m"][0, 0], 1.0)
                with h5netcdf.File(mask, "a") as file:
                    file.variables["t21_south_atlantic"][:] = np.ones((2, 3), dtype="int8")
                extract_raw_maps(inventory, destination, workers=1, refresh=True)
                with h5netcdf.File(mask) as file:
                    np.testing.assert_array_equal(file.variables["t21_south_atlantic"][:], 1)
                for source_path in (inventory.ice_paths[0], inventory.oce_paths[0]):
                    stat = source_path.stat()
                    os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
                    with self.assertRaisesRegex(ValueError, "Previously extracted source block changed"):
                        plan_extraction(inventory, archive, mask)
                    os.utime(source_path, ns=(stat.st_atime_ns, stat.st_mtime_ns))

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

    def test_cli_incomplete_quartet_fails_in_dry_run(self) -> None:
        spec = importlib.util.spec_from_file_location(
            "extract_plasim_raw_maps",
            Path(__file__).resolve().parents[1] / "scripts" / "extract_plasim_raw_maps.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with TemporaryDirectory() as temporary:
            experiment = Path(temporary) / "run"
            source = experiment / "output" / "spinup"
            source.mkdir(parents=True)
            for component in ("PLA", "LSG", "ICE"):
                (source / f"run_{component}.10-19.nc").touch()
            output = Path(temporary) / "archives"
            result = CliRunner().invoke(module.app, ["--experiment-dir", str(experiment), "--output-root", str(output), "--dry-run"])
            self.assertEqual(result.exit_code, 1, result.output)
            self.assertIn("Missing OCE", result.output)
            self.assertFalse(output.exists())

if __name__ == "__main__":
    unittest.main()
