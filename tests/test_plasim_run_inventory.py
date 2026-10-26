"""Checks for the read-only PlaSim run inventory on a synthetic experiment."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import os
import struct
import unittest
from unittest.mock import patch

import h5netcdf
import numpy as np

from gsebm import plasim_run_inventory
from gsebm.plasim_run_inventory import (
    atmosphere_restart_date,
    inventory_run,
    latest_inventory,
    lsg_restart_date,
    write_inventory,
)

EXP = "CONTROL_TEST_MU_1245"
NTSPD = 32


def _record(data: bytes) -> bytes:
    head = struct.pack("<i", len(data))
    return head + data + head


def _write_rest(path: Path, end_of_year: int) -> None:
    nstep = (end_of_year + 1) * 360 * NTSPD
    path.write_bytes(_record(b"nstep".ljust(16)) + _record(struct.pack("<i", nstep))
                     + _record(b"naccuout".ljust(16)) + _record(struct.pack("<i", 31)))


def _write_lsgrest(path: Path, year: int, truncated: bool = False) -> None:
    nddr = [0] * 512
    nddr[10], nddr[11] = -1230, year
    path.write_bytes(_record(struct.pack("<512q", *nddr)) + (b"" if truncated else _record(b"\0" * 48)))


def _write_output(path: Path, internal_years: list[int]) -> None:
    with h5netcdf.File(path, "w") as file:
        file.dimensions["time"] = len(internal_years)
        time = file.create_variable("time", ("time",), "f8")
        time[:] = np.array(internal_years) * 1e4 + 630.5
        time.attrs["units"] = "day as %Y%m%d.%f"


class PlaSimRunInventoryTest(unittest.TestCase):
    def _experiment(self, root: Path) -> Path:
        """Labels 100-139; block 120-129 is labelled +500 relative to its model years."""
        exp = root / EXP
        for sub in ("run", "slurm", "initial_conditions", "output/spinup", "restart/spinup", "diag/spinup"):
            (exp / sub).mkdir(parents=True)
        (exp / "run/plasim_namelist").write_text(" MPSTEP      =    45\n N_DAYS_PER_YEAR=   360\n N_RUN_YEARS =     10\n")
        (exp / "run/planet_namelist").write_text(" GSOL0       =  1245.0\n")
        (exp / "slurm/run.slurm").write_text("FIRST_YEAR=100\nSEGMENT_YEARS=10\nN_SEGMENTS=4\n")
        _write_rest(exp / "initial_conditions/PARENT_REST.99.12", 99)
        _write_lsgrest(exp / "initial_conditions/PARENT_LSGREST.99.12", 99)
        model_first = {100: 100, 110: 110, 120: -380, 130: 130}
        for label, model in model_first.items():
            for stream in ("PLA", "OCE", "ICE", "LSG"):
                years = list(range(model, model + 10))
                if label == 130 and stream == "ICE":
                    years = years[:7]
                _write_output(exp / f"output/spinup/{EXP}_{stream}.{label}-{label + 9}.nc", years)
            if label != 110:
                _write_rest(exp / f"restart/spinup/{EXP}_REST.{label + 9}.12", model + 9)
                _write_lsgrest(exp / f"restart/spinup/{EXP}_LSGREST.{label + 9}.12", model + 9)
                (exp / f"restart/spinup/{EXP}_KLEISWI.{label + 9}.12").write_text("          84           0\n")
        _write_rest(exp / "run/plasim_restart", 139)
        _write_lsgrest(exp / "run/kleiin1", 139)
        (exp / "run/kleiswi").write_text("          84           0\n")
        return exp

    def test_restart_dates(self):
        with TemporaryDirectory() as tmp:
            rest, lsg = Path(tmp) / "rest", Path(tmp) / "lsg"
            _write_rest(rest, 9999)
            _write_lsgrest(lsg, 9999)
            date = atmosphere_restart_date(rest, NTSPD)
            self.assertEqual(date["nstep"], 115200000)
            self.assertEqual((date["year"], date["month"], date["day"], date["end_of_year"]), (10000, 1, 1, 9999))
            self.assertEqual(lsg_restart_date(lsg), {"year": 9999, "month": 12, "day": 30, "end_of_year": 9999})

    def test_inventory_finds_offsets_gaps_and_short_blocks(self):
        with TemporaryDirectory() as tmp:
            exp = self._experiment(Path(tmp))
            inventory = inventory_run(exp)
            summary = inventory["summary"]
            self.assertEqual(inventory["steps_per_day"], 32)
            self.assertEqual(summary["streams"]["PLA"]["label_ranges"], [[100, 139]])
            self.assertEqual(summary["streams"]["ICE"]["bad_blocks"], [f"{EXP}_ICE.130-139.nc: short"])
            self.assertEqual(summary["complete_label_years"], [[100, 129]])
            offsets = [(r["label_first"], r["label_last"], r["value"]["PLA"]) for r in summary["output_label_offsets"]]
            self.assertEqual(offsets, [(100, 119, 0), (120, 129, 500), (130, 139, 0)])
            restart = [(r["label_first"], r["label_last"], r["value"]) for r in summary["restart_label_offsets"]]
            self.assertEqual(restart, [(109, 109, {"rest": 0, "lsg": 0}), (129, 129, {"rest": 500, "lsg": 500}),
                                       (139, 139, {"rest": 0, "lsg": 0})])
            self.assertEqual(summary["restart_label_gaps"], [[119, 119]])
            self.assertEqual(summary["run_state_end_of_year"], {"REST": 139, "LSGREST": 139})
            self.assertEqual([r["date"]["end_of_year"] for r in inventory["initial_conditions"]], [99, 99])
            self.assertTrue(any("differ from model years" in p for p in summary["problems"]))

            path = write_inventory(inventory, Path(tmp) / "inventory")
            self.assertEqual(path.parent, Path(tmp) / "inventory" / EXP)
            self.assertEqual(json.loads(path.read_text())["summary"], json.loads(json.dumps(summary)))
            with self.assertRaises(FileExistsError):
                write_inventory(inventory, Path(tmp) / "inventory")

    def test_failure_signs(self):
        """Logged-but-missing decades, DIAG without output, truncated run/ state, empty phase."""
        with TemporaryDirectory() as tmp:
            exp = self._experiment(Path(tmp))
            log = "".join(f"Starting PlaSim segment: {a}-{a + 9}\nCompleted {a}-{a + 9} in 900 seconds\n"
                          for a in range(100, 150, 10))
            (exp / "diag/spinup/slurm-1.out").write_text(log)
            (exp / f"diag/spinup/{EXP}_DIAG.140-149.txt").write_text("diag\n")
            _write_lsgrest(exp / "run/kleiin1", 139, truncated=True)
            summary = inventory_run(exp)["summary"]
            self.assertEqual(summary["logged_completed_without_output"], [[130, 149]])
            self.assertEqual(summary["diag_without_output"], [[140, 149]])
            self.assertTrue(any(p.startswith("run/ kleiin1 is") for p in summary["problems"]))

            (exp / "output/stationary").mkdir()
            empty = inventory_run(exp, phase="stationary")
            self.assertTrue(any("phase stationary is empty" in p for p in empty["summary"]["problems"]))

    def test_compare_and_reuse(self):
        with TemporaryDirectory() as tmp:
            exp = self._experiment(Path(tmp))
            root = Path(tmp) / "inventory"
            first = inventory_run(exp)
            write_inventory(first, root)

            for stream in ("PLA", "OCE", "ICE", "LSG"):
                _write_output(exp / f"output/spinup/{EXP}_{stream}.140-149.nc", list(range(140, 150)))
            changed = exp / f"output/spinup/{EXP}_PLA.100-109.nc"
            _write_output(changed, list(range(100, 109)))
            os.utime(changed, (1e9, 1e9))
            (exp / f"restart/spinup/{EXP}_KLEISWI.109.12").unlink()

            path, previous = latest_inventory(root, EXP, "spinup")
            with patch.object(plasim_run_inventory, "output_block", wraps=plasim_run_inventory.output_block) as reader:
                second = inventory_run(exp, previous=previous)
            self.assertEqual(reader.call_count, 5)  # four new blocks and the changed one
            changes = second["changes_since"]
            self.assertEqual(len(changes["output"]["new"]), 4)
            self.assertEqual(changes["output"]["changed"], [changed.name])
            self.assertEqual(changes["restart"]["removed"], [f"{EXP}_KLEISWI.109.12"])
            self.assertFalse(changes["namelists_changed"])
            problems = second["summary"]["problems"]
            self.assertTrue(any("1 output file(s) changed" in p for p in problems))
            self.assertTrue(any("1 restart file(s) removed" in p for p in problems))
            unchanged = f"{EXP}_OCE.110-119.nc"
            row = lambda inv: next(r for r in inv["outputs"] if r["file"] == unchanged)
            self.assertEqual(row(second), row(first))

            with patch.object(plasim_run_inventory, "output_block", wraps=plasim_run_inventory.output_block) as reader:
                inventory_run(exp, previous=previous, reuse=False)
            self.assertEqual(reader.call_count, 20)


if __name__ == "__main__":
    unittest.main()
