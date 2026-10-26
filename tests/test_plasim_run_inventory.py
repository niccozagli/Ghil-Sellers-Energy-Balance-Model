"""Checks for the read-only PlaSim run inventory on a synthetic experiment."""

from pathlib import Path
from tempfile import TemporaryDirectory
import json
import struct
import unittest

import h5netcdf
import numpy as np

from gsebm.plasim_run_inventory import (
    atmosphere_restart_date,
    inventory_run,
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


def _write_lsgrest(path: Path, year: int) -> None:
    nddr = [0] * 512
    nddr[10], nddr[11] = -1230, year
    path.write_bytes(_record(struct.pack("<512q", *nddr)) + _record(b"\0" * 48))


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


if __name__ == "__main__":
    unittest.main()
