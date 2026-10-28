"""Checks of step-ensemble members on small hand-written inventories."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from gsebm.plasim_step_members import (
    member_problems,
    namelist_differences,
    parent_restart_year,
    read_member_list,
)

MEMBER = {"index": 1, "experiment": "STEP_p2_from16999", "start_label": 16999, "gsol0": 1239.5}


def parent_inventory() -> dict:
    """Parent whose labels are 500 above its model years, with bundles at 16999 and 17009."""
    return {
        "restarts": [{"label": 16999, "rest_end_of_year": 16499}, {"label": 17009, "rest_end_of_year": 16509}],
        "namelists": {"planet_namelist": {"values": {"GSOL0": "1237.5"}},
                      "plasim_namelist": {"values": {"MPSTEP": "45", "N_RUN_YEARS": "10"}}},
    }


def member_inventory(first_label: int = 17000, first_model_year: int = 16500, gsol0: str = "1239.5",
                     complete: list | None = None) -> dict:
    """A member with two decades, inheriting the parent's +500 label offset."""
    return {
        "outputs": [{"stream": "PLA", "label_first": first_label, "internal_first": first_model_year}],
        "namelists": {"planet_namelist": {"values": {"GSOL0": gsol0}},
                      "plasim_namelist": {"values": {"MPSTEP": "45", "N_RUN_YEARS": "10"}}},
        "summary": {
            "complete_label_years": complete if complete is not None else [[first_label, first_label + 19]],
            "output_label_offsets": [{"label_first": first_label, "label_last": first_label + 19, "value": {}}],
            "problems": ["labels differ from model years in 1 run(s); see label offsets"],
        },
    }


class StepMemberTest(unittest.TestCase):
    def test_read_member_list(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "members.txt"
            path.write_text("# parent: CONTROL_MU_1237p5\n# base_mu: 1237.5\n"
                            "# idx  experiment  start_restart_year  GSOL0\n"
                            "1 STEP_p2_from16999 16999 1239.5\n2 STEP_m2_from16999 16999 1235.5\n")
            header, members = read_member_list(path)
        self.assertEqual(header, {"parent": "CONTROL_MU_1237p5", "base_mu": "1237.5"})
        self.assertEqual(members[1], {"index": 2, "experiment": "STEP_m2_from16999",
                                      "start_label": 16999, "gsol0": 1235.5})

    def test_parent_restart_year(self):
        self.assertEqual(parent_restart_year(parent_inventory(), 16999), 16499)
        self.assertIsNone(parent_restart_year(parent_inventory(), 17999))

    def test_good_member_passes(self):
        # The inherited label offset is expected and not reported.
        self.assertEqual(member_problems(member_inventory(), MEMBER, parent_inventory()), [])

    def test_wrong_first_label(self):
        problems = member_problems(member_inventory(first_label=17010, first_model_year=16510),
                                   MEMBER, parent_inventory())
        self.assertTrue(any("first complete label 17010" in p for p in problems))

    def test_gap(self):
        inventory = member_inventory(complete=[[17000, 17009], [17020, 17029]])
        problems = member_problems(inventory, MEMBER, parent_inventory())
        self.assertTrue(any("gaps" in p for p in problems))

    def test_wrong_first_model_year(self):
        # Started from another bundle than the label says: the offset note is then reported too.
        problems = member_problems(member_inventory(first_model_year=16510), MEMBER, parent_inventory())
        self.assertTrue(any("first model year 16510" in p for p in problems))
        self.assertTrue(any(p.startswith("labels differ") for p in problems))

    def test_missing_parent_restart(self):
        member = MEMBER | {"start_label": 17999}
        problems = member_problems(member_inventory(first_label=18000), member, parent_inventory())
        self.assertTrue(any("parent has no restart labelled 17999" in p for p in problems))

    def test_wrong_gsol0(self):
        problems = member_problems(member_inventory(gsol0="1237.5"), MEMBER, parent_inventory())
        self.assertTrue(any("GSOL0 1237.5" in p for p in problems))

    def test_namelist_differences(self):
        member = member_inventory()
        self.assertEqual(namelist_differences(member, parent_inventory()), ["planet_namelist:GSOL0"])
        # A control written as 1237.5 vs 1237.50, or 1235 vs 1235.0, is the same value.
        control = member_inventory(gsol0="1237.50")
        self.assertEqual(namelist_differences(control, parent_inventory()), [])
        member["namelists"]["plasim_namelist"]["values"]["N_RUN_YEARS"] = "1"
        member["namelists"]["extra_namelist"] = {"values": {}}
        self.assertEqual(namelist_differences(member, parent_inventory()),
                         ["extra_namelist", "planet_namelist:GSOL0", "plasim_namelist:N_RUN_YEARS"])


if __name__ == "__main__":
    unittest.main()
