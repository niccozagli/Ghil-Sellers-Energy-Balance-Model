"""Tests for the experiment files (Plasim_Indicators.md §7) on small synthetic products."""

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

from gsebm.plasim_experiments import collect_experiment, write_experiment_files
from gsebm.plasim_indicators import load_indicators, product_stem

sys.path.insert(0, str(Path(__file__).parent))
from test_plasim_indicators import synthetic_product  # noqa: E402


def write_product(root: Path, run: str, years) -> None:
    """A run's zonal and sections files with the synthetic values of test_plasim_indicators."""
    zonal, sections = synthetic_product(years)
    folder = root / run
    folder.mkdir(parents=True)
    stem = folder / product_stem(run, "spinup")
    zonal.to_netcdf(stem.with_name(stem.name + "_zonal.nc"), engine="h5netcdf")
    sections.to_netcdf(stem.with_name(stem.name + "_sections.nc"), engine="h5netcdf")


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        # Two members of different lengths from different starts, and one without a product.
        write_product(self.root, "RUN_m_from100", years=range(101, 104))
        write_product(self.root, "RUN_p_from200", years=range(201, 206))
        self.header = {"parent": "PARENT", "base_mu": "1232.25"}
        self.members = [
            {"index": 1, "experiment": "RUN_m_from100", "start_label": 100, "gsol0": 1232.0, "parent": "SOURCE_A"},
            {"index": 2, "experiment": "RUN_p_from200", "start_label": 200, "gsol0": 1233.0},
            {"index": 3, "experiment": "RUN_missing", "start_label": 300, "gsol0": 1232.0},
        ]

    def tearDown(self):
        self.tmp.cleanup()

    def test_members_aligned_by_years_since_the_step(self):
        series, fields, missing = collect_experiment(self.members, self.header, self.root)
        self.assertEqual(missing, ["RUN_missing"])
        self.assertEqual(list(series["year"].values), [1, 2, 3, 4, 5])
        self.assertEqual(list(series["role"].values), ["minus", "plus"])
        self.assertEqual(list(series["source"].values), ["SOURCE_A", "PARENT"])
        self.assertEqual(list(series["years_run"].values), [3, 5])
        # The shorter member is padded with NaN after its last year.
        self.assertTrue(np.all(np.isfinite(series["tas_global"].values[0, :3])))
        self.assertTrue(np.all(np.isnan(series["tas_global"].values[0, 3:])))
        np.testing.assert_allclose(series["t_0_700"].values[1], 1.0)
        self.assertEqual(fields["tas"].dims, ("member", "year", "t21_lat"))
        self.assertEqual(fields["t_2025_bottom"].dims, ("member", "year", "lsg_lat"))

    def test_files_round_trip(self):
        series, fields, _ = collect_experiment(self.members, self.header, self.root)
        paths = write_experiment_files(series, fields, self.root / "out", "test", {"member_list": "list.txt"})
        self.assertEqual([p.name for p in paths], ["test_series.nc", "test_latitude.nc"])
        back = load_indicators(paths[0])
        self.assertEqual(back.attrs["member_list"], "list.txt")
        self.assertEqual(list(back["experiment"].values), ["RUN_m_from100", "RUN_p_from200"])
        np.testing.assert_allclose(back["acc"].values[1], 14.0)

    def test_wrong_first_label_raises(self):
        members = [dict(self.members[0], start_label=99)]
        with self.assertRaises(ValueError):
            collect_experiment(members, self.header, self.root)


if __name__ == "__main__":
    unittest.main()
