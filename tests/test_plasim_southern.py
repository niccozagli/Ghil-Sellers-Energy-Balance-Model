"""Tests for the Southern-extratropics diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from gsebm import plasim_southern as southern


class LagTests(unittest.TestCase):
    def test_peak_at_the_imposed_delay(self) -> None:
        rng = np.random.default_rng(0)
        x = rng.standard_normal(2000)
        y = np.r_[np.zeros(7), x[:-7]]  # y follows x by 7 years

        lags, r = southern.lagged_correlation(x, y, 20)

        self.assertEqual(lags[np.argmax(r)], 7)
        self.assertGreater(r.max(), 0.99)


class BoxTests(unittest.TestCase):
    def test_rows_are_selected_by_centre_latitude_on_each_grid(self) -> None:
        from gsebm.plasim_global import GlobalSeries

        n = 3
        lat = np.array([-60.0, -40.0, -20.0, 20.0])
        lsg_lat = np.array([-65.0, -45.0, -25.0, 25.0])
        series = GlobalSeries(
            "x", np.arange(n), lat, np.ones(4) / 4, np.zeros((n, 4)), np.zeros((n, 4)),
            np.ones((n, 4)), np.zeros((n, 4)), np.zeros((n, 4)), lsg_lat, np.ones(4),
            -np.ones((n, 4)), (),
        )

        box = southern.box_series(series, (-50.0, -15.0))

        np.testing.assert_allclose(box.ocean_release, 2.0)  # LSG rows -45 and -25
        self.assertAlmostEqual(box.ocean_area, 2.0)
        np.testing.assert_allclose(box.toa_net, 2 * box.atmosphere_area / 2)


class CapTableTests(unittest.TestCase):
    def test_share_is_cap_release_over_cap_loss(self) -> None:
        years = np.arange(10)
        cap = southern.BoxSeries("x", years, np.full(10, 1e15), np.full(10, -4e15), 1.0, 1.0)
        band = southern.BoxSeries("x", years, np.full(10, 0.8e15), np.full(10, -2e15), 1.0, 1.0)

        row = southern.cap_table({"x": {"band": band, "cap": cap}}, {"x": 36.0}, {"x": (0, 9)})[0]

        self.assertEqual(row["cap ocean share"], 0.25)
        self.assertEqual(row["band / cap release"], 0.8)


if __name__ == "__main__":
    unittest.main()
