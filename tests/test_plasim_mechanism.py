"""Tests for the ice-cycle phase and composites of the mechanism analysis."""

from __future__ import annotations

import unittest

import numpy as np

from gsebm import plasim_mechanism as mechanism


class EventPhaseTests(unittest.TestCase):
    def test_phase_runs_linearly_between_maxima(self) -> None:
        phase = mechanism.event_phase(12, np.array([2, 6, 10]))

        self.assertTrue(np.all(np.isnan(phase[:2])))
        np.testing.assert_allclose(phase[2:6], [0, 0.25, 0.5, 0.75])
        np.testing.assert_allclose(phase[6:10], [0, 0.25, 0.5, 0.75])
        self.assertTrue(np.all(np.isnan(phase[10:])))

    def test_unequal_cycles_are_stretched_to_unit_phase(self) -> None:
        phase = mechanism.event_phase(20, np.array([0, 4, 14]))

        self.assertAlmostEqual(phase[2], 0.5)
        self.assertAlmostEqual(phase[9], 0.5)


class CompositeTests(unittest.TestCase):
    def test_composite_recovers_a_phase_locked_shape(self) -> None:
        peaks = np.cumsum(np.r_[0, np.random.default_rng(0).integers(40, 70, 30)])
        phase = mechanism.event_phase(peaks[-1] + 5, peaks)
        values = np.where(np.isfinite(phase), np.cos(2 * np.pi * np.nan_to_num(phase)), 9.0)

        composite = mechanism.phase_composite(values, phase, bins=10)

        centres = (np.arange(10) + 0.5) / 10
        np.testing.assert_allclose(composite, np.cos(2 * np.pi * centres), atol=0.05)

    def test_composite_keeps_trailing_dimensions(self) -> None:
        phase = mechanism.event_phase(50, np.array([0, 25, 50 - 1]))
        values = np.ones((50, 3, 2))

        self.assertEqual(mechanism.phase_composite(values, phase, bins=5).shape, (5, 3, 2))


class StageDurationTests(unittest.TestCase):
    def test_sawtooth_has_short_retreat_and_long_advance(self) -> None:
        cycle = np.r_[np.linspace(1, -1, 11)[:-1], np.linspace(-1, 1, 31)[:-1]]
        smoothed = np.r_[np.tile(cycle, 3), 1.0]
        peaks = np.arange(0, smoothed.size, 40)

        durations = mechanism.stage_durations(smoothed, peaks)

        np.testing.assert_array_equal(durations, [[10, 30]] * 3)


class GeometryTests(unittest.TestCase):
    def test_gaussian_cell_areas_cover_the_sphere(self) -> None:
        _, weights = np.polynomial.legendre.leggauss(32)
        area = mechanism.t21_cell_area(weights, 64)

        self.assertAlmostEqual(area.sum() / (4 * np.pi * mechanism.EARTH_RADIUS_M**2), 1.0)

    def test_mu_labels(self) -> None:
        self.assertEqual(mechanism.mu_value("1233p75"), 1233.75)
        self.assertEqual(mechanism.archive_path("root", "1240").name,
                         "CONTROL_360ppm_T21L10_10000Y_MU_1240_spinup_raw_maps.nc")


if __name__ == "__main__":
    unittest.main()
