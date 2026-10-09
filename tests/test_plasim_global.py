"""Tests for the global equilibrium diagnostics of the PlaSim μ runs."""

from __future__ import annotations

import unittest

import numpy as np

from gsebm import plasim_global as global_picture


def gaussian_grid(n: int = 32) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return np.degrees(np.arcsin(nodes)), weights


class IceEdgeTests(unittest.TestCase):
    def test_crossing_is_interpolated_between_rows(self) -> None:
        lat = np.array([-40.0, -30.0, -20.0, 20.0, 30.0, 40.0])
        ice = np.array([[1.0, 0.75, 0.25, 0.0, 0.0, 1.0]])

        self.assertAlmostEqual(global_picture.ice_edge_latitude(ice, lat, True)[0], 25.0)
        self.assertAlmostEqual(global_picture.ice_edge_latitude(ice, lat, False)[0], 35.0)

    def test_ice_free_and_snowball_limits(self) -> None:
        lat = np.array([-30.0, -10.0, 10.0, 30.0])

        self.assertEqual(global_picture.ice_edge_latitude(np.zeros((1, 4)), lat, True)[0], 90.0)
        self.assertEqual(global_picture.ice_edge_latitude(np.ones((1, 4)), lat, True)[0], 10.0)


class TransportTests(unittest.TestCase):
    def test_transport_vanishes_at_poles_and_is_symmetric(self) -> None:
        lat, weight = gaussian_grid()
        # Heating at low latitudes, cooling at high ones, plus an imbalance.
        net = 100 * (np.cos(np.radians(lat)) ** 2 - 2 / 3) + 3.0

        transport = global_picture.implied_northward_transport(net, weight)

        self.assertAlmostEqual(transport[-1], 0.0, places=10)
        # Southern half carries energy south, symmetric about the equator.
        np.testing.assert_allclose(transport[:15], -transport[-2:-17:-1], atol=1e-9)
        self.assertLess(transport[10], 0)


class OceanTransportTests(unittest.TestCase):
    def test_release_in_the_south_needs_southward_transport(self) -> None:
        # Uptake in the north, release in the south (rows south to north).
        uptake = np.array([-10.0, -10.0, 10.0, 10.0])

        transport = global_picture.ocean_northward_transport(uptake, np.ones(4) * 1e14)

        self.assertLess(transport[1], 0)
        self.assertAlmostEqual(transport[-1], 0.0)


class ValidRecordTests(unittest.TestCase):
    def test_trailing_zero_years_are_dropped(self) -> None:
        years = np.r_[np.arange(4500, 4510), np.zeros(3, dtype=int)]

        self.assertEqual(global_picture.valid_record_range(years), (4500, 4509))

    def test_range_stops_at_the_first_break(self) -> None:
        years = np.r_[np.arange(100, 105), np.arange(200, 203)]

        self.assertEqual(global_picture.valid_record_range(years), (100, 104))


class SliceTests(unittest.TestCase):
    def test_slice_keeps_matching_years_in_every_field(self) -> None:
        n = 10
        series = global_picture.GlobalSeries(
            "x", np.arange(100, 100 + n), np.zeros(2), np.ones(2) / 2,
            np.arange(n)[:, None] * np.ones(2), np.zeros((n, 2)), np.zeros((n, 2)),
            np.ones((n, 2)), np.zeros((n, 2)), np.zeros(3), np.ones(3), np.zeros((n, 3)), (101, 107),
        )

        part = global_picture.slice_series(series, (103, 105))

        np.testing.assert_array_equal(part.years, [103, 104, 105])
        np.testing.assert_array_equal(part.surface_temperature[:, 0], [3, 4, 5])
        self.assertEqual(part.ocean_heat_uptake.shape, (3, 3))
        self.assertEqual(part.unreadable_years, ())


class HemisphericModeTests(unittest.TestCase):
    def test_two_mode_profile_is_recovered(self) -> None:
        lat, weight = gaussian_grid()
        x = np.sin(np.radians(lat))
        profile = 260.0 - 30.0 * global_picture.legendre_p2(x)

        modes = global_picture.hemispheric_modes(profile[None], lat, weight, south=True)

        self.assertAlmostEqual(modes["p2"][0], -30.0, places=6)
        self.assertAlmostEqual(modes["mean"][0], 260.0, places=6)
        # Continuous value -0.75 T2 = 22.5; on T21 no row boundary falls at 30°,
        # so the grid split gives about 2% less (22.0).
        self.assertAlmostEqual(modes["delta"][0], 22.0, delta=0.05)


class BudgetStepTests(unittest.TestCase):
    def state(self, mu: float, temperature: float, albedo: float) -> global_picture.EquilibriumState:
        insolation = mu / 4
        absorbed = insolation * (1 - albedo)
        empty = np.zeros(1)
        return global_picture.EquilibriumState(
            "x", mu, "", temperature, albedo, insolation, absorbed, absorbed, 0.0, 0.0,
            empty, empty, empty, empty, empty, empty, empty, empty, (),
        )

    def test_terms_add_up_to_the_absorbed_change(self) -> None:
        warm, cold = self.state(1240, 262.0, 0.385), self.state(1235, 260.0, 0.390)

        step = global_picture.budget_step(warm, cold)

        self.assertAlmostEqual(
            step["direct ΔASR (W m⁻²)"] + step["albedo ΔASR (W m⁻²)"],
            cold.absorbed_shortwave - warm.absorbed_shortwave,
        )
        self.assertGreater(step["gain"], 1)
        self.assertAlmostEqual(step["ΔT per W m⁻² of μ (K)"], 0.4)

    def test_run_labels(self) -> None:
        self.assertEqual(global_picture.run_mu("1235_new_IC"), 1235.0)
        self.assertEqual(global_picture.run_mu("1228p5"), 1228.5)
        self.assertTrue(set(global_picture.RUNS) >= {"1225", "1230", "1245"})


if __name__ == "__main__":
    unittest.main()
