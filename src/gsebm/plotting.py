"""Plotting utilities for model outputs."""

from __future__ import annotations

import matplotlib.pyplot as plt


def plot_asymptotic_state_diagnostics(
    latitude,
    warm_temperature,
    cold_temperature,
    edge_latitude,
    edge_temperature,
    warm_albedo,
    cold_albedo,
    edge_albedo,
    warm_heat_transfer,
    cold_heat_transfer,
    edge_heat_transfer,
) -> plt.Figure:
    """Plot asymptotic temperature, albedo, and heat transfer."""

    figure, axes = plt.subplots(nrows=3, figsize=(8, 10), sharex=True)
    temperature_ax, albedo_ax, heat_transfer_ax = axes.ravel()

    temperature_ax.plot(
        latitude,
        warm_temperature,
        color="red",
        label="Warm State",
    )
    temperature_ax.plot(
        latitude,
        cold_temperature,
        color="blue",
        label="Cold State",
    )
    temperature_ax.plot(
        edge_latitude,
        edge_temperature,
        linestyle="--",
        color="green",
        label="Edge State",
    )
    temperature_ax.set_ylabel("Temperature [K]")
    temperature_ax.grid(linestyle="--", alpha=0.4)
    temperature_ax.legend()

    albedo_ax.plot(
        latitude,
        warm_albedo,
        color="red",
        label="Warm State",
    )
    albedo_ax.plot(
        latitude,
        cold_albedo,
        color="blue",
        label="Cold State",
    )
    albedo_ax.plot(
        edge_latitude,
        edge_albedo,
        linestyle="--",
        color="green",
        label="Edge State",
    )
    albedo_ax.set_ylabel("Albedo [-]")
    albedo_ax.grid(linestyle="--", alpha=0.4)
    albedo_ax.legend()

    heat_transfer_ax.plot(
        latitude,
        warm_heat_transfer,
        color="red",
        label="Warm State",
    )
    heat_transfer_ax.plot(
        latitude,
        cold_heat_transfer,
        color="blue",
        label="Cold State",
    )
    heat_transfer_ax.plot(
        edge_latitude,
        edge_heat_transfer,
        linestyle="--",
        color="green",
        label="Edge State",
    )
    heat_transfer_ax.set_xlim(left=0, right=1)
    heat_transfer_ax.set_xlabel("Normalized latitude")
    heat_transfer_ax.set_ylabel(r"Heat transfer [W m$^{-2}$]")
    heat_transfer_ax.grid(linestyle="--", alpha=0.4)
    heat_transfer_ax.set_ylim(bottom=-0.5)
    heat_transfer_ax.legend()

    figure.tight_layout()
    return figure
