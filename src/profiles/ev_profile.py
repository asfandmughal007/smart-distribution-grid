"""Seeded generation of an EV home-charging fleet (session parameters)."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src.models.ev_model import build_fleet

logger = logging.getLogger(__name__)

EV_SEED_STREAM = 3


def _hours_to_timestamp(hours: float, reference: pd.Timestamp) -> pd.Timestamp:
    """Absolute timestamp for a time of day, on the first day at/after ``reference``."""
    stamp = reference.normalize() + pd.Timedelta(hours=hours)
    if stamp < reference:
        stamp += pd.Timedelta(days=1)
    return stamp


def generate_ev_fleet(
    config: dict[str, Any],
    timestamps: pd.DatetimeIndex,
    n_evs: int,
    candidate_buses: list[int],
    bus_weights: list[float],
) -> pd.DataFrame:
    """Sample ``n_evs`` home-charging sessions.

    Arrival and departure times are drawn from truncated normal distributions
    (evening arrival, next-morning departure), rounded to the simulation
    resolution. EVs are allocated to ``candidate_buses`` with probability
    proportional to ``bus_weights`` (e.g. nominal residential load).

    Returns:
        Fleet DataFrame (see :func:`src.models.ev_model.build_fleet`).
    """
    if n_evs == 0:
        return build_fleet([], timestamps)
    cfg = config["ev"]["charging"]
    rng = np.random.default_rng([config["simulation"]["seed"], EV_SEED_STREAM, n_evs])
    resolution_h = config["simulation"]["resolution_min"] / 60.0
    weights = np.asarray(bus_weights, dtype=float)
    buses = rng.choice(candidate_buses, size=n_evs, p=weights / weights.sum())

    def truncated(mean: float, std: float, low: float, high: float) -> np.ndarray:
        values = rng.normal(mean, std, n_evs)
        return np.round(np.clip(values, low, high) / resolution_h) * resolution_h

    arrival_h = truncated(cfg["arrival_mean_h"], cfg["arrival_std_h"], cfg["arrival_min_h"], cfg["arrival_max_h"])
    departure_h = truncated(cfg["departure_mean_h"], cfg["departure_std_h"],
                            cfg["departure_min_h"], cfg["departure_max_h"])
    capacity = rng.choice(cfg["battery_capacity_kwh"], size=n_evs)
    initial_soc = rng.uniform(cfg["initial_soc_min"], cfg["initial_soc_max"], n_evs)
    max_power = rng.choice(cfg["max_power_kw"], size=n_evs)

    start = timestamps[0]
    records = []
    for i in range(n_evs):
        arrival = _hours_to_timestamp(arrival_h[i], start)
        departure = _hours_to_timestamp(departure_h[i], arrival)
        records.append({
            "ev_id": f"EV{i:03d}",
            "bus": int(buses[i]),
            "arrival": arrival,
            "departure": departure,
            "capacity_kwh": float(capacity[i]),
            "initial_soc": float(initial_soc[i]),
            "target_soc": float(cfg["target_soc"]),
            "max_power_kw": float(max_power[i]),
            "efficiency": float(cfg["efficiency"]),
        })
    fleet = build_fleet(records, timestamps)
    logger.info("Generated %d EVs on %d buses, %.0f kWh required in total",
                n_evs, fleet.bus.nunique(), fleet.required_energy_kwh.sum())
    return fleet
