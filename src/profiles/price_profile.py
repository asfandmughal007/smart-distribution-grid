"""Synthetic time-of-use electricity price profile.

This is a synthetic time-of-use electricity price used for simulation. It does
not represent any specific German (or other) retail tariff.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.utils.config import hourly_step_timeseries


def generate_price_profile(timestamps: pd.DatetimeIndex, import_hourly_eur_per_kwh: list[float],
                           export_eur_per_kwh: float) -> pd.DataFrame:
    """Return import and export prices (EUR/kWh) for every time step.

    The import price is a step function of the hour (tariff periods do not
    change within an hour); the export (feed-in) price is flat.
    """
    if export_eur_per_kwh < 0 or min(import_hourly_eur_per_kwh) < 0:
        raise ValueError("Prices must be non-negative")
    import_price = hourly_step_timeseries(import_hourly_eur_per_kwh, timestamps)
    return pd.DataFrame(
        {"import_eur_per_kwh": import_price.to_numpy(),
         "export_eur_per_kwh": float(export_eur_per_kwh)},
        index=timestamps,
    )


def price_profile_from_config(config: dict[str, Any], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Convenience wrapper reading the tariff from the configuration."""
    cfg = config["price"]
    return generate_price_profile(timestamps, cfg["import_eur_per_kwh"], cfg["export_eur_per_kwh"])
