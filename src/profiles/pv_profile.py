"""Synthetic PV generation profile: clear-sky shape x seeded cloud process."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

PV_SEED_STREAM = 2


def clear_sky_shape(timestamps: pd.DatetimeIndex, sunrise_h: float, sunset_h: float,
                    exponent: float = 1.2) -> np.ndarray:
    """Normalized clear-sky output (0 at night, 1 at solar noon).

    Uses ``sin(pi * (t - sunrise) / daylength) ** exponent``, a common
    approximation of the daily irradiance curve on a fixed tilted plane.
    Values are evaluated at the interval midpoint.
    """
    if not 0.0 <= sunrise_h < sunset_h <= 24.0:
        raise ValueError("Require 0 <= sunrise_h < sunset_h <= 24")
    step_h = (timestamps[1] - timestamps[0]).total_seconds() / 3600.0 if len(timestamps) > 1 else 0.25
    hours = np.asarray(timestamps.hour + timestamps.minute / 60.0 + step_h / 2.0, dtype=float)
    phase = (hours - sunrise_h) / (sunset_h - sunrise_h)
    shape = np.where((phase > 0) & (phase < 1), np.sin(np.pi * np.clip(phase, 0, 1)), 0.0)
    return np.power(shape, exponent)


def cloud_process(n_steps: int, mean: float, std: float, autocorrelation: float,
                  minimum: float, rng: np.random.Generator) -> np.ndarray:
    """Clear-sky index from a stationary AR(1) process, clipped to [minimum, 1]."""
    if not 0.0 <= autocorrelation < 1.0:
        raise ValueError("autocorrelation must be in [0, 1)")
    innovation_std = std * np.sqrt(1.0 - autocorrelation**2)
    deviation = np.empty(n_steps)
    deviation[0] = rng.normal(0.0, std)
    for t in range(1, n_steps):
        deviation[t] = autocorrelation * deviation[t - 1] + rng.normal(0.0, innovation_std)
    return np.clip(mean + deviation, minimum, 1.0)


def generate_pv_profile(
    timestamps: pd.DatetimeIndex,
    installed_capacity_kw: float,
    seed: int = 42,
    sunrise_h: float = 5.0,
    sunset_h: float = 21.0,
    shape_exponent: float = 1.2,
    performance_ratio: float = 0.80,
    seasonal_factor: float = 1.0,
    mean_clearness: float = 0.92,
    variability: float = 0.08,
    autocorrelation: float = 0.85,
    min_clearness: float = 0.30,
) -> pd.DataFrame:
    """Deterministic PV output profile.

    ``pv_pu`` is AC output per unit of installed DC capacity:
    ``clear_sky * clearness * performance_ratio * seasonal_factor``.

    Returns:
        DataFrame with ``clear_sky``, ``clearness``, ``pv_pu`` and ``pv_kw``.
    """
    if installed_capacity_kw < 0:
        raise ValueError("installed_capacity_kw must be non-negative")
    rng = np.random.default_rng([seed, PV_SEED_STREAM])
    clear = clear_sky_shape(timestamps, sunrise_h, sunset_h, shape_exponent)
    clearness = cloud_process(len(timestamps), mean_clearness, variability, autocorrelation,
                              min_clearness, rng)
    pv_pu = np.clip(clear * clearness * performance_ratio * seasonal_factor, 0.0, 1.0)
    return pd.DataFrame(
        {"clear_sky": clear, "clearness": clearness, "pv_pu": pv_pu,
         "pv_kw": pv_pu * installed_capacity_kw},
        index=timestamps,
    )


def pv_profile_from_config(config: dict[str, Any], timestamps: pd.DatetimeIndex,
                           installed_capacity_kw: float) -> pd.DataFrame:
    """Convenience wrapper reading all parameters from the configuration."""
    cfg = config["pv"]
    cloud = cfg["cloud"]
    return generate_pv_profile(
        timestamps, installed_capacity_kw, seed=config["simulation"]["seed"],
        sunrise_h=cfg["sunrise_h"], sunset_h=cfg["sunset_h"], shape_exponent=cfg["shape_exponent"],
        performance_ratio=cfg["performance_ratio"], seasonal_factor=cfg["seasonal_factor"],
        mean_clearness=cloud["mean_clearness"], variability=cloud["variability"],
        autocorrelation=cloud["autocorrelation"], min_clearness=cloud["min_clearness"],
    )
