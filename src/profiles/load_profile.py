"""Synthetic daily base-load profiles (residential and commercial)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.utils.config import hourly_to_timeseries

LOAD_SEED_STREAM = 1  # sub-stream id so each profile draws independent random numbers


def generate_load_profile(
    timestamps: pd.DatetimeIndex,
    residential_shape: list[float],
    commercial_shape: list[float],
    commercial_share: float,
    noise_std: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate normalized load multipliers for every time step.

    Each class shape is interpolated from hourly anchor values, perturbed with
    seeded multiplicative Gaussian noise and normalized to a peak of 1.0, so a
    bus load at time t is ``p_nominal * multiplier(t)``.

    Args:
        timestamps: Simulation time index.
        residential_shape: 24 hourly values (00:00 ... 23:00).
        commercial_shape: 24 hourly values (00:00 ... 23:00).
        commercial_share: Share of feeder nominal load that is commercial (0-1);
            used only for the aggregate ``load_multiplier`` column.
        noise_std: Relative standard deviation of the noise.
        seed: Random seed.

    Returns:
        DataFrame indexed by timestamp with ``residential_multiplier``,
        ``commercial_multiplier`` and ``load_multiplier`` (aggregate).
    """
    if not 0.0 <= commercial_share <= 1.0:
        raise ValueError("commercial_share must be within [0, 1]")
    rng = np.random.default_rng([seed, LOAD_SEED_STREAM])
    profiles = {}
    for name, shape in (("residential", residential_shape), ("commercial", commercial_shape)):
        base = hourly_to_timeseries(shape, timestamps).to_numpy()
        noisy = base * (1.0 + noise_std * rng.standard_normal(len(base)))
        noisy = np.clip(noisy, 0.05, None)  # loads never drop to zero or below
        profiles[f"{name}_multiplier"] = noisy / noisy.max()
    df = pd.DataFrame(profiles, index=timestamps)
    df["load_multiplier"] = (
        (1.0 - commercial_share) * df.residential_multiplier
        + commercial_share * df.commercial_multiplier
    )
    return df


def load_profile_from_config(config: dict[str, Any], timestamps: pd.DatetimeIndex,
                             commercial_share: float) -> pd.DataFrame:
    """Convenience wrapper reading all parameters from the configuration."""
    cfg = config["load"]
    return generate_load_profile(
        timestamps, cfg["residential_shape"], cfg["commercial_shape"], commercial_share,
        cfg["noise_std"], config["simulation"]["seed"],
    )
