"""Configuration loading and simulation time-axis helpers."""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

REQUIRED_SECTIONS = (
    "simulation", "network", "limits", "load", "pv", "ev", "bess", "price", "optimization",
)


class ConfigError(ValueError):
    """Raised when the configuration file is missing or inconsistent."""


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load the YAML configuration and validate that required sections exist.

    Args:
        path: Path to a YAML file. Defaults to ``config/config.yaml``.

    Returns:
        The configuration as a nested dictionary.
    """
    config_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not config_path.is_file():
        raise ConfigError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    missing = [section for section in REQUIRED_SECTIONS if section not in config]
    if missing:
        raise ConfigError(f"Configuration is missing sections: {missing}")
    sim = config["simulation"]
    if (sim["horizon_h"] * 60) % sim["resolution_min"] != 0:
        raise ConfigError("horizon_h must be an integer multiple of resolution_min")
    return config


def override(config: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """Return a deep copy of ``config`` with dotted-key ``updates`` applied.

    Example:
        ``override(cfg, {"pv.scenario_penetration_pct": 50})``
    """
    new_config = copy.deepcopy(config)
    for dotted_key, value in updates.items():
        node = new_config
        *parents, leaf = dotted_key.split(".")
        for key in parents:
            if key not in node:
                raise ConfigError(f"Unknown configuration key: {dotted_key}")
            node = node[key]
        if leaf not in node:
            raise ConfigError(f"Unknown configuration key: {dotted_key}")
        node[leaf] = value
    return new_config


def time_index(config: dict[str, Any]) -> pd.DatetimeIndex:
    """Build the simulation time index (e.g. 96 x 15-min steps)."""
    sim = config["simulation"]
    n_steps = sim["horizon_h"] * 60 // sim["resolution_min"]
    return pd.date_range(
        start=sim["start"], periods=n_steps, freq=f"{sim['resolution_min']}min", name="timestamp"
    )


def step_hours(config: dict[str, Any]) -> float:
    """Length of one time step in hours (0.25 h for 15-min resolution)."""
    return config["simulation"]["resolution_min"] / 60.0


def hourly_to_timeseries(hourly_values: list[float], timestamps: pd.DatetimeIndex) -> pd.Series:
    """Map a 24-value hourly shape (index 0 = 00:00) onto ``timestamps``.

    Values are linearly interpolated between hour midpoints (cyclically across
    midnight) so the 15-min profile has no artificial steps.
    """
    if len(hourly_values) != 24:
        raise ConfigError("Hourly shapes must contain exactly 24 values")
    hours = timestamps.hour + timestamps.minute / 60.0
    anchor_hours = [h + 0.5 for h in range(24)]
    # Pad cyclically so interpolation wraps around midnight.
    xp = [anchor_hours[-1] - 24.0] + anchor_hours + [anchor_hours[0] + 24.0]
    fp = [hourly_values[-1]] + list(hourly_values) + [hourly_values[0]]
    return pd.Series(np.interp(hours, xp, fp), index=timestamps)


def hourly_step_timeseries(hourly_values: list[float], timestamps: pd.DatetimeIndex) -> pd.Series:
    """Map a 24-value hourly tariff onto ``timestamps`` as a step function (no interpolation)."""
    if len(hourly_values) != 24:
        raise ConfigError("Hourly tariffs must contain exactly 24 values")
    return pd.Series([hourly_values[h] for h in timestamps.hour], index=timestamps)


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a concise console logger for scripts."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    # pandapower and numba are verbose at INFO level.
    for noisy in ("pandapower", "numba", "pyomo"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
