"""Tests for configuration loading and time-axis helpers."""

import pytest

from src.utils.config import (
    ConfigError, hourly_step_timeseries, hourly_to_timeseries, load_config, override, step_hours, time_index,
)


def test_config_loads_all_sections(config):
    for section in ("simulation", "network", "limits", "pv", "ev", "bess", "price", "optimization", "scenarios"):
        assert section in config


def test_missing_config_file_raises(tmp_path):
    with pytest.raises(ConfigError):
        load_config(tmp_path / "missing.yaml")


def test_override_is_deep_copy(config):
    changed = override(config, {"pv.performance_ratio": 0.5})
    assert changed["pv"]["performance_ratio"] == 0.5
    assert config["pv"]["performance_ratio"] != 0.5
    with pytest.raises(ConfigError):
        override(config, {"pv.not_a_key": 1})


def test_time_axis(config):
    index = time_index(config)
    assert len(index) == 96
    assert step_hours(config) == pytest.approx(0.25)


def test_hourly_interpolation_and_steps(config):
    index = time_index(config)
    shape = list(range(24))
    stepped = hourly_step_timeseries(shape, index)
    assert (stepped.to_numpy() == index.hour).all()
    smooth = hourly_to_timeseries([1.0] * 24, index)
    assert (smooth == 1.0).all()
    with pytest.raises(ConfigError):
        hourly_to_timeseries([1.0] * 23, index)
