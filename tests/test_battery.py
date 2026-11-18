"""Tests for the battery energy storage model."""

import numpy as np
import pandas as pd
import pytest

from src.models.battery import (
    Battery, BatteryParameters, BatteryScheduleError, rule_based_dispatch, soc_from_schedule,
    validate_battery_schedule,
)

DT_H = 0.25


@pytest.fixture
def params():
    return BatteryParameters(capacity_kwh=100.0, max_charge_kw=50.0, max_discharge_kw=50.0,
                             soc_min=0.1, soc_max=0.9, soc_initial=0.5,
                             charge_efficiency=0.95, discharge_efficiency=0.9)


def test_charging_applies_efficiency(params):
    battery = Battery(params)
    applied = battery.step(40.0, DT_H)
    assert applied == pytest.approx(40.0)
    assert battery.energy_kwh == pytest.approx(50.0 + 40.0 * 0.95 * DT_H)


def test_discharging_applies_efficiency(params):
    battery = Battery(params)
    applied = battery.step(-36.0, DT_H)
    assert applied == pytest.approx(-36.0)
    assert battery.energy_kwh == pytest.approx(50.0 - 36.0 * DT_H / 0.9)


def test_power_limits_enforced(params):
    battery = Battery(params)
    assert battery.step(500.0, DT_H) == pytest.approx(50.0)
    assert battery.step(-500.0, DT_H) == pytest.approx(-50.0)


def test_soc_stays_within_limits(params):
    battery = Battery(params)
    for _ in range(40):
        battery.step(50.0, DT_H)
    assert battery.soc == pytest.approx(0.9)
    for _ in range(40):
        battery.step(-50.0, DT_H)
    assert battery.soc == pytest.approx(0.1)


def test_soc_from_schedule_matches_stateful_model(params):
    rng = np.random.default_rng(0)
    setpoints = rng.uniform(-20, 20, 24)
    battery = Battery(params)
    applied = np.array([battery.step(p, DT_H) for p in setpoints])
    soc = soc_from_schedule(params, np.clip(applied, 0, None), np.clip(-applied, 0, None), DT_H)
    assert soc[-1] == pytest.approx(battery.soc)


def test_simultaneous_charge_discharge_rejected(params):
    charge = np.array([10.0, 0.0])
    discharge = np.array([5.0, 0.0])
    with pytest.raises(BatteryScheduleError, match="Simultaneous"):
        validate_battery_schedule(params, charge, discharge, DT_H)


def test_validation_detects_soc_violation(params):
    charge = np.full(20, 50.0)  # would exceed 90 % SOC
    with pytest.raises(BatteryScheduleError, match="SOC"):
        validate_battery_schedule(params, charge, np.zeros(20), DT_H)


def test_invalid_parameters_rejected():
    with pytest.raises(ValueError):
        BatteryParameters(capacity_kwh=100, max_charge_kw=10, max_discharge_kw=10,
                          soc_min=0.5, soc_initial=0.2, soc_max=0.9)


def test_rule_based_dispatch_shaves_peak(params):
    index = pd.date_range("2025-01-01", periods=4, freq="15min")
    demand = pd.Series([100.0, 400.0, 100.0, 50.0], index=index)
    result = rule_based_dispatch(params, demand, DT_H, charge_below_kw=80.0, discharge_above_kw=300.0)
    assert result.discharge_kw.iloc[1] == pytest.approx(50.0)  # limited by power rating
    assert result.charge_kw.iloc[3] == pytest.approx(30.0)
    assert ((result.charge_kw > 0) & (result.discharge_kw > 0)).sum() == 0


def test_from_config(config):
    params = BatteryParameters.from_config(config, capacity_kwh=500)
    assert params.max_charge_kw == pytest.approx(500 * config["bess"]["c_rate"])
    assert params.bus == config["bess"]["bus"]
