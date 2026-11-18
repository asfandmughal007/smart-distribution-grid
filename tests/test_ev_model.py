"""Tests for the EV fleet generator and charging model."""

import numpy as np
import pandas as pd
import pytest

from src.models.ev_model import (
    EVScheduleError, aggregate_by_bus, build_fleet, connection_matrix, delivered_energy_kwh,
    uncontrolled_charging, validate_ev_schedule,
)
from src.profiles.ev_profile import generate_ev_fleet
from src.utils.config import time_index

DT_H = 0.25


@pytest.fixture(scope="module")
def timestamps(config):
    return time_index(config)


def _single_ev(timestamps, arrival="2025-06-16 18:00", departure="2025-06-17 07:00",
               capacity=60.0, soc0=0.3, target=0.8, pmax=11.0, eta=0.92):
    return build_fleet([{
        "ev_id": "EV000", "bus": 5, "arrival": pd.Timestamp(arrival),
        "departure": pd.Timestamp(departure), "capacity_kwh": capacity, "initial_soc": soc0,
        "target_soc": target, "max_power_kw": pmax, "efficiency": eta,
    }], timestamps)


def test_no_charging_before_arrival_or_after_departure(timestamps):
    fleet = _single_ev(timestamps)
    power = uncontrolled_charging(fleet, len(timestamps), DT_H)
    connected = connection_matrix(fleet, len(timestamps))
    assert (power[~connected] == 0).all()
    first = int(np.argmax(power[0] > 0))
    assert timestamps[first] == pd.Timestamp("2025-06-16 18:00")


def test_required_energy_delivered_exactly(timestamps):
    fleet = _single_ev(timestamps)
    power = uncontrolled_charging(fleet, len(timestamps), DT_H)
    required = (0.8 - 0.3) * 60.0
    assert delivered_energy_kwh(power, fleet, DT_H)[0] == pytest.approx(required)
    report = validate_ev_schedule(power, fleet, DT_H)
    assert report.final_soc.iloc[0] == pytest.approx(0.8)


def test_max_power_respected(timestamps):
    fleet = _single_ev(timestamps, pmax=7.4)
    power = uncontrolled_charging(fleet, len(timestamps), DT_H)
    assert power.max() <= 7.4 + 1e-9


def test_validation_rejects_charging_while_disconnected(timestamps):
    fleet = _single_ev(timestamps)
    power = uncontrolled_charging(fleet, len(timestamps), DT_H)
    power[0, 0] = 5.0  # 12:00, before arrival
    with pytest.raises(EVScheduleError, match="not connected"):
        validate_ev_schedule(power, fleet, DT_H)


def test_validation_rejects_overpower_and_shortfall(timestamps):
    fleet = _single_ev(timestamps)
    power = uncontrolled_charging(fleet, len(timestamps), DT_H)
    too_high = power.copy()
    too_high[0, fleet.arrival_step.iloc[0]] = 20.0
    with pytest.raises(EVScheduleError, match="rating"):
        validate_ev_schedule(too_high, fleet, DT_H)
    with pytest.raises(EVScheduleError, match="requirement"):
        validate_ev_schedule(power * 0.5, fleet, DT_H)


def test_infeasible_target_is_capped(timestamps):
    fleet = _single_ev(timestamps, arrival="2025-06-17 06:00", departure="2025-06-17 07:00",
                       capacity=77.0, soc0=0.2, pmax=7.4)
    assert not fleet.target_feasible.iloc[0]
    assert fleet.required_energy_kwh.iloc[0] == pytest.approx(7.4 * 0.92 * 1.0)


def test_fleet_generation_deterministic(config, timestamps):
    kwargs = dict(n_evs=20, candidate_buses=[1, 2, 3], bus_weights=[1, 1, 2])
    a = generate_ev_fleet(config, timestamps, **kwargs)
    b = generate_ev_fleet(config, timestamps, **kwargs)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.bus) <= {1, 2, 3}
    assert (a.departure > a.arrival).all()
    assert a.target_feasible.all()


def test_aggregate_by_bus_conserves_power(config, timestamps):
    fleet = generate_ev_fleet(config, timestamps, 30, [1, 2, 3], [1, 1, 1])
    power = uncontrolled_charging(fleet, len(timestamps), DT_H)
    by_bus = aggregate_by_bus(power, fleet, timestamps)
    assert by_bus.to_numpy().sum() == pytest.approx(power.sum())
