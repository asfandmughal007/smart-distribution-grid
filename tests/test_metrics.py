"""Tests for metric calculations on small hand-checkable results."""

import numpy as np
import pandas as pd
import pytest

from src.analysis.metrics import (
    economic_metrics, energy_metrics, loading_metrics, summary_table, to_markdown, voltage_metrics,
)

DT_H = 0.25
INDEX = pd.date_range("2025-06-16 12:00", periods=4, freq="15min")


def test_voltage_metrics():
    vm = pd.DataFrame({0: [1.00, 1.00, 1.00, 1.00],
                       5: [0.97, 0.94, 0.93, 0.96],
                       9: [0.98, 0.96, 0.94, 1.06]}, index=INDEX)
    m = voltage_metrics(vm, 0.95, 1.05, DT_H)
    assert m["v_min_pu"] == pytest.approx(0.93)
    assert m["v_min_bus"] == 5
    assert m["v_max_pu"] == pytest.approx(1.06)
    assert m["undervoltage_bus_steps"] == 3
    assert m["overvoltage_bus_steps"] == 1
    assert m["voltage_violation_steps"] == 3          # steps 1, 2, 3
    assert m["voltage_violation_duration_h"] == pytest.approx(0.75)
    assert m["buses_with_violation"] == 2


def test_loading_metrics():
    lines = pd.DataFrame({0: [50.0, 80.0, 110.0, 60.0], 1: [10.0, 20.0, 30.0, 40.0]}, index=INDEX)
    trafo = pd.Series([70.0, 101.0, 102.0, 90.0], index=INDEX)
    m = loading_metrics(lines, trafo, 100.0, 100.0, DT_H)
    assert m["line_loading_max_pct"] == pytest.approx(110.0)
    assert m["line_loading_avg_pct"] == pytest.approx(50.0)
    assert m["line_overload_duration_h"] == pytest.approx(0.25)
    assert m["trafo_loading_max_pct"] == pytest.approx(102.0)
    assert m["trafo_overload_duration_h"] == pytest.approx(0.5)


@pytest.fixture
def system():
    return pd.DataFrame({
        "grid_kw": [100.0, -20.0, 50.0, 0.0],
        "grid_import_kw": [100.0, 0.0, 50.0, 0.0],
        "grid_export_kw": [0.0, 20.0, 0.0, 0.0],
        "losses_kw": [2.0, 1.0, 1.0, 0.0],
        "load_kw": [80.0, 40.0, 60.0, 30.0],
        "ev_kw": [18.0, 0.0, 0.0, 0.0],
        "pv_available_kw": [0.0, 70.0, 20.0, 40.0],
        "pv_curtailed_kw": [0.0, 9.0, 0.0, 0.0],
        "pv_kw": [0.0, 61.0, 20.0, 40.0],
        "bess_charge_kw": [0.0, 0.0, 9.0, 10.0],
        "bess_discharge_kw": [0.0, 0.0, 0.0, 0.0],
    }, index=INDEX)


def test_energy_totals_consistent(system):
    m = energy_metrics(system, DT_H)
    assert m["grid_import_kwh"] == pytest.approx(37.5)
    assert m["grid_export_kwh"] == pytest.approx(5.0)
    assert m["pv_curtailment_kwh"] == pytest.approx(2.25)
    assert m["pv_available_kwh"] == pytest.approx(m["pv_generation_kwh"] + m["pv_curtailment_kwh"])
    supply = m["grid_import_kwh"] + m["pv_generation_kwh"] + m["bess_discharge_kwh"]
    demand = (m["load_kwh"] + m["ev_energy_kwh"] + m["bess_charge_kwh"] + m["losses_kwh"]
              + m["grid_export_kwh"])
    assert supply == pytest.approx(demand)
    assert m["bess_throughput_kwh"] == pytest.approx(19.0 * DT_H / 2)


def test_economic_metrics(system):
    price = pd.DataFrame({"import_eur_per_kwh": [0.2, 0.2, 0.4, 0.4], "export_eur_per_kwh": 0.1}, index=INDEX)
    m = economic_metrics(system, price, DT_H, peak_charge_eur_per_kw=0.5, degradation_eur_per_kwh=0.1)
    expected_energy = (100 * 0.2 + 50 * 0.4) * DT_H - 20 * 0.1 * DT_H
    assert m["energy_cost_eur"] == pytest.approx(expected_energy)
    assert m["peak_import_kw"] == pytest.approx(100.0)
    assert m["peak_demand_charge_eur"] == pytest.approx(50.0)
    assert m["total_cost_eur"] == pytest.approx(expected_energy + 50.0 + 0.1 * 19.0 * DT_H / 2)


def test_summary_table_and_markdown(system):
    metrics = {key: 1.0 for key in (
        "peak_import_kw", "v_min_pu", "v_max_pu", "voltage_violation_duration_h",
        "voltage_violation_bus_steps", "line_loading_max_pct", "trafo_loading_max_pct", "losses_kwh",
        "grid_import_kwh", "grid_export_kwh", "pv_curtailment_kwh", "ev_energy_kwh",
        "bess_throughput_kwh", "energy_cost_eur", "total_cost_eur")}
    table = summary_table({"S1": metrics})
    assert table.iloc[0]["Scenario"] == "S1"
    md = to_markdown(table)
    assert md.count("\n") == 2 and md.startswith("| Scenario")
    assert not np.isnan(table.select_dtypes("number").to_numpy()).any()
