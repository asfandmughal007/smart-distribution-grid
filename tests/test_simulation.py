"""Integration tests for the time-series power flow and scenario framework."""

import numpy as np
import pytest

from src.analysis.metrics import compute_metrics
from src.simulation.power_flow import energy_balance_residual_kwh
from src.simulation.scenarios import Scenario, get_scenario, load_scenarios, run_scenario


@pytest.fixture(scope="module")
def baseline(config):
    return run_scenario(config, get_scenario(config, "S1_baseline"))


def test_scenarios_load_from_config(config):
    scenarios = load_scenarios(config)
    assert len(scenarios) == 7
    assert len({s.name for s in scenarios}) == 7


def test_inconsistent_scenario_rejected():
    with pytest.raises(ValueError):
        Scenario(name="bad", label="bad", ev_level="high", ev_control="none")
    with pytest.raises(ValueError):
        Scenario(name="bad", label="bad", bess_capacity_kwh=100, bess_control="none")


def test_baseline_time_series_converges(baseline):
    sim = baseline.simulation
    assert sim.all_converged
    assert len(sim.bus_vm_pu) == 96
    assert sim.bus_vm_pu.shape[1] == 33


def test_baseline_energy_balance(baseline):
    assert abs(energy_balance_residual_kwh(baseline.simulation)) < 1e-6


def test_baseline_has_no_der(baseline):
    s = baseline.simulation.system
    assert (s[["pv_kw", "ev_kw", "bess_charge_kw", "bess_discharge_kw"]] == 0).all().all()
    assert (s.grid_export_kw == 0).all()


def test_metrics_consistent_with_raw_results(baseline, config):
    m = compute_metrics(baseline.simulation, baseline.data.price, config)
    sim = baseline.simulation
    assert m["v_min_pu"] == pytest.approx(np.nanmin(sim.bus_vm_pu.to_numpy()))
    assert m["trafo_loading_max_pct"] == pytest.approx(sim.trafo.loading_pct.max())
    assert m["grid_import_kwh"] == pytest.approx(sim.system.grid_import_kw.sum() * 0.25)


def test_figures_are_written(baseline, config, tmp_path):
    import matplotlib

    matplotlib.use("Agg")
    from src.analysis import visualization as viz

    viz.apply_style()
    limits = config["limits"]
    paths = [
        viz.plot_load_profile(baseline, tmp_path, 60),
        viz.plot_min_voltage({"S1": baseline}, limits["v_min_pu"], tmp_path, 60),
        viz.plot_voltage_profile({"S1": baseline}, limits["v_min_pu"], tmp_path, 60),
        viz.plot_transformer_loading({"S1": baseline}, limits["trafo_max_loading_pct"], tmp_path, 60),
    ]
    assert all(path.is_file() and path.stat().st_size > 10_000 for path in paths)
