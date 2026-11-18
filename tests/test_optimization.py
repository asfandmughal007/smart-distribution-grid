"""Tests for the LinDistFlow model and the optimization + AC validation loop."""

import numpy as np
import pytest

from src.models.battery import validate_battery_schedule
from src.models.ev_model import connection_matrix, delivered_energy_kwh
from src.optimization.lindistflow import compute_sensitivities, to_bus_matrix
from src.simulation.scenarios import Scenario, get_scenario, run_scenario
from src.utils.config import override


@pytest.fixture(scope="module")
def baseline(config):
    return run_scenario(config, get_scenario(config, "S1_baseline"))


def test_sensitivity_matrices(baseline):
    sens = compute_sensitivities(baseline.data.net)
    assert np.allclose(sens.r_path_ohm, sens.r_path_ohm.T)
    assert sens.downstream.shape == (32, 33)
    assert sens.downstream[:, 0].sum() == 0          # busbar is upstream of every line
    # Resistance to the far end of the main feeder is the sum of its series lines.
    assert sens.r_path_ohm[17, 17] == pytest.approx(
        sum(baseline.data.net.line.r_ohm_per_km.iloc[:17]), rel=1e-9)


def test_lindistflow_close_to_ac(baseline):
    sens = compute_sensitivities(baseline.data.net)
    p = to_bus_matrix(baseline.data.base_load_kw, sens.buses)
    q = to_bus_matrix(baseline.data.base_load_kvar, sens.buses)
    v_busbar = baseline.simulation.bus_vm_pu[0].to_numpy()
    v_lin = v_busbar[None, :] - sens.voltage_drop_pu(p, q)
    v_ac = baseline.simulation.bus_vm_pu[sens.buses].to_numpy().T
    error = v_lin - v_ac
    assert np.abs(error).max() < 0.006
    assert (error > -1e-4).all()  # neglecting losses makes LinDistFlow optimistic


@pytest.fixture(scope="module")
def small_optimized(config):
    """Reduced fleet/BESS so the integration test runs quickly."""
    cfg = override(config, {"ev.levels": {"low": 10, "medium": 20, "high": 30}})
    scenario = Scenario(name="test_opt", label="test", pv_penetration_pct=150, ev_level="high",
                        ev_control="optimized", bess_capacity_kwh=500, bess_control="optimized")
    uncontrolled = Scenario(name="test_unc", label="test", pv_penetration_pct=150, ev_level="high",
                            ev_control="uncontrolled")
    return run_scenario(cfg, scenario), run_scenario(cfg, uncontrolled)


def test_optimized_ev_constraints(small_optimized):
    result, _ = small_optimized
    fleet = result.data.fleet
    power = result.ev_power_kw
    assert (power[~connection_matrix(fleet, power.shape[1])] == 0).all()
    assert (power <= fleet.max_power_kw.to_numpy()[:, None] + 1e-6).all()
    delivered = delivered_energy_kwh(power, fleet, result.data.dt_h)
    assert delivered == pytest.approx(fleet.required_energy_kwh.to_numpy(), abs=1e-3)


def test_optimized_bess_constraints(small_optimized):
    result, _ = small_optimized
    bess = result.data.bess
    sched = result.bess_schedule
    soc = validate_battery_schedule(bess, sched.charge_kw.to_numpy(), sched.discharge_kw.to_numpy(),
                                    result.data.dt_h)
    assert soc[-1] >= bess.soc_initial - 1e-6       # terminal SOC constraint


def test_optimization_validated_by_ac(small_optimized, config):
    result, uncontrolled = small_optimized
    outcome = result.optimization
    assert result.simulation.all_converged
    assert outcome.converged
    assert outcome.history.max_voltage_mismatch_pu.iloc[-1] <= config["optimization"]["iterative_validation"]["voltage_tolerance_pu"]
    assert abs(result.validation["energy_balance_residual_kwh"]) < 1e-6
    # Coordinated dispatch must not raise the substation peak above the uncontrolled case.
    assert result.simulation.system.grid_import_kw.max() <= uncontrolled.simulation.system.grid_import_kw.max() + 1.0
