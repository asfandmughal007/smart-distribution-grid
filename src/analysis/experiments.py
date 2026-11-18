"""Systematic parameter sweeps built on the scenario framework.

Every sweep returns a tidy DataFrame (one row per run) containing the varied
parameters and the AC-validated metrics.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from src.analysis.metrics import compute_metrics
from src.simulation.scenarios import Scenario, ScenarioResult, run_scenario
from src.utils.config import override

logger = logging.getLogger(__name__)

EXPERIMENT_METRICS = [
    "v_min_pu", "v_max_pu", "voltage_violation_bus_steps", "voltage_violation_duration_h",
    "trafo_loading_max_pct", "line_loading_max_pct", "peak_import_kw", "min_grid_kw",
    "reverse_flow_duration_h", "grid_import_kwh", "grid_export_kwh", "pv_available_kwh",
    "pv_curtailment_kwh", "ev_energy_kwh", "bess_throughput_kwh", "losses_kwh",
    "energy_cost_eur", "peak_demand_charge_eur", "total_cost_eur",
]


def _row(params: dict[str, Any], result: ScenarioResult, config: dict[str, Any]) -> dict[str, Any]:
    metrics = compute_metrics(result.simulation, result.data.price, config)
    row = dict(params)
    row.update({key: metrics[key] for key in EXPERIMENT_METRICS})
    if result.optimization is not None:
        row["opt_iterations"] = len(result.optimization.history)
        row["opt_converged"] = result.optimization.converged
        row["objective_eur"] = result.optimization.solution.objective_eur
    return row


def pv_ev_sweep(config: dict[str, Any]) -> pd.DataFrame:
    """Uncontrolled operation for every PV penetration x EV level (no BESS)."""
    rows = []
    for pv in config["pv"]["penetration_levels_pct"]:
        for level in config["ev"]["levels"]:
            scenario = Scenario(name=f"pv{pv}_ev{level}", label="", pv_penetration_pct=pv,
                                ev_level=level, ev_control="uncontrolled")
            rows.append(_row({"pv_penetration_pct": pv, "ev_level": level,
                              "n_evs": config["ev"]["levels"][level]}, run_scenario(config, scenario), config))
    return pd.DataFrame(rows)


def hosting_capacity_sweep(config: dict[str, Any], ev_level: str, bess_capacity_kwh: float) -> pd.DataFrame:
    """Extended PV sweep: uncontrolled vs optimized (smart charging + BESS + curtailment)."""
    rows = []
    for pv in config["pv"]["hosting_capacity_levels_pct"]:
        cases = {
            "uncontrolled": Scenario(name=f"host_unc_{pv}", label="", pv_penetration_pct=pv,
                                     ev_level=ev_level, ev_control="uncontrolled"),
            "optimized": Scenario(name=f"host_opt_{pv}", label="", pv_penetration_pct=pv,
                                  ev_level=ev_level, ev_control="optimized",
                                  bess_capacity_kwh=bess_capacity_kwh,
                                  bess_control="optimized" if bess_capacity_kwh > 0 else "none"),
        }
        for control, scenario in cases.items():
            rows.append(_row({"pv_penetration_pct": pv, "control": control}, run_scenario(config, scenario), config))
    return pd.DataFrame(rows)


def bess_sweep(config: dict[str, Any], pv_penetration_pct: float, ev_level: str) -> pd.DataFrame:
    """Optimized EV + BESS for every BESS capacity level (0 kWh = smart charging only)."""
    rows = []
    for capacity in config["bess"]["capacity_levels_kwh"]:
        scenario = Scenario(name=f"bess{capacity}", label="", pv_penetration_pct=pv_penetration_pct,
                            ev_level=ev_level, ev_control="optimized", bess_capacity_kwh=capacity,
                            bess_control="optimized" if capacity > 0 else "none")
        rows.append(_row({"bess_capacity_kwh": capacity,
                          "bess_power_kw": capacity * config["bess"]["c_rate"]},
                         run_scenario(config, scenario), config))
    return pd.DataFrame(rows)


def ev_control_sweep(config: dict[str, Any], pv_penetration_pct: float) -> pd.DataFrame:
    """Uncontrolled vs optimized charging for every EV level (no BESS)."""
    rows = []
    for level in config["ev"]["levels"]:
        for control in ("uncontrolled", "optimized"):
            scenario = Scenario(name=f"ev_{level}_{control}", label="", pv_penetration_pct=pv_penetration_pct,
                                ev_level=level, ev_control=control)
            rows.append(_row({"ev_level": level, "n_evs": config["ev"]["levels"][level], "control": control},
                             run_scenario(config, scenario), config))
    return pd.DataFrame(rows)


def peak_charge_tradeoff(config: dict[str, Any], scenario: Scenario) -> pd.DataFrame:
    """Re-optimize one scenario for several demand-charge rates (cost vs peak trade-off)."""
    rows = []
    for rate in config["experiments"]["pareto_peak_charges_eur_per_kw"]:
        cfg = override(config, {"price.peak_demand_charge_eur_per_kw": rate})
        result = run_scenario(cfg, scenario)
        # Evaluate every run with the *same* reference tariff so costs are comparable.
        row = _row({"peak_charge_eur_per_kw": rate}, result, config)
        rows.append(row)
    return pd.DataFrame(rows)
