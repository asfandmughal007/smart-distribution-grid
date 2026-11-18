"""Configurable scenario framework.

A :class:`Scenario` only describes *what* is connected and *how* it is
controlled. :func:`prepare_scenario` turns it into a network plus input
profiles, :func:`run_scenario` dispatches EVs/BESS (uncontrolled, rule-based
or optimized), runs the AC time-series power flow and validates the result.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandapower as pp
import pandas as pd

from src.grid.network import (
    add_bess, add_ev_loads, add_pv_generators, build_network, feeder_peak_load_mw, residential_buses,
)
from src.models.battery import BatteryParameters, rule_based_dispatch, validate_battery_schedule
from src.models.ev_model import aggregate_by_bus, uncontrolled_charging, validate_ev_schedule
from src.profiles.ev_profile import generate_ev_fleet
from src.profiles.load_profile import load_profile_from_config
from src.profiles.price_profile import price_profile_from_config
from src.profiles.pv_profile import pv_profile_from_config
from src.simulation.power_flow import (
    SimulationResult, TimeSeriesInputs, energy_balance_residual_kwh, run_time_series,
)
from src.utils.config import step_hours, time_index

logger = logging.getLogger(__name__)

EV_CONTROLS = {"none", "uncontrolled", "optimized"}
BESS_CONTROLS = {"none", "rule_based", "optimized"}


@dataclass(frozen=True)
class Scenario:
    """Declarative description of one study case."""

    name: str
    label: str
    pv_penetration_pct: float = 0.0
    ev_level: str | None = None
    ev_control: str = "none"
    bess_capacity_kwh: float = 0.0
    bess_control: str = "none"

    def __post_init__(self) -> None:
        if self.ev_control not in EV_CONTROLS:
            raise ValueError(f"{self.name}: ev_control must be one of {EV_CONTROLS}")
        if self.bess_control not in BESS_CONTROLS:
            raise ValueError(f"{self.name}: bess_control must be one of {BESS_CONTROLS}")
        if (self.ev_level is None) != (self.ev_control == "none"):
            raise ValueError(f"{self.name}: ev_level and ev_control must both be set or both be none")
        if (self.bess_capacity_kwh > 0) != (self.bess_control != "none"):
            raise ValueError(f"{self.name}: bess_capacity_kwh and bess_control are inconsistent")
        if self.pv_penetration_pct < 0:
            raise ValueError(f"{self.name}: pv_penetration_pct must be non-negative")

    @property
    def is_optimized(self) -> bool:
        return self.ev_control == "optimized" or self.bess_control == "optimized"


def load_scenarios(config: dict[str, Any]) -> list[Scenario]:
    """Scenarios defined in the ``scenarios`` section of the configuration."""
    return [Scenario(**entry) for entry in config["scenarios"]]


def get_scenario(config: dict[str, Any], name: str) -> Scenario:
    for scenario in load_scenarios(config):
        if scenario.name == name:
            return scenario
    raise KeyError(f"Unknown scenario: {name}")


@dataclass
class ScenarioData:
    """Network and all input data of a prepared scenario."""

    config: dict[str, Any]
    scenario: Scenario
    net: pp.pandapowerNet
    timestamps: pd.DatetimeIndex
    dt_h: float
    load_multipliers: pd.DataFrame
    base_load_kw: pd.DataFrame          # [time x bus] base active load
    base_load_kvar: pd.DataFrame        # [time x bus] base reactive load
    pv_available_kw: pd.DataFrame       # [time x bus]
    fleet: pd.DataFrame
    price: pd.DataFrame
    bess: BatteryParameters | None


@dataclass
class ScenarioResult:
    """Dispatch, AC power-flow results and validation of one scenario."""

    data: ScenarioData
    ev_power_kw: np.ndarray             # [n_ev x time]
    bess_schedule: pd.DataFrame         # charge_kw, discharge_kw
    pv_curtailment_kw: pd.DataFrame     # [time x bus]
    simulation: SimulationResult
    validation: dict[str, Any] = field(default_factory=dict)
    optimization: Any = None            # OptimizationOutcome for optimized scenarios

    @property
    def name(self) -> str:
        return self.data.scenario.name


def prepare_scenario(config: dict[str, Any], scenario: Scenario) -> ScenarioData:
    """Build the network with PV/EV/BESS elements and generate all input profiles."""
    timestamps = time_index(config)
    dt_h = step_hours(config)
    net = build_network(config)

    base = net.load[net.load.element_type == "base_load"]
    commercial_share = float(base.loc[base.profile == "commercial", "p_nom_mw"].sum() / base.p_nom_mw.sum())
    load_multipliers = load_profile_from_config(config, timestamps, commercial_share)
    base_load_kw, base_load_kvar = _base_load_by_bus(base, load_multipliers)

    pv_capacity_mw = feeder_peak_load_mw(net) * scenario.pv_penetration_pct / 100.0
    if pv_capacity_mw > 0:
        add_pv_generators(net, pv_capacity_mw)
        pv_pu = pv_profile_from_config(config, timestamps, installed_capacity_kw=1.0)["pv_pu"]
        capacities_kw = net.sgen.set_index("bus").sn_mva * 1000.0
        pv_available_kw = pd.DataFrame(np.outer(pv_pu, capacities_kw), index=timestamps,
                                       columns=capacities_kw.index.astype(int))
    else:
        pv_available_kw = pd.DataFrame(index=timestamps)

    if scenario.ev_level is not None:
        n_evs = config["ev"]["levels"][scenario.ev_level]
        res_buses = residential_buses(net)
        weights = base.set_index("bus").loc[res_buses, "p_nom_mw"].tolist()
        fleet = generate_ev_fleet(config, timestamps, n_evs, res_buses, weights)
        add_ev_loads(net, fleet.bus.tolist())
    else:
        fleet = generate_ev_fleet(config, timestamps, 0, [], [])

    bess = None
    if scenario.bess_capacity_kwh > 0:
        bess = BatteryParameters.from_config(config, scenario.bess_capacity_kwh)
        add_bess(net, bess.bus, bess.capacity_kwh, bess.max_charge_kw)

    return ScenarioData(
        config=config, scenario=scenario, net=net, timestamps=timestamps, dt_h=dt_h,
        load_multipliers=load_multipliers, base_load_kw=base_load_kw, base_load_kvar=base_load_kvar,
        pv_available_kw=pv_available_kw, fleet=fleet, price=price_profile_from_config(config, timestamps),
        bess=bess,
    )


def _base_load_by_bus(base: pd.DataFrame, multipliers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Active/reactive base load per bus in kW/kvar (constant power factor per load)."""
    mult = np.where(
        (base.profile == "residential").to_numpy()[None, :],
        multipliers.residential_multiplier.to_numpy()[:, None],
        multipliers.commercial_multiplier.to_numpy()[:, None],
    )
    buses = base.bus.astype(int).to_numpy()
    p = pd.DataFrame(mult * base.p_nom_mw.to_numpy() * 1000.0, index=multipliers.index, columns=buses)
    q = pd.DataFrame(mult * base.q_nom_mvar.to_numpy() * 1000.0, index=multipliers.index, columns=buses)
    return p.T.groupby(level=0).sum().T, q.T.groupby(level=0).sum().T


def net_demand_kw(data: ScenarioData, ev_power_kw: np.ndarray) -> pd.Series:
    """Feeder active net demand (load + EV - PV) without losses, in kW."""
    pv = data.pv_available_kw.sum(axis=1) if len(data.pv_available_kw.columns) else 0.0
    return data.base_load_kw.sum(axis=1) + ev_power_kw.sum(axis=0) - pv


def simulate_dispatch(data: ScenarioData, ev_power_kw: np.ndarray, bess_schedule: pd.DataFrame,
                      pv_curtailment_kw: pd.DataFrame | None = None) -> SimulationResult:
    """Apply a dispatch to a fresh copy of the network and run the AC time series."""
    net = copy.deepcopy(data.net)  # independent copy (tap position, results)
    inputs = TimeSeriesInputs(
        timestamps=data.timestamps,
        load_multipliers=data.load_multipliers,
        pv_available_kw=data.pv_available_kw,
        pv_curtailment_kw=pv_curtailment_kw,
        ev_kw=aggregate_by_bus(ev_power_kw, data.fleet, data.timestamps),
        bess_kw=bess_schedule.charge_kw - bess_schedule.discharge_kw,
    )
    return run_time_series(net, inputs, data.dt_h)


def validate_result(data: ScenarioData, ev_power_kw: np.ndarray, bess_schedule: pd.DataFrame,
                    simulation: SimulationResult) -> dict[str, Any]:
    """Independent physical checks of a dispatched and simulated scenario.

    Raises on EV/BESS constraint violations or an energy-balance residual above
    the configured tolerance; these indicate modelling errors, not grid issues.
    """
    validation: dict[str, Any] = {}
    if len(data.fleet):
        report = validate_ev_schedule(ev_power_kw, data.fleet, data.dt_h)
        validation["ev_min_final_soc"] = float(report.final_soc.min())
        validation["ev_energy_shortfall_kwh"] = float((report.required_kwh - report.delivered_kwh).clip(lower=0).sum())
    if data.bess is not None:
        soc = validate_battery_schedule(data.bess, bess_schedule.charge_kw.to_numpy(),
                                        bess_schedule.discharge_kw.to_numpy(), data.dt_h)
        validation["bess_soc"] = soc
    validation["power_flow_converged_steps"] = int(simulation.converged.sum())
    if not simulation.all_converged:
        failed = simulation.converged.index[~simulation.converged]
        raise RuntimeError(f"{data.scenario.name}: power flow failed at {list(failed)}")
    residual = energy_balance_residual_kwh(simulation)
    validation["energy_balance_residual_kwh"] = residual
    tolerance = data.config["limits"]["energy_balance_tolerance_kwh"]
    if abs(residual) > tolerance:
        raise RuntimeError(f"{data.scenario.name}: energy balance residual {residual:.3f} kWh > {tolerance} kWh")
    return validation


def run_scenario(config: dict[str, Any], scenario: Scenario) -> ScenarioResult:
    """Prepare, dispatch, simulate and validate one scenario."""
    logger.info("Running scenario %s", scenario.name)
    data = prepare_scenario(config, scenario)
    n_steps = len(data.timestamps)

    if scenario.is_optimized:
        from src.optimization.validation import optimize_and_validate  # local import: heavy dependency

        outcome = optimize_and_validate(data)
        result = ScenarioResult(
            data=data, ev_power_kw=outcome.ev_power_kw, bess_schedule=outcome.bess_schedule,
            pv_curtailment_kw=outcome.pv_curtailment_kw, simulation=outcome.simulation, optimization=outcome,
        )
    else:
        ev_power = (uncontrolled_charging(data.fleet, n_steps, data.dt_h)
                    if scenario.ev_control == "uncontrolled" else np.zeros((0, n_steps)))
        bess_schedule = pd.DataFrame({"charge_kw": 0.0, "discharge_kw": 0.0}, index=data.timestamps)
        if scenario.bess_control == "rule_based":
            rule = config["bess"]["rule_based"]
            bess_schedule = rule_based_dispatch(
                data.bess, net_demand_kw(data, ev_power), data.dt_h,
                charge_below_kw=rule["charge_below_mw"] * 1000.0,
                discharge_above_kw=rule["discharge_above_mw"] * 1000.0,
            )[["charge_kw", "discharge_kw"]]
        curtailment = pd.DataFrame(0.0, index=data.timestamps, columns=data.pv_available_kw.columns)
        simulation = simulate_dispatch(data, ev_power, bess_schedule, curtailment)
        result = ScenarioResult(data=data, ev_power_kw=ev_power, bess_schedule=bess_schedule,
                                pv_curtailment_kw=curtailment, simulation=simulation)

    result.validation = validate_result(data, result.ev_power_kw, result.bess_schedule, result.simulation)
    logger.info("Scenario %s done: energy balance residual %.4f kWh",
                scenario.name, result.validation["energy_balance_residual_kwh"])
    return result
