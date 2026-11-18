"""Battery energy storage system (BESS) model.

Sign convention: positive power = charging (load), negative = discharging,
grid-side active power in kW. SOC dynamics per step of length dt_h:

    E(t+1) = E(t) + eta_ch * P_ch(t) * dt_h - P_dis(t) * dt_h / eta_dis

A single signed setpoint per step makes simultaneous charging and discharging
impossible by construction in the simulation model; the optimization model
enforces the same with a binary variable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


class BatteryScheduleError(ValueError):
    """Raised when a BESS schedule violates a physical limit."""


@dataclass(frozen=True)
class BatteryParameters:
    """Static BESS parameters (energies in kWh, powers in kW, SOC as fraction)."""

    capacity_kwh: float
    max_charge_kw: float
    max_discharge_kw: float
    soc_min: float = 0.1
    soc_max: float = 0.9
    soc_initial: float = 0.5
    charge_efficiency: float = 0.95
    discharge_efficiency: float = 0.95
    degradation_cost_eur_per_kwh: float = 0.0
    bus: int = 0

    def __post_init__(self) -> None:
        if self.capacity_kwh < 0 or self.max_charge_kw < 0 or self.max_discharge_kw < 0:
            raise ValueError("Capacity and power ratings must be non-negative")
        if not 0.0 <= self.soc_min <= self.soc_initial <= self.soc_max <= 1.0:
            raise ValueError("Require 0 <= soc_min <= soc_initial <= soc_max <= 1")
        if not (0 < self.charge_efficiency <= 1 and 0 < self.discharge_efficiency <= 1):
            raise ValueError("Efficiencies must be in (0, 1]")

    @property
    def energy_min_kwh(self) -> float:
        return self.soc_min * self.capacity_kwh

    @property
    def energy_max_kwh(self) -> float:
        return self.soc_max * self.capacity_kwh

    @property
    def energy_initial_kwh(self) -> float:
        return self.soc_initial * self.capacity_kwh

    @classmethod
    def from_config(cls, config: dict[str, Any], capacity_kwh: float) -> "BatteryParameters":
        """Build parameters from the ``bess`` config section (power = c_rate x capacity)."""
        cfg = config["bess"]
        power = cfg["c_rate"] * capacity_kwh
        return cls(
            capacity_kwh=capacity_kwh, max_charge_kw=power, max_discharge_kw=power,
            soc_min=cfg["soc_min"], soc_max=cfg["soc_max"], soc_initial=cfg["soc_initial"],
            charge_efficiency=cfg["charge_efficiency"], discharge_efficiency=cfg["discharge_efficiency"],
            degradation_cost_eur_per_kwh=cfg["degradation_cost_eur_per_kwh"], bus=cfg["bus"],
        )


class Battery:
    """Stateful BESS that clips requested setpoints to all physical limits."""

    def __init__(self, params: BatteryParameters) -> None:
        self.params = params
        self.energy_kwh = params.energy_initial_kwh

    @property
    def soc(self) -> float:
        return self.energy_kwh / self.params.capacity_kwh if self.params.capacity_kwh > 0 else 0.0

    def feasible_power_range(self, dt_h: float) -> tuple[float, float]:
        """(max discharge as negative kW, max charge kW) achievable in the next step."""
        p = self.params
        headroom = max(p.energy_max_kwh - self.energy_kwh, 0.0)
        available = max(self.energy_kwh - p.energy_min_kwh, 0.0)
        max_charge = min(p.max_charge_kw, headroom / (p.charge_efficiency * dt_h))
        max_discharge = min(p.max_discharge_kw, available * p.discharge_efficiency / dt_h)
        return -max_discharge, max_charge

    def step(self, requested_kw: float, dt_h: float) -> float:
        """Apply a signed power setpoint for one step and return the power actually applied."""
        low, high = self.feasible_power_range(dt_h)
        power = float(np.clip(requested_kw, low, high))
        if power >= 0:
            self.energy_kwh += power * self.params.charge_efficiency * dt_h
        else:
            self.energy_kwh += power * dt_h / self.params.discharge_efficiency
        # Guard against floating-point drift at the limits.
        self.energy_kwh = float(np.clip(self.energy_kwh, self.params.energy_min_kwh, self.params.energy_max_kwh))
        return power


def soc_from_schedule(params: BatteryParameters, charge_kw: np.ndarray, discharge_kw: np.ndarray,
                      dt_h: float) -> np.ndarray:
    """SOC trajectory (fraction, length n+1) for separate charge/discharge schedules."""
    delta = (params.charge_efficiency * charge_kw - discharge_kw / params.discharge_efficiency) * dt_h
    energy = params.energy_initial_kwh + np.concatenate([[0.0], np.cumsum(delta)])
    return energy / params.capacity_kwh if params.capacity_kwh > 0 else np.zeros_like(energy)


def validate_battery_schedule(params: BatteryParameters, charge_kw: np.ndarray, discharge_kw: np.ndarray,
                              dt_h: float, tolerance: float = 1e-6) -> np.ndarray:
    """Check a schedule against power, SOC and exclusivity limits; return the SOC trajectory.

    Raises:
        BatteryScheduleError: describing the first violated limit.
    """
    charge_kw = np.asarray(charge_kw, dtype=float)
    discharge_kw = np.asarray(discharge_kw, dtype=float)
    if (charge_kw < -tolerance).any() or (discharge_kw < -tolerance).any():
        raise BatteryScheduleError("Charge and discharge powers must be non-negative")
    if (charge_kw > params.max_charge_kw + tolerance).any():
        raise BatteryScheduleError("Charging power exceeds rating")
    if (discharge_kw > params.max_discharge_kw + tolerance).any():
        raise BatteryScheduleError("Discharging power exceeds rating")
    if ((charge_kw > tolerance) & (discharge_kw > tolerance)).any():
        raise BatteryScheduleError("Simultaneous charging and discharging")
    soc = soc_from_schedule(params, charge_kw, discharge_kw, dt_h)
    if params.capacity_kwh > 0:
        if soc.min() < params.soc_min - tolerance or soc.max() > params.soc_max + tolerance:
            raise BatteryScheduleError(f"SOC outside [{params.soc_min}, {params.soc_max}]: "
                                       f"range {soc.min():.4f}-{soc.max():.4f}")
    return soc


def rule_based_dispatch(params: BatteryParameters, net_demand_kw: pd.Series, dt_h: float,
                        charge_below_kw: float, discharge_above_kw: float) -> pd.DataFrame:
    """Simple peak-shaving / valley-filling heuristic (no forecast, no optimization).

    Charges when feeder net demand is below ``charge_below_kw`` (e.g. PV
    surplus hours) and discharges when it exceeds ``discharge_above_kw``, in
    both cases by just the amount needed to reach the threshold.
    """
    if charge_below_kw >= discharge_above_kw:
        raise ValueError("charge_below_kw must be lower than discharge_above_kw")
    battery = Battery(params)
    rows = []
    for demand in net_demand_kw.to_numpy():
        if demand > discharge_above_kw:
            request = -(demand - discharge_above_kw)
        elif demand < charge_below_kw:
            request = charge_below_kw - demand
        else:
            request = 0.0
        soc_before = battery.soc
        power = battery.step(request, dt_h)
        rows.append({"charge_kw": max(power, 0.0), "discharge_kw": max(-power, 0.0), "soc_start": soc_before})
    return pd.DataFrame(rows, index=net_demand_kw.index)
