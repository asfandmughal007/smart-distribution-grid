"""EV charging model: session bookkeeping, uncontrolled charging and validation.

Sign/units convention: charging power is grid-side active power in kW (>= 0).
Energy stored in the battery per step = ``power * efficiency * dt_h``.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FLEET_COLUMNS = [
    "ev_id", "bus", "arrival", "departure", "capacity_kwh", "initial_soc", "target_soc",
    "max_power_kw", "efficiency",
]


class EVScheduleError(ValueError):
    """Raised when an EV charging schedule violates a session constraint."""


def build_fleet(records: list[dict[str, Any]], timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Create a fleet table and derive time-step indices and energy requirements.

    Derived columns:
        arrival_step / departure_step: connected for steps ``arrival_step <= t < departure_step``.
        required_energy_kwh: battery-side energy needed to reach ``target_soc``.
        max_deliverable_kwh: energy achievable at full power while connected.
        target_feasible: False if the target cannot be reached; the requirement is
            then reduced to the deliverable energy (and a warning is logged).
    """
    fleet = pd.DataFrame(records, columns=FLEET_COLUMNS)
    if fleet.empty:
        for col in ("arrival_step", "departure_step"):
            fleet[col] = pd.Series(dtype=int)
        for col in ("required_energy_kwh", "max_deliverable_kwh"):
            fleet[col] = pd.Series(dtype=float)
        fleet["target_feasible"] = pd.Series(dtype=bool)
        return fleet

    if (fleet.departure <= fleet.arrival).any():
        raise ValueError("Every EV must depart after it arrives")
    if not ((fleet.initial_soc >= 0) & (fleet.target_soc <= 1)).all():
        raise ValueError("SOC values must be within [0, 1]")

    start = timestamps[0]
    dt = timestamps[1] - timestamps[0]
    n_steps = len(timestamps)
    dt_h = dt.total_seconds() / 3600.0

    arrival_step = ((fleet.arrival - start) / dt).apply(np.ceil).astype(int).clip(0, n_steps)
    departure_step = ((fleet.departure - start) / dt).apply(np.floor).astype(int).clip(0, n_steps)
    clipped = (fleet.departure > start + n_steps * dt).sum()
    if clipped:
        logger.warning("%d EV departures lie beyond the horizon and were clipped", clipped)
    fleet["arrival_step"] = arrival_step
    fleet["departure_step"] = departure_step

    connected_h = (departure_step - arrival_step).clip(lower=0) * dt_h
    fleet["max_deliverable_kwh"] = fleet.max_power_kw * fleet.efficiency * connected_h
    required = ((fleet.target_soc - fleet.initial_soc).clip(lower=0) * fleet.capacity_kwh)
    fleet["target_feasible"] = required <= fleet.max_deliverable_kwh + 1e-9
    n_infeasible = int((~fleet.target_feasible).sum())
    if n_infeasible:
        logger.warning("%d EVs cannot reach their target SOC; requirement capped at deliverable energy",
                       n_infeasible)
    fleet["required_energy_kwh"] = np.minimum(required, fleet.max_deliverable_kwh)
    return fleet


def connection_matrix(fleet: pd.DataFrame, n_steps: int) -> np.ndarray:
    """Boolean array [n_ev, n_steps]: True while the EV is plugged in."""
    steps = np.arange(n_steps)
    return (steps[None, :] >= fleet.arrival_step.to_numpy()[:, None]) & (
        steps[None, :] < fleet.departure_step.to_numpy()[:, None]
    )


def uncontrolled_charging(fleet: pd.DataFrame, n_steps: int, dt_h: float) -> np.ndarray:
    """Plug-in-and-charge schedule: full power from arrival until the requirement is met.

    Returns grid-side charging power in kW, shape [n_ev, n_steps]. The last
    charging step uses partial power so exactly the required energy is delivered.
    """
    power = np.zeros((len(fleet), n_steps))
    for i, ev in enumerate(fleet.itertuples(index=False)):
        remaining = ev.required_energy_kwh  # battery side
        for t in range(ev.arrival_step, ev.departure_step):
            if remaining <= 1e-9:
                break
            stored = min(ev.max_power_kw * ev.efficiency * dt_h, remaining)
            power[i, t] = stored / (ev.efficiency * dt_h)
            remaining -= stored
    return power


def soc_trajectory(power_kw: np.ndarray, fleet: pd.DataFrame, dt_h: float) -> np.ndarray:
    """SOC (fraction) at the start of every step plus the final value, shape [n_ev, n_steps + 1]."""
    stored = power_kw * fleet.efficiency.to_numpy()[:, None] * dt_h
    energy = np.concatenate([np.zeros((len(fleet), 1)), np.cumsum(stored, axis=1)], axis=1)
    return fleet.initial_soc.to_numpy()[:, None] + energy / fleet.capacity_kwh.to_numpy()[:, None]


def delivered_energy_kwh(power_kw: np.ndarray, fleet: pd.DataFrame, dt_h: float) -> np.ndarray:
    """Battery-side energy delivered to every EV over the horizon."""
    return (power_kw * dt_h).sum(axis=1) * fleet.efficiency.to_numpy()


def validate_ev_schedule(power_kw: np.ndarray, fleet: pd.DataFrame, dt_h: float,
                         tolerance_kwh: float = 0.01) -> pd.DataFrame:
    """Check a schedule against all session constraints.

    Raises:
        EVScheduleError: on charging while disconnected, power above the
            charger rating, negative power, or unmet energy requirements.

    Returns:
        Per-EV table with required/delivered energy and final SOC.
    """
    if power_kw.shape[0] != len(fleet):
        raise EVScheduleError("Schedule rows must match the fleet")
    connected = connection_matrix(fleet, power_kw.shape[1])
    tol_kw = 1e-6
    if (power_kw < -tol_kw).any():
        raise EVScheduleError("Negative EV charging power (V2G is not modelled)")
    if (power_kw[~connected] > tol_kw).any():
        raise EVScheduleError("EV charging while not connected")
    if (power_kw > fleet.max_power_kw.to_numpy()[:, None] + tol_kw).any():
        raise EVScheduleError("EV charging power exceeds charger rating")
    delivered = delivered_energy_kwh(power_kw, fleet, dt_h)
    shortfall = fleet.required_energy_kwh.to_numpy() - delivered
    if (shortfall > tolerance_kwh).any():
        worst = fleet.ev_id.iloc[int(np.argmax(shortfall))]
        raise EVScheduleError(f"EV energy requirement not met (worst: {worst}, {shortfall.max():.3f} kWh)")
    final_soc = soc_trajectory(power_kw, fleet, dt_h)[:, -1]
    if (final_soc > 1.0 + 1e-6).any():
        raise EVScheduleError("EV SOC exceeds 100 %")
    return pd.DataFrame({
        "ev_id": fleet.ev_id, "bus": fleet.bus,
        "required_kwh": fleet.required_energy_kwh, "delivered_kwh": delivered,
        "final_soc": final_soc,
    })


def aggregate_by_bus(power_kw: np.ndarray, fleet: pd.DataFrame,
                     timestamps: pd.DatetimeIndex) -> pd.DataFrame:
    """Sum EV charging power per bus. Returns a [time x bus] DataFrame in kW."""
    if len(fleet) == 0:
        return pd.DataFrame(index=timestamps)
    df = pd.DataFrame(power_kw.T, index=timestamps, columns=fleet.bus.to_numpy())
    return df.T.groupby(level=0).sum().T
