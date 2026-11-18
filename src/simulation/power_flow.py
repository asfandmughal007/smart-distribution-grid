"""Quasi-static time-series AC power flow.

For every time step the base loads, PV infeed, EV charging and BESS power are
written to the pandapower network, a Newton-Raphson power flow is run (with
the OLTC controller active), and bus, line, transformer and system results
are stored. The OLTC tap position carries over between steps, as in a real
substation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandapower as pp
import pandas as pd

from src.grid.network import NetworkError, run_power_flow

logger = logging.getLogger(__name__)

KW_PER_MW = 1000.0


@dataclass
class TimeSeriesInputs:
    """Exogenous and scheduled power values for one 24-h simulation (all in kW).

    Attributes:
        load_multipliers: DataFrame with ``residential_multiplier`` and
            ``commercial_multiplier`` columns.
        pv_available_kw: [time x bus] available PV power (before curtailment).
        pv_curtailment_kw: [time x bus] curtailed PV power (same shape) or None.
        ev_kw: [time x bus] aggregated EV charging power.
        bess_kw: Signed BESS power (+ charging, - discharging) or None.
    """

    timestamps: pd.DatetimeIndex
    load_multipliers: pd.DataFrame
    pv_available_kw: pd.DataFrame
    ev_kw: pd.DataFrame
    pv_curtailment_kw: pd.DataFrame | None = None
    bess_kw: pd.Series | None = None

    def __post_init__(self) -> None:
        for name in ("load_multipliers", "pv_available_kw", "ev_kw"):
            frame = getattr(self, name)
            if len(frame) and not frame.index.equals(self.timestamps):
                raise ValueError(f"{name} index does not match timestamps")
        if self.pv_curtailment_kw is None:
            self.pv_curtailment_kw = pd.DataFrame(0.0, index=self.timestamps, columns=self.pv_available_kw.columns)
        if ((self.pv_curtailment_kw - self.pv_available_kw) > 1e-6).any().any():
            raise ValueError("PV curtailment exceeds available PV power")
        if self.bess_kw is None:
            self.bess_kw = pd.Series(0.0, index=self.timestamps)


@dataclass
class SimulationResult:
    """Time-series power-flow results. Powers in kW, voltages in pu, loading in %."""

    bus_vm_pu: pd.DataFrame
    line_loading_pct: pd.DataFrame
    line_p_kw: pd.DataFrame
    line_q_kvar: pd.DataFrame
    trafo: pd.DataFrame          # loading_pct, p_hv_kw, q_hv_kvar, tap_pos
    system: pd.DataFrame         # system-level energy flows per step
    converged: pd.Series
    dt_h: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def all_converged(self) -> bool:
        return bool(self.converged.all())


def run_time_series(net: pp.pandapowerNet, inputs: TimeSeriesInputs, dt_h: float) -> SimulationResult:
    """Run the AC power flow for every time step.

    Non-converged steps are logged with their operating point and stored as
    NaN; they are never silently dropped (see ``SimulationResult.converged``).
    """
    base_mask = net.load.element_type == "base_load"
    base_idx = net.load.index[base_mask]
    is_residential = (net.load.loc[base_idx, "profile"] == "residential").to_numpy()
    p_nom = net.load.loc[base_idx, "p_nom_mw"].to_numpy()
    q_nom = net.load.loc[base_idx, "q_nom_mvar"].to_numpy()

    pv_idx_by_bus = {int(bus): idx for idx, bus in net.sgen.bus.items()}
    ev_idx_by_bus = {int(row.bus): idx for idx, row in net.load[net.load.element_type == "ev"].iterrows()}
    storage_idx = net.storage.index[0] if len(net.storage) else None
    _check_columns(inputs.pv_available_kw.columns, pv_idx_by_bus, "PV")
    _check_columns(inputs.ev_kw.columns, ev_idx_by_bus, "EV")
    if storage_idx is None and (inputs.bess_kw.abs() > 1e-9).any():
        raise ValueError("BESS schedule given but network has no storage element")

    pv_net_kw = inputs.pv_available_kw - inputs.pv_curtailment_kw
    mv_buses = net.bus.index[net.bus.vn_kv < net.trafo.vn_hv_kv.iloc[0]]

    n = len(inputs.timestamps)
    vm = np.full((n, len(mv_buses)), np.nan)
    line_loading = np.full((n, len(net.line)), np.nan)
    line_p = np.full_like(line_loading, np.nan)
    line_q = np.full_like(line_loading, np.nan)
    trafo_rows, system_rows = [], []
    converged = np.zeros(n, dtype=bool)

    for t, stamp in enumerate(inputs.timestamps):
        mult = np.where(is_residential, inputs.load_multipliers.residential_multiplier.iloc[t],
                        inputs.load_multipliers.commercial_multiplier.iloc[t])
        net.load.loc[base_idx, "p_mw"] = p_nom * mult
        net.load.loc[base_idx, "q_mvar"] = q_nom * mult
        for bus, idx in pv_idx_by_bus.items():
            net.sgen.at[idx, "p_mw"] = pv_net_kw.at[stamp, bus] / KW_PER_MW
        for bus, idx in ev_idx_by_bus.items():
            net.load.at[idx, "p_mw"] = inputs.ev_kw.at[stamp, bus] / KW_PER_MW
        bess_kw = float(inputs.bess_kw.iloc[t])
        if storage_idx is not None:
            net.storage.at[storage_idx, "p_mw"] = bess_kw / KW_PER_MW

        base_load_kw = float(net.load.loc[base_idx, "p_mw"].sum() * KW_PER_MW)
        ev_kw = float(inputs.ev_kw.iloc[t].sum()) if len(inputs.ev_kw.columns) else 0.0
        pv_avail = float(inputs.pv_available_kw.iloc[t].sum()) if len(inputs.pv_available_kw.columns) else 0.0
        pv_curt = float(inputs.pv_curtailment_kw.iloc[t].sum()) if len(inputs.pv_curtailment_kw.columns) else 0.0

        try:
            run_power_flow(net)
            converged[t] = True
        except NetworkError:
            logger.error(
                "Power flow did not converge at %s (step %d): load %.0f kW, EV %.0f kW, PV %.0f kW, BESS %.0f kW",
                stamp, t, base_load_kw, ev_kw, pv_avail - pv_curt, bess_kw,
            )

        if converged[t]:
            vm[t] = net.res_bus.loc[mv_buses, "vm_pu"].to_numpy()
            line_loading[t] = net.res_line.loading_percent.to_numpy()
            line_p[t] = net.res_line.p_from_mw.to_numpy() * KW_PER_MW
            line_q[t] = net.res_line.q_from_mvar.to_numpy() * KW_PER_MW
            grid_kw = float(net.res_ext_grid.p_mw.sum() * KW_PER_MW)
            losses_kw = float((net.res_line.pl_mw.sum() + net.res_trafo.pl_mw.sum()) * KW_PER_MW)
            trafo_rows.append({
                "loading_pct": float(net.res_trafo.loading_percent.iloc[0]),
                "p_hv_kw": float(net.res_trafo.p_hv_mw.iloc[0] * KW_PER_MW),
                "q_hv_kvar": float(net.res_trafo.q_hv_mvar.iloc[0] * KW_PER_MW),
                "tap_pos": float(net.trafo.tap_pos.iloc[0]),
            })
        else:
            grid_kw = losses_kw = np.nan
            trafo_rows.append({"loading_pct": np.nan, "p_hv_kw": np.nan, "q_hv_kvar": np.nan, "tap_pos": np.nan})

        system_rows.append({
            "grid_kw": grid_kw,
            "grid_import_kw": max(grid_kw, 0.0) if converged[t] else np.nan,
            "grid_export_kw": max(-grid_kw, 0.0) if converged[t] else np.nan,
            "losses_kw": losses_kw,
            "load_kw": base_load_kw,
            "ev_kw": ev_kw,
            "pv_available_kw": pv_avail,
            "pv_curtailed_kw": pv_curt,
            "pv_kw": pv_avail - pv_curt,
            "bess_charge_kw": max(bess_kw, 0.0),
            "bess_discharge_kw": max(-bess_kw, 0.0),
        })

    n_failed = int((~converged).sum())
    if n_failed:
        logger.error("%d of %d time steps did not converge", n_failed, n)

    index = inputs.timestamps
    return SimulationResult(
        bus_vm_pu=pd.DataFrame(vm, index=index, columns=mv_buses),
        line_loading_pct=pd.DataFrame(line_loading, index=index, columns=net.line.index),
        line_p_kw=pd.DataFrame(line_p, index=index, columns=net.line.index),
        line_q_kvar=pd.DataFrame(line_q, index=index, columns=net.line.index),
        trafo=pd.DataFrame(trafo_rows, index=index),
        system=pd.DataFrame(system_rows, index=index),
        converged=pd.Series(converged, index=index),
        dt_h=dt_h,
    )


def _check_columns(columns: pd.Index, element_map: dict[int, int], label: str) -> None:
    missing = set(int(c) for c in columns) - set(element_map)
    if missing:
        raise ValueError(f"{label} profile has buses without a network element: {sorted(missing)}")


def energy_balance_residual_kwh(result: SimulationResult) -> float:
    """Residual of the 24-h energy balance in kWh (should be ~0).

    Import + PV + BESS discharge = Load + EV + BESS charge + losses + export
    """
    s = result.system
    supply = s.grid_import_kw + s.pv_kw + s.bess_discharge_kw
    demand = s.load_kw + s.ev_kw + s.bess_charge_kw + s.losses_kw + s.grid_export_kw
    return float(((supply - demand) * result.dt_h).sum())
