"""Grid performance, energy and economic metrics computed from AC power-flow results.

All metrics are derived from the AC simulation (not from the optimization
model), so optimized and non-optimized scenarios are evaluated identically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.simulation.power_flow import SimulationResult


def voltage_metrics(vm_pu: pd.DataFrame, v_min: float, v_max: float, dt_h: float) -> dict[str, Any]:
    """Voltage extremes and violation counts/durations.

    A *violation step* is a time step in which at least one bus is outside
    [v_min, v_max]; *bus-step violations* count every (bus, step) pair.
    """
    under = vm_pu < v_min
    over = vm_pu > v_max
    any_violation = (under | over).any(axis=1)
    return {
        "v_min_pu": float(vm_pu.min().min()),
        "v_max_pu": float(vm_pu.max().max()),
        "v_min_bus": int(vm_pu.min().idxmin()),
        "undervoltage_bus_steps": int(under.to_numpy().sum()),
        "overvoltage_bus_steps": int(over.to_numpy().sum()),
        "voltage_violation_bus_steps": int((under | over).to_numpy().sum()),
        "voltage_violation_steps": int(any_violation.sum()),
        "voltage_violation_duration_h": float(any_violation.sum() * dt_h),
        "buses_with_violation": int((under | over).any(axis=0).sum()),
    }


def loading_metrics(line_loading_pct: pd.DataFrame, trafo_loading_pct: pd.Series,
                    line_limit_pct: float, trafo_limit_pct: float, dt_h: float) -> dict[str, Any]:
    """Line and transformer loading statistics."""
    line_max_per_step = line_loading_pct.max(axis=1)
    return {
        "line_loading_max_pct": float(line_loading_pct.max().max()),
        "line_loading_avg_pct": float(line_loading_pct.to_numpy().mean()),
        "line_overload_duration_h": float((line_max_per_step > line_limit_pct).sum() * dt_h),
        "trafo_loading_max_pct": float(trafo_loading_pct.max()),
        "trafo_loading_avg_pct": float(trafo_loading_pct.mean()),
        "trafo_overload_duration_h": float((trafo_loading_pct > trafo_limit_pct).sum() * dt_h),
    }


def energy_metrics(system: pd.DataFrame, dt_h: float) -> dict[str, float]:
    """24-h energy totals in kWh. Battery throughput = (charge + discharge) / 2."""
    energy = system.sum() * dt_h
    return {
        "grid_import_kwh": float(energy.grid_import_kw),
        "grid_export_kwh": float(energy.grid_export_kw),
        "load_kwh": float(energy.load_kw),
        "pv_available_kwh": float(energy.pv_available_kw),
        "pv_generation_kwh": float(energy.pv_kw),
        "pv_curtailment_kwh": float(energy.pv_curtailed_kw),
        "ev_energy_kwh": float(energy.ev_kw),
        "bess_charge_kwh": float(energy.bess_charge_kw),
        "bess_discharge_kwh": float(energy.bess_discharge_kw),
        "bess_throughput_kwh": float((energy.bess_charge_kw + energy.bess_discharge_kw) / 2.0),
        "losses_kwh": float(energy.losses_kw),
        "losses_pct_of_supply": float(100.0 * energy.losses_kw /
                                      max(energy.grid_import_kw + energy.pv_kw + energy.bess_discharge_kw, 1e-9)),
    }


def economic_metrics(system: pd.DataFrame, price: pd.DataFrame, dt_h: float,
                     peak_charge_eur_per_kw: float, degradation_eur_per_kwh: float) -> dict[str, float]:
    """Operating cost at the substation (EUR). Assumptions are documented in the config.

    energy cost      = sum(import * import_price - export * export_price) * dt
    peak charge      = max(import) * daily demand charge
    degradation cost = BESS throughput * degradation cost per kWh
    """
    energy_cost = float(((system.grid_import_kw * price.import_eur_per_kwh
                          - system.grid_export_kw * price.export_eur_per_kwh) * dt_h).sum())
    peak_kw = float(system.grid_import_kw.max())
    throughput = float(((system.bess_charge_kw + system.bess_discharge_kw) / 2.0).sum() * dt_h)
    peak_cost = peak_kw * peak_charge_eur_per_kw
    degradation = throughput * degradation_eur_per_kwh
    return {
        "peak_import_kw": peak_kw,
        "energy_cost_eur": energy_cost,
        "peak_demand_charge_eur": peak_cost,
        "bess_degradation_cost_eur": degradation,
        "total_cost_eur": energy_cost + peak_cost + degradation,
    }


def compute_metrics(simulation: SimulationResult, price: pd.DataFrame, config: dict[str, Any]) -> dict[str, Any]:
    """All metrics for one simulated scenario."""
    limits = config["limits"]
    dt_h = simulation.dt_h
    metrics: dict[str, Any] = {}
    metrics.update(voltage_metrics(simulation.bus_vm_pu, limits["v_min_pu"], limits["v_max_pu"], dt_h))
    metrics.update(loading_metrics(simulation.line_loading_pct, simulation.trafo.loading_pct,
                                   limits["line_max_loading_pct"], limits["trafo_max_loading_pct"], dt_h))
    metrics.update(energy_metrics(simulation.system, dt_h))
    metrics.update(economic_metrics(simulation.system, price, dt_h,
                                    config["price"]["peak_demand_charge_eur_per_kw"],
                                    config["bess"]["degradation_cost_eur_per_kwh"]))
    metrics["min_grid_kw"] = float(simulation.system.grid_kw.min())
    metrics["reverse_flow_duration_h"] = float((simulation.system.grid_kw < 0).sum() * dt_h)
    return metrics


# Columns of the published comparison table: (metric key, column header, decimals)
SUMMARY_COLUMNS = [
    ("peak_import_kw", "Peak grid import [kW]", 0),
    ("v_min_pu", "Min voltage [pu]", 4),
    ("v_max_pu", "Max voltage [pu]", 4),
    ("voltage_violation_duration_h", "Voltage violation [h]", 2),
    ("voltage_violation_bus_steps", "Violation bus-steps [-]", 0),
    ("line_loading_max_pct", "Max line loading [%]", 1),
    ("trafo_loading_max_pct", "Max trafo loading [%]", 1),
    ("losses_kwh", "Grid losses [kWh]", 0),
    ("grid_import_kwh", "Grid import [kWh]", 0),
    ("grid_export_kwh", "Grid export [kWh]", 0),
    ("pv_curtailment_kwh", "PV curtailment [kWh]", 0),
    ("ev_energy_kwh", "EV energy [kWh]", 0),
    ("bess_throughput_kwh", "BESS throughput [kWh]", 0),
    ("energy_cost_eur", "Energy cost [EUR]", 0),
    ("total_cost_eur", "Total cost [EUR]", 0),
]


def summary_table(metrics_by_scenario: dict[str, dict[str, Any]]) -> pd.DataFrame:
    """Comparison table (one row per scenario) with readable, unit-bearing headers."""
    rows = []
    for label, metrics in metrics_by_scenario.items():
        row = {"Scenario": label}
        for key, header, decimals in SUMMARY_COLUMNS:
            value = metrics[key]
            row[header] = round(value, decimals) if decimals else int(round(value))
        rows.append(row)
    return pd.DataFrame(rows)


def write_table(table: pd.DataFrame, csv_path: str | Path, markdown_path: str | Path | None = None) -> None:
    """Write a table as CSV and (optionally) as a GitHub-flavoured Markdown table."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(csv_path, index=False)
    if markdown_path is not None:
        Path(markdown_path).write_text(to_markdown(table) + "\n", encoding="utf-8")


def to_markdown(table: pd.DataFrame) -> str:
    """Render a DataFrame as a Markdown table without the optional ``tabulate`` dependency."""
    header = "| " + " | ".join(str(c) for c in table.columns) + " |"
    align = "|" + "|".join(":---" if i == 0 else "---:" for i in range(len(table.columns))) + "|"
    body = []
    for row in table.itertuples(index=False):
        cells = [f"{v:,.4f}".rstrip("0").rstrip(".") if isinstance(v, float) and not np.isnan(v)
                 else (f"{v:,}" if isinstance(v, (int, np.integer)) else str(v)) for v in row]
        body.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, align, *body])
