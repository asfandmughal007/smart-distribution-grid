"""Engineering figures generated from simulation and experiment results.

Every function takes result objects / tables produced by the simulation and
writes one PNG to ``results/figures``. No numbers are hard-coded.

Styling: a fixed categorical palette (validated for colour-vision deficiency;
the lighter hues are always paired with a legend and a distinct line style),
thin marks, recessive grid, units on every axis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless rendering for scripts and CI
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Categorical slots in fixed order (never cycled) + text/grid tokens.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1)), (0, (1, 1)), (0, (8, 2))]
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3df"
LIMIT = "#d03b3b"        # reserved status colour (critical) for operating limits only
SURFACE = "#fcfcfb"


def apply_style() -> None:
    """Project-wide matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold", "axes.labelsize": 10,
        "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
        "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "legend.frameon": False, "legend.fontsize": 9, "lines.linewidth": 1.8,
        "figure.dpi": 100,
    })


def _time_axis(ax: plt.Axes) -> None:
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=range(0, 24, 3)))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set_xlabel("Time of day (noon-to-noon horizon)")


def _limit_line(ax: plt.Axes, value: float, label: str, horizontal: bool = True) -> None:
    (ax.axhline if horizontal else ax.axvline)(value, color=LIMIT, linewidth=1.2, linestyle=(0, (4, 2)), zorder=1)
    if horizontal:
        ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"), xytext=(-4, 3),
                    textcoords="offset points", ha="right", va="bottom", fontsize=8, color=LIMIT)


def _legend_below(ax: plt.Axes, ncol: int = 4) -> None:
    """Legend under the x-axis label so it never covers data."""
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=ncol)


def _save(fig: plt.Figure, out_dir: Path, name: str, dpi: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def _series(ax: plt.Axes, x: Any, y: Any, slot: int, label: str, **kwargs: Any) -> None:
    ax.plot(x, y, color=SERIES[slot], linestyle=LINESTYLES[slot], label=label, **kwargs)


# --------------------------------------------------------------------------- 1-4 inputs
def plot_load_profile(result: Any, out_dir: Path, dpi: int) -> Path:
    data = result.data
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
    lm = data.load_multipliers
    _series(axes[0], lm.index, lm.residential_multiplier, 0, "Residential")
    _series(axes[0], lm.index, lm.commercial_multiplier, 1, "Commercial")
    axes[0].set(title="Normalized load shapes", ylabel="Load multiplier [p.u. of nominal]", ylim=(0, 1.1))
    _time_axis(axes[0])
    axes[0].legend(loc="lower right")
    total = data.base_load_kw.sum(axis=1)
    _series(axes[1], total.index, total / 1000, 0, "Feeder base load")
    nominal = data.net.load.loc[data.net.load.element_type == "base_load", "p_nom_mw"].sum()
    _limit_line(axes[1], nominal, f"Sum of nominal loads {nominal:.2f} MW")
    axes[1].set(title="Feeder base load (IEEE 33-bus)", ylabel="Active power [MW]", ylim=(0, nominal * 1.15))
    _time_axis(axes[1])
    return _save(fig, out_dir, "01_load_profile.png", dpi)


def plot_pv_profile(result: Any, out_dir: Path, dpi: int) -> Path:
    data = result.data
    pv = data.pv_available_kw.sum(axis=1) / 1000
    capacity = data.net.sgen.sn_mva.sum()
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.fill_between(pv.index, pv, color=SERIES[3], alpha=0.25, linewidth=0)
    _series(ax, pv.index, pv, 3, f"Available PV ({capacity:.2f} MWp installed)")
    ax.set(title=f"PV generation profile - {data.scenario.pv_penetration_pct:.0f} % penetration",
           ylabel="Active power [MW]")
    _time_axis(ax)
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "02_pv_profile.png", dpi)


def plot_ev_profile(uncontrolled: Any, optimized: Any, out_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(8, 3.8))
    idx = uncontrolled.data.timestamps
    _series(ax, idx, uncontrolled.ev_power_kw.sum(axis=0), 1, "Uncontrolled (plug-in and charge)")
    _series(ax, idx, optimized.ev_power_kw.sum(axis=0), 0, "Optimized (smart charging)")
    n = len(uncontrolled.data.fleet)
    ax.set(title=f"Aggregate EV charging power - {n} EVs", ylabel="Charging power [kW]")
    _time_axis(ax)
    _legend_below(ax)
    return _save(fig, out_dir, "03_ev_charging_profile.png", dpi)


def plot_price(result: Any, out_dir: Path, dpi: int) -> Path:
    price = result.data.price
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.step(price.index, price.import_eur_per_kwh * 100, where="post", color=SERIES[0],
            label="Import (time-of-use)")
    ax.step(price.index, price.export_eur_per_kwh * 100, where="post", color=SERIES[1],
            linestyle=LINESTYLES[1], label="Export (feed-in)")
    ax.set(title="Synthetic time-of-use electricity price (simulation input)", ylabel="Price [ct/kWh]",
           ylim=(0, price.import_eur_per_kwh.max() * 115))
    _time_axis(ax)
    ax.legend(loc="upper right")
    return _save(fig, out_dir, "04_electricity_price.png", dpi)


# --------------------------------------------------------------------------- 5-9 outcomes
def plot_grid_import(results: dict[str, Any], out_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    for slot, (label, res) in enumerate(results.items()):
        _series(ax, res.data.timestamps, res.simulation.system.grid_kw / 1000, slot, label)
    ax.axhline(0, color=INK_2, linewidth=0.8)
    ax.set(title="Substation active power before vs after optimization",
           ylabel="Grid exchange [MW]  (+ import / - export)")
    _time_axis(ax)
    _legend_below(ax)
    return _save(fig, out_dir, "05_grid_import.png", dpi)


def plot_transformer_loading(results: dict[str, Any], limit_pct: float, out_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    for slot, (label, res) in enumerate(results.items()):
        _series(ax, res.data.timestamps, res.simulation.trafo.loading_pct, slot, label)
    _limit_line(ax, limit_pct, f"Rated loading {limit_pct:.0f} %")
    ax.set(title="HV/MV transformer loading", ylabel="Loading [% of 5 MVA]", ylim=(0, limit_pct * 1.1))
    _time_axis(ax)
    _legend_below(ax)
    return _save(fig, out_dir, "06_transformer_loading.png", dpi)


def plot_min_voltage(results: dict[str, Any], v_min: float, out_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    for slot, (label, res) in enumerate(results.items()):
        _series(ax, res.data.timestamps, res.simulation.bus_vm_pu.min(axis=1), slot, label)
    _limit_line(ax, v_min, f"Lower limit {v_min:.2f} pu")
    ax.set(title="Minimum feeder voltage over time", ylabel="Minimum bus voltage [pu]")
    _time_axis(ax)
    _legend_below(ax)
    return _save(fig, out_dir, "07_min_voltage.png", dpi)


def plot_battery(results: dict[str, Any], out_dir: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    params = None
    for slot, (label, res) in enumerate(results.items()):
        params = res.data.bess
        soc = res.validation["bess_soc"]
        times = res.data.timestamps.append(pd.DatetimeIndex([res.data.timestamps[-1] + pd.Timedelta(hours=res.data.dt_h)]))
        _series(axes[0], times, soc * 100, slot, label)
        net = res.bess_schedule.charge_kw - res.bess_schedule.discharge_kw
        axes[1].step(net.index, net, where="post", color=SERIES[slot], linestyle=LINESTYLES[slot], label=label)
    _limit_line(axes[0], params.soc_max * 100, f"SOC max {params.soc_max:.0%}")
    _limit_line(axes[0], params.soc_min * 100, f"SOC min {params.soc_min:.0%}")
    axes[0].set(title=f"BESS operation - {params.capacity_kwh:.0f} kWh / {params.max_charge_kw:.0f} kW at bus {params.bus}",
                ylabel="State of charge [%]", ylim=(0, 100))
    _legend_below(axes[1])
    axes[1].axhline(0, color=INK_2, linewidth=0.8)
    axes[1].set(ylabel="Power [kW]\n(+ charge / - discharge)")
    _time_axis(axes[1])
    return _save(fig, out_dir, "08_battery_soc.png", dpi)


def plot_pv_curtailment(hosting: pd.DataFrame, v_max: float, trafo_limit: float, out_dir: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
    for slot, control in enumerate(["uncontrolled", "optimized"]):
        sub = hosting[hosting.control == control]
        label = "Uncontrolled" if control == "uncontrolled" else "Optimized (EV + BESS + curtailment)"
        axes[0].plot(sub.pv_penetration_pct, sub.v_max_pu, marker="o", markersize=5, color=SERIES[slot],
                     linestyle=LINESTYLES[slot], label=label)
        axes[1].plot(sub.pv_penetration_pct, sub.trafo_loading_max_pct, marker="o", markersize=5,
                     color=SERIES[slot], linestyle=LINESTYLES[slot], label=label)
    _limit_line(axes[0], v_max, f"Upper limit {v_max:.2f} pu")
    _limit_line(axes[1], trafo_limit, f"Rated {trafo_limit:.0f} %")
    axes[0].set(title="Maximum voltage vs PV penetration", xlabel="PV penetration [% of nominal peak load]",
                ylabel="Maximum bus voltage [pu]")
    axes[1].set(title="Transformer loading vs PV", xlabel="PV penetration [% of nominal peak load]",
                ylabel="Max transformer loading [%]")
    axes[0].legend(loc="upper left")
    opt = hosting[hosting.control == "optimized"]
    share = 100 * opt.pv_curtailment_kwh / opt.pv_available_kwh.replace(0, np.nan)
    bars = axes[2].bar(opt.pv_penetration_pct, opt.pv_curtailment_kwh / 1000, width=30, color=SERIES[0])
    for bar, pct in zip(bars, share.fillna(0)):
        if bar.get_height() > 0:
            axes[2].annotate(f"{pct:.1f} %", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                             xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    axes[2].set(title="PV curtailment (optimized)", xlabel="PV penetration [% of nominal peak load]",
                ylabel="Curtailed energy [MWh/day]")
    return _save(fig, out_dir, "09_pv_curtailment.png", dpi)


# --------------------------------------------------------------------------- 10-12 comparison
def plot_scenario_comparison(metrics: dict[str, dict[str, Any]], out_dir: Path, dpi: int) -> Path:
    panels = [
        ("peak_import_kw", "Peak grid import [kW]", "{:,.0f}"),
        ("v_min_pu", "Minimum voltage [pu]", "{:.3f}"),
        ("voltage_violation_bus_steps", "Voltage violations [bus-steps]", "{:,.0f}"),
        ("trafo_loading_max_pct", "Max transformer loading [%]", "{:.1f}"),
        ("losses_kwh", "Network losses [kWh/day]", "{:,.0f}"),
        ("total_cost_eur", "Operating cost [EUR/day]", "{:,.0f}"),
    ]
    labels = list(metrics)
    y = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, (key, title, fmt) in zip(axes.flat, panels):
        values = np.array([metrics[label][key] for label in labels])
        ax.barh(y, values, color=SERIES[0], height=0.62)
        for yi, value in zip(y, values):
            ax.annotate(fmt.format(value), (value, yi), xytext=(3, 0), textcoords="offset points",
                        va="center", fontsize=8, color=INK_2)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", visible=False)
        if key == "v_min_pu":
            ax.set_xlim(0.88, 1.0)
            ax.axvline(0.95, color=LIMIT, linewidth=1.2, linestyle=(0, (4, 2)))
        else:
            ax.set_xlim(0, values.max() * 1.22 if values.max() > 0 else 1)
    axes[0, 0].set_yticks(y, labels)
    axes[0, 0].invert_yaxis()
    fig.suptitle("Scenario comparison (AC power-flow results, 24 h)", fontweight="bold")
    return _save(fig, out_dir, "10_scenario_comparison.png", dpi)


def _feeder_branches(net: Any) -> list[list[int]]:
    """Bus sequences (0-based) from each branch's junction to its end, for radial plotting."""
    children: dict[int, list[int]] = {}
    for fb, tb in zip(net.line.from_bus, net.line.to_bus):
        children.setdefault(int(fb), []).append(int(tb))
    branches, stack = [], [[0]]
    while stack:
        path = stack.pop()
        nxt = sorted(children.get(path[-1], []))
        if not nxt:
            branches.append(path)
            continue
        stack.append(path + [nxt[0]])           # continue along the lowest-numbered child
        for child in nxt[1:]:
            stack.append([path[-1], child])     # a lateral starts at its junction bus
    branches.sort(key=lambda b: (b[0] != 0, b[1]))  # main feeder first, then laterals by bus number
    return branches


def plot_voltage_profile(results: dict[str, Any], v_min: float, out_dir: Path, dpi: int,
                         reference: str | None = None) -> Path:
    """Voltage along every feeder branch at the critical step of the ``reference`` scenario.

    The critical step is the time of the lowest voltage in the reference case
    (default: second scenario, i.e. uncontrolled PV + EV), so the plot shows how
    the other control strategies perform at that same moment.
    """
    labels = list(results)
    ref_label = reference or labels[min(1, len(labels) - 1)]
    ref = results[ref_label]
    step = ref.simulation.bus_vm_pu.min(axis=1).idxmin()
    branches = _feeder_branches(ref.data.net)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    for slot, (label, res) in enumerate(results.items()):
        vm = res.simulation.bus_vm_pu.loc[step]
        for i, branch in enumerate(branches):
            buses = branch if i == 0 else branch[1:]   # laterals drawn without their junction bus
            ax.plot(np.array(buses) + 1, vm.loc[buses].to_numpy(), marker="o", markersize=3.5,
                    color=SERIES[slot], linestyle=LINESTYLES[slot], label=label if i == 0 else None)
    for branch in branches[1:]:
        ax.axvline(branch[1] + 0.5, color=GRID, linewidth=1.0, zorder=0)
    _limit_line(ax, v_min, f"Lower limit {v_min:.2f} pu")
    ax.set(title=f"Voltage along the feeder at {step:%H:%M} (most critical step of {ref_label})",
           xlabel="IEEE bus number  (main feeder 1-18 | laterals 19-22, 23-25, 26-33)", ylabel="Voltage magnitude [pu]",
           xticks=range(1, 34, 2))
    _legend_below(ax)
    return _save(fig, out_dir, "11_voltage_profile.png", dpi)


def plot_tradeoff(tradeoff: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.plot(tradeoff.peak_import_kw, tradeoff.energy_cost_eur, color=SERIES[0], marker="o", markersize=7)
    for row in tradeoff.itertuples():
        ax.annotate(f"{row.peak_charge_eur_per_kw:.1f} EUR/kW", (row.peak_import_kw, row.energy_cost_eur),
                    xytext=(6, 4), textcoords="offset points", fontsize=8, color=INK_2)
    ax.set(title="Energy cost vs peak demand (S7, varying demand charge)",
           xlabel="Peak grid import [kW]", ylabel="Energy cost excl. demand charge [EUR/day]")
    return _save(fig, out_dir, "12_cost_peak_tradeoff.png", dpi)


# --------------------------------------------------------------------------- extra experiment views
def plot_bess_sweep(bess: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    x = bess.bess_capacity_kwh.astype(int).astype(str) + " kWh"
    for ax, key, title, fmt in [
        (axes[0], "peak_import_kw", "Peak grid import [kW]", "{:,.0f}"),
        (axes[1], "voltage_violation_bus_steps", "Voltage violations [bus-steps]", "{:,.0f}"),
        (axes[2], "total_cost_eur", "Operating cost [EUR/day]", "{:,.0f}"),
    ]:
        bars = ax.bar(x, bess[key], color=SERIES[0], width=0.6)
        for bar, value in zip(bars, bess[key]):
            ax.annotate(fmt.format(value), (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
        ax.set(title=title, xlabel="BESS capacity (C/2)")
        ax.grid(axis="x", visible=False)
        ax.set_ylim(0, max(bess[key].max() * 1.18, 1))
    fig.suptitle("Effect of BESS size (optimized EV + BESS, 150 % PV, high EV)", fontweight="bold")
    return _save(fig, out_dir, "13_bess_sweep.png", dpi)


def plot_pv_ev_heatmap(pv_ev: pd.DataFrame, out_dir: Path, dpi: int) -> Path:
    from matplotlib.colors import LinearSegmentedColormap

    blues = LinearSegmentedColormap.from_list("blue_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
    levels = list(dict.fromkeys(pv_ev.ev_level))
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for ax, key, title, fmt in [
        (axes[0], "voltage_violation_bus_steps", "Undervoltage bus-steps (uncontrolled)", "{:.0f}"),
        (axes[1], "peak_import_kw", "Peak grid import [kW] (uncontrolled)", "{:,.0f}"),
    ]:
        grid = pv_ev.pivot(index="ev_level", columns="pv_penetration_pct", values=key).loc[levels]
        image = ax.imshow(grid.to_numpy(), cmap=blues, aspect="auto")
        for (i, j), value in np.ndenumerate(grid.to_numpy()):
            ink = "white" if value > grid.to_numpy().mean() else INK
            ax.text(j, i, fmt.format(value), ha="center", va="center", fontsize=9, color=ink)
        ax.set_xticks(range(len(grid.columns)), [f"{c:.0f} %" for c in grid.columns])
        ax.set_yticks(range(len(levels)), [f"{lvl} ({pv_ev[pv_ev.ev_level == lvl].n_evs.iloc[0]} EVs)" for lvl in levels])
        ax.set(title=title, xlabel="PV penetration [% of nominal peak load]")
        ax.grid(False)
        fig.colorbar(image, ax=ax, shrink=0.85)
    return _save(fig, out_dir, "14_pv_ev_sweep.png", dpi)
