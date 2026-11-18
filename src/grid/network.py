"""IEEE 33-bus distribution network model built on pandapower.

The electrical data (bus voltages, branch impedances, nominal loads) come
unchanged from pandapower's ``case33bw`` implementation of the Baran & Wu
(1989) 33-bus radial feeder. Documented modifications for this study:

1. The five normally-open tie lines (out of service in the standard case) are
   removed, leaving the 32-branch radial feeder.
2. The ideal slack at bus 0 is replaced by a 110 kV external grid feeding bus 0
   through an HV/MV transformer with an on-load tap changer (OLTC). This makes
   transformer loading observable; the OLTC keeps the MV busbar voltage close
   to the 1.0 pu that the standard case assumes.
3. Thermal ratings are assigned to all lines (the standard case has none).
4. Loads are labelled residential or commercial to receive different profiles.
5. PV generators, EV charging loads and a BESS are added as separate elements so
   they can be updated independently in the time-series simulation.

Bus indices are pandapower's 0-based indices; IEEE bus number = index + 1.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandapower as pp
import pandapower.networks as pn
from pandapower.control import DiscreteTapControl

logger = logging.getLogger(__name__)

MV_BUSBAR = 0  # bus index of the substation MV busbar (IEEE bus 1)


class NetworkError(RuntimeError):
    """Raised when the network cannot be built or does not converge."""


def build_network(config: dict[str, Any]) -> pp.pandapowerNet:
    """Create the IEEE 33-bus feeder with substation transformer and line ratings.

    PV, EV and BESS elements are not included; add them with
    :func:`add_pv_generators`, :func:`add_ev_loads` and :func:`add_bess`.
    """
    net_cfg = config["network"]
    limits = config["limits"]
    if net_cfg["source"] != "case33bw":
        raise NetworkError(f"Unsupported network source: {net_cfg['source']}")

    net = pn.case33bw()
    _remove_open_tie_lines(net)

    net.bus["min_vm_pu"] = limits["v_min_pu"]
    net.bus["max_vm_pu"] = limits["v_max_pu"]
    net.bus["name"] = [f"Bus {i + 1}" for i in net.bus.index]  # IEEE numbering
    net.line["name"] = [f"Line {fb + 1}-{tb + 1}" for fb, tb in zip(net.line.from_bus, net.line.to_bus)]

    _add_substation(net, net_cfg)

    commercial = set(net_cfg["commercial_buses"])
    net.load["profile"] = ["commercial" if bus in commercial else "residential" for bus in net.load.bus]
    net.load["element_type"] = "base_load"
    net.load["p_nom_mw"] = net.load["p_mw"]
    net.load["q_nom_mvar"] = net.load["q_mvar"]

    assign_line_ratings(net, net_cfg["line_ratings"])
    net.line["max_loading_percent"] = limits["line_max_loading_pct"]
    net.trafo["max_loading_percent"] = limits["trafo_max_loading_pct"]

    if net_cfg["transformer"]["oltc"]["enabled"]:
        add_oltc_controller(net, net_cfg["transformer"]["oltc"])

    logger.info(
        "Built %s: %d buses, %d lines, %d loads (%.3f MW / %.3f Mvar nominal)",
        net_cfg["source"], len(net.bus), len(net.line), len(net.load),
        net.load.p_mw.sum(), net.load.q_mvar.sum(),
    )
    return net


def _remove_open_tie_lines(net: pp.pandapowerNet) -> None:
    tie_lines = net.line.index[~net.line.in_service]
    net.line.drop(tie_lines, inplace=True)
    if len(net.line) != len(net.bus) - 1:
        raise NetworkError("Feeder is not radial after removing tie lines")


def _add_substation(net: pp.pandapowerNet, net_cfg: dict[str, Any]) -> None:
    """Replace the MV slack with an HV external grid and an OLTC transformer."""
    trafo_cfg = net_cfg["transformer"]
    oltc_cfg = trafo_cfg["oltc"]
    net.ext_grid.drop(net.ext_grid.index, inplace=True)
    hv_bus = pp.create_bus(net, vn_kv=net_cfg["hv_kv"], name="HV grid", min_vm_pu=0.9, max_vm_pu=1.1)
    pp.create_ext_grid(net, bus=hv_bus, vm_pu=net_cfg["hv_vm_pu"], name=f"{net_cfg['hv_kv']:g} kV grid")
    pp.create_transformer_from_parameters(
        net,
        hv_bus=hv_bus,
        lv_bus=MV_BUSBAR,
        sn_mva=trafo_cfg["sn_mva"],
        vn_hv_kv=net_cfg["hv_kv"],
        vn_lv_kv=net_cfg["mv_kv"],
        vk_percent=trafo_cfg["vk_percent"],
        vkr_percent=trafo_cfg["vkr_percent"],
        pfe_kw=trafo_cfg["pfe_kw"],
        i0_percent=trafo_cfg["i0_percent"],
        shift_degree=trafo_cfg["shift_degree"],
        tap_side="hv",
        tap_neutral=0,
        tap_min=oltc_cfg["tap_min"],
        tap_max=oltc_cfg["tap_max"],
        tap_step_percent=oltc_cfg["tap_step_percent"],
        tap_pos=0,
        tap_changer_type="Ratio",
        name="HV/MV substation transformer",
    )


def add_oltc_controller(net: pp.pandapowerNet, oltc_cfg: dict[str, Any]) -> None:
    """Attach a discrete OLTC voltage regulator to the substation transformer."""
    DiscreteTapControl(
        net,
        element_index=net.trafo.index[0],
        vm_lower_pu=oltc_cfg["vm_lower_pu"],
        vm_upper_pu=oltc_cfg["vm_upper_pu"],
        side="lv",
    )


def assign_line_ratings(net: pp.pandapowerNet, rating_cfg: dict[str, Any]) -> None:
    """Assign a discrete thermal rating to each line.

    Each line receives the smallest standard ampacity whose design utilization
    (``design_utilization``) covers the line current at nominal peak load. This
    mimics a planner who sized conductors with headroom for load growth.
    """
    sizes_ka = np.sort(np.array(rating_cfg["standard_ampacities_a"], dtype=float)) / 1000.0
    utilization = rating_cfg["design_utilization"]
    net.line["max_i_ka"] = sizes_ka[-1] * 100  # temporary: avoid limits during the sizing run
    run_power_flow(net, run_control=False)
    peak_current_ka = net.res_line["i_ka"].to_numpy()
    ratings = []
    for current in peak_current_ka:
        candidates = sizes_ka[sizes_ka * utilization >= current]
        if len(candidates) == 0:
            logger.warning("Line current %.0f A exceeds the largest standard size", current * 1000)
            ratings.append(sizes_ka[-1])
        else:
            ratings.append(candidates[0])
    net.line["max_i_ka"] = ratings


def add_pv_generators(net: pp.pandapowerNet, total_capacity_mw: float) -> list[int]:
    """Add rooftop PV (unity power factor) at every load bus, proportional to its nominal load.

    Returns the sgen indices. The installed capacity is stored in ``sn_mva``;
    the time-series simulation sets ``p_mw`` each step.
    """
    base = net.load[net.load.element_type == "base_load"]
    shares = base.p_nom_mw / base.p_nom_mw.sum()
    indices = []
    for bus, share in zip(base.bus, shares):
        capacity = total_capacity_mw * share
        indices.append(
            pp.create_sgen(net, bus=bus, p_mw=0.0, q_mvar=0.0, sn_mva=capacity,
                           name=f"PV bus {bus}", type="PV")
        )
    return indices


def add_ev_loads(net: pp.pandapowerNet, buses: list[int]) -> dict[int, int]:
    """Add one aggregated EV charging load element per bus. Returns {bus: load index}."""
    mapping = {}
    for bus in sorted(set(buses)):
        idx = pp.create_load(net, bus=bus, p_mw=0.0, q_mvar=0.0, name=f"EV bus {bus}")
        net.load.loc[idx, ["element_type", "profile"]] = ["ev", "ev"]
        mapping[bus] = idx
    return mapping


def add_bess(net: pp.pandapowerNet, bus: int, capacity_kwh: float, max_power_kw: float) -> int:
    """Add a grid-connected BESS as a pandapower storage element (p_mw > 0 = charging)."""
    return pp.create_storage(
        net, bus=bus, p_mw=0.0, max_e_mwh=capacity_kwh / 1000.0,
        sn_mva=max_power_kw / 1000.0, name=f"BESS bus {bus}",
    )


def run_power_flow(net: pp.pandapowerNet, run_control: bool = True) -> None:
    """Run an AC Newton-Raphson power flow (with OLTC control if present).

    Raises:
        NetworkError: if the power flow does not converge.
    """
    try:
        if run_control and len(net.controller):
            pp.runpp(net, run_control=True, numba=True)
        else:
            pp.runpp(net, numba=True)
    except pp.LoadflowNotConverged as exc:
        raise NetworkError("AC power flow did not converge") from exc


def feeder_peak_load_mw(net: pp.pandapowerNet) -> float:
    """Sum of nominal (non-coincident peak) base loads in MW."""
    return float(net.load.loc[net.load.element_type == "base_load", "p_nom_mw"].sum())


def residential_buses(net: pp.pandapowerNet) -> list[int]:
    """Buses whose base load is residential."""
    base = net.load[(net.load.element_type == "base_load") & (net.load.profile == "residential")]
    return base.bus.tolist()
