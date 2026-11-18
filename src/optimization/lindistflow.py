"""Linearized radial power flow (LinDistFlow) used inside the optimization.

For a radial feeder with lossless branch flows and voltages close to 1 pu,
the voltage drop from the MV busbar (bus 0) to bus j is approximately

    v_j = v_0 - (1 / V_n^2) * sum_k ( R_jk * P_k + X_jk * Q_k )

where P_k, Q_k are the net consumptions at bus k (MW, Mvar), V_n is the
nominal voltage (kV) and R_jk (X_jk) is the resistance (reactance) of the
path shared by bus j and bus k from the busbar (Ohm). The active power
flowing through line l is the sum of the net consumption downstream of l.

The approximation neglects losses and the quadratic voltage terms; the
optimization corrects for this with offsets obtained from AC power flow
(see :mod:`src.optimization.validation`).
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import pandapower as pp
import pandas as pd

from src.grid.network import MV_BUSBAR


@dataclass(frozen=True)
class FeederSensitivity:
    """Topology and impedance matrices of the radial MV feeder."""

    buses: np.ndarray            # MV bus indices (column order of all matrices)
    lines: np.ndarray            # line indices (row order of ``downstream``)
    downstream: np.ndarray       # [line x bus] 1 if bus is downstream of the line
    r_path_ohm: np.ndarray       # [bus x bus] shared-path resistance
    x_path_ohm: np.ndarray       # [bus x bus] shared-path reactance
    vn_kv: float
    line_rating_kva: np.ndarray  # sqrt(3) * V_n * I_max per line

    def voltage_drop_pu(self, p_kw: np.ndarray, q_kvar: np.ndarray) -> np.ndarray:
        """Voltage drop [bus x time] for net consumptions p_kw, q_kvar [bus x time]."""
        return (self.r_path_ohm @ p_kw + self.x_path_ohm @ q_kvar) / 1000.0 / self.vn_kv**2

    def line_flows_kw(self, p_kw: np.ndarray) -> np.ndarray:
        """Lossless active line flows [line x time] (positive = away from the substation)."""
        return self.downstream @ p_kw


def compute_sensitivities(net: pp.pandapowerNet) -> FeederSensitivity:
    """Derive the LinDistFlow matrices from the pandapower line data."""
    hv_kv = net.trafo.vn_hv_kv.iloc[0] if len(net.trafo) else np.inf
    buses = np.array(sorted(net.bus.index[net.bus.vn_kv < hv_kv]))
    bus_pos = {bus: i for i, bus in enumerate(buses)}
    lines = net.line.index.to_numpy()
    r_line = (net.line.r_ohm_per_km * net.line.length_km / net.line.parallel).to_numpy()
    x_line = (net.line.x_ohm_per_km * net.line.length_km / net.line.parallel).to_numpy()

    adjacency: dict[int, list[tuple[int, int]]] = {b: [] for b in buses}
    for pos, (fb, tb) in enumerate(zip(net.line.from_bus, net.line.to_bus)):
        adjacency[fb].append((tb, pos))
        adjacency[tb].append((fb, pos))

    # Breadth-first search from the busbar gives the line path to every bus.
    path: dict[int, list[int]] = {MV_BUSBAR: []}
    queue = deque([MV_BUSBAR])
    while queue:
        bus = queue.popleft()
        for neighbour, line_pos in adjacency[bus]:
            if neighbour not in path:
                path[neighbour] = path[bus] + [line_pos]
                queue.append(neighbour)
    if len(path) != len(buses):
        raise ValueError("Feeder is not connected")

    downstream = np.zeros((len(lines), len(buses)))
    for bus, line_positions in path.items():
        downstream[line_positions, bus_pos[bus]] = 1.0
    if not np.allclose(downstream.sum(axis=1)[:, None] >= 1, True):
        raise ValueError("Feeder contains lines without downstream buses (not radial)")

    vn_kv = float(net.bus.loc[buses, "vn_kv"].iloc[0])
    rating = np.sqrt(3.0) * vn_kv * net.line.max_i_ka.to_numpy() * 1000.0  # kVA
    return FeederSensitivity(
        buses=buses,
        lines=lines,
        downstream=downstream,
        r_path_ohm=downstream.T @ np.diag(r_line) @ downstream,
        x_path_ohm=downstream.T @ np.diag(x_line) @ downstream,
        vn_kv=vn_kv,
        line_rating_kva=rating,
    )


def to_bus_matrix(frame: pd.DataFrame, buses: np.ndarray) -> np.ndarray:
    """[time x bus] DataFrame -> [bus x time] array with zero rows for missing buses."""
    if len(frame.columns) == 0:
        return np.zeros((len(buses), len(frame.index)))
    return frame.reindex(columns=buses, fill_value=0.0).to_numpy().T
