"""Pyomo MILP for coordinated EV charging, BESS dispatch and PV curtailment.

Decision variables (kW unless noted), t = time step, e = EV, n = MV bus:
    p_ev[e,t]        EV charging power (only for connected (e,t) pairs)
    p_ch[t], p_dis[t] BESS charge/discharge power, y_bess[t] binary mode
    e_bess[t]        BESS stored energy (kWh), t = 0..T
    p_curt[n,t]      PV curtailment
    p_imp[t], p_exp[t] grid import/export at the substation
    p_peak           peak import
    p_net[n,t]       net active consumption per bus (auxiliary)
    s_lo[n,t], s_hi[n,t] voltage-limit slack (pu)

Objective (EUR, each term multiplied by a configurable weight):
    energy cost         sum_t (c_imp p_imp - c_exp p_exp) dt
    peak demand charge  c_peak p_peak
    voltage penalty     c_v sum_{n,t} (s_lo + s_hi) dt      (soft constraint, not a real cost)
    curtailment penalty c_curt sum_{n,t} p_curt dt
    BESS degradation    c_deg sum_t (p_ch + p_dis) / 2 dt

Network representation: LinDistFlow voltages and lossless line flows, plus
per-step loss and voltage corrections obtained from AC power flow. The
transformer and line limits are linearized by fixing the (uncontrolled)
reactive power: |P| <= sqrt((margin * S_rated)^2 - Q^2).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from src.models.battery import BatteryParameters
from src.models.ev_model import connection_matrix
from src.optimization.lindistflow import FeederSensitivity

logger = logging.getLogger(__name__)


class OptimizationError(RuntimeError):
    """Raised when the solver fails or returns an unusable solution."""


@dataclass
class NetworkCorrections:
    """Quantities fed back from the AC power flow (all shape [T] or [bus x T])."""

    v_busbar_pu: np.ndarray        # MV busbar voltage per step (set by the OLTC)
    v_offset_pu: np.ndarray        # AC voltage minus LinDistFlow voltage [bus x T]
    losses_kw: np.ndarray          # active losses per step
    q_trafo_kvar: np.ndarray       # reactive power through the transformer per step
    q_line_kvar: np.ndarray        # reactive line flows [line x T]
    trafo_margin: float = 1.0      # fraction of rated power usable
    line_margin: float = 1.0


@dataclass
class OptimizationInputs:
    """Everything the MILP needs; built from a prepared scenario."""

    dt_h: float
    timestamps: pd.DatetimeIndex
    buses: np.ndarray
    base_p_kw: np.ndarray          # [bus x T]
    base_q_kvar: np.ndarray        # [bus x T]
    pv_available_kw: np.ndarray    # [bus x T]
    fleet: pd.DataFrame
    optimize_ev: bool
    fixed_ev_kw: np.ndarray        # [n_ev x T] used when optimize_ev is False
    bess: BatteryParameters | None
    price_import: np.ndarray
    price_export: np.ndarray
    sensitivity: FeederSensitivity
    trafo_rating_kva: float
    v_min: float
    v_max: float
    weights: dict[str, float]
    peak_charge_eur_per_kw: float
    voltage_penalty: float
    curtailment_penalty: float
    allow_curtailment: bool = True
    enforce: dict[str, bool] = field(default_factory=lambda: {"voltage": True, "transformer": True, "lines": True})


@dataclass
class OptimizationSolution:
    ev_power_kw: np.ndarray
    bess_schedule: pd.DataFrame
    pv_curtailment_kw: pd.DataFrame
    grid_kw: np.ndarray
    v_model_pu: np.ndarray          # [bus x T] voltages predicted by the model
    objective_eur: float
    objective_terms: dict[str, float]
    solve_time_s: float
    solver_status: str


def build_model(inp: OptimizationInputs, corr: NetworkCorrections) -> pyo.ConcreteModel:
    """Construct the Pyomo model for given inputs and AC corrections."""
    n_steps = len(inp.timestamps)
    n_bus = len(inp.buses)
    dt = inp.dt_h
    sens = inp.sensitivity
    m = pyo.ConcreteModel(name="ev_bess_coordination")
    m.T = pyo.RangeSet(0, n_steps - 1)
    m.N = pyo.RangeSet(0, n_bus - 1)                         # bus positions
    m.L = pyo.RangeSet(0, len(sens.lines) - 1)

    # ----- EVs --------------------------------------------------------------
    fleet = inp.fleet
    ev_bus_pos = {i: int(np.where(inp.buses == b)[0][0]) for i, b in enumerate(fleet.bus)}
    if inp.optimize_ev and len(fleet):
        connected = connection_matrix(fleet, n_steps)
        pairs = [(e, t) for e in range(len(fleet)) for t in range(n_steps) if connected[e, t]]
        m.EVT = pyo.Set(initialize=pairs, dimen=2)
        m.E = pyo.RangeSet(0, len(fleet) - 1)
        pmax = fleet.max_power_kw.to_numpy()
        m.p_ev = pyo.Var(m.EVT, bounds=lambda _m, e, t: (0.0, pmax[e]))
        eta = fleet.efficiency.to_numpy()
        required = fleet.required_energy_kwh.to_numpy()
        steps_of = {e: [t for (ee, t) in pairs if ee == e] for e in range(len(fleet))}
        m.ev_energy = pyo.Constraint(
            m.E, rule=lambda _m, e: sum(_m.p_ev[e, t] for t in steps_of[e]) * eta[e] * dt == required[e])
        evs_at_bus_t: dict[tuple[int, int], list] = {}
        for e, t in pairs:
            evs_at_bus_t.setdefault((ev_bus_pos[e], t), []).append(e)

        def ev_load(_m, n, t):
            return sum(_m.p_ev[e, t] for e in evs_at_bus_t.get((n, t), []))
    else:
        fixed = np.zeros((n_bus, n_steps))
        for e in range(len(fleet)):
            fixed[ev_bus_pos[e]] += inp.fixed_ev_kw[e]

        def ev_load(_m, n, t):
            return fixed[n, t]

    # ----- BESS -------------------------------------------------------------
    bess = inp.bess
    if bess is not None and bess.capacity_kwh > 0:
        bess_pos = int(np.where(inp.buses == bess.bus)[0][0])
        m.TE = pyo.RangeSet(0, n_steps)
        m.p_ch = pyo.Var(m.T, bounds=(0.0, bess.max_charge_kw))
        m.p_dis = pyo.Var(m.T, bounds=(0.0, bess.max_discharge_kw))
        m.y_bess = pyo.Var(m.T, domain=pyo.Binary)            # 1 = charging mode
        m.e_bess = pyo.Var(m.TE, bounds=(bess.energy_min_kwh, bess.energy_max_kwh))
        m.e_bess[0].fix(bess.energy_initial_kwh)
        m.bess_dyn = pyo.Constraint(m.T, rule=lambda _m, t: _m.e_bess[t + 1] == _m.e_bess[t]
                                    + bess.charge_efficiency * _m.p_ch[t] * dt - _m.p_dis[t] * dt / bess.discharge_efficiency)
        m.bess_ch_mode = pyo.Constraint(m.T, rule=lambda _m, t: _m.p_ch[t] <= bess.max_charge_kw * _m.y_bess[t])
        m.bess_dis_mode = pyo.Constraint(m.T, rule=lambda _m, t: _m.p_dis[t] <= bess.max_discharge_kw * (1 - _m.y_bess[t]))
        # No "free" energy: end the day with at least the initial state of charge.
        m.bess_terminal = pyo.Constraint(expr=m.e_bess[n_steps] >= bess.energy_initial_kwh)

        def bess_load(_m, n, t):
            return (_m.p_ch[t] - _m.p_dis[t]) if n == bess_pos else 0.0
    else:
        def bess_load(_m, n, t):
            return 0.0

    # ----- PV curtailment ---------------------------------------------------
    pv = inp.pv_available_kw
    if inp.allow_curtailment:
        m.p_curt = pyo.Var(m.N, m.T, bounds=lambda _m, n, t: (0.0, float(pv[n, t])))
        curt = lambda _m, n, t: _m.p_curt[n, t]
    else:
        curt = lambda _m, n, t: 0.0

    # ----- Bus balance and grid exchange ------------------------------------
    m.p_net = pyo.Var(m.N, m.T)
    m.bus_balance = pyo.Constraint(m.N, m.T, rule=lambda _m, n, t: _m.p_net[n, t] ==
                                   inp.base_p_kw[n, t] + ev_load(_m, n, t) + bess_load(_m, n, t)
                                   - pv[n, t] + curt(_m, n, t))
    m.p_imp = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.p_exp = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.p_peak = pyo.Var(domain=pyo.NonNegativeReals)
    m.grid_balance = pyo.Constraint(m.T, rule=lambda _m, t: _m.p_imp[t] - _m.p_exp[t] ==
                                    sum(_m.p_net[n, t] for n in _m.N) + corr.losses_kw[t])
    m.peak_def = pyo.Constraint(m.T, rule=lambda _m, t: _m.p_peak >= _m.p_imp[t])

    # ----- Transformer capacity --------------------------------------------
    if inp.enforce.get("transformer", True):
        s_max = corr.trafo_margin * inp.trafo_rating_kva
        p_lim = np.sqrt(np.clip(s_max**2 - corr.q_trafo_kvar**2, 0.0, None))
        m.trafo_fwd = pyo.Constraint(m.T, rule=lambda _m, t: _m.p_imp[t] - _m.p_exp[t] <= p_lim[t])
        m.trafo_rev = pyo.Constraint(m.T, rule=lambda _m, t: _m.p_imp[t] - _m.p_exp[t] >= -p_lim[t])

    # ----- Line capacity (lossless flows) -----------------------------------
    if inp.enforce.get("lines", True):
        s_line = corr.line_margin * sens.line_rating_kva
        p_line_lim = np.sqrt(np.clip(s_line[:, None]**2 - corr.q_line_kvar**2, 0.0, None))
        down = [np.nonzero(sens.downstream[l])[0].tolist() for l in range(len(sens.lines))]
        m.line_fwd = pyo.Constraint(m.L, m.T, rule=lambda _m, l, t:
                                    sum(_m.p_net[n, t] for n in down[l]) <= p_line_lim[l, t])
        m.line_rev = pyo.Constraint(m.L, m.T, rule=lambda _m, l, t:
                                    sum(_m.p_net[n, t] for n in down[l]) >= -p_line_lim[l, t])

    # ----- Voltage (LinDistFlow + AC correction), soft limits ---------------
    q_drop = (sens.x_path_ohm @ inp.base_q_kvar) / 1000.0 / sens.vn_kv**2      # fixed reactive part
    r_coef = sens.r_path_ohm / 1000.0 / sens.vn_kv**2
    v_const = corr.v_busbar_pu[None, :] - q_drop + corr.v_offset_pu             # [bus x T]
    nz = [np.nonzero(r_coef[n])[0].tolist() for n in range(n_bus)]

    def v_expr(_m, n, t):
        return v_const[n, t] - sum(r_coef[n, k] * _m.p_net[k, t] for k in nz[n])
    m.v = pyo.Expression(m.N, m.T, rule=v_expr)
    m.s_lo = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)
    m.s_hi = pyo.Var(m.N, m.T, domain=pyo.NonNegativeReals)
    if inp.enforce.get("voltage", True):
        m.v_low = pyo.Constraint(m.N, m.T, rule=lambda _m, n, t: _m.v[n, t] + _m.s_lo[n, t] >= inp.v_min)
        m.v_high = pyo.Constraint(m.N, m.T, rule=lambda _m, n, t: _m.v[n, t] - _m.s_hi[n, t] <= inp.v_max)
    else:
        for var in (m.s_lo, m.s_hi):
            var.fix(0.0)

    # ----- Objective --------------------------------------------------------
    w = inp.weights
    m.cost_energy = pyo.Expression(expr=sum((inp.price_import[t] * m.p_imp[t] - inp.price_export[t] * m.p_exp[t]) * dt
                                            for t in m.T))
    m.cost_peak = pyo.Expression(expr=inp.peak_charge_eur_per_kw * m.p_peak)
    m.cost_voltage = pyo.Expression(expr=inp.voltage_penalty * dt *
                                    sum(m.s_lo[n, t] + m.s_hi[n, t] for n in m.N for t in m.T))
    m.cost_curtail = pyo.Expression(expr=(inp.curtailment_penalty * dt * sum(m.p_curt[n, t] for n in m.N for t in m.T))
                                    if inp.allow_curtailment else 0.0)
    m.cost_degradation = pyo.Expression(expr=(bess.degradation_cost_eur_per_kwh * dt *
                                              sum((m.p_ch[t] + m.p_dis[t]) / 2 for t in m.T))
                                        if hasattr(m, "p_ch") else 0.0)
    m.objective = pyo.Objective(
        expr=w["energy_cost"] * m.cost_energy + w["peak_demand"] * m.cost_peak
        + w["voltage_violation"] * m.cost_voltage + w["pv_curtailment"] * m.cost_curtail
        + w["battery_degradation"] * m.cost_degradation,
        sense=pyo.minimize,
    )
    return m


def solve_model(m: pyo.ConcreteModel, solver_cfg: dict[str, Any]) -> tuple[str, float]:
    """Solve with HiGHS (via Pyomo APPSI); fall back to CBC if HiGHS is unavailable."""
    start = time.perf_counter()
    for name in (solver_cfg["solver"], solver_cfg.get("fallback_solver")):
        if not name:
            continue
        solver = pyo.SolverFactory(name)
        if not solver.available(exception_flag=False):
            logger.warning("Solver %s not available", name)
            continue
        if name.endswith("highs"):
            solver.highs_options.update({"mip_rel_gap": solver_cfg["mip_gap"],
                                         "time_limit": float(solver_cfg["time_limit_s"])})
        elif name == "cbc":
            solver.options.update({"ratioGap": solver_cfg["mip_gap"], "seconds": solver_cfg["time_limit_s"]})
        result = solver.solve(m, load_solutions=True)
        status = str(result.solver.termination_condition)
        if status not in ("optimal", "TerminationCondition.optimal", "maxTimeLimit"):
            raise OptimizationError(f"Solver {name} terminated with status {status}")
        return f"{name}:{status}", time.perf_counter() - start
    raise OptimizationError("No MILP solver available (install highspy or CBC)")


def extract_solution(m: pyo.ConcreteModel, inp: OptimizationInputs, status: str, solve_time: float) -> OptimizationSolution:
    """Read variable values into arrays/DataFrames."""
    n_steps = len(inp.timestamps)
    n_bus = len(inp.buses)
    if inp.optimize_ev and hasattr(m, "p_ev"):
        ev = np.zeros((len(inp.fleet), n_steps))
        for (e, t), var in m.p_ev.items():
            ev[e, t] = max(pyo.value(var), 0.0)
    else:
        ev = inp.fixed_ev_kw.copy()
    if hasattr(m, "p_ch"):
        charge = np.array([max(pyo.value(m.p_ch[t]), 0.0) for t in m.T])
        discharge = np.array([max(pyo.value(m.p_dis[t]), 0.0) for t in m.T])
        # Remove numerical noise (< 1 W) so validation sees exact exclusivity.
        charge[charge < 1e-3] = 0.0
        discharge[discharge < 1e-3] = 0.0
    else:
        charge = discharge = np.zeros(n_steps)
    bess = pd.DataFrame({"charge_kw": charge, "discharge_kw": discharge}, index=inp.timestamps)
    curt = np.zeros((n_bus, n_steps))
    if inp.allow_curtailment:
        for (n, t), var in m.p_curt.items():
            curt[n, t] = min(max(pyo.value(var), 0.0), inp.pv_available_kw[n, t])
        curt[curt < 1e-3] = 0.0
    pv_buses = inp.pv_available_kw.sum(axis=1) > 0
    curt_df = pd.DataFrame(curt[pv_buses].T, index=inp.timestamps, columns=inp.buses[pv_buses].astype(int))
    v_model = np.array([[pyo.value(m.v[n, t]) for t in m.T] for n in m.N])
    terms = {
        "energy_cost": pyo.value(m.cost_energy),
        "peak_demand": pyo.value(m.cost_peak),
        "voltage_violation": pyo.value(m.cost_voltage),
        "pv_curtailment": pyo.value(m.cost_curtail),
        "battery_degradation": pyo.value(m.cost_degradation),
    }
    return OptimizationSolution(
        ev_power_kw=ev, bess_schedule=bess, pv_curtailment_kw=curt_df,
        grid_kw=np.array([pyo.value(m.p_imp[t] - m.p_exp[t]) for t in m.T]),
        v_model_pu=v_model, objective_eur=float(pyo.value(m.objective)), objective_terms=terms,
        solve_time_s=solve_time, solver_status=status,
    )


def optimize(inp: OptimizationInputs, corr: NetworkCorrections, solver_cfg: dict[str, Any]) -> OptimizationSolution:
    """Build, solve and extract in one call."""
    build_start = time.perf_counter()
    m = build_model(inp, corr)
    logger.info("Model built in %.1f s (%d variables, %d constraints)", time.perf_counter() - build_start,
                m.nvariables(), m.nconstraints())
    status, solve_time = solve_model(m, solver_cfg)
    solution = extract_solution(m, inp, status, solve_time)
    logger.info("Solved (%s) in %.1f s, objective %.2f EUR", status, solve_time, solution.objective_eur)
    return solution
