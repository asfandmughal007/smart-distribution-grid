"""Optimization with independent AC power-flow validation.

Architecture (the optimizer never has the final word on grid compliance):

    reference AC run -> corrections -> MILP -> schedule -> AC power flow
           ^                                                   |
           +---- update corrections / derate limits <-- check -+

1. A reference dispatch (uncontrolled EVs, idle BESS) is simulated with AC
   power flow to obtain the OLTC busbar voltage, losses, reactive flows and
   the LinDistFlow voltage error at every bus and step.
2. The MILP is solved with those corrections.
3. The optimized schedule is applied to the pandapower network and a full AC
   time-series power flow is run.
4. If the AC voltages differ from the model's prediction by more than the
   tolerance, or a thermal limit is exceeded in AC, the corrections are
   recomputed at the new operating point (and violated limits derated) and
   the MILP is re-solved.

If the tolerance is not met within ``max_iterations`` (discrete OLTC tap
changes can make the loop oscillate), the iteration with the best AC result
is kept and the outcome is flagged as not converged. Residual violations that
the controllable resources cannot remove (e.g. base-load undervoltage) are
reported, never hidden.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.models.ev_model import aggregate_by_bus, uncontrolled_charging
from src.optimization.lindistflow import FeederSensitivity, compute_sensitivities, to_bus_matrix
from src.optimization.model import NetworkCorrections, OptimizationInputs, OptimizationSolution, optimize
from src.simulation.power_flow import SimulationResult

logger = logging.getLogger(__name__)


@dataclass
class OptimizationOutcome:
    """Final validated schedule and the history of the iterative loop."""

    ev_power_kw: np.ndarray
    bess_schedule: pd.DataFrame
    pv_curtailment_kw: pd.DataFrame
    simulation: SimulationResult           # AC validation of the final schedule
    solution: OptimizationSolution         # MILP solution of the final iteration
    history: pd.DataFrame                  # one row per iteration
    converged: bool                        # model and AC agree within tolerance (else best AC iteration kept)


def build_inputs(data: Any, sensitivity: FeederSensitivity) -> OptimizationInputs:
    """Translate a prepared :class:`ScenarioData` into MILP inputs."""
    config = data.config
    scenario = data.scenario
    opt = config["optimization"]
    if data.bess is not None and scenario.bess_control != "optimized":
        raise ValueError("A BESS present in an optimized scenario must also be optimized")
    n_steps = len(data.timestamps)
    fixed_ev = (uncontrolled_charging(data.fleet, n_steps, data.dt_h)
                if scenario.ev_control == "uncontrolled" else np.zeros((len(data.fleet), n_steps)))
    return OptimizationInputs(
        dt_h=data.dt_h,
        timestamps=data.timestamps,
        buses=sensitivity.buses,
        base_p_kw=to_bus_matrix(data.base_load_kw, sensitivity.buses),
        base_q_kvar=to_bus_matrix(data.base_load_kvar, sensitivity.buses),
        pv_available_kw=to_bus_matrix(data.pv_available_kw, sensitivity.buses),
        fleet=data.fleet,
        optimize_ev=scenario.ev_control == "optimized",
        fixed_ev_kw=fixed_ev,
        bess=data.bess,
        price_import=data.price.import_eur_per_kwh.to_numpy(),
        price_export=data.price.export_eur_per_kwh.to_numpy(),
        sensitivity=sensitivity,
        trafo_rating_kva=float(data.net.trafo.sn_mva.iloc[0] * 1000.0),
        v_min=config["limits"]["v_min_pu"] + opt["voltage_safety_margin_pu"],
        v_max=config["limits"]["v_max_pu"] - opt["voltage_safety_margin_pu"],
        weights=opt["weights"],
        peak_charge_eur_per_kw=config["price"]["peak_demand_charge_eur_per_kw"],
        voltage_penalty=opt["voltage_violation_penalty_eur_per_pu_h"],
        curtailment_penalty=opt["pv_curtailment_penalty_eur_per_kwh"],
        allow_curtailment=opt["allow_pv_curtailment"],
        enforce=opt["network_constraints"],
    )


def dispatch_injections(inp: OptimizationInputs, ev_power_kw: np.ndarray, bess: pd.DataFrame,
                        curtailment: pd.DataFrame) -> np.ndarray:
    """Net active consumption per bus [bus x T] for a given dispatch (kW)."""
    ev_bus = to_bus_matrix(aggregate_by_bus(ev_power_kw, inp.fleet, inp.timestamps), inp.buses)
    p = inp.base_p_kw + ev_bus - inp.pv_available_kw + to_bus_matrix(curtailment, inp.buses)
    if inp.bess is not None:
        pos = int(np.where(inp.buses == inp.bess.bus)[0][0])
        p[pos] += (bess.charge_kw - bess.discharge_kw).to_numpy()
    return p


def corrections_from_ac(inp: OptimizationInputs, sim: SimulationResult, p_kw: np.ndarray,
                        trafo_margin: float = 1.0, line_margin: float = 1.0) -> NetworkCorrections:
    """Derive LinDistFlow corrections from an AC result at the dispatch ``p_kw``."""
    sens = inp.sensitivity
    v_ac = sim.bus_vm_pu[sens.buses].to_numpy().T
    v_busbar = sim.bus_vm_pu[sens.buses[0]].to_numpy()
    v_lin = v_busbar[None, :] - sens.voltage_drop_pu(p_kw, inp.base_q_kvar)
    return NetworkCorrections(
        v_busbar_pu=v_busbar,
        v_offset_pu=v_ac - v_lin,
        losses_kw=sim.system.losses_kw.to_numpy(),
        q_trafo_kvar=sim.trafo.q_hv_kvar.to_numpy(),
        q_line_kvar=sim.line_q_kvar[sens.lines].to_numpy().T,
        trafo_margin=trafo_margin,
        line_margin=line_margin,
    )


def optimize_and_validate(data: Any) -> OptimizationOutcome:
    """Run the iterative optimize -> AC validate -> correct loop for a prepared scenario."""
    from src.simulation.scenarios import simulate_dispatch  # avoid circular import

    config = data.config
    opt_cfg = config["optimization"]
    loop_cfg = opt_cfg["iterative_validation"]
    limits = config["limits"]
    sens = compute_sensitivities(data.net)
    inp = build_inputs(data, sens)

    # Reference operating point: uncontrolled EVs, idle BESS, no curtailment.
    ref_ev = uncontrolled_charging(data.fleet, len(data.timestamps), data.dt_h)
    idle = pd.DataFrame({"charge_kw": 0.0, "discharge_kw": 0.0}, index=data.timestamps)
    no_curt = pd.DataFrame(0.0, index=data.timestamps, columns=data.pv_available_kw.columns)
    ref_sim = simulate_dispatch(data, ref_ev, idle, no_curt)
    corr = corrections_from_ac(inp, ref_sim, dispatch_injections(inp, ref_ev, idle, no_curt))

    max_iter = loop_cfg["max_iterations"] if loop_cfg["enabled"] else 1
    tol = loop_cfg["voltage_tolerance_pu"]
    history: list[dict[str, Any]] = []
    candidates: list[tuple[OptimizationSolution, SimulationResult]] = []
    converged = False
    for iteration in range(1, max_iter + 1):
        solution = optimize(inp, corr, opt_cfg)
        sim = simulate_dispatch(data, solution.ev_power_kw, solution.bess_schedule, solution.pv_curtailment_kw)
        if not sim.all_converged:
            raise RuntimeError(f"AC validation failed to converge in iteration {iteration}")
        v_ac = sim.bus_vm_pu[sens.buses].to_numpy().T
        mismatch = float(np.abs(v_ac - solution.v_model_pu).max())
        trafo_max = float(sim.trafo.loading_pct.max())
        line_max = float(sim.line_loading_pct.max().max())
        trafo_over = trafo_max > limits["trafo_max_loading_pct"] + 1e-6
        line_over = line_max > limits["line_max_loading_pct"] + 1e-6
        history.append({
            "iteration": iteration, "objective_eur": solution.objective_eur,
            "solve_time_s": solution.solve_time_s, "max_voltage_mismatch_pu": mismatch,
            "v_min_model_pu": float(solution.v_model_pu.min()), "v_min_ac_pu": float(v_ac.min()),
            "v_max_ac_pu": float(v_ac.max()),
            "ac_undervoltage_bus_steps": int((v_ac < limits["v_min_pu"]).sum()),
            "ac_overvoltage_bus_steps": int((v_ac > limits["v_max_pu"]).sum()),
            "trafo_max_pct": trafo_max, "line_max_pct": line_max,
            "trafo_margin": corr.trafo_margin, "line_margin": corr.line_margin,
        })
        logger.info("Iteration %d: objective %.2f EUR, |V_ac - V_model| max %.4f pu, AC Vmin %.4f, trafo %.1f %%",
                    iteration, solution.objective_eur, mismatch, v_ac.min(), trafo_max)
        candidates.append((solution, sim))
        if mismatch <= tol and not trafo_over and not line_over:
            converged = True
            break
        step = loop_cfg["loading_margin_step"]
        p_kw = dispatch_injections(inp, solution.ev_power_kw, solution.bess_schedule, solution.pv_curtailment_kw)
        corr = corrections_from_ac(
            inp, sim, p_kw,
            trafo_margin=corr.trafo_margin - step if trafo_over else corr.trafo_margin,
            line_margin=corr.line_margin - step if line_over else corr.line_margin,
        )

    history_df = pd.DataFrame(history)
    if converged:
        best = len(candidates) - 1
    else:
        # Discrete OLTC tap changes can make the loop oscillate around the tolerance.
        # Keep the iteration that is best *in AC*: fewest thermal overloads, then fewest
        # voltage violations, then lowest objective. The outcome stays flagged as not converged.
        score = list(zip(
            (history_df.trafo_max_pct > limits["trafo_max_loading_pct"] + 1e-6).astype(int)
            + (history_df.line_max_pct > limits["line_max_loading_pct"] + 1e-6).astype(int),
            history_df.ac_undervoltage_bus_steps + history_df.ac_overvoltage_bus_steps,
            history_df.objective_eur,
        ))
        best = min(range(len(score)), key=score.__getitem__)
        logger.warning("Iterative validation did not meet the tolerance in %d iterations; "
                       "selected iteration %d (best AC result)", max_iter, best + 1)
    history_df["selected"] = history_df.index == best
    solution, sim = candidates[best]
    return OptimizationOutcome(
        ev_power_kw=solution.ev_power_kw, bess_schedule=solution.bess_schedule,
        pv_curtailment_kw=solution.pv_curtailment_kw, simulation=sim, solution=solution,
        history=history_df, converged=converged,
    )
