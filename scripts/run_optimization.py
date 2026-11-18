"""Run the optimized scenarios (smart charging, smart charging + BESS) with AC validation.

Usage:
    python scripts/run_optimization.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401
import pandas as pd

from src.simulation.scenarios import load_scenarios, run_scenario
from src.utils.config import PROJECT_ROOT, load_config, setup_logging
from src.utils.storage import save_result

logger = logging.getLogger("run_optimization")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the optimized scenarios with AC validation.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    tables = PROJECT_ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    histories, objectives = [], []
    for scenario in load_scenarios(config):
        if not scenario.is_optimized:
            continue
        result = run_scenario(config, scenario)
        save_result(result, scenario.name)
        outcome = result.optimization
        history = outcome.history.assign(scenario=scenario.name)
        histories.append(history)
        objectives.append({
            "scenario": scenario.name,
            "solver": outcome.solution.solver_status,
            "objective_eur": outcome.solution.objective_eur,
            **{f"term_{key}_eur": value for key, value in outcome.solution.objective_terms.items()},
            "iterations": len(history),
            "validated": outcome.converged,
        })
        print(f"\n{scenario.label}: iterative AC validation")
        print(history.drop(columns="scenario").round(4).to_string(index=False))

    pd.concat(histories).to_csv(tables / "optimization_iterations.csv", index=False)
    pd.DataFrame(objectives).to_csv(tables / "optimization_objective.csv", index=False)
    print()
    print(pd.DataFrame(objectives).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
