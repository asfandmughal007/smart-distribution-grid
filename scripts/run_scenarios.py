"""Run the non-optimized scenarios (uncontrolled EVs / rule-based BESS).

Usage:
    python scripts/run_scenarios.py [--config config/config.yaml] [--all]

``--all`` also runs the optimized scenarios (same as running
scripts/run_optimization.py afterwards).
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401

from src.analysis.metrics import compute_metrics, summary_table, write_table
from src.simulation.scenarios import load_scenarios, run_scenario
from src.utils.config import PROJECT_ROOT, load_config, setup_logging
from src.utils.storage import save_result

logger = logging.getLogger("run_scenarios")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the scenarios defined in the configuration.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--all", action="store_true", help="include optimized scenarios")
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)

    metrics = {}
    for scenario in load_scenarios(config):
        if scenario.is_optimized and not args.all:
            continue
        result = run_scenario(config, scenario)
        save_result(result, scenario.name)
        metrics[scenario.label] = compute_metrics(result.simulation, result.data.price, config)

    table = summary_table(metrics)
    out = PROJECT_ROOT / "results" / "tables" / "scenarios_run.csv"
    write_table(table, out)
    print(table.to_string(index=False))
    logger.info("Wrote %s", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
