"""Run the baseline scenario (S1) and report its key grid metrics.

Usage:
    python scripts/run_baseline.py [--config config/config.yaml]
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401  (adds project root to sys.path)
import pandas as pd

from src.analysis.metrics import compute_metrics
from src.simulation.scenarios import get_scenario, run_scenario
from src.utils.config import PROJECT_ROOT, load_config, setup_logging
from src.utils.storage import save_result

logger = logging.getLogger("run_baseline")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the baseline scenario (S1).")
    parser.add_argument("--config", default=None, help="Path to a YAML configuration file")
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)

    result = run_scenario(config, get_scenario(config, "S1_baseline"))
    save_result(result, "S1_baseline")
    m = compute_metrics(result.simulation, result.data.price, config)

    report = [
        ("Minimum bus voltage", f"{m['v_min_pu']:.4f} pu at bus {m['v_min_bus']} (IEEE bus {m['v_min_bus'] + 1})"),
        ("Maximum bus voltage", f"{m['v_max_pu']:.4f} pu"),
        ("Voltage violations", f"{m['voltage_violation_bus_steps']} bus-steps, "
                               f"{m['voltage_violation_duration_h']:.2f} h, {m['buses_with_violation']} buses"),
        ("Maximum line loading", f"{m['line_loading_max_pct']:.1f} %"),
        ("Maximum transformer loading", f"{m['trafo_loading_max_pct']:.1f} %"),
        ("Peak grid import", f"{m['peak_import_kw']:.0f} kW"),
        ("Total grid import", f"{m['grid_import_kwh']:.0f} kWh"),
        ("Total losses", f"{m['losses_kwh']:.0f} kWh ({m['losses_pct_of_supply']:.2f} % of supply)"),
        ("Energy balance residual", f"{result.validation['energy_balance_residual_kwh']:.2e} kWh"),
    ]
    width = max(len(name) for name, _ in report)
    print("\nBaseline (S1) - 24 h AC time-series power flow")
    for name, value in report:
        print(f"  {name:<{width}} : {value}")

    out = PROJECT_ROOT / "results" / "tables" / "baseline_metrics.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(m, name="value").rename_axis("metric").to_csv(out, header=True)
    logger.info("Wrote %s", out.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
