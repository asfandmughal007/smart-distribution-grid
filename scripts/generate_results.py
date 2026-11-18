"""Build the comparison tables and all figures from saved simulation results.

Usage:
    python scripts/generate_results.py [--config config/config.yaml]

Requires the outputs of run_scenarios.py, run_optimization.py and (for the
experiment figures) run_experiments.py.
"""

from __future__ import annotations

import argparse
import logging

import _bootstrap  # noqa: F401
import pandas as pd

from src.analysis import visualization as viz
from src.analysis.metrics import compute_metrics, summary_table, to_markdown, write_table
from src.simulation.scenarios import load_scenarios
from src.utils.config import PROJECT_ROOT, load_config, setup_logging
from src.utils.storage import load_result

logger = logging.getLogger("generate_results")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate tables and figures from saved results.")
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    figures = PROJECT_ROOT / config["output"]["figures_dir"]
    tables = PROJECT_ROOT / config["output"]["tables_dir"]
    dpi = config["output"]["figure_dpi"]
    limits = config["limits"]

    scenarios = load_scenarios(config)
    results = {s.name: load_result(s.name) for s in scenarios}
    labels = {s.name: s.label for s in scenarios}
    metrics = {labels[name]: compute_metrics(res.simulation, res.data.price, config) for name, res in results.items()}

    # ---- scenario comparison tables ------------------------------------------------
    table = summary_table(metrics)
    write_table(table, results_dir / "scenario_comparison.csv", results_dir / "scenario_comparison.md")
    pd.DataFrame(metrics).T.rename_axis("scenario").to_csv(tables / "scenario_metrics_full.csv")
    print(table.to_string(index=False))

    # ---- figures -------------------------------------------------------------------
    viz.apply_style()
    r = results
    paths = [
        viz.plot_load_profile(r["S1_baseline"], figures, dpi),
        viz.plot_pv_profile(r["S2_high_pv"], figures, dpi),
        viz.plot_ev_profile(r["S4_pv_ev"], r["S6_smart_charging"], figures, dpi),
        viz.plot_price(r["S1_baseline"], figures, dpi),
    ]
    compare = {labels[n]: r[n] for n in ("S4_pv_ev", "S6_smart_charging", "S7_smart_bess")}
    paths += [
        viz.plot_grid_import(compare, figures, dpi),
        viz.plot_transformer_loading(compare, limits["trafo_max_loading_pct"], figures, dpi),
    ]
    voltage_cases = {labels[n]: r[n] for n in ("S1_baseline", "S4_pv_ev", "S6_smart_charging", "S7_smart_bess")}
    paths += [
        viz.plot_min_voltage(voltage_cases, limits["v_min_pu"], figures, dpi),
        viz.plot_battery({labels[n]: r[n] for n in ("S5_pv_ev_bess", "S7_smart_bess")}, figures, dpi),
        viz.plot_scenario_comparison(metrics, figures, dpi),
        viz.plot_voltage_profile(voltage_cases, limits["v_min_pu"], figures, dpi),
    ]

    experiments = {name: tables / f"experiment_{name}.csv" for name in ("pv_ev", "hosting", "bess", "ev_control", "tradeoff")}
    available = {name: pd.read_csv(path) for name, path in experiments.items() if path.is_file()}
    if "hosting" in available:
        paths.append(viz.plot_pv_curtailment(available["hosting"], limits["v_max_pu"],
                                             limits["trafo_max_loading_pct"], figures, dpi))
    if "tradeoff" in available:
        paths.append(viz.plot_tradeoff(available["tradeoff"], figures, dpi))
    if "bess" in available:
        paths.append(viz.plot_bess_sweep(available["bess"], figures, dpi))
    if "pv_ev" in available:
        paths.append(viz.plot_pv_ev_heatmap(available["pv_ev"], figures, dpi))
    missing = set(experiments) - set(available)
    if missing:
        logger.warning("Experiment tables missing (run scripts/run_experiments.py): %s", sorted(missing))

    # Markdown versions of the experiment tables for the documentation.
    for name, frame in available.items():
        rounded = frame.round(4)
        (tables / f"experiment_{name}.md").write_text(to_markdown(rounded) + "\n", encoding="utf-8")

    for path in paths:
        logger.info("Figure %s", path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
