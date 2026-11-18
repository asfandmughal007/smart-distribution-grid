"""Run the parameter sweeps (PV x EV, hosting capacity, BESS size, EV control, demand charge).

Usage:
    python scripts/run_experiments.py [--config config/config.yaml] [--only NAME ...]

Takes several minutes because every optimized run includes iterative AC validation.
"""

from __future__ import annotations

import argparse
import logging
import time

import _bootstrap  # noqa: F401

from src.analysis import experiments
from src.simulation.scenarios import get_scenario
from src.utils.config import PROJECT_ROOT, load_config, setup_logging

logger = logging.getLogger("run_experiments")

SWEEPS = ["pv_ev", "hosting", "bess", "ev_control", "tradeoff"]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the parameter sweeps.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--only", nargs="*", default=None, choices=SWEEPS)
    args = parser.parse_args(argv)
    setup_logging()
    config = load_config(args.config)
    out_dir = PROJECT_ROOT / "results" / "tables"
    out_dir.mkdir(parents=True, exist_ok=True)

    # The sweeps vary one dimension around the full smart-control scenario S7.
    ref = get_scenario(config, "S7_smart_bess")
    runners = {
        "pv_ev": lambda: experiments.pv_ev_sweep(config),
        "hosting": lambda: experiments.hosting_capacity_sweep(config, ref.ev_level, ref.bess_capacity_kwh),
        "bess": lambda: experiments.bess_sweep(config, ref.pv_penetration_pct, ref.ev_level),
        "ev_control": lambda: experiments.ev_control_sweep(config, ref.pv_penetration_pct),
        "tradeoff": lambda: experiments.peak_charge_tradeoff(config, ref),
    }
    for name in SWEEPS:
        if args.only and name not in args.only:
            continue
        start = time.perf_counter()
        table = runners[name]()
        path = out_dir / f"experiment_{name}.csv"
        table.to_csv(path, index=False)
        logger.info("Experiment %s finished in %.0f s -> %s", name, time.perf_counter() - start,
                    path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
