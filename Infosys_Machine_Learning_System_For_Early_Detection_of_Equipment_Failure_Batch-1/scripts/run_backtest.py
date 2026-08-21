#!/usr/bin/env python3
"""Run the offline backtest on a historical data file.

Usage:
    python scripts/run_backtest.py --data data/BTCUSDT_2021-05-19.parquet
"""
import argparse
import logging
import sys
from pathlib import Path

# Insert the ml directory at the FRONT of sys.path so we import the local
# package (with all submodules: data/, features/, models/, eval/, alert/)
# rather than any pip-installed version that might be incomplete.
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

# Also add the project root so 'flash_crash_watchdog' resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.cascade import DetectionCascade
from flash_crash_watchdog.data.historical_loader import load_parquet
from flash_crash_watchdog.eval.backtest import run_backtest


def main() -> int:
    parser = argparse.ArgumentParser(description="Run offline backtest")
    parser.add_argument("--data", required=True, help="Parquet or CSV file")
    parser.add_argument("--config", default="configs/pipeline.yml")
    parser.add_argument("--output", default="results/backtest_results.json")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_parquet(args.data)
    cascade = DetectionCascade.from_config(args.config)
    results = run_backtest(cascade, df)
    results.print_summary()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.save(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
