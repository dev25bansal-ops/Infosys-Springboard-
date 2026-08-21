#!/usr/bin/env python3
"""Run the backtest with TRAINED models (loads saved model weights).

Usage:
    python scripts/run_backtest_trained.py --data data/parquet/BTCUSDT_2021-05-19.parquet
    python scripts/run_backtest_trained.py --data data/parquet/LUNAUSDT_2022-05-10.parquet
"""
import argparse
import logging
import sys
from pathlib import Path

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.cascade import DetectionCascade
from flash_crash_watchdog.data.historical_loader import load_parquet
from flash_crash_watchdog.eval.backtest import run_backtest
from flash_crash_watchdog.models.stage2_isolation_forest import Stage2IsolationForest
from flash_crash_watchdog.models.stage3_tcn import Stage3TCN, TCNConfig


def main() -> int:
    parser = argparse.ArgumentParser(description="Run backtest with trained models")
    parser.add_argument("--data", required=True, help="Parquet file of crash data")
    parser.add_argument("--config", default="configs/pipeline.yml")
    parser.add_argument("--models", default="models/", help="Directory with trained models")
    parser.add_argument("--output", default="results/backtest_trained.json")
    parser.add_argument("--max-ticks", type=int, default=0,
                        help="Max ticks to process (0 = all, for speed use 500000)")
    parser.add_argument("--stage2", default=None, help="Explicit Stage-2 (.joblib) path (overrides auto-discovery)")
    parser.add_argument("--stage3", default=None, help="Explicit Stage-3 (.pt) path (overrides auto-discovery)")
    parser.add_argument("--stage3-threshold", type=float, default=None,
                        help="Override the Stage-3 pass threshold (default: whatever the checkpoint uses, 0.6)")
    parser.add_argument("--stage5-threshold", type=float, default=None,
                        help="Override the Stage-5 (Bayesian) posterior alert threshold (default: keep config/pipeline.yml value)")
    parser.add_argument("--cooldown-ms", type=int, default=0,
                        help="Coalesce alert bursts: keep at most one alert per interval (ms). 0 = every alert")
    parser.add_argument("--scan", action="store_true",
                        help="Also print the window-scan (5s-lookahead) metrics on the same data so deployed and scan F1 are comparable")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logger = logging.getLogger(__name__)

    # Load cascade
    cascade = DetectionCascade.from_config(args.config)

    # Locate trained models (canonical names first, legacy fallbacks, explicit overrides).
    # Stage-3 checkpoint is canonical as stage3_tcn_trained.pt (train_tcn_windows.py);
    # older train_models/train_gpu wrote stage3_tcn.pt — accept both. Stage3TCN.load
    # rebuilds the model to match its own checkpoint config, so architecture always lines up.
    models_dir = Path(args.models)
    stage2_path = args.stage2 or next(
        (p for p in [models_dir / "stage2_isolation_forest.joblib"] if p.exists()), None)
    stage3_path = args.stage3 or next(
        (p for p in [models_dir / "stage3_tcn_prod.pt",
                     models_dir / "stage3_tcn_v2.pt",
                     models_dir / "stage3_tcn_trained.pt",
                     models_dir / "stage3_tcn.pt"] if p.exists()),
        None)

    if stage2_path is not None:
        logger.info("Loading trained Stage 2 from %s", stage2_path)
        cascade.s2.load(stage2_path)
    else:
        logger.warning("No trained Stage 2 found under %s — using untrained fallback", models_dir)

    if stage3_path is not None:
        logger.info("Loading trained Stage 3 from %s", stage3_path)
        cascade.s3.load(stage3_path)
        if args.stage3_threshold is not None:
            cascade.s3._threshold = args.stage3_threshold
            logger.info("Stage-3 threshold set to %.2f", args.stage3_threshold)
    else:
        logger.warning("No trained Stage 3 found under %s — using untrained fallback", models_dir)

    if args.stage5_threshold is not None:
        cascade.s5.config.alert_threshold = args.stage5_threshold
        logger.info("Stage-5 posterior alert threshold set to %.3f", args.stage5_threshold)

    # Load data
    df = load_parquet(args.data)
    if args.max_ticks > 0 and len(df) > args.max_ticks:
        logger.info("Sampling down to %d ticks (from %d) for speed", args.max_ticks, len(df))
        indices = range(0, len(df), len(df) // args.max_ticks)
        df = df.iloc[indices[:args.max_ticks]].copy()

    # Run backtest
    results = run_backtest(cascade, df, cooldown_ms=args.cooldown_ms)
    results.print_summary()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.save(output_path)

    # Optional: print the window-scan (5s-lookahead) metrics on the same data so
    # the deployed (60s-window + cooldown) and scan F1 are directly comparable.
    if args.scan and stage3_path is not None:
        import subprocess
        logger.info("Window-scan metrics (5s-lookahead, stride-20) for comparison:")
        subprocess.run([
            sys.executable, str(Path(__file__).parent / "tcn_score_diag.py"),
            "--data", args.data,
            "--model", str(args.stage3 or stage3_path),
            "--lookahead", "500", "--normalize",
        ])
    elif args.scan:
        logger.warning("--scan requested but no Stage-3 checkpoint found to scan with")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
