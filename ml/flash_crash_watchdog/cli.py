"""Command-line interface for the Flash Crash Early Warning detector.

Usage:
    python -m flash_crash_watchdog.cli live --symbol BTCUSDT
    python -m flash_crash_watchdog.cli backtest --data data/BTCUSDT_2021-05-19.parquet
    python -m flash_crash_watchdog.cli train --data data/fi2010/ --model configs/tcn_baseline.yml
"""
import argparse
import asyncio
import logging
import sys
from pathlib import Path

from flash_crash_watchdog.cascade import DetectionCascade


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_live(args: argparse.Namespace) -> int:
    """Run the detector against a live Binance WebSocket stream."""
    from flash_crash_watchdog.data.live_stream import BinanceLiveStream

    cascade = DetectionCascade.from_config(args.config)
    stream = BinanceLiveStream(
        symbol=args.symbol,
        depth_levels=args.depth,
        on_tick=cascade.process_tick,
    )
    logging.info("Starting live detector on %s (depth=%d)", args.symbol, args.depth)
    try:
        asyncio.run(stream.run())
    except KeyboardInterrupt:
        logging.info("Shutting down...")
    return 0


def cmd_backtest(args: argparse.Namespace) -> int:
    """Run the detector on a historical data file."""
    from flash_crash_watchdog.data.historical_loader import load_parquet
    from flash_crash_watchdog.eval.backtest import run_backtest

    ticks = load_parquet(args.data)
    cascade = DetectionCascade.from_config(args.config)
    results = run_backtest(cascade, ticks, window_ms=args.window)
    results.print_summary()
    if args.output:
        results.save(args.output)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the TCN model on a dataset."""
    from flash_crash_watchdog.data.fi2010_loader import load_fi2010
    from flash_crash_watchdog.models.stage3_tcn import TCNDetector

    train_ds, val_ds = load_fi2010(args.data)
    model = TCNDetector.from_config(args.model)
    model.train(train_ds, val_ds, epochs=args.epochs)
    model.save(args.model.replace(".yml", ".pt"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="flash-crash-watchdog",
        description="Real-time flash-crash detector on LOB streams.",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    # live
    p_live = sub.add_parser("live", help="Run against live Binance WebSocket")
    p_live.add_argument("--symbol", default="BTCUSDT")
    p_live.add_argument("--depth", type=int, default=20)
    p_live.add_argument("--source", default=None, help="tcp://host:port for Rust proxy")
    p_live.add_argument("--config", default="configs/pipeline.yml")
    p_live.set_defaults(func=cmd_live)

    # backtest
    p_bt = sub.add_parser("backtest", help="Run on historical data")
    p_bt.add_argument("--data", required=True, help="Path to parquet/csv file")
    p_bt.add_argument("--config", default="configs/pipeline.yml")
    p_bt.add_argument("--window", type=int, default=500, help="Window size (ms)")
    p_bt.add_argument("--output", default=None, help="Save results to JSON")
    p_bt.set_defaults(func=cmd_backtest)

    # train
    p_tr = sub.add_parser("train", help="Train the TCN model")
    p_tr.add_argument("--data", required=True, help="FI-2010 directory")
    p_tr.add_argument("--model", default="configs/tcn_baseline.yml")
    p_tr.add_argument("--epochs", type=int, default=50)
    p_tr.set_defaults(func=cmd_train)

    args = parser.parse_args()
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
