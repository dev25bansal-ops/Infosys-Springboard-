"""Command-line interface for the Flash Crash Early Warning detector.

Usage:
    python -m flash_crash_watchdog.cli live --symbol BTCUSDT
    python -m flash_crash_watchdog.cli backtest --data data/BTCUSDT_2021-05-19.parquet
    python -m flash_crash_watchdog.cli train --data data/fi2010/ --model configs/tcn_baseline.yml
"""
import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from flash_crash_watchdog.cascade import DetectionCascade

# ENH-05: resolve default config paths against the REPO ROOT (ml/../..), not the
# process CWD, so `python -m flash_crash_watchdog.cli ...` works from any
# directory (the Makefile runs it from ml/, the README from the repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = str(REPO_ROOT / "configs" / "pipeline.yml")
DEFAULT_MODEL_CONFIG = str(REPO_ROOT / "configs" / "tcn_baseline.yml")
MODELS_DIR = str(REPO_ROOT / "models")


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
        datefmt="%H:%M:%S",
    )


def cmd_live(args: argparse.Namespace) -> int:
    """Run the detector against a live Binance WebSocket stream."""
    from flash_crash_watchdog.alert import AlertRouter
    from flash_crash_watchdog.data.live_stream import BinanceLiveStream

    cascade = DetectionCascade.from_config(args.config, models_dir=MODELS_DIR)

    # Route fired alerts to console (always) + JSONL file (--log) and
    # Slack/PagerDuty/webhook/email when the corresponding env vars are set.
    smtp = {
        "host": os.environ.get("SMTP_HOST"),
        "port": os.environ.get("SMTP_PORT", "587"),
        "user": os.environ.get("SMTP_USER"),
        "password": os.environ.get("SMTP_PASS"),
        "to": os.environ.get("SMTP_TO"),
        "from": os.environ.get("SMTP_FROM"),
    }
    router = AlertRouter(
        log_path=args.log,
        slack_webhook=os.environ.get("SLACK_WEBHOOK"),
        pagerduty_key=os.environ.get("PAGERDUTY_KEY"),
        webhook_url=args.webhook or os.environ.get("ALERT_WEBHOOK"),
        telegram_bot_token=os.environ.get("TELEGRAM_BOT_TOKEN"),
        telegram_chat_id=os.environ.get("TELEGRAM_CHAT_ID"),
        smtp_config=smtp,
        dry_run=args.dry_run,
    )
    cascade.on_alert(router.route)
    if args.log:
        logging.info("Alerts will be appended to %s", args.log)
    if router.slack_webhook or router.pagerduty_key or router.webhook_url or router.telegram_bot_token or smtp.get("to"):
        logging.info(
            "Alert delivery enabled: slack=%s pagerduty=%s webhook=%s telegram=%s email=%s dry_run=%s",
            bool(router.slack_webhook), bool(router.pagerduty_key), bool(router.webhook_url),
            bool(router.telegram_bot_token), bool(smtp.get("to")), args.dry_run,
        )
    if args.source:
        logging.warning(
            "--source (Rust proxy TCP) is parsed but not wired yet — ignoring it and "
            "using the Binance WebSocket directly (BUG-01)."
        )

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
    # ENH-05: `--model` is a README/quickstart alias for the config YAML (the
    # documented command is `cli backtest --data ... --model configs/tcn_baseline.yml`).
    config_path = args.model or args.config
    cascade = DetectionCascade.from_config(config_path, models_dir=MODELS_DIR)
    results = run_backtest(cascade, ticks, window_ms=args.window)
    results.print_summary()
    if args.output:
        results.save(args.output)
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the TCN model on labeled windows (or build them from tick data).

    ``--data`` may be:
      * a ``.npz`` window file (windows/labels keys),
      * a directory of ``.npz`` window files or ``.parquet`` tick files, or
      * a single ``.parquet``/``.csv`` tick file (windows+labels are built
        on the fly with data.windows.build_windows_from_df).
    """
    import torch

    from flash_crash_watchdog.data.windows import resolve_windows_source
    from flash_crash_watchdog.models.stage3_tcn import TCNDetector

    windows, labels, _feature_names = resolve_windows_source(args.data, max_ticks=args.max_ticks)
    model = TCNDetector.from_config(args.model)  # yaml -> model config
    model.train_on_windows(
        windows, labels,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    out_path = Path(args.output) if args.output else Path(args.model).with_suffix(".pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "config": model.config}, out_path)
    logging.info("Saved TCN model to %s", out_path)
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
    p_live.add_argument("--source", default=None, help="tcp://host:port for Rust proxy (NOT wired — see BUG-01)")
    p_live.add_argument("--config", default=DEFAULT_CONFIG)
    p_live.add_argument("--log", default=None,
                        help="Path to append fired alerts as JSONL (optional)")
    p_live.add_argument("--webhook", default=None,
                        help="Generic JSON webhook URL for alerts (or env ALERT_WEBHOOK)")
    p_live.add_argument("--dry-run", action="store_true",
                        help="Log intended Slack/PD/webhook/email deliveries without sending")
    p_live.set_defaults(func=cmd_live)

    # backtest
    p_bt = sub.add_parser("backtest", help="Run on historical data")
    p_bt.add_argument("--data", required=True, help="Path to parquet/csv file")
    p_bt.add_argument("--config", default=DEFAULT_CONFIG)
    p_bt.add_argument("--model", default=None,
                      help="Alias for --config (the README quickstart passes --model configs/tcn_baseline.yml)")
    p_bt.add_argument("--window", type=int, default=500, help="Window size (ms)")
    p_bt.add_argument("--output", default=None, help="Save results to JSON")
    p_bt.set_defaults(func=cmd_backtest)

    # train
    p_tr = sub.add_parser("train", help="Train the TCN model on labeled windows")
    p_tr.add_argument("--data", required=True,
                      help="Path: a .npz window file, a directory of .npz/.parquet, or a tick .parquet/.csv")
    p_tr.add_argument("--model", default=DEFAULT_MODEL_CONFIG,
                      help="TCN config YAML (model architecture)")
    p_tr.add_argument("--epochs", type=int, default=50)
    p_tr.add_argument("--batch-size", type=int, default=128)
    p_tr.add_argument("--lr", type=float, default=1e-3)
    p_tr.add_argument("--device", default="auto", help="cuda, cpu, or auto")
    p_tr.add_argument("--max-ticks", type=int, default=0,
                        help="Cap ticks used when building windows from a parquet/csv (0 = all)")
    p_tr.add_argument("--output", default=None,
                        help="Output .pt path (default: <model>.yml -> <model>.pt)")
    p_tr.set_defaults(func=cmd_train)

    args = parser.parse_args()
    setup_logging(args.verbose)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
