#!/usr/bin/env python3
"""Replay a historical crash through the detector at real-time speed.

Usage:
    python scripts/replay_crash.py --data data/BTCUSDT_2021-05-19.parquet --speed 10
"""
import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

# Insert the ml directory at the FRONT of sys.path
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.cascade import DetectionCascade
from flash_crash_watchdog.data.historical_loader import df_to_ticks, load_parquet


async def replay(df, cascade: DetectionCascade, speed: float = 1.0) -> None:
    """Replay ticks at real-time speed (or speed*x)."""
    ticks = list(df_to_ticks(df))
    print(f"Replaying {len(ticks)} ticks at {speed}x speed...")

    for i, tick in enumerate(ticks):
        if i > 0:
            delay_ms = (tick.timestamp_ms - ticks[i - 1].timestamp_ms) / speed
            if 0 < delay_ms < 10_000:
                await asyncio.sleep(delay_ms / 1000)
        cascade.process_tick(tick)
        if i % 1000 == 0:
            print(f"  Progress: {i}/{len(ticks)}  alerts={cascade.stats.alerts_fired}")

    cascade.print_stats()


def main() -> int:
    parser = argparse.ArgumentParser(description="Replay a historical crash")
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/pipeline.yml")
    parser.add_argument("--speed", type=float, default=10.0, help="Replay speed multiplier")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = load_parquet(args.data)
    cascade = DetectionCascade.from_config(args.config)
    asyncio.run(replay(df, cascade, args.speed))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
