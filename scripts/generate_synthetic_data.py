#!/usr/bin/env python3
"""Generate synthetic LOB data for testing without downloading real data.

Creates a parquet file with 1 hour of synthetic ticks, including a flash-crash
window at the 30-minute mark.
"""
import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd


def generate_synthetic_ticks(
    n_ticks: int = 360_000,  # 1 hour at 10 ticks/sec
    base_price: float = 100.0,
    crash_at_tick: int = 180_000,  # crash at 30 min
    crash_duration_ticks: int = 600,  # 60-second crash
    crash_drop_pct: float = 0.05,  # 5% drop
) -> pd.DataFrame:
    """Generate synthetic LOB data with a flash crash."""
    rng = np.random.default_rng(42)

    timestamps = np.arange(n_ticks) * 10  # 10ms per tick
    prices = np.full(n_ticks, base_price)

    # Normal price movement (random walk)
    returns = rng.normal(0, 0.0001, n_ticks)
    prices = base_price * np.cumprod(1 + returns)

    # Inject flash crash
    crash_start = crash_at_tick
    crash_end = crash_at_tick + crash_duration_ticks
    crash_returns = np.linspace(0, -crash_drop_pct, crash_duration_ticks)
    prices[crash_start:crash_end] *= (1 + crash_returns)

    # Recovery (V-shape)
    recovery_returns = np.linspace(0, crash_drop_pct * 0.7, crash_duration_ticks)
    prices[crash_end:crash_end + crash_duration_ticks] *= (1 + recovery_returns)

    # Generate bid/ask around mid
    spreads = rng.uniform(0.01, 0.03, n_ticks)
    best_bids = prices - spreads / 2
    best_asks = prices + spreads / 2

    # Generate sizes (with liquidity withdrawal during crash)
    base_size = 1.0
    sizes = rng.uniform(0.5, 2.0, n_ticks)
    # Liquidity withdrawal during crash
    crash_window = slice(crash_start, crash_end)
    sizes[crash_window] *= rng.uniform(0.1, 0.3, crash_duration_ticks)

    df = pd.DataFrame({
        "timestamp_ms": timestamps,
        "best_bid": best_bids,
        "best_ask": best_asks,
        "bid_size": sizes,
        "ask_size": sizes * rng.uniform(0.8, 1.2, n_ticks),
        "mid_price": prices,
    })
    return df


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate synthetic LOB data")
    parser.add_argument("--out", default="data/synthetic_crash.parquet")
    parser.add_argument("--ticks", type=int, default=360_000)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = generate_synthetic_ticks(n_ticks=args.ticks)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Generated {len(df)} synthetic ticks -> {out_path}")
    print(f"Crash window: ticks 180000-180600 (at ~30 min)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
