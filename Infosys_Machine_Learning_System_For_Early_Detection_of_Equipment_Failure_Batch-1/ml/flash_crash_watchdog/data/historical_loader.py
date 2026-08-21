"""Load historical tick data from parquet or CSV files."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator

import pandas as pd

from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.tick import Tick, Trade

logger = logging.getLogger(__name__)


def load_parquet(path: str | Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    logger.info("Loaded %d ticks from %s", len(df), path)
    return df


def load_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    logger.info("Loaded %d ticks from %s", len(df), path)
    return df


def df_to_ticks(df: pd.DataFrame, symbol: str = "UNKNOWN") -> Iterator[Tick]:
    # itertuples (namedtuple per row, immutable & C-backed) is 2-5x faster than
    # iterrows(), which materializes a pandas Series per row. This is on the hot
    # path of every backtest/capture loop.
    for row in df.itertuples(index=False):
        ts = int(getattr(row, "timestamp_ms", 0) or 0)
        best_bid = float(getattr(row, "best_bid", 0) or 0)
        best_ask = float(getattr(row, "best_ask", 0) or 0)
        bid_size = float(getattr(row, "bid_size", 0) or 0)
        ask_size = float(getattr(row, "ask_size", 0) or 0)

        bids = [PriceLevel(best_bid, bid_size)] if best_bid > 0 else []
        asks = [PriceLevel(best_ask, ask_size)] if best_ask > 0 else []
        book = OrderBookSnapshot(timestamp_ms=ts, bids=bids, asks=asks)

        trades = []
        if hasattr(row, "trade_price"):
            tp = getattr(row, "trade_price", None)
            try:
                f_tp = float(tp)
            except (TypeError, ValueError):
                f_tp = float("nan")
            if f_tp == f_tp:  # skip NaN
                trades.append(Trade(
                    timestamp_ms=ts,
                    price=f_tp,
                    size=float(getattr(row, "trade_size", 0) or 0),
                    side=str(getattr(row, "trade_side", "buy")),
                ))

        yield Tick(book=book, trades=trades, symbol=symbol)
