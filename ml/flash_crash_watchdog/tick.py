"""Tick data structure for the detection pipeline.

A Tick is the atomic unit processed by the cascade. It bundles an
OrderBookSnapshot with the most recently computed feature vector
and any trade events that occurred since the last tick.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from flash_crash_watchdog.lob import OrderBookSnapshot


@dataclass
class Trade:
    """A single executed trade."""
    timestamp_ms: int
    price: float
    size: float
    side: str  # "buy" or "sell" (aggressor side)


@dataclass
class Tick:
    """One observation passed through the detection cascade.

    Attributes:
        book: Current order-book snapshot.
        trades: Trades since the last tick (may be empty).
        features: Computed feature vector (populated by the feature
            extractor before the cascade runs).
        symbol: The instrument this tick belongs to.
    """
    book: OrderBookSnapshot
    trades: List[Trade] = field(default_factory=list)
    features: dict = field(default_factory=dict)
    symbol: str = ""

    @property
    def timestamp_ms(self) -> int:
        return self.book.timestamp_ms
