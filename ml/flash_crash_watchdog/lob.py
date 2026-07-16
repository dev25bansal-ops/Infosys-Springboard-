"""Limit Order Book (LOB) data structures.

A LOB is the real-time queue of outstanding buy (bid) and sell (ask) orders
at every price level. This module provides a simple in-memory representation
plus utilities for reconstruction from a stream of order events.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class PriceLevel:
    """A single price level in the order book."""
    price: float
    size: float

    def __repr__(self) -> str:
        return f"PriceLevel(price={self.price:.4f}, size={self.size:.4f})"


@dataclass
class OrderBookSnapshot:
    """A snapshot of the LOB at a point in time.

    Attributes:
        timestamp_ms: Unix timestamp in milliseconds.
        bids: List of (price, size) tuples, sorted descending by price.
        asks: List of (price, size) tuples, sorted ascending by price.
    """
    timestamp_ms: int
    bids: List[PriceLevel] = field(default_factory=list)
    asks: List[PriceLevel] = field(default_factory=list)

    @property
    def best_bid(self) -> Optional[float]:
        return self.bids[0].price if self.bids else None

    @property
    def best_ask(self) -> Optional[float]:
        return self.asks[0].price if self.asks else None

    @property
    def mid_price(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return (self.best_bid + self.best_ask) / 2.0

    @property
    def spread(self) -> Optional[float]:
        if self.best_bid is None or self.best_ask is None:
            return None
        return self.best_ask - self.best_bid

    @property
    def spread_bps(self) -> Optional[float]:
        """Spread in basis points, relative to mid-price."""
        if self.spread is None or self.mid_price is None or self.mid_price == 0:
            return None
        return (self.spread / self.mid_price) * 10_000

    @property
    def micro_price(self) -> Optional[float]:
        """Volume-weighted mid-price.

        micro_price = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)

        Reflects which side has more resting volume — a better fair-value
        estimate than the simple mid-price when the book is imbalanced.
        """
        if not self.bids or not self.asks:
            return None
        bb, bs = self.bids[0].price, self.bids[0].size
        ba, as_ = self.asks[0].price, self.asks[0].size
        denom = bs + as_
        if denom == 0:
            return self.mid_price
        return (bb * as_ + ba * bs) / denom

    def bid_depth(self, levels: int = 10) -> float:
        """Total bid volume in the top N levels."""
        return sum(level.size for level in self.bids[:levels])

    def ask_depth(self, levels: int = 10) -> float:
        """Total ask volume in the top N levels."""
        return sum(level.size for level in self.asks[:levels])

    def order_book_imbalance(self, levels: int = 10) -> float:
        """OBI = (V_bid - V_ask) / (V_bid + V_ask), range [-1, 1].

        Positive = more bid-side volume (bullish pressure).
        Negative = more ask-side volume (bearish pressure).
        """
        vb = self.bid_depth(levels)
        va = self.ask_depth(levels)
        if vb + va == 0:
            return 0.0
        return (vb - va) / (vb + va)

    def weighted_mid_price(self, levels: int = 10) -> Optional[float]:
        """Depth-weighted mid-price across top N levels."""
        if not self.bids or not self.asks:
            return None
        vb = self.bid_depth(levels)
        va = self.ask_depth(levels)
        if vb + va == 0:
            return self.mid_price
        bb = self.bids[0].price
        ba = self.asks[0].price
        return (bb * vb + ba * va) / (vb + va)

    def to_feature_dict(self) -> dict:
        """Convert to a flat dict suitable for feature extraction."""
        return {
            "timestamp_ms": self.timestamp_ms,
            "best_bid": self.best_bid,
            "best_ask": self.best_ask,
            "mid_price": self.mid_price,
            "micro_price": self.micro_price,
            "spread": self.spread,
            "spread_bps": self.spread_bps,
            "bid_depth_10": self.bid_depth(10),
            "ask_depth_10": self.ask_depth(10),
            "obi_10": self.order_book_imbalance(10),
            "weighted_mid_10": self.weighted_mid_price(10),
        }


class OrderBookReconstructor:
    """Reconstructs a limit order book from a stream of order events.

    Maintains two sorted lists (bids descending, asks ascending) and
    applies insert/cancel/execute events to keep them up to date.
    """

    def __init__(self) -> None:
        self._bids: dict[float, float] = {}  # price -> size
        self._asks: dict[float, float] = {}

    def update_level(self, side: str, price: float, size: float) -> None:
        """Insert, update, or remove a price level.

        Args:
            side: "bid" or "ask"
            price: The price level
            size: New size. 0 removes the level.
        """
        book = self._bids if side.lower() == "bid" else self._asks
        if size == 0:
            book.pop(price, None)
        else:
            book[price] = size

    def snapshot(self, timestamp_ms: int, levels: int = 20) -> OrderBookSnapshot:
        """Take a snapshot of the top N levels."""
        bids = sorted(self._bids.items(), key=lambda x: -x[0])[:levels]
        asks = sorted(self._asks.items(), key=lambda x: x[0])[:levels]
        return OrderBookSnapshot(
            timestamp_ms=timestamp_ms,
            bids=[PriceLevel(p, s) for p, s in bids],
            asks=[PriceLevel(p, s) for p, s in asks],
        )
