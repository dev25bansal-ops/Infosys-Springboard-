"""Feature Family F2 — Depth & Imbalance (5 features).

Computed per tick from the order-book snapshot.
Latency budget: ~0.3 ms.

Features:
    6.  bid_depth_10             — total bid volume in top 10 levels
    7.  ask_depth_10             — total ask volume in top 10 levels
    8.  order_book_imbalance_10  — OBI = (V_bid - V_ask) / (V_bid + V_ask)
    9.  weighted_mid_price_10     — depth-weighted mid across top 10 levels
    10. depth_slope               — linear regression slope of size vs level index
"""
from __future__ import annotations

import numpy as np

from flash_crash_watchdog.tick import Tick


class DepthImbalanceFeatures:
    """F2 features — stateless (computed from the current snapshot)."""

    def update(self, tick: Tick) -> dict:
        book = tick.book

        bid_depth = book.bid_depth(10)
        ask_depth = book.ask_depth(10)
        obi = book.order_book_imbalance(10)
        weighted_mid = book.weighted_mid_price(10)

        # Depth slope: linear regression of (level_index, cumulative_size).
        # A steep negative bid slope or steep positive ask slope indicates
        # that depth is concentrated at the top of the book (fragile).
        bid_slope = self._depth_slope(book.bids[:10])
        ask_slope = self._depth_slope(book.asks[:10])
        depth_slope = bid_slope - ask_slope  # positive = bid-side fragile

        return {
            "f2_bid_depth_10": bid_depth,
            "f2_ask_depth_10": ask_depth,
            "f2_obi_10": obi,
            "f2_weighted_mid_10": weighted_mid,
            "f2_depth_slope": depth_slope,
        }

    @staticmethod
    def _depth_slope(levels) -> float:
        """Slope of cumulative size vs level index. Steep = fragile."""
        if len(levels) < 2:
            return 0.0
        cumulative = np.cumsum([lvl.size for lvl in levels])
        x = np.arange(len(cumulative))
        # Normalize so the slope is scale-invariant
        if cumulative[-1] == 0:
            return 0.0
        x_norm = x / max(1, len(cumulative))
        y_norm = cumulative / cumulative[-1]
        slope = np.polyfit(x_norm, y_norm, 1)[0]
        return float(slope)
