"""Feature Family F3 — Flow & Toxicity (4 features).

Computed per tick from the order-book snapshot and recent trade list.
Latency budget: ~0.5 ms.

Features:
    11. vpin              — Volume-Synchronized Probability of Informed Trading
    12. kyle_lambda       — Price impact coefficient (price change per unit volume)
    13. effective_spread  — 2 * (exec_price - mid_price) signed by trade side
    14. realized_spread   — 5-minute ahead price change minus effective spread
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, List

from flash_crash_watchdog.tick import Tick, Trade


@dataclass
class _TradeWithMid:
    timestamp_ms: int
    price: float
    size: float
    side: str
    mid: float


class FlowToxicityFeatures:
    """Maintains rolling state for F3 features."""

    def __init__(self, vpin_bucket_volume: float = 1.0) -> None:
        # VPIN state
        self._vpin_bucket_volume = vpin_bucket_volume
        self._current_bucket_buy = 0.0
        self._current_bucket_sell = 0.0
        self._buckets: Deque[float] = deque(maxlen=50)

        # Kyle's lambda state — rolling regression of (volume, price_change)
        self._recent_flows: Deque[tuple] = deque(maxlen=200)  # (signed_volume, price_change)

        # Effective spread state
        self._recent_trades: Deque[_TradeWithMid] = deque(maxlen=500)

    def update(self, tick: Tick) -> dict:
        mid = tick.book.mid_price
        if mid is None:
            return self._empty()

        # VPIN: bucket trades by volume, compute buy-sell imbalance per bucket
        for trade in tick.trades:
            signed_vol = trade.size if trade.side == "buy" else -trade.size
            if trade.side == "buy":
                self._current_bucket_buy += trade.size
            else:
                self._current_bucket_sell += trade.size

            bucket_total = self._current_bucket_buy + self._current_bucket_sell
            if bucket_total >= self._vpin_bucket_volume:
                imbalance = abs(self._current_bucket_buy - self._current_bucket_sell)
                self._buckets.append(imbalance / bucket_total)
                self._current_bucket_buy = 0.0
                self._current_bucket_sell = 0.0

            # Kyle's lambda: accumulate (signed_volume, price_change vs prev mid)
            if self._recent_flows:
                prev_mid = self._recent_flows[-1][2]
                price_change = trade.price - prev_mid
            else:
                price_change = 0.0
            self._recent_flows.append((signed_vol, price_change, mid))

            self._recent_trades.append(
                _TradeWithMid(trade.timestamp_ms, trade.price, trade.size, trade.side, mid)
            )

        vpin = self._compute_vpin()
        kyle_lambda = self._compute_kyle_lambda()
        effective_spread = self._compute_effective_spread()
        realized_spread = self._compute_realized_spread()

        return {
            "f3_vpin": vpin,
            "f3_kyle_lambda": kyle_lambda,
            "f3_effective_spread_bps": effective_spread,
            "f3_realized_spread_bps": realized_spread,
        }

    def _compute_vpin(self) -> float:
        """VPIN = mean(|buy_vol - sell_vol| / (buy_vol + sell_vol)) over recent buckets."""
        if not self._buckets:
            return 0.0
        return sum(self._buckets) / len(self._buckets)

    def _compute_kyle_lambda(self) -> float:
        """Kyle's lambda = price impact per unit volume.

        Simple OLS regression: price_change = lambda * signed_volume + epsilon
        """
        if len(self._recent_flows) < 20:
            return 0.0
        import numpy as np
        flows = list(self._recent_flows)
        x = np.array([f[0] for f in flows])
        y = np.array([f[1] for f in flows])
        if x.std() == 0:
            return 0.0
        # lambda = cov(x, y) / var(x)
        cov = np.cov(x, y, ddof=0)[0, 1]
        var = np.var(x, ddof=0)
        if var == 0:
            return 0.0
        return float(cov / var)

    def _compute_effective_spread(self) -> float:
        """Effective spread in bps: 2 * (exec_price - mid) signed by side."""
        if not self._recent_trades:
            return 0.0
        # Convert deque to list for safe slicing (deque doesn't support negative slices)
        recent = list(self._recent_trades)
        spreads = []
        for t in recent[-50:]:
            if t.mid == 0:
                continue
            if t.side == "buy":
                es = 2 * (t.price - t.mid) / t.mid * 10_000
            else:
                es = 2 * (t.mid - t.price) / t.mid * 10_000
            spreads.append(es)
        if not spreads:
            return 0.0
        return sum(spreads) / len(spreads)

    def _compute_realized_spread(self) -> float:
        """Realized spread (5-min ahead, simplified as 0 in MVP)."""
        # Full implementation would track future mid-price 5 min ahead.
        # For MVP, we return 0 — this is a placeholder.
        return 0.0

    @staticmethod
    def _empty() -> dict:
        return {
            "f3_vpin": 0.0,
            "f3_kyle_lambda": 0.0,
            "f3_effective_spread_bps": 0.0,
            "f3_realized_spread_bps": 0.0,
        }
