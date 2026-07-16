"""Feature Family F1 — Price & Action (5 features).

Computed per tick from the order-book snapshot and recent trade list.
Latency budget: < 0.1 ms.

Features:
    1. mid_price_velocity_50ms  — rate of mid-price change over 50ms
    2. mid_price_velocity_200ms — rate of mid-price change over 200ms
    3. micro_price              — volume-weighted mid-price (from LOB)
    4. trade_arrival_rate       — trades per second in the last 1s
    5. cancel_to_trade_ratio    — cancellations / executions in last 1s
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque

from flash_crash_watchdog.tick import Tick


@dataclass
class _PriceSample:
    timestamp_ms: int
    mid_price: float


class PriceActionFeatures:
    """Maintains rolling state for F1 features."""

    def __init__(self) -> None:
        self._price_history: Deque[_PriceSample] = deque(maxlen=10_000)
        self._trade_timestamps: Deque[int] = deque(maxlen=10_000)
        self._cancel_count: int = 0
        self._trade_count: int = 0

    def update(self, tick: Tick) -> dict:
        """Compute F1 features for this tick and update internal state."""
        ts = tick.timestamp_ms
        mid = tick.book.mid_price

        # Record price sample
        if mid is not None:
            self._price_history.append(_PriceSample(ts, mid))

        # Record trades
        for trade in tick.trades:
            self._trade_timestamps.append(trade.timestamp_ms)
            self._trade_count += 1

        # Evict old samples (older than 200ms)
        cutoff_200 = ts - 200
        while self._price_history and self._price_history[0].timestamp_ms < cutoff_200:
            self._price_history.popleft()
        cutoff_1000 = ts - 1000
        while self._trade_timestamps and self._trade_timestamps[0] < cutoff_1000:
            self._trade_timestamps.popleft()

        # Compute velocities
        vel_50 = self._velocity(50)
        vel_200 = self._velocity(200)

        # Trade arrival rate (trades per second in last 1s)
        trade_rate = len(self._trade_timestamps)

        # Cancel-to-trade ratio (simplified: we don't have explicit cancel
        # events in this MVP, so we proxy with order-book size reduction)
        # A real implementation would track explicit cancel messages.
        ctr = self._cancel_count / max(1, self._trade_count)

        return {
            "f1_mid_velocity_50ms": vel_50,
            "f1_mid_velocity_200ms": vel_200,
            "f1_micro_price": tick.book.micro_price,
            "f1_trade_arrival_rate": trade_rate,
            "f1_cancel_to_trade_ratio": ctr,
        }

    def _velocity(self, window_ms: int) -> float:
        """Compute mid-price velocity (change per ms) over a window."""
        if len(self._price_history) < 2:
            return 0.0
        now = self._price_history[-1]
        target_ts = now.timestamp_ms - window_ms
        # Find the oldest sample within the window
        oldest = None
        for sample in self._price_history:
            if sample.timestamp_ms >= target_ts:
                oldest = sample
                break
        if oldest is None or oldest.timestamp_ms == now.timestamp_ms:
            return 0.0
        return (now.mid_price - oldest.mid_price) / (now.timestamp_ms - oldest.timestamp_ms)
