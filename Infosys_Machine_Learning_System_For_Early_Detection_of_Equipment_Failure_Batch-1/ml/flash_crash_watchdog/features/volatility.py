"""Feature Family F4 - Volatility (3 features).

Computed per tick from rolling mid-price history.
Latency budget: ~0.2 ms.

Features:
    15. realized_volatility_1s  - sqrt(sum(r_t^2)) over last 1s
    16. variance_ratio          - var(long_horizon) / (k * var(short_horizon))
                                  Lo-MacKinlay variance ratio test
    17. garman_klass           - GK volatility estimator using H/L prices

FIXED version - snapshots the deque before iterating to prevent
"deque mutated during iteration" errors, AND properly defines ts_now
in _realized_vol (was missing in a previous partial fix).
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

from flash_crash_watchdog.tick import Tick


@dataclass
class _MidSample:
    timestamp_ms: int
    mid: float


class VolatilityFeatures:
    """Maintains rolling mid-price history for F4 features."""

    def __init__(self, short_window_ms: int = 100, long_window_ms: int = 1000) -> None:
        self._short_window = short_window_ms
        self._long_window = long_window_ms
        self._history: Deque[_MidSample] = deque(maxlen=10_000)

    def update(self, tick: Tick) -> dict:
        try:
            mid = tick.book.mid_price
            if mid is None or mid <= 0:
                return self._empty()

            ts = tick.timestamp_ms
            self._history.append(_MidSample(ts, mid))

            # Evict old samples
            cutoff = ts - self._long_window * 2
            while self._history and self._history[0].timestamp_ms < cutoff:
                self._history.popleft()

            rv = self._realized_vol(self._long_window)
            vr = self._variance_ratio(self._short_window, self._long_window)
            gk = self._garman_klass(tick)

            return {
                "f4_realized_vol_1s": rv,
                "f4_variance_ratio": vr,
                "f4_garman_klass": gk,
            }
        except Exception:
            # Defensive: never let F4 break the whole feature extraction
            return self._empty()

    def _realized_vol(self, window_ms: int) -> float:
        """Annualized realized volatility over a window."""
        if len(self._history) < 2:
            return 0.0
        # Snapshot the deque to avoid "deque mutated during iteration"
        history_snapshot = list(self._history)
        ts_now = history_snapshot[-1].timestamp_ms
        samples = [s for s in history_snapshot if s.timestamp_ms >= ts_now - window_ms]
        if len(samples) < 2:
            return 0.0
        returns = [
            math.log(samples[i].mid / samples[i - 1].mid)
            for i in range(1, len(samples))
            if samples[i].mid > 0 and samples[i - 1].mid > 0
        ]
        if not returns:
            return 0.0
        var = sum(r * r for r in returns) / len(returns)
        return math.sqrt(var)

    def _variance_ratio(self, short_ms: int, long_ms: int) -> float:
        """Lo-MacKinlay variance ratio: VR(k) = var(long) / (k * var(short)).

        VR = 1 under random walk. VR > 1 = trending. VR < 1 = mean-reverting.
        A sudden drop in VR often precedes flash crashes (correlation breakdown).
        """
        if len(self._history) < 10:
            return 1.0
        # Snapshot the deque to avoid "deque mutated during iteration"
        history_snapshot = list(self._history)
        ts_now = history_snapshot[-1].timestamp_ms
        short_samples = [s for s in history_snapshot if s.timestamp_ms >= ts_now - short_ms]
        long_samples = [s for s in history_snapshot if s.timestamp_ms >= ts_now - long_ms]
        if len(short_samples) < 2 or len(long_samples) < 4:
            return 1.0

        short_rets = [
            math.log(short_samples[i].mid / short_samples[i - 1].mid)
            for i in range(1, len(short_samples))
            if short_samples[i].mid > 0 and short_samples[i - 1].mid > 0
        ]
        long_rets = [
            math.log(long_samples[i].mid / long_samples[i - 1].mid)
            for i in range(1, len(long_samples))
            if long_samples[i].mid > 0 and long_samples[i - 1].mid > 0
        ]
        if not short_rets or not long_rets:
            return 1.0
        var_short = sum(r * r for r in short_rets) / max(1, len(short_rets))
        var_long = sum(r * r for r in long_rets) / max(1, len(long_rets))
        if var_short == 0:
            return 1.0
        k = long_ms / short_ms
        return var_long / (k * var_short)

    def _garman_klass(self, tick: Tick) -> float:
        """Garman-Klass volatility estimator: 0.5 * (ln(H/L))^2 - (2ln2-1) * (ln(C/O))^2.

        Simplified for LOB: use best_bid as High, best_ask as Low, mid as Close.
        """
        book = tick.book
        if not book.bids or not book.asks or book.mid_price is None or book.mid_price <= 0:
            return 0.0
        h = book.best_bid
        l = book.best_ask
        c = book.mid_price
        if h is None or l is None or h <= 0 or l <= 0:
            return 0.0
        try:
            term1 = 0.5 * (math.log(h / l)) ** 2
            term2 = (2 * math.log(2) - 1) * (math.log(c / c)) ** 2  # C/O term is 0 for snapshot
            return math.sqrt(max(0, term1 - term2))
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def _empty() -> dict:
        return {
            "f4_realized_vol_1s": 0.0,
            "f4_variance_ratio": 1.0,
            "f4_garman_klass": 0.0,
        }
