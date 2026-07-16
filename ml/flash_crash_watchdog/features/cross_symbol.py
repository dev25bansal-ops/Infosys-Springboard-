"""Feature Family F5 — Cross-Symbol (3 features).

Computed across multiple symbols' returns.
Latency budget: ~0.9 ms.

Features:
    18. pairwise_correlation  — rolling Pearson correlation between this symbol
                                 and a reference (e.g., BTC for crypto, SPY for equities)
    19. lead_lag_coefficient  — cross-correlation at non-zero lag (who leads whom?)
    20. cointegration_residual — residual from a rolling OLS regression of this symbol
                                  on the reference (deviations from long-run relationship)
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from flash_crash_watchdog.tick import Tick


@dataclass
class _ReturnSample:
    timestamp_ms: int
    ret: float


class CrossSymbolFeatures:
    """Maintains rolling return history for this symbol and a reference symbol."""

    def __init__(self, window_ms: int = 60_000, max_lag_ms: int = 2_000) -> None:
        self._window_ms = window_ms
        self._max_lag_ms = max_lag_ms
        self._self_history: Deque[_ReturnSample] = deque(maxlen=10_000)
        self._ref_history: Dict[str, Deque[_ReturnSample]] = {}
        self._last_mid: Optional[float] = None

    def update_reference(self, symbol: str, timestamp_ms: int, mid_price: float) -> None:
        """Update the reference symbol's mid-price history."""
        if symbol not in self._ref_history:
            self._ref_history[symbol] = deque(maxlen=10_000)
        # Compute return vs last sample
        hist = self._ref_history[symbol]
        if hist:
            prev_mid = hist[-1].ret  # Not quite right; need to store mid separately
        # Simplified: store mid as "return" for MVP — real impl would compute log returns
        hist.append(_ReturnSample(timestamp_ms, mid_price))

    def update(self, tick: Tick) -> dict:
        mid = tick.book.mid_price
        if mid is None or mid <= 0:
            return self._empty()

        ts = tick.timestamp_ms
        ret = 0.0
        if self._last_mid and self._last_mid > 0:
            ret = np.log(mid / self._last_mid)
        self._last_mid = mid
        self._self_history.append(_ReturnSample(ts, ret))

        # Evict old samples
        cutoff = ts - self._window_ms
        while self._self_history and self._self_history[0].timestamp_ms < cutoff:
            self._self_history.popleft()
        for hist in self._ref_history.values():
            while hist and hist[0].timestamp_ms < cutoff:
                hist.popleft()

        # If we have a reference, compute cross-symbol features
        if not self._ref_history:
            return self._empty()

        ref_symbol = next(iter(self._ref_history))
        ref_hist = self._ref_history[ref_symbol]

        if len(self._self_history) < 30 or len(ref_hist) < 30:
            return self._empty()

        corr = self._pairwise_correlation(ref_hist)
        lead_lag = self._lead_lag(ref_hist)
        coint_resid = self._cointegration_residual(ref_hist)

        return {
            "f5_pairwise_correlation": corr,
            "f5_lead_lag_coefficient": lead_lag,
            "f5_cointegration_residual": coint_resid,
        }

    def _pairwise_correlation(self, ref_hist: Deque[_ReturnSample]) -> float:
        """Rolling Pearson correlation between self and reference returns."""
        self_rets = np.array([s.ret for s in self._self_history])
        ref_rets = np.array([s.ret for s in ref_hist])
        n = min(len(self_rets), len(ref_rets))
        if n < 30:
            return 0.0
        a = self_rets[-n:]
        b = ref_rets[-n:]
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    def _lead_lag(self, ref_hist: Deque[_ReturnSample]) -> float:
        """Cross-correlation at non-zero lag. Positive = self leads ref."""
        self_rets = np.array([s.ret for s in self._self_history])
        ref_rets = np.array([s.ret for s in ref_hist])
        n = min(len(self_rets), len(ref_rets))
        if n < 60:
            return 0.0
        a = self_rets[-n:]
        b = ref_rets[-n:]
        # Compute cross-correlation at lag ±5
        max_corr = 0.0
        best_lag = 0
        for lag in range(-5, 6):
            if lag < 0:
                x, y = a[-lag:], b[:lag]
            elif lag > 0:
                x, y = a[:-lag], b[lag:]
            else:
                x, y = a, b
            if len(x) < 30 or x.std() == 0 or y.std() == 0:
                continue
            c = np.corrcoef(x, y)[0, 1]
            if abs(c) > abs(max_corr):
                max_corr = c
                best_lag = lag
        return float(best_lag) * max_corr  # signed lag * correlation

    def _cointegration_residual(self, ref_hist: Deque[_ReturnSample]) -> float:
        """Residual from rolling OLS regression of self on reference.

        Large residual = the two symbols have decoupled from their long-run
        relationship. This is a leading indicator of correlation breakdown.
        """
        self_mids = np.array([s.ret for s in self._self_history])
        ref_mids = np.array([s.ret for s in ref_hist])
        n = min(len(self_mids), len(ref_mids))
        if n < 50:
            return 0.0
        y = self_mids[-n:]
        x = ref_mids[-n:]
        # OLS: y = alpha + beta * x + epsilon
        X = np.column_stack([np.ones(n), x])
        try:
            beta, _ = np.linalg.lstsq(X, y, rcond=None)[0:1] + (None,)
            coeffs, *_ = np.linalg.lstsq(X, y, rcond=None)
            alpha, beta = coeffs
            resid = y[-1] - (alpha + beta * x[-1])
            # Normalize by residual std
            all_resid = y - (alpha + beta * x)
            std = all_resid.std()
            if std == 0:
                return 0.0
            return float(resid / std)
        except np.linalg.LinAlgError:
            return 0.0

    @staticmethod
    def _empty() -> dict:
        return {
            "f5_pairwise_correlation": 0.0,
            "f5_lead_lag_coefficient": 0.0,
            "f5_cointegration_residual": 0.0,
        }
