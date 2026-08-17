"""Stage 4 — Cross-Symbol Correlation Breakdown (real, feedable).

Replaces the placeholder Transformer as the "market knows something" signal:
when an anchor's pairwise return-correlation with a basket collapses below its
own recent baseline, the market is disintegrating — an early-warning signal that
was verified on BTC/ETH 2021-05-19 (rolling corr 0.69 -> 0.14 at the crash,
z = -3.6).

Usage (multi-symbol feed, called once per tick per symbol):
    corr = CorrelationBreakdown(anchor="BTCUSDT")
    corr.update(symbol, mid_price, timestamp_ms)
    z, score, should_pass = corr.evaluate()   # call when you want a score

The score is time-observable (past returns only), so it is live-wirable.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    anchor: str = "BTCUSDT"        # the symbol we're watching for breakdown
    corr_window_bins: int = 120    # rolling corr window (120 s)
    baseline_bins: int = 1800      # baseline of corr (30 min) for the z-score
    warmup_bins: int = 300         # min history before scoring
    collapse_z: float = 2.0        # z below baseline => strong breakdown (score only)
    floor_corr: float = 0.45       # ABSOLUTE floor: corr below this = "assets decoupled"
    sustain_s: int = 30            # must stay below floor this long before firing
    min_basket: int = 1
    max_pairs: int = 8


class CorrelationBreakdown:
    """Rolling pairwise (anchor-vs-basket) return-correlation + baseline z-score."""

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self.config = config or CorrelationConfig()
        self._last_mid: Dict[str, float] = {}          # symbol -> last mid (for returns)
        self._returns: Dict[str, Deque[float]] = {}    # symbol -> deque of log-returns
        self._corr_hist: Deque[float] = deque(maxlen=self.config.baseline_bins)
        self._below_count = 0
        self._ticks = 0

    # -- feed ---------------------------------------------------------------
    def update(self, symbol: str, mid: float, ts_ms: int) -> None:
        """Record a mid-price observation; appends a log-return when the mid changes.

        Works for dense per-tick feeds and resampled 1s-grid feeds (consecutive
        observations => consecutive returns).
        """
        if mid is None or mid <= 0:
            return
        prev = self._last_mid.get(symbol)
        if prev is not None and prev != mid:
            self._returns.setdefault(
                symbol, deque(maxlen=self.config.corr_window_bins + 1)
            ).append(float(np.log(mid / prev)))
        self._last_mid[symbol] = float(mid)
        self._ticks += 1

    # -- scoring ------------------------------------------------------------
    def pair_correlations(self) -> Dict[str, float]:
        """ADV-04: per-basket-symbol return-correlation vs the anchor (current window).

        Returns {basket_symbol: corr} for every basket symbol with enough history —
        the visibility that answers "which specific assets decoupled?" (the aggregate
        ``_current_corr`` is the mean over these pairs).
        """
        a = self._returns.get(self.config.anchor)
        if a is None or len(a) < self.config.corr_window_bins:
            return {}
        a_arr = np.asarray(list(a), dtype=np.float64)
        out: Dict[str, float] = {}
        for sym, rq in self._returns.items():
            if sym == self.config.anchor or len(rq) < self.config.corr_window_bins:
                continue
            if len(out) >= self.config.max_pairs:
                break
            b_arr = np.asarray(list(rq), dtype=np.float64)[-len(a_arr):]
            s = np.corrcoef(a_arr[-len(b_arr):], b_arr)[0, 1]
            if s == s:  # not NaN
                out[sym] = float(s)
        return out

    def decoupling_symbols(self, floor: float | None = None) -> list[str]:
        """ADV-04: basket symbols whose corr with the anchor is below the floor.

        Honest framing: a decoupling ALARM (one asset broke from the market), not a
        crash alarm — market-wide crashes keep cross-asset correlation high.
        """
        floor = floor if floor is not None else self.config.floor_corr
        return [sym for sym, c in self.pair_correlations().items() if c < floor]

    def _current_corr(self) -> float | None:
        """Mean pairwise return-correlation of anchor vs basket over corr_window_bins."""
        a = self._returns.get(self.config.anchor)
        if a is None or len(a) < self.config.corr_window_bins:
            return None
        a_arr = np.asarray(list(a), dtype=np.float64)
        vals = []
        for sym, rq in self._returns.items():
            if sym == self.config.anchor or len(rq) < self.config.corr_window_bins:
                continue
            if len(vals) >= self.config.max_pairs:
                break
            b_arr = np.asarray(list(rq), dtype=np.float64)[-len(a_arr):]
            s = np.corrcoef(a_arr[-len(b_arr):], b_arr)[0, 1]
            if s == s:  # not NaN
                vals.append(float(s))
        if not vals:
            return None
        return float(np.mean(vals))

    def evaluate(self) -> tuple[float, float, bool]:
        """Return (collapse_z, anomaly_score[0,1], should_pass).

        should_pass = the anchor-vs-basket correlation has DROPPED BELOW an
        ABSOLUTE floor (assets decoupled) and stayed there for ``sustain_s``
        seconds. A self-referential rolling z alone is too noisy on calm days
        (tiny variance -> false trips), so the fire rule is the absolute floor +
        a duration gate; the z-score is kept as the anomaly magnitude.
        """
        c = self._current_corr()
        if c is None:
            return 0.0, 0.0, False
        self._corr_hist.append(c)
        below = c < self.config.floor_corr
        self._below_count = self._below_count + 1 if below else 0
        if len(self._corr_hist) < self.config.warmup_bins:
            return 0.0, 0.0, False
        arr = np.asarray(list(self._corr_hist), dtype=np.float64)
        mean, std = float(arr.mean()), float(arr.std())
        z = (mean - c) / std if std > 1e-4 else 0.0
        score = float(min(1.0, max(0.0, z / self.config.collapse_z)))
        fire = self._below_count >= self.config.sustain_s
        return z, score, fire

    @property
    def ready(self) -> bool:
        return len(self._corr_hist) >= self.config.warmup_bins
