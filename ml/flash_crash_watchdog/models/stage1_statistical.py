"""Stage 1 — Statistical Pre-Filter.

Cheap statistical tests on every tick. Rejects obviously normal ticks.
Latency: < 0.1 ms. Pass-through: ~5%.

Tests:
    1. Micro-price velocity z-score over 50ms
    2. Spread z-score (vs rolling baseline)
    3. OBI z-score
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Deque

from flash_crash_watchdog.tick import Tick


@dataclass
class Stage1Config:
    velocity_z_threshold: float = 3.0
    spread_z_threshold: float = 3.0
    obi_z_threshold: float = 2.5
    baseline_window: int = 1000  # ticks for rolling baseline


class Stage1Statistical:
    """Cheap statistical pre-filter. Runs on every tick."""

    def __init__(self, config: Stage1Config | None = None) -> None:
        self.config = config or Stage1Config()
        self._vel_history: Deque[float] = deque(maxlen=self.config.baseline_window)
        self._spread_history: Deque[float] = deque(maxlen=self.config.baseline_window)
        self._obi_history: Deque[float] = deque(maxlen=self.config.baseline_window)
        self._ticks_processed = 0
        self._ticks_passed = 0

    def score(self, tick: Tick) -> tuple[float, bool]:
        """Returns (anomaly_score, should_pass_to_stage2).

        anomaly_score in [0, 1]: 0 = normal, 1 = highly anomalous.
        """
        features = tick.features
        vel = features.get("f1_mid_velocity_50ms", 0.0)
        spread = features.get("f2_obi_10", 0.0)  # placeholder; use spread_bps from book
        obi = features.get("f2_obi_10", 0.0)

        # Use spread_bps from book if available
        spread_bps = tick.book.spread_bps if tick.book.spread_bps is not None else 0.0

        self._vel_history.append(vel)
        self._spread_history.append(spread_bps)
        self._obi_history.append(obi)
        self._ticks_processed += 1

        # Need enough history for a baseline
        if len(self._vel_history) < 100:
            return 0.0, False  # warmup; pass nothing

        vel_z = self._z_score(vel, list(self._vel_history))
        spread_z = self._z_score(spread_bps, list(self._spread_history))
        obi_z = self._z_score(obi, list(self._obi_history))

        # Composite anomaly score
        score = max(
            abs(vel_z) / self.config.velocity_z_threshold,
            abs(spread_z) / self.config.spread_z_threshold,
            abs(obi_z) / self.config.obi_z_threshold,
        )
        score = min(1.0, score)

        should_pass = score >= 1.0
        if should_pass:
            self._ticks_passed += 1

        return score, should_pass

    @property
    def pass_through_rate(self) -> float:
        if self._ticks_processed == 0:
            return 0.0
        return self._ticks_passed / self._ticks_processed

    @staticmethod
    def _z_score(value: float, history: list[float]) -> float:
        if len(history) < 2:
            return 0.0
        mean = sum(history) / len(history)
        var = sum((x - mean) ** 2 for x in history) / len(history)
        std = math.sqrt(var)
        if std == 0:
            return 0.0
        return (value - mean) / std
