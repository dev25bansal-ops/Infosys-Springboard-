"""Stage 5 — Bayesian Aggregator.

Fuses scores from Stages 2, 3, and 4 via Bayesian model averaging.
Produces a final alert decision.
Latency: ~1 ms.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Optional

from flash_crash_watchdog.tick import Tick

logger = logging.getLogger(__name__)


@dataclass
class Alert:
    """A flash-crash alert."""
    timestamp_ms: int
    symbol: str
    posterior: float  # final probability of anomaly
    stage2_score: float
    stage3_score: float
    stage4_score: float
    affected_symbols: list[str]
    features_snapshot: dict
    rationale: dict = field(default_factory=dict)  # ADV-01: which stages drove the decision

    def __repr__(self) -> str:
        return (
            f"Alert(ts={self.timestamp_ms}, symbol={self.symbol}, "
            f"posterior={self.posterior:.3f}, s2={self.stage2_score:.2f}, "
            f"s3={self.stage3_score:.2f}, s4={self.stage4_score:.2f})"
        )


@dataclass
class Stage5Config:
    alert_threshold: float = 0.7
    # Weights for Bayesian fusion (sum to 1.0)
    stage2_weight: float = 0.25
    stage3_weight: float = 0.45
    stage4_weight: float = 0.30
    # Per-stage confidence calibration
    stage2_confidence: float = 0.85
    stage3_confidence: float = 0.90
    stage4_confidence: float = 0.80


class Stage5Bayesian:
    """Bayesian model averaging aggregator."""

    def __init__(self, config: Stage5Config | None = None) -> None:
        self.config = config or Stage5Config()
        self._alerts_fired = 0

    def aggregate(
        self,
        tick: Tick,
        stage2_score: float,
        stage3_score: float,
        stage4_score: float,
    ) -> Optional[Alert]:
        """Fuse stage scores and decide whether to fire an alert.

        Uses a weighted geometric mean (log-odds fusion), which is the
        Bayesian-optimal combination when the stage scores are calibrated
        probabilities.
        """
        # Convert scores to log-odds (NaN/None -> neutral 0.5 so a bad score
        # can never silently suppress an alert)
        def to_log_odds(p: float) -> float:
            if p is None or p != p:
                p = 0.5
            p = max(1e-6, min(1 - 1e-6, p))
            return math.log(p / (1 - p))

        def from_log_odds(lo: float) -> float:
            return 1.0 / (1.0 + math.exp(-lo))

        w2 = self.config.stage2_weight * self.config.stage2_confidence
        w3 = self.config.stage3_weight * self.config.stage3_confidence
        w4 = self.config.stage4_weight * self.config.stage4_confidence
        total_w = w2 + w3 + w4
        w2, w3, w4 = w2 / total_w, w3 / total_w, w4 / total_w

        log_odds = (
            w2 * to_log_odds(stage2_score)
            + w3 * to_log_odds(stage3_score)
            + w4 * to_log_odds(stage4_score)
        )
        posterior = from_log_odds(log_odds)

        if posterior >= self.config.alert_threshold:
            self._alerts_fired += 1
            # ADV-01: lazy import — the eval package eagerly pulls eval.backtest, which
            # would otherwise create a load-time import cycle through the cascade.
            from flash_crash_watchdog.eval.explain import alert_rationale
            rationale = alert_rationale(stage2_score, stage3_score, stage4_score, self.config)
            return Alert(
                timestamp_ms=tick.timestamp_ms,
                symbol=tick.symbol,
                posterior=posterior,
                stage2_score=stage2_score,
                stage3_score=stage3_score,
                stage4_score=stage4_score,
                affected_symbols=[tick.symbol],
                features_snapshot=dict(tick.features),
                rationale=rationale,
            )
        return None

    @property
    def alerts_fired(self) -> int:
        return self._alerts_fired
