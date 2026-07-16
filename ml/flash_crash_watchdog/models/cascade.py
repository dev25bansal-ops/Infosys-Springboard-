"""5-stage hybrid detection cascade.

Orchestrates the 5 detector stages:
    Stage 1: Statistical pre-filter (every tick, < 0.1 ms)
    Stage 2: Isolation Forest (~1 ms, suspects only)
    Stage 3: TCN (~8 ms, deeper suspects)
    Stage 4: Cross-symbol Transformer (~15 ms)
    Stage 5: Bayesian aggregator (~1 ms, final decision)

Most ticks exit at Stage 1. The cascade keeps average per-tick cost low
while preserving deep-model sensitivity.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

from flash_crash_watchdog.features import FeatureExtractor
from flash_crash_watchdog.models.stage1_statistical import Stage1Statistical
from flash_crash_watchdog.models.stage2_isolation_forest import Stage2IsolationForest
from flash_crash_watchdog.models.stage3_tcn import Stage3TCN, TCNConfig
from flash_crash_watchdog.models.stage4_transformer import Stage4Transformer, TransformerConfig
from flash_crash_watchdog.models.stage5_bayesian import Alert, Stage5Bayesian
from flash_crash_watchdog.tick import Tick

logger = logging.getLogger(__name__)


@dataclass
class CascadeStats:
    ticks_total: int = 0
    stage1_passed: int = 0
    stage2_passed: int = 0
    stage3_passed: int = 0
    stage4_passed: int = 0
    alerts_fired: int = 0
    total_latency_ms: float = 0.0

    def summary(self) -> str:
        return (
            f"ticks={self.ticks_total}, "
            f"s1_pass={self.stage1_passed} ({self._rate(self.stage1_passed, self.ticks_total):.1%}), "
            f"s2_pass={self.stage2_passed}, s3_pass={self.stage3_passed}, s4_pass={self.stage4_passed}, "
            f"alerts={self.alerts_fired}, "
            f"avg_latency={self.total_latency_ms / max(1, self.ticks_total):.2f}ms"
        )

    @staticmethod
    def _rate(num: int, denom: int) -> float:
        return num / denom if denom else 0.0


class DetectionCascade:
    """The full 5-stage detection cascade."""

    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        stage1: Stage1Statistical,
        stage2: Stage2IsolationForest,
        stage3: Stage3TCN,
        stage4: Stage4Transformer,
        stage5: Stage5Bayesian,
    ) -> None:
        self.feature_extractor = feature_extractor
        self.s1 = stage1
        self.s2 = stage2
        self.s3 = stage3
        self.s4 = stage4
        self.s5 = stage5
        self.stats = CascadeStats()
        self._alert_callback = None

    @classmethod
    def from_config(cls, config_path: str | Path) -> "DetectionCascade":
        """Load cascade from a YAML config file."""
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return cls._from_dict(config)

    @classmethod
    def _from_dict(cls, config: dict) -> "DetectionCascade":
        from flash_crash_watchdog.models.stage1_statistical import Stage1Config
        from flash_crash_watchdog.models.stage5_bayesian import Stage5Config

        return cls(
            feature_extractor=FeatureExtractor(),
            stage1=Stage1Statistical(Stage1Config(**config.get("stage1", {}))),
            stage2=Stage2IsolationForest(**config.get("stage2", {})),
            stage3=Stage3TCN(TCNConfig(**config.get("stage3", {}))),
            stage4=Stage4Transformer(TransformerConfig(**config.get("stage4", {}))),
            stage5=Stage5Bayesian(Stage5Config(**config.get("stage5", {}))),
        )

    def on_alert(self, callback) -> None:
        """Register a callback fired when an alert is emitted."""
        self._alert_callback = callback

    def process_tick(self, tick: Tick) -> Optional[Alert]:
        """Run a tick through the full cascade.

        Returns an Alert if one was fired, else None.
        """
        t0 = time.perf_counter()
        self.stats.ticks_total += 1

        # Extract features
        self.feature_extractor.extract(tick)

        # Stage 1 — statistical pre-filter (every tick)
        s1_score, s1_pass = self.s1.score(tick)
        if not s1_pass:
            self._record_latency(t0)
            return None
        self.stats.stage1_passed += 1

        # Stage 2 — Isolation Forest
        s2_score, s2_pass = self.s2.score(tick)
        if not s2_pass:
            self._record_latency(t0)
            return None
        self.stats.stage2_passed += 1

        # Stage 3 — TCN
        s3_score, s3_pass = self.s3.score(tick)
        if not s3_pass:
            self._record_latency(t0)
            return None
        self.stats.stage3_passed += 1

        # Stage 4 — Cross-symbol Transformer
        s4_score, s4_pass = self.s4.score(tick)
        if not s4_pass:
            self._record_latency(t0)
            return None
        self.stats.stage4_passed += 1

        # Stage 5 — Bayesian aggregator
        alert = self.s5.aggregate(tick, s2_score, s3_score, s4_score)
        if alert is not None:
            self.stats.alerts_fired += 1
            logger.warning("ALERT: %s", alert)
            if self._alert_callback:
                self._alert_callback(alert)

        self._record_latency(t0)
        return alert

    def _record_latency(self, t0: float) -> None:
        latency_ms = (time.perf_counter() - t0) * 1000
        self.stats.total_latency_ms += latency_ms

    def print_stats(self) -> None:
        logger.info("Cascade stats: %s", self.stats.summary())
