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
        corr=None,  # optional CorrelationBreakdown (Stage-4 real multi-symbol signal)
    ) -> None:
        self.feature_extractor = feature_extractor
        self.s1 = stage1
        self.s2 = stage2
        self.s3 = stage3
        self.s4 = stage4
        self.s5 = stage5
        self.corr = corr
        self.stats = CascadeStats()
        self._alert_callback = None
        # Stage 4 needs a multi-symbol feed that the current data paths don't
        # provide; when disabled it is skipped (neutral 0.5) so single-symbol
        # pipelines can still alert via Stage-3.
        self._stage4_enabled = True

    @classmethod
    def from_config(cls, config_path: str | Path, models_dir: str | Path | None = None) -> "DetectionCascade":
        """Load cascade from a YAML config file.

        ``models_dir``: if given, load the trained Stage-2 (stage2_isolation_forest
        .joblib) and Stage-3 (stage3_tcn_prod.pt) checkpoints when present, so the
        live/backtest path uses the trained models instead of random/untrained ones.
        """
        with open(config_path) as f:
            config = yaml.safe_load(f)
        cascade = cls._from_dict(config)
        if models_dir is not None:
            cascade._load_models(models_dir)
        return cascade

    def _load_models(self, models_dir: str | Path) -> None:
        """Load trained Stage-2 + Stage-3 checkpoints from a models directory."""
        from pathlib import Path

        md = Path(models_dir)
        s2p = md / "stage2_isolation_forest.joblib"
        s3p = md / "stage3_tcn_prod.pt"
        if s2p.exists():
            try:
                self.s2.load(s2p)
                logger.info("Loaded trained Stage 2 from %s", s2p)
            except Exception as e:
                logger.warning("Failed to load Stage 2 %s: %s", s2p, e)
        if s3p.exists():
            try:
                self.s3.load(s3p)
                logger.info("Loaded trained Stage 3 from %s", s3p)
            except Exception as e:
                logger.warning("Failed to load Stage 3 %s: %s", s3p, e)

    @classmethod
    def _from_dict(cls, config: dict) -> "DetectionCascade":
        from flash_crash_watchdog.models.stage1_statistical import Stage1Config
        from flash_crash_watchdog.models.stage5_bayesian import Stage5Config

        stage4_cfg = dict(config.get("stage4", {}))
        stage4_enabled = bool(stage4_cfg.pop("enabled", True))

        cascade = cls(
            feature_extractor=FeatureExtractor(),
            stage1=Stage1Statistical(Stage1Config(**config.get("stage1", {}))),
            stage2=Stage2IsolationForest(**config.get("stage2", {})),
            stage3=Stage3TCN(TCNConfig(**config.get("stage3", {}))),
            stage4=Stage4Transformer(TransformerConfig(**stage4_cfg)),
            stage5=Stage5Bayesian(Stage5Config(**config.get("stage5", {}))),
        )
        # Allow pipeline.yml to disable the (currently unfed) Stage-4.
        cascade._stage4_enabled = stage4_enabled
        return cascade

    def on_alert(self, callback) -> None:
        """Register a callback fired when an alert is emitted."""
        self._alert_callback = callback

    def update_reference(self, symbol: str, mid_price: float, timestamp_ms: int) -> None:
        """ENH-03: feed a reference/basket symbol's mid to the correlation detector.

        The cascade's ``process_tick`` feeds the ANCHOR (tick symbol) into the
        configured ``corr``, but not the basket. Call this for every non-anchor
        symbol observation so the correlation breakdown has both sides. Safe
        no-op when no correlation module is configured.
        """
        if self.corr is not None:
            self.corr.update(symbol, mid_price, timestamp_ms)

    def process_tick(self, tick: Tick) -> Optional[Alert]:
        """Run a tick through the full cascade.

        Stage-3 (the TCN) is the crash classifier and is fed **and scored on every
        tick**, so no upstream gate can starve it of a contiguous window or of a
        scoring opportunity (a strict Stage-1/2 pre-filter was found to reject the
        very ticks the TCN needs, causing LUNA misses). Stage-1/Stage-2 remain
        advisory — their scores feed the Stage-5 fusion but never veto an alert.
        An alert fires only when Stage-3 passes its (per-asset calibrated)
        threshold AND the Stage-5 posterior crosses the alert threshold.
        """
        t0 = time.perf_counter()
        self.stats.ticks_total += 1

        # Extract features once per tick
        self.feature_extractor.extract(tick)

        # Advisory Stage-1 (statistical) + Stage-2 (isolation forest)
        s1_score, s1_pass = self.s1.score(tick)
        if s1_pass:
            self.stats.stage1_passed += 1
        s2_score, s2_pass = self.s2.score(tick)
        if s2_pass:
            self.stats.stage2_passed += 1

        # Stage 3 — TCN: feed + score on EVERY tick (contiguous window)
        self.s3.feed(tick)
        s3_score, s3_pass = self.s3.score_current()
        if not s3_pass:
            self._record_latency(t0)
            return None
        self.stats.stage3_passed += 1

        # Stage 4 — Cross-symbol Transformer / correlation breakdown.
        # When a correlation module is configured (multi-symbol feed), it is fed
        # every tick and its anomaly score becomes the real s4 (advisory, never a
        # veto). Otherwise fall back to the Transformer (if enabled) or neutral 0.5.
        if self.corr is not None:
            mp = tick.book.mid_price
            if mp is not None:
                self.corr.update(tick.symbol or "UNKNOWN", float(mp), tick.timestamp_ms)
            z4, s4_score, s4_fire = self.corr.evaluate()
            s4_pass = True  # advisory: correlation contributes to fusion, doesn't veto
        elif self._stage4_enabled:
            s4_score, s4_pass = self.s4.score(tick)
            if not s4_pass:
                self._record_latency(t0)
                return None
        else:
            s4_score, s4_pass = 0.5, True
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
