"""Offline backtest — replay historical ticks through the cascade."""
from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from flash_crash_watchdog.cascade import DetectionCascade
from flash_crash_watchdog.data.historical_loader import df_to_ticks
from flash_crash_watchdog.data.labels import label_crashes
from flash_crash_watchdog.models.stage3_tcn import normalize_z, TCNDetector
from flash_crash_watchdog.models.stage5_bayesian import Alert

logger = logging.getLogger(__name__)

# Batched Stage-3 scoring defaults (shared by the RSR-15 harness and scripts).
WINDOW = 200        # trained TCN window length
NORM_WINDOW = 500   # rolling-z window (matches training / Stage3TCN.feed)
CHUNK = 4096        # batched forward chunk


def stage3_scores_batched(
    model: TCNDetector,
    F: np.ndarray,
    dev: str,
    window: int = WINDOW,
    norm_window: int = NORM_WINDOW,
    chunk: int = CHUNK,
) -> np.ndarray:
    """Every stride-1 ``window``-window Stage-3 score in one batched GPU pass.

    The canonical batched scorer (BUG-03 / RSR-15): applies the shared
    ``normalize_z`` transform, then computes scores[j] for the normalized window
    ``norm[j : j+window]`` — i.e. scores[j] is the model output at tick
    ``j + window - 1`` and uses ONLY data up to that tick (causal). This is what
    ``scripts/run_backtest_batched.py`` and ``scripts/run_validation.py`` use.
    """
    if len(F) <= window:
        return np.zeros(0, dtype=np.float32)
    norm = normalize_z(F, norm_window)
    n = len(norm) - window + 1
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(s + chunk, n)
        win = np.stack([norm[i:i + window] for i in range(s, e)])
        x = torch.from_numpy(np.ascontiguousarray(win)).permute(0, 2, 1).float().to(dev)
        with torch.no_grad():
            out[s:e] = model(x)[:, -1].cpu().numpy()
    return out


@dataclass
class BacktestResults:
    """Results of an offline backtest."""
    total_ticks: int = 0
    alerts_fired: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    ttd_ms: List[int] = field(default_factory=list)
    cascade_stats: dict = field(default_factory=dict)
    alerts: List[dict] = field(default_factory=list)

    @property
    def precision(self) -> float:
        pp = self.true_positives + self.false_positives
        return self.true_positives / pp if pp else 0.0

    @property
    def recall(self) -> float:
        ap = self.true_positives + self.false_negatives
        return self.true_positives / ap if ap else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) > 0 else 0.0

    @property
    def median_ttd_ms(self) -> float:
        return statistics.median(self.ttd_ms) if self.ttd_ms else 0.0

    def print_summary(self) -> None:
        logger.info("=" * 60)
        logger.info("BACKTEST RESULTS")
        logger.info("=" * 60)
        logger.info("Ticks processed:        %d", self.total_ticks)
        logger.info("Alerts fired:           %d", self.alerts_fired)
        logger.info("True positives:         %d", self.true_positives)
        logger.info("False positives:        %d", self.false_positives)
        logger.info("False negatives:        %d", self.false_negatives)
        logger.info("Precision:              %.3f", self.precision)
        logger.info("Recall:                 %.3f", self.recall)
        logger.info("F1:                     %.3f", self.f1)
        logger.info("Median TTD:             %.1f ms", self.median_ttd_ms)
        logger.info("=" * 60)

    def save(self, path: str | Path) -> None:
        with open(path, "w") as f:
            json.dump({
                "total_ticks": self.total_ticks,
                "alerts_fired": self.alerts_fired,
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "precision": self.precision,
                "recall": self.recall,
                "f1": self.f1,
                "median_ttd_ms": self.median_ttd_ms,
                "ttd_ms": self.ttd_ms,
                "alerts": self.alerts,
            }, f, indent=2)
        logger.info("Saved results to %s", path)


def run_backtest(
    cascade: DetectionCascade,
    df: pd.DataFrame,
    window_ms: int = 500,
    crash_threshold_pct: float = 2.0,
    cooldown_ms: int = 0,
) -> BacktestResults:
    """Run the cascade on a historical DataFrame.

    ``cooldown_ms`` coalesces alert bursts: at most one alert per cooldown
    interval is kept (report-wise), so a crash event that fires dozens of
    adjacent per-tick alerts is reported once. This gives a realistic
    "alerts per hour" and precision rather than counting every passing tick.
    """
    results = BacktestResults()
    ticks = list(df_to_ticks(df))
    results.total_ticks = len(ticks)

    crashes = label_crashes(ticks, drop_threshold_pct=crash_threshold_pct)
    logger.info("Found %d ground-truth crash windows", len(crashes))

    alerts: List[Alert] = []
    last_ts: int | None = None
    for tick in ticks:
        alert = cascade.process_tick(tick)
        if alert is not None:
            if cooldown_ms > 0 and last_ts is not None and alert.timestamp_ms - last_ts < cooldown_ms:
                continue
            last_ts = alert.timestamp_ms
            alerts.append(alert)
            results.alerts.append({
                "timestamp_ms": alert.timestamp_ms,
                "symbol": alert.symbol,
                "posterior": alert.posterior,
            })

    results.alerts_fired = len(alerts)

    for crash in crashes:
        earliest = None
        for alert in alerts:
            if crash.start_ts <= alert.timestamp_ms <= crash.end_ts:
                if earliest is None or alert.timestamp_ms < earliest:
                    earliest = alert.timestamp_ms
        if earliest is not None:
            results.true_positives += 1
            results.ttd_ms.append(crash.end_ts - earliest)
        else:
            results.false_negatives += 1

    matched = 0
    for alert in alerts:
        for crash in crashes:
            if crash.start_ts <= alert.timestamp_ms <= crash.end_ts:
                matched += 1
                break
    results.false_positives = len(alerts) - matched

    results.cascade_stats = {
        "ticks_total": cascade.stats.ticks_total,
        "stage1_passed": cascade.stats.stage1_passed,
        "stage2_passed": cascade.stats.stage2_passed,
        "stage3_passed": cascade.stats.stage3_passed,
        "stage4_passed": cascade.stats.stage4_passed,
        "alerts_fired": cascade.stats.alerts_fired,
        "avg_latency_ms": cascade.stats.total_latency_ms / max(1, cascade.stats.ticks_total),
    }

    cascade.print_stats()
    return results
