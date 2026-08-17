"""Shared evaluation metrics for the crash-detection cascade.

This module is the single source of truth for precision/recall/F1 and lead-time
(ttd) so the various backtest/eval paths (eval/backtest, run_backtest_batched,
run_cascade_backtest, generate_plots) don't each re-implement the math.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Iterable, List


@dataclass
class Metrics:
    """Aggregate detection metrics."""

    alerts: int = 0
    true_positives: int = 0
    false_positives: int = 0
    false_negatives: int = 0
    total_ticks: int = 0
    ttd_ms: List[int] = field(default_factory=list)

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

    @property
    def alert_rate_pct(self) -> float:
        return 100.0 * self.alerts / max(1, self.total_ticks) if self.total_ticks else 0.0

    def as_dict(self) -> dict:
        return {
            "alerts_fired": self.alerts,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "median_ttd_ms": round(self.median_ttd_ms, 1),
            "alert_rate_pct": round(self.alert_rate_pct, 4),
        }


def compute_metrics(
    total_ticks: int,
    crashes: int,
    matched: Iterable[int],
    alert_ts_total: int,
    alert_tp: int,
    ttd: Iterable[int] = (),
) -> Metrics:
    """Build Metrics from raw counts (matches eval/backtest's crash-window matching)."""
    m = Metrics(
        alerts=alert_ts_total,
        true_positives=alert_tp,
        false_positives=max(0, alert_ts_total - alert_tp),
        false_negatives=crashes - len(set(matched)),
        total_ticks=total_ticks,
        ttd_ms=list(ttd),
    )
    return m


def precision_recall_f1(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    """Direct precision/recall/F1 from counts (no extra state)."""
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Two-sided Wilson score interval (RSR-12).

    For small event counts (n=2-9 crash events/day) a naive
    p_hat ± z*sqrt(pq/n) is unreliable (can exceed [0,1] and undercovers).
    The Wilson interval is the honest default for proportions with small n.
    Returns (lower, upper) bounds for a z=1.96 (95%) interval.
    """
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / denom
    lo = max(0.0, centre - half)
    hi = min(1.0, centre + half)
    return (lo, hi)


def match_alerts_to_crashes(
    alert_ts: Iterable[int],
    crash_ranges: Iterable[tuple[int, int]],
    grace_ms: int = 5_000,
) -> Metrics:
    """Match alerts to crash events using the canonical RSR-04 protocol.

    THE single matching convention every offline backtest must use (replaces four
    divergent per-script implementations):

        alert matches an event  <=>  start - grace_ms <= alert_ts <= end

    Alerts are processed chronologically. An alert is a true positive iff it is
    the *earliest* alert for a crash event it falls within; each event claims
    exactly one (the earliest) alert, so an alert burst yields one TP per event.
    Every other alert is a false positive. Events claimed by no alert are
    false negatives.

    Args:
        alert_ts: alert timestamps (ms), not necessarily sorted.
        crash_ranges: iterable of (start_ts, end_ts), e.g. from LabelCrashList.
        grace_ms: an alert up to this far before a crash's start still counts as
            a true positive (event window = [start - grace_ms, end]).

    Returns:
        Metrics with alerts/tp/fp/fn/ttd_ms populated (total_ticks left 0).
    """
    alerts = sorted(alert_ts)
    ranges = list(crash_ranges)
    matched = [False] * len(ranges)
    tp = 0
    ttd: List[int] = []

    for ts in alerts:
        claimed = -1
        for ci, (start, end) in enumerate(ranges):
            if not matched[ci] and (start - grace_ms) <= ts <= end:
                claimed = ci
                break
        if claimed >= 0:
            matched[claimed] = True
            tp += 1
            ttd.append(ranges[claimed][1] - ts)

    fn = len(ranges) - int(sum(matched))
    return Metrics(
        alerts=len(alerts),
        true_positives=tp,
        false_positives=len(alerts) - tp,
        false_negatives=fn,
        ttd_ms=ttd,
    )


def ranges_from_crashlabels(crashes) -> list[tuple[int, int]]:
    """Convert label_crashes output (CrashLabel list) to (start, end) ranges."""
    return [(c.start_ts, c.end_ts) for c in crashes]