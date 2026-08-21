"""Tests for the canonical alert-vs-event matcher (RSR-04)."""
import numpy as np

from flash_crash_watchdog.eval.metrics import match_alerts_to_crashes


def test_single_alert_hits_crash():
    m = match_alerts_to_crashes([150], [(100, 200)])
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 0, 0)
    assert m.median_ttd_ms == 50  # end - alert


def test_grace_before_start_is_a_tp():
    # alert at 98ms is within [start-5000, end] = [95000, 200000]... use ms scale
    m = match_alerts_to_crashes([98_000], [(100_000, 200_000)], grace_ms=5_000)
    assert m.true_positives == 1
    m0 = match_alerts_to_crashes([90_000], [(100_000, 200_000)], grace_ms=5_000)
    assert m0.true_positives == 0  # 90k < start - 5000


def test_burst_yields_one_tp_rest_fp():
    # two alerts in the same event: one TP (earliest), one FP
    m = match_alerts_to_crashes([150, 160], [(100, 200)])
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 1, 0)


def test_missed_event_is_false_negative():
    m = match_alerts_to_crashes([300], [(100, 200), (250, 400)])
    assert m.true_positives == 1  # 300 matches event 2
    assert m.false_negatives == 1  # event 1 missed
    assert m.false_positives == 0


def test_off_window_alert_is_false_positive():
    m = match_alerts_to_crashes([150, 1_000_000], [(100, 200)])
    assert (m.true_positives, m.false_positives, m.false_negatives) == (1, 1, 0)


def test_greedy_earliest_claims_event():
    # events overlap; earliest alert claims the first event it falls in
    m = match_alerts_to_crashes([150, 260], [(100, 300), (200, 400)])
    assert m.true_positives == 2
    assert m.false_negatives == 0
    assert m.median_ttd_ms == 145.0  # median of [150, 140]


def test_no_alerts_no_crashes():
    m = match_alerts_to_crashes([], [])
    assert (m.true_positives, m.false_positives, m.false_negatives) == (0, 0, 0)

def test_wilson_ci_reasonable():
    from flash_crash_watchdog.eval.metrics import wilson_ci
    # 1 success in 3 events: the Wilson CI is wide but within [0,1]
    lo, hi = wilson_ci(1, 3)
    assert 0.0 <= lo <= hi <= 1.0
    assert lo < 0.9 and hi > 0.05  # honest wide interval for n=3
    # 9/9 successes: CI hugs 1 but is not exactly [1,1]
    lo2, hi2 = wilson_ci(9, 9)
    assert lo2 > 0.6 and hi2 <= 1.0
    # n=0 -> (0,0)
    assert wilson_ci(0, 0) == (0.0, 0.0)
