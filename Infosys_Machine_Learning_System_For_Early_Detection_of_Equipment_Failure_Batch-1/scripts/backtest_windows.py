#!/usr/bin/env python3
"""Backtest the trained TCN on real crash data using the window approach.

Slides the trained TCN over crash-day data and fires alerts when it
predicts "crash" with high confidence. Compares against a naive
threshold baseline.

Usage:
    python scripts/backtest_windows.py \
        --data data/parquet/BTCUSDT_2021-05-19.parquet \
        --model models/stage3_tcn_trained.pt \
        --output results/window_backtest.json
"""
import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.data.historical_loader import df_to_ticks, load_parquet
from flash_crash_watchdog.data.labels import label_crashes
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor
from flash_crash_watchdog.models.stage3_tcn import normalize_z, TCNDetector, TCNConfig
torch.serialization.add_safe_globals([TCNConfig])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TCN_FEATURES = FEATURE_NAMES[:17]
WINDOW_SIZE = 200
NORM_WINDOW = 500  # rolling-z window (matches training / Stage3TCN.feed)
ALERT_THRESHOLD = 0.5  # operating-point Stage-3 gate (was 0.6 — BUG-03)
BASELINE_DROP_PCT = 2.0  # baseline: alert if price drops 2% in 60s
BASELINE_WINDOW_MS = 60_000


def load_trained_tcn(model_path: str, device: str = "auto") -> TCNDetector:
    """Load a trained TCN from disk."""
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    data = torch.load(model_path, map_location=device, weights_only=True)
    config = data["config"]
    model = TCNDetector(config).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    logger.info("Loaded TCN from %s (device=%s)", model_path, device)
    return model


def run_tcn_backtest(
    model: TCNDetector,
    df: pd.DataFrame,
    max_ticks: int = 500_000,
    device: str = "cpu",
) -> dict:
    """Slide the TCN over crash-day data, fire alerts, measure TTD."""
    if max_ticks > 0 and len(df) > max_ticks:
        indices = np.linspace(0, len(df) - 1, max_ticks, dtype=int)
        df = df.iloc[indices].copy()
        logger.info("Sampled to %d ticks", len(df))

    extractor = FeatureExtractor()
    feature_window = deque(maxlen=WINDOW_SIZE)

    # Extract ground-truth crash labels
    ticks = list(df_to_ticks(df, symbol="BACKTEST"))
    crashes = label_crashes(ticks, drop_threshold_pct=BASELINE_DROP_PCT,
                            window_ms=BASELINE_WINDOW_MS)
    logger.info("Found %d ground-truth crash windows", len(crashes))

    # One sequential pass: raw feature matrix, then the shared rolling-z (BUG-03)
    # so the TCN window matches training / Stage3TCN.feed (not raw features).
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    for i, tick in enumerate(ticks):
        fd = extractor.extract(tick)
        F[i] = [float(fd.get(f, 0.0)) or 0.0 for f in TCN_FEATURES]
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    norm = normalize_z(F, NORM_WINDOW)

    # Run TCN
    alerts = []
    tcn_scores = []
    t0 = time.time()

    for i, tick in enumerate(ticks):
        if i % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(1, elapsed)
            logger.info("  Processing tick %d/%d (%.0f/sec, %.0fs elapsed)",
                        i, len(ticks), rate, elapsed)

        feature_window.append(norm[i])

        if len(feature_window) < WINDOW_SIZE:
            continue

        # Score the window
        window_array = np.array(list(feature_window))
        with torch.no_grad():
            x = torch.FloatTensor(window_array).T.unsqueeze(0).to(device)
            scores = model(x)
            score = float(scores[0, -1].item())

        tcn_scores.append({"timestamp_ms": tick.timestamp_ms, "score": score})

        if score >= ALERT_THRESHOLD:
            alerts.append({
                "timestamp_ms": tick.timestamp_ms,
                "score": score,
                "mid_price": tick.book.mid_price,
            })

    logger.info("TCN backtest: %d ticks, %d alerts, %.1fs",
                len(ticks), len(alerts), time.time() - t0)

    # Evaluate against ground truth
    results = evaluate_alerts(alerts, crashes)
    results["n_ticks"] = len(ticks)
    results["n_crashes"] = len(crashes)
    results["n_alerts"] = len(alerts)
    results["tcn_scores_sample"] = tcn_scores[::max(1, len(tcn_scores) // 1000)]  # downsample

    return results


def run_baseline_backtest(df: pd.DataFrame, max_ticks: int = 500_000) -> dict:
    """Run a naive threshold-based baseline detector.

    Fires an alert when the mid-price drops ≥ 2% within 60 seconds.
    This is what production circuit breakers do.
    """
    if max_ticks > 0 and len(df) > max_ticks:
        indices = np.linspace(0, len(df) - 1, max_ticks, dtype=int)
        df = df.iloc[indices].copy()

    ticks = list(df_to_ticks(df, symbol="BASELINE"))
    crashes = label_crashes(ticks, drop_threshold_pct=BASELINE_DROP_PCT,
                            window_ms=BASELINE_WINDOW_MS)

    # Baseline: alert at the moment the price has dropped ≥ threshold
    alerts = []
    peak_price = ticks[0].book.mid_price or 0.0
    peak_ts = ticks[0].timestamp_ms

    for tick in ticks:
        mid = tick.book.mid_price
        if mid is None or mid <= 0:
            continue

        ts = tick.timestamp_ms
        if ts - peak_ts > BASELINE_WINDOW_MS:
            peak_price = mid
            peak_ts = ts
            continue

        if mid > peak_price:
            peak_price = mid
            peak_ts = ts

        if peak_price > 0:
            drop_pct = (peak_price - mid) / peak_price * 100
            if drop_pct >= BASELINE_DROP_PCT:
                alerts.append({
                    "timestamp_ms": ts,
                    "drop_pct": drop_pct,
                    "mid_price": mid,
                })
                peak_price = mid
                peak_ts = ts

    results = evaluate_alerts(alerts, crashes)
    results["n_ticks"] = len(ticks)
    results["n_crashes"] = len(crashes)
    results["n_alerts"] = len(alerts)
    return results


def evaluate_alerts(alerts: list[dict], crashes: list) -> dict:
    """Evaluate alerts against ground-truth crash windows."""
    true_positives = 0
    false_positives = 0
    ttd_ms = []
    matched_crashes = set()

    for alert in alerts:
        alert_ts = alert["timestamp_ms"]
        matched = False
        for j, crash in enumerate(crashes):
            if j in matched_crashes:
                continue
            # Alert fires within the crash window or up to 5s before
            if crash.start_ts - 5000 <= alert_ts <= crash.end_ts:
                true_positives += 1
                matched_crashes.add(j)
                ttd = crash.end_ts - alert_ts  # positive = before crash
                ttd_ms.append(ttd)
                matched = True
                break
        if not matched:
            false_positives += 1

    false_negatives = len(crashes) - true_positives
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, len(crashes))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "ttd_ms": ttd_ms,
        "median_ttd_ms": float(np.median(ttd_ms)) if ttd_ms else 0.0,
        "mean_ttd_ms": float(np.mean(ttd_ms)) if ttd_ms else 0.0,
    }


def main() -> int:
    global ALERT_THRESHOLD
    parser = argparse.ArgumentParser(description="Window-based backtest with trained TCN")
    parser.add_argument("--data", required=True, help="Parquet file of crash data")
    parser.add_argument("--model", required=True, help="Trained TCN model path")
    parser.add_argument("--output", default="results/window_backtest.json")
    parser.add_argument("--max-ticks", type=int, default=500_000)
    parser.add_argument("--threshold", type=float, default=ALERT_THRESHOLD)
    args = parser.parse_args()

    
    ALERT_THRESHOLD = args.threshold

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    df = load_parquet(args.data)
    logger.info("Loaded %d ticks", len(df))

    # Load model
    model = load_trained_tcn(args.model, device=device)

    # Run TCN backtest
    logger.info("\n" + "=" * 60)
    logger.info("TCN DETECTOR BACKTEST")
    logger.info("=" * 60)
    tcn_results = run_tcn_backtest(model, df, max_ticks=args.max_ticks, device=device)

    # Run baseline
    logger.info("\n" + "=" * 60)
    logger.info("BASELINE (threshold circuit breaker)")
    logger.info("=" * 60)
    baseline_results = run_baseline_backtest(df, max_ticks=args.max_ticks)

    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("RESULTS COMPARISON")
    logger.info("=" * 60)
    logger.info("%-25s  %-15s  %-15s", "", "TCN Detector", "Baseline")
    logger.info("-" * 60)
    logger.info("%-25s  %-15d  %-15d", "Alerts", tcn_results["n_alerts"], baseline_results["n_alerts"])
    logger.info("%-25s  %-15d  %-15d", "True positives", tcn_results["true_positives"], baseline_results["true_positives"])
    logger.info("%-25s  %-15d  %-15d", "False positives", tcn_results["false_positives"], baseline_results["false_positives"])
    logger.info("%-25s  %-15d  %-15d", "False negatives", tcn_results["false_negatives"], baseline_results["false_negatives"])
    logger.info("%-25s  %-15.3f  %-15.3f", "Precision", tcn_results["precision"], baseline_results["precision"])
    logger.info("%-25s  %-15.3f  %-15.3f", "Recall", tcn_results["recall"], baseline_results["recall"])
    logger.info("%-25s  %-15.3f  %-15.3f", "F1", tcn_results["f1"], baseline_results["f1"])
    logger.info("%-25s  %-15.1f  %-15.1f", "Median TTD (ms)", tcn_results["median_ttd_ms"], baseline_results["median_ttd_ms"])
    logger.info("=" * 60)

    if tcn_results["median_ttd_ms"] > 0:
        logger.info("TCN fires %.1f ms BEFORE the crash (early warning!)", tcn_results["median_ttd_ms"])
    if baseline_results["median_ttd_ms"] > 0:
        logger.info("Baseline fires %.1f ms AFTER the crash (too late)", -baseline_results["median_ttd_ms"])

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({
            "tcn": tcn_results,
            "baseline": baseline_results,
        }, f, indent=2, default=str)
    logger.info("Saved to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
