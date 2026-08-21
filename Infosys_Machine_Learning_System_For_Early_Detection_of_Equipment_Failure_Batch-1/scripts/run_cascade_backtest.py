#!/usr/bin/env python3
"""Full 5-stage cascade backtest with trained TCN + per-crash breakdown table.

Wires the trained TCN into Stage 3 of the cascade, runs the full pipeline
(Stage 1 → 2 → 3), and outputs:
    1. Cascade pass-through funnel (Stage 1 → Stage 2 → Stage 3)
    2. Per-crash breakdown table (each true positive with TTD, features)
    3. Comparison vs baseline

Usage:
    python scripts/run_cascade_backtest.py \
        --data data/parquet/BTCUSDT_2021-05-19.parquet \
        --model models/stage3_tcn_trained.pt \
        --out results/cascade_backtest.json \
        --max-ticks 500000
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
from flash_crash_watchdog.models.stage1_statistical import Stage1Statistical, Stage1Config
from flash_crash_watchdog.models.stage2_isolation_forest import Stage2IsolationForest
from flash_crash_watchdog.models.stage3_tcn import normalize_z, TCNDetector, TCNConfig
torch.serialization.add_safe_globals([TCNConfig])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TCN_FEATURES = FEATURE_NAMES[:17]
TCN_FEAT_IDX = {name: j for j, name in enumerate(TCN_FEATURES)}
WINDOW_SIZE = 200
NORM_WINDOW = 500  # rolling-z window (matches training / Stage3TCN.feed)
ALERT_THRESHOLD = 0.3
BASELINE_DROP_PCT = 2.0
BASELINE_WINDOW_MS = 60_000


def load_trained_tcn(model_path: str, device: str = "auto") -> TCNDetector:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(model_path, map_location=device, weights_only=True)
    config = data["config"]
    model = TCNDetector(config).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    logger.info("Loaded TCN from %s (device=%s)", model_path, device)
    return model


def run_full_cascade(
    tcn_model: TCNDetector,
    df: pd.DataFrame,
    max_ticks: int = 500_000,
    device: str = "cpu",
) -> dict:
    """Run the full Stage 1 → 2 → 3 cascade with per-crash breakdown."""
    if max_ticks > 0 and len(df) > max_ticks:
        indices = np.linspace(0, len(df) - 1, max_ticks, dtype=int)
        df = df.iloc[indices].copy()
        logger.info("Sampled to %d ticks", len(df))

    extractor = FeatureExtractor()
    feature_window = deque(maxlen=WINDOW_SIZE)

    # Initialize stages. Gates aligned with configs/pipeline.yml (BUG-03): the
    # previous 1.0/1.0/0.5 + contamination 0.30 silently diverged from the
    # configured operating cascade (3.0/3.0/2.5, contamination 0.05).
    stage1 = Stage1Statistical(Stage1Config(velocity_z_threshold=3.0, spread_z_threshold=3.0, obi_z_threshold=2.5))
    stage2 = Stage2IsolationForest(n_estimators=100, contamination=0.05)
    ticks = list(df_to_ticks(df, symbol="CASCADE"))

    # One sequential pass: raw feature matrix F (for the shared rolling-z) while
    # training Stage 2 on the first 50K ticks (normal data warmup).
    logger.info("Training Stage 2 (Isolation Forest) on warmup data...")
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    warmup_features = []
    for i, tick in enumerate(ticks):
        fd = extractor.extract(tick)
        F[i] = [float(fd.get(f, 0.0)) or 0.0 for f in TCN_FEATURES]
        if i < 50_000:
            warmup_features.append([fd.get(f, 0.0) for f in FEATURE_NAMES[:12]])
    stage2.fit(np.array(warmup_features))
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    # Shared rolling-z transform — the window fed to the TCN now matches the
    # online Stage3TCN.feed normalization (BUG-03), not raw features.
    norm = normalize_z(F, NORM_WINDOW)

    # Get ground-truth crash labels
    crashes = label_crashes(ticks, drop_threshold_pct=BASELINE_DROP_PCT,
                            window_ms=BASELINE_WINDOW_MS)
    logger.info("Found %d ground-truth crash windows", len(crashes))

    # Cascade stats
    stats = {
        "total_ticks": 0,
        "stage1_passed": 0,
        "stage2_passed": 0,
        "stage3_passed": 0,
        "alerts_fired": 0,
    }

    alerts = []
    t0 = time.time()

    for i, tick in enumerate(ticks):
        if i % 50000 == 0:
            elapsed = time.time() - t0
            logger.info("  Processing tick %d/%d (%.0f/sec, %.0fs)",
                        i, len(ticks), (i+1)/max(1,elapsed), elapsed)

        stats["total_ticks"] += 1

        # Update the TCN feature window on EVERY tick (continuous sliding window)
        feature_window.append(norm[i])

        # Stage 1 — Statistical pre-filter
        s1_score, s1_pass = stage1.score(tick)
        if not s1_pass:
            continue
        stats["stage1_passed"] += 1

        # Stage 2 — Isolation Forest
        s2_score, s2_pass = stage2.score(tick)
        if not s2_pass:
            continue
        stats["stage2_passed"] += 1

        # Stage 3 — TCN (only run on suspects, but window is continuous)
        if len(feature_window) < WINDOW_SIZE:
            continue

        window_array = np.array(list(feature_window))
        with torch.no_grad():
            x = torch.FloatTensor(window_array).T.unsqueeze(0).to(device)
            scores = tcn_model(x)
            s3_score = float(scores[0, -1].item())

        if s3_score >= ALERT_THRESHOLD:
            stats["stage3_passed"] += 1
            stats["alerts_fired"] += 1
            alerts.append({
                "timestamp_ms": tick.timestamp_ms,
                "score": s3_score,
                "mid_price": tick.book.mid_price or 0.0,
                "s1_score": s1_score,
                "s2_score": s2_score,
                "s3_score": s3_score,
                "features": {
                    "obi_10": float(F[i][TCN_FEAT_IDX["f2_obi_10"]]),
                    "bid_depth_10": float(F[i][TCN_FEAT_IDX["f2_bid_depth_10"]]),
                    "ask_depth_10": float(F[i][TCN_FEAT_IDX["f2_ask_depth_10"]]),
                    "spread_bps": tick.book.spread_bps or 0.0,
                    "vpin": float(F[i][TCN_FEAT_IDX["f3_vpin"]]),
                    "realized_vol_1s": float(F[i][TCN_FEAT_IDX["f4_realized_vol_1s"]]),
                    "variance_ratio": float(F[i][TCN_FEAT_IDX["f4_variance_ratio"]]),
                    "trade_arrival_rate": float(F[i][TCN_FEAT_IDX["f1_trade_arrival_rate"]]),
                },
            })

    logger.info("Cascade complete: %d ticks in %.1fs", len(ticks), time.time() - t0)

    # Evaluate alerts against ground truth
    results = evaluate_alerts(alerts, crashes)
    results["cascade_stats"] = stats
    results["per_crash_breakdown"] = build_breakdown(alerts, crashes)

    return results


def evaluate_alerts(alerts: list, crashes: list) -> dict:
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
            if crash.start_ts - 5000 <= alert_ts <= crash.end_ts:
                true_positives += 1
                matched_crashes.add(j)
                ttd = crash.end_ts - alert_ts
                ttd_ms.append(ttd)
                alert["matched_crash"] = j
                alert["ttd_ms"] = ttd
                alert["crash_drop_pct"] = crash.drop_pct
                matched = True
                break
        if not matched:
            false_positives += 1

    false_negatives = len(crashes) - true_positives
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, len(crashes))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    return {
        "alerts_fired": len(alerts),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_ttd_ms": float(np.median(ttd_ms)) if ttd_ms else 0.0,
        "ttd_ms": ttd_ms,
    }


def build_breakdown(alerts: list, crashes: list) -> list:
    """Build per-crash breakdown table for true positives."""
    breakdown = []
    for alert in alerts:
        if "matched_crash" not in alert:
            continue
        crash = crashes[alert["matched_crash"]]
        breakdown.append({
            "alert_timestamp_ms": alert["timestamp_ms"],
            "crash_start_ms": crash.start_ts,
            "crash_end_ms": crash.end_ts,
            "ttd_ms": alert["ttd_ms"],
            "ttd_seconds": alert["ttd_ms"] / 1000.0,
            "crash_drop_pct": round(crash.drop_pct, 2),
            "peak_price": round(crash.peak_price, 2),
            "trough_price": round(crash.trough_price, 2),
            "alert_price": round(alert["mid_price"], 2),
            "tcn_score": round(alert["score"], 4),
            "s1_score": round(alert["s1_score"], 4),
            "s2_score": round(alert["s2_score"], 4),
            "features": {k: round(v, 6) for k, v in alert["features"].items()},
        })
    return breakdown


def print_cascade_funnel(stats: dict) -> None:
    logger.info("\n" + "=" * 60)
    logger.info("CASCADE FUNNEL")
    logger.info("=" * 60)
    total = stats["total_ticks"]
    s1 = stats["stage1_passed"]
    s2 = stats["stage2_passed"]
    s3 = stats["stage3_passed"]
    alerts = stats["alerts_fired"]
    logger.info("  Total ticks:      %d", total)
    logger.info("  Stage 1 passed:   %d (%.1f%%)", s1, s1/max(1,total)*100)
    logger.info("  Stage 2 passed:   %d (%.1f%% of S1)", s2, s2/max(1,s1)*100)
    logger.info("  Stage 3 passed:   %d (%.1f%% of S2)", s3, s3/max(1,s2)*100)
    logger.info("  Alerts fired:     %d", alerts)
    logger.info("=" * 60)


def print_breakdown_table(breakdown: list) -> None:
    logger.info("\n" + "=" * 80)
    logger.info("PER-CRASH BREAKDOWN TABLE (%d true positives)", len(breakdown))
    logger.info("=" * 80)
    logger.info("%-5s  %-10s  %-8s  %-8s  %-8s  %-8s  %-8s",
                "#", "TTD (s)", "Drop%", "Price", "OBI", "VPIN", "Vol")
    logger.info("-" * 80)
    for i, b in enumerate(breakdown):
        logger.info("%-5d  %-10.2f  %-8.2f  %-8.2f  %-8.4f  %-8.4f  %-8.6f",
                    i+1,
                    b["ttd_seconds"],
                    b["crash_drop_pct"],
                    b["alert_price"],
                    b["features"]["obi_10"],
                    b["features"]["vpin"],
                    b["features"]["realized_vol_1s"])
    logger.info("=" * 80)


def main() -> int:
    global ALERT_THRESHOLD
    parser = argparse.ArgumentParser(description="Full cascade backtest + breakdown")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", default="results/cascade_backtest.json")
    parser.add_argument("--max-ticks", type=int, default=500_000)
    parser.add_argument("--threshold", type=float, default=ALERT_THRESHOLD)
    args = parser.parse_args()

    
    ALERT_THRESHOLD = args.threshold

    device = "cuda" if torch.cuda.is_available() else "cpu"
    df = load_parquet(args.data)
    tcn_model = load_trained_tcn(args.model, device=device)

    results = run_full_cascade(tcn_model, df, max_ticks=args.max_ticks, device=device)

    print_cascade_funnel(results["cascade_stats"])
    logger.info("\nPrecision: %.3f | Recall: %.3f | F1: %.3f | Median TTD: %.1f ms",
                results["precision"], results["recall"], results["f1"],
                results["median_ttd_ms"])
    print_breakdown_table(results["per_crash_breakdown"])

    output_path = Path(args.out)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved to %s", output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
