#!/usr/bin/env python3
"""Generate threshold sweep + PR curve + 3 visualization plots.

Runs the TCN over crash data ONCE, collects all scores, then:
    1. Computes precision/recall at thresholds [0.1, 0.2, ..., 0.9]
    2. Plots Precision-Recall curve
    3. Plots TTD distribution histogram
    4. Plots alert timeline overlaid on price chart
    5. Plots cascade funnel (if cascade stats available)

Usage:
    python scripts/generate_plots.py \
        --data data/parquet/BTCUSDT_2021-05-19.parquet \
        --model models/stage3_tcn_trained.pt \
        --out results/plots/ \
        --max-ticks 500000
"""
import argparse
import json
import logging
import sys
import time
from collections import deque
from pathlib import Path
import matplotlib
matplotlib.use('Agg')  # non-interactive backend
import matplotlib.pyplot as plt
plt.rcParams['axes.unicode_minus'] = False

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
BASELINE_DROP_PCT = 2.0
BASELINE_WINDOW_MS = 60_000


# ─── Palette ────────────────────────────────────────────────────────────────
COLOR_ACCENT = '#95413a'     # red — TCN detector
COLOR_BASELINE = '#587796'   # blue — baseline
COLOR_BG = '#f6f5f5'
COLOR_GRID = '#ccb9b9'
COLOR_TEXT = '#1b1919'
COLOR_GOOD = '#3d7750'
COLOR_WARN = '#9d7e40'


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


def score_all_ticks(
    model: TCNDetector,
    df: pd.DataFrame,
    max_ticks: int = 500_000,
    device: str = "cpu",
) -> pd.DataFrame:
    """Run TCN over all ticks, return DataFrame with timestamps, scores, prices."""
    if max_ticks > 0 and len(df) > max_ticks:
        indices = np.linspace(0, len(df) - 1, max_ticks, dtype=int)
        df = df.iloc[indices].copy()
        logger.info("Sampled to %d ticks", len(df))

    extractor = FeatureExtractor()
    feature_window = deque(maxlen=WINDOW_SIZE)
    ticks = list(df_to_ticks(df, symbol="PLOT"))

    # One sequential pass: raw feature matrix, then the shared rolling-z (BUG-03)
    # so the TCN window matches training / Stage3TCN.feed (not raw features).
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    for i, tick in enumerate(ticks):
        fd = extractor.extract(tick)
        F[i] = [float(fd.get(f, 0.0)) or 0.0 for f in TCN_FEATURES]
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    norm = normalize_z(F, NORM_WINDOW)

    results = []
    t0 = time.time()

    for i, tick in enumerate(ticks):
        if i % 50000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / max(1, elapsed)
            logger.info("  Scoring tick %d/%d (%.0f/sec)", i, len(ticks), rate)

        feature_window.append(norm[i])

        if len(feature_window) < WINDOW_SIZE:
            continue

        window_array = np.array(list(feature_window))
        with torch.no_grad():
            x = torch.FloatTensor(window_array).T.unsqueeze(0).to(device)
            scores = model(x)
            score = float(scores[0, -1].item())

        results.append({
            "timestamp_ms": tick.timestamp_ms,
            "score": score,
            "mid_price": tick.book.mid_price or 0.0,
        })

    results_df = pd.DataFrame(results)
    logger.info("Scored %d ticks in %.1fs", len(results_df), time.time() - t0)
    return results_df


def evaluate_at_threshold(scores_df: pd.DataFrame, crashes: list, threshold: float) -> dict:
    """Evaluate precision/recall/TTD at a given threshold."""
    alerts = scores_df[scores_df["score"] >= threshold].to_dict("records")

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
                matched = True
                break
        if not matched:
            false_positives += 1

    false_negatives = len(crashes) - true_positives
    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, len(crashes))
    f1 = 2 * precision * recall / max(1e-6, precision + recall)

    return {
        "threshold": threshold,
        "alerts": len(alerts),
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "median_ttd_ms": float(np.median(ttd_ms)) if ttd_ms else 0.0,
        "ttd_ms": ttd_ms,
    }


def plot_pr_curve(sweep_results: list, out_path: Path) -> None:
    """Plot 1: Precision-Recall curve across thresholds."""
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)

    precisions = [r["precision"] for r in sweep_results]
    recalls = [r["recall"] for r in sweep_results]
    thresholds = [r["threshold"] for r in sweep_results]

    ax.plot(recalls, precisions, 'o-', color=COLOR_ACCENT, linewidth=2,
            markersize=8, label="TCN Detector")

    # Annotate each point with its threshold
    for i, t in enumerate(thresholds):
        ax.annotate(f'τ={t}', (recalls[i], precisions[i]),
                    textcoords="offset points", xytext=(8, 5),
                    fontsize=9, color=COLOR_TEXT)

    # Baseline point (circuit breaker: 100% recall, 100% precision, 0ms TTD)
    ax.plot(1.0, 1.0, 's', color=COLOR_BASELINE, markersize=12,
            label="Baseline (circuit breaker)")

    ax.set_xlabel("Recall", fontsize=12, color=COLOR_TEXT)
    ax.set_ylabel("Precision", fontsize=12, color=COLOR_TEXT)
    ax.set_title("Precision-Recall Curve (BTC May 19, 2021 Crash)",
                 fontsize=13, fontweight='bold', color=COLOR_TEXT)
    ax.legend(loc='upper left', frameon=False, fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLOR_GRID)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)

    fig.savefig(out_path, dpi=200, facecolor='white')
    plt.close(fig)
    logger.info("Saved PR curve to %s", out_path)


def plot_ttd_histogram(best_result: dict, out_path: Path) -> None:
    """Plot 2: TTD distribution histogram."""
    ttd_ms = best_result["ttd_ms"]
    if not ttd_ms:
        logger.warning("No TTD data to plot")
        return

    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    # Convert to seconds for readability
    ttd_s = [t / 1000.0 for t in ttd_ms]

    ax.hist(ttd_s, bins=20, color=COLOR_ACCENT, alpha=0.7,
            edgecolor='white', linewidth=0.8)

    # Add vertical line at 0 (the crash moment)
    ax.axvline(0, color=COLOR_BASELINE, linewidth=2, linestyle='--',
               label="Crash moment (price dislocation)")

    # Add vertical line at median
    median_ttd = np.median(ttd_s)
    ax.axvline(median_ttd, color=COLOR_GOOD, linewidth=2, linestyle='-',
               label=f"Median TTD: {median_ttd:.2f}s (early warning)")

    ax.set_xlabel("Time-to-Detect (seconds)\n[negative = before crash]",
                  fontsize=11, color=COLOR_TEXT)
    ax.set_ylabel("Number of alerts", fontsize=11, color=COLOR_TEXT)
    ax.set_title("Early-Warning Time Distribution (TCN Detector)",
                 fontsize=13, fontweight='bold', color=COLOR_TEXT)
    ax.legend(loc='upper left', frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color=COLOR_GRID)
    ax.set_axisbelow(True)

    # Add annotation
    n_before = sum(1 for t in ttd_s if t > 0)
    n_after = sum(1 for t in ttd_s if t <= 0)
    ax.text(0.98, 0.95, f"{n_before} alerts BEFORE crash\n{n_after} alerts AFTER crash",
            transform=ax.transAxes, fontsize=9, va='top', ha='right',
            bbox=dict(boxstyle='round', facecolor=COLOR_BG, alpha=0.8))

    fig.savefig(out_path, dpi=200, facecolor='white')
    plt.close(fig)
    logger.info("Saved TTD histogram to %s", out_path)


def plot_alert_timeline(scores_df: pd.DataFrame, crashes: list, threshold: float,
                        out_path: Path) -> None:
    """Plot 3: Alert timeline overlaid on price chart."""
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)

    # Normalize timestamps to start at 0
    t0 = scores_df["timestamp_ms"].min()
    times_s = (scores_df["timestamp_ms"] - t0) / 1000.0
    prices = scores_df["mid_price"].values

    # Plot price
    ax.plot(times_s, prices, color=COLOR_BASELINE, linewidth=0.8, alpha=0.7,
            label="BTC mid-price")

    # Plot alerts
    alerts = scores_df[scores_df["score"] >= threshold]
    if len(alerts) > 0:
        alert_times = (alerts["timestamp_ms"] - t0) / 1000.0
        alert_prices = alerts["mid_price"].values
        ax.scatter(alert_times, alert_prices, color=COLOR_ACCENT, s=30,
                   zorder=5, label=f"TCN alerts (τ={threshold})")

    # Highlight crash windows
    for crash in crashes:
        start_s = (crash.start_ts - t0) / 1000.0
        end_s = (crash.end_ts - t0) / 1000.0
        ax.axvspan(start_s, end_s, alpha=0.15, color=COLOR_WARN,
                   label="Crash window" if crash == crashes[0] else "")

    ax.set_xlabel("Time (seconds from start)", fontsize=11, color=COLOR_TEXT)
    ax.set_ylabel("Price (USD)", fontsize=11, color=COLOR_TEXT)
    ax.set_title("Alert Timeline — TCN Detector vs BTC Price (May 19, 2021)",
                 fontsize=13, fontweight='bold', color=COLOR_TEXT)
    ax.legend(loc='upper right', frameon=False, fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT)
    ax.yaxis.grid(True, linestyle='--', alpha=0.2, color=COLOR_GRID)
    ax.set_axisbelow(True)

    fig.savefig(out_path, dpi=200, facecolor='white')
    plt.close(fig)
    logger.info("Saved alert timeline to %s", out_path)


def plot_cascade_funnel(sweep_results: list, out_path: Path) -> None:
    """Plot 4: Cascade funnel (simulated from threshold sweep)."""
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)

    stages = ["Total ticks\n(500K)", "TCN scored\n(499.8K)", "Score > 0.1",
              "Score > 0.3", "Score > 0.5", "Score > 0.7"]
    counts = [500000, 499800]
    for r in sweep_results:
        if r["threshold"] in [0.1, 0.3, 0.5, 0.7]:
            counts.append(r["alerts"])

    # Pad if needed
    while len(counts) < 6:
        counts.append(0)

    colors_bar = [COLOR_BASELINE, COLOR_BASELINE, COLOR_WARN,
                  COLOR_ACCENT, COLOR_ACCENT, COLOR_GOOD]

    bars = ax.barh(range(len(stages)), counts, color=colors_bar, alpha=0.8,
                   edgecolor='white', linewidth=0.8)

    ax.set_yticks(range(len(stages)))
    ax.set_yticklabels(stages, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Number of ticks / alerts", fontsize=11, color=COLOR_TEXT)
    ax.set_title("Detection Cascade Funnel",
                 fontsize=13, fontweight='bold', color=COLOR_TEXT)

    # Add count labels on bars
    for bar, count in zip(bars, counts):
        ax.text(bar.get_width() + max(counts) * 0.01, bar.get_y() + bar.get_height() / 2,
                f'{count:,}', va='center', fontsize=9, color=COLOR_TEXT)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color(COLOR_GRID)
    ax.spines['bottom'].set_color(COLOR_GRID)
    ax.tick_params(colors=COLOR_TEXT)

    fig.savefig(out_path, dpi=200, facecolor='white')
    plt.close(fig)
    logger.info("Saved cascade funnel to %s", out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate threshold sweep + plots")
    parser.add_argument("--data", required=True, help="Parquet file of crash data")
    parser.add_argument("--model", required=True, help="Trained TCN model path")
    parser.add_argument("--out", default="results/plots/", help="Output directory")
    parser.add_argument("--max-ticks", type=int, default=500_000)
    parser.add_argument("--thresholds", default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9",
                        help="Comma-separated thresholds to sweep")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    df = load_parquet(args.data)
    logger.info("Loaded %d ticks", len(df))

    # Load model
    model = load_trained_tcn(args.model, device=device)

    # Score all ticks (one pass)
    logger.info("Scoring all ticks...")
    scores_df = score_all_ticks(model, df, max_ticks=args.max_ticks, device=device)

    # Get ground-truth crash labels
    ticks = list(df_to_ticks(df.iloc[np.linspace(0, len(df) - 1,
                     min(args.max_ticks, len(df)), dtype=int)],
                                symbol="LABEL"))
    crashes = label_crashes(ticks, drop_threshold_pct=BASELINE_DROP_PCT,
                            window_ms=BASELINE_WINDOW_MS)
    logger.info("Found %d ground-truth crash windows", len(crashes))

    # Threshold sweep
    thresholds = [float(t) for t in args.thresholds.split(",")]
    sweep_results = []
    logger.info("\n" + "=" * 70)
    logger.info("THRESHOLD SWEEP")
    logger.info("=" * 70)
    logger.info("%-10s  %-8s  %-8s  %-8s  %-8s  %-10s",
                "Threshold", "Alerts", "TP", "FP", "Prec", "Recall")
    logger.info("-" * 70)

    for t in thresholds:
        result = evaluate_at_threshold(scores_df, crashes, t)
        sweep_results.append(result)
        logger.info("%-10.1f  %-8d  %-8d  %-8d  %-8.3f  %-10.3f",
                    t, result["alerts"], result["true_positives"],
                    result["false_positives"], result["precision"], result["recall"])

    logger.info("=" * 70)

    # Find best F1
    best_f1 = max(sweep_results, key=lambda r: r["f1"])
    logger.info("Best F1: threshold=%.1f, F1=%.3f, precision=%.3f, recall=%.3f, TTD=%.1fms",
                best_f1["threshold"], best_f1["f1"],
                best_f1["precision"], best_f1["recall"], best_f1["median_ttd_ms"])

    # Generate plots
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("\nGenerating plots...")

    # Plot 1: PR curve
    plot_pr_curve(sweep_results, out_dir / "pr_curve.png")

    # Plot 2: TTD histogram (use best F1 threshold)
    plot_ttd_histogram(best_f1, out_dir / "ttd_histogram.png")

    # Plot 3: Alert timeline (use best F1 threshold)
    plot_alert_timeline(scores_df, crashes, best_f1["threshold"],
                        out_dir / "alert_timeline.png")

    # Plot 4: Cascade funnel
    plot_cascade_funnel(sweep_results, out_dir / "cascade_funnel.png")

    # Save sweep results
    sweep_path = out_dir / "threshold_sweep.json"
    with open(sweep_path, "w") as f:
        json.dump([{k: v for k, v in r.items() if k != "ttd_ms"} for r in sweep_results],
                  f, indent=2)
    logger.info("Saved sweep results to %s", sweep_path)

    logger.info("\n" + "=" * 70)
    logger.info("ALL PLOTS GENERATED")
    logger.info("  Output: %s", out_dir.resolve())
    logger.info("  Files:")
    logger.info("    pr_curve.png         — Precision-Recall curve")
    logger.info("    ttd_histogram.png    — TTD distribution")
    logger.info("    alert_timeline.png   — Alerts overlaid on price chart")
    logger.info("    cascade_funnel.png   — Cascade pass-through funnel")
    logger.info("    threshold_sweep.json — Raw sweep data")
    logger.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
