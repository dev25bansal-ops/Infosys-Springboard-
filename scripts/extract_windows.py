#!/usr/bin/env python3
"""Extract sliding windows from real tick data with lookahead crash labels.

For each window of 200 ticks, the label is:
    1 (crash) if the mid-price drops ≥ threshold% within the next lookahead_ms
    0 (normal) otherwise

This gives us REAL labeled training data — no synthetic anomalies.

Usage:
    python scripts/extract_windows.py --data data/parquet/BTCUSDT_2021-05-19.parquet --out data/windows/
    python scripts/extract_windows.py --data data/parquet/BTCUSDT_2024-01-15.parquet --out data/windows/ --label normal
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.data.historical_loader import df_to_ticks, load_parquet
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Features used by the TCN (F1-F4, 17 features)
TCN_FEATURES = FEATURE_NAMES[:17]

WINDOW_SIZE = 200       # ticks per window (~20 seconds at 10 ticks/sec)
STRIDE = 10             # sliding window stride (overlap = 190 ticks)
LOOKAHEAD_MS = 5_000    # label = crash if price drops ≥ threshold within next 5 seconds
CRASH_THRESHOLD_PCT = 2.0  # 2% drop = crash


def extract_features_from_df(df: pd.DataFrame, max_ticks: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract features + mid-prices + timestamps from a DataFrame.

    Returns:
        features: shape (N, 17) — feature vector per tick
        mid_prices: shape (N,) — mid-price per tick (for labeling)
        timestamps: shape (N,) — tick timestamps in ms (for wall-clock labels)
    """
    if max_ticks > 0 and len(df) > max_ticks:
        indices = np.linspace(0, len(df) - 1, max_ticks, dtype=int)
        df = df.iloc[indices].copy()
        logger.info("Sampled to %d ticks", len(df))

    extractor = FeatureExtractor()
    features_list = []
    mid_prices = []

    for i, tick in enumerate(df_to_ticks(df, symbol="EXTRACT")):
        if i % 100000 == 0:
            logger.info("  Extracting features: %d/%d", i, len(df))
        features = extractor.extract(tick)
        features_list.append([features.get(f, 0.0) for f in TCN_FEATURES])
        mid_prices.append(tick.book.mid_price or 0.0)

    features = np.array(features_list, dtype=np.float32)
    mid_prices = np.array(mid_prices, dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    timestamps = df["timestamp_ms"].to_numpy(dtype=np.float64)
    return features, mid_prices, timestamps


def label_windows(
    features: np.ndarray,
    mid_prices: np.ndarray,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    lookahead_ms: int = LOOKAHEAD_MS,
    crash_threshold: float = CRASH_THRESHOLD_PCT,
    timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build sliding windows with lookahead crash labels.

    For each window starting at index i:
        - Window: features[i : i+window_size]
        - Label: 1 if mid_price drops ≥ crash_threshold% within the next
                 lookahead_ms *wall-clock* after the window ends, else 0

    When ``timestamps`` is provided (recommended), the lookahead is a real time
    window (RSR-02): the label horizon for a window ending at tick ``end_idx`` is
    [timestamps[end_idx], timestamps[end_idx] + lookahead_ms]. Windows whose
    horizon is not fully covered by data are skipped. When ``timestamps`` is None,
    the old tick-count heuristic (lookahead_ms // 100, ~10 ticks/s) is used.

    Returns:
        windows: shape (N, window_size, 17)
        labels: shape (N,)
    """
    n = len(features)
    windows = []
    labels = []
    n_positive = 0
    n_negative = 0

    for i in range(0, n - window_size, stride):
        end_idx = i + window_size - 1
        current_price = mid_prices[end_idx]

        # Future price series that defines the crash label.
        if timestamps is not None:
            t_horizon = timestamps[end_idx] + float(lookahead_ms)
            j = int(np.searchsorted(timestamps, t_horizon, side="right")) - 1
            if j < end_idx or timestamps[j] < t_horizon:
                continue  # lookahead horizon not fully covered by data
            future_prices = mid_prices[end_idx + 1 : j + 1]
        else:
            lookahead_ticks = max(1, lookahead_ms // 100)  # legacy ~10/s heuristic
            if end_idx + 1 + lookahead_ticks > n:
                continue
            future_prices = mid_prices[end_idx + 1 : end_idx + 1 + lookahead_ticks]

        # Extract window
        window = features[i : i + window_size]
        windows.append(window)

        # Label: does the price drop ≥ threshold% within the next lookahead window?
        if current_price > 0 and len(future_prices) > 0:
            min_future = np.min(future_prices)
            drop_pct = (current_price - min_future) / current_price * 100
            label = 1 if drop_pct >= crash_threshold else 0
        else:
            label = 0

        labels.append(label)
        if label == 1:
            n_positive += 1
        else:
            n_negative += 1

    windows = np.array(windows, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    logger.info("Windows: %d total | %d positive (crash) | %d negative (normal)",
                len(windows), n_positive, n_negative)
    logger.info("Positive rate: %.4f%%", n_positive / max(1, len(windows)) * 100)

    return windows, labels


def balance_windows(windows: np.ndarray, labels: np.ndarray, max_ratio: float = 5.0) -> tuple[np.ndarray, np.ndarray]:
    """Balance positive/negative windows by subsampling negatives.

    Keeps all positives. Subsamples negatives to at most max_ratio × positives.
    """
    n_pos = np.sum(labels == 1)
    n_neg = np.sum(labels == 0)

    if n_pos == 0:
        logger.warning("No positive windows found — cannot balance. Returning all negatives.")
        return windows, labels

    max_neg = int(n_pos * max_ratio)
    if n_neg <= max_neg:
        logger.info("Already balanced enough (pos=%d, neg=%d). No subsampling needed.", n_pos, n_neg)
        return windows, labels

    # Subsample negatives
    neg_indices = np.where(labels == 0)[0]
    pos_indices = np.where(labels == 1)[0]
    selected_neg = np.random.choice(neg_indices, size=max_neg, replace=False)

    all_indices = np.concatenate([pos_indices, selected_neg])
    np.random.shuffle(all_indices)

    balanced_windows = windows[all_indices]
    balanced_labels = labels[all_indices]

    logger.info("Balanced: %d pos + %d neg = %d total (ratio %.1f:1)",
                n_pos, max_neg, len(balanced_windows), max_ratio)
    return balanced_windows, balanced_labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract sliding windows with crash labels")
    parser.add_argument("--data", required=True, help="Parquet file")
    parser.add_argument("--out", default="data/windows/", help="Output directory")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    parser.add_argument("--stride", type=int, default=STRIDE)
    parser.add_argument("--lookahead-ms", type=int, default=LOOKAHEAD_MS)
    parser.add_argument("--crash-threshold", type=float, default=CRASH_THRESHOLD_PCT)
    parser.add_argument("--max-ticks", type=int, default=500_000,
                        help="Max ticks to process (0 = all)")
    parser.add_argument("--balance-ratio", type=float, default=5.0,
                        help="Max neg:pos ratio (subsampling)")
    parser.add_argument("--name", default=None,
                        help="Output filename (default: based on input)")
    args = parser.parse_args()

    # Load data
    df = load_parquet(args.data)
    logger.info("Loaded %d ticks from %s", len(df), args.data)

    # Extract features
    features, mid_prices, timestamps = extract_features_from_df(df, max_ticks=args.max_ticks)
    logger.info("Feature matrix: %s", features.shape)

    # Build windows with wall-clock labels (RSR-02)
    windows, labels = label_windows(
        features, mid_prices,
        window_size=args.window_size,
        stride=args.stride,
        lookahead_ms=args.lookahead_ms,
        crash_threshold=args.crash_threshold,
        timestamps=timestamps,
    )

    # Balance
    windows, labels = balance_windows(windows, labels, max_ratio=args.balance_ratio)

    # Save
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.name:
        out_path = out_dir / f"{args.name}.npz"
    else:
        stem = Path(args.data).stem
        out_path = out_dir / f"{stem}_windows.npz"

    np.savez_compressed(
        out_path,
        windows=windows,
        labels=labels,
        feature_names=np.array(TCN_FEATURES),
        config=np.array({
            "window_size": args.window_size,
            "stride": args.stride,
            "lookahead_ms": args.lookahead_ms,
            "crash_threshold": args.crash_threshold,
        }),
    )
    logger.info("Saved %d windows to %s (%.1f MB)",
                len(windows), out_path, out_path.stat().st_size / 1e6)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
