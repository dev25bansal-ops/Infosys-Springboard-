#!/usr/bin/env python3
"""Build CONTIGUOUS labeled windows from a tick parquet.

Fixes the non-contiguous sampling bug in extract_windows.py: that script
even-samples the day via --max-ticks (np.linspace), so its "200-tick" training
windows are NOT temporally contiguous and don't match the contiguous 200-tick
windows the cascade/Stage3TCN.score builds at inference. This script walks
every tick in order, extracts the 17-dim TCN features, and builds true
contiguous windows with the same crash label (>=2% mid-price drop within the
next ``lookahead`` ticks).

The per-tick feature matrix is cached (default ``<data>_features.npz``) so the
expensive extraction pass runs once; window params can be retuned cheaply after.

Usage:
    python scripts/extract_windows_contiguous.py \
        --data data/parquet/BTCUSDT_2021-05-19.parquet \
        --out data/windows/BTCUSDT_2021-05-19_contig.npz \
        --stride 10 --max-pos 20000 --max-neg-per-pos 4
"""
import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.historical_loader import df_to_ticks
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor

logger = logging.getLogger(__name__)

TCN_FEATURES = FEATURE_NAMES[:17]


def normalize_features(features: np.ndarray, window: int = 2000) -> np.ndarray:
    """Rolling z-score standardization (streaming-compatible, scale-invariant).

    The raw 17-dim vector mixes huge price-level features (f1_micro_price ~ 1e4-1e5)
    with small dimensionless ones; unnormalized inputs make the TCN saturate to
    all-0 or all-1. Rolling (mean, std) over the last ``window`` ticks removes the
    scale problem while remaining computable online. Constant features (std ~ 0)
    map to 0.
    """
    df = pd.DataFrame(features)
    mean = df.rolling(window, min_periods=1).mean()
    std = df.rolling(window, min_periods=1).std()
    norm = (df - mean) / std
    norm = norm.where(std.abs() > 1e-8, 0.0).fillna(0.0).to_numpy(np.float32)
    return norm


def extract_features(df: pd.DataFrame, cache_path: Path, max_ticks: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Extract the 17-dim TCN feature matrix + mid prices over CONTIGUOUS ticks.

    Caches to cache_path so repeat runs skip the ~10+ min extraction.
    """
    if cache_path.exists():
        d = np.load(cache_path)  # trusted local cache (plain arrays)
        logger.info("Loaded cached features %s from %s", d["features"].shape, cache_path)
        return d["features"], d["mids"]

    if max_ticks > 0 and len(df) > max_ticks:
        idx = np.linspace(0, len(df) - 1, max_ticks, dtype=int)
        df = df.iloc[idx]

    extractor = FeatureExtractor()
    feats: list = []
    mids: list = []
    t0 = time.time()
    for i, tick in enumerate(df_to_ticks(df, symbol="EXTRACT")):
        f = extractor.extract(tick)
        feats.append([f.get(k, 0.0) for k in TCN_FEATURES])
        mids.append(float(tick.book.mid_price) if tick.book.mid_price else 0.0)
        if i and i % 500_000 == 0:
            rate = (i + 1) / max(1e-9, time.time() - t0)
            logger.info("  features %d/%d (%.0f ticks/s)", i, len(df), rate)

    features = np.asarray(feats, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    mids = np.asarray(mids, dtype=np.float64)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, features=features, mids=mids)
    logger.info("Saved feature cache %s (%s) in %.1fs", cache_path, features.shape, time.time() - t0)
    return features, mids


def build_windows(
    features: np.ndarray,
    mids: np.ndarray,
    window: int,
    stride: int,
    lookahead: int,
    crash_pct: float,
    max_pos: int,
    max_neg_per_pos: float,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sliding contiguous windows; keep all positives (capped) + subsampled negatives."""
    n = len(features)
    pos: list = []
    neg: list = []
    for i in range(0, n - window - lookahead, stride):
        cur = mids[i + window - 1]
        future = mids[i + window: i + window + lookahead]
        lab = 1 if (cur > 0 and len(future) > 0 and (cur - future.min()) / cur * 100 >= crash_pct) else 0
        (pos if lab else neg).append(features[i: i + window])
        if (len(pos) + len(neg)) % 200_000 == 0:
            logger.info("  scanned windows %d (pos %d)", len(pos) + len(neg), len(pos))

    pos_arr = np.asarray(pos, dtype=np.float32)
    neg_arr = np.asarray(neg, dtype=np.float32)
    rng = np.random.default_rng(seed)
    if len(pos_arr) == 0:
        # All-negative day (e.g. a normal day) — keep up to max_neg_per_pos*max_pos negatives.
        max_neg = int(max(1, len(neg_arr)))
        if len(neg_arr) > 10_000:
            neg_arr = neg_arr[rng.choice(len(neg_arr), size=10_000, replace=False)]
        windows, labels = neg_arr, np.zeros(len(neg_arr), np.int32)
        logger.info("Windows: %s  pos 0  neg %d (all-negative day)", windows.shape, len(windows))
        return windows, labels
    if len(pos_arr) > max_pos:
        pos_arr = pos_arr[rng.choice(len(pos_arr), size=max_pos, replace=False)]
    max_neg = int(len(pos_arr) * max_neg_per_pos)
    if len(neg_arr) > max_neg:
        neg_arr = neg_arr[rng.choice(len(neg_arr), size=max_neg, replace=False)]

    windows = np.concatenate([pos_arr, neg_arr], axis=0)
    labels = np.concatenate([np.ones(len(pos_arr), np.int32), np.zeros(len(neg_arr), np.int32)])
    perm = rng.permutation(len(windows))
    windows, labels = windows[perm], labels[perm]
    logger.info("Windows: %s  pos %d  neg %d (%.1f:1)",
                windows.shape, len(pos_arr), len(neg_arr), len(neg_arr) / max(1, len(pos_arr)))
    return windows, labels


def main() -> int:
    parser = argparse.ArgumentParser(description="Build contiguous labeled windows")
    parser.add_argument("--data", required=True, help="Tick parquet")
    parser.add_argument("--out", required=True, help="Output .npz")
    parser.add_argument("--cache", default=None, help="Feature cache path (default <data>_features.npz)")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--lookahead", type=int, default=50)
    parser.add_argument("--crash-pct", type=float, default=2.0)
    parser.add_argument("--max-pos", type=int, default=20_000, help="Cap positives kept")
    parser.add_argument("--max-neg-per-pos", type=float, default=4.0)
    parser.add_argument("--normalize", action="store_true",
                        help="Rolling z-score normalize features before windowing")
    parser.add_argument("--norm-window", type=int, default=2000)
    parser.add_argument("--max-ticks", type=int, default=0, help="0 = full day (contiguous)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    df = pd.read_parquet(args.data)
    logger.info("Loaded %d ticks from %s (contiguous)", len(df), args.data)
    stem = Path(args.data).stem
    cache = Path(args.cache) if args.cache else Path(args.data).with_name(f"{stem}_features.npz")

    features, mids = extract_features(df, cache, max_ticks=args.max_ticks)
    if args.normalize:
        features = normalize_features(features, window=args.norm_window)
        logger.info("Features rolling-z normalized (window=%d)", args.norm_window)
    windows, labels = build_windows(
        features, mids,
        window=args.window, stride=args.stride, lookahead=args.lookahead,
        crash_pct=args.crash_pct, max_pos=args.max_pos, max_neg_per_pos=args.max_neg_per_pos,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, windows=windows, labels=labels,
        feature_names=np.array(TCN_FEATURES),
        config=np.array({"window": args.window, "stride": args.stride,
                         "lookahead": args.lookahead, "crash_pct": args.crash_pct}),
    )
    logger.info("Saved %d windows to %s", len(windows), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())