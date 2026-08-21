"""Build / load labeled (window, label) training sets for the TCN.

This is the canonical way to turn raw tick data (parquet/csv) or an already
extracted ``.npz`` window file into the ``(N, window, features)`` tensors plus
binary crash labels that ``TCNDetector.train_on_windows`` (stage3_tcn) expects.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor
from flash_crash_watchdog.data.historical_loader import df_to_ticks

logger = logging.getLogger(__name__)

# 17 features consumed by the TCN (F1-F4), matching STAGE3_FEATURES.
TCN_FEATURES = FEATURE_NAMES[:17]

WINDOW_SIZE = 200   # ticks per window
STRIDE = 10         # sliding-window stride (overlap 190)
LOOKAHEAD_MS = 5000  # label: crash if price drops >= threshold within next 5s
CRASH_THRESHOLD_PCT = 2.0


def build_windows_from_df(
    df: pd.DataFrame,
    window_size: int = WINDOW_SIZE,
    stride: int = STRIDE,
    lookahead_ms: int = LOOKAHEAD_MS,
    crash_pct: float = CRASH_THRESHOLD_PCT,
    max_ticks: int = 0,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract per-tick features + mid prices and build labeled sliding windows.

    Args:
        df: detector-schema tick frame (timestamp_ms, best_bid/ask, bid/ask_size,
            mid_price, trade_*), sorted ascending by timestamp_ms. The crash label
            is defined over the *wall-clock* lookahead_ms after each window's last
            tick (RSR-02), so the timestamps must be present and monotonic.
        max_ticks: if >0, sample the frame EVENLY to at most this many rows
            (kept for a hard cap; note even sampling is fine for window building
            only if you accept the coarse timeline).

    Returns:
        (windows, labels, feature_names): windows (N, window_size, 17),
        labels (N,) int 0/1, and the 17 feature names used.
    """
    n = len(df)
    if max_ticks > 0 and n > max_ticks:
        idx = np.linspace(0, n - 1, max_ticks, dtype=int)
        df = df.iloc[idx].copy()
        n = len(df)  # re-bind to the sampled length used by the window loop below
        logger.info("Sampled window source to %d ticks (evenly spaced)", n)

    extractor = FeatureExtractor()
    feats: List[np.ndarray] = []
    mids: List[float] = []
    for i, tick in enumerate(df_to_ticks(df, symbol="WINDOW")):
        info = extractor.extract(tick)
        feats.append(np.array([info.get(f, 0.0) for f in TCN_FEATURES], dtype=np.float32))
        mids.append(float(tick.book.mid_price) if tick.book.mid_price else 0.0)
        if i % 200000 == 0:
            logger.info("  features: %d/%d", i, len(df))

    features = np.asarray(feats, dtype=np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    mids = np.asarray(mids, dtype=np.float64)

    # Wall-clock lookahead (RSR-02): the crash label horizon is a *real time*
    # window (next lookahead_ms), not a fixed tick count. The old
    # `lookahead_ms // 100` assumed 100ms/tick (10/s), but real tick rates are
    # 8-42/s, so it turned the "2% in 5s" label into ~1.2-1.8s on fast crash days
    # and ~6.5s on slow days. Find the future by timestamp instead.
    if "timestamp_ms" in df.columns:
        timestamps = df["timestamp_ms"].to_numpy(dtype=np.float64)
        if len(timestamps) != n or not np.all(np.diff(timestamps) >= 0):
            raise ValueError(
                "build_windows_from_df requires df sorted by timestamp_ms "
                "(len(timestamps) == len(df), monotonic non-decreasing)"
            )
    else:
        timestamps = None
        logger.warning(
            "No timestamp_ms column; falling back to the tick-count lookahead "
            "heuristic (lookahead_ms // 100). Prefer wall-clock labels (RSR-02)."
        )

    windows: List[np.ndarray] = []
    labels: List[int] = []
    for i in range(0, n - window_size, stride):
        end_idx = i + window_size - 1
        cur = mids[end_idx]
        if timestamps is not None:
            # Last index whose timestamp is within the lookahead window of the
            # window-end tick. Only emit windows whose horizon is fully covered.
            t_horizon = timestamps[end_idx] + float(lookahead_ms)
            j = int(np.searchsorted(timestamps, t_horizon, side="right")) - 1
            if j < end_idx or timestamps[j] < t_horizon:
                continue  # not enough future data to define the label
            future = mids[end_idx + 1 : j + 1]
        else:
            lookahead_ticks = max(1, lookahead_ms // 100)  # legacy heuristic
            if end_idx + 1 + lookahead_ticks > n:
                continue
            future = mids[end_idx + 1 : end_idx + 1 + lookahead_ticks]
        windows.append(features[i : i + window_size])
        if cur > 0 and len(future) > 0:
            drop_pct = (cur - float(np.min(future))) / cur * 100.0
        else:
            drop_pct = 0.0
        labels.append(1 if drop_pct >= crash_pct else 0)

    windows_arr = np.asarray(windows, dtype=np.float32)
    labels_arr = np.asarray(labels, dtype=np.int32)
    n_pos = int(labels_arr.sum())
    logger.info("Windows: %d total | %d positive (%.2f%%)",
                len(windows_arr), n_pos, (n_pos / max(1, len(windows_arr))) * 100)
    return windows_arr, labels_arr, list(TCN_FEATURES)


def load_windows(paths: List[Path]) -> Tuple[np.ndarray, np.ndarray]:
    """Load + concat labeled windows from one or more .npz files.

    The .npz files must contain keys ``windows`` (N, window_size, 17) and
    ``labels`` (N,).
    """
    all_w = []
    all_l = []
    for p in paths:
        p = Path(p)
        if not p.exists():
            raise FileNotFoundError(f"Windows file not found: {p}")
        d = np.load(p, allow_pickle=True)  # project-local .npz of plain arrays (windows/labels) produced by extract_windows.py
        w, l = d["windows"], d["labels"]
        all_w.append(w)
        all_l.append(l)
        logger.info("Loaded %s: %d windows (%d positive)", p, len(w), int(np.sum(l)))
    windows = np.concatenate(all_w, axis=0)
    labels = np.concatenate(all_l, axis=0)
    logger.info("Total: %d windows (%d positive = %.1f%%)",
                len(windows), int(np.sum(labels)), float(np.sum(labels)) / max(1, len(labels)) * 100)
    return windows, labels


def resolve_windows_source(data: str | Path, max_ticks: int = 0) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Make a ``(windows, labels, feature_names)`` set from a parquet/csv or npz.

    - A ``.npz`` file -> loaded directly.
    - A ``.parquet``/``.csv`` -> windows are built on the fly (build_windows_from_df).
    - A directory  -> first non-empty .npz found (or built from parquet files).
    """
    data = Path(data)
    if data.is_dir():
        zips = sorted(data.glob("*.npz"))
        if zips:
            return (*load_windows(zips), None)  # feature_names unknown for npz
        parqs = sorted(data.glob("*.parquet"))
        if parqs:
            frames = [pd.read_parquet(p) for p in parqs]
            df = pd.concat(frames, ignore_index=True)
            return build_windows_from_df(df, max_ticks=max_ticks)
        raise FileNotFoundError(f"No .npz or .parquet found in {data}")
    if data.suffix == ".npz":
        w, l = load_windows([data])
        return w, l, None
    df = pd.read_parquet(data) if data.suffix == ".parquet" else pd.read_csv(data)
    return build_windows_from_df(df, max_ticks=max_ticks)