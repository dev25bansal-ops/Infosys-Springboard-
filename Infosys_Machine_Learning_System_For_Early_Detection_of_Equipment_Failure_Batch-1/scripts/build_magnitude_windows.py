#!/usr/bin/env python3
"""Build magnitude-labeled windows for the Stage-3 magnitude head.

For each contiguous 200-tick window, emit:
    label = (mid[end] - min(mid[end+1 .. end+lookahead])) / mid[end] * 100
clipped to [0, clip] and scaled by /scale so the range is ~[0,1].

Outputs an .npz with windows (N, 200, 17), labels (N,) in [0,1],
plus the config used. Reuses the detector feature pipeline.
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))  # for sibling imports like magnitude_config

from flash_crash_watchdog.data.historical_loader import df_to_ticks  # noqa: E402
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402

# RSR-19: single source of truth for the magnitude label constants.
from magnitude_config import CLIP_PCT as DEFAULT_CLIP, SCALE_PCT as DEFAULT_SCALE, POSITIVE_FLOOR_PCT  # noqa: E402

logger = logging.getLogger(__name__)
TCN_FEATURES = FEATURE_NAMES[:17]
W = 200
NORM_WINDOW = 500
DEFAULT_LOOKAHEAD = 500


def build_from_parquet(parquet: Path, lookahead: int, clip: float, scale: float,
                       stride: int, max_ticks: int):
    df = pd.read_parquet(parquet)
    if max_ticks > 0 and len(df) > max_ticks:
        df = df.iloc[:max_ticks]
    ticks = list(df_to_ticks(df, symbol="MAG"))
    ext = FeatureExtractor()
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    mid = np.full(len(ticks), np.nan, dtype=np.float64)
    for i, t in enumerate(ticks):
        fd = ext.extract(t)
        F[i] = [float(fd.get(k, 0.0)) or 0.0 for k in TCN_FEATURES]
        mid[i] = t.book.mid_price if t.book.mid_price is not None else np.nan
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    mid = pd.Series(mid).ffill().bfill().to_numpy()

    # rolling-z normalization (window 500) matching the Stage-3 inference path
    pdf = pd.DataFrame(F)
    mean = pdf.rolling(NORM_WINDOW, min_periods=1).mean()
    std = pdf.rolling(NORM_WINDOW, min_periods=1).std()
    norm = ((pdf - mean) / std).where(std.abs() > 1e-8, 0.0).fillna(0.0).to_numpy(np.float32)

    n = len(norm)
    starts = np.arange(0, max(0, n - W - lookahead + 1), stride)
    X = np.stack([norm[s:s + W] for s in starts])
    end = starts + W - 1
    fut_min = np.empty(len(end), dtype=np.float64)
    for j, e in enumerate(end):
        seg = mid[e + 1:e + 1 + lookahead]
        fut_min[j] = float(seg.min()) if len(seg) >= max(10, lookahead // 5) else mid[e]
    cur = mid[end]
    fwd = np.where(cur > 0, (cur - fut_min) / cur * 100.0, 0.0)
    fwd = np.clip(fwd, 0.0, clip) / scale  # -> [0,1]

    logger.info("%s: %d windows (%.2f%% with forward-drop >=0.1%% → label>=%.3f)",
                Path(parquet).name, len(X), 100.0 * np.mean(fwd >= 0.1 / scale), 0.1 / scale)
    return X, fwd.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="tick parquet")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--lookahead", type=int, default=DEFAULT_LOOKAHEAD)
    ap.add_argument("--clip", type=float, default=DEFAULT_CLIP)
    ap.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-ticks", type=int, default=0)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    X, y = build_from_parquet(Path(args.data), args.lookahead, args.clip, args.scale,
                              args.stride, args.max_ticks)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, windows=X, labels=y,
        feature_names=np.array(TCN_FEATURES),
        config=np.array({"window": W, "lookahead": args.lookahead, "clip": args.clip,
                         "scale": args.scale, "norm_window": NORM_WINDOW}),
    )
    logger.info("Saved %d magnitude windows to %s", len(X), out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
