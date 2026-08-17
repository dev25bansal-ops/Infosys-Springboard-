#!/usr/bin/env python3
"""Score a trained TCN on a contiguous tick slice and return a threshold.

Fast (batched GPU) version of the diagnostic. Builds contiguous windows the same
way the cascade does, applies the same rolling-z feature normalization as
training, scores all windows at once, and prints the score distribution plus a
precision/recall/F1 threshold sweep so the operating threshold can be picked.

Usage:
    python scripts/tcn_score_diag.py --data <slice.parquet> \
        --model models/stage3_tcn_oos.pt --lookahead 500 --normalize
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.historical_loader import df_to_ticks
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector
torch.serialization.add_safe_globals([TCNConfig])

logger = logging.getLogger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="TCN score distribution + threshold sweep")
    parser.add_argument("--data", required=True, help="Contiguous tick parquet to score")
    parser.add_argument("--model", default="models/stage3_tcn_trained.pt")
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--lookahead", type=int, default=500,
                        help="Crash-label lookahead (ticks) — should match the model's training label")
    parser.add_argument("--crash-pct", type=float, default=2.0)
    parser.add_argument("--normalize", action="store_true",
                        help="Rolling-z normalize features (same as training)")
    parser.add_argument("--norm-window", type=int, default=500,
                        help="Rolling-z window; MUST match the model's training normalization (models are trained on *_norm500 windows)")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-ticks", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    device = "cuda" if (args.device == "auto" and torch.cuda.is_available()) else args.device

    df = pd.read_parquet(args.data)
    if args.max_ticks > 0 and len(df) > args.max_ticks:
        df = df.iloc[: args.max_ticks]

    extractor = FeatureExtractor()
    tcf = FEATURE_NAMES[:17]
    feats, mids = [], []
    for tick in df_to_ticks(df, symbol="DIAG"):
        f = extractor.extract(tick)
        feats.append([f.get(k, 0.0) for k in tcf])
        mids.append(float(tick.book.mid_price) if tick.book.mid_price else 0.0)
    features = np.nan_to_num(np.asarray(feats, np.float32))
    mids = np.asarray(mids, np.float64)

    if args.normalize:
        fdf = pd.DataFrame(features)
        mean = fdf.rolling(args.norm_window, min_periods=1).mean()
        std = fdf.rolling(args.norm_window, min_periods=1).std()
        features = ((fdf - mean) / std).where(std.abs() > 1e-8, 0.0).fillna(0.0).to_numpy(np.float32)

    W, S, LA = args.window, args.stride, args.lookahead
    win, labs = [], []
    for i in range(0, len(features) - W - LA, S):
        win.append(features[i:i + W])
        cur = mids[i + W - 1]
        fut = mids[i + W:i + W + LA]
        labs.append(1 if cur > 0 and len(fut) and (cur - fut.min()) / cur * 100 >= args.crash_pct else 0)
    if not win:
        logger.error("No windows; need >%d ticks, got %d", W + LA, len(features))
        return 1

    X = torch.from_numpy(np.asarray(win)).permute(0, 2, 1).to(device)
    labs = np.asarray(labs)

    st = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to(device)
    model.load_state_dict(st["model_state"])
    model.eval()
    with torch.no_grad():
        scores = model(X)[:, -1].cpu().numpy()

    print("windows=%d positive=%d" % (len(scores), int(labs.sum())))
    print("score dist: p50=%.3f p90=%.3f p99=%.3f max=%.3f" % (
        np.percentile(scores, 50), np.percentile(scores, 90),
        np.percentile(scores, 99), scores.max()))
    print("threshold sweep (Stage-3 alone):")
    best = (0.5, 0.0)
    for thr in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        pred = scores > thr
        tp = int(((pred) & (labs == 1)).sum())
        fp = int(((pred) & (labs == 0)).sum())
        fn = int(((~pred) & (labs == 1)).sum())
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        print("  thr=%.2f alerts=%d tp=%d fp=%d fn=%d prec=%.3f recall=%.3f f1=%.3f" %
              (thr, tp + fp, tp, fp, fn, prec, rec, f1))
        if f1 >= best[1]:
            best = (thr, f1)
    print("BEST Stage-3 operating threshold: %.2f (F1=%.3f)" % (best[0], best[1]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())