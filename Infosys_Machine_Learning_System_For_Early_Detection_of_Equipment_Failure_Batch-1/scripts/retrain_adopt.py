#!/usr/bin/env python3
"""ADV-05/06 retrain: train a new Stage-3 TCN on the CORRECTED pipeline.

Adopts the improved order-flow features (ADV-05: real cancel-to-trade proxy +
realized-spread proxy) and optional market-context features (ADV-06, --context),
with the corrected training path:
  - wall-clock lookahead labels (RSR-02),
  - rolling-z normalization window 500 (BUG-03, matches the live Stage3TCN.feed),
  - deterministic seed (RSR-09) + provenance stamp.

The new checkpoint is a SEPARATE file (default stage3_tcn_v2.pt) — the operating
prod model is untouched until this is validated and promoted.

Usage (GPU):
    PYTHONPATH=ml python scripts/retrain_adopt.py \
        --data data/parquet/LUNAUSDT_2022-05-11.parquet \
        --normal data/parquet/BTCUSDT_2021-05-18.parquet \
        --out models/stage3_tcn_v2.pt --epochs 20 --device cuda
"""
import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
SCRIPTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS_DIR))

from flash_crash_watchdog.data.historical_loader import df_to_ticks  # noqa: E402
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from flash_crash_watchdog.models.stage3_tcn import normalize_z, TCNDetector, TCNConfig  # noqa: E402
from flash_crash_watchdog.data.windows import build_windows_from_df  # noqa: E402

logger = logging.getLogger(__name__)
TCN_FEATURES = FEATURE_NAMES[:17]
W = 200
NORM_WINDOW = 500


def build_corrected_windows(parquet: str, max_ticks: int, slice_spec: str | None):
    """Raw features -> rolling-z(500) -> wall-clock-labeled windows (RSR-02)."""
    df = pd.read_parquet(parquet)
    if slice_spec:
        start, end = (int(x) for x in slice_spec.split(":"))
        df = df.iloc[start:end]
    elif max_ticks > 0 and len(df) > max_ticks:
        df = df.iloc[: max_ticks]
    df = df.reset_index(drop=True)
    ticks = list(df_to_ticks(df, symbol="R"))
    extractor = FeatureExtractor()
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    for i, t in enumerate(ticks):
        fd = extractor.extract(t)
        F[i] = [float(fd.get(f, 0.0)) or 0.0 for f in TCN_FEATURES]
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    norm = normalize_z(F, NORM_WINDOW)  # BUG-03: shared rolling-z, window 500

    mids = np.array([t.book.mid_price or 0.0 for t in ticks], dtype=np.float64)
    times = np.array([t.book.timestamp_ms for t in ticks], dtype=np.int64)
    starts = np.arange(0, len(norm) - W, 10)
    windows = np.stack([norm[s:s + W] for s in starts])
    labels = np.zeros(len(starts), dtype=np.int64)
    for j, s in enumerate(starts):
        e = s + W - 1
        t_h = times[e] + 5000  # RSR-02: wall-clock lookahead (5s)
        j_end = int(np.searchsorted(times, t_h, side="right"))
        fut = mids[e + 1: j_end] if j_end > e + 1 else np.array([])
        cur = mids[e]
        if cur > 0 and fut.size:
            drop = (cur - fut.min()) / cur * 100.0
            labels[j] = 1 if drop >= 2.0 else 0
    return windows.astype(np.float32), labels, len(ticks)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="crash training days: comma list of parquet files")
    ap.add_argument("--slices", default=None,
                    help="comma list 'file-or-date:start:end' for crash-region slices, e.g. "
                         "'LUNAUSDT_2022-05-11.parquet:5600000:5650000,ETHUSDT_2021-05-19.parquet:2700000:2730000'")
    ap.add_argument("--normal", default=None, help="normal day parquet(s), comma list (adds negatives)")
    ap.add_argument("--out", default="models/stage3_tcn_v2.pt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-pos", type=int, default=40000, help="cap positives")
    ap.add_argument("--max-neg-per-pos", type=float, default=3.0)
    ap.add_argument("--normal-max-ticks", type=int, default=200_000,
                    help="head cap per normal day (they are negatives; no need for the full day)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

    dev = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    # parse slices: {basename: (start, end)}
    slices = {}
    if args.slices:
        for spec in args.slices.split(","):
            f, s, e = spec.strip().split(":")
            slices[f] = (int(s), int(e))

    data_files = [f.strip() for f in args.data.split(",")]
    normal_files = [f.strip() for f in (args.normal or "").split(",") if f.strip()]

    wins, labs, n_ticks = [], [], 0
    for f in data_files:
        sl = slices.get(f)
        w, l, n = build_corrected_windows(f, 0, (f"{sl[0]}:{sl[1]}" if sl else None))
        wins.append(w); labs.append(l); n_ticks += n
        logger.info("crash day %s: %d windows, %d pos%s", f, len(w), int(l.sum()),
                    f" slice[{sl[0]}:{sl[1]}]" if sl else "")
    for f in normal_files:
        w, l, n = build_corrected_windows(f, args.normal_max_ticks, None)
        l = np.zeros(len(l), dtype=np.int64)
        wins.append(w); labs.append(l); n_ticks += n
        logger.info("normal day %s: %d windows (negatives, head %d ticks)", f, len(w), args.normal_max_ticks)

    wins = np.concatenate(wins); labs = np.concatenate(labs)
    logger.info("Total %d windows (pos=%d) from %d source ticks on %s",
                len(wins), int(labs.sum()), n_ticks, dev)

    # balance: cap positives + subsample negatives
    pos_idx = np.where(labs == 1)[0]
    neg_idx = np.where(labs == 0)[0]
    rng = np.random.default_rng(args.seed)
    if len(pos_idx) > args.max_pos:
        pos_idx = rng.choice(pos_idx, args.max_pos, replace=False)
    keep_neg = min(len(neg_idx), int(len(pos_idx) * args.max_neg_per_pos))
    neg_idx = rng.choice(neg_idx, keep_neg, replace=False)
    idx = np.sort(np.concatenate([pos_idx, neg_idx]))
    wins, labs = wins[idx], labs[idx]
    logger.info("After balance: %d windows (pos=%d)", len(wins), int(labs.sum()))

    from train_tcn_windows import train_tcn
    model = train_tcn(wins, labs, epochs=args.epochs, batch_size=args.batch,
                      learning_rate=args.lr, channels=256, device=dev, seed=args.seed)
    out = Path(args.out)
    torch.save({
        "model_state": model.state_dict(),
        "config": model.config,
        "provenance": {
            "trainer": "retrain_adopt.py", "seed": args.seed,
            "torch_version": str(torch.__version__),
            "source": args.data, "label_mode": "wall-clock", "normalize": "rolling-z-500",
            "features": "adopt-ADV05(ctr,realized_spread)",
        },
    }, out)
    logger.info("Saved adopted checkpoint -> %s (pos=%d)", out, int(labs.sum()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())