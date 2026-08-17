#!/usr/bin/env python3
"""ADV-03: per-feature drift check between a reference day and a live/observed day.

Extracts the 17 TCN features on a slice of two parquet days and reports per-feature
PSI, flagging features beyond the MODERATE/DRIFT bands. ADVISORY-ONLY — a drift
banner is a re-validation signal, never an auto-tune.

Usage:
    PYTHONPATH=ml python scripts/drift_check.py \
        --reference data/parquet/<train-day>.parquet --observed data/parquet/<live-day>.parquet \
        --max-ticks 50000 [--threshold 0.25]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.historical_loader import df_to_ticks  # noqa: E402
from flash_crash_watchdog.eval.drift import drift_flags  # noqa: E402
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402

TCN_FEATURES = FEATURE_NAMES[:17]


def features_of(parquet: str, max_ticks: int) -> np.ndarray:
    df = pd.read_parquet(parquet)
    if max_ticks > 0 and len(df) > max_ticks:
        df = df.iloc[: max_ticks]
    extractor = FeatureExtractor()
    F = np.zeros((len(df), 17), dtype=np.float32)
    for i, t in enumerate(df_to_ticks(df, symbol="D")):
        fd = extractor.extract(t)
        F[i] = [float(fd.get(f, 0.0)) or 0.0 for f in TCN_FEATURES]
    return F


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", required=True)
    ap.add_argument("--observed", required=True)
    ap.add_argument("--max-ticks", type=int, default=50000)
    ap.add_argument("--threshold", type=float, default=0.25)
    args = ap.parse_args()

    ref = features_of(args.reference, args.max_ticks)
    obs = features_of(args.observed, args.max_ticks)
    flags = drift_flags(ref, obs, TCN_FEATURES, threshold=args.threshold)
    print(f"reference={Path(args.reference).name}  observed={Path(args.observed).name}  "
          f"n(ref)={len(ref)} n(obs)={len(obs)}  threshold={args.threshold}")
    if not flags:
        print("  no features beyond threshold (STABLE)")
    for f in flags:
        print(f"  {f['band']:8s} {f['feature']:<28s} PSI={f['psi']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())