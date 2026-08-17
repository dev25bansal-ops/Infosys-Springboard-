#!/usr/bin/env python3
"""AF-1: correlation-breakdown backtest, parametric on a (anchor, basket) pair.

Drives the real CorrelationBreakdown feature on aligned, resampled mid-price
grids. Validated scenarios:
  - LUNA-vs-BTC 2022-05-11 (collapse): corr should BREAK (LUNA->0, BTC flat).
  - LUNA-vs-BTC 2022-05-09 (calm pre): corr should stay high (no breakdown).
  - BTC-vs-ETH  2021-05-19 (market-wide crash): corr stays HIGH (both crashed).

Usage:
    PYTHONPATH=ml python scripts/run_correlation_backtest.py \
        --a LUNAUSDT --a-parq data/parquet/LUNAUSDT_2022-05-11.parquet \
        --b BTCUSDT --b-parq data/parquet/BTCUSDT_2022-05-11.parquet \
        --bin-s 60 --corr-s 7200 --floor 0.4 --sustain 180 --cooldown 600
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "ml")

from flash_crash_watchdog.features.correlation import CorrelationBreakdown, CorrelationConfig  # noqa


def load_binned(parquet: str, bin_s: int) -> pd.Series:
    df = pd.read_parquet(parquet, columns=["timestamp_ms", "mid_price"]).dropna(subset=["mid_price"])
    df["bin"] = df.timestamp_ms // (bin_s * 1000)
    return df.groupby("bin").mid_price.mean()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="LUNAUSDT")
    ap.add_argument("--a-parq", required=True)
    ap.add_argument("--b", default="BTCUSDT")
    ap.add_argument("--b-parq", required=True)
    ap.add_argument("--bin-s", type=int, default=60)
    ap.add_argument("--corr-s", type=int, default=7200, help="rolling corr window (seconds)")
    ap.add_argument("--floor", type=float, default=0.4)
    ap.add_argument("--sustain", type=int, default=180, help="seconds below floor to fire")
    ap.add_argument("--cooldown", type=int, default=600)
    ap.add_argument("--label", default="day")
    args = ap.parse_args()

    a = load_binned(args.a_parq, args.bin_s)
    b = load_binned(args.b_parq, args.bin_s)
    both = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna().sort_index()
    if len(both) < 200:
        print("insufficient aligned samples:", len(both))
        return 1

    win_bins = max(2, args.corr_s // args.bin_s)
    sustain_bins = max(1, args.sustain // args.bin_s)
    cooldown_bins = max(1, args.cooldown // args.bin_s)
    corr = CorrelationBreakdown(CorrelationConfig(
        anchor=args.a, corr_window_bins=win_bins, baseline_bins=max(win_bins * 4, 600),
        warmup_bins=win_bins * 2, collapse_z=2.0,
        floor_corr=args.floor, sustain_s=sustain_bins))
    # NOTE: module's sustain is in bins here (fed at bin cadence).

    alerts = []
    last = -10 ** 9
    trough_bin = int(both.a.idxmin())
    n = len(both)
    for i, (bin_, row) in enumerate(both.iterrows()):
        corr.update(args.a, float(row.a), int(bin_) * args.bin_s * 1000)
        corr.update(args.b, float(row.b), int(bin_) * args.bin_s * 1000)
        z, score, fire = corr.evaluate()
        if fire and (i - last) > cooldown_bins:
            last = i
            alerts.append(int(bin_))
        if i % 5000 == 0 and i:
            sys.stderr.write(".")  # progress dots
    sys.stderr.write("\n")

    first = alerts[0] if alerts else None
    lead = (trough_bin - first) * args.bin_s if first is not None else None
    print("[%s] bins=%d  corr-win=%ds floor=%.2f sustain=%ds cooldown=%ds" % (
        args.label, n, args.corr_s, args.floor, args.sustain, args.cooldown))
    print("  %s-vs-%s correlation-breakdown alerts=%d  first=bin%d  lead_to_trough=%s"
          % (args.a, args.b, len(alerts), first if first is not None else -1,
             ("%.0f s" % lead) if lead is not None else "n/a"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())