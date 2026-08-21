#!/usr/bin/env python3
"""AF-1 core-signal MVP: cross-asset correlation-breakdown detection.

Aligns two symbols' mid-prices on a common time grid and measures the rolling
pairwise correlation. Test: on a CRASH day (BTC/ETH 2021-05-19), does the
correlation COLLAPSE around the crash trough vs the calm baseline of the same
day? If yes, a rolling-correlation-drop rule is a viable early-warning signal
that the (disabled) Stage-4 was supposed to provide.

Usage: PYTHONPATH=ml python scripts/correlation_breakdown.py
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "ml")


def load_mid(parquet, start_ts=None, end_ts=None, step=1):
    df = pd.read_parquet(parquet, columns=["timestamp_ms", "mid_price"])
    if start_ts is not None:
        df = df[(df.timestamp_ms >= start_ts) & (df.timestamp_ms <= end_ts)]
    df = df.dropna(subset=["mid_price"])
    # resample to 1s grid via mean mid per second
    df["sec"] = df.timestamp_ms // 1000
    s = df.groupby("sec").mid_price.mean()
    return s


def main() -> int:
    btc = load_mid("data/parquet/BTCUSDT_2021-05-19.parquet")
    eth = load_mid("data/parquet/ETHUSDT_2021-05-19.parquet")
    # inner join on the 1s grid
    both = pd.concat([btc, eth], axis=1, keys=["btc", "eth"]).dropna()
    b = both.btc; e = both.eth
    print("aligned 1s samples:", len(b))
    print("BTC day range %.2f -> %.2f | ETH %.2f -> %.2f"
          % (b.min(), b.max(), e.min(), e.max()))
    # crash trough in this slice
    trough = int(e.idxmin())
    print("ETH trough at sec", trough)

    # rolling correlation (log-returns over a ~120s window)
    rb = np.log(b / b.shift(1)); re = np.log(e / e.shift(1))
    corr = rb.rolling(120).corr(re)
    # z-score vs the day's calm (pre-crash) baseline
    pre = corr.loc[: trough - 600].dropna()
    calm_mean, calm_std = pre.mean(), pre.std()
    print("pre-crash (calm) rolling-corr: mean=%.3f std=%.3f" % (calm_mean, calm_std))
    crash_win = corr.loc[trough - 300: trough + 60].dropna()
    if len(crash_win) and calm_std > 1e-6:
        z = (crash_win.mean() - calm_mean) / calm_std
        print("crash-window rolling-corr mean=%.3f  (z=%.2f vs calm)" % (crash_win.mean(), z))
        print("correlation COLLAPSE" if z <= -2.0 else
              ("correlation weakens (z=%.1f)" % z))
    else:
        print("insufficient data for crash-window corr")
    # how early does the corr drop? find first time corr < calm_mean - 2*std
    thr = calm_mean - 2.0 * calm_std
    below = corr[corr < thr]
    if len(below):
        first = below.index.min()
        print("corr first < calm-2std at sec %d (%.0f s before ETH trough)" % (first, trough - first))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
