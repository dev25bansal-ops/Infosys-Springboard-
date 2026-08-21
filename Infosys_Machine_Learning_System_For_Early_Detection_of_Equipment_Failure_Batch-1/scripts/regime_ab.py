#!/usr/bin/env python3
"""ADV-08: honest A/B — a 2-state volatility-regime model vs the fixed trailing-vol gate.

The operating point uses a fixed gate: alert only when trailing realized-vol
(tv) >= 2 bps. This compares that gate against a 2-state regime model — an EM
Gaussian-mixture on log(trailing_vol) that learns LOW/HIGH vol regimes per day —
on the SAME crash + normal days, reporting:

    crash-day: fraction of crash-window ticks the regime flags as HIGH-vol
               (the gate-equivalent "recall"), and
    normal-day: fraction of ticks flagged (calm-day quiet; lower is better).

A regime model only wins if it keeps crash recall while reducing normal-day
chatter. Dependency-free (numpy EM), advisory only.

Usage:
    PYTHONPATH=ml python scripts/regime_ab.py --crash data/parquet/BTCUSDT_2021-05-19.parquet \
        --normal data/parquet/BTCUSDT_2024-01-16.parquet --max-ticks 50000
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.historical_loader import df_to_ticks  # noqa: E402
from flash_crash_watchdog.data.labels import label_crashes  # noqa: E402

W = 200
GATE_BPS = 2.0


def trailing_vol_bps(mid: np.ndarray) -> np.ndarray:
    out = np.zeros(len(mid))
    for i in range(W - 1, len(mid)):
        seg = mid[i - W + 1:i + 1]
        m = seg.mean()
        out[i] = (seg.std() / m) * 10000.0 if m > 0 else 0.0
    return out


def gmm2_fit(logv: np.ndarray, iters: int = 100):
    """2-state Gaussian mixture on log-vol via EM. Returns (weights, means, stds)."""
    rng = np.random.default_rng(0)
    w = np.array([0.5, 0.5])
    m = np.array([np.quantile(logv, 0.25), np.quantile(logv, 0.75)])
    s = np.array([logv.std() / 2, logv.std() / 2]) + 1e-3
    for _ in range(iters):
        # E
        p0 = w[0] * np.exp(-0.5 * ((logv - m[0]) / s[0]) ** 2) / (s[0] * np.sqrt(2 * np.pi))
        p1 = w[1] * np.exp(-0.5 * ((logv - m[1]) / s[1]) ** 2) / (s[1] * np.sqrt(2 * np.pi))
        denom = p0 + p1 + 1e-12
        r0, r1 = p0 / denom, p1 / denom
        # M
        n0, n1 = r0.sum(), r1.sum()
        w = np.array([n0, n1]) / (n0 + n1)
        m[0] = (r0 * logv).sum() / (n0 + 1e-12)
        m[1] = (r1 * logv).sum() / (n1 + 1e-12)
        s[0] = np.sqrt((r0 * (logv - m[0]) ** 2).sum() / (n0 + 1e-12)) + 1e-3
        s[1] = np.sqrt((r1 * (logv - m[1]) ** 2).sum() / (n1 + 1e-12)) + 1e-3
    hi = int(np.argmax(m))  # the HIGH-vol state is the larger mean
    return w[hi], m[hi], s[hi]


def evaluate_day(parquet: str, max_ticks: int, threshold: float) -> dict:
    df = pd.read_parquet(parquet)
    if max_ticks > 0 and len(df) > max_ticks:
        df = df.iloc[: max_ticks]
    ticks = list(df_to_ticks(df, symbol="AB"))
    mid = np.array([t.book.mid_price or 0.0 for t in ticks])
    tv = trailing_vol_bps(mid)
    crashes = label_crashes(ticks, drop_threshold_pct=2.0, window_ms=60000)
    crash_tick = np.zeros(len(ticks), dtype=bool)
    for c in crashes:
        for i in range(len(ticks)):
            if c.start_ts <= ticks[i].timestamp_ms <= c.end_ts:
                crash_tick[i] = True

    active = tv > 0
    logv = np.log(np.clip(tv[active], 1e-6, None))
    w_hi, m_hi, s_hi = gmm2_fit(logv)
    # regime HIGH if the high-vol state posterior > 0.5
    z = (logv - m_hi) / s_hi
    reg_hi = np.zeros(len(ticks), dtype=bool)
    reg_hi[active] = z > 0  # within the fitted high-vol state's positive side
    gate_hi = tv >= threshold

    mask = np.arange(len(ticks)) >= W - 1
    return {
        "ticks": len(ticks),
        "n_crash_ticks": int(crash_tick.sum()),
        "gate_crash_recall": float(crash_tick[mask & gate_hi].sum() / max(1, crash_tick[mask].sum())),
        "regime_crash_recall": float(crash_tick[mask & reg_hi].sum() / max(1, crash_tick[mask].sum())),
        "gate_calm_frac": float(gate_hi[~crash_tick & mask].mean()),
        "regime_calm_frac": float(reg_hi[~crash_tick & mask].mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crash", required=True)
    ap.add_argument("--normal", required=True)
    ap.add_argument("--max-ticks", type=int, default=50000)
    args = ap.parse_args()

    crash = evaluate_day(args.crash, args.max_ticks, GATE_BPS)
    normal = evaluate_day(args.normal, args.max_ticks, GATE_BPS)
    print(f"{'metric':<28}{'gate':>10}{'regime':>10}")
    print(f"{'crash-day recall':<28}{crash['gate_crash_recall']:>10.3f}{crash['regime_crash_recall']:>10.3f}")
    print(f"{'normal-day flagged frac':<28}{normal['gate_calm_frac']:>10.4f}{normal['regime_calm_frac']:>10.4f}")
    print(f"(crash day n_crash_ticks={crash['n_crash_ticks']}, ticks={crash['ticks']}; "
          f"normal ticks={normal['ticks']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())