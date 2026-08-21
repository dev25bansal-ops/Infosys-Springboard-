#!/usr/bin/env python3
"""Verify the magnitude head separates normal-day FPs from crash detections.

For each eval day, score every 200-tick window with models/stage3_tcn_magnitude.pt,
interpret the sigmoid output as the SCALED forward drop (x5 => %), and report:
  - how many windows predict a drop >= thr-pct (on a normal day expect ~0),
  - crash-detection recall/precision by matching predicted-drop>=thr windows against
    the 2%-in-60s crash windows (with a 10s cooldown),
  - the Spearman correlation between predicted and actual forward drop.

Usage:
    PYTHONPATH=ml python scripts/eval_magnitude.py --model models/stage3_tcn_magnitude.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.historical_loader import df_to_ticks  # noqa: E402
from flash_crash_watchdog.data.labels import label_crashes  # noqa: E402
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector  # noqa: E402
torch.serialization.add_safe_globals([TCNConfig])

FC = FEATURE_NAMES[:17]
W = 200
LA = 500
NORM_WIN = 500


def load_day(parquet):
    df = pd.read_parquet(parquet)
    ticks = list(df_to_ticks(df, symbol="E"))
    ext = FeatureExtractor()
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    mid = np.full(len(ticks), np.nan, dtype=np.float64)
    ts = np.empty(len(ticks), dtype=np.int64)
    for i, t in enumerate(ticks):
        fd = ext.extract(t)
        F[i] = [float(fd.get(k, 0.0)) or 0.0 for k in FC]
        if t.book.mid_price is not None:
            mid[i] = t.book.mid_price
        ts[i] = t.book.timestamp_ms
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    mid = pd.Series(mid).ffill().bfill().to_numpy()
    pdf = pd.DataFrame(F)
    mean = pdf.rolling(NORM_WIN, min_periods=1).mean()
    std = pdf.rolling(NORM_WIN, min_periods=1).std()
    norm = ((pdf - mean) / std).where(std.abs() > 1e-8, 0.0).fillna(0.0).to_numpy(np.float32)
    return norm, mid, ts, ticks


def score_windows(model, norm, dev):
    n = len(norm)
    out = np.zeros(max(0, n - W + 1), dtype=np.float32)
    if len(out) == 0:
        return out
    for s in range(0, len(out), 4096):
        e = min(s + 4096, len(out))
        win = np.stack([norm[i:i + W] for i in range(s, e)])
        x = torch.from_numpy(np.ascontiguousarray(win)).permute(0, 2, 1).float().to(dev)
        with torch.no_grad():
            out[s:e] = model(x)[:, -1].cpu().numpy()
    return out


def eval_day(model, parquet, dev, thr_pct, cooldown_ms):
    norm, mid, ts, ticks = load_day(parquet)
    pred = score_windows(model, norm, dev) * 5.0  # scaled->percent
    n = len(pred)
    # actual forward min-drop % (for correlation)
    actual = np.full(n, np.nan)
    for j in range(n):
        e = j + W - 1
        seg = mid[e + 1:e + 1 + LA]
        if len(seg) >= 10 and mid[e] > 0:
            actual[j] = (mid[e] - float(seg.min())) / mid[e] * 100.0

    # alerts: windows predicting >= thr_pct, coalesced by cooldown
    alert_ts = []
    last = None
    for j in range(n):
        if pred[j] >= thr_pct:
            e = j + W - 1
            if e >= len(ts):
                continue
            t = ts[e]
            if last is not None and t - last < cooldown_ms:
                continue
            last = t
            alert_ts.append(t)

    crashes = label_crashes(ticks, drop_threshold_pct=2.0)
    ranges = [(c.start_ts, c.end_ts) for c in crashes]
    matched = set(); tp = 0; ttd = []
    for t in alert_ts:
        hit = [i for i, (a, b) in enumerate(ranges) if a <= t <= b]
        if hit:
            tp += 1; matched.add(hit[0]); ttd.append(ranges[hit[0]][1] - t)
    fn = len(crashes) - len(matched)
    fp = len(alert_ts) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    corr = float(np.nan)
    m = ~np.isnan(actual)
    if m.sum() > 1:
        corr = float(np.corrcoef(pred[m], actual[m])[0, 1])
    return {
        "windows": n, "pred_ge_thr": int((pred >= thr_pct).sum()), "alerts": len(alert_ts),
        "tp": tp, "fp": fp, "fn": fn, "precision": prec, "recall": rec, "f1": f1,
        "median_ttd_ms": float(np.median(ttd)) if ttd else 0.0,
        "corr": corr,
        "pred_p10_p50_p90": (round(float(np.nanpercentile(pred[~np.isnan(pred)], 10)), 3),
                             round(float(np.nanpercentile(pred[~np.isnan(pred)], 50)), 3),
                             round(float(np.nanpercentile(pred[~np.isnan(pred)], 90)), 3)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stage3_tcn_magnitude.pt")
    ap.add_argument("--data", action="append",
                    default=["C:/Users/dev25/AppData/Local/Temp/fcw_prod/BTC_0519_val.parquet",
                             "C:/Users/dev25/AppData/Local/Temp/fcw_prod/LUNA_0510_val.parquet",
                             "C:/Users/dev25/AppData/Local/Temp/fcw_prod/BTC_2024_0116_norm.parquet"])
    ap.add_argument("--labels", action="append",
                    default=["BTC-0519(crash)", "LUNA-0510(crash)", "BTC-0116(normal)"])
    ap.add_argument("--thr-pct", type=float, default=1.0)
    ap.add_argument("--cooldown-ms", type=int, default=10000)
    ap.add_argument("--base", default="models/stage3_tcn_prod.pt",
                    help="also score with the binary model to compare pred-vs-actual")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # torch.load(weights_only=True): checkpoint stores a TCNConfig dataclass (trusted).
    st = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to(dev)
    model.load_state_dict(st["model_state"]); model.eval()

    print("=== magnitude head: predicted-drop >= %.1f%% threshold ===" % args.thr_pct)
    print("%-16s %6s %6s %6s %5s %5s | %5s %5s %5s | %7s %10s | corr" % (
        "day", "win", "pred>=", "alerts", "tp", "fp", "prec", "rec", "f1", "ttd_ms", "pred p50"))
    for parquet, label in zip(args.data, args.labels):
        r = eval_day(model, parquet, dev, args.thr_pct, args.cooldown_ms)
        print("%-16s %6d %6d %6d %5d %5d | %5.3f %5.3f %5.3f | %7.0f %7.3f | %+.3f" % (
            label, r["windows"], r["pred_ge_thr"], r["alerts"], r["tp"], r["fp"],
            r["precision"], r["recall"], r["f1"], r["median_ttd_ms"], r["pred_p10_p50_p90"][1], r["corr"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())