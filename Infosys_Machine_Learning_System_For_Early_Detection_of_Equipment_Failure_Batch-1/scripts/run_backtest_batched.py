#!/usr/bin/env python3
"""Batched (GPU) backtest for the Stage-3 flash-crash detector.

The full cascade scores Stage-3 once per tick (a TCN forward per tick), which is
correct in production but makes offline backtests impractically slow. This script
precomputes every stride-1 200-tick window score in one batched GPU pass, then
replays the (fixed) cascade decision cheaply:

    alert(t)  <=  s3(t) >= stage3_threshold  AND  Stage-5 posterior >= alert_threshold

Stage-5 fusion uses the trained Stage-2 score (cheap) + neutral Stage-4 (0.5).
Stage-1 is advisory (never a veto). Crash windows use the same 2%-drop-in-60s
label rule as the cascade backtest, and alerts are coalesced by --cooldown-ms.

Usage:
    PYTHONPATH=ml python scripts/run_backtest_batched.py \
        --data data/parquet/LUNAUSDT_2022-05-10.parquet --model models/stage3_tcn_prod.pt
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

from flash_crash_watchdog.data.historical_loader import df_to_ticks  # noqa: E402
from flash_crash_watchdog.data.labels import label_crashes  # noqa: E402
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from flash_crash_watchdog.models.stage3_tcn import normalize_z, TCNConfig, TCNDetector  # noqa: E402
torch.serialization.add_safe_globals([TCNConfig])
from flash_crash_watchdog.models.stage2_isolation_forest import Stage2IsolationForest  # noqa: E402
from flash_crash_watchdog.models.stage5_bayesian import Stage5Config, Stage5Bayesian  # noqa: E402

logger = logging.getLogger(__name__)
TCN_FEATURES = FEATURE_NAMES[:17]
W = 200          # trained window length
NORM_WINDOW = 500  # rolling-z window (matches training / Stage3TCN.feed)
CH = 4096        # GPU batch per forward chunk


def device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def stage3_scores(model: TCNDetector, F: np.ndarray, dev: str) -> np.ndarray:
    """Every stride-1 200-window Stage-3 score (batched GPU)."""
    if len(F) <= W:
        return np.zeros(0, dtype=np.float32)
    norm = normalize_z(F, NORM_WINDOW)
    n = len(norm) - W + 1
    out = np.empty(n, dtype=np.float32)
    for s in range(0, n, CH):
        e = min(s + CH, n)
        win = np.stack([norm[i:i + W] for i in range(s, e)])
        x = torch.from_numpy(np.ascontiguousarray(win)).permute(0, 2, 1).float().to(dev)
        with torch.no_grad():
            out[s:e] = model(x)[:, -1].cpu().numpy()
    return out


def score_day(model: TCNDetector, dev: str, parquet: str | Path, max_ticks: int, crash_pct: float):
    """Feature-extract + score one day. Returns (scores, times, ticks, mids, crash_ranges)."""
    df = pd.read_parquet(parquet)
    if max_ticks > 0 and len(df) > max_ticks:
        df = df.iloc[: max_ticks]
    ticks = list(df_to_ticks(df, symbol="DAY"))
    extractor = FeatureExtractor()
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    times = np.empty(len(ticks), dtype=np.int64)
    mids = np.full(len(ticks), np.nan, dtype=np.float64)
    for i, t in enumerate(ticks):
        fd = extractor.extract(t)
        F[i] = [float(fd.get(k, 0.0)) or 0.0 for k in TCN_FEATURES]
        times[i] = t.book.timestamp_ms
        mp = t.book.mid_price
        if mp is not None:
            mids[i] = mp
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    filled = pd.Series(mids).ffill().bfill().to_numpy()
    mids = np.where(np.isnan(filled), 1.0, filled)
    scores = stage3_scores(model, F, dev)  # scores[j] => window ending at tick j+W-1
    crashes = label_crashes(ticks, drop_threshold_pct=crash_pct)
    crash_ranges = [(c.start_ts, c.end_ts) for c in crashes]
    return scores, times, ticks, mids, crash_ranges


def evaluate(scores, times, ticks, s2, s5, threshold, crash_ranges, cooldown_ms,
             mids=None, confirmation_drop_pct=0.0, drop_lookback=0, min_tv_bps=0.0):
    """Replay + metrics for one threshold. Returns (alerts,tp,fp,fn,prec,rec,f1,med_ttd,rate).

    When confirmation_drop_pct > 0, an alert fires only if the mid is in a REAL
    sustained decline: mids[alert_tick] <= mids[alert_tick - drop_lookback] *
    (1 - confirmation_drop_pct/100). This rejects transient normal-day spikes
    while keeping crash-day sustained drops. drop_lookback==0 => W//2 (100 ticks).

    When min_tv_bps > 0, an alert fires only if the window's trailing realized
    volatility (std/mean of the mid over the 200-tick window, in bps) is above
    the threshold. Normal-day false positives are on ~0-vol flat windows; crash
    onsets are on high-vol windows — so this suppresses calm-day chatter while
    keeping crash detections (and it is observable in real time).
    """
    n_scores = len(scores)
    if drop_lookback <= 0:
        drop_lookback = W // 2
    alerts = []
    last_ts = None
    for j, tick_idx in enumerate(np.arange(n_scores) + W - 1):
        s3v = float(scores[j])
        if s3v < threshold:
            continue
        tick = ticks[tick_idx]
        s2v, _ = s2.score(tick)
        alert = s5.aggregate(tick, s2v, s3v, 0.5)
        if alert is None:
            continue
        ts = times[tick_idx]
        if min_tv_bps > 0:
            if mids is None or tick_idx < W:
                continue
            seg = mids[tick_idx - W + 1:tick_idx + 1]
            mn = float(seg.mean())
            tv = float(seg.std() / mn) * 10000.0 if mn > 0 else 0.0
            if tv < min_tv_bps:
                continue
        if confirmation_drop_pct > 0:
            if mids is None or tick_idx < drop_lookback:
                continue
            if not (mids[tick_idx] <= mids[tick_idx - drop_lookback] * (1.0 - confirmation_drop_pct / 100.0)):
                continue
        if last_ts is not None and ts - last_ts < cooldown_ms:
            continue
        last_ts = ts
        alerts.append(ts)
    matched = set()
    ttd = []
    tp = 0
    for ts in alerts:
        hit = [i for i, (a, b) in enumerate(crash_ranges) if a <= ts <= b]
        if hit:
            tp += 1
            matched.add(hit[0])
            ttd.append(crash_ranges[hit[0]][1] - ts)
    fn = len(crash_ranges) - len(matched)
    fp = len(alerts) - tp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return (len(alerts), tp, fp, fn, prec, rec, f1,
            float(np.median(ttd)) if ttd else 0.0, 100.0 * len(alerts) / max(1, len(ticks)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Batched GPU Stage-3 backtest")
    parser.add_argument("--data", required=True, help="Tick parquet")
    parser.add_argument("--model", default="models/stage3_tcn_prod.pt")
    parser.add_argument("--stage2", default="models/stage2_isolation_forest.joblib")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Stage-3 pass gate (operating point: 0.5; or use --thresholds)")
    parser.add_argument("--thresholds", default=None, help="comma list to sweep, e.g. '0.3,0.4,0.5,0.6,0.7'")
    parser.add_argument("--stage5-threshold", type=float, default=0.5)
    parser.add_argument("--cooldown-ms", type=int, default=10000)
    parser.add_argument("--confirmation-drop-pct", type=float, default=0.0,
                        help="0.0 = OFF. Confirmation gate: an alert counts only when the "
                             "mid is in a REAL sustained decline: "
                             "mids[alert_tick] <= mids[alert_tick - drop_lookback] * (1 - pct/100). "
                             "Rejects transient normal-day spikes, keeps crash-day sustained drops.")
    parser.add_argument("--drop-lookback", type=int, default=0,
                        help="ticks back to compare mid for the confirmation gate; 0 = W//2 (100). "
                             "Ignored when --confirmation-drop-pct == 0.")
    parser.add_argument("--min-trailing-vol-bps", type=float, default=0.0,
                        help="Regime gate (bps): only alert if the window's trailing realized "
                             "volatility (mid std/mean over 200 ticks) exceeds this. Suppresses "
                             "calm-day chatter (normal FPs are ~0 bps; crash onsets are 10s of bps). "
                             "0.0 = off.")
    parser.add_argument("--out-json", default=None,
                        help="Write the threshold-sweep table to a JSON file (for the leaderboard)")
    parser.add_argument("--crash-pct", type=float, default=2.0)
    parser.add_argument("--max-ticks", type=int, default=0, help="0 = all")
    parser.add_argument("--calibrate-on", default=None,
                        help="RSR-11: a SEPARATE parquet day used ONLY to select the "
                             "threshold (max F1). The eval day (--data) is never used "
                             "for threshold selection, which would overfit the metric.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    dev = device()
    st = torch.load(args.model, map_location="cpu", weights_only=True)  # MLOPS-06
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to(dev)
    model.load_state_dict(st["model_state"])
    model.eval()

    s2 = Stage2IsolationForest()
    if Path(args.stage2).exists():
        s2.load(args.stage2)
    s5 = Stage5Bayesian(Stage5Config(alert_threshold=args.stage5_threshold))

    scores, times, ticks, mids, crash_ranges = score_day(model, dev, args.data, args.max_ticks, args.crash_pct)
    logger.info("Found %d ground-truth crash windows on eval day", len(crash_ranges))

    thrs = [float(t) for t in args.thresholds.split(",")] if args.thresholds else [args.threshold]

    # RSR-11: threshold selection must never happen on the evaluation day.
    if args.calibrate_on:
        cal = score_day(model, dev, args.calibrate_on, args.max_ticks, args.crash_pct)
        best_f1, best_thr = -1.0, thrs[0]
        for thr in thrs:
            _an, tp, fp, fn, _prec, rec, f1, _med, _rate = evaluate(
                cal[0], cal[1], cal[2], s2, s5, thr, cal[4], args.cooldown_ms,
                mids=cal[3], confirmation_drop_pct=args.confirmation_drop_pct,
                drop_lookback=args.drop_lookback, min_tv_bps=args.min_trailing_vol_bps)
            if f1 > best_f1:
                best_f1, best_thr = f1, thr
        logger.info("RSR-11: selected threshold %.3f on calibration day %s (F1=%.3f); "
                    "eval day %s is untouched by selection",
                    best_thr, Path(args.calibrate_on).name, best_f1, Path(args.data).name)
        thrs = [best_thr]
    elif len(thrs) > 1:
        logger.warning(
            "RSR-11: sweeping %d thresholds ON the evaluation day overfits the metric — "
            "use --calibrate-on to select the threshold on a separate day", len(thrs))

    print("day=%s  ticks=%d  crashes=%d  gate_drop_pct=%.2f  drop_lookback=%d  min_tv_bps=%.1f"
          % (Path(args.data).name, len(ticks), len(crash_ranges),
             args.confirmation_drop_pct, (args.drop_lookback or W // 2), args.min_trailing_vol_bps))
    rows = []
    print("threshold,alerts,tp,fp,fn,precision,recall,f1,alert_rate_pct,median_ttd_ms")
    for thr in thrs:
        alerts_n, tp, fp, fn, prec, rec, f1, med, rate = evaluate(
            scores, times, ticks, s2, s5, thr, crash_ranges, args.cooldown_ms,
            mids=mids, confirmation_drop_pct=args.confirmation_drop_pct,
            drop_lookback=args.drop_lookback, min_tv_bps=args.min_trailing_vol_bps)
        print("%.3f,%d,%d,%d,%d,%.3f,%.3f,%.3f,%.3f,%.0f" % (thr, alerts_n, tp, fp, fn, prec, rec, f1, rate, med))
        rows.append({"threshold": thr, "alerts": alerts_n, "tp": tp, "fp": fp, "fn": fn,
                     "precision": prec, "recall": rec, "f1": f1,
                     "alert_rate_pct": rate, "median_ttd_ms": med})
    if args.out_json:
        import json as _json
        payload = {
            "day": Path(args.data).name,
            "ticks": len(ticks),
            "crashes": len(crash_ranges),
            "gate_bps": args.min_trailing_vol_bps,
            "cooldown_ms": args.cooldown_ms,
            "model": Path(args.model).name,
            "thresholds": rows,
        }
        Path(args.out_json).write_text(_json.dumps(payload, indent=2))
        logger.info("wrote leaderboard table -> %s", args.out_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())