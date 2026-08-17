#!/usr/bin/env python3
"""RSR-15: standardized validation harness — one evaluator, one 6-day table.

Consumes ``configs/operating.yml`` (model, threshold, gate_bps, cooldown_ms,
stage5_threshold) plus the machine-readable ``validation.days`` manifest, runs the
batched Stage-3 backtest at the operating point on every day, and emits ONE
per-day table plus a machine-readable ledger.

This is the canonical evaluation path. It unifies the previously-divergent
pipelines:
  - event-based crash labels            (RSR-03, data.labels.label_crashes)
  - shared rolling-z normalization      (BUG-03, models.stage3_tcn.normalize_z)
  - canonical alert-vs-event matching   (RSR-04, eval.metrics.match_alerts_to_crashes)
  - documented alert-rate denominator   (GAP-02: alerts per wall-clock hour)

Usage:
    PYTHONPATH=ml python scripts/run_validation.py \
        --config configs/operating.yml \
        --models-dir models --data-dir data/parquet \
        --out results/validation.json [--max-ticks 500000]
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.historical_loader import df_to_ticks, load_parquet  # noqa: E402
from flash_crash_watchdog.data.labels import label_crashes  # noqa: E402
from flash_crash_watchdog.eval.backtest import stage3_scores_batched  # noqa: E402
from flash_crash_watchdog.eval.metrics import match_alerts_to_crashes, ranges_from_crashlabels, wilson_ci  # noqa: E402
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from flash_crash_watchdog.models.stage3_tcn import TCNDetector, TCNConfig  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TCN_FEATURES = FEATURE_NAMES[:17]
W = 200  # trained window length (matches eval.backtest.WINDOW)


def run_day(df: pd.DataFrame, model: TCNDetector, dev: str, op: dict) -> dict:
    """Score one day at the operating point and return canonical metrics."""
    ticks = list(df_to_ticks(df, symbol="VAL"))
    extractor = FeatureExtractor()
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    times = np.empty(len(ticks), dtype=np.int64)
    mids = np.full(len(ticks), np.nan, dtype=np.float64)
    for i, t in enumerate(ticks):
        fd = extractor.extract(t)
        F[i] = [float(fd.get(k, 0.0)) or 0.0 for k in TCN_FEATURES]
        times[i] = t.book.timestamp_ms
        if t.book.mid_price is not None:
            mids[i] = t.book.mid_price
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    filled = pd.Series(mids).ffill().bfill().to_numpy()
    mids = np.where(np.isnan(filled), 1.0, filled)

    scores = stage3_scores_batched(model, F, dev)  # scores[j] => window ending at tick j+W-1

    # Alert decision rule (operating point): s3 >= threshold AND trailing realized
    # vol >= gate_bps AND cooldown_ms spacing.
    threshold = float(op["threshold"])
    gate_bps = float(op.get("gate_bps", 0.0))
    cooldown_ms = int(op.get("cooldown_ms", 0))
    alerts: list[int] = []
    last_ts = None
    for j, tick_idx in enumerate(np.arange(len(scores)) + W - 1):
        if scores[j] < threshold:
            continue
        ts = int(times[tick_idx])
        if gate_bps > 0:
            seg = mids[tick_idx - W + 1: tick_idx + 1]
            mn = float(seg.mean())
            tv = float(seg.std() / mn) * 10000.0 if mn > 0 else 0.0
            if tv < gate_bps:
                continue
        if last_ts is not None and ts - last_ts < cooldown_ms:
            continue
        last_ts = ts
        alerts.append(ts)

    crashes = label_crashes(ticks, drop_threshold_pct=float(op.get("crash_drop_pct", 2.0)),
                            window_ms=int(op.get("crash_window_ms", 60_000)))
    m = match_alerts_to_crashes(alerts, ranges_from_crashlabels(crashes),
                                grace_ms=int(op.get("grace_ms", 5_000)))

    span_s = max(1.0, (times[-1] - times[0]) / 1000.0)
    # STR-02: surface lead-time (TTD) distribution, not just the median.
    ttd = sorted(m.ttd_ms)
    p50 = ttd[len(ttd) // 2] if ttd else 0.0
    p95 = ttd[int(len(ttd) * 0.95)] if ttd else 0.0
    return {
        "ticks": len(ticks),
        "crashes": len(crashes),
        "exposure_hours": round(span_s / 3600.0, 3),
        "alerts": m.alerts,
        "alerts_per_hour": round(3600.0 * m.alerts / span_s, 4),
        # RSR-12: with n=2-9 events/day the point recall needs an honest CI.
        "recall_ci": [round(x, 3) for x in wilson_ci(m.true_positives, len(crashes))],
        "lead_time_p50_ms": round(p50, 1),
        "lead_time_p95_ms": round(p95, 1),
        **m.as_dict(),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="RSR-15 validation harness")
    parser.add_argument("--config", default="configs/operating.yml")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--model", default=None,
                        help="override the operating model (default: configs/operating.yml's model)")
    parser.add_argument("--data-dir", default="data/parquet")
    parser.add_argument("--out", default="results/validation.json")
    parser.add_argument("--max-ticks", type=int, default=500_000, help="0 = full day")
    parser.add_argument("--days", default=None, help="comma list of day 'date' to run (default: all manifest rows)")
    parser.add_argument("--slices", default=None,
                        help="ROW-RANGE slices for crash-region validation, comma list of "
                             "'date:start:end' e.g. '2021-05-19:3000000:3035000,2024-01-16:0:40000' "
                             "(overrides --max-ticks for those days)")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        op = yaml.safe_load(f)
    days = op["validation"]["days"]
    if args.days:
        want = set(d.strip() for d in args.days.split(","))
        days = [d for d in days if d["date"] in want]

    # parse slices: {date: (start, end)}
    slices: dict[str, tuple[int, int]] = {}
    if args.slices:
        for spec in args.slices.split(","):
            date, s, e = spec.strip().split(":")
            slices[date] = (int(s), int(e))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model_path = Path(args.models_dir) / (args.model or op["model"])
    # Secure load (MLOPS-06): weights_only=True + the TCNConfig dataclass allowlisted.
    torch.serialization.add_safe_globals([TCNConfig])
    st = torch.load(model_path, map_location="cpu", weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to(dev)
    model.load_state_dict(st["model_state"])
    model.eval()
    logger.info("Loaded %s (threshold=%s, gate_bps=%s, cooldown_ms=%s)",
                model_path.name, op["threshold"], op.get("gate_bps"), op.get("cooldown_ms"))

    rows = []
    for day in days:
        df = load_parquet(Path(args.data_dir) / day["file"])
        sl = slices.get(day["date"])
        if sl:
            start, end = sl
            df = df.iloc[start:end]
            logger.info("Validating %s (%s, %s) slice [%d:%d]", day["symbol"], day["date"], day["kind"], start, end)
        elif args.max_ticks > 0 and len(df) > args.max_ticks:
            df = df.iloc[: args.max_ticks]
            logger.info("Validating %s (%s, %s)", day["symbol"], day["date"], day["kind"])
        r = run_day(df, model, dev, {**op, **op.get("validation", {})})
        r.update({"symbol": day["symbol"], "date": day["date"], "kind": day["kind"]})
        rows.append(r)
        print("%-10s %-8s %-7s ticks=%-8d crashes=%-4d alerts=%-4d tp=%-3d fp=%-3d fn=%-3d "
              "prec=%.3f rec=%.3f f1=%.3f med_ttd=%.0f alerts/h=%.4f"
              % (day["symbol"], day["date"], day["kind"], r["ticks"], r["crashes"],
                 r["alerts"], r["true_positives"], r["false_positives"], r["false_negatives"],
                 r["precision"], r["recall"], r["f1"], r["median_ttd_ms"], r["alerts_per_hour"]))

    payload = {
        "operating_point": {k: op.get(k) for k in
                            ("model", "threshold", "gate_bps", "cooldown_ms", "stage5_threshold")},
        "config": str(args.config),
        "max_ticks": args.max_ticks,
        "days": rows,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote validation ledger -> %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
