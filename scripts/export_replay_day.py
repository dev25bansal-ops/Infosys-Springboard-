#!/usr/bin/env python3
"""AF-2: export a crash day's per-tick staging as a JSON timeseries for the replay UI.

Replays a historical parquet through the TRAINED cascade fast enough to stream:
  - Batched GPU Stage-3 scores (the slow part, done in one pass),
  - Cheap per-tick Stage-1 + Stage-2 pass flags and trailing-volatility per tick,
  - The cascade alert decision (Stage-3 >= threshold AND trailing-vol >= gate, coalesced).

Writes data/replay/<day>.json: {symbol, columns:[...], start_ms, series:[...], alerts:[...]}
for the replay-service to stream with speed/seek.

Usage:
    PYTHONPATH=ml python scripts/export_replay_day.py --data <slice.parquet> --label btc-0519
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.historical_loader import df_to_ticks  # noqa: E402
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor  # noqa: E402
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector  # noqa: E402
from flash_crash_watchdog.models.stage1_statistical import Stage1Statistical, Stage1Config  # noqa: E402
from flash_crash_watchdog.models.stage2_isolation_forest import Stage2IsolationForest  # noqa: E402

FC = FEATURE_NAMES[:17]
W = 200
NORM_WIN = 500


def normalize_z(f: np.ndarray, window: int) -> np.ndarray:
    pdf = pd.DataFrame(f)
    mean = pdf.rolling(window, min_periods=1).mean()
    std = pdf.rolling(window, min_periods=1).std()
    return ((pdf - mean) / std).where(std.abs() > 1e-8, 0.0).fillna(0.0).to_numpy(np.float32)


def trailing_vol_bps(mid, end_idx, window=200):
    seg = mid[max(0, end_idx - window + 1):end_idx + 1]
    if len(seg) < 50:
        return 0.0
    m = float(seg.mean())
    return float(seg.std() / m) * 10000.0 if m else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="crash slice parquet")
    ap.add_argument("--label", required=True, help="short id, e.g. btc-0519")
    ap.add_argument("--model", default="models/stage3_tcn_prod.pt")
    ap.add_argument("--stage2", default="models/stage2_isolation_forest.joblib")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--gate-bps", type=float, default=2.0)
    ap.add_argument("--cooldown-ms", type=int, default=10000)
    ap.add_argument("--max-ticks", type=int, default=0)
    ap.add_argument("--symbol", default=None,
                    help="symbol for the replay JSON (default: derived from the parquet filename, "
                         "e.g. ETHUSDT_2024-08-05.parquet -> ETHUSDT)")
    ap.add_argument("--out", default="data/replay/{label}.json")
    args = ap.parse_args()

    # NEW-05: derive the symbol from the file unless given explicitly (the old code
    # hardcoded "BTCUSDT", so ETH/LUNA replay files were mislabeled).
    symbol = args.symbol or Path(args.data).stem.split("_")[0]
    if not symbol:
        symbol = "BTCUSDT"

    df = pd.read_parquet(args.data)
    if args.max_ticks > 0 and len(df) > args.max_ticks:
        df = df.iloc[: args.max_ticks]

    ticks = list(df_to_ticks(df, symbol="R"))
    ext = FeatureExtractor()
    F = np.zeros((len(ticks), 17), dtype=np.float32)
    mid = np.full(len(ticks), np.nan, dtype=np.float64)
    times = np.empty(len(ticks), dtype=np.int64)
    for i, t in enumerate(ticks):
        fd = ext.extract(t)
        F[i] = [float(fd.get(k, 0.0)) or 0.0 for k in FC]
        if t.book.mid_price is not None:
            mid[i] = t.book.mid_price
        times[i] = t.book.timestamp_ms
    F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)
    mid = pd.Series(mid).ffill().bfill().to_numpy()

    # ---- batched GPU Stage-3 scores ----
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # Secure load (MLOPS-06): weights_only=True + the TCNConfig dataclass allowlisted.
    torch.serialization.add_safe_globals([TCNConfig])
    st = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to(dev)
    model.load_state_dict(st["model_state"]); model.eval()
    norm = normalize_z(F, NORM_WIN)
    n_win = len(norm) - W + 1
    s3 = np.zeros(n_win, dtype=np.float32)
    if n_win > 0:
        for s in range(0, n_win, 4096):
            e = min(s + 4096, n_win)
            win = np.stack([norm[i:i + W] for i in range(s, e)])
            x = torch.from_numpy(np.ascontiguousarray(win)).permute(0, 2, 1).float().to(dev)
            with torch.no_grad():
                s3[s:e] = model(x)[:, -1].cpu().numpy()

    # ---- cheap per-tick stage flags ----
    s1 = Stage1Statistical(Stage1Config())
    s2 = Stage2IsolationForest()
    if Path(args.stage2).exists():
        s2.load(args.stage2)

    series = []
    alerts = []
    last_alert = -10 ** 9
    for i, t in enumerate(ticks):
        _, p1 = s1.score(t)
        _, p2 = s2.score(t)
        # Stage-3 score of the window ENDING at the current tick (causal, RSR-01).
        # s3[j] is the score of window [j, j+W) -> ends at tick j+W-1, so the score
        # available at tick i is s3[i-W+1]. During warmup (i < W-1) no full window
        # has ended yet -> 0.0. (The old `s3[i]` plotted a window STARTING at i,
        # leaking 199 ticks of future data into every alert.)
        j = i - W + 1
        s3v = float(s3[j]) if 0 <= j < n_win else 0.0
        tv = trailing_vol_bps(mid, i)
        fire = s3v >= args.threshold and tv >= args.gate_bps
        if fire and (times[i] - last_alert) >= args.cooldown_ms:
            last_alert = int(times[i])
            alerts.append({"t": int(times[i]), "price": float(mid[i]), "s3": float(s3v), "tv": float(tv)})
        series.append({
            "t": int(times[i]), "p": float(mid[i]),
            "s3": float(s3v), "tv": float(tv),
            "s1": int(1 if p1 else 0), "s2": int(1 if p2 else 0),
        })

    out = Path(str(args.out).replace("{label}", args.label))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "label": args.label,
        "symbol": symbol,  # NEW-05: derived, not hardcoded
        "threshold": args.threshold, "gate_bps": args.gate_bps,
        "start_ms": int(times[0]), "ticks": len(series),
        "columns": ["t", "p", "s3", "tv", "s1", "s2"],
        "series": series, "alerts": alerts,
    }))
    print(f"exported {len(series)} ticks, {len(alerts)} alerts -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())