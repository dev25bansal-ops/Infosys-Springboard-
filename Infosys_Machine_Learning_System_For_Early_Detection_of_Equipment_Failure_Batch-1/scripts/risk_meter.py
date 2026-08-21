#!/usr/bin/env python3
"""STR-08: magnitude head as a risk meter (predicted forward drop-%).

Scores a labeled window set with the magnitude checkpoint (TCNMagnitudeDetector,
which REGRESSES the forward min-drop-% rather than emitting a binary crash score)
and surfaces the predicted-drawdown distribution — the "how bad could this be"
risk meter. This is an alert-RANKING/severity signal, NOT a replacement for the
binary detector.

Usage:
    PYTHONPATH=ml python scripts/risk_meter.py \
        --model models/stage3_tcn_magnitude.pt --data data/windows/<day>_windows.npz
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))

from flash_crash_watchdog.data.windows import load_windows  # noqa: E402
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNMagnitudeDetector  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/stage3_tcn_magnitude.pt")
    ap.add_argument("--data", required=True, help=".npz with windows (+ optional labels)")
    ap.add_argument("--mag-cap", type=float, default=5.0)
    args = ap.parse_args()

    torch.serialization.add_safe_globals([TCNConfig])  # MLOPS-06
    st = torch.load(args.model, map_location="cpu", weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNMagnitudeDetector(cfg, mag_cap=args.mag_cap).to("cpu")
    model.load_state_dict(st["model_state"])
    model.eval()

    windows, labels = load_windows([Path(args.data)])
    x = torch.FloatTensor(windows).permute(0, 2, 1)
    with torch.no_grad():
        pred = model(x)[:, -1].squeeze().numpy()  # predicted drop-% per window
    pred = np.clip(pred, 0.0, args.mag_cap)

    print(f"model={Path(args.model).name}  windows={len(windows)}  label_pos={int(labels.sum())}")
    print(f"  predicted drop-%%: p50={np.percentile(pred,50):.3f}  p90={np.percentile(pred,90):.3f}  "
          f"p99={np.percentile(pred,99):.3f}  max={pred.max():.3f}")
    severe = float(np.mean(pred >= 1.0))  # fraction flagged as "severe" (>=1% drawdown)
    print(f"  fraction with predicted drop >= 1.0%%: {severe:.4f}")
    pos = labels > 0
    if pos.sum() >= 3 and pos.sum() < labels.size:
        print(f"  positive (crash) windows: {int(pos.sum())}  "
              f"their predicted drop-%% p50={np.percentile(pred[pos],50):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())