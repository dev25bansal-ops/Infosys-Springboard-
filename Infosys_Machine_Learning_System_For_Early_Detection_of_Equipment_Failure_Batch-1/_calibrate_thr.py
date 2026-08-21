"""Per-asset Stage-3 threshold calibration on the online path.

Computes the stride-1 (per-tick, rolling-z normalized) score distribution of a
model on a given day's contiguous features and prints the threshold that bounds
the false-positive rate on that asset (p999 => ~0.1% of ticks exceed it).
Usage: python _calibrate_thr.py <parquet> <model.pt>
"""
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "ml")

from flash_crash_watchdog.data.historical_loader import df_to_ticks
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector
torch.serialization.add_safe_globals([TCNConfig])

parquet, model_path = sys.argv[1], sys.argv[2]

df = pd.read_parquet(parquet)
ext = FeatureExtractor()
tcf = FEATURE_NAMES[:17]
feats, mids = [], []
for t in df_to_ticks(df, symbol="CAL"):
    f = ext.extract(t)
    feats.append([f.get(k, 0.0) for k in tcf])
    mids.append(float(t.book.mid_price) if t.book.mid_price else 0.0)
features = np.nan_to_num(np.asarray(feats, np.float32))

fdf = pd.DataFrame(features)
mean = fdf.rolling(2000, min_periods=1).mean()
std = fdf.rolling(2000, min_periods=1).std()
norm = ((fdf - mean) / std).where(std.abs() > 1e-8, 0.0).fillna(0.0).to_numpy(np.float32)

W = 200
# stride-1 windows (chunked GPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
st = torch.load(model_path, map_location="cpu", weights_only=True)
cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
model = TCNDetector(cfg).to(device)
model.load_state_dict(st["model_state"]); model.eval()

scores = []
CH = 8192
for i in range(0, len(norm) - W + 1, CH):
    hi = min(i + CH, len(norm) - W + 1)
    chunk = norm[i:hi + W - 1]           # enough rows for hi-i stride-1 windows
    win = np.stack([chunk[j:j + W] for j in range(hi - i)])   # (B, W, F)
    x = torch.from_numpy(np.ascontiguousarray(win)).permute(0, 2, 1).float().to(device)
    with torch.no_grad():
        scores.append(model(x)[:, -1].cpu().numpy())
s = np.concatenate(scores)

print(f"asset={parquet.split('/')[-1]} ticks={len(norm)} windows={len(s)}")
print(f"score dist p50={np.percentile(s,50):.4f} p90={np.percentile(s,90):.4f} "
      f"p99={np.percentile(s,99):.4f} p999={np.percentile(s,99.9):.4f} max={s.max():.4f}")
print(f"THRESHOLD_P999={np.percentile(s,99.9):.4f}  implied_fp_pct={100.0*(s>np.percentile(s,99.9)).mean():.3f}")