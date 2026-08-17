#!/usr/bin/env python3
"""Find which normal-day windows score > 0.45 with the prod TCN and characterize
their price action (small dip? noise?) vs the full normal-day distribution."""
import argparse, logging, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
from flash_crash_watchdog.data.historical_loader import df_to_ticks
from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor
from flash_crash_watchdog.models.stage3_tcn import TCNConfig, TCNDetector
torch.serialization.add_safe_globals([TCNConfig])

logging.basicConfig(level=logging.WARNING)
W = 200
NORM = 500
CH = 4096

def normalize_z(f, window):
    pdf = pd.DataFrame(f)
    mean = pdf.rolling(window, min_periods=1).mean()
    std = pdf.rolling(window, min_periods=1).std()
    return ((pdf - mean) / std).where(std.abs() > 1e-8, 0.0).fillna(0.0).to_numpy(np.float32)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", default="models/stage3_tcn_prod.pt")
    ap.add_argument("--thr", type=float, default=0.45)
    ap.add_argument("--lookahead", type=int, default=500)
    ap.add_argument("--max-ticks", type=int, default=0)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    df = pd.read_parquet(a.data)
    if a.max_ticks and len(df) > a.max_ticks: df = df.iloc[:a.max_ticks]
    ticks = list(df_to_ticks(df, symbol="EVAL"))
    ex = FeatureExtractor(); F = np.zeros((len(ticks),17), np.float32); mid = np.zeros(len(ticks), np.float64)
    ts = np.zeros(len(ticks), np.float64)
    for i,t in enumerate(ticks):
        fd = ex.extract(t); F[i]=[float(fd.get(k,0.0)) or 0.0 for k in FEATURE_NAMES[:17]]
        mid[i]=float(t.book.mid_price) if t.book.mid_price else 0.0
        ts[i]=float(t.book.timestamp_ms)
    st = torch.load(a.model, map_location='cpu', weights_only=True)
    cfg = st["config"] if isinstance(st["config"], TCNConfig) else TCNConfig(**st["config"])
    model = TCNDetector(cfg).to(dev); model.load_state_dict(st["model_state"]); model.eval()
    norm = normalize_z(F, NORM)
    n = len(norm)-W+1
    scores = np.empty(n, np.float32)
    for s in range(0,n,CH):
        e=min(s+CH,n)
        win=np.stack([norm[i:i+W] for i in range(s,e)])
        x=torch.from_numpy(np.ascontiguousarray(win)).permute(0,2,1).float().to(dev)
        with torch.no_grad(): scores[s:e]=model(x)[:, -1].cpu().numpy()
    hi = np.where(scores > a.thr)[0]
    print("total windows=%d  >%.2f: %d" % (n, a.thr, len(hi)))
    print("score dist p50=%.3f p90=%.3f p99=%.3f max=%.3f"%(
        np.percentile(scores,50),np.percentile(scores,90),np.percentile(scores,99),scores.max()))
    # characterize the high windows
    print("\nidx     score    win_drop%  fut_drop%   mean_|1s return|%")
    drops=[]; big=0
    for j in hi:
        w0 = mid[j]; wend = mid[j+W-1]
        fut = mid[j+W:j+W+a.lookahead]
        win_drop = (w0-wend)/w0*100 if w0 else 0
        fut_drop = (wend-min(fut))/wend*100 if wend>0 and len(fut) else 0
        # intra-window volatility: mean abs per-tick return
        rets=np.abs(np.diff(mid[j:j+W])); r=rets.mean()/wend*100 if wend else 0
        drops={"win":win_drop,"fut":fut_drop}
        big += 1 if (win_drop>1.0 or fut_drop>1.0) else 0
        print("%5d %.3f  %8.2f  %8.2f   %7.3f"%(j+W-1, scores[j], win_drop, fut_drop, r))
    print("high windows with >1%% window or future drop: %d of %d"%(big,len(hi)))
    # overall normal-day drop stats for context
    ov=np.abs(np.diff(mid)); print("normal-day median abs-per-tick%%=%.4f  p99=%.4f"%(np.mean(ov)/np.mean(mid)*100, np.percentile(ov,99)/np.mean(mid)*100))

if __name__=="__main__": main()