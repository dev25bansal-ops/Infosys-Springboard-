#!/usr/bin/env python3
"""Analyze the downloaded market-context data: per-day funding + futures move.

Question: does funding rate magnitude or futures-volatility separate the CRASH
days from the NORMAL days at the day level? If yes, a cheap "regime gate" (only
publish alerts when funding|basis is elevated) could suppress calm-day chatter.

Usage:  PYTHONPATH=ml python scripts/analyze_context.py
"""
import glob
import json
from pathlib import Path
import numpy as np

OUT = Path("data/more/context")

print("\nfunding-rate magnitude per day:")
for f in sorted(OUT.glob("*_funding.json")):
    parts = f.stem.split("_")
    sym, day = parts[0], parts[1]
    d = json.load(open(f))
    rates = [abs(float(x.get("fundingRate", 0))) for x in d if x.get("fundingRate")]
    print(f"{sym:10} {day:12} n_fund={len(rates):3d}  mean|fund|={np.mean(rates):.6f}")

print("\n1h futures range (max-close - min-close relative to day low) + peak volume:")
for f in sorted(OUT.glob("*_klines.json")):
    parts = f.name.split("_")
    j, day = parts[0], parts[1]
    d = json.load(open(f))
    lo = min(float(x[3]) for x in d)
    closes = [float(x[4]) for x in d]
    vols = [float(x[5]) for x in d]
    range_pct = (max(closes) - min(closes)) / lo * 100
    peak = max(vols)
    print(f"{j:10} {day:12} n={len(d):3d} fut_range%={range_pct:6.1f}  vol_peak={peak/1e6:.0f}M")