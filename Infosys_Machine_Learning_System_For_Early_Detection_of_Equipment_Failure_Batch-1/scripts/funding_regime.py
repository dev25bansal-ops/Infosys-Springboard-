#!/usr/bin/env python3
"""STR-10: funding-rate regime as an advisory deleveraging flag.

Reads the downloaded per-day funding history (data/more/context/<SYM>_<date>_funding.json)
and classifies the funding regime:
  - DELEVERAGING: funding turned DEEPLY negative (< -5 bps) — longs being squeezed out
                  (a capitulation / deleveraging signal, as on BTC 2021-05-19).
  - STRESSED / STRAINED: intermediate negative-fuel bands.
  - NORMAL: no significant negative funding.

The full "basis" layer additionally needs spot-vs-futures price pairs; only futures
klines are downloaded, so basis is noted-but-not-computed here. Advisory only — a
regime flag, never an auto-trade signal.

Usage:
    python scripts/funding_regime.py --context data/more/context --day 2021-05-19 --symbol BTCUSDT
"""
import argparse
import json
import sys
from pathlib import Path

DELEVERAGING_BPS = -5.0   # funding < -5 bps => deep deleveraging
STRESSED_BPS = -2.0       # mean < -2 bps => stressed


def funding_regime(records) -> dict:
    rates = []
    for r in records:
        raw = r.get("fundingRate") if isinstance(r, dict) else None
        try:
            f = float(raw)
        except (TypeError, ValueError):
            continue
        rates.append(f)
    if not rates:
        return {"band": "NO_DATA", "n": 0}
    mean = sum(rates) / len(rates)
    mn = min(rates)
    n_neg = sum(1 for r in rates if r < 0) / len(rates)
    if mn < DELEVERAGING_BPS / 10000:
        band = "DELEVERAGING"
    elif mean < STRESSED_BPS / 10000:
        band = "STRESSED"
    elif n_neg > 0.5:
        band = "STRAINED"
    else:
        band = "NORMAL"
    return {"band": band, "n": len(rates), "mean_bps": round(mean * 10000, 2),
            "min_bps": round(mn * 10000, 2), "neg_frac": round(n_neg, 2)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--context", default="data/more/context")
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--day", required=True)
    args = ap.parse_args()

    f = Path(args.context) / f"{args.symbol}_{args.day}_funding.json"
    if not f.exists():
        print(f"no funding file for {args.symbol} {args.day}")
        return 1
    records = json.loads(f.read_text())
    regime = funding_regime(records)
    print(f"{args.symbol} {args.day}: band={regime['band']}  n={regime['n']}  "
          f"mean={regime['mean_bps']}bps  min={regime['min_bps']}bps  neg_frac={regime['neg_frac']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())