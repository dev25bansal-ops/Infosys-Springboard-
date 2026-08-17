#!/usr/bin/env python3
"""Download market-context data (funding rate + futures klines) from Binance's
public futures REST API for a set of (symbol, day) pairs.

Data source: https://fapi.binance.com (public endpoints, no auth):
  - /fapi/v1/fundingRate      -> 8h funding rate, full history
  - /fapi/v1/klines  (interval 1h or m) -> futures klines (for mark/vs-spot basis + volume)

Saves to data/more/context/<SYMBOL>_<DATE>_funding.json and _klines.json.

Usage:  PYTHONPATH=ml python scripts/download_market_context.py
"""
import json
import sys
from datetime import datetime
from pathlib import Path
from urllib import request, error

sys.path.insert(0, "ml")

FAPI = "https://fapi.binance.com"
OUT = Path(__file__).resolve().parent.parent / "data" / "more" / "context"

# (symbol, ISO date, is this a crash day?) — BTC/ETH/LUNA crash + normal days
JOBS = [
    ("BTCUSDT", "2021-05-19", "crash"),
    ("BTCUSDT", "2022-05-10", "crash"),
    ("BTCUSDT", "2024-08-05", "crash"),
    ("BTCUSDT", "2024-01-15", "normal"),
    ("BTCUSDT", "2024-01-16", "normal"),
    ("ETHUSDT", "2021-05-19", "crash"),
    ("ETHUSDT", "2022-06-13", "crash"),
    ("ETHUSDT", "2024-08-05", "crash"),
    ("ETHUSDT", "2024-01-15", "normal"),
    ("ETHUSDT", "2024-01-16", "normal"),
    ("LUNAUSDT", "2022-05-09", "crash"),
    ("LUNAUSDT", "2022-05-10", "crash"),
    ("LUNAUSDT", "2022-05-11", "crash"),
    ("LUNAUSDT", "2022-05-12", "crash"),
]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    fail = []
    for sym, day, tag in JOBS:
        s, e = day_range(day)
        # --- funding (whole day) ---
        furl = f"{FAPI}/fapi/v1/fundingRate?symbol={sym}&startTime={s}&endTime={e}&limit=1000"
        try:
            fund = get(furl)
        except Exception as ex:
            fail.append(f"{sym}_{day}_funding {ex}"); fund = []
        if fund:
            (OUT / f"{sym}_{day}_funding.json").write_text(json.dumps(fund, indent=1))
        # --- futures close klines (1h) for basis / volume ---
        kurl = f"{FAPI}/fapi/v1/klines?symbol={sym}&interval=1h&startTime={s}&endTime={e}&limit=64"
        try:
            kl = get(kurl)
        except Exception as ex:
            fail.append(f"{sym}_{day}_klines {ex}"); kl = []
        if kl:
            (OUT / f"{sym}_{day}_klines.json").write_text(json.dumps(kl))
        cnt = f" fund={len(fund)} klines={len(kl)}"
        print(f"{tag:6} {sym} {day}:{cnt}")
        if fund or kl:
            ok += 1
    print(f"\nDone: {ok}/{len(JOBS)} symbols/days downloaded -> {OUT}")
    if fail:
        print("failures:"); [print("  ", f) for f in fail]
    return 0


def day_range(date_str):
    from datetime import timezone
    start = int(datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)
    return start, start + 24 * 3600 * 1000


def get(url):
    import time
    for attempt in range(3):
        try:
            with request.urlopen(request.Request(url, headers={"User-Agent": "fc/0.4"}), timeout=60) as r:
                return json.loads(r.read().decode())
        except (error.HTTPError, error.URLError, TimeoutError, ValueError) as e:
            if attempt == 2:
                raise
            time.sleep(1.0)
    raise RuntimeError("unreachable")


FAPI = "https://fapi.binance.com"


if __name__ == "__main__":
    raise SystemExit(main())