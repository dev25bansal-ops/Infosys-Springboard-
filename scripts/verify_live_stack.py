"""End-to-end live-stack verification.

Drives the ACTUAL ml-inference/server /score FastAPI (which serves
stage3_tcn_prod.pt + rolling-z normalization) with the same batched-tick payloads
the mini-services/binance-stream sends, then applies the mini-stream's exact
alert gate (score>0.5 && trailing-vol>=2bps && 10s cooldown) on:
  - a CRASH day (should fire alerts), and
  - a NORMAL day (should fire ~0).

This confirms the whole server->score->gate->alert chain works live and that the
trailing-vol gate zeros calm-day chatter on the real served model.
"""
import sys
import pandas as pd
import numpy as np

sys.path.insert(0, "flash-crash-watchdog-web/ml-inference")
import server  # noqa: E402  (loads prod model + extractor at import)
from fastapi.testclient import TestClient  # noqa: E402

client = TestClient(server.app)

THR = 0.5
MIN_TV_BPS = 2.0
COOLDOWN_MS = 10000
BATCH = 10


def trailing_vol_bps(mids):
    w = mids[-200:]
    if len(w) < 50:
        return 0.0
    mean = float(np.mean(w))
    return float(np.std(w) / mean) * 10000.0 if mean else 0.0


def run_day(parquet, max_ticks=0):
    # reset server state so each day is scored independently (as a fresh session)
    client.post("/reset")
    server.extractor = server.FeatureExtractor()

    df = pd.read_parquet(parquet)
    if max_ticks > 0:
        df = df.iloc[: max_ticks]

    mids = []
    alerts = 0
    last_ts = 0
    batch = []
    n_req = 0
    tray = server.s3.model  # keep the loaded model referenced

    for row in df.itertuples(index=False):
        mid = (float(row.best_bid) + float(row.best_ask)) / 2.0
        mids.append(mid)
        ts = int(row.timestamp_ms)
        batch.append({
            "timestamp_ms": ts,
            "price": mid,
            "bids": [[float(row.best_bid), float(row.bid_size)]],
            "asks": [[float(row.best_ask), float(row.ask_size)]],
            "symbol": "BTCUSDT",
            "trade": {"price": float(row.trade_price), "size": float(row.trade_size), "timestamp_ms": ts},
        })
        if len(batch) >= BATCH:
            data = client.post("/score", json={"ticks": batch}).json()
            batch = []
            n_req += 1
            score = float(data.get("score", 0.0))
            tv = trailing_vol_bps(mids)
            if score > THR and tv >= MIN_TV_BPS and (ts - last_ts) > COOLDOWN_MS:
                alerts += 1
                last_ts = ts
    return len(df), n_req, alerts


def main():
    P = "C:/Users/dev25/AppData/Local/Temp/fcw_prod/"
    print("server serves:", server.MODEL_PATH, "| norm_window:", server.s3._norm_window,
          "| threshold:", server.s3._threshold)
    for name, parquet, max_t in [
        ("BTC-0519 (crash)", P + "BTC_0519_val.parquet", 8000),
        ("BTC-0116 (normal)", P + "BTC_2024_0116_norm.parquet", 8000),
    ]:
        n, req, alerts = run_day(parquet, max_t)
        print("%-16s ticks=%5d req=%4d  alerts(with gate)=%d" % (name, n, req, alerts))


if __name__ == "__main__":
    main()