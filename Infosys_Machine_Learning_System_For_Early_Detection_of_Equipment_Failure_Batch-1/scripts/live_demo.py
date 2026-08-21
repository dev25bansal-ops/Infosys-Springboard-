#!/usr/bin/env python3
"""Live demo — stream Binance WebSocket, run trained TCN, fire alerts.

Connects to Binance WebSocket, maintains a 200-tick feature window,
runs the trained TCN on each new tick, and prints alerts to console
with colored output.

Usage:
    python scripts/live_demo.py --model models/stage3_tcn_trained.pt --symbol BTCUSDT
    python scripts/live_demo.py --model models/stage3_tcn_trained.pt --symbol BTCUSDT --threshold 0.3
"""
import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

ML_DIR = Path(__file__).resolve().parent.parent / "ml"
sys.path.insert(0, str(ML_DIR))
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from flash_crash_watchdog.features import FEATURE_NAMES, FeatureExtractor
from flash_crash_watchdog.models.stage3_tcn import TCNDetector, TCNConfig
torch.serialization.add_safe_globals([TCNConfig])

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TCN_FEATURES = FEATURE_NAMES[:17]
WINDOW_SIZE = 200

# ANSI colors for console output
class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    BOLD = '\033[1m'
    END = '\033[0m'


def load_trained_tcn(model_path: str, device: str = "auto") -> TCNDetector:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(model_path, map_location=device, weights_only=True)
    config = data["config"]
    model = TCNDetector(config).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, device


async def run_live_demo(model: TCNDetector, device: str, symbol: str,
                        threshold: float, alert_log: str) -> None:
    """Stream Binance WebSocket and run the trained TCN in real time."""
    import websockets
    from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
    from flash_crash_watchdog.tick import Tick, Trade

    extractor = FeatureExtractor()
    feature_window = deque(maxlen=WINDOW_SIZE)
    url = f"wss://stream.binance.com:9443/stream?streams={symbol.lower()}@depth20@100ms/{symbol.lower()}@trade"

    ticks_processed = 0
    alerts_fired = 0
    start_time = time.time()

    # Console header
    print(f"\n{Color.BOLD}{Color.CYAN}{'='*60}")
    print(f"  FLASH CRASH EARLY WARNING — LIVE DEMO")
    print(f"  Symbol:    {symbol}")
    print(f"  Model:     TCN (trained on real crash data)")
    print(f"  Threshold: {threshold}")
    print(f"  Window:    {WINDOW_SIZE} ticks")
    print(f"  Press Ctrl+C to stop")
    print(f"{'='*60}{Color.END}\n")

    log_file = open(alert_log, "a") if alert_log else None

    def print_alert(score, price, features):
        nonlocal alerts_fired
        alerts_fired += 1
        ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        obi = features.get("f2_obi_10", 0.0)
        vol = features.get("f4_realized_vol_1s", 0.0)
        vpin = features.get("f3_vpin", 0.0)
        print(f"{Color.RED}{Color.BOLD}🚨 ALERT #{alerts_fired}{Color.END}  "
              f"{Color.YELLOW}{ts}{Color.END}  "
              f"score={score:.3f}  price=${price:,.2f}  "
              f"OBI={obi:+.4f}  VPIN={vpin:.4f}  Vol={vol:.6f}")
        if log_file:
            log_file.write(json.dumps({
                "timestamp": ts, "alert_num": alerts_fired,
                "score": score, "price": price,
                "obi_10": obi, "vpin": vpin, "realized_vol": vol,
            }) + "\n")
            log_file.flush()

    # Reconnect loop
    while True:
        try:
            async with websockets.connect(url, ping_interval=20, ping_timeout=60, max_size=8*1024*1024) as ws:
                if ticks_processed == 0:
                    logger.info("Connected. Waiting for %d ticks to fill window...", WINDOW_SIZE)
                else:
                    logger.info("Reconnected. Continuing (window preserved)...")

                async for raw in ws:
                    msg = json.loads(raw)
                    if "data" not in msg:
                        continue
                    data = msg["data"]
                    stream = msg.get("stream", "")

                    if "depth" in stream:
                        bids = [PriceLevel(float(p), float(s)) for p, s in data.get("bids", [])[:20]]
                        asks = [PriceLevel(float(p), float(s)) for p, s in data.get("asks", [])[:20]]
                        ts_ms = int(time.time() * 1000)
                        tick = Tick(book=OrderBookSnapshot(timestamp_ms=ts_ms, bids=bids, asks=asks), symbol=symbol)
                    elif "trade" in stream:
                        price = float(data.get("p", 0))
                        size = float(data.get("q", 0))
                        is_buyer_maker = data.get("m", False)
                        side = "sell" if is_buyer_maker else "buy"
                        ts_ms = data.get("T", int(time.time() * 1000))
                        trade = Trade(timestamp_ms=ts_ms, price=price, size=size, side=side)
                        tick = Tick(
                            book=OrderBookSnapshot(timestamp_ms=ts_ms,
                                                   bids=[PriceLevel(price, size)],
                                                   asks=[PriceLevel(price, size)]),
                            trades=[trade], symbol=symbol)
                    else:
                        continue

                    features = extractor.extract(tick)
                    vec = np.array([features.get(f, 0.0) for f in TCN_FEATURES])
                    feature_window.append(vec)
                    ticks_processed += 1

                    if ticks_processed % 100 == 0 and ticks_processed <= WINDOW_SIZE:
                        pct = ticks_processed / WINDOW_SIZE * 100
                        sys.stdout.write(f"\r{Color.BLUE}Filling window: {pct:.0f}% ({ticks_processed}/{WINDOW_SIZE}){Color.END}")
                        sys.stdout.flush()

                    if ticks_processed == WINDOW_SIZE:
                        print(f"\n{Color.GREEN}Window filled. Running TCN detector...{Color.END}\n")

                    if len(feature_window) >= WINDOW_SIZE:
                        window_array = np.array(list(feature_window))
                        with torch.no_grad():
                            x = torch.FloatTensor(window_array).T.unsqueeze(0).to(device)
                            scores = model(x)
                            score = float(scores[0, -1].item())

                        if ticks_processed % 500 == 0:
                            price = tick.book.mid_price or 0.0
                            elapsed = time.time() - start_time
                            rate = ticks_processed / max(1, elapsed)
                            status = (f"{Color.BLUE}[{ticks_processed:>6} ticks | {rate:.0f}/s | {alerts_fired} alerts] "
                                      f"score={score:.3f} price=${price:,.2f}{Color.END}")
                            sys.stdout.write(f"\r{status}")
                            sys.stdout.flush()

                        if score >= threshold:
                            price = tick.book.mid_price or 0.0
                            print()
                            print_alert(score, price, features)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error("WebSocket error: %s — reconnecting in 2s...", e)
            await asyncio.sleep(2)

    if log_file:
        log_file.close()
    print(f"\n\n{Color.CYAN}Demo ended. {ticks_processed} ticks processed, {alerts_fired} alerts fired.{Color.END}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Live flash-crash detection demo")
    parser.add_argument("--model", required=True, help="Trained TCN model")
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--log", default="", help="Alert log file (JSONL)")
    args = parser.parse_args()

    model, device = load_trained_tcn(args.model)

    try:
        asyncio.run(run_live_demo(model, device, args.symbol, args.threshold, args.log))
    except KeyboardInterrupt:
        print(f"\n{Color.YELLOW}Stopped by user.{Color.END}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
