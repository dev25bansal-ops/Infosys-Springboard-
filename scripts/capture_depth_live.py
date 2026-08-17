#!/usr/bin/env python3
"""Live WebSocket depth capture — records L2 order book snapshots to parquet.

Records Binance depth20 @ 100ms updates for one or more symbols.

Usage:
    python scripts/capture_depth_live.py --symbols BTCUSDT,ETHUSDT --hours 2
    python scripts/capture_depth_live.py --symbols BTCUSDT --minutes 30
    python scripts/capture_depth_live.py --symbols BTCUSDT --hours 0  # until Ctrl+C
"""
import argparse
import asyncio
import json
import logging
import signal
import time
from datetime import datetime
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="


class DepthRecorder:
    """Records L2 depth snapshots from Binance WebSocket."""

    def __init__(self, symbols: list[str], out_path: Path) -> None:
        self.symbols = [s.lower() for s in symbols]
        self.out_path = out_path
        self.records: list[dict] = []
        self._running = False
        self._start_time = 0.0
        self._last_save = 0.0
        self._save_interval = 30.0  # save every 30 seconds

    async def run(self, duration_seconds: float) -> None:
        """Record depth for a fixed duration."""
        self._running = True
        self._start_time = time.time()

        # Binance combined stream URL — using /stream?streams= format
        # This format auto-subscribes without needing a separate SUBSCRIBE message
        streams = [f"{s}@depth20@100ms" for s in self.symbols]
        url = BINANCE_WS_BASE + "/".join(streams)

        def signal_handler(sig, frame):
            logger.info("\nStopping capture...")
            self._running = False
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("=" * 70)
        logger.info("  LIVE DEPTH CAPTURE")
        logger.info("  Symbols:  %s", ", ".join(self.symbols).upper())
        logger.info("  Duration: %.0f seconds (%.1f minutes)", duration_seconds, duration_seconds / 60)
        logger.info("  Output:   %s", self.out_path)
        logger.info("  Press Ctrl+C to stop early (partial data is saved)")
        logger.info("=" * 70)

        import websockets

        while self._running and (duration_seconds == float("inf") or (time.time() - self._start_time) < duration_seconds):
            try:
                # max_size=8MB to handle large depth snapshots
                async with websockets.connect(url, ping_interval=10, max_size=8 * 1024 * 1024) as ws:
                    logger.info("Connected to Binance WebSocket. Recording...")
                    while self._running and (duration_seconds == float("inf") or (time.time() - self._start_time) < duration_seconds):
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            msg = json.loads(raw)
                            self._handle_message(msg)
                        except asyncio.TimeoutError:
                            pass  # check duration periodically

                        # Periodic save + progress
                        if time.time() - self._last_save > self._save_interval:
                            self._save()
                            elapsed = time.time() - self._start_time
                            rate = len(self.records) / max(1, elapsed)
                            logger.info("  %d records (%.1f rec/sec, %.0fs elapsed)",
                                        len(self.records), rate, elapsed)
            except Exception as e:
                logger.error("WebSocket error: %s — reconnecting in 2s...", e)
                await asyncio.sleep(2)

        self._save()
        logger.info("=" * 70)
        logger.info("  CAPTURE COMPLETE")
        logger.info("  Total records: %d", len(self.records))
        logger.info("  Duration:      %.1f seconds", time.time() - self._start_time)
        logger.info("  Output:        %s", self.out_path)
        logger.info("=" * 70)

    def _handle_message(self, msg: dict) -> None:
        """Parse a depth message and append to records.

        Binance combined stream format:
        {"stream": "btcusdt@depth20@100ms", "data": {...}}
        """
        # Combined stream wraps in {"stream": ..., "data": ...}
        if "data" in msg:
            data = msg["data"]
            stream = msg.get("stream", "")
        else:
            data = msg
            stream = msg.get("stream", "")

        # Extract symbol from stream name (e.g., "btcusdt@depth20@100ms" -> "BTCUSDT")
        symbol = ""
        if stream:
            symbol = stream.split("@")[0].upper()
        elif "s" in data:  # Binance uses "s" for symbol in some messages
            symbol = data["s"]
        elif "symbol" in data:
            symbol = data["symbol"]

        if not symbol:
            return

        # Depth20 messages have "bids" and "asks" arrays
        bids = data.get("bids") or data.get("b") or []
        asks = data.get("asks") or data.get("a") or []

        if not bids or not asks:
            return

        ts = int(time.time() * 1000)
        # Binance also sends E (event time) or lastUpdateId — use event time if available
        if "E" in data:
            ts = int(data["E"])
        elif "lastUpdateId" in data:
            pass  # keep our timestamp

        try:
            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            bid_size = float(bids[0][1])
            ask_size = float(asks[0][1])

            bid_depth_10 = sum(float(b[1]) for b in bids[:10])
            ask_depth_10 = sum(float(a[1]) for a in asks[:10])
            total_depth = bid_depth_10 + ask_depth_10
            obi = (bid_depth_10 - ask_depth_10) / total_depth if total_depth > 0 else 0.0
            mid = (best_bid + best_ask) / 2.0
            spread_bps = ((best_ask - best_bid) / mid * 10_000) if mid > 0 else 0.0

            self.records.append({
                "timestamp_ms": ts,
                "symbol": symbol,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "bid_depth_10": bid_depth_10,
                "ask_depth_10": ask_depth_10,
                "obi_10": obi,
                "mid_price": mid,
                "spread_bps": spread_bps,
                "n_levels_bid": len(bids),
                "n_levels_ask": len(asks),
            })
        except (ValueError, IndexError, TypeError) as e:
            logger.debug("Parse error: %s", e)

    def _save(self) -> None:
        """Save current records to parquet."""
        if not self.records:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_parquet(self.out_path, index=False)
        self._last_save = time.time()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Capture live Binance L2 depth data via WebSocket",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/capture_depth_live.py --symbols BTCUSDT,ETHUSDT --hours 2
    python scripts/capture_depth_live.py --symbols BTCUSDT --minutes 30
    python scripts/capture_depth_live.py --symbols BTCUSDT --hours 0
        """,
    )
    parser.add_argument("--symbols", default="BTCUSDT,ETHUSDT",
                        help="Comma-separated symbols (default: BTCUSDT,ETHUSDT)")
    parser.add_argument("--hours", type=float, default=2.0,
                        help="Duration in hours (0 = until Ctrl+C)")
    parser.add_argument("--minutes", type=float, default=0,
                        help="Duration in minutes (overrides --hours)")
    parser.add_argument("--out", default=None,
                        help="Output parquet path")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(",")]

    if args.minutes > 0:
        duration = args.minutes * 60
    elif args.hours > 0:
        duration = args.hours * 3600
    else:
        duration = float("inf")

    if args.out:
        out_path = Path(args.out)
    else:
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = Path(f"data/live_depth_{ts_str}.parquet")

    recorder = DepthRecorder(symbols, out_path)
    asyncio.run(recorder.run(duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
