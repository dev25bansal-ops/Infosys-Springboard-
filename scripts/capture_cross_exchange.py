#!/usr/bin/env python3
"""Live WebSocket capture for Coinbase + Kraken — cross-exchange depth.

Records L2 order book snapshots from Coinbase and Kraken simultaneously,
so you can compute cross-exchange spreads (the Stage 4 Transformer signal).

Usage:
    # Record Coinbase BTC-USD depth for 2 hours
    python scripts/capture_cross_exchange.py --exchange coinbase --hours 2

    # Record Kraken XBT-USD depth for 30 minutes
    python scripts/capture_cross_exchange.py --exchange kraken --minutes 30

    # Record both exchanges simultaneously
    python scripts/capture_cross_exchange.py --exchange both --hours 2
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


class CoinbaseDepthRecorder:
    """Records L2 depth from Coinbase Advanced Trade WebSocket.

    Coinbase uses a different protocol — subscribes to level2_batch channel.
    """

    WS_URL = "wss://ws-feed.exchange.coinbase.com"

    def __init__(self, product_ids: list[str], out_path: Path) -> None:
        self.product_ids = product_ids
        self.out_path = out_path
        self.records: list[dict] = []
        self._running = False
        self._start_time = 0.0
        self._last_save = 0.0

    async def run(self, duration_seconds: float) -> None:
        self._running = True
        self._start_time = time.time()

        def signal_handler(sig, frame):
            self._running = False
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("=" * 70)
        logger.info("  COINBASE DEPTH CAPTURE")
        logger.info("  Products: %s", ", ".join(self.product_ids))
        logger.info("  Duration: %.0f seconds", duration_seconds)
        logger.info("  Output:   %s", self.out_path)
        logger.info("=" * 70)

        import websockets

        subscribe_msg = {
            "type": "subscribe",
            "product_ids": self.product_ids,
            "channels": ["level2_batch"],
        }

        while self._running and (time.time() - self._start_time) < duration_seconds:
            try:
                # max_size=16MB — Coinbase level2_batch can be large
                async with websockets.connect(self.WS_URL, ping_interval=10, max_size=16 * 1024 * 1024) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info("Connected to Coinbase WebSocket. Recording...")

                    while self._running and (time.time() - self._start_time) < duration_seconds:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            msg = json.loads(raw)
                            self._handle_message(msg)
                        except asyncio.TimeoutError:
                            pass

                        if time.time() - self._last_save > 60:
                            self._save()
                            elapsed = time.time() - self._start_time
                            rate = len(self.records) / max(1, elapsed)
                            logger.info("  %d records (%.1f rec/sec, %.0fs elapsed)",
                                        len(self.records), rate, elapsed)
            except Exception as e:
                logger.error("Coinbase WebSocket error: %s — reconnecting in 2s...", e)
                await asyncio.sleep(2)

        self._save()
        logger.info("Capture complete: %d records -> %s", len(self.records), self.out_path)

    def _handle_message(self, msg) -> None:
        """Handle Coinbase level2_batch messages.

        Coinbase sends several message types:
        - Subscriptions confirmations (dict with type='subscriptions')
        - Heartbeats (dict with type='heartbeat')
        - Tickers (dict with type='ticker')
        - Level2 snapshots (dict with type='snapshot')
        - Level2 updates (dict with type='l2update')
        - Some messages come as lists (arrays) — skip those

        Defensive: every .get() call is preceded by an isinstance check.
        """
        # Skip non-dict messages (Coinbase sometimes sends arrays)
        if not isinstance(msg, dict):
            return

        msg_type = msg.get("type")
        ts = int(time.time() * 1000)
        product = msg.get("product_id", "")

        # Only process level2 data — skip subscriptions, heartbeats, tickers
        if msg_type not in ("snapshot", "l2update"):
            return

        try:
            if msg_type == "snapshot":
                # Initial full book snapshot
                # Format: {"type": "snapshot", "product_id": "BTC-USD",
                #          "bids": [["price", "size"], ...], "asks": [["price", "size"], ...]}
                bids = msg.get("bids", [])[:10]
                asks = msg.get("asks", [])[:10]
                self._record_snapshot(ts, product, bids, asks)
            elif msg_type == "l2update":
                # Incremental update
                # Format: {"type": "l2update", "product_id": "BTC-USD",
                #          "changes": [["side", "price", "size"], ...]}
                # NOTE: changes is a list of LISTS, not dicts!
                changes = msg.get("changes", [])
                bids = []
                asks = []
                for change in changes:
                    if not isinstance(change, (list, tuple)) or len(change) < 3:
                        continue
                    side, price, size = change[0], change[1], change[2]
                    if side == "buy":
                        bids.append([price, size])
                    elif side == "sell":
                        asks.append([price, size])
                bids = bids[:10]
                asks = asks[:10]
                if bids or asks:
                    self._record_snapshot(ts, product, bids, asks)
        except Exception as e:
            logger.debug("Handle message error: %s", e)

    def _record_snapshot(self, ts: int, product: str, bids: list, asks: list) -> None:
        try:
            best_bid = float(bids[0][0]) if bids else 0.0
            best_ask = float(asks[0][0]) if asks else 0.0
            bid_depth_10 = sum(float(b[1]) for b in bids[:10])
            ask_depth_10 = sum(float(a[1]) for a in asks[:10])
            obi = (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10) if (bid_depth_10 + ask_depth_10) > 0 else 0.0
            mid = (best_bid + best_ask) / 2.0 if best_bid and best_ask else 0.0
            spread_bps = ((best_ask - best_bid) / mid * 10_000) if mid > 0 else 0.0

            self.records.append({
                "timestamp_ms": ts,
                "exchange": "coinbase",
                "symbol": product,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_depth_10": bid_depth_10,
                "ask_depth_10": ask_depth_10,
                "obi_10": obi,
                "mid_price": mid,
                "spread_bps": spread_bps,
            })
        except (ValueError, IndexError, TypeError):
            pass

    def _save(self) -> None:
        if not self.records:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_parquet(self.out_path, index=False)
        self._last_save = time.time()


class KrakenDepthRecorder:
    """Records L2 depth from Kraken WebSocket.

    Kraken uses the 'book' channel with depth parameter.
    """

    WS_URL = "wss://ws.kraken.com"

    def __init__(self, pairs: list[str], out_path: Path) -> None:
        self.pairs = pairs
        self.out_path = out_path
        self.records: list[dict] = []
        self._running = False
        self._start_time = 0.0
        self._last_save = 0.0

    async def run(self, duration_seconds: float) -> None:
        self._running = True
        self._start_time = time.time()

        def signal_handler(sig, frame):
            self._running = False
        signal.signal(signal.SIGINT, signal_handler)

        logger.info("=" * 70)
        logger.info("  KRAKEN DEPTH CAPTURE")
        logger.info("  Pairs:    %s", ", ".join(self.pairs))
        logger.info("  Duration: %.0f seconds", duration_seconds)
        logger.info("  Output:   %s", self.out_path)
        logger.info("=" * 70)

        import websockets

        # Kraken subscription format
        subscribe_msg = {
            "event": "subscribe",
            "pair": self.pairs,
            "subscription": {"name": "book", "depth": 10},
        }

        while self._running and (time.time() - self._start_time) < duration_seconds:
            try:
                # max_size=16MB + longer ping interval for Kraken
                async with websockets.connect(
                    self.WS_URL,
                    ping_interval=30,
                    ping_timeout=60,
                    max_size=16 * 1024 * 1024,
                    close_timeout=10,
                ) as ws:
                    await ws.send(json.dumps(subscribe_msg))
                    logger.info("Connected to Kraken WebSocket. Recording...")

                    while self._running and (time.time() - self._start_time) < duration_seconds:
                        try:
                            raw = await asyncio.wait_for(ws.recv(), timeout=5.0)
                            self._handle_message(raw)
                        except asyncio.TimeoutError:
                            pass

                        if time.time() - self._last_save > 60:
                            self._save()
                            elapsed = time.time() - self._start_time
                            rate = len(self.records) / max(1, elapsed)
                            logger.info("  %d records (%.1f rec/sec, %.0fs elapsed)",
                                        len(self.records), rate, elapsed)
            except Exception as e:
                logger.error("Kraken WebSocket error: %s — reconnecting in 2s...", e)
                await asyncio.sleep(2)

        self._save()
        logger.info("Capture complete: %d records -> %s", len(self.records), self.out_path)

    def _handle_message(self, raw: str) -> None:
        """Handle Kraken book messages (array format)."""
        try:
            msg = json.loads(raw)
            if not isinstance(msg, list) or len(msg) < 4:
                return
            # Kraken book format: [channelID, [bids], [asks], pair]
            _, book_data, _, pair = msg[0], msg[1], msg[2], msg[-1]
            ts = int(time.time() * 1000)

            # book_data can be {"bs": [...], "as": [...]} or {"b": [...], "a": [...]}
            bids = book_data.get("bs") or book_data.get("b") or []
            asks = book_data.get("as") or book_data.get("a") or []

            if not bids or not asks:
                return

            best_bid = float(bids[0][0])
            best_ask = float(asks[0][0])
            bid_depth_10 = sum(float(b[1]) for b in bids[:10])
            ask_depth_10 = sum(float(a[1]) for a in asks[:10])
            obi = (bid_depth_10 - ask_depth_10) / (bid_depth_10 + ask_depth_10) if (bid_depth_10 + ask_depth_10) > 0 else 0.0
            mid = (best_bid + best_ask) / 2.0
            spread_bps = ((best_ask - best_bid) / mid * 10_000) if mid > 0 else 0.0

            self.records.append({
                "timestamp_ms": ts,
                "exchange": "kraken",
                "symbol": pair,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_depth_10": bid_depth_10,
                "ask_depth_10": ask_depth_10,
                "obi_10": obi,
                "mid_price": mid,
                "spread_bps": spread_bps,
            })
        except Exception:
            pass

    def _save(self) -> None:
        if not self.records:
            return
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame(self.records)
        df.to_parquet(self.out_path, index=False)
        self._last_save = time.time()


async def run_both(duration: float, out_dir: Path) -> None:
    """Record both Coinbase and Kraken simultaneously."""
    ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    coinbase_recorder = CoinbaseDepthRecorder(
        ["BTC-USD", "ETH-USD"],
        out_dir / f"coinbase_depth_{ts_str}.parquet",
    )
    kraken_recorder = KrakenDepthRecorder(
        ["XBT/USD", "ETH/USD"],
        out_dir / f"kraken_depth_{ts_str}.parquet",
    )
    await asyncio.gather(
        coinbase_recorder.run(duration),
        kraken_recorder.run(duration),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture cross-exchange depth data")
    parser.add_argument("--exchange", default="coinbase", choices=["coinbase", "kraken", "both"])
    parser.add_argument("--hours", type=float, default=2.0)
    parser.add_argument("--minutes", type=float, default=0)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    if args.minutes > 0:
        duration = args.minutes * 60
    elif args.hours > 0:
        duration = args.hours * 3600
    else:
        duration = float("inf")

    out_dir = Path(args.out) if args.out else Path("data")
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.exchange == "both":
        asyncio.run(run_both(duration, out_dir))
    elif args.exchange == "coinbase":
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder = CoinbaseDepthRecorder(
            ["BTC-USD", "ETH-USD"],
            out_dir / f"coinbase_depth_{ts_str}.parquet",
        )
        asyncio.run(recorder.run(duration))
    elif args.exchange == "kraken":
        ts_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        recorder = KrakenDepthRecorder(
            ["XBT/USD", "ETH/USD"],
            out_dir / f"kraken_depth_{ts_str}.parquet",
        )
        asyncio.run(recorder.run(duration))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
