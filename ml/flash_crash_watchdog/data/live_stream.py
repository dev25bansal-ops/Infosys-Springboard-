"""Load Binance WebSocket live stream and emit ticks to the cascade."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Awaitable, Callable

import websockets

from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
from flash_crash_watchdog.tick import Tick, Trade

logger = logging.getLogger(__name__)

BINANCE_WS_BASE = "wss://stream.binance.com:9443/stream?streams="


class BinanceLiveStream:
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        depth_levels: int = 20,
        on_tick: Callable[[Tick], Awaitable[None] | None] | None = None,
    ) -> None:
        self.symbol = symbol.lower()
        self.depth_levels = depth_levels
        self.on_tick = on_tick
        self._running = False
        # ENH-08: carry the last real depth book so a trade-only message can be
        # attached to a real book instead of a fabricated degenerate one.
        self._last_book: OrderBookSnapshot | None = None

    async def run(self) -> None:
        streams = [
            f"{self.symbol}@depth{self.depth_levels}@100ms",
            f"{self.symbol}@trade",
        ]
        url = BINANCE_WS_BASE + "/".join(streams)
        self._running = True
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=10, max_size=8 * 1024 * 1024) as ws:
                    logger.info("Connected to Binance WebSocket: %s", url)
                    async for raw in ws:
                        if not self._running:
                            break
                        await self._handle_message(json.loads(raw))
            except websockets.ConnectionClosed:
                logger.warning("WebSocket closed, reconnecting in 1s...")
                await asyncio.sleep(1)
            except Exception as e:
                logger.error("WebSocket error: %s, reconnecting in 1s...", e)
                await asyncio.sleep(1)

    def stop(self) -> None:
        self._running = False

    async def _handle_message(self, msg: dict) -> None:
        if "stream" not in msg or "data" not in msg:
            return
        stream = msg["stream"]
        data = msg["data"]

        if "depth" in stream:
            tick = self._parse_depth(data)
        elif "trade" in stream:
            tick = self._parse_trade(data)
        else:
            return

        if tick and self.on_tick:
            result = self.on_tick(tick)
            if asyncio.iscoroutine(result):
                await result

    def _parse_depth(self, data: dict) -> Tick | None:
        try:
            # ENH-08: use Binance's event time (E), not the local clock, so the
            # tick timestamp reflects exchange time.
            ts_ms = int(data.get("E", int(time.time() * 1000)))
            bids = [PriceLevel(float(p), float(s)) for p, s in data.get("bids", [])[: self.depth_levels]]
            asks = [PriceLevel(float(p), float(s)) for p, s in data.get("asks", [])[: self.depth_levels]]
            book = OrderBookSnapshot(timestamp_ms=ts_ms, bids=bids, asks=asks)
            self._last_book = book
            return Tick(book=book, symbol=self.symbol.upper())
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Failed to parse depth: %s", e)
            return None

    def _parse_trade(self, data: dict) -> Tick | None:
        try:
            # ENH-08: Binance event time.
            ts_ms = int(data.get("E", data.get("T", 0)))
            price = float(data.get("p", 0))
            size = float(data.get("q", 0))
            is_buyer_maker = data.get("m", False)
            side = "sell" if is_buyer_maker else "buy"
            trade = Trade(timestamp_ms=ts_ms, price=price, size=size, side=side)
            # ENH-08: attach the last REAL depth book — the old code fabricated a
            # single-level book with the trade price on both sides (spread=0,
            # degenerate for downstream features). Fall back to an empty book only
            # when no depth has arrived yet.
            book = self._last_book
            if book is None:
                book = OrderBookSnapshot(timestamp_ms=ts_ms, bids=[], asks=[])
            return Tick(book=book, trades=[trade], symbol=self.symbol.upper())
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Failed to parse trade: %s", e)
            return None
