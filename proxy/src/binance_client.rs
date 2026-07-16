//! Binance WebSocket client — connects to Binance and parses depth + trade messages.

use std::sync::Arc;
use tokio::sync::Mutex;
use tracing::{debug, info, warn};

use crate::lob::OrderBookReconstructor;

pub struct BinanceClient {
    symbol: String,
    depth: usize,
    book: Arc<Mutex<OrderBookReconstructor>>,
}

impl BinanceClient {
    pub fn new(symbol: String, depth: usize) -> Self {
        Self {
            symbol: symbol.to_lowercase(),
            depth,
            book: Arc::new(Mutex::new(OrderBookReconstructor::new())),
        }
    }

    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        let url = format!(
            "wss://stream.binance.com:9443/ws/{}@depth{}@100ms/{}@trade",
            self.symbol, self.depth, self.symbol
        );

        info!("Connecting to Binance WebSocket: {}", url);

        loop {
            match tokio_tungstenite::connect_async(&url).await {
                Ok((ws_stream, _)) => {
                    info!("Connected to Binance WebSocket");
                    use futures_util::StreamExt;
                    let (_, mut read) = ws_stream.split();

                    while let Some(msg) = read.next().await {
                        match msg {
                            Ok(msg) => {
                                if let tokio_tungstenite::Message::Text(text) = msg {
                                    if let Err(e) = self.handle_message(&text).await {
                                        warn!("Message handling error: {}", e);
                                    }
                                }
                            }
                            Err(e) => {
                                warn!("WebSocket read error: {}", e);
                                break;
                            }
                        }
                    }
                    warn!("WebSocket closed, reconnecting in 1s...");
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
                Err(e) => {
                    warn!("Connection failed: {}, retrying in 1s...", e);
                    tokio::time::sleep(std::time::Duration::from_secs(1)).await;
                }
            }
        }
    }

    async fn handle_message(&self, text: &str) -> Result<(), serde_json::Error> {
        let v: serde_json::Value = serde_json::from_str(text)?;

        // Check if it's a depth update or trade
        if let Some(stream) = v.get("stream").and_then(|s| s.as_str()) {
            let data = &v["data"];
            if stream.contains("depth") {
                self.handle_depth(data).await;
            } else if stream.contains("trade") {
                self.handle_trade(data).await;
            }
        }
        Ok(())
    }

    async fn handle_depth(&self, data: &serde_json::Value) {
        let mut book = self.book.lock().await;

        // Update bid levels
        if let Some(bids) = data.get("bids").and_then(|b| b.as_array()) {
            for level in bids.iter().take(self.depth) {
                if let (Some(price), Some(size)) = (
                    level.get(0).and_then(|p| p.as_str()),
                    level.get(1).and_then(|s| s.as_str()),
                ) {
                    if let (Ok(p), Ok(s)) = (price.parse::<f64>(), size.parse::<f64>()) {
                        book.update_level("bid", p, s);
                    }
                }
            }
        }

        // Update ask levels
        if let Some(asks) = data.get("asks").and_then(|a| a.as_array()) {
            for level in asks.iter().take(self.depth) {
                if let (Some(price), Some(size)) = (
                    level.get(0).and_then(|p| p.as_str()),
                    level.get(1).and_then(|s| s.as_str()),
                ) {
                    if let (Ok(p), Ok(s)) = (price.parse::<f64>(), size.parse::<f64>()) {
                        book.update_level("ask", p, s);
                    }
                }
            }
        }

        // Publish snapshot
        let ts = chrono::Utc::now().timestamp_millis() as u64;
        let snapshot = book.snapshot(ts, self.depth);
        debug!("Depth update: mid={:?}", snapshot.mid_price());
        // In a real impl, we'd publish the snapshot to the TCP server here
    }

    async fn handle_trade(&self, data: &serde_json::Value) {
        let price = data.get("p").and_then(|p| p.as_str()).and_then(|s| s.parse::<f64>().ok());
        let size = data.get("q").and_then(|q| q.as_str()).and_then(|s| s.parse::<f64>().ok());
        let is_buyer_maker = data.get("m").and_then(|m| m.as_bool()).unwrap_or(false);
        let ts = data.get("T").and_then(|t| t.as_u64()).unwrap_or(0);

        if let (Some(p), Some(s)) = (price, size) {
            let side = if is_buyer_maker { "sell" } else { "buy" };
            debug!("Trade: ts={} price={} size={} side={}", ts, p, s, side);
        }
    }
}
