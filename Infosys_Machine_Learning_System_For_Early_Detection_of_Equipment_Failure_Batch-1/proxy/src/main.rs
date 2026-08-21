//! Flash Crash Proxy — high-performance WebSocket ingest for Binance LOB streams.
//!
//! Connects to Binance WebSocket, parses depth + trade messages, and publishes
//! normalized LOB snapshots to a TCP socket for downstream Python consumers.
//!
//! Usage:
//!     ./flash-crash-proxy --symbol BTCUSDT --out tcp://127.0.0.1:5555

mod binance_client;
mod lob;
mod publisher;

use clap::Parser;
use tracing::{info, Level};

#[derive(Parser, Debug)]
#[command(name = "flash-crash-proxy", version, about)]
struct Args {
    /// Binance symbol (e.g., BTCUSDT)
    #[arg(long, default_value = "BTCUSDT")]
    symbol: String,

    /// Number of depth levels to track (10 or 20)
    #[arg(long, default_value_t = 20)]
    depth: usize,

    /// Output TCP address (e.g., tcp://127.0.0.1:5555)
    #[arg(long, default_value = "tcp://127.0.0.1:5555")]
    out: String,

    /// Verbose logging
    #[arg(short, long)]
    verbose: bool,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    // Initialize logging
    let level = if args.verbose { Level::DEBUG } else { Level::INFO };
    tracing_subscriber::fmt().with_max_level(level).init();

    info!("Flash Crash Proxy starting");
    info!("  Symbol: {}", args.symbol);
    info!("  Depth:  {} levels", args.depth);
    info!("  Output: {}", args.out);

    // Start the publisher (TCP server)
    let publisher = publisher::Publisher::new(&args.out)?;
    let publisher_task = tokio::spawn(async move {
        if let Err(e) = publisher.run().await {
            tracing::error!("Publisher error: {}", e);
        }
    });

    // Start the Binance WebSocket client
    let client = binance_client::BinanceClient::new(
        args.symbol.clone(),
        args.depth,
    );
    client.run().await?;

    publisher_task.await?;
    Ok(())
}
