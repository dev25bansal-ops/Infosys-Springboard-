//! TCP publisher — publishes LOB snapshots to downstream Python consumers.

use std::error::Error;
use tokio::net::TcpListener;
use tracing::{error, info};

pub struct Publisher {
    addr: String,
}

impl Publisher {
    pub fn new(addr: &str) -> Result<Self, Box<dyn Error>> {
        // Strip the tcp:// prefix
        let addr = addr.strip_prefix("tcp://").unwrap_or(addr);
        Ok(Self { addr: addr.to_string() })
    }

    pub async fn run(&self) -> Result<(), Box<dyn Error>> {
        let listener = TcpListener::bind(&self.addr).await?;
        info!("Publisher listening on {}", self.addr);

        loop {
            match listener.accept().await {
                Ok((stream, peer)) => {
                    info!("Client connected: {}", peer);
                    // In a real impl, we'd handle each client connection
                    // and broadcast LOB snapshots to all connected clients
                    tokio::spawn(async move {
                        use tokio::io::AsyncWriteExt;
                        let mut stream = stream;
                        let _ = stream.write_all(b"Flash Crash Proxy ready\n").await;
                        // Keep connection alive
                        loop {
                            tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                        }
                    });
                }
                Err(e) => {
                    error!("Accept error: {}", e);
                }
            }
        }
    }
}
