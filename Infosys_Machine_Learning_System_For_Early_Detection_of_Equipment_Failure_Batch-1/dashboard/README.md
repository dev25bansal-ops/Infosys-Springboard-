# Flash Crash Watchdog — Dashboard

Real-time Next.js dashboard for visualizing the detector's live output.

## Quickstart

```bash
cd dashboard
npm install
npm run dev
# Open http://localhost:3000
```

## Features

- Live LOB depth visualization (top 20 bid/ask levels)
- Real-time anomaly score gauge (Stages 1-5)
- Alert history feed (last 50 alerts)
- Cascade pass-through funnel (Stage 1 → Stage 5)
- Latency monitor (p50, p95, p99)

## Architecture

```
Binance WebSocket → Rust Proxy → Python Cascade → WebSocket (ws://localhost:8000) → Next.js Dashboard
```

The dashboard subscribes to a WebSocket endpoint exposed by the Python cascade
and renders the live state.
