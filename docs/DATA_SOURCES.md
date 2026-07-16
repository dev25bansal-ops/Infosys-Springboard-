# Data Sources

All datasets are **free and publicly accessible**.

## Primary Datasets

### 1. Binance Public Data ⭐⭐⭐⭐⭐
- **What**: Free historical trade + order-book data for every Binance-listed symbol
- **Granularity**: Trade-by-trade CSVs, daily/monthly aggregates, depth snapshots
- **Real-time**: Free WebSocket (`wss://stream.binance.com:9443`)
- **Coverage**: All Binance symbols, 2017–present
- **Cost**: Free, no auth for public data
- **URLs**:
  - Historical: https://data.binance.vision
  - Live: `wss://stream.binance.com:9443/ws`
  - Docs: https://github.com/binance/binance-public-data
- **Use for**: 80% of the demo. Live stream + historical crash windows (May 2021 BTC, May 2022 LUNA)

### 2. FI-2010 Benchmark ⭐⭐⭐⭐⭐
- **What**: First publicly available HFT LOB benchmark
- **Granularity**: ~10ms ticks, 10 levels of depth
- **Coverage**: 5 Finnish stocks, 10 days
- **Labels**: Pre-annotated mid-price movement (up/down/stationary, 3 horizons)
- **Cost**: Free
- **URL**: https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649
- **Paper**: Ntakaris et al. (2018), Journal of Forecasting
- **Use for**: Academic benchmark, training, comparing to published baselines

### 3. LOBSTER ⭐⭐⭐⭐
- **What**: Reconstructed NASDAQ TotalView-ITCH limit order books
- **Granularity**: Nanosecond, every order event
- **Coverage**: Any NASDAQ stock, 2007–present
- **Cost**: Free for academic use (university email); paid for commercial
- **URL**: https://lobsterdata.com
- **Use for**: Equity-market credibility, May 6, 2010 flash crash reconstruction

### 4. NASDAQ TotalView-ITCH Raw ⭐⭐⭐⭐
- **What**: Raw NASDAQ direct-feed protocol
- **Spec**: Public PDF (nasdaqtrader.com, NQTVITCH 5.0)
- **Parsers**: Open-source (Python, Julia, Rust)
- **Sample**: Free at https://data.nasdaq.com/databases/NTV
- **Cost**: Full historical is paid; samples free
- **Use for**: Low-latency parsing demo

### 5. Tardis.dev ⭐⭐⭐
- **What**: Commercial crypto tick data (L2/L3 snapshots)
- **Coverage**: All major crypto exchanges
- **Cost**: Paid (limited free tier)
- **URL**: https://tardis.dev
- **Use for**: LUNA crash L3 data

## Labeled Crash Windows

### May 6, 2010 — US Equities Flash Crash
- Dow dropped 998.5 points (9.2%) in 36 minutes
- $1 trillion wiped, recovered in 20 minutes
- Source: LOBSTER reconstruction
- SEC/CFTC report: https://www.sec.gov/files/marketevents-report.pdf

### May 19, 2021 — Bitcoin Flash Crash
- BTC -30% in hours
- $8B liquidations
- Binance outage during crash
- Source: Binance public data (data.binance.vision)

### May 2022 — LUNA / UST Death Spiral
- LUNA -99.9% in 48 hours
- $40B wiped
- UST stablecoin depeg
- Source: Binance public data + Tardis.dev

## Data Pipeline

```
Ingest → Normalize → Features → Detect → Alert
  │         │          │          │        │
  │         │          │          │        └── Slack/PagerDuty
  │         │          │          └── 5-stage cascade (27ms p99)
  │         │          └── Flink (20 features, 2ms)
  │         └── Rust proxy (sub-ms)
  └── Binance WebSocket / CSVs
```
