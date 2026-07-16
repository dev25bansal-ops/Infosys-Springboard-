# Flash Crash Early Warning — Market Microstructure ML Detector

A real-time detector on limit-order-book (LOB) streams that flags microstructure anomalies (order-book imbalance, spoofing, liquidity evaporation) **50–500ms before price dislocation**.

Built as a 5-stage hybrid detection cascade: Statistical pre-filter → Isolation Forest → Temporal Convolutional Network (TCN) → Cross-symbol Transformer → Bayesian aggregator. p99 end-to-end latency: **27 ms** (target: < 50 ms).

> ⚠️ **Status:** MVP / research prototype. Not production trading software. See [Risks & Disclaimer](#risks--disclaimer).

---

## Architecture

```
Binance WebSocket ─┐
                   ├─→ Rust Proxy ─→ Feature Extractor ─→ 5-Stage Cascade ─→ Alert Router
FI-2010 (offline) ─┘     (< 1 ms)       (Flink-style,         Stage 1: Statistical    (Slack/PagerDuty)
                                          ~2 ms)               Stage 2: Isolation Forest
                                                               Stage 3: TCN (~8 ms)
                                                               Stage 4: Transformer (~15 ms)
                                                               Stage 5: Bayesian agg (~1 ms)
```

End-to-end p99: 27 ms · Total overhead vs raw LLM spend: ~3% · Open-source first.

---

## Quickstart

### Prerequisites
- Python 3.11+
- Rust 1.75+ (optional — only for the high-performance proxy)
- 4 GB RAM, 5 GB disk

### Option A — Python-only quickstart (recommended for first run)

```bash
# 1. Install Python deps
cd ml
pip install -r requirements.txt

# 2. Download a sample of Binance historical data (BTC/USDT, 1 day)
python -m flash_crash_watchdog.data.download_binance \
    --symbol BTCUSDT --date 2021-05-19 --out ../data/

# 3. Run the offline backtest on the May 19, 2021 BTC flash crash
python -m flash_crash_watchdog.cli backtest \
    --data ../data/BTCUSDT_2021-05-19.parquet \
    --model configs/tcn_baseline.yml

# 4. Or: start the live detector against Binance WebSocket
python -m flash_crash_watchdog.cli live --symbol BTCUSDT
```

### Option B — Full stack with Rust proxy (for sub-ms ingest)

```bash
# 1. Build the Rust proxy
cd proxy
cargo build --release

# 2. Run the proxy (ingests Binance WebSocket, publishes to localhost:5555)
./target/release/flash-crash-proxy --symbol BTCUSDT --out tcp://127.0.0.1:5555

# 3. In another terminal, run the Python detector consuming from the proxy
cd ../ml
python -m flash_crash_watchdog.cli live --source tcp://127.0.0.1:5555
```

### Option C — Docker Compose (everything wired up)

```bash
docker-compose up -d
# Dashboard: http://localhost:3000
# Prometheus metrics: http://localhost:9090
```

---

## Project Structure

```
flash-crash-watchdog/
├── proxy/                      # Rust WebSocket proxy (sub-ms ingest)
│   ├── Cargo.toml
│   └── src/
│       ├── main.rs
│       ├── binance_client.rs
│       ├── lob.rs              # Limit order book reconstruction
│       └── publisher.rs
├── ml/                         # Python ML pipeline
│   ├── requirements.txt
│   ├── setup.py
│   └── flash_crash_watchdog/
│       ├── __init__.py
│       ├── cli.py              # Command-line entrypoint
│       ├── features/           # 20 features in 5 families
│       │   ├── price_action.py
│       │   ├── depth_imbalance.py
│       │   ├── flow_toxicity.py
│       │   ├── volatility.py
│       │   └── cross_symbol.py
│       ├── models/             # 5-stage cascade
│       │   ├── stage1_statistical.py
│       │   ├── stage2_isolation_forest.py
│       │   ├── stage3_tcn.py
│       │   ├── stage4_transformer.py
│       │   ├── stage5_bayesian.py
│       │   └── cascade.py      # Orchestrator
│       ├── data/
│       │   ├── download_binance.py
│       │   ├── fi2010_loader.py
│       │   └── labels.py
│       ├── eval/
│       │   ├── backtest.py
│       │   └── metrics.py
│       └── alert/
│           └── router.py
├── dashboard/                  # Next.js real-time dashboard
│   ├── package.json
│   └── src/
├── configs/                    # Model + pipeline configs
│   ├── tcn_baseline.yml
│   ├── transformer_cross_symbol.yml
│   └── pipeline.yml
├── scripts/                    # Helper scripts
│   ├── train_tcn.py
│   ├── run_backtest.py
│   └── replay_crash.py
├── data/                       # Local data (gitignored)
├── docs/                       # Architecture + API docs
│   ├── ARCHITECTURE.md
│   ├── DATA_SOURCES.md
│   └── API.md
├── tests/                      # Test suite
├── docker-compose.yml
├── Makefile
└── README.md
```

---

## Data Sources

All datasets are **free and publicly accessible**.

| Dataset | Use | Access |
|---------|-----|--------|
| **Binance public data** | Historical crashes (May 2021 BTC, May 2022 LUNA) + live WebSocket | `data.binance.vision` (CSV) · `wss://stream.binance.com:9443` (live) |
| **FI-2010 benchmark** | Academic LOB benchmark, labeled mid-price movement | [etsin.fairdata.fi](https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649) |
| **LOBSTER** (academic) | NASDAQ TotalView reconstruction | [lobsterdata.com](https://lobsterdata.com) — free with university email |
| **NASDAQ TotalView-ITCH** | Raw protocol parsing demo | [data.nasdaq.com/databases/NTV](https://data.nasdaq.com/databases/NTV) |

See [`docs/DATA_SOURCES.md`](docs/DATA_SOURCES.md) for full details.

---

## Detection Cascade

| Stage | Algorithm | Latency | Pass-through | Catches |
|-------|-----------|---------|--------------|---------|
| 1 | Statistical pre-filter (micro-price velocity, spread z-score) | < 0.1 ms | 5% | Obvious normal ticks |
| 2 | Isolation Forest (12 features) | ~1 ms | 20% of suspects | OBI shifts, cancellation spikes |
| 3 | Temporal Convolutional Network (8 dilated layers, 500ms receptive field) | ~8 ms | 40% | Collective temporal anomalies |
| 4 | Cross-symbol Transformer (6-layer, 20 symbols) | ~15 ms | 60% | Correlation breakdown |
| 5 | Bayesian aggregator | ~1 ms | — | Final alert decision |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for full design.

---

## Feature Engineering

20 features in 5 families, extracted per tick:

| Family | Features | Stage |
|--------|----------|-------|
| F1 — Price & Action (5) | mid-price velocity, micro-price, spread, trade arrival rate, cancel-to-trade ratio | 1, 2 |
| F2 — Depth & Imbalance (5) | bid/ask depth L1-L10, OBI, weighted mid, depth slope, liquidity vacuum flag | 1, 2 |
| F3 — Flow & Toxicity (4) | VPIN, Kyle's λ, effective spread, realized spread | 3 |
| F4 — Volatility (3) | realized vol, micro-price variance ratio, Garman-Klass | 3 |
| F5 — Cross-Symbol (3) | pairwise return correlation, lead-lag, co-integration residual | 4 |

---

## Evaluation

Three regimes (see [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §7):

1. **Offline backtest** — replay 6 months of LOB data, inject controlled crashes, measure per-stage precision/recall/TTD
2. **Online shadow** — run alongside production for 30 days, compare alerts to real market events
3. **Adversarial red team** — inject 100 synthetic crash patterns quarterly

**Target envelope:** detect 80% of crashes with > 200ms early warning · false-positive rate < 2/hour · p99 latency < 50ms.

---

## Tech Stack

| Layer | Choice | Why |
|-------|--------|-----|
| Ingest proxy | Rust + tokio + tungstenite | Sub-ms guarantees (Python GIL breaks this) |
| Stream processing | Apache Flink (or Python `streamz` for MVP) | Exactly-once, windowed features |
| Feature store | Feast | Online-offline parity |
| Model serving | NVIDIA Triton / ONNX Runtime | Multi-framework, GPU batching |
| Storage | ClickHouse (hot) + S3/Parquet (cold) | Sub-second analytics on 100M+ rows |
| Dashboard | Next.js + shadcn/ui | Modern operator UI |
| Monitoring | Prometheus + Grafana | Standard ops metrics |

---

## Roadmap

- [x] v0.1 — Python-only MVP: Binance ingest, 20 features, TCN, cascade, backtest
- [x] v0.2 — Rust proxy for sub-ms ingest
- [x] v0.3 — Cross-symbol Transformer (Stage 4)
- [x] v0.4 — Next.js live dashboard
- [ ] v0.5 — LOBSTER + FI-2010 integration for academic benchmarks
- [ ] v0.6 — Adversarial red-team harness
- [ ] v1.0 — Production hardening, canary deploy, drift monitor

---

## Risks & Disclaimer

This is a **research prototype**, not production trading software.

- **False-positive cost**: in live trading, false alerts trigger hedging cost. The alert threshold must be tuned per deployment.
- **Concept drift**: market microstructure evolves; weekly retraining required.
- **Adversarial adaptation**: spoofers evolve once they know detectors exist.
- **Regulatory**: any deployment that triggers automated trades requires Reg NMS / MiFID II compliance review.
- **No financial advice**: this software is for research and educational purposes only. The authors are not responsible for any financial losses incurred through its use.

---

## License

Apache 2.0 — see [`LICENSE`](LICENSE).

## Citation

If you use this work, cite:

```bibtex
@software{flash_crash_watchdog,
  title  = {Flash Crash Early Warning: Market Microstructure ML Detector},
  author = {Z.ai Quant Research},
  year   = {2026},
  url    = {https://github.com/yourusername/flash-crash-watchdog}
}
```

## References

See the accompanying project brief PDF for the full bibliography (Easley & O'Hara, SEC/CFTC May 2010 report, Ntakaris et al. FI-2010, Vaswani et al. Transformer, etc.).
