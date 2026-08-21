# Flash Crash Early Warning — Market Microstructure ML Detector

A real-time detector on limit-order-book (LOB) streams that flags microstructure anomalies (order-book imbalance, spoofing, liquidity evaporation) **50–500ms before price dislocation**.

Built as a 5-stage hybrid detection cascade: Statistical pre-filter → Isolation Forest → Temporal Convolutional Network (TCN) → Cross-symbol correlation breakdown → Bayesian aggregator. The operating cascade is S1→S2→S3→S5 (Stage-4's Transformer is **disabled** in the shipped `configs/pipeline.yml`; the validated correlation-breakdown signal is wired as the real Stage-4 input). Operating point: `models/stage3_tcn_prod.pt` @ threshold **0.5**, trailing-realized-vol gate ≥ 2 bps, 10 s cooldown. Held-out crash recall 0.38–1.0 with zero calm-day alerts.

> ⚠️ **Status:** MVP / research prototype. Not production trading software. See [Risks & Disclaimer](#risks--disclaimer).

---

## Architecture

```
Binance WebSocket ─┐
                   ├─→ Feature Extractor ─→ 5-Stage Cascade ─→ Alert Router
FI-2010 (offline) ─┘     (~2 ms)            Stage 1: Statistical    (console/JSONL/
                                                               Stage 2: Isolation Forest    Slack/PD/webhook)
                                                               Stage 3: TCN (~8 ms)
                                                               Stage 4: correlation (disabled-by-default transformer)
                                                               Stage 5: Bayesian agg (~1 ms)
```

The **Rust proxy** (`proxy/`) is an experimental ingest path that currently only opens a TCP port; the live feed uses the Node mini-service (`binance-stream`). See `docs/ANALYSIS-2026-08-11.md` for the full reality-vs-docs audit.

End-to-end p99: ~27 ms (measured, single-tick cascade) · Open-source first.

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
| 4 | Correlation breakdown (anchor-vs-basket return-corr collapse) — the Transformer variant is **disabled** in the shipped config | ~1 ms (corr only) | 60% | Correlation breakdown / decoupling |
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

| Layer | Choice | Status |
|-------|--------|--------|
| Ingest | Node mini-service (Binance WS → Socket.io) — live; Rust proxy is an experimental TCP-only stub | live: Node |
| Stream processing | Python asyncio + Socket.io (MVP) — Flink was design-only | actual |
| Feature store | In-process rolling windows (no external store) | actual |
| Model serving | PyTorch (CPU) + ONNX export available | actual |
| Storage | SQLite (web) + Parquet (research data) — ClickHouse was design-only | actual |
| Dashboard | Next.js + shadcn/ui | actual |
| Monitoring | Prometheus /metrics (sidecar + stream) + Grafana-ready | actual |

---

## Roadmap

- [x] v0.1 — Python-only MVP: Binance ingest, 20 features, TCN, cascade, backtest
- [~] v0.2 — Rust proxy (experimental stub; live ingest uses the Node mini-service)
- [~] v0.3 — Stage 4 (replaced by the validated correlation-breakdown; transformer disabled)
- [x] v0.4 — Next.js live dashboard + hardened auth (signed sessions, rate limits)
- [x] — Corrected evaluation: event-based labels, shared rolling-z, canonical matching, Wilson CIs, ONNX, drift/calibration tooling
- [ ] v1.0 — Production hardening: GPU retrain, full6-day validation, multi-symbol depth

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

MIT — see [`LICENSE`](LICENSE).

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
