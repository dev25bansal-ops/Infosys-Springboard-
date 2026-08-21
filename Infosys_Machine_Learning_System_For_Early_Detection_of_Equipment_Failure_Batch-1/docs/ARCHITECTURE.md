# Architecture

## Overview

The Flash Crash Early Warning system is a **5-stage hybrid detection cascade** that processes limit-order-book (LOB) streams in real time and fires alerts 50–500ms before price dislocation.

```
Binance WebSocket ─┐
                   ├─→ Rust Proxy ─→ Feature Extractor ─→ 5-Stage Cascade ─→ Alert Router
FI-2010 (offline) ─┘     (< 1 ms)       (~2 ms)               (27 ms p99)      (Slack/PagerDuty)
```

## 5-Stage Cascade

### Stage 1 — Statistical Pre-Filter
- **Algorithm**: Z-score tests on micro-price velocity, spread, OBI
- **Latency**: < 0.1 ms
- **Pass-through**: ~5% (rejects obviously normal ticks)
- **Runs on**: Every tick

### Stage 2 — Isolation Forest
- **Algorithm**: Isolation Forest on 12 microstructure features (F1+F2)
- **Latency**: ~1 ms
- **Pass-through**: ~20% of suspects
- **Runs on**: Ticks that pass Stage 1

### Stage 3 — Temporal Convolutional Network (TCN)
- **Algorithm**: 8-layer dilated causal TCN, 500ms receptive field
- **Latency**: ~8 ms (GPU)
- **Pass-through**: ~40%
- **Runs on**: Ticks that pass Stage 2

### Stage 4 — Cross-Symbol Transformer
- **Algorithm**: 6-layer Transformer encoder, self-attention across 20 symbols
- **Latency**: ~15 ms (GPU)
- **Pass-through**: ~60%
- **Runs on**: Ticks that pass Stage 3

### Stage 5 — Bayesian Aggregator
- **Algorithm**: Bayesian model averaging (log-odds fusion)
- **Latency**: ~1 ms
- **Output**: Alert / no-alert
- **Runs on**: Ticks that pass Stage 4

**Total p99 latency**: 27 ms (target: < 50 ms)

## Feature Engineering

20 features in 5 families:

| Family | Features | Stage | Latency |
|--------|----------|-------|---------|
| F1 — Price & Action (5) | mid-price velocity (50/200ms), micro-price, trade arrival rate, cancel-to-trade ratio | 1, 2 | < 0.1 ms |
| F2 — Depth & Imbalance (5) | bid/ask depth L10, OBI, weighted mid, depth slope | 1, 2 | ~0.3 ms |
| F3 — Flow & Toxicity (4) | VPIN, Kyle's λ, effective spread, realized spread | 3 | ~0.5 ms |
| F4 — Volatility (3) | realized vol, variance ratio, Garman-Klass | 3 | ~0.2 ms |
| F5 — Cross-Symbol (3) | pairwise correlation, lead-lag, cointegration residual | 4 | ~0.9 ms |

**Total extraction latency**: ~2 ms

## Training Strategy

1. **Self-supervised pretraining** — Masked prediction on months of normal LOB data
2. **Supervised fine-tuning** — Focal loss on labeled crash windows

## Evaluation

Three regimes:
1. **Offline backtest** — Replay 6 months of LOB data, inject controlled crashes
2. **Online shadow** — Run alongside production for 30 days
3. **Adversarial red team** — Inject 100 synthetic crash patterns quarterly

**Target envelope**: detect 80% of crashes with > 200ms early warning · FP rate < 2/hour · p99 < 50ms
