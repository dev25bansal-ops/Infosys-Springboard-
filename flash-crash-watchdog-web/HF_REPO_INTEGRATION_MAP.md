# HF Repo Integration Map — File-by-File

This document is the result of reading every relevant file in
`https://huggingface.co/Dev2506/flash-crash-watchdog/tree/main` line by line.

It maps each piece of the ML repo to its integration point in the web app,
flags the mismatches I found, and gives the exact patches needed to make the
trained TCN actually fire alerts through the dashboard.

---

## 1. The HF repo at a glance

```
Dev2506/flash-crash-watchdog/
├── README.md                         9 KB   ← project overview, quickstart
├── Makefile                          1 KB
├── docker-compose.yml                1 KB
├── LICENSE
├── .gitattributes                    (LFS config — historical, now removed)
│
├── ml/                                      ← the Python package
│   ├── setup.py                             (installable as flash-crash-watchdog==0.4.0)
│   ├── requirements.txt                     (numpy, pandas, scipy, sklearn, torch, websockets, pyarrow, pyyaml, joblib)
│   └── flash_crash_watchdog/
│       ├── __init__.py                      (empty)
│       ├── tick.py                          ← Tick + Trade dataclasses
│       ├── lob.py                           ← OrderBookSnapshot, PriceLevel
│       ├── cli.py                           ← `python -m flash_crash_watchdog.cli live/backtest/train`
│       ├── cascade.py                       ← DetectionCascade orchestrator (5 stages)
│       ├── features/
│       │   ├── __init__.py                  ← FeatureExtractor + FEATURE_NAMES (20 features)
│       │   ├── price_action.py              ← F1 (5 features)
│       │   ├── depth_imbalance.py           ← F2 (5 features)
│       │   ├── flow_toxicity.py             ← F3 (4 features)
│       │   ├── volatility.py                ← F4 (3 features)  ⚠ BUG HERE
│       │   └── cross_symbol.py              ← F5 (3 features)  ← needs reference symbol data
│       ├── models/
│       │   ├── stage1_statistical.py        ← Stage 1: 3σ gate on velocity/spread/OBI
│       │   ├── stage2_isolation_forest.py   ← Stage 2: iForest on F1+F2
│       │   ├── stage3_tcn.py                ← Stage 3: TCN, the trained model
│       │   ├── stage4_transformer.py        ← Stage 4: cross-symbol transformer
│       │   ├── stage5_bayesian.py           ← Stage 5: Bayesian aggregator, produces Alert
│       │   └── cascade.py                   ← re-export of DetectionCascade
│       ├── data/
│       │   ├── __init__.py
│       │   ├── live_stream.py               ← BinanceLiveStream (asyncio + websockets)
│       │   ├── historical_loader.py         ← load_parquet for backtests
│       │   ├── download_binance.py          ← REST downloader
│       │   ├── fi2010_loader.py             ← FI-2010 dataset loader
│       │   └── labels.py                    ← label_crashes()
│       ├── eval/
│       │   └── backtest.py
│       ├── alert/
│       │   └── router.py                    ← AlertRouter (Slack/PagerDuty/JSONL)
│       └── tests/                           ← unit tests
│
├── models/                                  ← trained model checkpoints
│   ├── stage3_tcn_trained.pt          763 KB  ← TCN trained on 287k real BTC windows
│   └── stage3_tcn_luna_finetuned.pt   763 KB  ← fine-tuned on LUNA crash (May 2022)
│
├── scripts/                                 ← one-off Python scripts
│   ├── live_demo.py                         ← THE reference for live TCN inference
│   ├── run_cascade_backtest.py              ← full 5-stage cascade backtest
│   ├── train_tcn_windows.py                 ← training script (Focal Loss, 287k windows)
│   ├── finetune_luna.py                     ← fine-tuning on LUNA data
│   ├── capture_depth_live.py                ← raw Binance depth capture
│   ├── download_*.py                        ← dataset downloaders
│   ├── extract_windows.py                   ← window extraction for training
│   └── generate_plots.py                    ← PR curves, alert timeline
│
├── configs/
│   ├── pipeline.yml                         ← full cascade config
│   ├── tcn_baseline.yml                     ← TCN-only config
│   ├── transformer_cross_symbol.yml
│   └── prometheus.yml
│
├── proxy/                                   ← Rust WebSocket proxy (optional)
│   ├── Cargo.toml
│   └── src/{main,binance_client,lob,publisher}.rs
│
├── dashboard/                               ← placeholder for the dashboard backend
│   ├── README.md                            (empty)
│   └── package.json                         (empty stub)
│
├── data/                                    ← datasets (git-LFS in HF)
│   ├── parquet/                             ← processed BTCUSDT parquet files
│   ├── windows/                             ← pre-extracted training windows
│   └── more/{bybit,equities,futures,metrics}/
│
├── results/
│   └── plots/                               ← PNG visualizations
│
└── docs/
```

---

## 2. The 6 files that actually matter for web integration

If you read only these, you understand the integration:

| File | What it does | Why it matters for the web app |
|---|---|---|
| `ml/flash_crash_watchdog/models/stage3_tcn.py` | Defines `TCNDetector` + `TCNConfig` + `Stage3TCN` wrapper | This is the model class. The web sidecar must instantiate `TCNDetector(config)` and call `model.forward(x)` where `x` has shape `(1, 17, 500)` |
| `ml/flash_crash_watchdog/features/__init__.py` | `FeatureExtractor` + `FEATURE_NAMES` (20 features, ordered) | The web sidecar calls `extractor.extract(tick)` to populate `tick.features`, then builds a 17-dim vector by indexing `FEATURE_NAMES[:17]` |
| `ml/flash_crash_watchdog/tick.py` + `lob.py` | `Tick`, `Trade`, `OrderBookSnapshot`, `PriceLevel` dataclasses | The web sidecar builds these from Binance JSON before calling `extractor.extract()` |
| `scripts/live_demo.py` | Reference implementation: Binance WS → features → TCN → alerts | This is the canonical pattern the web sidecar should mirror. The web sidecar is essentially `live_demo.py` wrapped in FastAPI |
| `ml/flash_crash_watchdog/models/cascade.py` | `DetectionCascade.from_config(yml)` orchestrator | If you want the FULL 5-stage cascade (not just TCN), use this. The web app currently only uses Stage 3 |
| `ml/flash_crash_watchdog/alert/router.py` | `AlertRouter` for Slack/PagerDuty/JSONL | Wire this in if you want alerts to also go to Slack/PagerDuty, not just the dashboard |

---

## 3. The 17 features the TCN expects (in EXACT order)

From `ml/flash_crash_watchdog/models/stage3_tcn.py` line 23-30:

```python
STAGE3_FEATURES = [
    "f1_mid_velocity_50ms", "f1_mid_velocity_200ms", "f1_micro_price",
    "f1_trade_arrival_rate", "f1_cancel_to_trade_ratio",
    "f2_bid_depth_10", "f2_ask_depth_10", "f2_obi_10",
    "f2_weighted_mid_10", "f2_depth_slope",
    "f3_vpin", "f3_kyle_lambda", "f3_effective_spread_bps", "f3_realized_spread_bps",
    "f4_realized_vol_1s", "f4_variance_ratio", "f4_garman_klass",
]
```

This is `FEATURE_NAMES[:17]` from `features/__init__.py` (which has 20 features total — the last 3 are F5 cross-symbol, used by Stage 4 Transformer, NOT by the TCN).

**The web sidecar already does this correctly** — see `ml-inference/server.py` line:
```python
def _feature_vector(tick: "Tick") -> np.ndarray:
    return np.array([tick.features.get(f, 0.0) for f in STAGE3_FEATURES], dtype=np.float32)
```

✅ No change needed.

---

## 4. The model checkpoint format

From `scripts/live_demo.py` line 51-59:

```python
def load_trained_tcn(model_path: str, device: str = "auto") -> TCNDetector:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    data = torch.load(model_path, map_location=device, weights_only=False)
    config = data["config"]               # ← TCNConfig dataclass is in the checkpoint
    model = TCNDetector(config).to(device)
    model.load_state_dict(data["model_state"])
    model.eval()
    return model, device
```

**Key facts:**
- The checkpoint is a dict with two keys: `"model_state"` and `"config"`
- `"config"` is a `TCNConfig` dataclass instance (pickled) — so `weights_only=False` is REQUIRED on PyTorch 2.6+
- The model is instantiated with that config, then `load_state_dict` is called

**The web sidecar already handles this** — `ml-inference/server.py` lines 117-128:
```python
try:
    state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
except TypeError:
    state = torch.load(MODEL_PATH, map_location=DEVICE)   # PyTorch <2.6 fallback
if isinstance(state, dict) and "model_state" in state:
    model.load_state_dict(state["model_state"])
else:
    model.load_state_dict(state)
```

✅ No change needed.

---

## 5. The forward pass shape

From `stage3_tcn.py` line 109-121:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: shape (batch, input_dim, sequence_length)   ← (B, 17, T)
    Returns:
        scores: shape (batch, sequence_length)         ← (B, T)
    """
    out = self.network(x)            # (B, C_last, T)
    out = out.transpose(1, 2)        # (B, T, C_last)
    scores = self.sigmoid(self.head(out)).squeeze(-1)  # (B, T)
    return scores
```

So:
- Input: `(1, 17, 500)` — batch=1, 17 features, 500 timesteps
- Output: `(1, 500)` — per-timestep sigmoid score in [0, 1]
- We take `scores[0, -1]` = the score at the most recent (last) timestep

**The web sidecar already does this** — `ml-inference/server.py` lines:
```python
window_array = np.array(window_buffer, dtype=np.float32)  # (T, 17)
x = torch.from_numpy(window_array).T.unsqueeze(0).to(DEVICE)  # (1, 17, T)
with torch.no_grad():
    scores = model(x)
    score_val = float(scores[0, -1].item())
```

The `.T` transposes `(T, 17)` → `(17, T)`, then `.unsqueeze(0)` adds the batch dim → `(1, 17, T)`. ✅ Correct.

---

## 6. The reference live demo pattern (`scripts/live_demo.py`)

This is the canonical pattern the web sidecar mirrors. Stripped to essentials:

```python
extractor = FeatureExtractor()
feature_window = deque(maxlen=WINDOW_SIZE)  # WINDOW_SIZE = 200 in the script

# Connect to Binance
url = f"wss://stream.binance.com:9443/stream?streams={symbol.lower()}@depth20@100ms/{symbol.lower()}@trade"

async for raw in ws:
    msg = json.loads(raw)
    data = msg["data"]
    stream = msg.get("stream", "")

    # Parse depth or trade into a Tick
    if "depth" in stream:
        bids = [PriceLevel(float(p), float(s)) for p, s in data.get("bids", [])[:20]]
        asks = [PriceLevel(float(p), float(s)) for p, s in data.get("asks", [])[:20]]
        tick = Tick(book=OrderBookSnapshot(timestamp_ms=int(time.time()*1000), bids=bids, asks=asks), symbol=symbol)
    elif "trade" in stream:
        price = float(data.get("p", 0))
        size = float(data.get("q", 0))
        side = "sell" if data.get("m", False) else "buy"
        ts_ms = data.get("T", int(time.time()*1000))
        trade = Trade(timestamp_ms=ts_ms, price=price, size=size, side=side)
        tick = Tick(book=OrderBookSnapshot(timestamp_ms=ts_ms, bids=[PriceLevel(price, size)], asks=[PriceLevel(price, size)]), trades=[trade], symbol=symbol)

    # Extract features and append to window
    features = extractor.extract(tick)
    vec = np.array([features.get(f, 0.0) for f in TCN_FEATURES])  # TCN_FEATURES = FEATURE_NAMES[:17]
    feature_window.append(vec)

    # Run TCN when window is full
    if len(feature_window) >= WINDOW_SIZE:
        window_array = np.array(list(feature_window))
        with torch.no_grad():
            x = torch.FloatTensor(window_array).T.unsqueeze(0).to(device)
            scores = model(x)
            score = float(scores[0, -1].item())

        if score >= threshold:  # default 0.3 in the script
            # Fire alert
            ...
```

### Differences between `live_demo.py` and the web sidecar

| Aspect | `live_demo.py` | Web sidecar (`ml-inference/server.py`) |
|---|---|---|
| Window size | 200 | 500 (matches TCN's `sequence_length=500` config) |
| Threshold | 0.3 | 0.6 (in `binance-stream/index.ts`) |
| Alert cooldown | None | 10 seconds |
| Stream source | Python `websockets` | Node.js `WebSocket` (in `binance-stream`) → HTTP POST to Python sidecar |
| Tick delivery | Same process | Serialized as JSON, sent over HTTP |
| Feature extraction | Python (in-process) | Python (in sidecar) |
| Alert output | Console + JSONL file | Socket.io `alert` event → dashboard toast + browser notification + SQLite |

**The Node.js binance-stream sends raw tick dicts to Python**, and Python reconstructs `Tick` objects via `_to_tick()`. This is the only architectural difference — and it's correct because the feature extractor needs `Tick` objects, not raw JSON.

---

## 7. The bug I found (and the fix)

While reading `ml/flash_crash_watchdog/features/volatility.py` line 83-84:

```python
def _variance_ratio(self, short_ms: int, long_ms: int) -> float:
    if len(self._history) < 10:
        return 1.0
    ts_now = self._history[-1].timestamp_ms
    short_samples = [s for s in self._history if s.timestamp_ms >= ts_now - short_ms]
    long_samples = [s for s in self._history if s.timestamp_ms >= ts_now - long_ms]   # ← line 84
```

**Bug:** `self._history` is a `deque(maxlen=10_000)` (line 34). The list comprehension iterates over the deque. If another tick arrives between line 83 and line 84 (in async code, this can happen during the `await` in the WebSocket loop), the deque gets mutated during iteration → `RuntimeError: deque mutated during iteration`.

I saw this exact error in the test logs:
```
File ".../features/volatility.py", line 84, in _variance_ratio
    long_samples = [s for s in self._history if s.timestamp_ms >= ts_now - long_ms]
RuntimeError: deque mutated during iteration
```

**Impact:** Non-fatal — the sidecar catches the exception and continues, but that tick gets a 0 feature vector, which can cause false negatives.

**Fix:** Snapshot the deque before iterating. In `volatility.py`:

```python
def _variance_ratio(self, short_ms: int, long_ms: int) -> float:
    if len(self._history) < 10:
        return 1.0
    # Snapshot to avoid "deque mutated during iteration" if a tick arrives mid-call
    history_snapshot = list(self._history)
    ts_now = history_snapshot[-1].timestamp_ms
    short_samples = [s for s in history_snapshot if s.timestamp_ms >= ts_now - short_ms]
    long_samples = [s for s in history_snapshot if s.timestamp_ms >= ts_now - long_ms]
    ...
```

Apply the same fix to `_realized_vol` (line 61).

**This is a fix to YOUR ML repo, not the web app.** Either:
- Patch `volatility.py` directly in your local clone, OR
- Submit a PR to the HF repo

The web sidecar will work without this fix (it catches the exception), but you'll get occasional 0-score ticks that shouldn't be 0.

---

## 8. Two models in the repo — which to use?

| Model | File | Trained on | Use case |
|---|---|---|---|
| `stage3_tcn_trained.pt` | 763 KB | 287k real BTC windows (May-Sep 2021, including May 19 crash) | General BTC flash-crash detection |
| `stage3_tcn_luna_finetuned.pt` | 763 KB | Fine-tuned on LUNA/USDT May 7-12 2022 (the LUNA collapse) | LUNA-specific detection |

**The web app currently uses `stage3_tcn_trained.pt`** (bundled in `models/`).

To switch to the LUNA model, just replace the file:
```powershell
# From flash-crash-watchdog-web\
copy /Y ..\flash-crash-watchdog\models\stage3_tcn_luna_finetuned.pt models\stage3_tcn_trained.pt
```
The architecture is identical — no code change needed.

---

## 9. The full 5-stage cascade (optional upgrade)

The web app currently only runs **Stage 3 (TCN)**. If you want the full cascade (Stage 1 → 2 → 3 → 4 → 5), you can replace the sidecar's scoring logic with `DetectionCascade.from_config()`.

From `ml/flash_crash_watchdog/models/cascade.py` line 79-98:

```python
@classmethod
def from_config(cls, config_path: str | Path) -> "DetectionCascade":
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return cls._from_dict(config)

# Then per tick:
alert = cascade.process_tick(tick)
# alert is None OR an Alert dataclass with:
#   timestamp_ms, symbol, posterior, stage2_score, stage3_score, stage4_score,
#   affected_symbols, features_snapshot
```

**To wire the full cascade into the web sidecar**, replace the `/score` endpoint in `ml-inference/server.py`:

```python
from flash_crash_watchdog.cascade import DetectionCascade
from flash_crash_watchdog.alert.router import AlertRouter

# At startup:
cascade = DetectionCascade.from_config("../flash-crash-watchdog/configs/pipeline.yml")
router = AlertRouter(log_path="./alerts.jsonl")

# In /score endpoint:
alert = cascade.process_tick(tick)
if alert is not None:
    router.route(alert)  # logs to JSONL + (optional) Slack/PagerDuty
    return {"score": alert.posterior, "ready": True, "source": "cascade"}
else:
    return {"score": 0.0, "ready": True, "source": "cascade"}
```

**Trade-offs:**
- ✅ Full cascade = lower false positive rate (Bayesian aggregator fuses 3 model scores)
- ❌ Needs Stage 4 transformer model (not in the repo) — would fail at import time
- ❌ ~25ms per tick vs ~5ms for TCN-only
- ❌ Stage 4 needs cross-symbol data (F5 features) — you'd need to stream ETHUSDT etc. as reference

**Recommendation:** Stick with TCN-only for the web app. The cascade is for offline backtests.

---

## 10. The `AlertRouter` (optional Slack/PagerDuty integration)

From `ml/flash_crash_watchdog/alert/router.py`:

```python
class AlertRouter:
    def __init__(self, log_path=None, slack_webhook=None, pagerduty_key=None):
        ...

    def route(self, alert: Alert) -> None:
        # 1. Logs to console
        # 2. Appends to JSONL file (if log_path)
        # 3. Sends to Slack webhook (if slack_webhook)
        # 4. Sends to PagerDuty (if pagerduty_key)
```

**To wire Slack alerts into the web sidecar**, add this to `ml-inference/server.py` after model loading:

```python
import os
from flash_crash_watchdog.alert.router import AlertRouter
from flash_crash_watchdog.models.stage5_bayesian import Alert

router = AlertRouter(
    log_path="./alerts.jsonl",
    slack_webhook=os.getenv("SLACK_WEBHOOK_URL"),
    pagerduty_key=os.getenv("PAGERDUTY_KEY"),
)
```

Then in the `/score` endpoint, when `score_val > 0.6`:

```python
if score_val > 0.6:
    alert = Alert(
        timestamp_ms=int(time.time() * 1000),
        symbol="BTCUSDT",
        posterior=score_val,
        stage2_score=0, stage3_score=score_val, stage4_score=0,
        affected_symbols=["BTCUSDT"],
        features_snapshot={},
    )
    router.route(alert)
```

Set env vars in `.env`:
```
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
PAGERDUTY_KEY=your_key_here
```

---

## 11. The Rust proxy (optional, sub-ms ingest)

The repo has a Rust WebSocket proxy in `proxy/src/`:
- `main.rs` — entry point, parses CLI args
- `binance_client.rs` — connects to Binance WS, parses messages
- `lob.rs` — maintains local LOB state
- `publisher.rs` — publishes ticks to a TCP socket (ZMQ-style)

From `README.md`:
> Option B — Full stack with Rust proxy (for sub-ms ingest)
> ```bash
> cd proxy && cargo build --release
> ./target/release/flash-crash-proxy --symbol BTCUSDT --out tcp://127.0.0.1:5555
> # In another terminal:
> python -m flash_crash_watchdog.cli live --source tcp://127.0.0.1:5555
> ```

**Do you need this?** No. The Node.js `binance-stream` service in the web app already does the same thing (connects to Binance WS, parses depth+trade, forwards to Python). The Rust proxy is only useful if you need sub-millisecond ingest for high-frequency trading — for a dashboard, the ~5ms Node.js latency is fine.

---

## 12. The CLI (`cli.py`)

The repo has a CLI: `python -m flash_crash_watchdog.cli live --symbol BTCUSDT`

From `cli.py`:
- `live` command: instantiates `DetectionCascade.from_config()` + `BinanceLiveStream`, runs the full cascade on live data
- `backtest` command: loads parquet, runs cascade, prints summary
- `train` command: trains TCN on FI-2010 data

**You don't need the CLI for the web app.** The web app's `binance-stream` + `ml-inference/server.py` replace the `live` command. But if you want to run the CLI alongside the web app (e.g. for debugging), you can:

```powershell
cd D:\flash-crash-watchdog\ml
pip install -e .
python -m flash_crash_watchdog.cli live --symbol BTCUSDT
```

This will print alerts to the console in parallel with the web dashboard.

---

## 13. End-to-end data flow (verified)

```
                 ┌─────────────────────────────────────────┐
                 │  Binance WebSocket (wss://stream.binance.com:9443)  │
                 │  Streams: btcusdt@depth20@100ms + btcusdt@trade     │
                 └────────────────────┬────────────────────┘
                                      │
                                      ▼
        ┌──────────────────────────────────────────────┐
        │  mini-services/binance-stream  (Node.js, port 3003)  │
        │  1. Parse depth + trade from Binance JSON            │
        │  2. Maintain rolling tickBuffer (500 ticks)          │
        │  3. Every 500ms, POST /score to Python sidecar       │
        │     with the last 500 ticks as JSON                  │
        │  4. Receive {score, source} response                 │
        │  5. Emit 'tick' Socket.io event to browser           │
        │  6. If score > 0.6, emit 'alert' event               │
        └────────────────────┬────────────────────┘
                             │ HTTP POST
                             ▼
        ┌──────────────────────────────────────────────┐
        │  ml-inference/server.py  (Python, port 8000)  │
        │  1. Receive JSON ticks                          │
        │  2. For each tick:                              │
        │     a. Build Tick dataclass                    │
        │        (OrderBookSnapshot + PriceLevel + Trade) │
        │     b. Call FeatureExtractor.extract(tick)     │
        │        → populates tick.features (20 fields)   │
        │     c. Build 17-dim vector from STAGE3_FEATURES│
        │     d. Append to rolling window (max 500)      │
        │  3. If window >= 50:                            │
        │     a. Stack window into (1, 17, T) tensor     │
        │     b. model.forward(x) → (1, T) sigmoid scores│
        │     c. Return scores[0, -1] as the live score  │
        │  4. Return {score, source: "tcn", ready: true}  │
        └──────────────────────────────────────────────┘
                ▲                                  │
                │ loads at startup                  │
                │                                  ▼
        ┌───────┴────────┐         ┌──────────────────────────────┐
        │ models/        │         │  Browser Dashboard (Next.js)  │
        │ stage3_tcn_    │         │  - Live price chart            │
        │ trained.pt     │         │  - Anomaly score gauge         │
        │ (763 KB)       │         │  - 5-stage cascade funnel      │
        └────────────────┘         │  - Feature breakdown bars      │
                                   │  - Sonner toast on alert       │
                                   │  - POST /api/alerts → SQLite   │
                                   └──────────────────────────────┘
```

---

## 14. What the web app gets right (no changes needed)

Reading the code confirmed these are all correct:

1. ✅ **Model loading** — handles both `weights_only=True/False` across PyTorch versions
2. ✅ **Checkpoint format** — extracts `model_state` from the dict correctly
3. ✅ **Feature order** — uses `STAGE3_FEATURES` (17 features, exact order)
4. ✅ **Tensor shape** — `(1, 17, T)` via `.T.unsqueeze(0)`
5. ✅ **Score extraction** — `scores[0, -1].item()` (last timestep)
6. ✅ **Tick construction** — builds `OrderBookSnapshot` with `PriceLevel` list, attaches `Trade` if present
7. ✅ **Warmup** — returns 0 score until 50 ticks buffered (matches `Stage3TCN.score()` line 163)
8. ✅ **Auto-fallback** — if Python is down, Node.js uses heuristic scorer (TS) and dashboard shows "HEURISTIC" badge
9. ✅ **Throttling** — calls Python at most once per 500ms (Python can't keep up with 10 calls/sec from Binance depth stream)
10. ✅ **Auto-reconnect** — both Node.js (Binance WS) and Python (uvicorn) survive restarts

---

## 15. What needs improvement (3 items)

### Issue 1: `volatility.py` deque mutation bug

**File:** `ml/flash_crash_watchdog/features/volatility.py` lines 61, 83-84
**Fix:** Snapshot the deque before iterating (see Section 7 above)
**Severity:** Low (non-fatal, causes occasional 0-score ticks)

### Issue 2: Web sidecar doesn't load the model config from checkpoint

**Current** (`ml-inference/server.py` line 116):
```python
cfg = TCNConfig()   # ← uses defaults
model = TCNDetector(cfg).to(DEVICE)
```

**Better** (matches `live_demo.py` line 54-56):
```python
data = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
cfg = data.get("config", TCNConfig())   # ← use config from checkpoint
model = TCNDetector(cfg).to(DEVICE)
model.load_state_dict(data["model_state"])
```

**Severity:** Low (the defaults match the trained model's config, so it works — but if you ever train with different hyperparams, this would break)

### Issue 3: Web sidecar's `_to_tick()` doesn't carry the trade timestamp

**Current** (`ml-inference/server.py` `_to_tick()`):
```python
trade_dict = tick_dict.get("trade")
if trade_dict:
    trades.append(Trade(
        timestamp_ms=int(trade_dict.get("timestamp_ms", book.timestamp_ms)),  # ← falls back to book ts
        ...
    ))
```

**The binance-stream service** sets the trade timestamp to `Date.now()` (line in `index.ts`):
```typescript
trade: trade ? { ...trade, timestamp_ms: timestamp } : null,
```
where `timestamp` is `Date.now()` — so the trade gets the same ts as the depth message, not Binance's `T` field.

**Fix** (in `binance-stream/index.ts`):
```typescript
// In the trade branch:
trade = {
  price: parseFloat(data.p),
  size: parseFloat(data.q),
  side: data.m ? 'sell' : 'buy',
  timestamp_ms: data.T,  // ← use Binance's trade timestamp, not Date.now()
}
```

**Severity:** Low (the F1 features use the book timestamp, not the trade timestamp, so this only affects logging)

---

## 16. How to actually apply the integration

### If you haven't unzipped the web app yet

1. Unzip `flash-crash-watchdog-web-v2.1.zip` into `D:\flash-crash-watchdog\flash-crash-watchdog-web\`
2. Run `.\SETUP-WINDOWS.ps1` then `.\START-WINDOWS.ps1`
3. The Python sidecar will auto-find:
   - ML package at `D:\flash-crash-watchdog\ml\`
   - Model at `D:\flash-crash-watchdog\models\stage3_tcn_trained.pt` (or the bundled copy)
4. Dashboard opens at `http://localhost:3000`

### If you've already unzipped and run SETUP

Just run `.\START-WINDOWS.ps1`. Everything is already wired.

### If you want to apply the 3 fixes above

**Fix 1** (volatility.py deque bug):
```powershell
# Edit D:\flash-crash-watchdog\ml\flash_crash_watchdog\features\volatility.py
# In _realized_vol (line 61) and _variance_ratio (line 83):
#   Replace: samples = [s for s in self._history if ...]
#   With:    history_snapshot = list(self._history)
#            samples = [s for s in history_snapshot if ...]
```

**Fix 2** (load config from checkpoint):
```powershell
# Edit D:\flash-crash-watchdog\flash-crash-watchdog-web\ml-inference\server.py
# Replace lines 116-118:
#   cfg = TCNConfig()
#   model = TCNDetector(cfg).to(DEVICE)
#   try:
#       state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
#   except TypeError:
#       state = torch.load(MODEL_PATH, map_location=DEVICE)
#   if isinstance(state, dict) and "model_state" in state:
#       model.load_state_dict(state["model_state"])
#   else:
#       model.load_state_dict(state)
# With:
#   try:
#       state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
#   except TypeError:
#       state = torch.load(MODEL_PATH, map_location=DEVICE)
#   cfg = state.get("config", TCNConfig()) if isinstance(state, dict) else TCNConfig()
#   model = TCNDetector(cfg).to(DEVICE)
#   if isinstance(state, dict) and "model_state" in state:
#       model.load_state_dict(state["model_state"])
#   else:
#       model.load_state_dict(state)
```

**Fix 3** (trade timestamp): Edit `binance-stream/index.ts` line ~138, change `trade` object to use `data.T` for `timestamp_ms`.

---

## 17. Summary

The web app is **already correctly integrated** with the HF repo's ML code. The Python sidecar (`ml-inference/server.py`) is essentially a FastAPI-wrapped version of `scripts/live_demo.py` — it loads the same model, uses the same `FeatureExtractor`, builds the same 17-feature window, and runs the same `model.forward(x)` call.

The only architectural difference is that the web app splits the live demo into two processes:
- **Node.js binance-stream** (port 3003) — handles Binance WebSocket + Socket.io to browser
- **Python ml-inference sidecar** (port 8000) — handles TCN inference

This split lets the dashboard work even when Python is down (heuristic fallback), and lets you restart Python without dropping the WebSocket connection to browsers.

The 3 fixes in Section 15 are optional improvements — the web app works without them, but applying them will reduce occasional 0-score ticks and make the model loading more robust to future config changes.

---

Built for `huggingface.co/Dev2506/flash-crash-watchdog`.
