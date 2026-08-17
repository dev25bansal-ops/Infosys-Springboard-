# Connecting the Web App to the ML Project (v2.1)

In v2.1, the Python sidecar and trained model are already bundled inside the zip. This guide explains the auto-detection logic and what to do if your folder layout is different.

---

## TL;DR - if you unzipped into `D:\flash-crash-watchdog\`

```
D:\flash-crash-watchdog\
+-- ml\flash_crash_watchdog\        <-- your existing ML package
+-- models\stage3_tcn_trained.pt    <-- your trained model (if present)
+-- scripts\
+-- flash-crash-watchdog-web\       <-- unzipped here
    +-- ml-inference\server.py      <-- auto-finds ML package at ..\flash-crash-watchdog\ml
    +-- models\stage3_tcn_trained.pt <-- backup copy bundled in zip
    +-- SETUP-WINDOWS.ps1
    +-- START-WINDOWS.ps1
```

Just run:
```powershell
cd D:\flash-crash-watchdog\flash-crash-watchdog-web
powershell -ExecutionPolicy Bypass -File .\SETUP-WINDOWS.ps1
powershell -ExecutionPolicy Bypass -File .\START-WINDOWS.ps1
```

The Python sidecar will:
1. Look for the ML package at `../flash-crash-watchdog/ml/` then **found** (your existing repo)
2. Look for the model at `../flash-crash-watchdog/models/stage3_tcn_trained.pt` then **found** if you have it
3. Fall back to `./models/stage3_tcn_trained.pt` then **bundled in zip** (764 KB)

Everything wires automatically.

---

## How the auto-detection works

In `ml-inference/server.py`:

```python
HERE = Path(__file__).resolve().parent                 # ml-inference/
PROJECT_ROOT = HERE.parent                              # flash-crash-watchdog-web/
ML_PACKAGE_ROOT = PROJECT_ROOT.parent / "flash-crash-watchdog" / "ml"   # ../flash-crash-watchdog/ml/
MODEL_PATH = PROJECT_ROOT.parent / "flash-crash-watchdog" / "models" / "stage3_tcn_trained.pt"

# Fallbacks if not found:
if not MODEL_PATH.exists():
    MODEL_PATH = PROJECT_ROOT / "models" / "stage3_tcn_trained.pt"   # inside web app folder
if not MODEL_PATH.exists():
    MODEL_PATH = HERE / "models" / "stage3_tcn_trained.pt"           # inside ml-inference/
```

If the ML package can't be imported (e.g. you placed the web app somewhere else), the sidecar logs a warning and runs in **fallback mode** - it uses a built-in heuristic scorer based on price velocity. Alerts still fire, but they're not from the trained TCN. The dashboard will show "HEURISTIC" instead of "TCN" next to the anomaly score.

To verify what mode the sidecar is in, visit `http://localhost:8000/health`:

```json
{
  "ok": true,
  "model_loaded": true,
  "model_path": "D:\\flash-crash-watchdog\\models\\stage3_tcn_trained.pt",
  "ml_package_available": true,
  "device": "cpu",
  "window_size": 0,
  "warmup_target": 50,
  "window_target": 500
}
```

---

## If your layout is different

### Case A - web app is standalone (no parent ML repo)

If you unzipped the web app into a folder by itself (e.g. `D:\flash-crash-watchdog-web\` without a parent `D:\flash-crash-watchdog\`), the sidecar will:

1. Fail to find `../flash-crash-watchdog/ml/` then log a warning
2. Fall back to `./models/stage3_tcn_trained.pt` then **found** (bundled in zip)
3. Try to import the ML package then **fail** (no `flash_crash_watchdog` module)
4. Run in **fallback mode** (heuristic scorer)

To use the real TCN in this layout, copy your ML package into the web app folder:

```powershell
# From D:\flash-crash-watchdog-web\
xcopy /E /I D:\path\to\your\flash-crash-watchdog\ml ml_package
```

Then edit `ml-inference/server.py` line ~32:

```python
ML_PACKAGE_ROOT = HERE / "ml_package"   # instead of PROJECT_ROOT.parent / "flash-crash-watchdog" / "ml"
```

### Case B - you have a different model filename

Edit `MODEL_PATH` in `ml-inference/server.py` to point to your file. Supported formats:
- PyTorch state_dict: `{"model_state": ..., "config": TCNConfig}`
- Raw state_dict: just the weights

### Case C - you want to use a different model entirely (e.g. the LUNA fine-tuned)

Replace `models/stage3_tcn_trained.pt` with `models/stage3_tcn_luna_finetuned.pt` (or whichever), then edit `MODEL_PATH` in `server.py`. The TCN architecture is the same.

---

## The 17 features the TCN expects

The model consumes these 17 features per timestep (in this exact order). Your existing `ml/flash_crash_watchdog/features/__init__.py` already produces all of them via `FeatureExtractor.extract(tick)`:

| # | Feature                       | Family        |
|---|-------------------------------|---------------|
| 1 | `f1_mid_velocity_50ms`        | Price action  |
| 2 | `f1_mid_velocity_200ms`       | Price action  |
| 3 | `f1_micro_price`              | Price action  |
| 4 | `f1_trade_arrival_rate`        | Price action  |
| 5 | `f1_cancel_to_trade_ratio`     | Price action  |
| 6 | `f2_bid_depth_10`              | Depth         |
| 7 | `f2_ask_depth_10`              | Depth         |
| 8 | `f2_obi_10`                    | Depth         |
| 9 | `f2_weighted_mid_10`           | Depth         |
|10 | `f2_depth_slope`               | Depth         |
|11 | `f3_vpin`                      | Flow toxicity |
|12 | `f3_kyle_lambda`               | Flow toxicity |
|13 | `f3_effective_spread_bps`      | Flow toxicity |
|14 | `f3_realized_spread_bps`       | Flow toxicity |
|15 | `f4_realized_vol_1s`           | Volatility    |
|16 | `f4_variance_ratio`            | Volatility    |
|17 | `f4_garman_klass`              | Volatility    |

The sidecar builds a `Tick` object from each Binance depth+trade message, runs `FeatureExtractor.extract(tick)`, then stacks the last 500 timesteps into a `(1, 17, 500)` tensor and forwards it through the TCN.

---

## The `/score` HTTP contract

Request:
```json
POST http://localhost:8000/score
Content-Type: application/json

{
  "ticks": [
    {
      "timestamp_ms": 1698000000000,
      "price": 64716.01,
      "bids": [["64716.00", "0.5"], ...],
      "asks": [["64716.50", "0.3"], ...],
      "symbol": "BTCUSDT",
      "trade": {
        "price": 64716.01,
        "size": 0.012,
        "side": "buy",
        "timestamp_ms": 1698000000000
      }
    }
  ]
}
```

Response:
```json
{
  "score": 0.732,
  "ready": true,
  "source": "tcn",
  "window_size": 500,
  "device": "cpu",
  "latency_ms": 4.2
}
```

The Node.js binance-stream service calls this endpoint with the last 500 ticks every depth message, with a 2-second timeout. If Python doesn't respond in time, it falls back to the heuristic. Calls are throttled to 1 per 500ms.

---

## Replacing the alert threshold / cooldown

In `mini-services/binance-stream/index.ts` line ~165:

```typescript
if (score > 0.6 && (now - lastAlertTime) > 10000) {
```

- `0.6` = alert threshold (raise to 0.7 for fewer false positives)
- `10000` = 10-second cooldown (ms)

---

## Multi-symbol setup

To monitor multiple symbols, edit the stream URL in `mini-services/binance-stream/index.ts`:

```typescript
const symbols = ['btcusdt', 'ethusdt', 'solusdt']
const url = 'wss://stream.binance.com:9443/stream?streams=' +
  symbols.flatMap(s => [`${s}@depth20@100ms`, `${s}@trade`]).join('/')
```

Then in the message handler, parse the symbol from `msg.stream` and track per-symbol price/score buffers.

---

## What's NOT included (intentional)

- **Cross-symbol features (F5)** - the TCN was trained on F1-F4 only (17 features). The F5 family (pairwise correlation, lead-lag, cointegration) is consumed by Stage 4 (Transformer), which is a separate model not bundled here.
- **The full cascade** - only Stage 3 (TCN) is wired. Stages 1 (Statistical), 2 (iForest), 4 (Transformer), 5 (Bayesian) are skipped. The dashboard's cascade funnel card is a visual representation of where the score would reach in the full cascade, computed heuristically from the TCN score.
- **GPU batching** - the sidecar runs single-tick inference. For high-throughput production, wrap with `gunicorn -k uvicorn.workers.UvicornWorker -w 4` or use Triton Inference Server.

---

Built for `huggingface.co/Dev2506/flash-crash-watchdog`.
