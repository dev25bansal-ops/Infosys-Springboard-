# API Reference

## Python API

### `DetectionCascade`

The main entrypoint. Loads from a YAML config file.

```python
from flash_crash_watchdog.cascade import DetectionCascade

cascade = DetectionCascade.from_config("configs/pipeline.yml")

# Register an alert callback
def on_alert(alert):
    print(f"Alert! {alert}")

cascade.on_alert(on_alert)

# Process ticks
alert = cascade.process_tick(tick)
```

### `Tick`

```python
from flash_crash_watchdog.tick import Tick, Trade
from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel

tick = Tick(
    book=OrderBookSnapshot(
        timestamp_ms=1700000000000,
        bids=[PriceLevel(99.5, 1.0)],
        asks=[PriceLevel(100.5, 1.0)],
    ),
    trades=[Trade(timestamp_ms=1700000000000, price=100.0, size=0.5, side="buy")],
    symbol="BTCUSDT",
)
```

### `FeatureExtractor`

```python
from flash_crash_watchdog.features import FeatureExtractor

extractor = FeatureExtractor()
features = extractor.extract(tick)
# Returns a dict of 20 features
```

### `Alert`

```python
@dataclass
class Alert:
    timestamp_ms: int
    symbol: str
    posterior: float          # final probability of anomaly [0, 1]
    stage2_score: float
    stage3_score: float
    stage4_score: float
    affected_symbols: list[str]
    features_snapshot: dict
```

## CLI

```bash
# Live mode
python -m flash_crash_watchdog.cli live --symbol BTCUSDT

# Backtest
python -m flash_crash_watchdog.cli backtest --data data/BTCUSDT_2021-05-19.parquet

# Train
python -m flash_crash_watchdog.cli train --data data/fi2010/ --model configs/tcn_baseline.yml
```

## Rust Proxy CLI

```bash
./flash-crash-proxy --symbol BTCUSDT --depth 20 --out tcp://127.0.0.1:5555
```
