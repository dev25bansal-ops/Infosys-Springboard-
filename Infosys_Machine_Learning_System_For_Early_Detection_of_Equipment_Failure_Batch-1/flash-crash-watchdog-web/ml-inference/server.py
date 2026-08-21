"""FastAPI sidecar that loads the trained TCN and exposes /score."""
from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

HERE = Path(__file__).resolve().parent                 # ml-inference/
PROJECT_ROOT = HERE.parent                              # flash-crash-watchdog-web/

# Try several common ML package locations — pick the first that contains
# a `flash_crash_watchdog/` subdirectory.
_ML_CANDIDATES = [
    PROJECT_ROOT.parent / "flash-crash-watchdog" / "ml",   # ../flash-crash-watchdog/ml/  (HF repo layout)
    PROJECT_ROOT.parent / "ml",                            # ../ml/                       (your layout)
    PROJECT_ROOT / "ml",                                   # ./ml/                        (inside web app)
    PROJECT_ROOT / "ml_package",                           # ./ml_package/                (manual copy)
    HERE / "ml",                                           # ./ml-inference/ml/
]
ML_PACKAGE_ROOT = next(
    (p for p in _ML_CANDIDATES if (p / "flash_crash_watchdog" / "__init__.py").exists()),
    _ML_CANDIDATES[0],  # fall back to the first candidate (will fail import, but logs cleanly)
)

# Same multi-location search for the model file. The operating checkpoint comes
# from configs/operating.yml (single source of truth, MLOPS-08); the hardcoded
# list is a fallback for when operating.yml is absent.
def _operating_model_path() -> Optional[Path]:
    import yaml
    cfg = PROJECT_ROOT.parent / "configs" / "operating.yml"
    try:
        if cfg.exists():
            op = yaml.safe_load(cfg.read_text(encoding="utf-8"))
            m = (op or {}).get("model")
            if m:
                p = PROJECT_ROOT.parent / "models" / m
                if p.exists():
                    return p
    except Exception:
        pass
    return None


_OP_MODEL = _operating_model_path()
_MODEL_CANDIDATES = ([_OP_MODEL] if _OP_MODEL else []) + [
    PROJECT_ROOT.parent / "flash-crash-watchdog" / "models" / "stage3_tcn_prod.pt",
    PROJECT_ROOT.parent / "models" / "stage3_tcn_prod.pt",
    PROJECT_ROOT / "models" / "stage3_tcn_prod.pt",
    HERE / "models" / "stage3_tcn_prod.pt",
    PROJECT_ROOT.parent / "models" / "stage3_tcn_v2.pt",
    PROJECT_ROOT.parent / "models" / "stage3_tcn_oos.pt",
    PROJECT_ROOT / "models" / "stage3_tcn_trained.pt",
]
MODEL_PATH = next((p for p in _MODEL_CANDIDATES if p.exists()), _MODEL_CANDIDATES[2])


def _load_dotenv() -> None:
    """BUG-14/MLOPS-08: load the web-root .env into os.environ (if not already set).

    The launcher starts uvicorn without exporting ALERT_THRESHOLD /
    MIN_TRAILING_VOL_BPS / ALERT_COOLDOWN_MS, so the operating point below was
    dead config. A real environment variable always wins.
    """
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        return
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = val
    except Exception as e:  # never break startup on a malformed .env
        log.warning("Could not load .env: %s", e)


_load_dotenv()


def _verify_model_hash(model_path: Path) -> None:
    """MLOPS-05: fail-closed sha256 pin against models/manifest.json.

    If the manifest has an entry for this model and the on-disk file differs,
    refuse to load. A missing manifest fails closed in production (NODE_ENV=
    production), warns in dev. Regenerate with scripts/model_manifest.py.
    """
    import hashlib
    import json

    manifest = PROJECT_ROOT.parent / "models" / "manifest.json"
    if not manifest.exists():
        if os.environ.get("NODE_ENV") == "production":
            raise RuntimeError("models/manifest.json missing — refusing to load an unpinned model (MLOPS-05)")
        log.warning("models/manifest.json missing — model hash not pinned (MLOPS-05)")
        return
    known = json.loads(manifest.read_text(encoding="utf-8")).get("models", {})
    entry = known.get(model_path.name)
    if entry is None:
        log.warning("model %s not listed in models/manifest.json — regenerate with "
                    "scripts/model_manifest.py (MLOPS-05)", model_path.name)
        return
    h = hashlib.sha256(model_path.read_bytes()).hexdigest()
    if h != entry["sha256"]:
        raise RuntimeError(
            f"model hash mismatch for {model_path.name} (MLOPS-05) — refusing to load "
            "(regenerate models/manifest.json if the checkpoint was intentionally replaced)"
        )
    log.info("model hash verified against manifest (MLOPS-05)")

# Operating point surfaced to the UI (matches configs/operating.yml / live mini-stream).
MODEL_KEY = MODEL_PATH.name if MODEL_PATH.exists() else "none"
OP_THRESHOLD = float(os.environ.get("ALERT_THRESHOLD", "0.5"))
OP_GATE_BPS = float(os.environ.get("MIN_TRAILING_VOL_BPS", "2"))
OP_COOLDOWN_MS = int(os.environ.get("ALERT_COOLDOWN_MS", "10000"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(name)s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("tcn-sidecar")
log.info("Project root: %s", PROJECT_ROOT)
log.info("ML package root: %s", ML_PACKAGE_ROOT)
log.info("Model path: %s", MODEL_PATH)
if not (ML_PACKAGE_ROOT / "flash_crash_watchdog" / "__init__.py").exists():
    log.warning("ML package not found in any of these locations:")
    for p in _ML_CANDIDATES:
        log.warning("  - %s", p)
    log.warning("Fix: copy your ml/ folder to one of these paths, OR set ML_PACKAGE_ROOT manually in server.py")

sys.path.insert(0, str(ML_PACKAGE_ROOT))

try:
    from flash_crash_watchdog.models.stage3_tcn import (
        STAGE3_FEATURES, TCNConfig, TCNDetector, Stage3TCN,
    )
    from flash_crash_watchdog.features import FeatureExtractor
    from flash_crash_watchdog.tick import Tick, Trade
    from flash_crash_watchdog.lob import OrderBookSnapshot, PriceLevel
    ML_AVAILABLE = True
    log.info("Loaded ML package from %s", ML_PACKAGE_ROOT)
except Exception as e:
    ML_AVAILABLE = False
    log.warning("ML package not available (%s) - sidecar will fall back to heuristic.", e)
    STAGE3_FEATURES = []
    FeatureExtractor = None
    Tick = None
    Trade = None
    OrderBookSnapshot = None
    PriceLevel = None
    PriceLevel = None
    TCNConfig = None
    TCNDetector = None
    Stage3TCN = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
WARMUP = 50  # ticks of feature history before the TCN produces a score

model: Optional["TCNDetector"] = None
s3: Optional["Stage3TCN"] = None   # rolling-window + rolling-z normalize + score (matches training)
extractor: Optional[Any] = None

if ML_AVAILABLE and MODEL_PATH.exists():
    try:
        # MLOPS-05: fail-closed sha256 pin before loading.
        _verify_model_hash(MODEL_PATH)
        # Secure load (MLOPS-06): weights_only=True + the TCNConfig dataclass
        # allowlisted — never unpickle arbitrary objects from a checkpoint file.
        torch.serialization.add_safe_globals([TCNConfig])
        state = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        cfg = state["config"] if isinstance(state, dict) and state.get("config") is not None else TCNConfig()
        s3 = Stage3TCN(cfg, device=DEVICE)
        if isinstance(state, dict) and "model_state" in state:
            s3.model.load_state_dict(state["model_state"])
        else:
            s3.model.load_state_dict(state)
        s3.model.eval()
        model = s3.model
        extractor = FeatureExtractor()
        log.info("Loaded TCN model (%s) on %s (config: %s)", MODEL_PATH.name, DEVICE.upper(), cfg)
    except Exception as e:
        log.error("Failed to load TCN model: %s", e)
        model = None
        s3 = None
        extractor = None
else:
    if not MODEL_PATH.exists():
        log.warning("Model file not found at %s - running in FALLBACK mode (heuristic).", MODEL_PATH)


def _to_price_level(pair: List[str]) -> "PriceLevel":
    return PriceLevel(price=float(pair[0]), size=float(pair[1]))


def _to_tick(tick_dict: dict) -> Optional["Tick"]:
    if not ML_AVAILABLE:
        return None
    bids_raw = tick_dict.get("bids", [])[:10]
    asks_raw = tick_dict.get("asks", [])[:10]
    if not bids_raw or not asks_raw:
        return None
    book = OrderBookSnapshot(
        timestamp_ms=int(tick_dict.get("timestamp_ms", time.time() * 1000)),
        bids=[_to_price_level(p) for p in bids_raw],
        asks=[_to_price_level(p) for p in asks_raw],
    )
    trade_dict = tick_dict.get("trade")
    trades: list = []
    if trade_dict:
        trades.append(Trade(
            timestamp_ms=int(trade_dict.get("timestamp_ms", book.timestamp_ms)),
            price=float(trade_dict["price"]),
            size=float(trade_dict.get("size", 0)),
            side=trade_dict.get("side", "buy"),
        ))
    tick = Tick(book=book, trades=trades, symbol=tick_dict.get("symbol", "BTCUSDT"))
    # Defensive: if feature extraction throws (e.g. due to a bug in one of the
    # feature modules), still return the tick with whatever features got populated.
    # A partial feature vector is better than a 500 error that kills the request.
    try:
        extractor.extract(tick)
    except Exception as e:
        log.warning("Feature extraction failed for tick (will use partial features): %s", e)
    return tick


def _feature_vector(tick: "Tick") -> np.ndarray:
    return np.array([tick.features.get(f, 0.0) for f in STAGE3_FEATURES], dtype=np.float32)


def heuristic_score(ticks: List[dict]) -> float:
    if len(ticks) < 30:
        return 0.0
    prices = [float(t["price"]) for t in ticks if "price" in t]
    if len(prices) < 30:
        return 0.0
    recent = prices[-10:]
    baseline = prices[-50:-10]
    if not baseline:
        return 0.0
    recent_avg = sum(recent) / len(recent)
    baseline_avg = sum(baseline) / len(baseline)
    velocity = abs((recent_avg - baseline_avg) / baseline_avg) * 100 if baseline_avg > 0 else 0
    return float(min(1.0, velocity / 1.5))


app = FastAPI(title="Flash Crash Watchdog - TCN Inference", version="1.0.0")
# SEC-3/4 hardening: restrict CORS to localhost (the sidecar is local-only) —
# previously `allow_origins=["*"]`. Server-to-server callers (mini-stream) don't
# send a browser Origin, so this only tightens browser exposure.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000", "http://localhost:3003",
        "http://127.0.0.1:3000", "http://127.0.0.1:3003",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)
TCN_API_KEY = os.environ.get("TCN_API_KEY", "")
if TCN_API_KEY:
    log.info("Inference sidecar requires x-api-key (TCN_API_KEY set)")
else:
    # SEC-8: without a shared key, /score and /reset are unauthenticated on the
    # local loopback — any local process/page can reset the window (suppressing
    # alerts) or spam scoring. Set TCN_API_KEY in .env and it is enforced.
    log.warning(
        "TCN_API_KEY is NOT set — /score and /reset are unauthenticated (localhost-only). "
        "Set TCN_API_KEY in .env to enforce the shared-secret check."
    )

# NEW-02: Prometheus metrics for the live path (configs/prometheus.yml scrapes this).
SCORE_REQUESTS = Counter("tcn_score_requests_total", "score requests received")
SCORE_LATENCY = Histogram("tcn_score_latency_seconds", "score request latency",
                          buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0))
ALERTS_EMITTED = Counter("tcn_alerts_total", "alerts emitted by the sidecar")


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


class ScoreRequest(BaseModel):
    ticks: List[dict]


class ScoreResponse(BaseModel):
    score: float
    ready: bool
    source: str
    window_size: int
    device: str
    latency_ms: float
    model_key: str = ""      # which checkpoint is live (e.g. stage3_tcn_prod.pt)
    threshold: float = 0.5   # operating Stage-3 gate
    gate_bps: float = 2.0    # operating calm-day trailing-vol regime gate
    cooldown_ms: int = 10000


@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None,
        "ml_package_available": ML_AVAILABLE,
        "device": DEVICE,
        "window_size": len(s3._window) if s3 else 0,
        "warmup_target": WARMUP,
        "window_target": s3._max_window if s3 else 200,
    }


@app.get("/version")
def version() -> dict:
    return {
        "service": "flash-crash-watchdog ml-inference",
        "model_key": MODEL_KEY,
        "device": DEVICE,
        "operating": {
            "threshold": OP_THRESHOLD,
            "gate_bps": OP_GATE_BPS,
            "cooldown_ms": OP_COOLDOWN_MS,
        },
    }


@app.post("/score", response_model=ScoreResponse)
def score(req: ScoreRequest, request: Request) -> ScoreResponse:
    if TCN_API_KEY and request.headers.get("x-api-key") != TCN_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    SCORE_REQUESTS.inc()
    t0 = time.perf_counter()
    if not req.ticks:
        return ScoreResponse(score=0.0, ready=False, source="warmup",
                              window_size=(len(s3._window) if s3 else 0),
                              device=DEVICE, latency_ms=0.0)

    if s3 is not None and extractor is not None and ML_AVAILABLE:
        # Feed every tick through the Stage3TCN wrapper: it applies the same
        # rolling-z normalization as training and maintains a CONTIGUOUS window.
        for td in req.ticks:
            tick = _to_tick(td)
            if tick is None:
                continue
            s3.feed(tick)
        if len(s3._window) < WARMUP:
            return ScoreResponse(
                score=0.0, ready=False, source="warmup",
                window_size=len(s3._window), device=DEVICE,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        # BUG-04: `ready` must mean a FULL trained-length window is available.
        # score_current() returns (0.0, False) until len(_window) reaches the
        # checkpoint's sequence_length, so with only the WARMUP=50 gate the
        # sidecar reported ready=true with a fabricated score 0.0 for ticks
        # 50..window-1 (and the mini-stream stamped those as source='tcn').
        if len(s3._window) < s3._max_window:
            return ScoreResponse(
                score=0.0, ready=False, source="warmup",
                window_size=len(s3._window), device=DEVICE,
                latency_ms=(time.perf_counter() - t0) * 1000,
            )
        score_val, _ready = s3.score_current()
        SCORE_LATENCY.observe(time.perf_counter() - t0)
        return ScoreResponse(
            score=score_val, ready=True, source="tcn",
            window_size=len(s3._window), device=DEVICE,
            latency_ms=(time.perf_counter() - t0) * 1000,
            model_key=MODEL_KEY, threshold=OP_THRESHOLD,
            gate_bps=OP_GATE_BPS, cooldown_ms=OP_COOLDOWN_MS,
        )

    s = heuristic_score(req.ticks)
    return ScoreResponse(
        score=s, ready=True, source="fallback",
        window_size=len(req.ticks), device=DEVICE,
        latency_ms=(time.perf_counter() - t0) * 1000,
        model_key=MODEL_KEY, threshold=OP_THRESHOLD,
        gate_bps=OP_GATE_BPS, cooldown_ms=OP_COOLDOWN_MS,
    )


@app.post("/reset")
def reset(request: Request) -> dict:
    if TCN_API_KEY and request.headers.get("x-api-key") != TCN_API_KEY:
        raise HTTPException(status_code=401, detail="unauthorized")
    if s3 is not None:
        # Reset rolling window + normalization history (as Stage3TCN.load does).
        s3._window = []
        s3._norm_hist = []
        s3._ticks_processed = 0
    return {"ok": True, "window_size": 0}


if __name__ == "__main__":
    import uvicorn
    # Bind localhost only: the sidecar should never be exposed on the network
    # (was 0.0.0.0). The web app / mini-services reach it on 127.0.0.1:8001.
    uvicorn.run(app, host="127.0.0.1", port=8001)
