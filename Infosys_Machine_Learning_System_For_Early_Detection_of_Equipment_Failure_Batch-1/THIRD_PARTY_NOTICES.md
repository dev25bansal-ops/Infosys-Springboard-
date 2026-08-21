# Third-Party Notices & Data Licensing

This project (Apache-2.0) reuses external data and software. Respect the terms below.

## Market data

- **Binance Public Market Data** — used for training, backtests, and the live feed
  (`wss://stream.binance.com`, `data.binance.vision`). Use is subject to
  [Binance Market Data Terms of Use](https://www.binance.com/en/legal/terms-of-use).
  The May 19 2021 BTC data notably spans a Binance outage window; treat that day
  as "continuous as downloaded" (largest gap ~1.8s), not a verified-continuous feed.
- **FI-2010** (financial limit-order-book dataset) — academic dataset. Note: the
  `fi2010_loader` reads the label column but **discards it** and returns 40-column
  feature matrices; the README row describing "FI-2010 crash labels" should be
  construed as feature extraction only, not labeled crashes.
- **LOBSTER / NASDAQ TotalView-ITCH** — cited as candidate loaders in `docs/DATA_SOURCES.md`;
  data is academic-licensed and, if used, must retain original attribution.

## Core software dependencies (indicative; `lock` files are authoritative)

- **Python (ml + ml-inference):** PyTorch, numpy, pandas, scikit-learn, scipy,
  fastapi/uvicorn, pydantic. See `ml/requirements.txt` and
  `ml-inference/requirements.txt`.
- **Web (flash-crash-watchdog-web):** Next.js, React, Prisma, Socket.IO,
  Zustand, shadcn/ui (MIT), Tailwind. See `package-lock.json`.
- **Rust (proxy):** tokio, tokio-tungstenite, clap, serde, tracing, chrono.
  See `proxy/Cargo.toml`.
- **AI assistance:** this codebase was substantially produced with assistance
  from a language model (Anthropic Claude).

## License

The project itself is Apache-2.0. Third-party components retain their own
licenses; this notice catalogues the noteworthy external data and libraries for
attribution and compliance review.