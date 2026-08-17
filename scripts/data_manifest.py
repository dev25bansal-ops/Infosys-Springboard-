#!/usr/bin/env python3
"""NEW-07: crash-calendar data manifest.

Scans ``data/parquet/*.parquet`` and writes ``data/manifest.yml`` cataloguing
every available day: symbol, date, file, and its role/kind from
``configs/operating.yml`` (train vs held-out validation vs unlisted). Makes the
data provenance explicit and powers the replay/validation catalogs.

Usage:
    python scripts/data_manifest.py --data-dir data/parquet --out data/manifest.yml
"""
import argparse
import sys
import yaml
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/parquet")
    ap.add_argument("--config", default="configs/operating.yml")
    ap.add_argument("--out", default="data/manifest.yml")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as f:
        op = yaml.safe_load(f)
    val = {(d["symbol"], d["date"]): {"role": "validate", "kind": d.get("kind", "unknown")}
           for d in op["validation"]["days"]}
    trn = {(d.get("symbol"), d.get("date")): {"role": "train", "kind": "train"}
           for d in op["validation"].get("train_days", [])}
    known = {**val, **trn}

    entries = []
    data_dir = Path(args.data_dir)
    for p in sorted(data_dir.glob("*.parquet")):
        name = p.name
        stem = p.stem  # e.g. BTCUSDT_2021-05-19
        parts = stem.split("_")
        symbol = parts[0] if parts else name
        date = parts[1] if len(parts) > 1 else ""
        meta = known.get((symbol, date), {"role": "unlisted", "kind": "unknown"})
        entries.append({
            "symbol": symbol, "date": date, "file": name,
            "role": meta["role"], "kind": meta["kind"],
        })

    payload = {"generated_from": args.config, "days": entries}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(yaml.safe_dump(payload, sort_keys=False, default_flow_style=False))
    roles = {e["role"] for e in entries}
    print(f"[manifest] wrote {len(entries)} days ({dict((r, sum(e['role'] == r for e in entries)) for r in roles)}) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())