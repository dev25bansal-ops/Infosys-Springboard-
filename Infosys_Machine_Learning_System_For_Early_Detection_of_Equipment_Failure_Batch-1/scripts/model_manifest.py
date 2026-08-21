#!/usr/bin/env python3
"""MLOPS-05: model registry / manifest — sha256 + provenance per checkpoint.

Generates ``models/manifest.json``: one entry per ``.pt`` checkpoint with its
sha256, size, and (when the checkpoint carries an RSR-16 provenance stamp) the
provenance. The inference sidecar (ml-inference/server.py) verifies the model it
loads against this manifest and FAILS CLOSED on a hash mismatch.

Usage:
    python scripts/model_manifest.py --models-dir models            # write
    python scripts/model_manifest.py --models-dir models --verify    # check hashes
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--manifest", default=None,
                    help="manifest path (default <models-dir>/manifest.json)")
    ap.add_argument("--verify", action="store_true",
                    help="verify existing hashes instead of writing")
    args = ap.parse_args()

    models_dir = Path(args.models_dir)
    manifest_path = Path(args.manifest or models_dir / "manifest.json")

    entries: dict[str, dict] = {}
    for p in sorted(models_dir.glob("*.pt")):
        entries[p.name] = {"sha256": sha256_of(p), "size_bytes": p.stat().st_size}

    if args.verify:
        if not manifest_path.exists():
            print(f"ERROR: no manifest at {manifest_path}", file=sys.stderr)
            return 1
        known = json.loads(manifest_path.read_text())
        problems = []
        for name, entry in known.get("models", {}).items():
            f = models_dir / name
            if not f.exists():
                problems.append(f"missing file: {name}")
                continue
            if sha256_of(f) != entry["sha256"]:
                problems.append(f"HASH MISMATCH: {name}")
        # also report new models not in the manifest
        unlisted = [n for n in entries if n not in known.get("models", {})]
        if problems:
            for p in problems:
                print(f"ERROR: {p}", file=sys.stderr)
            return 1
        status = "OK" if not unlisted else f"OK ({len(unlisted)} unlisted new model(s))"
        print(f"[manifest] {status}")
        return 0

    payload = {"models": entries}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))
    print(f"[manifest] wrote {len(entries)} checkpoint hashes -> {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
