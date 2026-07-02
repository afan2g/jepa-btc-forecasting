"""Pure, CI-safe helpers for the Binance Crypto Lake downloader / native reconstruction.

Stdlib/pandas-light ONLY. No boto3/lakeapi/pyarrow import at module top so CI unit tests import
this without the downloader's vendor deps (mirrors the ingest/_common.py split). The vendor-touching
Stage-1 CLI (ingest/download_lake_binance.py) and Stage-2 recon runner (scripts/run_binance_recon.py)
import lakeapi/pyarrow themselves; everything here is the pure substrate they share:

  * the frozen feed/instrument registry (Requirement 1) — one source of truth for every path,
    manifest key and quota estimate;
  * Hive partition-path builders for the normalized raw store and the processed store (Requirement 2);
  * append-only manifest read/write + resume/idempotency state (Requirements 3/6);
  * the joint engine-time resolver (origin_time first, documented received_time fallback — Requirement 4);
  * quota estimation + the broad-pull gate (Requirement 7 / exit-code contract Requirement 8).

See docs/superpowers/plans/2026-07-02-binance-downloader-plan.md.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

MANIFEST_NAME = "_manifest.jsonl"

# ----------------------------------------------------------------------------- registry (Req 1)
# Scoped OUTPUT feeds. `book` (the 20-level snapshot SEED product) is deliberately NOT here — it is a
# seed INPUT for book_delta_v2 reconstruction, downloaded alongside book_delta_v2 but never emitted.
FEEDS = ("book_delta_v2", "trades", "funding", "open_interest", "liquidations")

# Seed-input product (NOT a scoped output feed): Lake's `book` 20-level snapshot, pulled per
# instrument SOLELY to seed book_delta_v2 recon (book_delta_v2 starts mid-stream with no per-day
# snapshot, docs §4.1). Consumed by Stage 2, never emitted. Mirrors the Coinbase reference
# LAKE_PRODUCTS = ("book_delta_v2", "book").
SEED_PRODUCT = "book"

# Per-feed handling class — drives which Stage-2 path runs and error tolerance (docs §5b / Risk Q2/Q6).
FEED_KIND = {
    "book_delta_v2": "delta",     # → Stage-2 seed/reseed → top-K L2 (label ≠ the `book` seed product)
    "trades":        "trades",    # → normalize (origin_time-sorted); Binance already sorted (§5b)
    "funding":       "scalar",    # → normalize; ~8-hourly cadence (confirm in Phase 1, Risk Q6)
    "open_interest": "scalar",    # → normalize; periodic snapshots (confirm in Phase 1, Risk Q6)
    "liquidations":  "events",    # → normalize; SPARSE — missing/empty files are OK (Risk Q2)
}


@dataclass(frozen=True)
class Instrument:
    key: str                     # "binance-perp" | "binance-spot"
    exchange: str                # lakeapi `exchange` partition value
    symbol: str                  # lakeapi `symbol` partition value
    feeds: tuple[str, ...]       # scoped output feeds valid for this instrument


# Identifiers reused VERBATIM from the repo's existing verifiers (do not re-invent — plan Req 1):
#   perp  BINANCE_FUTURES / BTC-USDT-PERP  (ingest/verify_lake.py:21-22, liquidations :70-72)
#   spot  BINANCE / BTC-USDT               (ingest/verify_trades_and_calendar.py:98-99)
INSTRUMENTS = {
    "binance-perp": Instrument(
        "binance-perp", "BINANCE_FUTURES", "BTC-USDT-PERP",
        ("book_delta_v2", "trades", "funding", "open_interest", "liquidations")),
    "binance-spot": Instrument(
        "binance-spot", "BINANCE", "BTC-USDT",
        ("book_delta_v2", "trades")),
}

# book_delta_v2 reconstructs into the top-K L2 output; every other feed normalizes to a same-named
# processed table. The `book` seed product is never an output (it is consumed by Stage 2).
FEED_TO_OUTPUT = {
    "book_delta_v2": "topk_l2",
    "trades":        "trades",
    "funding":       "funding",
    "open_interest": "open_interest",
    "liquidations":  "liquidations",
}


def validate_feed(instrument_key: str, feed: str) -> None:
    """Reject an invalid (instrument, feed) pair (e.g. `funding` on spot) BEFORE any vendor call.
    Raises KeyError for an unknown instrument, ValueError for a feed the instrument does not carry."""
    inst = INSTRUMENTS[instrument_key]
    if feed not in inst.feeds:
        raise ValueError(f"feed {feed!r} not valid for {instrument_key} (valid: {inst.feeds})")


# ----------------------------------------------------------------------------- partition paths (Req 2)
def raw_partition_dir(out_root: str, feed: str, exchange: str, symbol: str, day_iso: str) -> str:
    """Hive-style partition DIR for the normalized raw store: keyed by feed/exchange/symbol/date.
    `feed` may be a scoped output feed OR the `book` SEED_PRODUCT (its own sibling partition)."""
    return os.path.join(out_root, feed, f"exchange={exchange}",
                        f"symbol={symbol}", f"dt={day_iso}")


def raw_parquet_path(out_root: str, feed: str, exchange: str, symbol: str, day_iso: str) -> str:
    """Final `data.parquet` path inside the raw-store partition (idempotency marker)."""
    return os.path.join(raw_partition_dir(out_root, feed, exchange, symbol, day_iso), "data.parquet")


def processed_parquet_path(out_root: str, output: str, exchange: str, symbol: str,
                           day_iso: str) -> str:
    """Final `data.parquet` for a PROCESSED output — keyed by OUTPUT NAME (topk_l2/trades/...),
    NOT the Lake feed, and with no `lake/` segment (distinct scheme from the raw store)."""
    return os.path.join(out_root, output, f"exchange={exchange}",
                        f"symbol={symbol}", f"dt={day_iso}", "data.parquet")


# ----------------------------------------------------------------------------- manifest + resume (Req 3/6)
def manifest_append(store_root: str, rec: dict) -> None:
    """Append one JSON record line to `<store_root>/_manifest.jsonl` (append-only resume ledger,
    one record per written partition; mirrors download_coinapi.py:manifest_append)."""
    os.makedirs(store_root, exist_ok=True)
    with open(os.path.join(store_root, MANIFEST_NAME), "a") as f:
        f.write(json.dumps(rec) + "\n")


def manifest_index(store_root: str) -> dict[tuple[str, str, str, str], str]:
    """Map (feed, exchange, symbol, dt) -> latest `status` from the manifest. Last record wins, so a
    --resume run that re-writes a unit supersedes the earlier record. Blank/malformed lines and
    records missing the key fields are skipped (a partial write must not crash resume). Empty dict
    when no manifest exists."""
    path = os.path.join(store_root, MANIFEST_NAME)
    idx: dict[tuple[str, str, str, str], str] = {}
    if not os.path.exists(path):
        return idx
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["feed"], rec["exchange"], rec["symbol"], rec["dt"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            idx[key] = rec.get("status")
    return idx


def is_done(out_root: str, feed: str, exchange: str, symbol: str, day_iso: str) -> bool:
    """A partition is done iff its FINAL `data.parquet` exists — a leftover `.tmp` from an
    interrupted run does NOT count (writes are atomic: stream to .tmp, os.replace on success)."""
    return os.path.exists(raw_parquet_path(out_root, feed, exchange, symbol, day_iso))


def cleanup_tmp(out_root: str) -> int:
    """Remove stale `*.parquet.tmp` left by an interrupted run (keeps --resume clean; mirrors
    download_coinapi.py:cleanup_tmp). Returns the number removed."""
    removed = 0
    for dirpath, _, files in os.walk(out_root):
        for fn in files:
            if fn.endswith(".parquet.tmp"):
                os.remove(os.path.join(dirpath, fn))
                removed += 1
    return removed
