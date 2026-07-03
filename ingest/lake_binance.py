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
import sys
from dataclasses import dataclass

# recon.ingest is pandas-only (NO boto3/lakeapi/pyarrow); importing it keeps this module CI-safe
# while reusing the ONE authority for the single-axis engine-time convention (Requirement 4) rather
# than duplicating it. pandas is a default dependency ("stdlib/pandas-light", plan file-structure).
from recon.ingest import _ns, shared_engine_time_col

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


# ----------------------------------------------------------------------------- engine-time resolver (Req 4)
def _is_fully_populated(df, col: str) -> bool:
    """True iff `col` is present and every row is populated (>0 ns) — the 100% gate recon applies
    (recon calls `_require_populated`, which raises on ANY <=0/NaT row: recon/ingest.py:83-87). The
    >99% `is_populated` selector is too coarse here — a 99.x% column passes it then crashes recon."""
    return col in df.columns and bool((_ns(df[col]) > 0).all())


def resolve_engine_time(*dfs):
    """Choose ONE fully-populated engine-time column shared across ALL frames handed to recon
    (deltas + the `book` seed), returning `(col, fallback_used, dfs_clean, dropped_rows)`.

    Joint, never per-frame (plan Requirement 4): selecting per-frame could put deltas on
    received_time and the seed on origin_time — a mixed exchange/capture axis that reorders
    seed/reseed events relative to deltas. Policy:
      1. `origin_time` if FULLY populated in every frame (exchange clock, no data loss);
      2. else `received_time` if FULLY populated in every frame (documented whole-day fallback,
         preferred over dropping rows);
      3. else keep the best shared (>99%) column and DROP its <=0/NaT rows from each frame,
         recording per-frame drop counts. Raises if no column is >99%-populated across all frames.

    `dfs_clean` (aligned to inputs) are the frames the caller feeds to recon — never the originals
    when rows were dropped. `col` + `dropped_rows` go in the manifest, never silent."""
    if not dfs:
        raise ValueError("resolve_engine_time requires at least one DataFrame")
    if all(_is_fully_populated(df, "origin_time") for df in dfs):
        return "origin_time", False, list(dfs), [0] * len(dfs)
    if all(_is_fully_populated(df, "received_time") for df in dfs):
        return "received_time", True, list(dfs), [0] * len(dfs)
    col = shared_engine_time_col(*dfs)  # raises if no column is >99% across every frame
    cleaned, dropped = [], []
    for df in dfs:
        keep = (_ns(df[col]) > 0).to_numpy()
        cleaned.append(df[keep].reset_index(drop=True))
        dropped.append(int((~keep).sum()))
    return col, col != "origin_time", cleaned, dropped


# ----------------------------------------------------------------------------- schema versions (Req 6)
RAW_SCHEMA_VERSION = "lake_binance/1"           # normalized raw store (all feeds + `book` seed)
PROCESSED_SCHEMA_VERSION = {                     # per-output processed store; bump on any schema change
    "topk_l2":       "topk_l2/1",
    "trades":        "trades/1",
    "funding":       "funding/1",
    "open_interest": "open_interest/1",
    "liquidations":  "liquidations/1",
}


# ----------------------------------------------------------------------------- quota + broad gate (Req 7/8)
QUOTA_GB = 300.0            # Crypto Lake individual-plan monthly download cap (docs §2.1/§8)
DEFAULT_MAX_GB = 5.0        # a request larger than this is a "broad" pull → needs --allow-broad
DEFAULT_HEADROOM_GB = 10.0  # never plan to use the last N GB of the quota (used_data lags ~60 min)
BROAD_GATE_EXIT = 4         # exit-code contract (Requirement 8; cf. _common.BACKFILL_GATE_EXIT)

# Conservative per-day Lake footprint by (exchange, symbol) then feed, GB. book_delta_v2 (perp
# 573.8 MB) is MEASURED (docs §6); the per-feed splits of the ~33 MB perp scalar/event feeds and the
# 261 MB spot combined total are DERIVED from §6 combined figures; the `book` seed (~180 MB/instrument)
# is DERIVED from the Coinbase reference (docs §5a-QualityMap) — NOT measured for Binance. Phase 1
# re-measures every value and updates this table (plan Requirement 7 / Risk Q7). Sum over both
# instruments (all feeds + seed) ≈ 1.23 GB/day (see full_pull_gb_per_day).
LAKE_GB_PER_DAY = {
    ("BINANCE_FUTURES", "BTC-USDT-PERP"): {
        "book_delta_v2": 0.5738,   # measured (§6: 573.8 MB/day)
        "trades":        0.025,    # derived (perp trades+funding+OI+liq ≈ 33 MB combined, §6)
        "funding":       0.001,    # derived
        "open_interest": 0.004,    # derived
        "liquidations":  0.003,    # derived (sparse, Risk Q2)
        "book":          0.18,     # seed product — derived (Coinbase ref ~180 MB/day)
    },
    ("BINANCE", "BTC-USDT"): {
        "book_delta_v2": 0.240,    # derived (spot book_delta_v2+trades ≈ 261 MB combined, §6)
        "trades":        0.021,    # derived
        "book":          0.18,     # seed product — derived (Coinbase ref ~180 MB/day)
    },
}


def estimate_gb(instrument_key: str, feeds, n_days: int) -> float:
    """Conservative upper-bound Lake download estimate (GB) for `n_days` × `feeds`. Whenever
    book_delta_v2 is requested the `book` SEED_PRODUCT is ALSO budgeted (it is pulled alongside it,
    Requirement 1) — added once regardless of how many feeds are listed. Rejects an invalid
    (instrument, feed) pair before any estimate. Reads the module's own LAKE_GB_PER_DAY constants,
    so a Phase-1 re-measurement updates the estimate without touching callers."""
    inst = INSTRUMENTS[instrument_key]
    per = LAKE_GB_PER_DAY[(inst.exchange, inst.symbol)]
    feeds = list(feeds)
    for feed in feeds:
        validate_feed(instrument_key, feed)
    per_day = sum(per[feed] for feed in feeds)
    if "book_delta_v2" in feeds:
        per_day += per[SEED_PRODUCT]        # `book` seed pulled with book_delta_v2 (Coinbase-mirrored)
    return float(n_days) * per_day


def full_pull_gb_per_day() -> float:
    """Per-day GB for a FULL pull of every in-scope instrument+feed (+ its `book` seed). The single
    source of truth the batch planner pins its quota-gate rate to (kept aligned by a contract test)."""
    return sum(estimate_gb(k, INSTRUMENTS[k].feeds, 1) for k in INSTRUMENTS)


def check_broad_gate(*, est_gb: float, max_gb: float, allow_broad: bool, used_gb: float,
                     quota_gb: float = QUOTA_GB, headroom_gb: float = DEFAULT_HEADROOM_GB) -> None:
    """Refuse a Lake pull that would breach the monthly quota headroom OR the soft --max-gb cap.

    Two gates (mirrors run_coinbase_quality_map.quota_decision, raised as SystemExit(4) to honor the
    exit-code contract like _common.check_backfill_gate):
      (1) HARD — the monthly quota is an external limit: refuse if the pull leaves less than
          `headroom_gb` of the cap, REGARDLESS of `allow_broad` (used_data lags ~60 min, so keep
          headroom). `--allow-broad` overrides ONLY the soft cap, never this.
      (2) SOFT — a pull larger than `max_gb` is "broad" and refused unless `allow_broad`.
    est_gb <= 0 (nothing to load) is always allowed."""
    if est_gb <= 0:
        return
    remaining = float(quota_gb) - float(used_gb)
    safe_remaining = remaining - float(headroom_gb)
    if est_gb > safe_remaining:
        print(f"REFUSING Lake pull: estimated {est_gb:.2f} GB would breach the monthly quota "
              f"headroom (used {used_gb:.2f} of {quota_gb:.0f} GB, safe remaining "
              f"{safe_remaining:.2f} GB) — this refusal holds regardless of --allow-broad "
              "(used_data lags ~60 min). Re-check lakeapi.used_data or wait for the next quota "
              "window.", file=sys.stderr)
        raise SystemExit(BROAD_GATE_EXIT)
    if est_gb > max_gb and not allow_broad:
        print(f"REFUSING broad Lake pull: estimated {est_gb:.2f} GB exceeds --max-gb {max_gb:.2f} "
              "GB. Pass --allow-broad for a deliberate broad pull (still capped by the "
              f"{quota_gb:.0f} GB/month quota headroom).", file=sys.stderr)
        raise SystemExit(BROAD_GATE_EXIT)
