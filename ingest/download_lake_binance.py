"""Stage-1 Binance Crypto Lake downloader → normalized, Hive-partitioned, ZSTD Parquet raw store.

Streams each `(feed, exchange, symbol, day)` partition (plus the `book` 20-level SEED_PRODUCT that
seeds Stage-2 reconstruction) row-group-by-row-group into `data/raw/lake/{feed}/exchange=…/symbol=…/
dt=…/data.parquet` — never decompressing to CSV, never holding a whole `book_delta_v2` day (109 M
rows) in RAM. Resumable and quota-aware: a partition whose final `data.parquet` exists is skipped,
writes are atomic (`.tmp` → `os.replace`), and the whole request is gated against the 300 GB/month
Lake quota BEFORE any transfer (`ingest.lake_binance.check_broad_gate`). Mirrors the CoinAPI
downloader's streaming/atomic/manifest shape (`ingest/download_coinapi.py`).

Design / safety (plan docs/superpowers/plans/2026-07-02-binance-downloader-plan.md, Task 5):
  * IMPORT-SAFE: no `boto3`/`lakeapi`/`pyarrow` at module top — the pure helpers stay importable in
    CI without vendor deps; every vendor touch (`pyarrow` write, `boto3` session, `lakeapi`
    list/used_data) is imported inside the live function that needs it.
  * process_unit(reader, …) is driven by an INJECTED reader (a callable yielding pyarrow batches /
    pandas frames, or None for a missing file), so the whole streaming/atomic/manifest/retry path is
    unit-tested with fakes — no live Lake in tests.
  * Lake-only session (`lake_session`) reads AWS keys straight from `.env`; it does NOT require the
    unrelated `COINAPI_KEY` (`ingest/_common.load_env` would `SystemExit` without it — Requirement 4).

Exit codes (Requirement 8): 0 all ok/skip/missing · 2 setup error or vendor quota/credit hard stop ·
3 completed with ≥1 errored unit (rerun with --resume) · 4 broad-pull / quota-headroom gate.

Runtime deps (pyarrow, lakeapi, boto3) are the `lake` extra — `pip install -e .[lake]` — kept out of
the base install so the pure `recon`/`ingest.lake_binance` helpers stay light. This module imports
them lazily inside the live functions, so it (and its tests) import fine without the extra.

Live Lake pulls are approval-gated (AGENTS.md). Even `--dry-run` issues a live `lakeapi.list_data`
metadata call (no parquet transfer). The only fully offline planner is `scripts/plan_lake_binance_batches.py`.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import os
import random
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from threading import Event, Lock

# Repo root on sys.path so `from ingest import lake_binance` works both as a script (script dir is
# ingest/, not the root) and when imported by tests. lake_binance is pure (pandas-only, no vendor
# deps), so importing it here keeps this module import-safe.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from ingest import lake_binance as lb  # noqa: E402

DEFAULT_OUT_ROOT = "data/raw/lake"
DEFAULT_REPORT_DIR = "data/reports/binance_download"

SETUP_ERROR_EXIT = 2       # bad args / missing keys / unreadable used_data / vendor quota hard stop
PARTIAL_EXIT = 3           # completed with ≥1 errored unit — rerun with --resume

# Retry policy for transient S3/botocore errors (SlowDown, timeout, reset, 5xx). A quota/credit error
# is a HARD stop (never retried). Retries are per-unit so one bad partition never re-pulls a done one.
DEFAULT_RETRIES = 5
DEFAULT_BACKOFF_BASE_S = 1.0
DEFAULT_BACKOFF_CAP_S = 60.0

# Four independent streaming units keep the Stage-1 S3 path bounded while overlapping high-latency
# reads. This is an operational ceiling, not a live-speedup guarantee. Stage 2 has a separate,
# memory-bound concurrency contract in g0bn_acquisition_preflight.py.
MAX_DOWNLOAD_JOBS = 4

# Certified serial evidence from the completed 2026-04-01 Binance-perp smoke. These timings are used
# only for clearly caveated runtime projections; quota and approval gates never depend on them.
CERTIFIED_SERIAL_DAY = "2026-04-01"
CERTIFIED_SERIAL_SECS_BY_FEED = {
    "book_delta_v2": 1441.344,
    "book": 242.671,
    "trades": 24.936,
}


# ----------------------------------------------------------------------------- error taxonomy
class HardStop(RuntimeError):
    """A RUN-fatal condition that aborts the whole pull immediately with setup exit 2 — never
    retried, never recorded as a per-unit failure. Subclassed by QuotaError and AuthError so one
    `except HardStop` in main() covers both."""


class QuotaError(HardStop):
    """Vendor quota/credit exhaustion — a HARD stop (never retried); the run exits 2 (fail-safe)."""


class AuthError(HardStop):
    """Vendor auth/permission failure (AccessDenied, InvalidAccessKeyId — wrong subscriber keys or
    the wrong AWS account, docs §2.1). A HARD stop: every pending unit would fail identically, so
    abort with setup exit 2 rather than retrying / recording each partition as an error (exit 3)."""


class TransientError(RuntimeError):
    """A retryable transient vendor/network error (throttle, timeout, reset, 5xx)."""


class CancelledByHardStop(RuntimeError):
    """Internal cooperative cancellation after another unit hits a run-fatal hard stop."""


# String markers so we never need to import botocore to classify a real ClientError (mirrors the
# stringly-typed ingest/_common.is_quota_error). Explicit Quota/Auth/TransientError bypass these.
# Markers are context-specific on purpose: BARE HTTP codes ("500"/"503"/"403") are deliberately
# excluded — they match unrelated text ("row 5000", "expected 500 columns") and would misclassify a
# fatal error. The PARENTHESIZED forms ("(500)") ARE matched: they are botocore's `An error occurred
# (500) ...` shape and cannot appear in "row 5000"/"expected 500 columns". Quota markers avoid the
# bare "you have exceeded" (matches rate-limit throttles).
_QUOTA_MARKERS = ("quotaexceeded", "quota exceeded", "insufficient usage credits",
                  "download quota", "no usable credit", "exceeded your quota",
                  "exceeded your download")
_AUTH_MARKERS = ("accessdenied", "access denied", "invalidaccesskeyid", "invalid access key",
                 "signaturedoesnotmatch", "signature does not match", "unrecognizedclient",
                 "unrecognized client", "authfailure", "auth failure", "not authorized",
                 "unauthorized", "expiredtoken", "expired token", "token has expired",
                 "invalidclienttokenid", "invalid client token id", "invalidtoken", "invalid token",
                 "permission denied", "forbidden")
_TRANSIENT_MARKERS = ("slowdown", "slow down", "throttl", "reduce your request rate",
                      "requesttimeout", "request timeout", "timed out", "timeout",
                      "connection reset", "connectionreset", "connection aborted",
                      "network connection", "serviceunavailable", "service unavailable",
                      "bad gateway", "gateway timeout", "internalerror", "internal error",
                      "internal failure", "internal server error",
                      "(500)", "(502)", "(503)", "(504)",
                      "http status 500", "http status 502", "http status 503", "http status 504",
                      "temporarily")


def classify_error(exc: BaseException) -> str:
    """Map an exception to 'quota' | 'auth' (both HARD stops → exit 2) | 'transient' (retry) |
    'fatal' (record + continue → exit 3). Underscores are normalized to spaces first so PyArrow's
    `AWS Error ACCESS_DENIED` / `SLOW_DOWN` / `SERVICE_UNAVAILABLE` codes match the same
    space-separated markers as botocore's `AccessDenied`/`SlowDown` strings."""
    if isinstance(exc, QuotaError):
        return "quota"
    if isinstance(exc, AuthError):
        return "auth"
    if isinstance(exc, TransientError):
        return "transient"
    msg = str(exc).lower().replace("_", " ")
    if any(m in msg for m in _QUOTA_MARKERS):
        return "quota"
    if any(m in msg for m in _AUTH_MARKERS):
        return "auth"
    if any(m in msg for m in _TRANSIENT_MARKERS):
        return "transient"
    return "fatal"


# ----------------------------------------------------------------------------- small helpers
def now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def _rm(path: str) -> None:
    with contextlib.suppress(FileNotFoundError):
        os.remove(path)


def _sha256_file(path: str, *, cancel_event=None) -> str:
    """Streaming sha256 of a file (never loads the whole parquet into RAM)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledByHardStop("run hard-stopped while hashing temporary output")
            h.update(chunk)
    return h.hexdigest()


def _format_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def _format_bytes(value: int) -> str:
    value = int(value)
    if value < 1_000_000:
        return f"{value / 1_000:.1f}KB"
    if value < 1_000_000_000:
        return f"{value / 1_000_000:.1f}MB"
    return f"{value / 1_000_000_000:.2f}GB"


def validate_download_jobs(value: int) -> int:
    """Return a strict Stage-1 unit concurrency, rejecting bool-as-int and unsafe fan-out."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("--jobs must be an integer (booleans are not valid concurrency values)")
    if not 1 <= value <= MAX_DOWNLOAD_JOBS:
        raise ValueError(f"--jobs must be between 1 and {MAX_DOWNLOAD_JOBS}; got {value}")
    return value


def _backoff_seconds(attempt: int, base: float, cap: float, rng) -> float:
    """Capped exponential backoff with 0.5–1.0× jitter (attempt is 1-based)."""
    raw = min(cap, base * (2 ** (attempt - 1)))
    return raw * (0.5 + 0.5 * rng())


def daterange(start_iso: str, end_iso: str) -> list[str]:
    start = dt.date.fromisoformat(start_iso)
    end = dt.date.fromisoformat(end_iso)
    if end < start:
        raise ValueError(f"--end {end_iso} is before --start {start_iso}")
    out, d = [], start
    while d <= end:
        out.append(d.isoformat())
        d += dt.timedelta(days=1)
    return out


def load_days_file(path: str) -> list[str]:
    """Sorted ISO day list from a --days-file (one YYYY-MM-DD per line; blanks/`#` ignored)."""
    days = []
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#"):
            dt.date.fromisoformat(line)              # validate; raises on a bad day
            days.append(line)
    return sorted(set(days))


# ----------------------------------------------------------------------------- streaming parquet write
def _iter_record_batches(item):
    """Yield pyarrow RecordBatch(es) from a streamed item (RecordBatch | Table | pandas frame)."""
    import pyarrow as pa
    if isinstance(item, pa.RecordBatch):
        yield item
    elif isinstance(item, pa.Table):
        yield from item.to_batches()
    else:                                            # assume a pandas DataFrame
        yield pa.RecordBatch.from_pandas(item, preserve_index=False)


def _stream_to_parquet(stream, dest_tmp: str, *, schema_version: str, feed: str,
                       exchange: str, symbol: str, day_iso: str,
                       cancel_event=None) -> int:
    """Consume an iterable of batches/frames → one ZSTD Parquet file. Returns total rows written
    (0 when the stream yields no batches OR only zero-row batches — a present-but-empty partition;
    the caller treats rows==0 as empty/missing under the sparse/required policy rather than
    publishing a 0-row parquet).

    schema_version + partition keys go in the parquet KV metadata (Requirement 6) — NOT the sha256
    (embedding the hash would change the very bytes being hashed; rows likewise stays in the manifest
    since a streaming writer cannot know the total up front)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    writer = None
    rows = 0
    try:
        for item in stream:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledByHardStop("run hard-stopped while streaming temporary output")
            for batch in _iter_record_batches(item):
                if cancel_event is not None and cancel_event.is_set():
                    raise CancelledByHardStop("run hard-stopped while streaming temporary output")
                if writer is None:
                    meta = dict(batch.schema.metadata or {})
                    meta[b"schema_version"] = schema_version.encode()
                    meta[b"feed"] = feed.encode()
                    meta[b"exchange"] = exchange.encode()
                    meta[b"symbol"] = symbol.encode()
                    meta[b"dt"] = day_iso.encode()
                    schema = batch.schema.with_metadata(meta)
                    writer = pq.ParquetWriter(dest_tmp, schema, compression="zstd")
                writer.write_table(pa.Table.from_batches([batch], schema=writer.schema))
                rows += batch.num_rows
    finally:
        if writer is not None:
            writer.close()
    # No raise on an empty stream: rows==0 (no batches, OR only zero-row batches) means a
    # present-but-empty partition, which the caller handles under the sparse/required policy (an
    # empty sparse liquidations day is a non-fatal quiet day; a required feed's emptiness is a gap).
    return rows


# ----------------------------------------------------------------------------- schema validation (Req 6)
# Engine-time is the ONE column every feed must carry — recon/passthrough aligns on it and
# recon.ingest.ENGINE_TIME_CANDIDATES only understands these names (raw vendor `timestamp`/
# `receipt_timestamp` OR lakeapi-normalized `origin_time`/`received_time`). Requiring its PRESENCE is
# safe even where the exact per-feed schema is unmeasured (funding/OI/liquidations, Risk Q6/Q9): a
# partition missing all four cannot be reconstructed at all, so it is real drift worth failing on.
_ENGINE_TIME_ALIASES = ("origin_time", "received_time", "timestamp", "receipt_timestamp")


def schema_fingerprint(schema) -> str:
    """Stable 16-hex fingerprint of a pyarrow schema (sha256 over sorted `name:type`). Recorded per
    partition so an audit/monitor can detect vendor schema drift day-over-day even for the feeds whose
    exact schema is not yet pinned — the fixed `lake_binance/1` alone cannot signal drift."""
    parts = sorted(f"{field.name}:{field.type}" for field in schema)
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:16]


def validate_raw_schema(feed: str, columns) -> None:
    """Fail loud BEFORE a partition is stamped `ok` if it lacks the universal engine-time column, so
    gross vendor drift surfaces at download time (status error → exit 3) instead of being copied,
    stamped as if it met the raw-store contract, marked done by resume, and only discovered in Stage-2
    after quota is spent (plan Requirement 6). Finer per-feed column/dtype checks land with the Task-3
    normalizers once the funding/OI/liquidations schemas are measured (Risk Q6); until then the
    recorded schema_fingerprint carries the rest for the Phase-1 probe."""
    cols = set(columns)
    if not any(c in cols for c in _ENGINE_TIME_ALIASES):
        raise ValueError(f"{feed} raw partition missing an engine-time column (need one of "
                         f"{_ENGINE_TIME_ALIASES}); saw {sorted(cols)} — vendor schema drift, "
                         "refusing to stamp the raw-store contract")


# ----------------------------------------------------------------------------- per-unit worker
@dataclass
class UnitResult:
    status: str                 # ok | skip | missing | error
    rows: int
    path: str | None
    record: dict | None


def process_unit(reader, out_root: str, feed: str, exchange: str, symbol: str, day_iso: str, *,
                 schema_version: str | None = None, overwrite: bool = False,
                 retries: int = DEFAULT_RETRIES, backoff_base: float = DEFAULT_BACKOFF_BASE_S,
                 backoff_cap: float = DEFAULT_BACKOFF_CAP_S, sleep=time.sleep, rng=None,
                 manifest_root: str | None = None, lock=None, sparse_ok: bool = False,
                 cancel_event=None) -> UnitResult:
    """Download ONE (feed, exchange, symbol, day) partition via an injected `reader`.

    `reader(feed, exchange, symbol, day_iso)` returns an iterable of pyarrow batches / pandas frames
    (streamed straight to ZSTD Parquet), None for a missing vendor file, or raises. Behaviors:
      * a partition whose final data.parquet exists is SKIPPED unless `overwrite`;
      * writes are ATOMIC — stream to `data.parquet.tmp`, `os.replace` on success, so an interrupted
        run never publishes a partial parquet;
      * transient errors retry with capped exponential backoff + jitter; a QuotaError is re-raised
        immediately (hard stop, never retried); a fatal error / exhausted retries record `status:
        error` and return (the run continues, then exits 3);
      * completed units append one manifest record (ok/skip/missing/error) to the resume ledger;
        cooperative hard-stop cancellations never mutate the deterministic resume ledger.
    Returns a UnitResult; raises QuotaError on a vendor quota/credit hard stop."""
    schema_version = schema_version or lb.RAW_SCHEMA_VERSION
    manifest_root = manifest_root or out_root
    rng = rng or random.random
    lock = lock or contextlib.nullcontext()

    final = lb.raw_parquet_path(out_root, feed, exchange, symbol, day_iso)
    base = {"feed": feed, "exchange": exchange, "symbol": symbol, "dt": day_iso}

    def _raise_if_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledByHardStop("run hard-stopped before this unit could complete")

    def _append(rec: dict) -> None:
        with lock:
            _raise_if_cancelled()
            lb.manifest_append(manifest_root, rec)

    _raise_if_cancelled()
    if not overwrite and lb.is_done(out_root, feed, exchange, symbol, day_iso):
        rec = {**base, "status": "skip", "ts": now_iso()}
        _append(rec)
        return UnitResult("skip", 0, final, rec)

    tmp = final + ".tmp"
    attempt = 0
    while True:
        _raise_if_cancelled()
        attempt += 1
        _rm(tmp)
        t0 = time.monotonic()
        try:
            stream = reader(feed, exchange, symbol, day_iso)
            _raise_if_cancelled()
            if stream is None:                       # no vendor file (sparse liquidations, gap, …)
                # `sparse_ok` (liquidations) → an expected quiet-day gap; otherwise a REQUIRED feed's
                # miss is a real hole the run must surface (main exits 3), never a silent success.
                # On --overwrite, drop any STALE published parquet so is_done doesn't keep obsolete
                # raw data for a partition the vendor now reports as absent (manifest says missing).
                rec = {**base, "status": "missing", "sparse_ok": bool(sparse_ok), "ts": now_iso()}
                with lock:
                    _raise_if_cancelled()
                    _rm(final)
                    lb.manifest_append(manifest_root, rec)
                return UnitResult("missing", 0, None, rec)
            os.makedirs(os.path.dirname(final), exist_ok=True)   # partition dir must exist for the .tmp
            rows = _stream_to_parquet(stream, tmp, schema_version=schema_version, feed=feed,
                                      exchange=exchange, symbol=symbol, day_iso=day_iso,
                                      cancel_event=cancel_event)
            if rows == 0:
                # present-but-empty partition — whether the stream had no batches OR only zero-row
                # batches (a schema-only empty parquet may sit in tmp; discard it, never publish a
                # 0-row data.parquet). Treated like a missing one under the SAME sparse/required
                # policy: a quiet-day liquidations file is non-fatal; a required feed's emptiness is
                # a real gap (main → exit 3). Drop any stale published parquet too (see above).
                _rm(tmp)
                rec = {**base, "status": "missing", "sparse_ok": bool(sparse_ok),
                       "empty": True, "ts": now_iso()}
                with lock:
                    _raise_if_cancelled()
                    _rm(final)
                    lb.manifest_append(manifest_root, rec)
                return UnitResult("missing", 0, None, rec)
            _raise_if_cancelled()
            import pyarrow.parquet as pq
            schema = pq.read_schema(tmp)             # cheap local footer read
            validate_raw_schema(feed, schema.names)  # fail loud on drift BEFORE publishing `ok`
            out_bytes = os.path.getsize(tmp)
            sha = _sha256_file(tmp, cancel_event=cancel_event)
            rec = {**base, "status": "ok", "rows": rows, "sha256": sha, "out_bytes": out_bytes,
                   "schema_version": schema_version, "schema_fingerprint": schema_fingerprint(schema),
                   "schema_cols": list(schema.names), "secs": round(time.monotonic() - t0, 3),
                   "ts": now_iso()}
            # Serialize the final cancellation check, atomic publication, and manifest append with
            # hard-stop signaling. A completed unit is therefore ordered either wholly before the
            # stop or wholly after it (and cancelled); no post-stop partial publication can win.
            with lock:
                _raise_if_cancelled()
                os.replace(tmp, final)
                lb.manifest_append(manifest_root, rec)
            return UnitResult("ok", rows, final, rec)
        except CancelledByHardStop:
            _rm(tmp)
            raise
        except Exception as exc:                     # noqa: BLE001 — classified below; never swallow SystemExit
            _rm(tmp)                                  # never leave a partial parquet behind
            kind = classify_error(exc)
            if kind == "quota":
                if cancel_event is not None:
                    with lock:
                        cancel_event.set()
                raise QuotaError(str(exc)) from exc
            if kind == "auth":                       # wrong keys/account → every unit fails the same
                if cancel_event is not None:
                    with lock:
                        cancel_event.set()
                raise AuthError(str(exc)) from exc
            if kind == "transient" and attempt < retries:
                _raise_if_cancelled()
                sleep(_backoff_seconds(attempt, backoff_base, backoff_cap, rng))
                continue
            rec = {**base, "status": "error", "error": f"{type(exc).__name__}: {exc}"[:500],
                   "attempts": attempt, "ts": now_iso()}
            _append(rec)
            return UnitResult("error", 0, None, rec)


# ----------------------------------------------------------------------------- unit planning
@dataclass(frozen=True)
class Unit:
    instrument_key: str
    exchange: str
    symbol: str
    feed: str            # a scoped output feed OR the `book` SEED_PRODUCT
    day: str


def _cleanup_unit_tmps(out_root: str, units: list[Unit]) -> int:
    """Remove stale temps only for the validated request, never scan the raw store."""
    tmp_paths = {
        lb.raw_parquet_path(out_root, unit.feed, unit.exchange, unit.symbol, unit.day) + ".tmp"
        for unit in units
    }
    removed = 0
    for path in sorted(tmp_paths):
        try:
            os.remove(path)
        except FileNotFoundError:
            continue
        removed += 1
    return removed


def runtime_projection(units: list[Unit], jobs: int) -> dict:
    """Caveated arithmetic runtime reference from the certified serial smoke, never a live bound."""
    validate_download_jobs(jobs)
    unknown_feeds = sorted({u.feed for u in units if u.feed not in CERTIFIED_SERIAL_SECS_BY_FEED})
    caveat = (
        "Certified serial timings are an arithmetic reference only. Object layout, latency, "
        "contention, retries, and quota can put actual runtime outside this range. It is not a "
        "bound and does not guarantee 4x live scaling."
    )
    basis = {
        "day": CERTIFIED_SERIAL_DAY,
        "seconds_by_feed": dict(CERTIFIED_SERIAL_SECS_BY_FEED),
        "minutes_per_three_product_day": round(
            sum(CERTIFIED_SERIAL_SECS_BY_FEED.values()) / 60, 3),
    }
    if unknown_feeds:
        return {"available": False, "basis": basis, "unknown_feeds": unknown_feeds,
                "caveat": caveat}

    serial_seconds = sum(CERTIFIED_SERIAL_SECS_BY_FEED[u.feed] for u in units)
    return {
        "available": True,
        "basis": basis,
        "serial_reference_seconds": round(serial_seconds, 3),
        "idealized_jobs_floor_seconds": round(serial_seconds / jobs, 3),
        "jobs": jobs,
        "caveat": caveat,
    }


def resolve_feeds(instrument_key: str, feeds_arg: str | None) -> list[str]:
    """Selected feeds for an instrument. Default = all feeds valid for it; each pair validated
    (an invalid pair like `funding` on spot raises ValueError before any vendor call). A `--feeds`
    that parses to nothing (e.g. `--feeds ,` from a wrapper) is REJECTED, not silently emptied —
    otherwise the run would produce zero units and exit 0 as if it had downloaded the batch."""
    inst = lb.INSTRUMENTS[instrument_key]
    if feeds_arg is None:                # OMITTED → default to all feeds (an explicit "" or "," is
        return list(inst.feeds)          # NOT omitted: it falls through and is rejected as empty)
    feeds, seen = [], set()
    for f in feeds_arg.split(","):
        f = f.strip()
        if f and f not in seen:          # de-dup repeated feeds so they never race on one partition
            seen.add(f)
            feeds.append(f)
    if not feeds:
        raise ValueError(f"--feeds {feeds_arg!r} is empty after parsing (only separators/whitespace); "
                         "provide at least one feed")
    for feed in feeds:
        lb.validate_feed(instrument_key, feed)
    return feeds


def plan_units(instrument_keys: list[str], feeds_arg: str | None, days: list[str]) -> list[Unit]:
    """The full (instrument, feed, day) work list, DE-DUPLICATED (order-preserving). Whenever
    `book_delta_v2` is selected the `book` SEED_PRODUCT is ALSO scheduled per day (it seeds Stage-2
    recon, Requirement 1). Dedup is load-bearing: a repeated instrument/feed (`--instrument
    binance-perp,binance-perp`, `--feeds trades,trades`) would otherwise emit identical
    (feed,E,S,dt) units that, under --jobs>1, race on the same data.parquet.tmp and corrupt it."""
    units: list[Unit] = []
    seen: set[Unit] = set()
    for key in instrument_keys:
        inst = lb.INSTRUMENTS[key]
        feeds = resolve_feeds(key, feeds_arg)
        pull_feeds = list(feeds)
        if "book_delta_v2" in feeds:
            pull_feeds.append(lb.SEED_PRODUCT)   # `book` seed pulled alongside book_delta_v2
        for feed in pull_feeds:
            for day in days:
                u = Unit(key, inst.exchange, inst.symbol, feed, day)
                if u not in seen:
                    seen.add(u)
                    units.append(u)
    return units


def unit_gb(unit: Unit) -> float:
    """Conservative per-day GB for ONE unit — a scoped feed OR the `book` seed (both are keys in the
    per-(exchange,symbol) table). Summing over the units that will actually transfer is what makes
    the gate estimate the real work, not the whole range (see pending_units)."""
    return lb.LAKE_GB_PER_DAY[(unit.exchange, unit.symbol)][unit.feed]


def sparse_accepted(manifest_root: str) -> set[tuple[str, str, str, str]]:
    """(feed,exchange,symbol,dt) units the manifest already ACCEPTED as a sparse-ok missing/empty
    partition (a quiet-day gap with no `data.parquet`, e.g. liquidations). These are DONE for resume:
    a later no-overwrite run must not re-hit Lake or re-charge the gate for them. Last record wins, so
    a subsequent `ok` (parquet written) or a required-feed miss (`sparse_ok` false, still pending)
    supersedes an earlier sparse acceptance. Malformed lines are skipped (mirrors manifest_index)."""
    path = os.path.join(manifest_root, lb.MANIFEST_NAME)
    if not os.path.exists(path):
        return set()
    latest: dict[tuple[str, str, str, str], bool] = {}
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
            latest[key] = (rec.get("status") == "missing" and bool(rec.get("sparse_ok")))
    return {k for k, accepted in latest.items() if accepted}


def pending_units(units: list[Unit], out_root: str, *, overwrite: bool,
                  manifest_root: str) -> list[Unit]:
    """Units that will actually transfer: all when --overwrite, else those that are neither already
    on disk (final data.parquet exists) NOR already accepted as a sparse-ok missing/empty partition
    in the manifest. The quota/broad gate estimates from THESE (not the full range) and the download
    loop iterates only THESE, so a --resume gates and re-hits Lake ONLY for the genuinely remaining
    work — never re-charging the gate for a mostly-complete batch or re-probing accepted quiet days."""
    if overwrite:
        return list(units)
    accepted = sparse_accepted(manifest_root)
    return [u for u in units
            if not lb.is_done(out_root, u.feed, u.exchange, u.symbol, u.day)
            and (u.feed, u.exchange, u.symbol, u.day) not in accepted]


FEED_MISS_NONFATAL_KIND = "events"   # FEED_KIND class whose missing partitions are non-fatal (sparse)


def feed_miss_is_fatal(feed: str) -> bool:
    """A missing partition is a real data gap for every feed EXCEPT sparse/event feeds (liquidations,
    FEED_KIND 'events' — genuinely absent on quiet days, Risk Q2). The `book` seed and every other
    feed are REQUIRED (recon can't seed/build without them), so a miss there is fatal → the run exits
    3 (partial) rather than reporting a hole as success (plan Requirement 3)."""
    return lb.FEED_KIND.get(feed) != FEED_MISS_NONFATAL_KIND


# ----------------------------------------------------------------------------- live vendor path (approval-gated)
def lake_session(env_path: str = ".env"):
    """Crypto Lake boto3 session from the .env subscriber keys (Lake-only — does NOT require the
    unrelated COINAPI_KEY, unlike ingest/_common.load_env). Mirrors
    scripts/verify_book_delta_v2.py::lake_session credential semantics: explicit keys (NOT the ~/.aws
    default chain, which would auth into the wrong account → AccessDenied), region eu-west-1."""
    import boto3
    env = {}
    if os.path.exists(env_path):
        for line in open(env_path):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env = {**env, **os.environ}
    try:
        return boto3.Session(
            aws_access_key_id=env["aws_access_key_id"],
            aws_secret_access_key=env["aws_secret_access_key"],
            region_name=env.get("region", "eu-west-1"),
        )
    except KeyError as e:
        # A normal exception (NOT SystemExit) so main() maps it to the documented setup exit 2,
        # rather than a bare SystemExit propagating as exit 1 / breaking the 0/2/3/4 contract.
        raise RuntimeError(f"Crypto Lake AWS key {e} not found in .env or environment "
                           "(need aws_access_key_id and aws_secret_access_key).") from None


def used_gb_from_response(resp) -> float:
    """GB used this window from a `lakeapi.used_data()` response — a dict ALREADY in GB
    (`{"downloaded_gb": 151.35, "timeframe_days": 31, …}`; docstring 'Get used data in gigabytes',
    mirrors run_coinbase_quality_map.py's `used.get("downloaded_gb")`). No bytes conversion. Raises
    (KeyError/TypeError) on an unexpected shape ON PURPOSE, so main() exits 2 fail-safe rather than
    gating against a bogus 0 usage (plan Requirement 4: unreadable used_data → fail safe)."""
    return float(resp["downloaded_gb"])


def _live_used_gb(session) -> float:
    """GB of Lake quota consumed this window (telemetry; the counter LAGS ~60 min so it is NOT the
    gate — our own estimate is)."""
    import lakeapi
    return used_gb_from_response(lakeapi.used_data(session))


def _lake_table(feed: str) -> str:
    """Lake `table` name for a scoped feed or the `book` SEED_PRODUCT (they share the raw scheme)."""
    return "book" if feed == lb.SEED_PRODUCT else feed


READ_BATCH_ROWS = 1_000_000    # row-group batch size for the streaming vendor read (never a full day)


def stream_parquet_batches(filesystem, paths, *, batch_size=READ_BATCH_ROWS):
    """Yield pyarrow RecordBatches from parquet `paths` on `filesystem`, one row-group batch at a time.

    This is the whole point of Stage 1's memory contract (Requirement 3 / repo perf rule): a
    book_delta_v2 day is ~109 M rows, so we open each partition object and `iter_batches` it —
    never materializing the day as a DataFrame/Table. Pulled out as a pure generator so it is
    unit-testable offline against a LocalFileSystem, with no live Lake.

    PyArrow 24's ``pre_buffer=True`` coalesces high-latency filesystem reads. Selecting one row
    group per ``iter_batches`` call bounds that read-ahead cache to one row group instead of a
    109 M-row day; yielded batches remain capped at ``batch_size``."""
    import pyarrow.parquet as pq
    for path in sorted(paths):
        with filesystem.open_input_file(path) as handle:
            parquet = pq.ParquetFile(handle, pre_buffer=True)
            for row_group in range(parquet.num_row_groups):
                yield from parquet.iter_batches(batch_size=batch_size, row_groups=[row_group])


def present_days_from_list_records(records) -> list[str]:
    """Sorted ISO day list from `lakeapi.list_data` records — dicts keyed by `dt` (docstring:
    'dicts containing keys table, exchange, symbol, dt, filename'; mirrors ingest/verify_lake.py's
    `{o["dt"] for o in objs}`). Skips any record missing `dt` rather than stringifying the whole
    dict (which would make every requested day read as missing)."""
    return sorted({r["dt"] for r in (records or []) if isinstance(r, dict) and "dt" in r})


def _lake_bucket() -> str:
    """Crypto Lake's S3 bucket+root prefix, resolved from lakeapi's own configured default
    ('qnt.data/market-data/cryptofeed') rather than hard-coded, so a lakeapi upgrade can't drift it."""
    import lakeapi
    bucket = lakeapi.load_data.__globals__.get("default_bucket")
    if not bucket:
        # A normal exception (NOT SystemExit, which is BaseException and would bypass main's
        # `except Exception` reader-setup guard → exit 1) so it maps to the documented setup exit 2.
        raise RuntimeError("could not resolve the Crypto Lake bucket from lakeapi (default_bucket "
                           "missing/changed — check the lakeapi version).")
    return bucket


def _s3_filesystem(session):
    import pyarrow.fs as pafs
    creds = session.get_credentials().get_frozen_credentials()
    return pafs.S3FileSystem(access_key=creds.access_key, secret_key=creds.secret_key,
                             session_token=creds.token,
                             region=session.region_name or "eu-west-1")


def _live_reader(session):
    """Build the live vendor reader: reader(feed, E, S, day) → iterable of pyarrow batches, or None.

    STREAMS the vendor parquet as compressed row-group batches over an S3FileSystem handle (plan
    Requirement 3: `ParquetFile(...).iter_batches(...)`), never calling `lakeapi.load_data`, which
    would return a whole ~109 M-row day as a DataFrame in RAM. Reads the raw vendor columns
    (`timestamp`/`receipt_timestamp`/`side_is_bid`/…) — losslessly; `recon.ingest` aliases them.
    NOT exercised in CI (tests inject a fake reader) and the exact bucket/layout is re-confirmed in
    Phase-1 before the first approval-gated live pull."""
    import pyarrow.fs as pafs
    fs = _s3_filesystem(session)
    bucket = _lake_bucket()

    def read(feed, exchange, symbol, day_iso):
        # Hive partition prefix, matching the Lake bucket layout (docs §Requirement 2).
        prefix = f"{bucket}/{_lake_table(feed)}/exchange={exchange}/symbol={symbol}/dt={day_iso}"
        entries = fs.get_file_info(pafs.FileSelector(prefix, recursive=True, allow_not_found=True))
        paths = [e.path for e in entries
                 if e.type == pafs.FileType.File and e.path.endswith(".parquet")]
        if not paths:
            return None                              # no vendor file for this partition → missing
        return stream_parquet_batches(fs, paths)

    return read


def _live_lister(session):
    """Build the live metadata lister for --dry-run: lister(feed, E, S, start, end) → present ISO days
    IN THE REQUESTED WINDOW. Passing start/end bounds list_data (mirrors
    ingest/verify_trades_and_calendar.py:48) so a one-day dry-run is a one-day metadata probe, not an
    unbounded full-history scan."""
    import lakeapi

    def list_days(feed, exchange, symbol, start, end):
        try:
            meta = lakeapi.list_data(table=_lake_table(feed), start=start, end=end, symbols=[symbol],
                                     exchanges=[exchange], boto3_session=session)
        except lakeapi.exceptions.NoFilesFound:
            return []                                # nothing present for this feed → no days
        return present_days_from_list_records(meta)

    return list_days


# ----------------------------------------------------------------------------- run report
def _write_report(report_dir: str, report: dict) -> str:
    os.makedirs(report_dir, exist_ok=True)
    run_id = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    path = os.path.join(report_dir, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    return path


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage-1 Binance Crypto Lake downloader → normalized ZSTD Parquet raw store "
                    "(streaming, resumable, quota-gated). Live pulls are approval-gated.")
    ap.add_argument("--instrument", default=",".join(lb.INSTRUMENTS),
                    help="comma list of instruments (default all in-scope: "
                         f"{','.join(lb.INSTRUMENTS)})")
    ap.add_argument("--feeds", default=None,
                    help="comma list of feeds (default: all valid for the instrument; book_delta_v2 "
                         "also pulls the `book` seed product)")
    ap.add_argument("--start", help="YYYY-MM-DD inclusive (with --end)")
    ap.add_argument("--end", help="YYYY-MM-DD inclusive (with --start)")
    ap.add_argument("--days-file", help="explicit day list (one YYYY-MM-DD per line); overrides "
                                        "--start/--end (matches the batch planner's batch_NNN_days.txt)")
    ap.add_argument("--out", "--raw", dest="out", default=DEFAULT_OUT_ROOT,
                    help=f"normalized raw store root (default {DEFAULT_OUT_ROOT})")
    ap.add_argument("--manifest", default=None,
                    help="override the _manifest.jsonl store root (default: --out)")
    ap.add_argument("--report-dir", default=DEFAULT_REPORT_DIR,
                    help=f"per-run JSON report dir (default {DEFAULT_REPORT_DIR})")
    ap.add_argument("--dry-run", action="store_true",
                    help="metadata + plan + GB estimate ONLY; zero parquet transfer (still a live "
                         "list_data call)")
    ap.add_argument("--resume", action="store_true",
                    help="rerun only missing/errored units (skip-done is always on unless --overwrite)")
    ap.add_argument("--overwrite", action="store_true", help="re-download existing partitions")
    ap.add_argument("--max-gb", type=float, default=lb.DEFAULT_MAX_GB,
                    help=f"refuse an estimated pull above this unless --allow-broad "
                         f"(default {lb.DEFAULT_MAX_GB})")
    ap.add_argument("--allow-broad", action="store_true",
                    help="permit a broad pull (still capped by the 300 GB/month headroom gate)")
    ap.add_argument("--jobs", type=int, default=1, help=f"parallel independent units, 1-{MAX_DOWNLOAD_JOBS} "
                                                        "(default 1; approved G0-BN Stage 1 uses 4)")
    ap.add_argument("--retries", type=int, default=DEFAULT_RETRIES,
                    help=f"per-unit transient-error retries (default {DEFAULT_RETRIES})")
    ap.add_argument("--engine", choices=("auto", "native", "python"), default="auto",
                    help="reserved; download is engine-agnostic (Stage-2 recon uses it)")
    return ap.parse_args(argv)


def main(argv=None, *, reader=None, lister=None, used_data_fn=None, sleep=time.sleep) -> int:
    """Entry point. Injectable `reader`/`lister`/`used_data_fn` keep the whole path unit-testable with
    fakes; when omitted, the live (approval-gated) Lake vendor path is built lazily."""
    args = parse_args(argv)

    # ---- resolve request (all setup errors → exit 2, before any vendor touch) -------------------
    try:
        args.jobs = validate_download_jobs(args.jobs)
        instrument_keys = list(dict.fromkeys(              # de-dup, order-preserving
            k.strip() for k in args.instrument.split(",") if k.strip()))
        if not instrument_keys:
            raise ValueError(f"--instrument {args.instrument!r} is empty after parsing (only "
                             "separators/whitespace); provide at least one instrument")
        for key in instrument_keys:
            if key not in lb.INSTRUMENTS:
                raise ValueError(f"unknown instrument {key!r} (valid: {list(lb.INSTRUMENTS)})")
        if args.days_file:
            days = load_days_file(args.days_file)
        elif args.start and args.end:
            days = daterange(args.start, args.end)
        else:
            raise ValueError("provide --start and --end, or --days-file")
        if not days:
            raise ValueError("day source resolved to zero days — nothing to download")
        units = plan_units(instrument_keys, args.feeds, days)       # validates + de-dups (instr,feed)
        if not units:                                # catch-all: bad config must never no-op to exit 0
            raise ValueError("resolved zero units to download — check --instrument/--feeds/day source")
        manifest_root = args.manifest or args.out
        pending = pending_units(units, args.out, overwrite=args.overwrite,
                                manifest_root=manifest_root)
        est_gb = sum(unit_gb(u) for u in pending)     # gate/estimate ONLY the not-yet-done units
        projection = runtime_projection(pending, args.jobs)
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return SETUP_ERROR_EXIT

    # A SIGINT may leave a closed but incomplete .tmp. It is never complete; every inert/local
    # resume removes requested-unit temps without scanning unrelated raw-store partitions.
    if not args.dry_run:
        _cleanup_unit_tmps(args.out, units)
    # ---- no-op fast path: a resume whose range is already complete (nothing pending, incl.
    # sparse-accepted quiet days) needs NO Lake session, NO used_data probe, and NO gate. Short-circuit
    # before any vendor touch so an idempotent resume never makes a live call or exits 2 on absent
    # credentials (dry-run still previews presence, so it is intentionally not short-circuited). ----
    if not pending and not args.dry_run:
        report = {"args": {"instruments": instrument_keys, "feeds": args.feeds,
                           "days": [days[0], days[-1]], "n_days": len(days), "out": args.out,
                           "overwrite": args.overwrite, "jobs": args.jobs},
                  "n_units": len(units), "n_pending": 0, "est_gb": 0.0, "dry_run": False,
                  "transferred_gb": 0, "used_data_before": None, "used_data_after": None,
                  "counts": {"ok": 0, "skip": 0, "missing": 0, "error": 0,
                             "hard_stop": 0, "cancelled": 0, "missing_required": 0},
                  "total_rows": 0, "total_out_bytes": 0, "per_unit": [],
                  "runtime_projection": projection,
                  "progress": {"completed": 0, "total": 0, "elapsed_seconds": 0,
                               "rows_per_second": None, "output_bytes_per_second": None,
                               "observed_eta_seconds": 0, "terminal_state": "complete",
                               "eta_caveat": "Observed-rate ETA is a projection, not a bound or "
                                             "live speedup guarantee."},
                  "note": "nothing pending — range already complete"}
        path = _write_report(args.report_dir, report)
        print(f"Nothing to do: all {len(units)} unit(s) already complete (no vendor call). "
              f"report: {path}")
        return 0

    # ---- session + used_data telemetry (fail-safe → exit 2: missing keys/deps OR unreadable
    # used_data — never proceed blind; live session construction is inside the guard too so a
    # missing AWS key / boto3 returns the documented setup exit, not a traceback / exit 1) --------
    try:
        session = None
        if (used_data_fn is None or (reader is None and not args.dry_run)
                or (lister is None and args.dry_run)):
            session = lake_session()                                # live path (approval-gated)
        if used_data_fn is None:
            used_data_fn = lambda: _live_used_gb(session)           # noqa: E731
        used_gb = float(used_data_fn())
    except Exception as e:                                          # noqa: BLE001 — fail safe, do not proceed blind
        print(f"ERROR: Lake setup failed or used_data unreadable (fail-safe, exit 2): {e}",
              file=sys.stderr)
        return SETUP_ERROR_EXIT

    # ---- quota / broad-pull gate BEFORE any transfer (may SystemExit(4)) ------------------------
    lb.check_broad_gate(est_gb=est_gb, max_gb=args.max_gb, allow_broad=args.allow_broad,
                        used_gb=used_gb)

    report = {"args": {"instruments": instrument_keys, "feeds": args.feeds,
                       "days": [days[0], days[-1]] if days else [], "n_days": len(days),
                       "out": args.out, "overwrite": args.overwrite, "jobs": args.jobs},
              "n_units": len(units), "n_pending": len(pending), "est_gb": round(est_gb, 4),
              "used_data_before": round(used_gb, 4),
              "runtime_projection": projection}

    # ---- dry-run: metadata + plan only, zero parquet transfer -----------------------------------
    if args.dry_run:
        # Building the lister AND list_data itself are LIVE Lake calls, so a reader/lister setup
        # failure (missing lakeapi) or an auth/permission wall (wrong keys/account) or any vendor
        # failure must return the documented setup exit 2 (fail-safe) — never a bare traceback / exit
        # 1 — the same hard-stop contract as the download path. Bound the metadata probe to the
        # REQUESTED window [first_day, last_day+1) (list_data end exclusive) so a one-day dry-run is a
        # one-day probe, not a full-history scan.
        probe_start = dt.datetime.fromisoformat(days[0])
        probe_end = dt.datetime.fromisoformat(days[-1]) + dt.timedelta(days=1)
        want = set(days)
        try:
            if lister is None:
                lister = _live_lister(session)
            presence = {}
            for key in instrument_keys:
                inst = lb.INSTRUMENTS[key]
                for feed in plan_feeds_for_presence(key, args.feeds):
                    present = want & set(lister(feed, inst.exchange, inst.symbol,
                                                probe_start, probe_end))
                    presence[f"{key}:{feed}"] = {"n_present": len(present),
                                                 "missing": sorted(want - present)}
        except Exception as e:                                      # noqa: BLE001 — fail safe on a live probe
            print(f"ERROR: --dry-run list_data failed ({classify_error(e)}, fail-safe exit 2): {e}",
                  file=sys.stderr)
            return SETUP_ERROR_EXIT
        report.update(dry_run=True, transferred_gb=0, presence=presence,
                      used_data_after=round(used_gb, 4))
        path = _write_report(args.report_dir, report)
        print(f"[DRY RUN] {len(units)} unit(s), est {est_gb:.2f} GB, zero transfer. report: {path}")
        return 0

    # ---- live download (only the pending units — done + sparse-accepted are skipped) -------------
    try:
        if reader is None:
            reader = _live_reader(session)         # imports pyarrow/lakeapi, resolves bucket + S3 fs
    except Exception as e:                          # noqa: BLE001 — fail safe: setup exit 2, not exit 1
        print(f"ERROR: could not build the live Lake reader (fail-safe, exit 2; is `.[lake]` "
              f"installed?): {e}", file=sys.stderr)
        return SETUP_ERROR_EXIT

    counts = {"ok": 0, "skip": 0, "missing": 0, "error": 0, "hard_stop": 0,
              "cancelled": 0, "missing_required": 0}
    total_rows = 0
    total_out_bytes = 0
    per_unit = []
    hard_stop = False
    hard_stop_errors = []
    lock = Lock()
    cancel_event = Event()
    run_started = time.monotonic()
    completed = 0

    if projection["available"]:
        print(f"Starting {len(pending)} unit(s) with jobs={args.jobs}. Certified "
              f"{projection['basis']['day']} arithmetic runtime reference: "
              f"{_format_duration(projection['idealized_jobs_floor_seconds'])} to "
              f"{_format_duration(projection['serial_reference_seconds'])}. "
              f"{projection['caveat']}")
    else:
        print(f"Starting {len(pending)} unit(s) with jobs={args.jobs}. Runtime reference unavailable "
              f"for feeds {projection['unknown_feeds']}. {projection['caveat']}")

    def _do(u: Unit) -> UnitResult:
        return process_unit(reader, args.out, u.feed, u.exchange, u.symbol, u.day,
                            overwrite=args.overwrite, retries=args.retries, sleep=sleep,
                            manifest_root=manifest_root, lock=lock,
                            sparse_ok=not feed_miss_is_fatal(u.feed),
                            cancel_event=cancel_event)

    def _print_progress(entry: dict) -> None:
        elapsed = max(time.monotonic() - run_started, 1e-9)
        eta = None
        if completed and not cancel_event.is_set():
            eta = elapsed * (len(pending) - completed) / completed
        fields = [
            f"[{completed}/{len(pending)}]",
            f"{entry['feed']} {entry['exchange']} {entry['symbol']} {entry['dt']}",
            f"status={entry['status']} rows={entry['rows']:,}",
        ]
        if entry.get("out_bytes") is not None:
            fields.append(f"bytes={_format_bytes(entry['out_bytes'])}")
        if entry.get("secs") is not None:
            fields.append(f"unit={_format_duration(entry['secs'])}")
        fields.append(
            f"aggregate elapsed={_format_duration(elapsed)} rows/s={total_rows / elapsed:,.0f} "
            f"output/s={_format_bytes(int(total_out_bytes / elapsed))}"
        )
        if eta is not None:
            fields.append(f"ETA~{_format_duration(eta)} (observed-rate projection; not a bound)")
        elif cancel_event.is_set():
            fields.append("ETA unavailable after hard stop")
        print(" | ".join(fields))

    def _account(u: Unit, status: str, *, started: float | None,
                 res: UnitResult | None = None, error: str | None = None) -> None:
        nonlocal total_rows, total_out_bytes, completed
        record = (res.record or {}) if res is not None else {}
        rows = res.rows if res is not None else 0
        out_bytes = record.get("out_bytes")
        unit_secs = round(time.monotonic() - started, 3) if started is not None else None
        counts[status] += 1
        if status == "missing" and feed_miss_is_fatal(u.feed):
            counts["missing_required"] += 1
        total_rows += rows
        total_out_bytes += int(out_bytes or 0)
        entry = {"feed": u.feed, "exchange": u.exchange, "symbol": u.symbol, "dt": u.day,
                 "status": status, "rows": rows}
        if out_bytes is not None:
            entry["out_bytes"] = int(out_bytes)
        if unit_secs is not None:
            entry["secs"] = unit_secs
        if error is not None:
            entry["error"] = error[:500]
        elif record.get("error"):
            entry["error"] = record["error"]
        per_unit.append(entry)
        completed += 1
        _print_progress(entry)

    def _finish(u: Unit, started: float, get_result) -> None:
        nonlocal hard_stop
        try:
            res = get_result()
        except CancelledByHardStop as exc:
            _account(u, "cancelled", started=started, error=str(exc))
        except HardStop as exc:
            # process_unit sets the event while holding the publication lock. Keep this defensive set
            # for injected workers and preserve the same ordering against any final publication.
            with lock:
                cancel_event.set()
            hard_stop = True
            detail = f"{type(exc).__name__}: {exc}"
            hard_stop_errors.append(detail)
            _account(u, "hard_stop", started=started, error=detail)
        else:
            _account(u, res.status, started=started, res=res)

    next_index = 0
    if args.jobs > 1:
        # Keep at most jobs independent units in flight. Each unit pre-buffers only one row group, so
        # the outer fan-out remains bounded and no queued work starts after a quota/auth hard stop.
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            inflight = {}
            while next_index < len(pending) and len(inflight) < args.jobs:
                u = pending[next_index]
                started = time.monotonic()
                inflight[ex.submit(_do, u)] = (u, started)
                next_index += 1

            while inflight:
                done, _ = wait(tuple(inflight), return_when=FIRST_COMPLETED)
                for fut in done:
                    u, started = inflight.pop(fut)
                    _finish(u, started, fut.result)
                while (not cancel_event.is_set() and next_index < len(pending)
                       and len(inflight) < args.jobs):
                    u = pending[next_index]
                    started = time.monotonic()
                    inflight[ex.submit(_do, u)] = (u, started)
                    next_index += 1
    else:
        while next_index < len(pending):
            u = pending[next_index]
            started = time.monotonic()
            _finish(u, started, lambda u=u: _do(u))
            next_index += 1
            if cancel_event.is_set():
                break

    # These units were never submitted after a hard stop. Report them, but do not append synthetic
    # manifest records: only actual unit attempts belong in the resume ledger.
    if cancel_event.is_set():
        for u in pending[next_index:]:
            _account(u, "cancelled", started=None, error="not started after run hard stop")

    if hard_stop:
        print(f"*** Lake hard stop ({'; '.join(hard_stop_errors)}) — exiting 2 (fail-safe); "
              "post-stop quota telemetry was not called. ***", file=sys.stderr)

    used_after = None
    if not hard_stop:
        try:
            used_after = float(used_data_fn())
        except Exception:                            # noqa: BLE001 — post-run telemetry is best-effort
            pass

    terminal_state = (
        "hard_stop" if hard_stop
        else "partial" if counts["error"] or counts["missing_required"]
        else "complete"
    )
    elapsed = max(time.monotonic() - run_started, 0.0)
    progress = {
        "completed": completed,
        "total": len(pending),
        "elapsed_seconds": round(elapsed, 3),
        "rows_per_second": round(total_rows / elapsed, 3) if elapsed else None,
        "output_bytes_per_second": round(total_out_bytes / elapsed, 3) if elapsed else None,
        "observed_eta_seconds": None if hard_stop else 0,
        "terminal_state": terminal_state,
        "eta_caveat": "Observed-rate ETA is a projection, not a bound or live speedup guarantee.",
    }
    report.update(dry_run=False, transferred_gb=None, counts=counts, total_rows=total_rows,
                  total_out_bytes=total_out_bytes,
                  used_data_after=(round(used_after, 4) if used_after is not None else None),
                  per_unit=per_unit, progress=progress, hard_stop_errors=hard_stop_errors)
    path = _write_report(args.report_dir, report)
    finish = "Stopped" if hard_stop else "Finished partial" if terminal_state == "partial" else "Done"
    print(f"{finish}. ok={counts['ok']} skip={counts['skip']} missing={counts['missing']} "
          f"(required-missing={counts['missing_required']}) error={counts['error']} "
          f"hard-stop={counts['hard_stop']} cancelled={counts['cancelled']} "
          f"| rows={total_rows:,} output={_format_bytes(total_out_bytes)} | report: {path}")

    if hard_stop:                                      # quota/credit exhaustion OR auth/permission
        return SETUP_ERROR_EXIT
    if counts["error"] or counts["missing_required"]:  # a transfer error OR a required-feed hole
        return PARTIAL_EXIT
    return 0


def plan_feeds_for_presence(instrument_key: str, feeds_arg: str | None) -> list[str]:
    """Feeds to probe in --dry-run: the selected output feeds + the `book` seed when book_delta_v2 is
    selected (so a dry run reports the seed's coverage too)."""
    feeds = resolve_feeds(instrument_key, feeds_arg)
    if "book_delta_v2" in feeds:
        feeds = feeds + [lb.SEED_PRODUCT]
    return feeds


if __name__ == "__main__":
    raise SystemExit(main())
