"""Stage-2 Binance local reconstruction runner: normalized raw store -> bar-ready processed store.

Reads ONLY the local Stage-1 normalized raw store (`data/raw/lake/...`, written by
`ingest/download_lake_binance.py`) — quota-free, re-runnable, NO vendor I/O of any kind. Per
`(instrument, day)` it:

  * reconstructs `book_delta_v2` into the top-K L2 frame (`data/processed/topk_l2/...`) through the
    EXISTING seed/reseed engines — `recon.reseed.reconstruct_lake_l2_at_samples_seeded` (the Python
    correctness oracle) or the native `recon_native` core via `recon.native` — never reimplementing
    reconstruction semantics. The `book` 20-level seed product is read from the raw store and both
    frames resolve to ONE shared, fully-populated engine-time column
    (`ingest.lake_binance.resolve_engine_time`, plan Requirement 4);
  * normalizes trades/funding/open_interest/liquidations through the explicit fail-loud
    `ingest.lake_binance.NORMALIZERS` contracts into same-named processed tables;
  * enforces the seed-source crossed-rate gate (plan Requirement 5, mirroring
    `scripts/run_coinbase_quality_map.py::classify_day`): a day whose thinned `book` seed candidates
    are >5% crossed is `inconclusive` and publishes NO certified top-K output, even when an
    individually-valid snapshot is accepted. The gate runs BEFORE the expensive replay (its
    candidate classifications are pinned equal to the replay's reason codes by test), so a
    109 M-row perp day that cannot certify never costs a full replay;
  * FAIL-CLOSED publishing: only a `certified` day writes `topk_l2/data.parquet`. `degraded`
    (over the usable thresholds) and `inconclusive` days publish nothing — they exist only as
    manifest records, so no consumer can mistake them for good data;
  * writes outputs atomically (`.tmp` -> `os.replace`) and appends one resumable processed-manifest
    record per unit with status, rows, sha256, schema version, engine + engine-time choice,
    dropped-row counts, quality classification, and the plan-Requirement-5 reconstruction meta
    (seed/reseed/crossed metrics, `seed_source_crossed_frac`).

Engine selection (`--engine {auto,python,native}`) resolves per instrument BEFORE any file is read
(`recon.native.resolve_engine`): explicit `native` aborts (exit 2) when the extension or a verified
tick scale is missing — never a silent fallback; `auto` falls back to Python with a printed note.
No Binance tick scale is registered in `recon/native.py::_TICK_SCALE` yet (plan Risk Q1 — no
recorded verification evidence), so Binance days currently run the Python oracle.

`--jobs N` fans out by `(instrument, feed, day)` — a single book stream is stateful/sequential and
is never split. Threads bound memory, not CPU: the Python replay holds the GIL, so real wall-clock
parallelism arrives with the native engine / process-level orchestration (plan Requirement 7,
native-recon plan §Follow-on architecture). Keep N small for perp book days (~4-6 GB each).

Exit codes: 0 all units ok/skip (inconclusive/degraded are recorded VERDICTS, not errors — rerunning
cannot change them) · 2 setup error (bad args, unresolvable explicit --engine native, bad grid) ·
3 completed with >=1 errored unit or a REQUIRED feed's raw partition missing (rerun after fixing
Stage-1). Plan: docs/superpowers/plans/2026-07-02-binance-downloader-plan.md Task 7; issue #36.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from threading import Lock

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ingest import lake_binance as lb                                          # noqa: E402
from ingest.download_lake_binance import (                                     # noqa: E402
    _rm, _sha256_file, _write_report, daterange, feed_miss_is_fatal, load_days_file, now_iso,
    resolve_feeds,
)
from recon import native as _native                                            # noqa: E402
from recon.reseed import (                                                     # noqa: E402
    ReseedPolicy, classify_snapshot, reconstruct_lake_l2_at_samples_seeded,
    snapshots_from_lake_book_df,
)

DEFAULT_RAW_ROOT = "data/raw/lake"
DEFAULT_OUT_ROOT = "data/processed"
DEFAULT_REPORT_DIR = "data/reports/binance_recon"

SETUP_ERROR_EXIT = 2    # bad args / explicit native unresolvable / invalid grid
PARTIAL_EXIT = 3        # >=1 errored unit or required raw partition missing

NS_PER_MS = 1_000_000
DAY_MS = 86_400_000
BOOK_MAX_LEVELS = 20    # the Lake `book` seed product is a 20-level snapshot

# Day-level quality classification of a topk_l2 unit (the manifest's `classification`).
CERTIFIED = "certified"
DEGRADED = "degraded"
INCONCLUSIVE = "inconclusive"
# Reason code for an accepted seed whose `book` SOURCE is itself crossed above the bar — the exact
# contract of run_coinbase_quality_map.py::classify_day (docs §5a-QualityMap, PR #13 fill policy).
SEED_SOURCE_UNRELIABLE = "seed_accepted_but_source_unreliable"


@dataclass(frozen=True)
class Thresholds:
    """Usable-day thresholds — the same conservative bars as the Coinbase quality map
    (`run_coinbase_quality_map.Thresholds`), applied to the seeded reconstruction and emitted in
    every manifest record so a verdict is always auditable against the bars that produced it."""
    crossed_usable_max: float = 0.01     # <=1% of grid samples crossed after reseed
    missing_usable_max: float = 0.02     # <=2% of grid samples with no top-of-book on a side
    thin_usable_max: float = 0.10        # <=10% of samples present+uncrossed but thin (< k/side)
    seed_crossed_frac_max: float = 0.05  # >5% crossed seed candidates => source unreliable

    def as_dict(self) -> dict:
        return {"crossed_usable_max": self.crossed_usable_max,
                "missing_usable_max": self.missing_usable_max,
                "thin_usable_max": self.thin_usable_max,
                "seed_crossed_frac_max": self.seed_crossed_frac_max}


def build_grid(day: dt.date, grid_ms: int) -> list[int]:
    """Exchange-time sample grid (int ns) spanning the partition day at `grid_ms` spacing —
    mirrors `run_coinbase_quality_map.build_grid` including the divisibility guard (a grid that
    does not divide the 24 h day would silently truncate before midnight)."""
    if grid_ms <= 0 or DAY_MS % grid_ms:
        raise ValueError(f"grid_ms must be positive and divide the {DAY_MS} ms day evenly "
                         f"(got {grid_ms})")
    day_open = int(pd.Timestamp(day).value)
    step_ns = grid_ms * NS_PER_MS
    return [day_open + i * step_ns for i in range(DAY_MS // grid_ms)]


# ----------------------------------------------------------------------------- seed-source gate
def preclassify_snapshots(snapshots, policy: ReseedPolicy) -> dict:
    """Classify every thinned `book` seed candidate BEFORE replay, with the same
    `classify_snapshot` + policy the replay itself applies — so the seed-source gate can refuse a
    day without paying for the full delta replay. The returned `reason_codes`, `seed_accepted`,
    `seed_reason` and `seed_ts` are pinned equal to the replay's own meta by test
    (tests/test_lake_binance_seed_source.py): the replay classifies each snapshot exactly once in
    ts order, which is what this reproduces."""
    codes: Counter = Counter()
    seed_accepted = False
    seed_reason = "no_snapshots"
    seed_ts: int | None = None
    for s in sorted(snapshots, key=lambda s: s.ts):
        r = classify_snapshot(s, min_levels_per_side=policy.min_levels_per_side,
                              max_spread_frac=policy.max_spread_frac)
        codes[r] += 1
        if not seed_accepted:
            if r == "ok":
                seed_accepted, seed_reason, seed_ts = True, "ok", int(s.ts)
            elif seed_reason == "no_snapshots":
                seed_reason = r    # remember the first rejection cause (replay semantics)
    n = sum(codes.values())
    frac = (codes.get("crossed", 0) / n) if n else 0.0
    return {"reason_codes": dict(codes), "n_candidates": n, "seed_accepted": seed_accepted,
            "seed_reason": seed_reason, "seed_ts": seed_ts, "seed_source_crossed_frac": frac}


def seed_gate_verdict(pre: dict, thresholds: Thresholds) -> tuple[str, list[str]] | None:
    """The pre-replay INCONCLUSIVE gate: None when the day may proceed to replay, else
    `(INCONCLUSIVE, reasons)`. Order mirrors `run_coinbase_quality_map.classify_day`: no/rejected
    seed first, then the crossed-source bar — an accepted seed off a >5%-crossed source is NOT
    certifiable (on Coinbase such days stayed crossed for hours, docs §5a-QualityMap)."""
    frac = pre["seed_source_crossed_frac"]
    if pre["n_candidates"] == 0:
        return INCONCLUSIVE, ["no_seed_snapshots"]
    if not pre["seed_accepted"]:
        return INCONCLUSIVE, [f"seed_rejected:{pre['seed_reason']}",
                              f"seed_source_crossed_frac={frac:.4f}"]
    if frac > thresholds.seed_crossed_frac_max:
        return INCONCLUSIVE, [SEED_SOURCE_UNRELIABLE,
                              f"seed_source_crossed_frac={frac:.4f}>"
                              f"{thresholds.seed_crossed_frac_max}"]
    return None


def classify_replay(meta: dict, thresholds: Thresholds) -> tuple[str, list[str], float]:
    """Post-replay classification `(classification, reasons, seed_source_crossed_frac)` from the
    engine's meta — the same rules as `run_coinbase_quality_map.classify_day` minus the
    Coinbase-fill routing. The seed checks are re-derived from meta (identical to the pre-gate by
    construction) so this function alone upholds the invariant even if a caller skips the gate."""
    codes = meta.get("snapshot_reason_codes") or {}
    n_snap = sum(codes.values())
    frac = (codes.get("crossed", 0) / n_snap) if n_snap else 0.0
    if not meta.get("seed_accepted"):
        sr = meta.get("seed_reason")
        code = "no_seed_snapshots" if sr in (None, "no_snapshots") else f"seed_rejected:{sr}"
        return INCONCLUSIVE, [code], frac
    if frac > thresholds.seed_crossed_frac_max:
        return INCONCLUSIVE, [SEED_SOURCE_UNRELIABLE,
                              f"seed_source_crossed_frac={frac:.4f}>"
                              f"{thresholds.seed_crossed_frac_max}"], frac
    crossed = float(meta["crossed_rate"])
    missing = float(meta["missing_book_fraction"])
    thin = float(meta.get("thin_depth_fraction") or 0.0)
    over: list[str] = []
    if crossed > thresholds.crossed_usable_max:
        over.append(f"crossed_rate_after={crossed:.4f}>{thresholds.crossed_usable_max}")
    if missing > thresholds.missing_usable_max:
        over.append(f"missing_book_fraction={missing:.4f}>{thresholds.missing_usable_max}")
    if thin > thresholds.thin_usable_max:
        over.append(f"thin_depth_fraction={thin:.4f}>{thresholds.thin_usable_max}")
    if over:
        return DEGRADED, ["seed_accepted", *over], frac
    return CERTIFIED, ["seed_accepted", f"crossed_rate_after={crossed:.4f}",
                       f"missing_book_fraction={missing:.4f}",
                       f"thin_depth_fraction={thin:.4f}"], frac


def _seeded_reconstruct(engine, price_scale, *, df, grid, k, engine_col, snapshots, policy):
    """Dispatch to the native or Python seeded reconstruction — identical `(frame, meta)` schema
    (mirrors `run_coinbase_quality_map._seeded_reconstruct`). Reuses the engines as-is."""
    if engine == "native":
        return _native.reconstruct_lake_l2_at_samples_seeded_native(
            df, grid, k=k, engine_time_col=engine_col, snapshots=snapshots, policy=policy,
            frame_out=True, price_scale=price_scale)
    return reconstruct_lake_l2_at_samples_seeded(
        df, grid, k=k, engine_time_col=engine_col, snapshots=snapshots, policy=policy,
        frame_out=True)


def _seed_block(pre: dict, meta: dict | None) -> dict:
    """The manifest's seed/reseed block. Replay-only fields are None when the pre-replay gate
    refused the day (they were never measured — never report an unmeasured 0)."""
    return {
        "snapshots_present": pre["n_candidates"] > 0,
        "snapshot_candidates": pre["n_candidates"],
        "seed_accepted": bool(meta["seed_accepted"]) if meta else pre["seed_accepted"],
        "seed_reason": meta["seed_reason"] if meta else pre["seed_reason"],
        "seed_ts": meta["seed_ts"] if meta else pre["seed_ts"],
        "reseed_count": int(meta["reseed_count"]) if meta else None,
        "reseed_ts": [int(x) for x in meta["reseed_ts"]] if meta else None,
        "reseed_blocked_invalid_snapshot":
            int(meta["reseed_blocked_invalid_snapshot"]) if meta else None,
        "snapshot_reason_codes": dict(meta["snapshot_reason_codes"]) if meta
            else dict(pre["reason_codes"]),
    }


def recon_topk_day(delta_df: pd.DataFrame, book_df: pd.DataFrame | None, *, day: dt.date, k: int,
                   grid_ms: int, engine: str, price_scale: int | None, policy: ReseedPolicy,
                   book_stride_ms: int, thresholds: Thresholds,
                   ) -> tuple[pd.DataFrame | None, dict]:
    """PURE per-day top-K pipeline (no I/O): joint engine-time resolution across the delta and
    `book` seed frames, snapshot parse/thin, the pre-replay seed-source gate, the seeded replay
    through the selected engine, and classification. Returns `(frame, info)` where `frame` is the
    engine's exact top-K output ONLY for a CERTIFIED day (else None — fail closed) and `info`
    carries everything the processed manifest records."""
    have_book = book_df is not None and len(book_df) > 0
    frames = [lb.canonicalize_time_columns(delta_df)]
    if have_book:
        frames.append(lb.canonicalize_time_columns(book_df))
    col, fallback, cleaned, dropped = lb.resolve_engine_time(*frames)
    delta_clean = cleaned[0]
    dropped_rows = {"book_delta_v2": dropped[0]}
    if have_book:
        dropped_rows["book"] = dropped[1]

    snaps = []
    if have_book:
        snaps = snapshots_from_lake_book_df(cleaned[1], engine_time_col=col,
                                            max_levels=BOOK_MAX_LEVELS,
                                            stride_ns=book_stride_ms * NS_PER_MS)
    pre = preclassify_snapshots(snaps, policy)

    info = {
        "engine": engine, "price_scale": price_scale,
        "engine_time_col": col, "engine_time_fallback": bool(fallback),
        "dropped_rows": dropped_rows,
        "src_rows": int(len(delta_df)),
        "book_rows": int(len(book_df)) if book_df is not None else 0,
        "book_present": book_df is not None,
        "k": int(k), "grid_ms": int(grid_ms),
        "policy": policy.as_dict(), "thresholds": thresholds.as_dict(),
    }

    gated = seed_gate_verdict(pre, thresholds)
    if gated is not None:
        cls, reasons = gated
        info.update(classification=cls, reasons=reasons, gated_before_replay=True,
                    seed_source_crossed_frac=pre["seed_source_crossed_frac"],
                    seed=_seed_block(pre, None), quality=None)
        return None, info

    grid = build_grid(day, grid_ms)
    frame, meta = _seeded_reconstruct(engine, price_scale, df=delta_clean, grid=grid, k=k,
                                      engine_col=col, snapshots=snaps, policy=policy)
    cls, reasons, frac = classify_replay(meta, thresholds)
    info.update(
        classification=cls, reasons=reasons, gated_before_replay=False,
        seed_source_crossed_frac=frac, seed=_seed_block(pre, meta),
        quality={
            "n_samples": int(meta["n_samples"]),
            "crossed_samples": int(meta["crossed_samples"]),
            "crossed_rate": float(meta["crossed_rate"]),
            "missing_book_samples": int(meta["missing_book_samples"]),
            "missing_book_fraction": float(meta["missing_book_fraction"]),
            "thin_depth_samples": int(meta["thin_depth_samples"]),
            "thin_depth_fraction": float(meta["thin_depth_fraction"]),
            "crossed_duration_s": float(meta["crossed_duration_s"]),
            "n_invalid_runs": int(meta["coverage"]["n_invalid_runs"]),
        })
    return (frame if cls == CERTIFIED else None), info


# ----------------------------------------------------------------------------- local parquet I/O
def _read_raw(raw_root: str, feed: str, exchange: str, symbol: str,
              day_iso: str) -> pd.DataFrame | None:
    """One raw-store partition as pandas, or None when absent. Loads ONE day of ONE feed at a time
    (the Requirement-7 memory contract); the streaming row-group reader into the native core is the
    documented follow-on for high perp --jobs."""
    path = lb.raw_parquet_path(raw_root, feed, exchange, symbol, day_iso)
    if not os.path.exists(path):
        return None
    import pyarrow.parquet as pq
    # ParquetFile, NOT read_table: the dataset-API reader infers hive partition columns from the
    # `exchange=/symbol=/dt=` path segments and APPENDS them to the frame (pyarrow >= 24).
    with pq.ParquetFile(path) as pf:
        return pf.read().to_pandas()


def _write_frame_atomic(frame: pd.DataFrame, final: str, *, schema_version: str, output: str,
                        exchange: str, symbol: str, day_iso: str) -> tuple[int, str, int]:
    """Write a processed frame as ZSTD parquet ATOMICALLY (`.tmp` -> os.replace), column order
    preserved. Returns (rows, sha256, out_bytes). The parquet KV metadata carries schema_version +
    rows + partition keys (Requirement 6); the sha256 lives ONLY in the manifest — embedding it
    would change the bytes being hashed."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    os.makedirs(os.path.dirname(final), exist_ok=True)
    tmp = final + ".tmp"
    table = pa.Table.from_pandas(frame, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[b"schema_version"] = schema_version.encode()
    meta[b"rows"] = str(len(frame)).encode()
    meta[b"output"] = output.encode()
    meta[b"exchange"] = exchange.encode()
    meta[b"symbol"] = symbol.encode()
    meta[b"dt"] = day_iso.encode()
    table = table.replace_schema_metadata(meta)
    try:
        pq.write_table(table, tmp, compression="zstd")
        out_bytes = os.path.getsize(tmp)
        sha = _sha256_file(tmp)
        os.replace(tmp, final)
    except BaseException:
        _rm(tmp)
        raise
    return len(frame), sha, out_bytes


# ----------------------------------------------------------------------------- units + resume
@dataclass(frozen=True)
class Unit:
    instrument_key: str
    exchange: str
    symbol: str
    feed: str       # scoped output feed (the `book` seed is an INPUT of the topk unit, not a unit)
    output: str     # processed table name (lb.FEED_TO_OUTPUT[feed])
    day: str


@dataclass(frozen=True)
class RunConfig:
    raw_root: str
    out_root: str
    manifest_root: str
    overwrite: bool
    k: int
    grid_ms: int
    book_stride_ms: int
    policy: ReseedPolicy
    thresholds: Thresholds
    engine_by_key: dict = field(default_factory=dict)   # instrument_key -> (engine, price_scale)


def plan_units(instrument_keys: list[str], feeds_arg: str | None, days: list[str]) -> list[Unit]:
    """The (instrument, feed, day) work list — validated + de-duplicated via the Stage-1
    `resolve_feeds`. The `book` seed product is NOT a unit: it is read as an input whenever a
    book_delta_v2 unit runs."""
    units: list[Unit] = []
    seen: set[Unit] = set()
    for key in instrument_keys:
        inst = lb.INSTRUMENTS[key]
        for feed in resolve_feeds(key, feeds_arg):
            for day in days:
                u = Unit(key, inst.exchange, inst.symbol, feed, lb.FEED_TO_OUTPUT[feed], day)
                if u not in seen:
                    seen.add(u)
                    units.append(u)
    return units


def processed_state(manifest_root: str) -> dict[tuple[str, str, str, str], dict]:
    """(output, exchange, symbol, dt) -> LAST manifest record (last wins, so a rerun supersedes).
    Malformed/blank lines are skipped (a partial write must not crash resume) — mirrors
    `lake_binance.manifest_index` / `download_lake_binance.sparse_accepted`."""
    import json
    path = os.path.join(manifest_root, lb.MANIFEST_NAME)
    state: dict[tuple[str, str, str, str], dict] = {}
    if not os.path.exists(path):
        return state
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                key = (rec["output"], rec["exchange"], rec["symbol"], rec["dt"])
            except (json.JSONDecodeError, KeyError, TypeError):
                continue
            state[key] = rec
    return state


def unit_is_done(u: Unit, cfg: RunConfig, state: dict) -> bool:
    """Resume policy. A unit is done iff its final parquet exists, OR its last manifest verdict is
    terminal AND the inputs that produced that verdict have not changed:

      * `degraded` — deterministic from local data, done until --overwrite;
      * `inconclusive` — done, UNLESS it was inconclusive with the `book` seed product absent and
        the seed parquet has since appeared in the raw store (a later Stage-1 pull unblocks it);
      * `missing` — done only while the raw partition is STILL absent (new raw data re-runs it);
      * `error` (or no record) — always re-run."""
    if cfg.overwrite:
        return False
    if os.path.exists(lb.processed_parquet_path(cfg.out_root, u.output, u.exchange, u.symbol,
                                                u.day)):
        return True
    rec = state.get((u.output, u.exchange, u.symbol, u.day))
    if not rec:
        return False
    status = rec.get("status")
    if status == "degraded":
        return True
    if status == "inconclusive":
        if rec.get("book_present") is False and os.path.exists(
                lb.raw_parquet_path(cfg.raw_root, lb.SEED_PRODUCT, u.exchange, u.symbol, u.day)):
            return False
        return True
    if status == "missing":
        return not os.path.exists(
            lb.raw_parquet_path(cfg.raw_root, u.feed, u.exchange, u.symbol, u.day))
    return False


# ----------------------------------------------------------------------------- per-unit workers
@dataclass
class UnitResult:
    status: str          # ok | skip | missing | inconclusive | degraded | error
    rows: int
    record: dict | None


def _finish(rec: dict, manifest_root: str, lock, t0: float) -> UnitResult:
    rec["secs"] = round(time.monotonic() - t0, 3)
    rec["ts"] = now_iso()
    with lock:
        lb.manifest_append(manifest_root, rec)
    return UnitResult(rec["status"], rec.get("rows", 0), rec)


def process_topk_unit(u: Unit, cfg: RunConfig, lock) -> UnitResult:
    """One (instrument, day) book_delta_v2 -> topk_l2 unit. Only a CERTIFIED day publishes
    `data.parquet`; degraded/inconclusive days remove any stale output and exist only as manifest
    verdicts (fail closed). Raw-partition absence is `missing` (required feed -> run exits 3)."""
    t0 = time.monotonic()
    final = lb.processed_parquet_path(cfg.out_root, u.output, u.exchange, u.symbol, u.day)
    base = {"output": u.output, "feed": u.feed, "exchange": u.exchange, "symbol": u.symbol,
            "dt": u.day, "schema_version": lb.PROCESSED_SCHEMA_VERSION[u.output]}
    engine, price_scale = cfg.engine_by_key[u.instrument_key]
    try:
        delta_df = _read_raw(cfg.raw_root, u.feed, u.exchange, u.symbol, u.day)
        if delta_df is None or len(delta_df) == 0:
            _rm(final)   # never keep a stale output for a partition the raw store no longer backs
            return _finish({**base, "status": "missing", "sparse_ok": False,
                            "empty": delta_df is not None}, cfg.manifest_root, lock, t0)
        book_df = _read_raw(cfg.raw_root, lb.SEED_PRODUCT, u.exchange, u.symbol, u.day)
        frame, info = recon_topk_day(
            delta_df, book_df, day=dt.date.fromisoformat(u.day), k=cfg.k, grid_ms=cfg.grid_ms,
            engine=engine, price_scale=price_scale, policy=cfg.policy,
            book_stride_ms=cfg.book_stride_ms, thresholds=cfg.thresholds)
    except Exception as exc:   # noqa: BLE001 — recorded, run continues, exit 3
        _rm(final + ".tmp")
        return _finish({**base, "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:500]},
                       cfg.manifest_root, lock, t0)
    rec = {**base, **info}
    if frame is not None:
        rows, sha, out_bytes = _write_frame_atomic(
            frame, final, schema_version=base["schema_version"], output=u.output,
            exchange=u.exchange, symbol=u.symbol, day_iso=u.day)
        rec.update(status="ok", rows=rows, sha256=sha, out_bytes=out_bytes)
    else:
        _rm(final)       # fail closed: an earlier certified output must not outlive its verdict
        rec.update(status=info["classification"], rows=0)
    return _finish(rec, cfg.manifest_root, lock, t0)


def process_table_unit(u: Unit, cfg: RunConfig, lock) -> UnitResult:
    """One passthrough unit: raw partition -> fail-loud normalizer -> engine-time resolution ->
    stable engine-time sort -> atomic parquet + manifest record. Missing/empty partitions follow
    the Stage-1 sparse/required policy (liquidations quiet days are non-fatal, Risk Q2)."""
    t0 = time.monotonic()
    final = lb.processed_parquet_path(cfg.out_root, u.output, u.exchange, u.symbol, u.day)
    base = {"output": u.output, "feed": u.feed, "exchange": u.exchange, "symbol": u.symbol,
            "dt": u.day, "schema_version": lb.PROCESSED_SCHEMA_VERSION[u.output]}
    try:
        raw = _read_raw(cfg.raw_root, u.feed, u.exchange, u.symbol, u.day)
        if raw is None or len(raw) == 0:
            _rm(final)
            return _finish({**base, "status": "missing",
                            "sparse_ok": not feed_miss_is_fatal(u.feed),
                            "empty": raw is not None}, cfg.manifest_root, lock, t0)
        norm = lb.NORMALIZERS[u.feed](raw)
        col, fallback, cleaned, dropped = lb.resolve_engine_time(norm)
        out = cleaned[0]
        was_sorted = bool(out[col].is_monotonic_increasing)
        out = out.sort_values(col, kind="stable").reset_index(drop=True)
        rows, sha, out_bytes = _write_frame_atomic(
            out, final, schema_version=base["schema_version"], output=u.output,
            exchange=u.exchange, symbol=u.symbol, day_iso=u.day)
    except Exception as exc:   # noqa: BLE001 — schema drift & friends: recorded, run exits 3
        _rm(final + ".tmp")
        return _finish({**base, "status": "error",
                        "error": f"{type(exc).__name__}: {exc}"[:500]},
                       cfg.manifest_root, lock, t0)
    return _finish({**base, "status": "ok", "rows": rows, "sha256": sha, "out_bytes": out_bytes,
                    "src_rows": int(len(raw)), "engine_time_col": col,
                    "engine_time_fallback": bool(fallback),
                    "dropped_rows": {u.feed: dropped[0]}, "resorted": not was_sorted},
                   cfg.manifest_root, lock, t0)


def process_unit(u: Unit, cfg: RunConfig, lock) -> UnitResult:
    if u.feed == "book_delta_v2":
        return process_topk_unit(u, cfg, lock)
    return process_table_unit(u, cfg, lock)


# ----------------------------------------------------------------------------- CLI
def parse_args(argv=None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Stage-2 Binance reconstruction: local normalized raw store -> certified "
                    "top-K L2 + normalized trades/funding/OI/liquidations. NO vendor I/O.")
    ap.add_argument("--instrument", default=",".join(lb.INSTRUMENTS),
                    help=f"comma list of instruments (default all: {','.join(lb.INSTRUMENTS)})")
    ap.add_argument("--feeds", default=None,
                    help="comma list of feeds (default: all valid for the instrument; "
                         "book_delta_v2 reconstructs, reading the `book` seed as input)")
    ap.add_argument("--start", help="YYYY-MM-DD inclusive (with --end)")
    ap.add_argument("--end", help="YYYY-MM-DD inclusive (with --start)")
    ap.add_argument("--days-file", help="explicit day list (one YYYY-MM-DD per line); overrides "
                                        "--start/--end")
    ap.add_argument("--raw", default=DEFAULT_RAW_ROOT,
                    help=f"Stage-1 normalized raw store root (default {DEFAULT_RAW_ROOT})")
    ap.add_argument("--out", default=DEFAULT_OUT_ROOT,
                    help=f"processed store root (default {DEFAULT_OUT_ROOT})")
    ap.add_argument("--manifest", default=None,
                    help="override the processed _manifest.jsonl root (default: --out)")
    ap.add_argument("--report-dir", default=DEFAULT_REPORT_DIR,
                    help=f"per-run JSON report dir (default {DEFAULT_REPORT_DIR})")
    ap.add_argument("--k", type=int, default=10, help="top-K book depth (default 10)")
    ap.add_argument("--grid-s", type=float, default=1.0,
                    help="sample grid spacing in seconds (default 1.0; must divide the day)")
    ap.add_argument("--engine", choices=("auto", "native", "python"), default="auto",
                    help="replay engine; explicit `native` ABORTS if the extension or a verified "
                         "tick scale is missing (never a silent fallback)")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel (instrument, feed, day) units (default 1; keep small for perp "
                         "book days — Requirement 7 RAM bound)")
    ap.add_argument("--overwrite", action="store_true",
                    help="re-run every unit, replacing outputs and superseding verdicts")
    ap.add_argument("--resume", action="store_true",
                    help="rerun only missing/errored units (skip-done is always on unless "
                         "--overwrite; this flag documents intent)")
    ap.add_argument("--book-stride-ms", type=int, default=1000,
                    help="thin the `book` seed stream to one candidate per stride (default 1000)")
    ap.add_argument("--seed-min-levels", type=int, default=5,
                    help="seed-validity depth floor per side (default 5, quality-map default)")
    ap.add_argument("--reseed-after-crossed-s", type=float, default=2.0,
                    help="reseed once the book stays crossed this long (default 2.0)")
    ap.add_argument("--no-reseed", action="store_true",
                    help="disable intraday reseed (seed-once A/B arm)")
    ap.add_argument("--max-spread-frac", type=float, default=None,
                    help="optional sane-spread guard for seed validation (off by default)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)

    # ---- resolve request (all setup errors -> exit 2, before touching any file) -----------------
    try:
        instrument_keys = list(dict.fromkeys(
            k.strip() for k in args.instrument.split(",") if k.strip()))
        if not instrument_keys:
            raise ValueError(f"--instrument {args.instrument!r} is empty after parsing")
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
            raise ValueError("day source resolved to zero days — nothing to reconstruct")
        units = plan_units(instrument_keys, args.feeds, days)
        if not units:
            raise ValueError("resolved zero units — check --instrument/--feeds/day source")
        grid_ms = int(round(args.grid_s * 1000))
        build_grid(dt.date.fromisoformat(days[0]), grid_ms)     # validate divisibility once
        if args.k <= 0:
            raise ValueError(f"--k must be positive (got {args.k})")
        if args.jobs is not None and args.jobs < 1:
            raise ValueError(f"--jobs must be >=1 (got {args.jobs})")
    except (ValueError, KeyError, FileNotFoundError) as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return SETUP_ERROR_EXIT

    # ---- engine resolution per instrument, BEFORE any load (plan Requirement 5) -----------------
    engine_by_key: dict[str, tuple[str, int | None]] = {}
    engine_notes: dict[str, str | None] = {}
    for key in instrument_keys:
        inst = lb.INSTRUMENTS[key]
        engine, scale, note = _native.resolve_engine(args.engine, exchange=inst.exchange,
                                                     symbol=inst.symbol)
        if args.engine == "native" and engine != "native":
            # An explicit native request must never silently fall back (plan Review Checklist).
            print(f"ERROR: --engine native unavailable for {key}: {note}", file=sys.stderr)
            return SETUP_ERROR_EXIT
        if note:
            print(f"note[{key}]: {note}", file=sys.stderr)
        engine_by_key[key] = (engine, scale)
        engine_notes[key] = note

    policy = ReseedPolicy(enabled=not args.no_reseed, min_levels_per_side=args.seed_min_levels,
                          reseed_after_crossed_s=args.reseed_after_crossed_s,
                          max_spread_frac=args.max_spread_frac)
    cfg = RunConfig(raw_root=args.raw, out_root=args.out,
                    manifest_root=args.manifest or args.out, overwrite=args.overwrite,
                    k=args.k, grid_ms=grid_ms, book_stride_ms=args.book_stride_ms,
                    policy=policy, thresholds=Thresholds(), engine_by_key=engine_by_key)

    lb.cleanup_tmp(cfg.out_root)
    state = processed_state(cfg.manifest_root)
    pending = [u for u in units if not unit_is_done(u, cfg, state)]
    n_skip = len(units) - len(pending)

    counts = {"ok": 0, "skip": n_skip, "missing": 0, "missing_required": 0,
              "inconclusive": 0, "degraded": 0, "error": 0}
    total_rows = 0
    per_unit = []
    lock = Lock()

    def _record(u: Unit, res: UnitResult) -> None:
        nonlocal total_rows
        counts[res.status] = counts.get(res.status, 0) + 1
        if res.status == "missing" and feed_miss_is_fatal(u.feed):
            counts["missing_required"] += 1
        total_rows += res.rows
        entry = {"output": u.output, "symbol": u.symbol, "dt": u.day, "status": res.status,
                 "rows": res.rows}
        if res.record and res.record.get("classification"):
            entry["classification"] = res.record["classification"]
        per_unit.append(entry)
        print(f"{u.day}  {u.output:<14} {res.status:<12} rows={res.rows:,}")

    if args.jobs > 1:
        with ThreadPoolExecutor(max_workers=args.jobs) as ex:
            futures = {ex.submit(process_unit, u, cfg, lock): u for u in pending}
            for fut in as_completed(futures):
                _record(futures[fut], fut.result())
    else:
        for u in pending:
            _record(u, process_unit(u, cfg, lock))

    report = {"args": {"instruments": instrument_keys, "feeds": args.feeds,
                       "days": [days[0], days[-1]], "n_days": len(days), "raw": args.raw,
                       "out": args.out, "k": args.k, "grid_ms": grid_ms,
                       "engine": args.engine, "jobs": args.jobs, "overwrite": args.overwrite},
              "engine_by_instrument": {k: {"engine": e, "price_scale": s,
                                           "note": engine_notes[k]}
                                       for k, (e, s) in engine_by_key.items()},
              "policy": policy.as_dict(), "thresholds": cfg.thresholds.as_dict(),
              "n_units": len(units), "n_pending": len(pending), "counts": counts,
              "total_rows": total_rows, "per_unit": per_unit}
    path = _write_report(args.report_dir, report)
    print(f"Done. ok={counts['ok']} skip={counts['skip']} missing={counts['missing']} "
          f"(required-missing={counts['missing_required']}) "
          f"inconclusive={counts['inconclusive']} degraded={counts['degraded']} "
          f"error={counts['error']} | rows={total_rows:,} | report: {path}")

    if counts["error"] or counts["missing_required"]:
        return PARTIAL_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
