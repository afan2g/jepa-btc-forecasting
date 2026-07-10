"""
CoinAPI Flat Files downloader -> partitioned Parquet. Throttled + resumable.

Pulls Coinbase BTC-USD `limitbook_full` (full-depth L3) day by day and writes
Hive-partitioned Parquet (spec §3 `ingest/`, §12.1):

    <out>/limitbook_full/exchange=COINBASE/symbol=BTC-USD/dt=YYYY-MM-DD/data.parquet

Design (see verification notes in coinapi_flatfiles.py):
  * ONE get_object stream per file = 1 request (NOT boto3 multipart download_file,
    which fires many GETs and trips the 10 req/min tier limit). All S3 calls go
    through the shared RateLimiter (8/min).
  * Memory-safe: decompress + parse the (tens-of-GB) CSV in chunks, append Parquet
    row groups via ParquetWriter — never materialize a whole day in RAM.
  * Resumable: a day whose final data.parquet exists is skipped; writes are atomic
    (tmp file -> os.replace); every completed day is logged to _manifest.jsonl.
  * Faithful, lossless schema — no date assumptions baked in (recon's job):
        seq               int64   row order within the day = canonical event order
        time_exchange_ns  int64   ns since midnight UTC (from HH:MM:SS.fffffff)
        time_coinapi_ns   int64   ns since midnight UTC
        update_type       string  SNAPSHOT|ADD|DELETE|MATCH|SET
        is_buy            bool
        entry_px          float64
        entry_sx          float64
        order_id          string  UUID (L3 order-by-order)
    dt/exchange/symbol live in the partition path. Recon combines dt + *_ns
    (and handles the SNAPSHOT block carrying the prior-day close time).

Targets CONSOLIDATED daily partitions (D-YYYYMMDD). The most recent ~3 weeks live
as an hourly tail / in coinapi-daily-tail and are out of scope here (live capture).

Reviewed-manifest mode (issue #53) executes EXACTLY the sparse book/trade fill units authorized
by the canonical reviewed backfill manifest (scripts/review_coinbase_backfill_manifest.py) —
never a contiguous date range, never an intervening non-fill day. It is fail-closed end to end
(planning/acceptance logic in ingest/coinapi_backfill.py): only a ready, scope-complete,
hash-valid, non-stale manifest plans; the default is a DRY-RUN plan with zero vendor I/O; a live
run additionally needs --execute, the operator-pinned --manifest-sha256, an --approve-usd cap
covering the plan's high-band cost, --spend-evidence, and (multi-day) the §5a --allow-backfill
override with CoinAPI Spend Management ON (§8). Resume state is keyed on
source/product/day + the manifest fingerprint, and every run emits a reconciled execution report.

Usage:
  python ingest/download_coinapi.py --start 2025-06-01 --end 2025-06-01               # one parity day
  python ingest/download_coinapi.py --start 2026-05-28 --end 2026-05-28 --sample-mb 8 # cheap smoke test
  # Multi-day BULK = backfill, GATED until the §5a parity+reseed gates pass (docs/data.md §5a/§8):
  # a >1-day full pull exits 4 unless you pass --allow-backfill (with CoinAPI Spend Management on).
  python ingest/download_coinapi.py --start 2025-01-01 --end 2025-06-30 --allow-backfill --keep-raw
  # Reviewed-manifest backfill: DRY-RUN plan by default (no vendor I/O, no credentials needed):
  python ingest/download_coinapi.py --manifest data/reports/backfill/coinbase_backfill_manifest.json
  # Execute the reviewed units (optionally a hash-pinned pilot window) — all gates apply:
  python ingest/download_coinapi.py --manifest ... --execute --manifest-sha256 <hex> \\
      --approve-usd 97 --spend-evidence "issue #33 spend approval" --allow-backfill \\
      [--pilot-start 2024-12-01 --pilot-end 2024-12-31]
"""
from __future__ import annotations
import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
from io import StringIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from _common import load_env, is_quota_error, QUOTA_HINT, check_backfill_gate  # noqa: E402
import coinapi_backfill as bf                                     # noqa: E402
import coinapi_flatfiles as ff                                    # noqa: E402
from botocore.exceptions import ClientError                       # noqa: E402

DATA_TYPE = "LIMITBOOK_FULL"
CHUNK_ROWS = 2_000_000
READ_DTYPE = {
    "time_exchange": str, "time_coinapi": str, "update_type": str,
    "is_buy": "int8", "entry_px": "float64", "entry_sx": "float64", "order_id": str,
}
SCHEMA = pa.schema([
    ("seq", pa.int64()),
    ("time_exchange_ns", pa.int64()),
    ("time_coinapi_ns", pa.int64()),
    ("update_type", pa.string()),
    ("is_buy", pa.bool_()),
    ("entry_px", pa.float64()),
    ("entry_sx", pa.float64()),
    ("order_id", pa.string()),
])

# CoinAPI Flat Files `trades` (T-TRADES/D-YYYYMMDD daily csv.gz). Same faithful-lossless
# discipline as the book: vendor columns preserved, times as int-ns since midnight UTC, seq =
# in-file row order. The column set is the documented flat-files trades layout; it has NOT been
# verified against a live file (no vendor calls in this change), so the executor validates the
# header before parsing and FAILS CLOSED on drift rather than normalizing wrong columns.
TRADES_DATA_TYPE = "TRADES"
TRADES_READ_DTYPE = {
    "time_exchange": str, "time_coinapi": str, "guid": str,
    "price": "float64", "base_amount": "float64", "taker_side": str,
}
TRADES_SCHEMA = pa.schema([
    ("seq", pa.int64()),
    ("time_exchange_ns", pa.int64()),
    ("time_coinapi_ns", pa.int64()),
    ("guid", pa.string()),
    ("price", pa.float64()),
    ("base_amount", pa.float64()),
    ("taker_side", pa.string()),
])


# ----------------------------------------------------------------------------- helpers
def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def find_file(s3, bucket, day_compact, exchange, symbol_tag, data_type=DATA_TYPE):
    """Exact daily file (key, size) for exchange+symbol on a day, or None."""
    objs, _ = ff.list_prefix(s3, bucket, f"T-{data_type}/D-{day_compact}/E-{exchange}/")
    match = f"+SC-{symbol_tag}+"
    hits = [o for o in objs if match in o["Key"]]
    if not hits:
        return None
    o = max(hits, key=lambda x: x["Size"])
    return o["Key"], o["Size"]


def stream_to_file(s3, bucket, key, dest) -> tuple[int, str]:
    """Single throttled get_object streamed to disk (1 request). Returns (bytes, sha256hex) —
    the source hash is computed on the fly so the execution report can pin what was pulled."""
    ff.RL.wait()
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    total, h = 0, hashlib.sha256()
    with open(dest, "wb") as f:
        while True:
            buf = body.read(8 * 1024 * 1024)
            if not buf:
                break
            f.write(buf)
            h.update(buf)
            total += len(buf)
    return total, h.hexdigest()


def to_table(df: pd.DataFrame, seq_start: int) -> pa.Table:
    n = len(df)
    out = pd.DataFrame({
        "seq": range(seq_start, seq_start + n),
        "time_exchange_ns": pd.to_timedelta(df["time_exchange"]).astype("int64"),
        "time_coinapi_ns": pd.to_timedelta(df["time_coinapi"]).astype("int64"),
        "update_type": df["update_type"].fillna("").astype(str),
        "is_buy": df["is_buy"].astype(bool),
        "entry_px": df["entry_px"].astype("float64"),
        "entry_sx": df["entry_sx"].astype("float64"),
        "order_id": df["order_id"].fillna("").astype(str),
    })
    return pa.Table.from_pandas(out, schema=SCHEMA, preserve_index=False)


def trades_to_table(df: pd.DataFrame, seq_start: int) -> pa.Table:
    n = len(df)
    out = pd.DataFrame({
        "seq": range(seq_start, seq_start + n),
        "time_exchange_ns": pd.to_timedelta(df["time_exchange"]).astype("int64"),
        "time_coinapi_ns": pd.to_timedelta(df["time_coinapi"]).astype("int64"),
        "guid": df["guid"].fillna("").astype(str),
        "price": df["price"].astype("float64"),
        "base_amount": df["base_amount"].astype("float64"),
        "taker_side": df["taker_side"].fillna("").astype(str),
    })
    return pa.Table.from_pandas(out, schema=TRADES_SCHEMA, preserve_index=False)


# explicit product handlers for the reviewed-manifest executor (issue #53). Keys are the
# manifest's unit products (coinapi_backfill.PRODUCTS); each pins the vendor flat-file data type,
# the faithful read/write schemas, and the normalizer — vendor schema knowledge stays here at the
# ingestion boundary.
HANDLERS = {
    bf.PRODUCT_BOOK: {"data_type": DATA_TYPE, "read_dtype": READ_DTYPE, "schema": SCHEMA,
                      "to_table": to_table, "dict_cols": ("update_type",)},
    bf.PRODUCT_TRADES: {"data_type": TRADES_DATA_TYPE, "read_dtype": TRADES_READ_DTYPE,
                        "schema": TRADES_SCHEMA, "to_table": trades_to_table,
                        "dict_cols": ("taker_side",)},
}


def write_parquet(chunks, dest_tmp, *, schema=SCHEMA, to_table_fn=to_table,
                  dict_cols=("update_type",)) -> tuple[int, int]:
    """Consume an iterator of DataFrames -> one Parquet file. Returns (rows, bytes).
    Defaults preserve the original LIMITBOOK_FULL behavior byte-for-byte."""
    writer = pq.ParquetWriter(dest_tmp, schema, compression="zstd",
                              use_dictionary=list(dict_cols))
    rows = 0
    try:
        for df in chunks:
            writer.write_table(to_table_fn(df, rows))
            rows += len(df)
    finally:
        writer.close()
    return rows, os.path.getsize(dest_tmp)


def manifest_append(out_root, rec):
    with open(os.path.join(out_root, "_manifest.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def cleanup_tmp(out_root):
    """Remove stale *.tmp / *.gz left by an interrupted run (keeps resume clean)."""
    removed = 0
    for dirpath, _, files in os.walk(out_root):
        for fn in files:
            if fn.endswith(".parquet.tmp") or fn.endswith(".csv.gz.partial"):
                os.remove(os.path.join(dirpath, fn)); removed += 1
    if removed:
        print(f"  cleaned {removed} stale temp file(s)")


# ----------------------------------------------------------------------------- per-day
def process_day(s3, bucket, day, args):
    iso = day.isoformat()
    compact = day.strftime("%Y%m%d")
    base = os.path.join(args.out, "_sample") if args.sample_mb else args.out  # never shadow real data
    part_dir = os.path.join(
        base, DATA_TYPE.lower(),
        f"exchange={args.exchange_out}", f"symbol={args.symbol_out}", f"dt={iso}",
    )
    final = os.path.join(part_dir, "data.parquet")
    if os.path.exists(final) and not args.overwrite:
        return "skip", 0

    located = find_file(s3, bucket, compact, args.exchange, args.symbol_tag)
    if not located:
        print(f"  {iso}  MISSING (no consolidated daily file)")
        manifest_append(args.out, {"dt": iso, "status": "missing", "ts": now_iso()})
        return "missing", 0
    key, src_bytes = located

    os.makedirs(part_dir, exist_ok=True)
    tmp_parq = final + ".tmp"
    tmp_gz = os.path.join(args.out, "_tmp"); os.makedirs(tmp_gz, exist_ok=True)
    gz_path = os.path.join(tmp_gz, key.split("/")[-1] + ".partial")
    t0 = dt.datetime.now()

    try:
        if args.sample_mb:                                   # cheap end-to-end smoke test
            ff.RL.wait()
            rng = s3.get_object(Bucket=bucket, Key=key,
                                Range=f"bytes=0-{args.sample_mb*1024*1024 - 1}")["Body"].read()
            text = ff.gunzip_partial(rng)
            df = pd.read_csv(StringIO("\n".join(text.split("\n")[:-1])),
                             sep=";", dtype=READ_DTYPE)
            chunks = [df]
        else:                                                # full streamed download
            got, _ = stream_to_file(s3, bucket, key, gz_path)
            assert got == src_bytes, f"size mismatch {got} != {src_bytes}"
            chunks = pd.read_csv(gz_path, compression="gzip", sep=";",
                                 chunksize=CHUNK_ROWS, dtype=READ_DTYPE)

        rows, out_bytes = write_parquet(chunks, tmp_parq)
        os.replace(tmp_parq, final)                          # atomic publish
    finally:
        if os.path.exists(tmp_parq):
            os.remove(tmp_parq)
        if os.path.exists(gz_path):
            if args.keep_raw:
                raw_dir = os.path.join(args.out, "_raw_gz",
                                       f"exchange={args.exchange_out}",
                                       f"symbol={args.symbol_out}")
                os.makedirs(raw_dir, exist_ok=True)
                os.replace(gz_path, os.path.join(raw_dir, f"{iso}.csv.gz"))
            else:
                os.remove(gz_path)

    secs = (dt.datetime.now() - t0).total_seconds()
    rec = {"dt": iso, "status": "sample" if args.sample_mb else "ok", "key": key,
           "src_bytes": src_bytes, "rows": rows, "out_bytes": out_bytes,
           "secs": round(secs, 1), "ts": now_iso()}
    manifest_append(args.out, rec)
    tag = f"SAMPLE({args.sample_mb}MB)" if args.sample_mb else f"{src_bytes/1e6:.0f}MB->{out_bytes/1e6:.0f}MB"
    print(f"  {iso}  OK  {rows:>12,} rows  {tag}  {secs:.0f}s")
    return "ok", rows


# ----------------------------------------------------------------------------- manifest units
def _validate_csv_columns(gz_path, required, what) -> None:
    """Fail-closed header check for manifest-mode units: normalizing the wrong vendor columns
    would silently corrupt the backfill, so a header without every expected column is an error
    (the unit is recorded as `error` and nothing is published)."""
    cols = list(pd.read_csv(gz_path, compression="gzip", sep=";", nrows=0).columns)
    missing = [c for c in required if c not in cols]
    if missing:
        raise ValueError(f"{what}: vendor csv is missing expected column(s) {missing} "
                         f"(got {cols}) — refusing to normalize (schema drift?)")


def process_unit(s3, bucket, unit, args, manifest_sha256, budget=None) -> dict:
    """Download + normalize ONE planned fill unit (product×day). Same discipline as process_day:
    one throttled streamed GET, chunked csv->parquet, atomic publish — plus a fingerprint-keyed
    state record so resume can never count a stale or foreign output as done, and a runtime
    budget guard (the vendor LIST size is known before the billable GET, so a unit that would
    push measured spend past --approve-usd is refused, never downloaded). Quota ClientErrors
    propagate (the run loop aborts); any other vendor/parse failure is a per-unit `error`."""
    product, day = unit["product"], unit["day"]
    handler = HANDLERS[product]
    compact = day.replace("-", "")
    part_dir = os.path.join(args.out, product, f"exchange={args.exchange_out}",
                            f"symbol={args.symbol_out}", f"dt={day}")
    final = os.path.join(part_dir, "data.parquet")
    res = {"source": bf.SOURCE, "product": product, "day": day, "kind": unit.get("kind"),
           "status": None, "key": None, "src_bytes": 0, "src_sha256": None, "billed_bytes": 0,
           "rows": 0, "out_bytes": 0, "out_sha256": None, "out_path": None, "secs": 0.0,
           "error": None}

    resume = bf.unit_resume_status(args.out, manifest_sha256, unit, final,
                                   overwrite=args.overwrite)
    if resume == "done":
        st = bf.load_state(bf.state_path(args.out, manifest_sha256, product, day)) or {}
        res.update(status="done", key=st.get("key"), out_bytes=st.get("out_bytes") or 0,
                   out_sha256=st.get("out_sha256"), out_path=final)
        print(f"  {day}  {product}  done (state + output bytes re-verified for this manifest)")
        return res
    if resume == "conflict":
        res.update(status="conflict", out_path=final, error=(
            "existing output has no state record matching this manifest fingerprint and these "
            "exact bytes — refusing to adopt or overwrite it (move it aside, or re-download "
            "with --overwrite)"))
        print(f"  {day}  {product}  CONFLICT (foreign/stale output at {final})")
        return res

    t0 = dt.datetime.now()
    try:
        located = find_file(s3, bucket, compact, args.exchange, args.symbol_tag,
                            data_type=handler["data_type"])
    except ClientError as e:
        if is_quota_error(str(e)):
            raise                                            # abort the whole run (quota gate)
        res.update(status="error", error=f"{type(e).__name__}: {e}"[:300])
        return res
    if not located:
        res.update(status="missing", error="no consolidated daily file")
        manifest_append(args.out, {"dt": day, "product": product, "status": "missing",
                                   "manifest_sha256": manifest_sha256, "ts": now_iso()})
        print(f"  {day}  {product}  MISSING (no consolidated daily file)")
        return res
    key, src_bytes = located

    if budget is not None:
        rate = budget["rates"][product]
        projected = src_bytes / 1e9 * rate
        if budget["spent_usd"] + projected > budget["approve_usd"] + 1e-9:
            res.update(status="refused_budget", key=key, src_bytes=src_bytes, error=(
                f"projected ${projected:.2f} ({src_bytes/1e9:.3f} GB actual) would push measured "
                f"spend past the approved --approve-usd ${budget['approve_usd']:.2f} "
                f"(${budget['spent_usd']:.2f} already committed) — refused before any billable "
                "GET; raise the approval or narrow the window"))
            print(f"  {day}  {product}  REFUSED (budget: projected ${projected:.2f} over cap)")
            return res
        # commit the projection NOW: once the GET below starts, CoinAPI bills these bytes
        # whether or not header validation / parsing / publishing succeeds afterwards —
        # later units must be budgeted against money already spent (Codex P1)
        budget["spent_usd"] += projected
    res["billed_bytes"] = src_bytes   # the GET is now committed; report it as billed even on error

    os.makedirs(part_dir, exist_ok=True)
    tmp_parq = final + ".tmp"
    tmp_gz = os.path.join(args.out, "_tmp"); os.makedirs(tmp_gz, exist_ok=True)
    gz_path = os.path.join(tmp_gz, key.split("/")[-1] + ".partial")
    ok = False
    try:
        try:
            got, src_sha = stream_to_file(s3, bucket, key, gz_path)
            if got != src_bytes:
                raise ValueError(f"size mismatch {got} != {src_bytes}")
            _validate_csv_columns(gz_path, list(handler["read_dtype"]), f"{product} {day}")
            chunks = pd.read_csv(gz_path, compression="gzip", sep=";", chunksize=CHUNK_ROWS,
                                 dtype=handler["read_dtype"])
            rows, _ = write_parquet(chunks, tmp_parq, schema=handler["schema"],
                                    to_table_fn=handler["to_table"],
                                    dict_cols=handler["dict_cols"])
            out_bytes = os.path.getsize(tmp_parq)
            out_sha = bf.sha256_file(tmp_parq)
            # state BEFORE publish: a crash between the two leaves state-without-output, which
            # resumes as a plain re-download (todo) — never a conflict demanding --overwrite
            bf.write_state(args.out, manifest_sha256, unit,
                           {"status": "ok", "key": key, "src_bytes": src_bytes,
                            "src_sha256": src_sha, "rows": rows, "out_bytes": out_bytes,
                            "out_sha256": out_sha, "out_path": final,
                            "completed_utc": now_iso()})
            os.replace(tmp_parq, final)                      # atomic publish
            ok = True
        except ClientError as e:
            if is_quota_error(str(e)):
                raise                                        # abort the whole run (quota gate)
            res.update(status="error", key=key, error=f"{type(e).__name__}: {e}"[:300])
            return res
        except Exception as e:
            res.update(status="error", key=key, error=f"{type(e).__name__}: {e}"[:300])
            return res
    finally:
        if os.path.exists(tmp_parq):
            os.remove(tmp_parq)
        if os.path.exists(gz_path):
            if ok and args.keep_raw:   # never archive a failed/partial download as canonical raw
                raw_dir = os.path.join(args.out, "_raw_gz", product,
                                       f"exchange={args.exchange_out}",
                                       f"symbol={args.symbol_out}")
                os.makedirs(raw_dir, exist_ok=True)
                os.replace(gz_path, os.path.join(raw_dir, f"{day}.csv.gz"))
            else:
                os.remove(gz_path)

    secs = round((dt.datetime.now() - t0).total_seconds(), 1)
    res.update(status="ok", key=key, src_bytes=src_bytes, src_sha256=src_sha, rows=rows,
               out_bytes=out_bytes, out_sha256=out_sha, out_path=final, secs=secs)
    manifest_append(args.out, {"dt": day, "product": product, "status": "ok", "key": key,
                               "src_bytes": src_bytes, "rows": rows, "out_bytes": out_bytes,
                               "manifest_sha256": manifest_sha256, "secs": secs,
                               "ts": now_iso()})
    print(f"  {day}  {product}  OK  {rows:>12,} rows  "
          f"{src_bytes/1e6:.0f}MB->{out_bytes/1e6:.0f}MB  {secs:.0f}s")
    return res


def _print_plan_summary(plan) -> None:
    m, t = plan["meta"], plan["totals"]
    print("=" * 74)
    print(f"  COINAPI BACKFILL PLAN — mode {m['mode']} | manifest sha256 "
          f"{m['manifest']['sha256']}")
    if m["window"]:
        print(f"  pilot window {m['window']['start']}..{m['window']['end']} "
              f"({plan['scope']['skipped_by_window']} unit(s) outside the window)")
    print(f"  units: {t['n_units']}  (book {t['book_units']}: {t['book_full_days']} full-day + "
          f"{t['book_partial_days']} partial-day file(s); trades {t['trade_units']})")
    print(f"  GB: book {t['book_gb_total']} ({t['book_gb_measured']} measured + "
          f"{t['book_gb_estimated']} estimated) | trades {t['trades_gb_total']}")
    print(f"  cost band: ${t['usd_low']:.2f}-${t['usd_high']:.2f} at the manifest's model rates")


def _live_s3_factory():
    s3 = ff.make_client(load_env()["COINAPI_KEY"])
    return s3, ff.discover_bucket(s3)


QUOTA_ABORT_EXIT = 5   # manifest-mode quota abort — distinct from input-error 2 / refusal 3 /
                       # gate 4 (range mode keeps its legacy exit 2 on quota)

# The reviewed manifest is pinned COINBASE/BTC-USD; a CLI market override in manifest mode would
# download another market's files and record them as Coinbase fills. (flag, args attr, pin)
_MANIFEST_MODE_MARKET_PINS = (("--exchange", "exchange", "COINBASE"),
                              ("--symbol-tag", "symbol_tag", "COINBASE_SPOT_BTC_USD"),
                              ("--exchange-out", "exchange_out", "COINBASE"),
                              ("--symbol-out", "symbol_out", "BTC-USD"))


def run_manifest_mode(args, s3_factory=None) -> int:
    """Reviewed-manifest execution (issue #53). Fail-closed order: refuse market overrides and
    missing authorization flags, build+reconcile the plan (refusals exit 3, input errors exit 2),
    write the plan, stop there on a dry run; else enforce the spend cap and the §5a backfill gate
    BEFORE any vendor client exists, run exactly the planned units (with a runtime budget guard
    against the approved cap), and emit the reconciled execution report."""
    generated = args.generated_utc or now_iso()
    overridden = [f"{flag}={getattr(args, attr)!r}" for flag, attr, pin in
                  _MANIFEST_MODE_MARKET_PINS if getattr(args, attr) != pin]
    if overridden:
        print("REFUSING manifest mode with market overrides (" + ", ".join(overridden) + "): "
              "the reviewed manifest is pinned to COINBASE/BTC-USD.", file=sys.stderr)
        return bf.REFUSAL_EXIT
    if args.execute:
        missing = [flag for flag, v in (("--manifest-sha256", args.manifest_sha256),
                                        ("--approve-usd", args.approve_usd),
                                        ("--spend-evidence", args.spend_evidence)) if not v]
        if missing:
            print("REFUSING --execute without " + ", ".join(missing) + ": a live backfill needs "
                  "the operator-pinned manifest hash and explicit spend authorization "
                  "(docs/data.md §8 — enable CoinAPI Spend Management first).", file=sys.stderr)
            return bf.REFUSAL_EXIT
    try:
        plan = bf.build_plan(args.manifest, generated_utc=generated,
                             expected_sha256=args.manifest_sha256,
                             window_start=args.pilot_start, window_end=args.pilot_end,
                             mode="execute" if args.execute else "dry_run")
    except bf.BackfillInputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return bf.INPUT_ERROR_EXIT
    except bf.BackfillRefusal as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return bf.REFUSAL_EXIT
    bf.write_json_atomic(plan, args.plan_out)
    _print_plan_summary(plan)
    print(f"  wrote plan: {args.plan_out}")
    if not args.execute:
        print("  DRY-RUN ONLY — no vendor I/O. To execute: add --execute --manifest-sha256 "
              f"{plan['meta']['manifest']['sha256']} --approve-usd <cap> "
              "--spend-evidence <approval ref> (multi-day also needs --allow-backfill; "
              "enable CoinAPI Spend Management, docs/data.md §8).")
        return 0

    high = plan["totals"]["usd_high"]
    approve = float(args.approve_usd)
    # positive-comparison form: NaN fails every comparison, so `approve < high` alone would let
    # --approve-usd nan through the spend gate. Require a finite, positive cap covering the band.
    if not (math.isfinite(approve) and approve > 0 and approve >= high):
        print(f"REFUSING: --approve-usd {args.approve_usd} does not cover the selected plan's "
              f"high-band cost ${high} (the cap must be a finite positive USD amount >= the "
              "band) — raise the approval or narrow the pilot window.", file=sys.stderr)
        return bf.REFUSAL_EXIT
    manifest_sha = plan["meta"]["manifest"]["sha256"]
    cm = plan["meta"]["cost_model"]
    budget = {"approve_usd": approve, "spent_usd": 0.0,
              "rates": {bf.PRODUCT_BOOK: float(cm["book_usd_per_gb"]),
                        bf.PRODUCT_TRADES: float(cm["trades_usd_per_gb"])}}
    results, quota_hit = [], False
    if plan["units"]:
        days = sorted({u["day"] for u in plan["units"]})
        check_backfill_gate(dt.date.fromisoformat(days[0]), dt.date.fromisoformat(days[-1]),
                            sample_mb=0, allow_backfill=args.allow_backfill)
        s3, bucket = (s3_factory or _live_s3_factory)()
        os.makedirs(args.out, exist_ok=True)
        cleanup_tmp(args.out)
        # durable, append-only record of the run authorization: report files at --report-out are
        # replaced per run, so the spend evidence for EVERY run (incl. deliberate --overwrite
        # re-bills) must survive somewhere the next run cannot clobber
        manifest_append(args.out, {"kind": "backfill_run", "manifest_sha256": manifest_sha,
                                   "approve_usd": approve,
                                   "spend_evidence": args.spend_evidence,
                                   "allow_backfill": bool(args.allow_backfill),
                                   "overwrite": bool(args.overwrite),
                                   "window": plan["meta"]["window"],
                                   "n_units": len(plan["units"]), "ts": now_iso()})
        print(f"Executing {len(plan['units'])} reviewed fill unit(s) -> {args.out}  "
              f"(bucket={bucket} throttle={ff.REQ_PER_MIN}/min)")
        for unit in plan["units"]:
            try:
                results.append(process_unit(s3, bucket, unit, args, manifest_sha,
                                            budget=budget))
            except ClientError as e:
                if is_quota_error(str(e)):
                    print(QUOTA_HINT)
                    quota_hit = True
                    break
                raise
    else:
        print("  0 units selected — nothing to download (empty fill scope)")
    report = bf.build_execution_report(
        plan, results, generated_utc=generated,
        spend={"approve_usd": approve, "spend_evidence": args.spend_evidence,
               "allow_backfill": bool(args.allow_backfill),
               "overwrite": bool(args.overwrite)})
    bf.write_json_atomic(report, args.report_out)
    rec = report["reconciliation"]
    print(f"\nDone. planned={rec['planned']} ok={rec['ok']} done_prior={rec['done_prior']} "
          f"missing={rec['missing']} conflict={rec['conflict']} error={rec['error']} "
          f"refused_budget={rec['refused_budget']} | "
          f"downloaded {rec['bytes_downloaded']/1e9:.3f} GB, {rec['rows_written']:,} rows | "
          f"complete={rec['complete']} | report: {args.report_out}")
    if quota_hit:
        return QUOTA_ABORT_EXIT
    return 0 if rec["complete"] else 1


# ----------------------------------------------------------------------------- main
def main(argv=None, s3_factory=None):
    ap = argparse.ArgumentParser(description="Download CoinAPI flat files -> partitioned Parquet")
    ap.add_argument("--start", default=None, help="YYYY-MM-DD inclusive (range mode)")
    ap.add_argument("--end", default=None, help="YYYY-MM-DD inclusive (range mode)")
    ap.add_argument("--out", default="data/raw")
    ap.add_argument("--exchange", default="COINBASE", help="CoinAPI exchange id (bucket path)")
    ap.add_argument("--symbol-tag", default="COINBASE_SPOT_BTC_USD", help="CoinAPI SC- symbol id")
    ap.add_argument("--exchange-out", default="COINBASE", help="partition exchange= value")
    ap.add_argument("--symbol-out", default="BTC-USD", help="partition symbol= value")
    ap.add_argument("--keep-raw", action="store_true", help="archive the source .csv.gz")
    ap.add_argument("--overwrite", action="store_true", help="re-download existing days")
    ap.add_argument("--sample-mb", type=int, default=0, help="dev: parse only first N MB (smoke test)")
    ap.add_argument("--allow-backfill", action="store_true",
                    help="override the §5a backfill gate for a multi-day full pull (blocked by "
                         "default until recon parity + reseed pass — docs/data.md §5a/§8)")
    ap.add_argument("--manifest", default=None,
                    help="MANIFEST MODE: canonical reviewed backfill manifest "
                         "(scripts/review_coinbase_backfill_manifest.py output). Executes exactly "
                         "its sparse fill units; dry-run plan by default")
    ap.add_argument("--manifest-sha256", default=None,
                    help="operator-pinned sha256 of the reviewed manifest (required with "
                         "--execute; the run refuses any other manifest bytes)")
    ap.add_argument("--pilot-start", default=None,
                    help="manifest mode: pilot-window subset start day (inclusive)")
    ap.add_argument("--pilot-end", default=None,
                    help="manifest mode: pilot-window subset end day (inclusive)")
    ap.add_argument("--execute", action="store_true",
                    help="manifest mode: actually download (default is a dry-run plan with no "
                         "vendor I/O); needs --manifest-sha256, --approve-usd, --spend-evidence")
    ap.add_argument("--approve-usd", type=float, default=None,
                    help="manifest mode: authorized spend cap in USD; must cover the selected "
                         "plan's high-band cost")
    ap.add_argument("--spend-evidence", default=None,
                    help="manifest mode: where the human spend approval lives (e.g. issue/comment "
                         "URL) — recorded in the execution report")
    ap.add_argument("--plan-out", default="data/reports/backfill/coinapi_backfill_plan.json",
                    help="manifest mode: where to write the deterministic dry-run/execute plan")
    ap.add_argument("--report-out",
                    default="data/reports/backfill/coinapi_backfill_execution_report.json",
                    help="manifest mode: where to write the reconciled execution report")
    ap.add_argument("--generated-utc", default=None,
                    help="override the plan/report timestamp (for deterministic tests)")
    args = ap.parse_args(argv)

    if args.manifest:
        if args.start or args.end or args.sample_mb:
            ap.error("--manifest mode is driven by the reviewed manifest's fill units; "
                     "--start/--end/--sample-mb do not apply")
        return run_manifest_mode(args, s3_factory=s3_factory)
    if not args.start or not args.end:
        ap.error("--start and --end are required (or use --manifest for the reviewed-manifest "
                 "backfill mode)")

    start = dt.date.fromisoformat(args.start)
    end = dt.date.fromisoformat(args.end)
    if end < start:
        raise SystemExit("--end is before --start")
    check_backfill_gate(start, end, sample_mb=args.sample_mb, allow_backfill=args.allow_backfill)

    api_key = load_env()["COINAPI_KEY"]
    os.makedirs(args.out, exist_ok=True)
    cleanup_tmp(args.out)
    s3 = ff.make_client(api_key)
    bucket = ff.discover_bucket(s3)

    print(f"Downloading {args.symbol_tag} {DATA_TYPE} {start}..{end}  -> {args.out}")
    print(f"  bucket={bucket} throttle={ff.REQ_PER_MIN}/min "
          f"{'[SAMPLE %dMB]' % args.sample_mb if args.sample_mb else ''}")

    counts = {"ok": 0, "skip": 0, "missing": 0}
    total_rows = 0
    for day in daterange(start, end):
        try:
            status, rows = process_day(s3, bucket, day, args)
        except ClientError as e:
            if is_quota_error(str(e)):
                print(QUOTA_HINT); sys.exit(2)
            raise
        counts[status] = counts.get(status, 0) + 1
        total_rows += rows
        if status == "skip":
            print(f"  {day.isoformat()}  skip (exists)")

    print(f"\nDone. ok={counts['ok']} skip={counts['skip']} missing={counts['missing']} "
          f"| rows written this run: {total_rows:,} | S3 requests: {ff.RL.calls}")


if __name__ == "__main__":
    # propagate manifest-mode exit codes (2/3/4/5/1) to the process; range mode returns None -> 0
    raise SystemExit(main())
