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

Usage:
  python ingest/download_coinapi.py --start 2025-01-01 --end 2025-01-31
  python ingest/download_coinapi.py --start 2025-01-01 --end 2025-01-03 --keep-raw
  python ingest/download_coinapi.py --start 2026-05-28 --end 2026-05-28 --sample-mb 8   # cheap smoke test
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import os
import sys
from io import StringIO

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(__file__))
from _common import load_env, is_quota_error, QUOTA_HINT          # noqa: E402
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


# ----------------------------------------------------------------------------- helpers
def daterange(start: dt.date, end: dt.date):
    d = start
    while d <= end:
        yield d
        d += dt.timedelta(days=1)


def find_file(s3, bucket, day_compact, exchange, symbol_tag):
    """Exact daily file (key, size) for exchange+symbol on a day, or None."""
    objs, _ = ff.list_prefix(s3, bucket, f"T-{DATA_TYPE}/D-{day_compact}/E-{exchange}/")
    match = f"+SC-{symbol_tag}+"
    hits = [o for o in objs if match in o["Key"]]
    if not hits:
        return None
    o = max(hits, key=lambda x: x["Size"])
    return o["Key"], o["Size"]


def stream_to_file(s3, bucket, key, dest) -> int:
    """Single throttled get_object streamed to disk (1 request). Returns bytes."""
    ff.RL.wait()
    body = s3.get_object(Bucket=bucket, Key=key)["Body"]
    total = 0
    with open(dest, "wb") as f:
        while True:
            buf = body.read(8 * 1024 * 1024)
            if not buf:
                break
            f.write(buf)
            total += len(buf)
    return total


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


def write_parquet(chunks, dest_tmp) -> tuple[int, int]:
    """Consume an iterator of DataFrames -> one Parquet file. Returns (rows, bytes)."""
    writer = pq.ParquetWriter(dest_tmp, SCHEMA, compression="zstd",
                              use_dictionary=["update_type"])
    rows = 0
    try:
        for df in chunks:
            writer.write_table(to_table(df, rows))
            rows += len(df)
    finally:
        writer.close()
    return rows, os.path.getsize(dest_tmp)


def manifest_append(out_root, rec):
    with open(os.path.join(out_root, "_manifest.jsonl"), "a") as f:
        f.write(json.dumps(rec) + "\n")


def now_iso():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# Backfill is GATED on the §5a recon-parity + reseed gates passing (docs/data.md §5a/§8). As of the
# 2025-06-01 live run the Lake-side gate FAILS (book_delta_v2 crosses ~67% intraday), so a paid bulk
# Coinbase pull must not proceed. A single overlap day (the parity pilot) and `--sample-mb` smoke
# tests are always allowed; a multi-day full pull requires an explicit `--allow-backfill` override so
# the documented "do not backfill" decision is actually enforced at the point of action, not just in
# prose. Pure + importable so it can be unit-tested without any vendor I/O.
BACKFILL_GATE_EXIT = 4  # mirrors run_coinbase_parity.py's small-int exit-code convention


def check_backfill_gate(start: dt.date, end: dt.date, *, sample_mb: int, allow_backfill: bool) -> None:
    """Block a multi-day full pull before the §5a gates pass. Prints the reason to stderr and
    raises SystemExit(4) (a string SystemExit would exit 1 and skip the int-code contract). A
    single day, a `--sample-mb` smoke, or an explicit `--allow-backfill` override returns cleanly."""
    n_days = (end - start).days + 1
    if n_days <= 1 or sample_mb or allow_backfill:
        return
    print(
        f"REFUSING multi-day backfill pull ({n_days} days {start}..{end}): the §5a Coinbase "
        "vendor-parity gate has NOT passed (Lake book_delta_v2 reseed pending — docs/data.md §5a). "
        "Bulk backfill is blocked until parity + reseed pass.\n"
        "  • For the parity pilot, pull ONE overlap day at a time: --start D --end D\n"
        "  • For a cheap smoke test, add --sample-mb 8\n"
        "  • To override once the gate passes (or for a deliberate, budgeted pull), pass "
        "--allow-backfill (ensure CoinAPI Spend Management is enabled, §8).",
        file=sys.stderr,
    )
    raise SystemExit(BACKFILL_GATE_EXIT)


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
            got = stream_to_file(s3, bucket, key, gz_path)
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


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Download CoinAPI flat files -> partitioned Parquet")
    ap.add_argument("--start", required=True, help="YYYY-MM-DD inclusive")
    ap.add_argument("--end", required=True, help="YYYY-MM-DD inclusive")
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
    args = ap.parse_args()

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
    main()
