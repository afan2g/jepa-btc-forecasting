"""One-day Coinbase vendor-parity gate (docs/data.md §5a hard gate #1).

Reconstructs the SAME overlap day two ways and compares them at top-K L2:
  1. Crypto Lake Coinbase `book_delta_v2` → reconstructed top-K L2.
  2. CoinAPI Coinbase `limitbook_full` (L3) → replayed/aggregated top-K L2.

It aligns both on an exchange-time grid and reports bid/ask/mid differences, per-level
price/size deltas, per-vendor crossed-book and missing-book rates, the |Δmid| spike
distribution (the known rare large-divergence concern — characterized, not assumed to wash
out), and directional label agreement at the project horizons. A small JSON report (+ a
top-spikes CSV) is written under data/reports/.

This is a bounded PILOT, NOT a backfill. It pulls ONE Crypto Lake day (subscription is flat,
unlimited) and reads a LOCAL CoinAPI parquet for the day — it never triggers a CoinAPI bulk
download. If the CoinAPI parquet is missing it prints the exact one-day download command and
exits 3.

Usage:
  .venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --k 10
  .venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --k 10 --size-policy decrement
  .venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --dump-grid   # full aligned grid CSV

Credentials: Crypto Lake AWS keys in .env (Lake-only — COINAPI_KEY is NOT required to read a
local parquet). Mirrors scripts/verify_book_delta_v2.py::lake_session credential semantics.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import sys

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.coinapi import reconstruct_coinapi_l2_at_samples          # noqa: E402
from recon.ingest import shared_engine_time_col                      # noqa: E402
from recon.parity import compare_topk, frame_quality                 # noqa: E402
from recon.reconstruct import (                                      # noqa: E402
    reconstruct_book_at_samples, reconstruct_lake_l2_at_samples,
)

NS_PER_MS = 1_000_000
DAY_MS = 86_400_000
DEFAULT_HORIZONS_S = (2, 10, 60)


# ----------------------------------------------------------------------------- credentials
def lake_session():
    """Crypto Lake boto3 session from .env subscriber keys (NOT the personal ~/.aws default,
    which auths into the wrong account → AccessDenied). Lake-only: does not require
    COINAPI_KEY (unlike ingest._common.load_env). Mirrors scripts/verify_book_delta_v2.py."""
    import boto3  # local import: importing this module must not require boto3/lakeapi
    env: dict = {}
    envpath = ROOT / ".env"
    if envpath.exists():
        for line in envpath.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, val = line.split("=", 1)
                env[key.strip()] = val.strip().strip('"').strip("'")
    env = {**env, **os.environ}
    try:
        return boto3.Session(
            aws_access_key_id=env["aws_access_key_id"],
            aws_secret_access_key=env["aws_secret_access_key"],
            region_name=env.get("region", "eu-west-1"),
        )
    except KeyError as e:
        raise SystemExit(
            f"Crypto Lake AWS key {e} not found in .env or environment "
            "(need aws_access_key_id and aws_secret_access_key)."
        ) from None


# ----------------------------------------------------------------------------- data loading
def load_lake_book_delta_v2(sess, day: dt.date, exchange: str, symbol: str) -> pd.DataFrame:
    """Load one day of Crypto Lake `book_delta_v2`. Projects the RAW parquet column names
    (docs/data.md §4.1) so `lakeapi` renames timestamp→origin_time / receipt_timestamp→
    received_time and the recon ingest seam resolves them."""
    import lakeapi  # local import: keep module import side-effect-free / vendor-free
    start = dt.datetime.combine(day, dt.time())
    end = start + dt.timedelta(days=1)
    return lakeapi.load_data(
        table="book_delta_v2", start=start, end=end, symbols=[symbol], exchanges=[exchange],
        columns=["timestamp", "receipt_timestamp", "sequence_number", "side_is_bid",
                 "price", "size"],
        boto3_session=sess, drop_partition_cols=True,
    )


def coinapi_parquet_path(out_root: str, day: dt.date, exchange: str, symbol: str) -> str:
    return os.path.join(out_root, "limitbook_full", f"exchange={exchange}",
                        f"symbol={symbol}", f"dt={day.isoformat()}", "data.parquet")


def iter_coinapi_chunks(path: str, chunk_rows: int):
    """Stream a CoinAPI parquet day as seq-ordered DataFrame chunks (Parquet row-groups are
    written in seq order by download_coinapi.py), so a multi-GB day is never fully resident."""
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=chunk_rows):
        yield batch.to_pandas()


# ----------------------------------------------------------------------------- core
def build_grid(day: dt.date, grid_ms: int) -> list[int]:
    """Exchange-time sample grid (int ns) spanning the partition day at `grid_ms` spacing."""
    day_open = int(pd.Timestamp(day).value)
    step_ns = grid_ms * NS_PER_MS
    n = DAY_MS // grid_ms
    return [day_open + i * step_ns for i in range(n)]


def run_parity_core(lake_delta_df: pd.DataFrame, coinapi_chunks, *, day: dt.date, k: int,
                    grid_ms: int = 1000, size_policy: str = "absolute",
                    on_unknown: str = "count", horizons_s=DEFAULT_HORIZONS_S,
                    band_bps: float = 0.0, n_spikes: int = 25):
    """Reconstruct both vendors onto one grid and compare. Pure w.r.t. its inputs (a Lake
    delta DataFrame and an iterable of CoinAPI chunks) — no vendor I/O — so the skip-guarded
    integration test can drive the exact production path from local fixtures.

    Returns `(report, lake_frame, coinapi_frame)`."""
    grid = build_grid(day, grid_ms)
    grid_s = grid_ms / 1000.0

    if lake_delta_df is None or len(lake_delta_df) == 0:
        lake = reconstruct_book_at_samples([], grid, k=k)  # empty book → all-missing frame
        engine_col = None
    else:
        engine_col = shared_engine_time_col(lake_delta_df)
        lake = reconstruct_lake_l2_at_samples(lake_delta_df, grid, k=k, engine_time_col=engine_col)

    capi, capi_q = reconstruct_coinapi_l2_at_samples(
        coinapi_chunks, k=k, day=day, sample_ts=grid,
        size_policy=size_policy, on_unknown=on_unknown,
    )
    lake_q = frame_quality(lake, source_rows=(0 if lake_delta_df is None else len(lake_delta_df)))
    parity = compare_topk(lake, capi, k=k, grid_s=grid_s, horizons_s=horizons_s,
                          band_bps=band_bps, n_spikes=n_spikes)

    report = {
        "meta": {
            "day": day.isoformat(), "k": int(k), "grid_ms": int(grid_ms),
            "grid_points": len(grid), "size_policy": size_policy, "on_unknown": on_unknown,
            "horizons_s": list(horizons_s), "band_bps": float(band_bps),
            "lake_engine_time_col": engine_col,
            "lake_delta_rows": (0 if lake_delta_df is None else int(len(lake_delta_df))),
            "coinapi_event_rows": int(capi_q.get("total_rows", 0)),
            "note": "PILOT — synthetic-validated tooling; live measured results only when run "
                    "against real vendor data (docs/data.md §5a hard gate #1).",
        },
        "lake_quality": lake_q,
        "coinapi_quality": capi_q,
        "parity": parity,
    }
    return report, lake, capi


# ----------------------------------------------------------------------------- reporting
def _json_safe(obj):
    """Recursively coerce to strict-JSON-valid types: non-finite floats → None, numpy scalars
    → python scalars, so the artifact passes `jq empty` (AGENTS.md testing rule)."""
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if hasattr(obj, "item"):  # numpy scalar
        v = obj.item()
        return _json_safe(v) if isinstance(v, float) else v
    return obj


def write_report(report: dict, lake: pd.DataFrame, capi: pd.DataFrame, out_dir: str,
                 day: dt.date, k: int, dump_grid: bool) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    stem = f"parity_coinbase_{day.isoformat()}_k{k}"
    json_path = os.path.join(out_dir, f"{stem}.json")
    with open(json_path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
        f.write("\n")

    spikes = report["parity"].get("top_spikes", [])
    spikes_path = os.path.join(out_dir, f"{stem}_spikes.csv")
    pd.DataFrame(spikes).to_csv(spikes_path, index=False)

    paths = {"json": json_path, "spikes_csv": spikes_path}
    if dump_grid:  # full aligned mid/top-of-book grid (large; data/reports is git-ignored)
        grid_path = os.path.join(out_dir, f"{stem}_grid.csv")
        merged = lake.set_index("sample_ts")[["mid", "bid_0_price", "ask_0_price"]].add_suffix("_lake").join(
            capi.set_index("sample_ts")[["mid", "bid_0_price", "ask_0_price"]].add_suffix("_capi"))
        merged.to_csv(grid_path)
        paths["grid_csv"] = grid_path
    return paths


def print_summary(report: dict, paths: dict) -> None:
    m, p = report["meta"], report["parity"]
    md = p.get("mid_diff", {})
    print("\n" + "=" * 74)
    print(f"  COINBASE VENDOR PARITY — {m['day']}  (k={m['k']}, grid={m['grid_ms']}ms, "
          f"size_policy={m['size_policy']})")
    print("=" * 74)
    print(f"  Lake delta rows: {m['lake_delta_rows']:,} | CoinAPI event rows: "
          f"{m['coinapi_event_rows']:,}")
    print(f"  crossed-book rate : lake {report['lake_quality']['crossed_rate']:.4%} | "
          f"capi {report['coinapi_quality'].get('crossed_rate', 0):.4%}")
    print(f"  missing-book frac : lake {report['lake_quality']['missing_book_fraction']:.4%} | "
          f"capi {report['coinapi_quality'].get('missing_book_fraction', 0):.4%}")
    if md.get("median") is not None:
        print(f"  |Δmid| $          : median {md['median']:.4f} | p95 {md['p95']:.4f} | "
              f"p99 {md['p99']:.4f} | max {md['max']:.2f} | corr {md.get('corr')}")
    print(f"  |Δmid| spikes     : " + " ".join(
        f"{kk}:{vv}" for kk, vv in p.get("spike_counts", {}).items()))
    for h, la in p.get("label_agreement", {}).items():
        ag = la.get("agreement")
        print(f"  label agree {h:>3}s  : {('%.4f' % ag) if ag is not None else 'n/a'}  "
              f"(n={la.get('n')})")
    print(f"\n  wrote {paths['json']}")
    print(f"        {paths['spikes_csv']}" + (f"\n        {paths['grid_csv']}"
          if "grid_csv" in paths else ""))


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="One-day Coinbase Lake↔CoinAPI top-K L2 parity gate (pilot, not a backfill)")
    ap.add_argument("--day", default="2025-06-01", help="overlap day YYYY-MM-DD (default 2025-06-01)")
    ap.add_argument("--k", type=int, default=10, help="top-K levels (default 10)")
    ap.add_argument("--grid-ms", type=int, default=1000, help="sample-grid spacing ms (default 1000)")
    ap.add_argument("--size-policy", choices=("absolute", "decrement"), default="absolute",
                    help="CoinAPI SUB/MATCH size convention (default absolute; see recon/coinapi.py)")
    ap.add_argument("--on-unknown", choices=("count", "raise"), default="count",
                    help="policy for unknown CoinAPI update_type (default count+skip)")
    ap.add_argument("--band-bps", type=float, default=0.0, help="no-trade band for label agreement (bps)")
    ap.add_argument("--horizons-s", default="2,10,60", help="label horizons in seconds (csv)")
    ap.add_argument("--exchange", default="COINBASE", help="Crypto Lake / partition exchange")
    ap.add_argument("--symbol", default="BTC-USD", help="Crypto Lake / partition symbol")
    ap.add_argument("--coinapi-root", default="data/raw", help="root of the CoinAPI parquet tree")
    ap.add_argument("--chunk-rows", type=int, default=2_000_000, help="CoinAPI parquet stream batch size")
    ap.add_argument("--out-dir", default="data/reports", help="report output dir (git-ignored)")
    ap.add_argument("--dump-grid", action="store_true", help="also write the full aligned grid CSV")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    day = dt.date.fromisoformat(args.day)
    horizons = tuple(int(x) for x in str(args.horizons_s).split(",") if x.strip())

    capi_path = coinapi_parquet_path(args.coinapi_root, day, args.exchange, args.symbol)
    if not os.path.exists(capi_path):
        print(f"ERROR: CoinAPI parquet for {day} not found at:\n  {capi_path}\n\n"
              "This pilot does NOT download it for you. Produce ONE overlap day first:\n"
              f"  .venv/bin/python ingest/download_coinapi.py --start {day} --end {day}\n"
              "(cheap smoke first: add --sample-mb 8). Enable CoinAPI Spend Management before "
              "any real pull — docs/data.md §2.2/§8.", file=sys.stderr)
        return 3

    print(f"Loading Crypto Lake {args.exchange} {args.symbol} book_delta_v2 for {day} …")
    sess = lake_session()
    lake_df = load_lake_book_delta_v2(sess, day, args.exchange, args.symbol)
    print(f"  Lake book_delta_v2 rows: {len(lake_df):,}")
    if len(lake_df) == 0:
        print(f"  WARNING: Crypto Lake has no book_delta_v2 for {day} (a gap day?) — the parity "
              "report will show the Lake book as fully missing.", file=sys.stderr)

    print(f"Replaying CoinAPI L3 {capi_path} (streaming, {args.chunk_rows:,} rows/chunk) …")
    report, lake, capi = run_parity_core(
        lake_df, iter_coinapi_chunks(capi_path, args.chunk_rows), day=day, k=args.k,
        grid_ms=args.grid_ms, size_policy=args.size_policy, on_unknown=args.on_unknown,
        horizons_s=horizons, band_bps=args.band_bps,
    )
    paths = write_report(report, lake, capi, args.out_dir, day, args.k, args.dump_grid)
    print_summary(report, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
