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

Lake `book_delta_v2` is reconstructed with the §5a-Recon SEED/RESEED policy by default: it seeds
from the validated Lake `book` snapshot product and reseeds when the book stays crossed (the cure
for the 2025-06-01 ~67% intraday-crossing failure). The report's `lake_reseed` block carries the
before(cold)/after(reseed) crossed rate. `--no-reseed` keeps the seed but disables intraday repair
(A/B); `--no-lake-seed` is the pre-§5a-Recon pure cold-start.

Usage:
  .venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --k 10   # seed+reseed; size_policy=decrement
  .venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --k 10 --no-reseed       # seed-only A/B
  .venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --k 10 --no-lake-seed    # cold-start A/B
  .venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --k 10 --size-policy absolute  # CoinAPI A/B
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
from recon.parity import compare_topk, frame_quality, lake_warmup_cutoff  # noqa: E402
from recon.reconstruct import (                                      # noqa: E402
    reconstruct_book_at_samples, reconstruct_lake_l2_at_samples,
)
from recon.reseed import (                                           # noqa: E402
    ReseedPolicy, reconstruct_lake_l2_at_samples_seeded, snapshots_from_lake_book_df,
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


def load_lake_book_snapshots(sess, day: dt.date, exchange: str, symbol: str, *,
                             max_levels: int = 20, stride_ms: int = 1000):
    """Load the Crypto Lake `book` (20-level snapshot) product for `day` as validated-candidate
    reseed sources (docs/data.md §5a-Recon), thinned to ~one snapshot per `stride_ms`.

    ONE day-aligned `load_data` call: lakeapi raises NoFilesFound on a SUB-day window for the `book`
    table (verified), and the Coinbase `book` product is small (~275k rows/day ≈ 180 MB, 83 cols),
    so a single-day load is memory-safe — reseeds only need a sparse validated set, not every row.
    The `book` product is NOT used for features; it is a reseed source only, and only on days where
    it is verified uncrossed (2025-06-01: 0% crossed; 2026-04-01: 31.75% crossed → rejected by
    `classify_snapshot`). Projects the RAW level columns (`bid_i_price/size`, `ask_i_price/size`).

    Single-clock invariant: snapshots, deltas, and the grid share ONE engine clock (origin_time — the
    grid is origin-time midnight). Projecting only `timestamp` resolves to origin_time when populated
    (100% for Coinbase, docs §5) or RAISES — the caller catches it and falls back to cold-start, so a
    clock split can never silently corrupt the merge."""
    import lakeapi  # local import: keep module import side-effect-free / vendor-free
    cols = ["timestamp"]
    for i in range(max_levels):
        cols += [f"bid_{i}_price", f"bid_{i}_size", f"ask_{i}_price", f"ask_{i}_size"]
    start = dt.datetime.combine(day, dt.time())
    end = start + dt.timedelta(days=1)
    df = lakeapi.load_data(table="book", start=start, end=end, symbols=[symbol],
                           exchanges=[exchange], columns=cols, boto3_session=sess,
                           drop_partition_cols=True)
    if not len(df):
        return []
    etc = shared_engine_time_col(df)
    return snapshots_from_lake_book_df(df, engine_time_col=etc, max_levels=max_levels,
                                       stride_ns=stride_ms * NS_PER_MS)


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
                    grid_ms: int = 1000, size_policy: str = "decrement",
                    on_unknown: str = "count", horizons_s=DEFAULT_HORIZONS_S,
                    band_bps: float = 0.0, n_spikes: int = 25, gate_warmup: bool = True,
                    warmup_consecutive: int = 3, warmup_min_levels: int = 1,
                    lake_book_snapshots=None, reseed: bool = True,
                    reseed_after_crossed_s: float = 2.0, seed_min_levels: int = 5,
                    max_spread_frac: float | None = None, exclude_lake_crossed: bool = True):
    """Reconstruct both vendors onto one grid and compare. Pure w.r.t. its inputs (a Lake
    delta DataFrame, an iterable of CoinAPI chunks, and optional pre-loaded Lake `book`
    snapshots) — no vendor I/O — so the skip-guarded integration test can drive the exact
    production path from local fixtures.

    Lake `book_delta_v2` is a mid-stream incremental feed with no per-day snapshot. Cold-started
    it strands levels and crosses (~67% of 2025-06-01 — docs/data.md §5a). When
    `lake_book_snapshots` is supplied, the Lake side is reconstructed with the §5a-Recon
    seed/reseed policy: seed from the first validated `book` snapshot, then reseed whenever the
    book stays crossed past `reseed_after_crossed_s`. The cold-start reconstruction is ALSO run
    (snapshots=None, byte-identical code path) to report the before/after crossed rate — the A/B
    that shows the reseed effect. `reseed=False` keeps the seed but disables intraday repair (the
    seed-once A/B arm). Residual crossed Lake samples (awaiting a reseed) are EXCLUDED from the
    parity comparison when `exclude_lake_crossed` — a crossed mid is not a real vendor mid — and
    counted transparently; the per-vendor `lake_quality` always reports the FULL-grid crossed rate.

    When `gate_warmup`, pre-seed samples (before the Lake book establishes) are also excluded.

    Returns `(report, lake_frame, coinapi_frame)`."""
    grid = build_grid(day, grid_ms)
    grid_s = grid_ms / 1000.0
    have_lake = lake_delta_df is not None and len(lake_delta_df) > 0
    have_snaps = have_lake and bool(lake_book_snapshots)

    reseed_meta = None
    cold_meta = None
    if not have_lake:
        lake = reconstruct_book_at_samples([], grid, k=k)  # empty book → all-missing frame
        engine_col = None
    else:
        engine_col = shared_engine_time_col(lake_delta_df)
        if have_snaps:
            policy = ReseedPolicy(enabled=reseed, min_levels_per_side=seed_min_levels,
                                  reseed_after_crossed_s=reseed_after_crossed_s,
                                  max_spread_frac=max_spread_frac)
            lake, reseed_meta = reconstruct_lake_l2_at_samples_seeded(
                lake_delta_df, grid, k=k, engine_time_col=engine_col,
                snapshots=lake_book_snapshots, policy=policy)
            # A/B "before": the SAME reconstruction cold-started (no seed/reseed). Metrics-only
            # (frame_out=False) — we need just its crossed rate, so no second 86,400-row frame is
            # built/discarded on the multi-GB day (the per-delta replay is intrinsic to the A/B).
            _, cold_meta = reconstruct_lake_l2_at_samples_seeded(
                lake_delta_df, grid, k=k, engine_time_col=engine_col, snapshots=None,
                frame_out=False)
        else:
            lake = reconstruct_lake_l2_at_samples(lake_delta_df, grid, k=k, engine_time_col=engine_col)

    capi, capi_q = reconstruct_coinapi_l2_at_samples(
        coinapi_chunks, k=k, day=day, sample_ts=grid,
        size_policy=size_policy, on_unknown=on_unknown,
    )
    lake_q = frame_quality(lake, source_rows=(0 if not have_lake else len(lake_delta_df)))

    cutoff = (lake_warmup_cutoff(lake, min_consecutive=warmup_consecutive,
                                 min_levels_per_side=warmup_min_levels) if gate_warmup else None)
    # A validated seed defines the true "book established" time. Samples before seed_ts are pre-seed
    # COLD-STARTED state (§5a-Recon warm-up) — and cold-started deltas can look two-sided/uncrossed
    # before the seed lands, so lake_warmup_cutoff alone could place the cutoff earlier and let the
    # gate compare unseeded Lake state. Clamp the cutoff to the accepted seed (only when warm-up
    # gating is on; --no-warmup-gate deliberately compares the full grid incl. cold-start).
    if gate_warmup and reseed_meta and reseed_meta.get("seed_accepted") and \
            reseed_meta.get("seed_ts") is not None:
        st = int(reseed_meta["seed_ts"])
        cutoff = st if cutoff is None else max(int(cutoff), st)
    excluded = sum(1 for t in grid if cutoff is not None and t < cutoff)

    # Exclude residual crossed Lake samples (awaiting a reseed) from the comparison — a crossed mid
    # is not a real vendor mid — but keep n_grid_full = the TRUE full grid (compare_topk drops them
    # via exclude_ts, not by pre-filtering the frame, so the honest grid size is never undercounted).
    # ONLY when a valid seed was accepted AND reseed is active: otherwise the crossed samples are a
    # genuine reconstruction FAILURE (a rejected seed on a crossed-`book` day, or the seed-only A/B
    # arm) and must surface as crossed in the gate, not be masked into a clean-looking parity.
    seed_active = bool(reseed_meta and reseed_meta.get("seed_accepted") and reseed)
    excluded_crossed = (set(reseed_meta["crossed_sample_ts"])
                        if (exclude_lake_crossed and seed_active) else set())
    parity = compare_topk(lake, capi, k=k, grid_s=grid_s, horizons_s=horizons_s,
                          band_bps=band_bps, n_spikes=n_spikes, since_ts=cutoff,
                          exclude_ts=excluded_crossed)

    crossed_before = (cold_meta["crossed_rate"] if cold_meta is not None else lake_q["crossed_rate"])
    report = {
        "meta": {
            "day": day.isoformat(), "k": int(k), "grid_ms": int(grid_ms),
            "grid_points": len(grid), "size_policy": size_policy, "on_unknown": on_unknown,
            "horizons_s": list(horizons_s), "band_bps": float(band_bps),
            "lake_engine_time_col": engine_col,
            "lake_delta_rows": (0 if not have_lake else int(len(lake_delta_df))),
            "coinapi_event_rows": int(capi_q.get("total_rows", 0)),
            "note": "PILOT — synthetic-validated tooling; live measured results only when run "
                    "against real vendor data (docs/data.md §5a hard gate #1).",
        },
        "warmup": {
            "gated": bool(gate_warmup),
            "established": cutoff is not None,
            "cutoff_ts": (int(cutoff) if cutoff is not None else None),
            "min_consecutive": int(warmup_consecutive),
            "min_levels_per_side": int(warmup_min_levels),
            "excluded_samples": int(excluded),
            "excluded_fraction": (float(excluded / len(grid)) if grid else 0.0),
            "note": "Lake book_delta_v2 cold-starts with no per-day snapshot (docs/data.md "
                    "§5a-Recon); pre-seed samples are excluded from the parity gate. With a "
                    "validated Lake `book` seed the book establishes at day-open so this window "
                    "collapses. established=False ⇒ Lake never seeded (e.g. a gap day) → parity "
                    "is on the full grid and reads as fully missing.",
        },
        "lake_reseed": {
            "applied": bool(have_snaps),
            "reseed_enabled": bool(reseed),
            "snapshot_candidates": (int(len(lake_book_snapshots)) if have_snaps else 0),
            "seed_accepted": (bool(reseed_meta["seed_accepted"]) if reseed_meta else False),
            "seed_ts": (reseed_meta["seed_ts"] if reseed_meta else None),
            "seed_reason": (reseed_meta["seed_reason"] if reseed_meta else "no_snapshots"),
            "reseed_count": (int(reseed_meta["reseed_count"]) if reseed_meta else 0),
            "reseed_ts": (reseed_meta["reseed_ts"][:100] if reseed_meta else []),
            "reseed_blocked_invalid_snapshot": (
                int(reseed_meta["reseed_blocked_invalid_snapshot"]) if reseed_meta else 0),
            "snapshot_reason_codes": (reseed_meta["snapshot_reason_codes"] if reseed_meta else {}),
            "crossed_rate_before": float(crossed_before),
            "crossed_rate_after": float(lake_q["crossed_rate"]),
            "crossed_samples_after": int(lake_q["crossed_samples"]),
            "crossed_duration_s_after": (
                float(reseed_meta["crossed_duration_s"]) if reseed_meta else None),
            "missing_book_fraction_after": float(lake_q["missing_book_fraction"]),
            "thin_depth_fraction_after": (
                float(reseed_meta["thin_depth_fraction"]) if reseed_meta else None),
            "excluded_crossed_samples": int(len(excluded_crossed)),
            "policy": (reseed_meta["policy"] if reseed_meta else None),
            "note": "Seed from the validated Lake `book` snapshot, reseed on sustained crossing "
                    "(docs/data.md §5a-Recon). crossed_rate_before = cold-start (no seed/reseed); "
                    "crossed_rate_after = this run. seq is NOT used as a gap detector (it is "
                    "per-event, duplicated ~91% of rows); crossing is the reseed trigger.",
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
    lr = report.get("lake_reseed", {})
    if lr.get("applied"):
        seed = (f"OK@{lr['seed_ts']}" if lr.get("seed_accepted")
                else f"REJECTED({lr.get('seed_reason')})")
        print(f"  Lake seed/reseed  : seed {seed} | candidates {lr.get('snapshot_candidates', 0):,}"
              f" | reseeds {lr.get('reseed_count', 0)}"
              f" (blocked {lr.get('reseed_blocked_invalid_snapshot', 0)})")
        print(f"  Lake crossed A/B  : before(cold) {lr.get('crossed_rate_before', 0):.4%} → "
              f"after(reseed) {lr.get('crossed_rate_after', 0):.4%} | "
              f"excluded {lr.get('excluded_crossed_samples', 0):,} crossed samples")
    w = report.get("warmup", {})
    if w.get("gated"):
        cut = "established" if w.get("established") else "NEVER established (gap day?)"
        print(f"  Lake warm-up gate : {cut}; excluded {w.get('excluded_samples', 0):,} "
              f"({w.get('excluded_fraction', 0):.4%}) pre-seed samples → parity on "
              f"{p.get('n_grid')}/{p.get('n_grid_full')} grid pts")
    print(f"  crossed-book rate : lake {report['lake_quality']['crossed_rate']:.4%} | "
          f"capi {report['coinapi_quality'].get('crossed_rate', 0):.4%}  (full grid)")
    print(f"  missing-book frac : lake {report['lake_quality']['missing_book_fraction']:.4%} | "
          f"capi {report['coinapi_quality'].get('missing_book_fraction', 0):.4%}")
    if md.get("median") is not None:
        print(f"  |Δmid| $          : median {md['median']:.4f} | p95 {md['p95']:.4f} | "
              f"p99 {md['p99']:.4f} | max {md['max']:.2f} | corr {md.get('corr')}")
    print(f"  |Δmid| spikes     : " + " ".join(
        f"{kk}:{vv}" for kk, vv in p.get("spike_counts", {}).items()))
    pl = p.get("per_level", {})
    if pl:  # deepest level's both-present coverage — flags thin/one-sided top-K depth
        deep = str(m["k"] - 1)
        cov = pl.get(deep, {}).get("coverage", {})
        bc, ac = cov.get("bid", {}), cov.get("ask", {})
        print(f"  L{deep} coverage     : bid both {bc.get('both_fraction', 0):.2%} "
              f"(only_lake {bc.get('only_lake', 0)}/only_capi {bc.get('only_capi', 0)}) | "
              f"ask both {ac.get('both_fraction', 0):.2%}")
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
    ap.add_argument("--size-policy", choices=("absolute", "decrement"), default="decrement",
                    help="CoinAPI SUB/MATCH size convention. Default 'decrement': the 2025-06-01 "
                         "live gate proved MATCH.entry_sx is the traded amount for Coinbase "
                         "limitbook_full, so 'absolute' leaves filled orders as stale residue and "
                         "crosses the book ~100%% (docs/data.md §5a). 'absolute' kept as the A/B "
                         "alternative for other venues (see recon/coinapi.py).")
    ap.add_argument("--on-unknown", choices=("count", "raise"), default="count",
                    help="policy for unknown CoinAPI update_type (default count+skip)")
    ap.add_argument("--band-bps", type=float, default=0.0, help="no-trade band for label agreement (bps)")
    ap.add_argument("--horizons-s", default="2,10,60", help="label horizons in seconds (csv)")
    ap.add_argument("--no-warmup-gate", action="store_true",
                    help="compare the FULL grid incl. Lake book_delta_v2 cold-start warm-up "
                         "(default: exclude warm-up; docs/data.md §5a-Recon)")
    ap.add_argument("--warmup-consecutive", type=int, default=3,
                    help="consecutive seeded samples required before the Lake book is trusted")
    ap.add_argument("--warmup-min-levels", type=int, default=1,
                    help="min levels per side for the Lake seed-established gate")
    ap.add_argument("--no-lake-seed", action="store_true",
                    help="do NOT load the Lake `book` snapshot product; pure cold-start "
                         "reconstruction (the pre-§5a-Recon behavior, for A/B). Default: seed.")
    ap.add_argument("--no-reseed", action="store_true",
                    help="seed once at day-open but DISABLE intraday reseed-on-crossing "
                         "(the seed-only A/B arm; docs/data.md §5a-Recon)")
    ap.add_argument("--reseed-after-crossed-s", type=float, default=2.0,
                    help="reseed only once the Lake book has been crossed continuously ≥ this many "
                         "seconds (default 2.0; a transient one-tick cross does not force a reseed)")
    ap.add_argument("--seed-min-levels", type=int, default=5,
                    help="min levels per side for a Lake `book` snapshot to be a VALID seed/reseed "
                         "source (rejects thin/broken snapshots; default 5)")
    ap.add_argument("--seed-max-spread-frac", type=float, default=None,
                    help="optional sane-spread guard for seed validation (spread/mid; off by default)")
    ap.add_argument("--book-stride-ms", type=int, default=1000,
                    help="thin the Lake `book` product to ~one seed candidate per N ms (default 1000)")
    ap.add_argument("--book-max-levels", type=int, default=20,
                    help="Lake `book` snapshot depth to load/seed from (default 20, the product depth)")
    ap.add_argument("--no-exclude-crossed", action="store_true",
                    help="keep residual crossed Lake samples in the parity comparison instead of "
                         "excluding them (default: exclude crossed Lake samples, report separately)")
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

    snaps = None
    if not args.no_lake_seed and len(lake_df):
        print(f"Loading Crypto Lake {args.symbol} `book` snapshots for seeding/reseeding "
              f"(stride {args.book_stride_ms}ms, {args.book_max_levels} levels) …")
        try:
            snaps = load_lake_book_snapshots(
                sess, day, args.exchange, args.symbol,
                max_levels=args.book_max_levels, stride_ms=args.book_stride_ms)
            print(f"  Lake book seed candidates: {len(snaps):,}")
        except Exception as e:  # noqa: BLE001 — graceful fallback to cold-start
            print(f"  WARNING: could not load Lake `book` snapshots ({e!r}); falling back to "
                  "cold-start (no seed/reseed). The Lake side will read as crossed (docs/data.md "
                  "§5a). Re-run with the `book` product available to seed.", file=sys.stderr)
            snaps = None

    print(f"Replaying CoinAPI L3 {capi_path} (streaming, {args.chunk_rows:,} rows/chunk) …")
    report, lake, capi = run_parity_core(
        lake_df, iter_coinapi_chunks(capi_path, args.chunk_rows), day=day, k=args.k,
        grid_ms=args.grid_ms, size_policy=args.size_policy, on_unknown=args.on_unknown,
        horizons_s=horizons, band_bps=args.band_bps, gate_warmup=not args.no_warmup_gate,
        warmup_consecutive=args.warmup_consecutive, warmup_min_levels=args.warmup_min_levels,
        lake_book_snapshots=snaps, reseed=not args.no_reseed,
        reseed_after_crossed_s=args.reseed_after_crossed_s, seed_min_levels=args.seed_min_levels,
        max_spread_frac=args.seed_max_spread_frac, exclude_lake_crossed=not args.no_exclude_crossed,
    )
    paths = write_report(report, lake, capi, args.out_dir, day, args.k, args.dump_grid)
    print_summary(report, paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
