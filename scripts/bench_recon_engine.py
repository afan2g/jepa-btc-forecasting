"""Local benchmark: Python vs native Lake `book_delta_v2` seed/reseed replay (docs/data.md §5a-Recon;
plan `docs/superpowers/plans/2026-07-01-native-recon-engine.md`).

Generates a DETERMINISTIC synthetic Lake-like `book_delta_v2` frame with a configurable live-book
WIDTH (`--levels`) and delete/`churn` rate, then times `recon.reseed.reconstruct_lake_l2_at_samples_seeded`
(Python oracle) against `recon.native.reconstruct_lake_l2_at_samples_seeded_native` (Rust). It also
asserts the two produce identical `(frame, meta)` on the benchmark fixture — a large-fixture
conformance check on top of the unit tests.

Why a wide book + churn matter (plan §"Benchmark"): the Python hot spot is `max(dict)/min(dict)`
best-bid/ask scans run per-delta (`update_crossed`) and per-sample (`emit`) — an O(N·L) cost. A
few-levels fixture makes Python look artificially cheap and would NOT catch a native port that still
scans all levels. Keep `--levels` in the thousands to exercise the real bottleneck.

No vendor credentials are needed — the fixture is synthetic, on a $0.01 tick grid (native price_scale
100, matching the COINBASE/BTC-USD contract).

    .venv/bin/python scripts/bench_recon_engine.py --rows 1000000 --samples 10000 --levels 10000 \\
        --churn 0.20 --engine both

Performance gates are recorded as local validation in the PR, NOT asserted in pytest.
"""
from __future__ import annotations

import argparse
import pathlib
import resource
import sys
import time

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon import native as _native                                  # noqa: E402
from recon.reseed import ReseedPolicy, book_snapshot, reconstruct_lake_l2_at_samples_seeded  # noqa: E402

PRICE_SCALE = 100        # $0.01 tick => integer cents; matches COINBASE/BTC-USD
MID_CENTS = 10_000_000   # $100,000.00 mid (arbitrary; keeps prices well away from 0)


def build_fixture(rows: int, samples: int, levels: int, churn: float, seed: int = 0):
    """Deterministic synthetic Lake `book_delta_v2` fixture with a `levels`-wide (per side) live book.

    Construction (non-crossing by design):
      * The price UNIVERSE is `levels` bid ticks below the mid and `levels` ask ticks above it.
      * A day-open snapshot seeds ALL `2*levels` levels, so the book starts full width.
      * Each delta picks a random universe tick and either RESIZES it (positive size) or, with prob
        `churn`, DELETES it (size 0). Since bids stay below the mid and asks above, the book never
        crosses; the innermost present level (the touch) drifts as inner levels are deleted/refilled,
        so Python's per-event max/min best-bid/ask scans are genuinely exercised over ~`levels`
        entries.

    Returns `(df, grid, seed_snapshots)`.
    """
    rng = np.random.default_rng(seed)
    bid_ticks = MID_CENTS - 1 - np.arange(levels, dtype=np.int64)      # [mid-1 .. mid-levels]
    ask_ticks = MID_CENTS + 1 + np.arange(levels, dtype=np.int64)      # [mid+1 .. mid+levels]

    is_bid = rng.integers(0, 2, size=rows).astype(bool)
    # pick a universe tick per row (bid rows -> a bid tick, ask rows -> an ask tick)
    bid_pick = bid_ticks[rng.integers(0, levels, size=rows)]
    ask_pick = ask_ticks[rng.integers(0, levels, size=rows)]
    ticks = np.where(is_bid, bid_pick, ask_pick)
    price = ticks / PRICE_SCALE

    is_delete = rng.random(size=rows) < churn
    size = np.where(is_delete, 0.0, rng.random(size=rows) * 10.0 + 0.001)

    # timestamps: strictly-positive, non-decreasing, with duplicates (avg ~5 rows/ns tick group).
    gaps = rng.integers(0, 2, size=rows).astype(np.int64)             # 0 => duplicate ts
    ts = 1 + np.cumsum(gaps)
    seq = rng.integers(0, max(1, rows // 8), size=rows).astype(np.int64)

    df = pd.DataFrame({
        "origin_time": ts,
        "sequence_number": seq,
        "side_is_bid": is_bid,
        "price": price,
        "size": size,
    })

    # even sample grid across the observed ts span
    grid = np.linspace(int(ts[0]), int(ts[-1]), num=samples, dtype=np.int64)
    grid = np.unique(grid).tolist()  # sorted ascending, deduped

    seed_bids = [(float(t) / PRICE_SCALE, 1.0) for t in bid_ticks]
    seed_asks = [(float(t) / PRICE_SCALE, 1.0) for t in ask_ticks]
    seed_snaps = [book_snapshot(0, bids=seed_bids, asks=seed_asks)]
    return df, grid, seed_snaps


def _rss_mb() -> float:
    """Peak RSS in MB (Linux ru_maxrss is in KB)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def _time_run(fn) -> tuple[object, dict, float]:
    t0 = time.perf_counter()
    frame, meta = fn()
    return frame, meta, time.perf_counter() - t0


def _report_engine(name: str, rows: int, n_samples: int, elapsed: float) -> None:
    print(f"  {name:<7} {elapsed:9.3f} s   {rows / elapsed:14,.0f} rows/s   "
          f"{n_samples / elapsed:12,.0f} samples/s")


def _frames_match(a: pd.DataFrame, b: pd.DataFrame) -> bool:
    try:
        pd.testing.assert_frame_equal(a, b, check_dtype=True)
        return True
    except AssertionError:
        return False


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Python vs native book_delta_v2 seed/reseed replay bench")
    ap.add_argument("--rows", type=int, default=300_000, help="synthetic delta rows (default 300k)")
    ap.add_argument("--samples", type=int, default=3_000, help="grid sample points (default 3000)")
    ap.add_argument("--levels", type=int, default=2_000,
                    help="live book WIDTH per side — keep in the thousands to exercise the Python "
                         "O(N·L) best-bid/ask scans that native fixes (default 2000)")
    ap.add_argument("--churn", type=float, default=0.20,
                    help="fraction of deltas that are deletes (size=0) (default 0.20)")
    ap.add_argument("--k", type=int, default=10, help="top-K levels emitted (default 10)")
    ap.add_argument("--engine", choices=("python", "native", "both"), default="both",
                    help="which engine(s) to time (default both)")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (default 0)")
    args = ap.parse_args(argv)

    want_native = args.engine in ("native", "both")
    if want_native and not _native.native_available():
        print(f"ERROR: --engine {args.engine} needs the recon_native extension, which is not "
              f"importable ({_native.native_import_error()!r}).\n"
              "  Build it: .venv/bin/maturin develop --release -m native/recon_native/Cargo.toml",
              file=sys.stderr)
        return 2

    print(f"Building fixture: rows={args.rows:,} samples={args.samples:,} levels={args.levels:,} "
          f"churn={args.churn} k={args.k} seed={args.seed} …")
    df, grid, seed_snaps = build_fixture(args.rows, args.samples, args.levels, args.churn, args.seed)
    policy = ReseedPolicy()
    print(f"  delta rows={len(df):,}  grid points={len(grid):,}  "
          f"deletes={int((df['size'] == 0.0).sum()):,}  RSS≈{_rss_mb():.0f} MB\n")

    print(f"  {'engine':<7} {'time':>11}   {'throughput':>14}")
    py_frame = py_meta = nat_frame = nat_meta = None

    if args.engine in ("python", "both"):
        py_frame, py_meta, py_t = _time_run(lambda: reconstruct_lake_l2_at_samples_seeded(
            df, grid, k=args.k, engine_time_col="origin_time", snapshots=seed_snaps, policy=policy,
            frame_out=True))
        _report_engine("python", args.rows, len(grid), py_t)

    if want_native:
        nat_frame, nat_meta, nat_t = _time_run(
            lambda: _native.reconstruct_lake_l2_at_samples_seeded_native(
                df, grid, k=args.k, engine_time_col="origin_time", snapshots=seed_snaps, policy=policy,
                frame_out=True, price_scale=PRICE_SCALE))
        _report_engine("native", args.rows, len(grid), nat_t)

    print(f"\n  peak RSS ≈ {_rss_mb():.0f} MB")

    if args.engine == "both":
        speedup = py_t / nat_t if nat_t > 0 else float("inf")
        matched = _frames_match(py_frame, nat_frame) and (py_meta == nat_meta)
        print(f"  native speedup: {speedup:.1f}x   (python {py_t:.3f}s -> native {nat_t:.3f}s)")
        print(f"  native output matches Python on the benchmark fixture: {matched}")
        if not matched:
            print("  ERROR: native/python MISMATCH on the benchmark fixture — investigate before "
                  "trusting the speedup.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
