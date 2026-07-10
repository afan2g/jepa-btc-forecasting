"""#54 CoinAPI snapshot-only seeding EXPERIMENT runner (offline; never downloads).

Runs the experiments/snapshot_seed.py arm set for ONE fixture day against data that is
ALREADY on disk:
  * the CoinAPI `limitbook_full` parquet under --coinapi-root (exit 3 if absent — this
    runner never touches the vendor);
  * the Crypto Lake day bodies inside an existing lakeapi joblib cache
    (--lake-cache-root, read-only; FileNotFoundError if not cached).

The day's full CoinAPI L3→L2 reference frame is expensive (a full-day Python L3 replay)
so it is cached as parquet under <out-dir>/cache/ keyed by the source file's sha256.
Reports land in --out-dir (git-ignored) as snapshot_seed_<day>[_<variant>].json plus an
arm-summary CSV.

This is EXPERIMENT tooling for a cost-reduction question. It does not alter the
canonical #33 manifest, the partial-day fill policy, or the #53 executor; a GO verdict
requires a separate reviewed policy PR. Wrap real-day runs in the workstation compute
lock:

  flock -w 14400 /tmp/jepa-expensive-compute.lock \
    .venv/bin/python scripts/run_snapshot_seed_experiment.py --day 2025-06-01 ...
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon import native as _native                                    # noqa: E402
from recon.coinapi import reconstruct_coinapi_l2_at_samples           # noqa: E402
from recon.ingest import shared_engine_time_col                       # noqa: E402
from recon.reseed import snapshots_from_lake_book_df                  # noqa: E402
from scripts.run_coinbase_parity import (                             # noqa: E402
    build_grid, coinapi_parquet_path, iter_coinapi_chunks)

NATIVE_UNAVAILABLE_EXIT = 6
DEFAULT_ARMS = ("cold_control,lake_book_control,coinapi_day_open_L20,"
                "coinapi_day_open_full,coinapi_stream_L5,coinapi_stream_L10,"
                "coinapi_stream_L20,coinapi_stream_L50,coinapi_on_demand_L20")


def arm_specs_from_names(names) -> list[dict]:
    """Parse arm names (the preregistration_54.json vocabulary) into arm specs."""
    specs = []
    for name in names:
        if name == "cold_control":
            specs.append({"name": name, "kind": "cold"})
        elif name == "lake_book_control":
            specs.append({"name": name, "kind": "lake_book"})
        elif name == "coinapi_day_open_full":
            specs.append({"name": name, "kind": "day_open", "levels": None})
        elif name.startswith("coinapi_day_open_L"):
            specs.append({"name": name, "kind": "day_open",
                          "levels": int(name.rsplit("L", 1)[1])})
        elif name.startswith("coinapi_stream_L"):
            specs.append({"name": name, "kind": "stream",
                          "levels": int(name.rsplit("L", 1)[1])})
        elif name.startswith("coinapi_on_demand_L"):
            specs.append({"name": name, "kind": "on_demand",
                          "levels": int(name.rsplit("L", 1)[1])})
        else:
            raise ValueError(f"unrecognized arm name {name!r}")
    return specs


def k_ref_for(arm_specs: list[dict], *, k: int) -> int:
    """Reference-frame depth: must cover the parity k AND every arm that reads the
    reference frame (stream candidates AND the on-demand frame provider)."""
    frame_backed = [sp["levels"] for sp in arm_specs
                    if sp["kind"] in ("stream", "on_demand") and sp.get("levels")]
    return max([k] + frame_backed)


def full_day_gb_from_manifest(manifest_path, day: str):
    """Billable GB for the day's vendor csv.gz from the decode manifest (`status=ok`
    rows carry `src_bytes` = the size CoinAPI actually billed at $1/GB)."""
    p = pathlib.Path(manifest_path)
    if not p.exists():
        return None, "missing"
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        if row.get("dt") == day and row.get("status") == "ok" and row.get("src_bytes"):
            return float(row["src_bytes"]) / 1e9, "measured_src_bytes"
    return None, "missing"


def sha256_file(path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_or_build_reference(capi_path: str, *, day, grid, k_ref: int, cache_dir,
                            chunk_rows: int, src_sha256: str):
    """The day's CoinAPI L3→L2 reference frame at k_ref on the grid, parquet-cached by
    (source sha, k_ref, grid size) — the L3 replay is the expensive step, and every arm
    reuses the same reference."""
    import pandas as pd
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = f"ref_{day}_k{k_ref}_n{len(grid)}_{src_sha256[:16]}"
    fpath = cache_dir / f"{stem}.parquet"
    qpath = cache_dir / f"{stem}_quality.json"
    if fpath.exists() and qpath.exists():
        return pd.read_parquet(fpath), json.loads(qpath.read_text()), True
    frame, quality = reconstruct_coinapi_l2_at_samples(
        iter_coinapi_chunks(capi_path, chunk_rows), k=k_ref, day=day, sample_ts=grid,
        size_policy="decrement", on_unknown="count")
    frame.to_parquet(fpath, index=False)
    with open(qpath, "w") as f:
        json.dump({k: v for k, v in quality.items()}, f, indent=2, default=int)
        f.write("\n")
    return frame, quality, False


def arm_summary_rows(report: dict) -> list[dict]:
    rows = []
    for name, arm in report["arms"].items():
        ev = arm["evaluation"]
        md = ev["parity"].get("mid_diff", {})
        la = ev["parity"].get("label_agreement", {})
        pre = ev.get("preregistered", {})
        row = {
            "day": report["day"], "arm": name,
            "crossed_rate": ev["day_quality"].get("crossed_rate"),
            "missing_frac": ev["day_quality"].get("missing_book_fraction"),
            "thin_frac": ev["day_quality"].get("thin_depth_fraction"),
            "crossed_duration_s": ev["day_quality"].get("crossed_duration_s"),
            "mid_median": md.get("median"), "mid_p95": md.get("p95"),
            "mid_p99": md.get("p99"), "mid_corr": md.get("corr"),
            "spike_gt50_frac": ev["parity"].get("spike_fraction", {}).get(">50"),
            "label_2s": (la.get("2") or {}).get("agreement"),
            "label_10s": (la.get("10") or {}).get("agreement"),
            "label_60s": (la.get("60") or {}).get("agreement"),
            "prereg_pass": pre.get("pass"),
            "prereg_guarded_pass": (ev.get("preregistered_guarded") or {}).get("pass"),
            "prereg_pass_effective": arm.get("prereg_pass_effective"),
            "econ_pass": (arm.get("economics") or {}).get("pass"),
            "prereg_failed": ";".join(pre.get("failed", [])),
            "n_grid": ev["parity"].get("n_grid"),
        }
        od = arm["meta"].get("on_demand")
        if od:
            row["n_requests"] = od["n_requests"]
        rows.append(row)
    return rows


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="#54 CoinAPI snapshot-only seeding experiment (offline, one day)")
    ap.add_argument("--day", required=True, help="fixture day YYYY-MM-DD")
    ap.add_argument("--k", type=int, default=10, help="parity top-K (default 10)")
    ap.add_argument("--grid-ms", type=int, default=1000)
    ap.add_argument("--arms", default=DEFAULT_ARMS,
                    help=f"csv of arm names (default: {DEFAULT_ARMS})")
    ap.add_argument("--variant", choices=("real", "leading_gap", "sparse"),
                    default="real",
                    help="emulated degradation of the Lake day (preregistered "
                         "emulations; 'real' = untouched)")
    ap.add_argument("--gap-start-hours", type=float, default=6.0,
                    help="leading_gap: drop Lake deltas before this hour (default 6h)")
    ap.add_argument("--sparse-keep-mod", type=int, default=10,
                    help="sparse: drop every Nth delta row (default 10 -> drop 10%%)")
    ap.add_argument("--seed-min-levels", type=int, default=5,
                    help="acceptance depth floor (production seed gate default 5)")
    ap.add_argument("--max-age-s", type=float, default=60.0,
                    help="snapshot staleness bar for acceptance (default 60s)")
    ap.add_argument("--trigger-after-crossed-s", type=float, default=2.0,
                    help="sustained-crossing trigger/reseed window (production 2.0s)")
    ap.add_argument("--max-requests", type=int, default=24,
                    help="on-demand arm request budget per day")
    ap.add_argument("--injection-guard-s", type=float, default=60.0,
                    help="shared-source guard window after each injection")
    ap.add_argument("--engine", choices=("auto", "python", "native"), default="auto")
    ap.add_argument("--exchange", default="COINBASE")
    ap.add_argument("--symbol", default="BTC-USD")
    ap.add_argument("--coinapi-root", default="data/raw")
    ap.add_argument("--lake-cache-root", default=".lake_cache",
                    help="existing lakeapi joblib cache root (read-only)")
    ap.add_argument("--book-stride-ms", type=int, default=1000)
    ap.add_argument("--book-max-levels", type=int, default=20)
    ap.add_argument("--chunk-rows", type=int, default=2_000_000)
    ap.add_argument("--out-dir", default="data/reports/snapshot_seed")
    ap.add_argument("--no-input-hash", action="store_true",
                    help="skip sha256 of the (multi-GB) inputs for quick smoke runs")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    import pandas as pd

    from experiments.snapshot_seed import (SnapshotAcceptance, emulate_degradation,
                                           load_lake_cached_day, run_experiment_day)

    args = parse_args(argv)
    day = dt.date.fromisoformat(args.day)
    arm_names = [a.strip() for a in args.arms.split(",") if a.strip()]
    arm_specs = arm_specs_from_names(arm_names)

    engine, price_scale, engine_note = _native.resolve_engine(
        args.engine, exchange=args.exchange, symbol=args.symbol)
    if args.engine == "native" and engine != "native":
        print(f"ERROR: {engine_note}", file=sys.stderr)
        return NATIVE_UNAVAILABLE_EXIT
    if engine_note:
        print(f"NOTE: {engine_note}", file=sys.stderr)
    print(f"Lake replay engine: {engine}"
          + (f" (tick scale {price_scale})" if engine == "native" else ""))

    capi_path = coinapi_parquet_path(args.coinapi_root, day, args.exchange, args.symbol)
    if not os.path.exists(capi_path):
        print(f"ERROR: CoinAPI parquet for {day} not found at:\n  {capi_path}\n"
              "This experiment NEVER downloads vendor data; use a locally available "
              "fixture day (see experiments/preregistration_54.json).", file=sys.stderr)
        return 3

    grid = build_grid(day, args.grid_ms)
    k_ref = k_ref_for(arm_specs, k=args.k)

    src_sha = "skipped" if args.no_input_hash else sha256_file(capi_path)
    print(f"CoinAPI parquet: {capi_path}\n  sha256: {src_sha}")

    print(f"Loading Lake book_delta_v2 {args.day} from cache {args.lake_cache_root} …")
    lake_df, lake_info = load_lake_cached_day(
        args.lake_cache_root, table="book_delta_v2", exchange=args.exchange,
        symbol=args.symbol, day=args.day)
    print(f"  rows: {len(lake_df):,} from {lake_info['n_files']} file(s)")

    lake_book_snaps = None
    if any(sp["kind"] == "lake_book" for sp in arm_specs):
        print("Loading Lake `book` snapshot product from cache …")
        book_df, _ = load_lake_cached_day(
            args.lake_cache_root, table="book", exchange=args.exchange,
            symbol=args.symbol, day=args.day)
        etc = shared_engine_time_col(book_df)
        lake_book_snaps = snapshots_from_lake_book_df(
            book_df, engine_time_col=etc, max_levels=args.book_max_levels,
            stride_ns=args.book_stride_ms * 1_000_000)
        print(f"  candidates: {len(lake_book_snaps):,}")
        del book_df

    variant_info = None
    if args.variant == "leading_gap":
        start_ts = grid[0] + int(args.gap_start_hours * 3600 * 1e9)
        lake_df, variant_info = emulate_degradation(lake_df, "leading_gap",
                                                    start_ts=start_ts)
    elif args.variant == "sparse":
        lake_df, variant_info = emulate_degradation(lake_df, "sparse",
                                                    keep_mod=args.sparse_keep_mod)
    if variant_info:
        print(f"Variant {args.variant}: rows {variant_info['rows_before']:,} -> "
              f"{variant_info['rows_after']:,}")

    print(f"Building/loading CoinAPI reference frame (k_ref={k_ref}) …")
    ref_frame, ref_quality, cached = load_or_build_reference(
        capi_path, day=day, grid=grid, k_ref=k_ref,
        cache_dir=os.path.join(args.out_dir, "cache"), chunk_rows=args.chunk_rows,
        src_sha256=src_sha)
    print(f"  reference: {len(ref_frame):,} samples ({'cache hit' if cached else 'built'})")

    gb, gb_basis = full_day_gb_from_manifest(
        os.path.join(args.coinapi_root, "_manifest.jsonl"), args.day)
    if gb is None:
        # No measured billable size (decode manifest absent or lacks the day): leave
        # the day UNPRICED rather than substituting the local parquet size — the
        # economics gate must run on the vendor's billable csv.gz bytes or not at
        # all, and effective_prereg_pass fails closed on unpriced snapshot arms.
        print("WARNING: no measured src_bytes for this day in _manifest.jsonl; "
              "snapshot arms run UNPRICED and the effective verdict fails closed "
              "on economics.", file=sys.stderr)

    acceptance = SnapshotAcceptance(min_levels_per_side=args.seed_min_levels,
                                    max_age_s=args.max_age_s,
                                    tick_scale=(price_scale if engine == "native"
                                                else _native.tick_scale_for(
                                                    args.exchange, args.symbol)))
    input_info = {
        "coinapi_parquet": capi_path, "coinapi_parquet_sha256": src_sha,
        "coinapi_reference_quality": {k: v for k, v in ref_quality.items()
                                      if not isinstance(v, (list, dict))},
        "lake_cache": lake_info, "lake_rows": int(len(lake_df)),
        "variant": args.variant, "variant_info": variant_info,
        "full_day_book_gb": gb, "full_day_book_gb_basis": gb_basis,
    }

    print(f"Running {len(arm_specs)} arm(s) …")
    report = run_experiment_day(
        day=args.day, lake_df=lake_df, coinapi_chunks_factory=lambda:
        iter_coinapi_chunks(capi_path, args.chunk_rows), reference_frame=ref_frame,
        arm_specs=arm_specs, acceptance=acceptance, grid=grid, k=args.k,
        lake_book_snapshots=lake_book_snaps,
        injection_guard_s=args.injection_guard_s,
        trigger_after_crossed_s=args.trigger_after_crossed_s,
        max_requests=args.max_requests, engine=engine, price_scale=price_scale,
        full_day_book_gb=gb, input_info=input_info)

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = "" if args.variant == "real" else f"_{args.variant}"
    jpath = os.path.join(args.out_dir, f"snapshot_seed_{args.day}{suffix}.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2, allow_nan=False)
        f.write("\n")
    cpath = os.path.join(args.out_dir, f"snapshot_seed_{args.day}{suffix}_arms.csv")
    pd.DataFrame(arm_summary_rows(report)).to_csv(cpath, index=False)

    print("\n" + "=" * 74)
    print(f"  #54 SNAPSHOT-SEED EXPERIMENT — {args.day} variant={args.variant} "
          f"(k={args.k}, engine={engine})")
    print("=" * 74)
    for row in arm_summary_rows(report):
        corr = f"{row['mid_corr']:.6f}" if row.get("mid_corr") is not None else "n/a"
        p99 = f"{row['mid_p99']:.2f}" if row.get("mid_p99") is not None else "n/a"
        print(f"  {row['arm']:<28} crossed {row['crossed_rate'] if row['crossed_rate'] is not None else float('nan'):.4%} "
              f"| p99 ${p99} | corr {corr} | 2s {row['label_2s']}"
              f" | prereg {'PASS' if row['prereg_pass_effective'] else 'fail'}"
              f" (guarded {'ok' if row['prereg_guarded_pass'] else '—'})"
              + (f" | econ {'ok' if row['econ_pass'] else 'FAIL'}"
                 if row["econ_pass"] is not None else "")
              + (f" | req {row['n_requests']}" if "n_requests" in row else ""))
    print(f"\n  wrote {jpath}\n        {cpath}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
