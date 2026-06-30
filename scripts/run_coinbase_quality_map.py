"""Quota-aware multi-day Coinbase `book_delta_v2` quality map (docs/data.md §5a-Recon, §10 TODO).

Answers the open §10 question — *how many PRESENT Coinbase days reconstruct to a usable
`book_delta_v2` book after the seed/reseed policy, and which present-but-degraded or missing days
need CoinAPI fill?* — without unlocking the backfill gate. For each requested day it runs the SAME
seed/reseed reconstruction-quality path the one-day parity gate uses on the Lake side
(`recon.reseed.reconstruct_lake_l2_at_samples_seeded` + `recon.parity.frame_quality`) and classifies
the day. It is the multi-day GENERALIZATION of `scripts/run_coinbase_parity.py`, restricted to the
Lake side (no CoinAPI replay), so it needs no CoinAPI download — it only RECORDS whether a local
CoinAPI parquet / a calendar-verified fill is available per day.

This is a VALIDATION / quality-map tool, NOT a backfill:
  * It does NOT download CoinAPI and does NOT unlock the §5a backfill gate (still enforced in
    `ingest/download_coinapi.py` / `ingest/_common.py`).
  * It respects Crypto Lake's **300 GB/month** download quota (docs/data.md §2.1/§6/§8): it prints
    current `lakeapi.used_data(sess)`, estimates the requested Lake GB from the §6 measured per-day
    sizes, and REFUSES a broad pull unless `--allow-broad` is passed — and always refuses a pull that
    would breach the monthly quota headroom, override or not.
  * Default day set is small (2025-06-01 = the validated clean day → expected `lake_usable`;
    2026-04-01 = the crossed-`book`-product day, ~31.75% of `book` seed candidates crossed → its seed
    SOURCE is too unreliable to trust → expected `inconclusive`). Add more with `--days`,
    `--days-file`, or `--include-gap-days N` (documented gap/seam days from the usable calendar). It
    writes the full report under `data/reports/` (git-ignored).

Classifications (explicit thresholds, emitted in the report JSON — see `Thresholds`):
  * lake_usable            — present, seed accepted, crossed/missing/thin all within the usable bar.
  * lake_present_degraded  — present, seed accepted, but a metric is over the usable bar → CoinAPI fill.
  * missing_needs_coinapi  — no Lake `book_delta_v2` for the day (a gap) → CoinAPI fill.
  * excluded              — out of the usable calendar for a non-Coinbase reason (e.g. a Binance gap);
                            skipped before any Lake load (saves quota).
  * inconclusive          — cannot validate the reconstruction: no/all-rejected seed, OR an accepted
                            seed whose `book` SOURCE is itself substantially crossed (e.g. 2026-04-01
                            ~31.75%), OR the Lake load failed. Needs a better seed source or CoinAPI
                            fill before a verdict.

Usage:
  .venv/bin/python scripts/run_coinbase_quality_map.py                       # default 2-day set
  .venv/bin/python scripts/run_coinbase_quality_map.py --days 2025-06-01,2026-04-01
  .venv/bin/python scripts/run_coinbase_quality_map.py --days-file days.txt --allow-broad
  .venv/bin/python scripts/run_coinbase_quality_map.py --include-gap-days 3   # + calendar gap/seam days

Credentials: Crypto Lake AWS keys in .env (Lake-only; no COINAPI_KEY needed). Mirrors
`scripts/run_coinbase_parity.py::lake_session`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import sys
from dataclasses import dataclass

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recon.ingest import shared_engine_time_col                       # noqa: E402
from recon.parity import frame_quality                               # noqa: E402
from recon.reseed import (                                           # noqa: E402
    ReseedPolicy, reconstruct_lake_l2_at_samples_seeded, snapshots_from_lake_book_df,
)

NS_PER_MS = 1_000_000
DAY_MS = 86_400_000

# ----------------------------------------------------------------------------- classifications
LAKE_USABLE = "lake_usable"
LAKE_PRESENT_DEGRADED = "lake_present_degraded"
MISSING_NEEDS_COINAPI = "missing_needs_coinapi"
EXCLUDED = "excluded"
INCONCLUSIVE = "inconclusive"
CLASSES = (LAKE_USABLE, LAKE_PRESENT_DEGRADED, MISSING_NEEDS_COINAPI, EXCLUDED, INCONCLUSIVE)

# ----------------------------------------------------------------------------- quota constants
QUOTA_GB = 300.0           # Crypto Lake individual-plan monthly download cap (docs/data.md §2.1/§8)
DEFAULT_MAX_AUTO_GB = 5.0  # a request larger than this is a "broad" pull → needs --allow-broad
DEFAULT_HEADROOM_GB = 10.0  # never plan to use the last N GB of the monthly quota (used_data lags ~60 min)
QUOTA_REFUSED_EXIT = 5     # small-int exit-code convention (cf. parity exit 3, backfill gate exit 4)

# Conservative per-day Lake footprint by product, GB (docs/data.md §6: Coinbase book_delta_v2+trades
# ~303 MB/day; the `book` 20-level snapshot product ~180 MB/day). We load book_delta_v2 (projected to
# 6 cols) + `book` (for seeding); estimate the FULL product sizes so the quota gate over-estimates
# rather than under-estimates a pull. Known gap days actually cost ~0, so this is an upper bound.
LAKE_GB_PER_DAY = {"book_delta_v2": 0.30, "book": 0.18}
LAKE_PRODUCTS = ("book_delta_v2", "book")

# Small default validation set: 2025-06-01 = the validated clean day (→ lake_usable); 2026-04-01 =
# the crossed-`book`-product day (~31.75% crossed seed source → inconclusive). docs §5a-QualityMap.
DEFAULT_DAYS = ("2025-06-01", "2026-04-01")


@dataclass(frozen=True)
class Thresholds:
    """Usable-day thresholds applied to the seeded reconstruction (emitted in the report JSON).

    Conservative initial values, tunable as multi-day evidence accrues. 2025-06-01 (the validated
    day) sits far inside `lake_usable`: 0.015% crossed, ~0% missing, 0% crossed seed source."""
    crossed_usable_max: float = 0.01    # ≤1% of grid samples crossed after reseed
    missing_usable_max: float = 0.02    # ≤2% of grid samples with no top-of-book on a side
    thin_usable_max: float = 0.10       # ≤10% of grid samples present+uncrossed but thin (< k/side)
    seed_crossed_frac_max: float = 0.05  # ≤5% of `book` seed candidates may be crossed; above this the
                                         # seed SOURCE is unreliable → inconclusive (2026-04-01: 31.75%)

    def as_dict(self) -> dict:
        return {"crossed_usable_max": self.crossed_usable_max,
                "missing_usable_max": self.missing_usable_max,
                "thin_usable_max": self.thin_usable_max,
                "seed_crossed_frac_max": self.seed_crossed_frac_max}


THRESHOLDS = Thresholds()


# ----------------------------------------------------------------------------- pure helpers
def build_grid(day: dt.date, grid_ms: int) -> list[int]:
    """Exchange-time sample grid (int ns) spanning the partition day at `grid_ms` spacing.
    Mirrors `scripts/run_coinbase_parity.py::build_grid` (the shared sampling convention)."""
    day_open = int(pd.Timestamp(day).value)
    step_ns = grid_ms * NS_PER_MS
    n = DAY_MS // grid_ms
    return [day_open + i * step_ns for i in range(n)]


def _json_safe(obj):
    """Recursively coerce to strict-JSON-valid types: non-finite floats → None, numpy scalars →
    python scalars, so the artifact passes `jq empty` (AGENTS.md). Mirrors run_coinbase_parity."""
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


def estimate_lake_gb(n_days, *, products=LAKE_PRODUCTS, per_product_gb=None) -> float:
    """Conservative upper-bound Lake download estimate (GB) for `n_days` × `products`."""
    per = per_product_gb or LAKE_GB_PER_DAY
    return float(n_days) * sum(per[p] for p in products)


def quota_decision(*, est_gb, used_gb, quota_gb=QUOTA_GB, max_auto_gb=DEFAULT_MAX_AUTO_GB,
                   allow_broad=False, headroom_gb=DEFAULT_HEADROOM_GB) -> dict:
    """Decide whether a Lake pull of `est_gb` is allowed given current monthly `used_gb`.

    Two gates: (1) the monthly quota is a HARD external limit — refuse if the pull would leave less
    than `headroom_gb` of the `quota_gb` cap, REGARDLESS of `allow_broad` (used_data lags ~60 min, so
    we keep headroom). (2) a soft auto-cap — a pull larger than `max_auto_gb` is "broad" and refused
    unless `allow_broad`. Returns a JSON-safe decision dict (`ok`, `reason`, and the inputs)."""
    remaining = float(quota_gb) - float(used_gb)
    safe_remaining = remaining - float(headroom_gb)
    base = {"est_gb": float(est_gb), "used_gb": float(used_gb), "quota_gb": float(quota_gb),
            "remaining_gb": remaining, "safe_remaining_gb": safe_remaining,
            "max_auto_gb": float(max_auto_gb), "allow_broad": bool(allow_broad),
            "headroom_gb": float(headroom_gb)}
    if est_gb <= 0:  # nothing to load (e.g. all days excluded) — always allowed, headroom irrelevant
        return {**base, "ok": True, "reason": "ok"}
    if est_gb > safe_remaining:
        return {**base, "ok": False, "reason": "quota_headroom"}
    if est_gb > max_auto_gb and not allow_broad:
        return {**base, "ok": False, "reason": "exceeds_auto_cap"}
    return {**base, "ok": True, "reason": "ok"}


def classify_day(*, have_lake: bool, meta: dict | None, lake_q: dict | None,
                 thresholds: Thresholds = THRESHOLDS) -> tuple[str, list[str]]:
    """Classify a day from its Lake reconstruction metrics. Pure: `meta` is the seed/reseed metrics
    dict, `lake_q` is `frame_quality(...)`. Returns `(classification, reason_codes)`.

    A confident `lake_usable`/`lake_present_degraded` verdict requires a VALIDATED accepted seed AND
    a trustworthy seed SOURCE — without an accepted seed (none/all rejected), a cold-started book can
    look uncrossed for the wrong reason; and even with an accepted seed, if the `book` seed source is
    itself substantially crossed (e.g. 2026-04-01: 31.75%), seeds during crossed episodes are
    unreliable and reseeds get blocked, so a clean-looking reconstruction can't be trusted. Both →
    `inconclusive` (the day needs a better seed source or CoinAPI fill), not silently usable."""
    if not have_lake:
        return MISSING_NEEDS_COINAPI, ["lake_book_delta_v2_absent"]
    crossed = float(lake_q["crossed_rate"])
    missing = float(lake_q["missing_book_fraction"])
    thin = float(meta.get("thin_depth_fraction") or 0.0)
    if not meta.get("seed_accepted"):
        sr = meta.get("seed_reason")
        code = "no_seed_snapshots" if sr in (None, "no_snapshots") else f"seed_rejected:{sr}"
        return INCONCLUSIVE, [code, f"crossed_rate_after={crossed:.4f}"]
    # Seed accepted, but is the seed SOURCE itself reliable? The `book` product is intermittently
    # crossed (2026-04-01: 31.75%); above seed_crossed_frac_max the accepted seed cannot be trusted.
    codes = meta.get("snapshot_reason_codes") or {}
    n_snap = sum(codes.values())
    seed_crossed_frac = (codes.get("crossed", 0) / n_snap) if n_snap else 0.0
    if seed_crossed_frac > thresholds.seed_crossed_frac_max:
        return INCONCLUSIVE, ["seed_accepted_but_source_unreliable",
                              f"seed_source_crossed_frac={seed_crossed_frac:.4f}>"
                              f"{thresholds.seed_crossed_frac_max}"]
    over: list[str] = []
    if crossed > thresholds.crossed_usable_max:
        over.append(f"crossed_rate_after={crossed:.4f}>{thresholds.crossed_usable_max}")
    if missing > thresholds.missing_usable_max:
        over.append(f"missing_book_fraction={missing:.4f}>{thresholds.missing_usable_max}")
    if thin > thresholds.thin_usable_max:
        over.append(f"thin_depth_fraction={thin:.4f}>{thresholds.thin_usable_max}")
    if over:
        return LAKE_PRESENT_DEGRADED, ["seed_accepted", *over]
    return LAKE_USABLE, ["seed_accepted", f"crossed_rate_after={crossed:.4f}",
                         f"missing_book_fraction={missing:.4f}", f"thin_depth_fraction={thin:.4f}"]


def _empty_seed_block(snapshots_present, candidates) -> dict:
    return {"snapshots_present": bool(snapshots_present), "snapshot_candidates": int(candidates),
            "seed_accepted": False, "seed_reason": None, "seed_ts": None, "reseed_count": 0,
            "reseed_blocked_invalid_snapshot": 0, "snapshot_reason_codes": {}}


def _empty_quality_block(k, grid_ms) -> dict:
    return {"k": int(k), "grid_ms": int(grid_ms), "n_grid": None, "engine_time_col": None,
            "crossed_rate_after": None, "crossed_samples_after": None, "crossed_rate_cold": None,
            "missing_book_fraction": None, "thin_depth_fraction": None,
            "crossed_duration_s_after": None}


def _default_coinapi_block() -> dict:
    return {"parquet_local": None, "parquet_path": None, "fillable": None}


def _default_calendar_block() -> dict:
    return {"in_usable_days": None, "in_lake_all_days": None, "is_coinbase_fill_day": None,
            "excluded_reason": None}


def assess_lake_day(lake_delta_df, lake_book_snapshots, *, day: dt.date, k: int = 10,
                    grid_ms: int = 1000, reseed: bool = True, reseed_after_crossed_s: float = 2.0,
                    seed_min_levels: int = 5, max_spread_frac: float | None = None,
                    cold_ab: bool = True, thresholds: Thresholds = THRESHOLDS,
                    coinapi: dict | None = None, calendar: dict | None = None) -> dict:
    """Run the seed/reseed Lake reconstruction-quality path for one day and classify it. PURE w.r.t.
    its inputs (a pre-loaded Lake `book_delta_v2` DataFrame + optional pre-parsed `book` snapshots) —
    no vendor I/O — so the offline tests drive the exact production classification path."""
    have_lake = lake_delta_df is not None and len(lake_delta_df) > 0
    n_rows = (0 if lake_delta_df is None else int(len(lake_delta_df)))
    snaps = list(lake_book_snapshots or [])
    have_snaps = bool(snaps)
    result = {
        "day": day.isoformat(),
        "lake_book_delta_v2_present": bool(have_lake),
        "lake_delta_rows": n_rows,
        "coinapi": coinapi if coinapi is not None else _default_coinapi_block(),
        "calendar": calendar if calendar is not None else _default_calendar_block(),
    }

    if not have_lake:
        cls, reasons = classify_day(have_lake=False, meta=None, lake_q=None, thresholds=thresholds)
        result.update(classification=cls, reasons=reasons,
                      seed=_empty_seed_block(have_snaps, len(snaps)),
                      quality=_empty_quality_block(k, grid_ms))
        return result

    grid = build_grid(day, grid_ms)
    engine_col = shared_engine_time_col(lake_delta_df)
    policy = ReseedPolicy(enabled=reseed, min_levels_per_side=seed_min_levels,
                          reseed_after_crossed_s=reseed_after_crossed_s,
                          max_spread_frac=max_spread_frac)
    frame, meta = reconstruct_lake_l2_at_samples_seeded(
        lake_delta_df, grid, k=k, engine_time_col=engine_col,
        snapshots=(snaps or None), policy=policy)
    lake_q = frame_quality(frame, source_rows=n_rows)

    cold_rate = None
    if cold_ab and have_snaps:
        # A/B "before": the byte-identical reconstruction cold-started (no seed/reseed); metrics-only
        # (frame_out=False) so the 86,400-row frame is not re-materialized. Doubles the per-day delta
        # replay — disable with --no-cold-ab on a multi-GB-day sweep where only the after-rate matters.
        _, cold_meta = reconstruct_lake_l2_at_samples_seeded(
            lake_delta_df, grid, k=k, engine_time_col=engine_col, snapshots=None, frame_out=False)
        cold_rate = float(cold_meta["crossed_rate"])

    cls, reasons = classify_day(have_lake=True, meta=meta, lake_q=lake_q, thresholds=thresholds)
    result.update(
        classification=cls, reasons=reasons,
        seed={
            "snapshots_present": have_snaps,
            "snapshot_candidates": len(snaps),
            "seed_accepted": bool(meta["seed_accepted"]),
            "seed_reason": meta["seed_reason"],
            "seed_ts": meta["seed_ts"],
            "reseed_count": int(meta["reseed_count"]),
            "reseed_blocked_invalid_snapshot": int(meta["reseed_blocked_invalid_snapshot"]),
            "snapshot_reason_codes": dict(meta["snapshot_reason_codes"]),
        },
        quality={
            "k": int(k), "grid_ms": int(grid_ms), "n_grid": len(grid),
            "engine_time_col": engine_col,
            "crossed_rate_after": float(lake_q["crossed_rate"]),
            "crossed_samples_after": int(lake_q["crossed_samples"]),
            "crossed_rate_cold": cold_rate,
            "missing_book_fraction": float(lake_q["missing_book_fraction"]),
            "thin_depth_fraction": float(meta["thin_depth_fraction"]),
            "crossed_duration_s_after": float(meta["crossed_duration_s"]),
        },
    )
    return result


def excluded_result(day: dt.date, reasons, *, k: int = 10, grid_ms: int = 1000,
                    coinapi: dict | None = None, calendar: dict | None = None) -> dict:
    """A schema-consistent `excluded` per-day record for a day skipped before any Lake load."""
    return {
        "day": day.isoformat(), "classification": EXCLUDED, "reasons": list(reasons),
        "lake_book_delta_v2_present": None, "lake_delta_rows": None,
        "seed": _empty_seed_block(False, 0), "quality": _empty_quality_block(k, grid_ms),
        "coinapi": coinapi if coinapi is not None else _default_coinapi_block(),
        "calendar": calendar if calendar is not None else _default_calendar_block(),
    }


def inconclusive_load_failure(day: dt.date, err: str, *, k: int = 10, grid_ms: int = 1000,
                              coinapi: dict | None = None, calendar: dict | None = None) -> dict:
    """A schema-consistent `inconclusive` record for a day whose Lake load raised (can't be judged)."""
    return {
        "day": day.isoformat(), "classification": INCONCLUSIVE,
        "reasons": [f"lake_load_failed:{err}"],
        "lake_book_delta_v2_present": None, "lake_delta_rows": None,
        "seed": _empty_seed_block(False, 0), "quality": _empty_quality_block(k, grid_ms),
        "coinapi": coinapi if coinapi is not None else _default_coinapi_block(),
        "calendar": calendar if calendar is not None else _default_calendar_block(),
    }


def build_report(per_day_results, *, meta: dict) -> dict:
    """Aggregate per-day results into the report: stable per-class counts + day lists + the rows."""
    by_class: dict[str, list] = {c: [] for c in CLASSES}
    for r in per_day_results:
        by_class.setdefault(r["classification"], []).append(r["day"])
    counts = {c: len(by_class[c]) for c in by_class}
    return {"meta": meta,
            "summary": {"n_days": len(per_day_results), "counts": counts, "by_class": by_class},
            "days": list(per_day_results)}


def write_report(report: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(_json_safe(report), f, indent=2, allow_nan=False)
        f.write("\n")
    return path


# ----------------------------------------------------------------------------- credentials (vendor)
def lake_session():
    """Crypto Lake boto3 session from .env subscriber keys (NOT the personal ~/.aws default).
    Lake-only: does not require COINAPI_KEY. Mirrors scripts/run_coinbase_parity.py::lake_session."""
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


def lake_used_data(sess) -> dict:
    """Current monthly Crypto Lake download usage: {downloaded_gb, timeframe_days, ...}. May lag
    up to ~60 min (vendor-side) and is cached 60 s (lakeapi FSLRUCache)."""
    import lakeapi  # local import: keep module import side-effect-free / vendor-free
    return lakeapi.used_data(sess)


# ----------------------------------------------------------------------------- data loading (vendor)
def load_lake_book_delta_v2(sess, day: dt.date, exchange: str, symbol: str) -> pd.DataFrame:
    """Load one day of Crypto Lake `book_delta_v2` (projected to the 6 recon columns). Mirrors
    scripts/run_coinbase_parity.py."""
    import lakeapi  # local import
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
    """Load one day of the Crypto Lake `book` (20-level snapshot) product as seed/reseed candidates,
    thinned to ~one per `stride_ms`. Mirrors scripts/run_coinbase_parity.py."""
    import lakeapi  # local import
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


def _is_no_files(exc: BaseException) -> bool:
    """True if `exc` signals an ABSENT Lake partition (a gap day). lakeapi (0.22.3) raises
    `NoFilesFound` for a fully-missing partition rather than returning an empty frame, so a gap must
    be detected here and routed to missing_needs_coinapi — NOT treated as a load failure."""
    return (type(exc).__name__ == "NoFilesFound"
            or "nofilesfound" in repr(exc).lower()
            or "no files found" in str(exc).lower())


def coinapi_parquet_path(out_root: str, day: dt.date, exchange: str, symbol: str) -> str:
    return os.path.join(out_root, "limitbook_full", f"exchange={exchange}",
                        f"symbol={symbol}", f"dt={day.isoformat()}", "data.parquet")


# ----------------------------------------------------------------------------- usable calendar
def load_usable_calendar(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def calendar_context(cal: dict | None, day_iso: str) -> dict:
    if cal is None:
        return _default_calendar_block()
    return {
        "in_usable_days": day_iso in set(cal.get("usable_days", [])),
        "in_lake_all_days": day_iso in set(cal.get("lake_all_days", [])),
        "is_coinbase_fill_day": day_iso in (cal.get("coinbase_fill_days") or {}),
        "excluded_reason": (cal.get("excluded_days_by_reason") or {}).get(day_iso),
    }


def coinapi_context(cal: dict | None, day: dt.date, coinapi_root: str,
                    exchange: str, symbol: str) -> dict:
    """Whether CoinAPI parity is AVAILABLE for the day — a local parquet on disk and/or a
    calendar-verified flat-files `book` fill — without downloading anything."""
    path = coinapi_parquet_path(coinapi_root, day, exchange, symbol)
    fillable = None
    if cal is not None:
        fs = (cal.get("fill_status") or {}).get(day.isoformat())
        if fs is not None:
            b = fs.get("book")
            fillable = bool(b and b.get("present"))
    return {"parquet_local": os.path.exists(path), "parquet_path": path, "fillable": fillable}


def gap_days_from_calendar(cal: dict | None, n: int) -> list[str]:
    """Up to `n` documented Coinbase gap/seam days (the calendar's coinbase_fill_days, time-sorted) —
    each a day where Lake lacks Coinbase data and CoinAPI must fill (expected missing_needs_coinapi)."""
    if cal is None or n <= 0:
        return []
    return sorted((cal.get("coinbase_fill_days") or {}).keys())[:n]


# ----------------------------------------------------------------------------- day-set resolution
def resolve_days(args) -> list[dt.date]:
    if args.days_file:
        text = pathlib.Path(args.days_file).read_text()
        toks = [t.strip() for line in text.splitlines() for t in line.split(",")]
    elif args.days:
        toks = [t.strip() for t in args.days.split(",")]
    else:
        toks = list(DEFAULT_DAYS)
    days = [dt.date.fromisoformat(t) for t in toks if t]
    if args.include_gap_days:
        cal = load_usable_calendar(args.usable_calendar)
        for d in gap_days_from_calendar(cal, args.include_gap_days):
            gd = dt.date.fromisoformat(d)
            if gd not in days:
                days.append(gd)
    # de-dupe, keep order
    seen: set = set()
    out: list[dt.date] = []
    for d in days:
        if d not in seen:
            seen.add(d)
            out.append(d)
    return out


# ----------------------------------------------------------------------------- reporting (stdout)
def print_summary(report: dict) -> None:
    s = report["summary"]
    m = report["meta"]
    print("\n" + "=" * 74)
    print(f"  COINBASE QUALITY MAP — {s['n_days']} day(s)  (k={m.get('k')}, "
          f"grid={m.get('grid_ms')}ms)")
    print("=" * 74)
    for c in CLASSES:
        days = s["by_class"].get(c, [])
        if days:
            shown = ", ".join(days[:8]) + (" …" if len(days) > 8 else "")
            print(f"  {c:<22} {len(days):>4}   {shown}")
        else:
            print(f"  {c:<22} {len(days):>4}")
    for d in report["days"]:
        q = d.get("quality", {})
        extra = ""
        if q.get("crossed_rate_after") is not None:
            extra = (f"crossed {q['crossed_rate_after']:.4%}"
                     + (f" (cold {q['crossed_rate_cold']:.2%})" if q.get("crossed_rate_cold")
                        is not None else "")
                     + f", missing {q['missing_book_fraction']:.4%}")
        print(f"    {d['day']}  {d['classification']:<22} {extra}")


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Quota-aware multi-day Coinbase book_delta_v2 quality map (validation, NOT a "
                    "backfill; does not unlock the §5a CoinAPI backfill gate)")
    ap.add_argument("--days", default=None,
                    help="explicit days, CSV YYYY-MM-DD,YYYY-MM-DD (default: the small validation set "
                         "2025-06-01,2026-04-01)")
    ap.add_argument("--days-file", default=None,
                    help="file of days (CSV and/or one-per-line); overrides --days")
    ap.add_argument("--include-gap-days", type=int, default=0,
                    help="also map the first N documented Coinbase gap/seam days from the usable "
                         "calendar (expected missing_needs_coinapi; default 0)")
    ap.add_argument("--k", type=int, default=10, help="top-K levels (default 10)")
    ap.add_argument("--grid-ms", type=int, default=1000, help="sample-grid spacing ms (default 1000)")
    ap.add_argument("--no-reseed", action="store_true",
                    help="seed once but DISABLE intraday reseed-on-crossing (A/B; §5a-Recon)")
    ap.add_argument("--no-lake-seed", action="store_true",
                    help="do NOT load the Lake `book` snapshot product (pure cold-start; every "
                         "present day then classifies inconclusive — no validated seed)")
    ap.add_argument("--no-cold-ab", action="store_true",
                    help="skip the cold-start A/B crossed-rate (halves per-day reconstruction cost)")
    ap.add_argument("--reseed-after-crossed-s", type=float, default=2.0,
                    help="reseed only after the book is crossed continuously ≥ this many seconds")
    ap.add_argument("--seed-min-levels", type=int, default=5,
                    help="min levels/side for a valid Lake `book` seed/reseed source (default 5)")
    ap.add_argument("--seed-max-spread-frac", type=float, default=None,
                    help="optional sane-spread guard for seed validation (off by default)")
    ap.add_argument("--book-stride-ms", type=int, default=1000,
                    help="thin the Lake `book` product to ~one seed candidate per N ms (default 1000)")
    ap.add_argument("--book-max-levels", type=int, default=20,
                    help="Lake `book` snapshot depth to load/seed from (default 20)")
    ap.add_argument("--exchange", default="COINBASE", help="Crypto Lake / partition exchange")
    ap.add_argument("--symbol", default="BTC-USD", help="Crypto Lake / partition symbol")
    ap.add_argument("--coinapi-root", default="data/raw", help="root of any LOCAL CoinAPI parquet tree")
    ap.add_argument("--usable-calendar", default="data/usable_calendar.json",
                    help="usable-calendar JSON for excluded/gap/fill context (§5b)")
    ap.add_argument("--out-dir", default="data/reports", help="report output dir (git-ignored)")
    # quota controls
    ap.add_argument("--quota-gb", type=float, default=QUOTA_GB,
                    help="Crypto Lake monthly download cap GB (default 300)")
    ap.add_argument("--max-auto-gb", type=float, default=DEFAULT_MAX_AUTO_GB,
                    help="auto-allowed request cap GB; larger needs --allow-broad (default 5)")
    ap.add_argument("--headroom-gb", type=float, default=DEFAULT_HEADROOM_GB,
                    help="quota GB to always leave unused (default 10)")
    ap.add_argument("--allow-broad", action="store_true",
                    help="override the auto cap for a deliberate broad pull (still refused if it "
                         "would breach the monthly quota headroom)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    days = resolve_days(args)
    cal = load_usable_calendar(args.usable_calendar)
    if cal is None:
        print(f"NOTE: usable calendar {args.usable_calendar} not found — excluded/gap/fill context "
              "will be null (run ingest/verify_trades_and_calendar.py to produce it).",
              file=sys.stderr)

    excluded_set = set((cal or {}).get("excluded_days_by_reason", {}))
    to_load = [d for d in days if d.isoformat() not in excluded_set]
    excluded = [d for d in days if d.isoformat() in excluded_set]

    est_gb = estimate_lake_gb(len(to_load))
    print(f"Quality map: {len(days)} day(s) requested — {len(to_load)} to load, "
          f"{len(excluded)} excluded by calendar.")
    print(f"Estimated Crypto Lake download: ~{est_gb:.2f} GB "
          f"({len(to_load)} day(s) × {sum(LAKE_GB_PER_DAY[p] for p in LAKE_PRODUCTS):.2f} GB/day, "
          "conservative upper bound).")

    sess = lake_session()
    try:
        used = lake_used_data(sess)
        used_gb = float(used.get("downloaded_gb", 0.0))
        print(f"Crypto Lake usage BEFORE: {used_gb:.2f} GB / {used.get('timeframe_days')} days "
              f"(cap {args.quota_gb:.0f} GB; may lag ~60 min).")
    except Exception as e:  # noqa: BLE001 — fail safe: cannot confirm headroom ⇒ refuse below
        print(f"WARNING: could not read lakeapi.used_data ({e!r}); assuming worst-case usage at the "
              "monthly cap, so the quota gate will REFUSE any non-empty pull this run regardless of "
              "--allow-broad (cannot confirm headroom). Re-run once used_data is readable.",
              file=sys.stderr)
        used_gb = args.quota_gb

    decision = quota_decision(est_gb=est_gb, used_gb=used_gb, quota_gb=args.quota_gb,
                              max_auto_gb=args.max_auto_gb, allow_broad=args.allow_broad,
                              headroom_gb=args.headroom_gb)
    if not decision["ok"]:
        why = ("would breach the monthly quota headroom"
               if decision["reason"] == "quota_headroom"
               else f"exceeds the {args.max_auto_gb:.0f} GB auto cap")
        print(f"\nREFUSING Lake load: estimate ~{est_gb:.2f} GB {why} "
              f"(remaining ~{decision['remaining_gb']:.1f} GB of {args.quota_gb:.0f} GB).\n"
              "  • Narrow the day set (--days), or\n"
              "  • pass --allow-broad for a deliberate pull that still fits the monthly quota "
              "(ensure headroom; used_data lags ~60 min).", file=sys.stderr)
        return QUOTA_REFUSED_EXIT

    results: list[dict] = []
    for d in excluded:
        di = d.isoformat()
        # d ∈ excluded ⟹ di ∈ excluded_days_by_reason (so cal is present and the key exists).
        results.append(excluded_result(
            d, cal["excluded_days_by_reason"][di], k=args.k, grid_ms=args.grid_ms,
            coinapi=coinapi_context(cal, d, args.coinapi_root, args.exchange, args.symbol),
            calendar=calendar_context(cal, di)))

    for d in to_load:
        di = d.isoformat()
        cctx = coinapi_context(cal, d, args.coinapi_root, args.exchange, args.symbol)
        calctx = calendar_context(cal, di)
        print(f"\nLoading Crypto Lake {args.exchange} {args.symbol} book_delta_v2 for {di} …")
        try:
            lake_df = load_lake_book_delta_v2(sess, d, args.exchange, args.symbol)
        except Exception as e:  # noqa: BLE001
            if _is_no_files(e):
                # An ABSENT Lake partition (a gap day): lakeapi raises NoFilesFound rather than
                # returning an empty frame, so this is missing_needs_coinapi, NOT a load failure.
                print(f"  Lake book_delta_v2 absent for {di} (gap) → missing_needs_coinapi.")
                results.append(assess_lake_day(pd.DataFrame(), None, day=d, k=args.k,
                                               grid_ms=args.grid_ms, coinapi=cctx, calendar=calctx))
            else:
                print(f"  WARNING: Lake book_delta_v2 load failed for {di} ({e!r}) — inconclusive.",
                      file=sys.stderr)
                results.append(inconclusive_load_failure(d, repr(e), k=args.k, grid_ms=args.grid_ms,
                                                         coinapi=cctx, calendar=calctx))
            continue
        print(f"  Lake book_delta_v2 rows: {len(lake_df):,}")

        snaps = None
        if not args.no_lake_seed and len(lake_df):
            try:
                snaps = load_lake_book_snapshots(sess, d, args.exchange, args.symbol,
                                                 max_levels=args.book_max_levels,
                                                 stride_ms=args.book_stride_ms)
                print(f"  Lake book seed candidates: {len(snaps):,}")
            except Exception as e:  # noqa: BLE001 — fall back to cold-start (→ inconclusive)
                print(f"  WARNING: could not load Lake `book` snapshots ({e!r}); cold-start (the day "
                      "will classify inconclusive without a validated seed).", file=sys.stderr)
                snaps = None

        results.append(assess_lake_day(
            lake_df, snaps, day=d, k=args.k, grid_ms=args.grid_ms, reseed=not args.no_reseed,
            reseed_after_crossed_s=args.reseed_after_crossed_s, seed_min_levels=args.seed_min_levels,
            max_spread_frac=args.seed_max_spread_frac, cold_ab=not args.no_cold_ab,
            coinapi=cctx, calendar=calctx))

    used_after = None
    try:
        used_after = float(lake_used_data(sess).get("downloaded_gb", 0.0))
    except Exception:  # noqa: BLE001
        pass

    meta = {
        "k": int(args.k), "grid_ms": int(args.grid_ms),
        "exchange": args.exchange, "symbol": args.symbol,
        "thresholds": THRESHOLDS.as_dict(),
        "policy": {"reseed": not args.no_reseed, "reseed_after_crossed_s": args.reseed_after_crossed_s,
                   "seed_min_levels": args.seed_min_levels, "no_lake_seed": args.no_lake_seed,
                   "cold_ab": not args.no_cold_ab},
        "lake_gb_per_day": LAKE_GB_PER_DAY,
        "quota": {**decision, "used_gb_before": used_gb, "used_gb_after": used_after},
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "note": "VALIDATION quality map (docs/data.md §5a-Recon / §10). The CoinAPI backfill gate "
                "stays LOCKED until the multi-day quality map passes; this tool does not unlock it.",
    }
    report = build_report(results, meta=meta)
    out_path = os.path.join(args.out_dir, "coinbase_quality_map.json")
    write_report(report, out_path)
    print_summary(report)
    print(f"\n  wrote {out_path}")
    if used_after is not None:
        print(f"  Crypto Lake usage AFTER: {used_after:.2f} GB (Δ ~{used_after - used_gb:+.2f} GB; "
              "may lag ~60 min).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
