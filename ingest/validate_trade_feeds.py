"""Thin Crypto Lake CLI wrapper over the pure `ingest/trade_checks.py` checks
(docs/data.md §5b / §10 "trade validation breadth"; plan
docs/superpowers/plans/2026-07-02-trade-validation-breadth-plan.md, Phase 1b / Task 2).

This is the VENDOR half of the repo's established pure/vendor split (pure `ingest.trade_checks` vs.
this loader/CLI — mirroring pure `recon.stitch_policy` vs. the vendor
`scripts/run_coinbase_quality_map.py`). It resolves a BOUNDED `(venue, day)` plan, gates it against
the Crypto Lake monthly quota, loads each required `trades` partition, and feeds the loaded frame to
`trade_checks.validate_trade_frame` UNCHANGED — the pure module is source-agnostic and never imports
a vendor client. `lakeapi`/`boto3` are imported ONLY inside the live seam functions
(`lake_session`/`lake_used_data`/`load_lake_trades`), never at module import, so importing this module
(and the whole synthetic test path) touches no vendor. Fill/excluded days route via the calendar
(`trade_checks.calendar_state`) and are NOT loaded — only required days spend quota.

The final report `trade_feed_validation.json` (git-ignored `data/reports/`) IS the pure module's
`build_report`/`write_report` output; this wrapper only supplies the loaded frames + the meta block.

Usage (Phase 2 is ask-first per AGENTS.md; --dry-run is always vendor-free):
  .venv/bin/python ingest/validate_trade_feeds.py --dry-run
  .venv/bin/python ingest/validate_trade_feeds.py --days 2025-06-01,2024-08-05
  .venv/bin/python ingest/validate_trade_feeds.py --start 2024-06-22 --end 2026-05-05 \
      --sample-n 20 --seed 7 --allow-broad

Credentials: Crypto Lake AWS keys in .env (Lake-only; no COINAPI_KEY). Mirrors
scripts/run_coinbase_quality_map.py::lake_session. CoinAPI is NOT touched; the backfill gate stays
LOCKED and this tool does not unlock it.
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pathlib
import random
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest import trade_checks as tc                                  # noqa: E402  (pure; vendor-free)

# --------------------------------------------------------------------------- selection constants (§3)
# Venue evaluation order = the pure module's canonical `VENUES` order (do NOT invent a new order):
# binance_perp, binance_spot, coinbase. Selection is normalized to this order so a report is
# byte-deterministic regardless of the --venues input order.
CANONICAL_VENUES = tuple(tc.VENUES)

# The safe small default sample (5 curated days, ≈1.3 GB, under the auto cap) run with no day args
# (§3.4). 2024-08-06 is a full Coinbase gap kept ON PURPOSE so the one bounded run exercises the
# `coinapi_fill` route §8 later makes a gate condition.
DEFAULT_SAMPLE_DAYS = ("2025-06-01", "2024-08-05", "2024-08-06", "2025-01-07", "2026-04-15")

# The regime cohort (§3): curated days spanning distinct regimes. Included AS-IS for a --start/--end
# run when in range (curated, so gap days like 2024-08-06 are kept even though absent from
# usable_days — the deliberate fill-routing case); the stratified random sample is drawn from
# usable_days only and added on top.
REGIME_COHORT = ("2025-06-01", "2024-08-05", "2024-08-06", "2025-01-07", "2024-12-04",
                 "2026-04-15", "2026-06-15")

# Provisional, disjoint train/val/test split spans for the seeded stratified sample (§3), covering
# the effective modeling calendar's contiguous usable runs (docs/data.md §8: 2024-06-22→2026-02-04,
# 2026-02-06→2026-05-05, OOS ≈ April 2026). First-pass boundaries — the sample only feeds the
# Phase-2 bounded live run (ask-first); refine once the split calendar is finalized.
SPLIT_SPANS = (
    ("2024-06-22", "2025-12-31"),      # SSL-pretrain
    ("2026-01-01", "2026-02-04"),      # head-finetune / validation
    ("2026-02-06", "2026-06-22"),      # OOS + late window (incl. the untouched ≈ April-2026 run)
)

DEFAULT_END = os.environ.get("END", "2026-06-22")   # the repo `END=` anchor convention (§7)
DEFAULT_CALENDAR = "data/usable_calendar.json"
REPORT_NAME = "trade_feed_validation.json"


# --------------------------------------------------------------------------- day / venue resolution
def _canonical_day(token: str) -> str:
    """Validate and CANONICALIZE a day token to `YYYY-MM-DD`. `date.fromisoformat` (≥3.11) accepts
    non-canonical ISO forms (basic `20240806`, week dates), but the usable-calendar keys are all
    `YYYY-MM-DD`; keeping a raw token would make a known fill/excluded day miss its calendar entry
    and route as a required Lake day. A malformed token raises `ValueError` (fail fast)."""
    return dt.date.fromisoformat(token).isoformat()


def _validate_days(tokens) -> list[str]:
    """Canonicalize each token to `YYYY-MM-DD` and de-dupe on the canonical key, preserving
    first-seen order. A malformed date raises `ValueError`."""
    seen: set[str] = set()
    out: list[str] = []
    for t in tokens:
        iso = _canonical_day(t)
        if iso not in seen:
            seen.add(iso)
            out.append(iso)
    return out


def stratified_sample_days(usable_days, sample_n: int, *, seed: int,
                           spans=SPLIT_SPANS) -> list[str]:
    """A deterministic seeded stratified sample of `sample_n` usable days across the split `spans`
    (§3). Each day is assigned to the FIRST span it falls in (spans are disjoint), buckets are drawn
    round-robin one at a time so the sample is spread across splits even when `sample_n < len(spans)`,
    and each bucket is pre-shuffled with `random.Random(seed)`. Reproducible: the same
    `(usable_days, sample_n, seed)` yields the same days regardless of input order (inputs are sorted
    first)."""
    if sample_n <= 0:
        return []
    rng = random.Random(seed)
    pool = sorted(set(usable_days))
    buckets: list[list[str]] = [[] for _ in spans]
    for d in pool:
        for i, (lo, hi) in enumerate(spans):
            if lo <= d <= hi:
                buckets[i].append(d)
                break
    shuffled = [rng.sample(b, len(b)) for b in buckets]   # deterministic per-bucket order
    picked: list[str] = []
    depth = 0
    while len(picked) < sample_n:
        progressed = False
        for b in shuffled:
            if depth < len(b):
                picked.append(b[depth])
                progressed = True
                if len(picked) >= sample_n:
                    break
        if not progressed:                           # every bucket exhausted
            break
        depth += 1
    return sorted(picked)


def resolve_days(args, cal: dict | None) -> tuple[list[str], str]:
    """Resolve the `(days, selection_mode)` per the §3 precedence (first match wins):
    `--days` → `--days-file` → `--start/--end` (in-range regime cohort + seeded stratified sample of
    `--sample-n` usable days) → the safe 5-day default sample. Bounded by design: no broad default."""
    # `is not None` (not truthiness): an EXPLICITLY-provided-but-empty selector (`--days ""` from an
    # unset shell var) must resolve to [] and hit the empty-selection guard, NOT silently fall
    # through to the default sample and pull data on a live run.
    if args.days is not None:
        return _validate_days(t.strip() for t in args.days.split(",") if t.strip()), "explicit_days"
    if args.days_file is not None:
        text = pathlib.Path(args.days_file).read_text() if args.days_file else ""
        toks = (t.strip() for line in text.splitlines() for t in line.split(",") if t.strip())
        return _validate_days(toks), "days_file"
    if args.start:
        start = _canonical_day(args.start)                        # canonicalize the range endpoints
        end = _canonical_day(args.end)                            # so YYYY-MM-DD string compares hold
        cohort = [d for d in REGIME_COHORT if start <= d <= end]
        usable = [d for d in (cal or {}).get("usable_days", []) if start <= d <= end]
        sample = stratified_sample_days([d for d in usable if d not in cohort],
                                        args.sample_n, seed=args.seed)
        return sorted(set(cohort) | set(sample)), "range_sample"
    return list(DEFAULT_SAMPLE_DAYS), "default_sample"


def resolve_venues(venues_arg: str | None) -> list[str]:
    """Resolve `--venues` (CSV) to a subset of `trade_checks.VENUES` in canonical order (default: all
    three). An unknown venue key raises `ValueError` rather than silently dropping a requested feed."""
    if not venues_arg:
        return list(CANONICAL_VENUES)
    requested = {t.strip() for t in venues_arg.split(",") if t.strip()}
    unknown = requested - set(tc.VENUES)
    if unknown:
        raise ValueError(f"unknown venue(s): {sorted(unknown)}; valid: {sorted(tc.VENUES)}")
    return [v for v in CANONICAL_VENUES if v in requested]


def estimate_plan_gb(venues, days) -> float:
    """Conservative upper-bound Lake `trades` GB for `venues` × `days` — reuses the pure
    `trade_checks.estimate_trades_gb` (over the full grid; fill/excluded days that won't be loaded
    make this err high, which is what the quota gate wants)."""
    return tc.estimate_trades_gb(venues, days)


# --------------------------------------------------------------------------- live seams (VENDOR only)
# lakeapi/boto3 are imported ONLY inside these functions, so importing this module stays vendor-free
# (the case-13 guard). Tests inject fakes for all three, so no synthetic path ever reaches lakeapi.
def lake_session():
    """Crypto Lake boto3 session from .env subscriber keys (NOT ~/.aws). Lake-only: no COINAPI_KEY.
    Mirrors scripts/run_coinbase_quality_map.py::lake_session."""
    import boto3                                     # local import: keep module import vendor-free
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
    """Current monthly Crypto Lake download usage `{downloaded_gb, ...}`. May lag ~60 min (vendor
    side) and is cached 60 s (lakeapi FSLRUCache)."""
    import lakeapi                                   # local import: keep module import vendor-free
    return lakeapi.used_data(sess)


def load_lake_trades(sess, venue: str, day: str):
    """Load ONE `(venue, day)` of Crypto Lake `trades` [00:00, next-00:00), normalized to the
    `trade_checks` schema. lakeapi renames `timestamp→origin_time`, `receipt_timestamp→received_time`
    on load, so the frame already carries the expected column names; `trade_checks` also accepts the
    raw names defensively. Column-projected to the six checked fields to reduce bytes. Mirrors the
    §2 load shape."""
    import lakeapi                                   # local import: keep module import vendor-free
    exch, sym = tc.VENUES[venue]
    start = dt.datetime.combine(dt.date.fromisoformat(day), dt.time())
    end = start + dt.timedelta(days=1)
    return lakeapi.load_data(
        table="trades", start=start, end=end, symbols=[sym], exchanges=[exch],
        columns=["timestamp", "receipt_timestamp", "price", "quantity", "side", "trade_id"],
        boto3_session=sess, drop_partition_cols=True,
    )


def _is_no_files(exc: BaseException) -> bool:
    """True if `exc` signals an ABSENT Lake partition (a gap day): lakeapi raises `NoFilesFound`
    rather than returning an empty frame, so a gap must route to `missing_partition`, NOT a
    `load_error`. Mirrors scripts/run_coinbase_quality_map.py::_is_no_files."""
    return (type(exc).__name__ == "NoFilesFound"
            or "nofilesfound" in repr(exc).lower()
            or "no files found" in str(exc).lower())


def _load_error_record(venue: str, day: str, cs: dict, err: BaseException) -> dict:
    """A schema-consistent `load_error` per-day record, built from the pure module's OWN record
    builders (`_route_schema_fail`/`_record`/`_empty_metrics`) so it is identical in shape to every
    other record and honours the calendar route (required → fail; fill → coinapi_fill; excluded →
    excluded). Reused, not re-implemented, to keep the report schema authoritative in trade_checks."""
    route = cs.get("route", tc.ROUTE_REQUIRED)
    status, reasons = tc._route_schema_fail(
        [tc.LOAD_ERROR, f"{tc.LOAD_ERROR}:{type(err).__name__}"], route)
    return tc._record(day=day, venue=venue, status=status, reason_codes=reasons,
                      metrics=tc._empty_metrics(), calendar_state=cs, vendor_source="lake")


# --------------------------------------------------------------------------- validation core
def _validate_pairs(venues, days, cal, *, sess, load_fn, load_calls: list) -> list[dict]:
    """Cross each `(venue, day)` with the calendar and validate it. Required days are LOADED
    (via the injected `load_fn`) and metric-classified; fill/excluded days route via the calendar
    WITHOUT a load (only required days spend quota). Every record is the pure module's
    `validate_trade_frame` output (or the schema-identical `_load_error_record` on an unexpected
    load exception). Per-day order is `days × venues` (canonical venue order) for determinism."""
    records: list[dict] = []
    for day in days:
        for venue in venues:
            cs = tc.calendar_state(cal, day, venue)
            if cs.get("route") != tc.ROUTE_REQUIRED:
                # Fill / excluded day: the pure module routes df=None to coinapi_fill / excluded —
                # no Lake load (the missing/deferred side is expected, §8).
                records.append(tc.validate_trade_frame(None, venue, day, calendar_state=cs))
                continue
            try:
                df = load_fn(sess, venue, day)
            except Exception as e:                   # noqa: BLE001 — surface, never crash the run
                if _is_no_files(e):
                    df = None                        # absent partition → missing_partition (§1)
                else:
                    records.append(_load_error_record(venue, day, cs, e))
                    continue
            load_calls.append((venue, day))
            records.append(tc.validate_trade_frame(df, venue, day, calendar_state=cs))
    return records


def _print_plan(*, days, venues, mode, decision, cal_path, cal, dry_run) -> None:
    """Print the resolved plan (days × venues, est GB, quota decision, per-(venue,day) routes) —
    the shared preamble for both --dry-run and a live run."""
    est = decision["est_gb"]
    print("=" * 74)
    print(f"  TRADE FEED VALIDATION — {len(days)} day(s) × {len(venues)} venue(s)  "
          f"[{mode}]{'  (dry-run)' if dry_run else ''}")
    print("=" * 74)
    print(f"  venues: {', '.join(venues)}")
    print(f"  days:   {', '.join(days)}")
    print(f"  est Crypto Lake trades download: ~{est:.2f} GB "
          f"(conservative upper bound; fill/excluded days not loaded)")
    print(f"  quota decision: ok={decision['ok']} reason={decision['reason']} "
          f"(used {decision['used_gb']:.1f} GB, max_auto {decision['max_auto_gb']:.0f} GB, "
          f"allow_broad={decision['allow_broad']})")
    if cal is None:
        print(f"  NOTE: usable calendar {cal_path} not found — fill/excluded routing OFF; every "
              "day treated as a required Lake day.", file=sys.stderr)
    else:
        n_fill = n_excl = 0
        for day in days:
            for venue in venues:
                route = tc.calendar_state(cal, day, venue).get("route")
                n_fill += route == tc.ROUTE_COINAPI_FILL
                n_excl += route == tc.ROUTE_EXCLUDED
        print(f"  calendar routes: {n_fill} coinapi_fill, {n_excl} excluded "
              "(routed without a Lake load)")


# --------------------------------------------------------------------------- run
def run(args, *, load_fn=load_lake_trades, session_factory=lake_session,
        used_data_fn=lake_used_data, generated_utc: str | None = None) -> int:
    """Resolve the plan, gate the quota, (unless --dry-run) load + validate, write the report, and
    return the process exit code. Vendor access is confined to the three injected seams
    (`load_fn`/`session_factory`/`used_data_fn`); --dry-run touches none of them.

    Exit codes (§7): 0 ok; 5 (`QUOTA_REFUSED_EXIT`) a refused plan; 7 (`VALIDATION_FAILED_EXIT`)
    when --strict and ≥1 blocking fail. A blocking fail on a non-strict run still writes the report
    and exits 0 (the report is the artifact; --strict is the CI gate)."""
    venues = resolve_venues(args.venues)
    cal = tc.load_usable_calendar(args.calendar)
    days, mode = resolve_days(args, cal)
    if not days or not venues:
        # A selection that resolves to zero (venue, day) pairs must never emit a vacuously-green
        # report (n_days=0 → lake_required_pass/bars_ready both True, which Phase 4 would consume as
        # "buildable"). Refuse loudly instead — main() maps the ValueError to exit 2.
        raise ValueError(f"empty selection: no (venue, day) pairs to validate "
                         f"(days={days}, venues={venues})")
    est_gb = estimate_plan_gb(venues, days)

    # --- quota gate (§7) ------------------------------------------------------------------------
    # --dry-run reads NO vendor usage, so it gates optimistically at used_gb=0 (the auto-cap arm is
    # pure — a broad plan is still refused; the headroom arm can only trip live). A live run reads
    # the real monthly usage; a failure to read it fail-safes to the cap so the gate refuses.
    used_gb: float | None = 0.0
    sess = None
    used_after = None
    if not args.dry_run:
        sess = session_factory()
        try:
            used_gb = float(used_data_fn(sess).get("downloaded_gb", 0.0))
        except Exception as e:                       # noqa: BLE001 — cannot confirm headroom ⇒ refuse
            print(f"WARNING: could not read Crypto Lake used_data ({e!r}); assuming worst-case usage "
                  "at the monthly cap, so the quota gate will REFUSE this pull (cannot confirm "
                  "headroom). Re-run once used_data is readable.", file=sys.stderr)
            used_gb = float(args.quota_gb)

    decision = tc.quota_decision(est_gb=est_gb, used_gb=used_gb, quota_gb=args.quota_gb,
                                 max_auto_gb=args.max_auto_gb, allow_broad=args.allow_broad,
                                 headroom_gb=args.headroom_gb)
    _print_plan(days=days, venues=venues, mode=mode, decision=decision,
                cal_path=args.calendar, cal=cal, dry_run=args.dry_run)

    if not decision["ok"]:
        why = ("would breach the monthly quota headroom" if decision["reason"] == "quota_headroom"
               else f"exceeds the {args.max_auto_gb:.0f} GB auto cap")
        print(f"\nREFUSING Lake load: estimate ~{est_gb:.2f} GB {why} "
              f"(remaining ~{decision['remaining_gb']:.1f} GB of {args.quota_gb:.0f} GB).\n"
              "  • Narrow the day/venue set, or\n"
              "  • pass --allow-broad for a deliberate pull that still fits the monthly quota.",
              file=sys.stderr)
        return tc.QUOTA_REFUSED_EXIT

    if args.dry_run:
        print("\n  --dry-run: plan only, no Crypto Lake session created, no report written.")
        return 0

    # --- load + validate (required days only) ---------------------------------------------------
    load_calls: list = []
    records = _validate_pairs(venues, days, cal, sess=sess, load_fn=load_fn, load_calls=load_calls)
    try:
        used_after = float(used_data_fn(sess).get("downloaded_gb", 0.0))
    except Exception:                                # noqa: BLE001 — post-run usage is informational
        used_after = None

    if generated_utc is None:
        generated_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    meta = {
        "script": "ingest/validate_trade_feeds.py",
        "table": "trades",
        "anchor_end": args.end,
        "venues": venues,
        "days_requested": days,
        "days_selected": days,
        "selection_mode": mode,
        "seed": args.seed,
        "timestamp_policy": {"engine_clock": "origin_time", "fallback": "received_time",
                             "sort": "stable_by_engine_clock_then_file_order"},
        "thresholds": tc.THRESHOLDS.as_dict(),
        "trades_gb_per_day": tc.TRADES_GB_PER_DAY,
        "quota": {**decision, "used_gb_before": used_gb, "used_gb_after": used_after},
        "vendor_api": {"source": "crypto_lake", "region": "eu-west-1", "table": "trades",
                       "lakeapi_calls": len(load_calls), "dry_run": False, "coinapi_used": False},
        "source_artifacts": {"usable_calendar": args.calendar,
                             "usable_calendar_anchor_end": args.end},
        "generated_utc": generated_utc,
        "note": "VALIDATION only (docs/data.md §5b / §10 trade-validation breadth). Reads Crypto "
                "Lake trades; the CoinAPI backfill gate stays LOCKED and this tool does not unlock "
                "it.",
    }
    report = tc.build_report(records, meta=meta)
    out_path = os.path.join(args.out_dir, REPORT_NAME)
    tc.write_report(report, out_path)

    gate = report["summary"]["gate"]
    counts = report["summary"]["counts"]
    print(f"\n  wrote {out_path}")
    print("  statuses: " + ", ".join(f"{s}={counts.get(s, 0)}" for s in tc.STATUSES))
    print(f"  gate: lake_required_pass={gate['lake_required_pass']} "
          f"bars_ready={gate['bars_ready']} "
          f"({len(gate['blocking_failures'])} blocking, "
          f"{len(gate['coinapi_fill_deferred'])} coinapi_fill deferred)")
    if used_after is not None and used_gb is not None:
        print(f"  Crypto Lake usage: {used_gb:.2f} → {used_after:.2f} GB "
              f"(Δ ~{used_after - used_gb:+.2f} GB; may lag ~60 min).")

    if args.strict and gate["blocking_failures"]:
        print(f"\n  --strict: {len(gate['blocking_failures'])} blocking fail(s) → exit "
              f"{tc.VALIDATION_FAILED_EXIT}.", file=sys.stderr)
        return tc.VALIDATION_FAILED_EXIT
    return 0


# --------------------------------------------------------------------------- argparse / main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Thin Crypto Lake `trades` validator over the pure ingest.trade_checks module "
                    "(VALIDATION, not a backfill; does not unlock the CoinAPI backfill gate).")
    ap.add_argument("--start", default=None,
                    help="range start YYYY-MM-DD for cohort + stratified-sample selection (§3)")
    ap.add_argument("--end", default=DEFAULT_END,
                    help=f"range end / anchor YYYY-MM-DD (the END= convention; default {DEFAULT_END})")
    ap.add_argument("--days", default=None,
                    help="explicit days, CSV YYYY-MM-DD,YYYY-MM-DD (highest precedence)")
    ap.add_argument("--days-file", default=None,
                    help="file of days (CSV and/or one-per-line); precedence below --days")
    ap.add_argument("--venues", default=None,
                    help="CSV subset of binance_perp,binance_spot,coinbase (default: all three)")
    ap.add_argument("--sample-n", type=int, default=0,
                    help="extra deterministic stratified-random usable days for --start/--end (§3)")
    ap.add_argument("--seed", type=int, default=0, help="seed for the stratified sample (default 0)")
    ap.add_argument("--calendar", "--usable-calendar", dest="calendar", default=DEFAULT_CALENDAR,
                    help="usable-calendar JSON for fill/excluded routing (§8; default "
                         f"{DEFAULT_CALENDAR})")
    ap.add_argument("--max-auto-gb", type=float, default=tc.DEFAULT_MAX_AUTO_GB,
                    help=f"auto-allowed est-GB cap; larger needs --allow-broad (default "
                         f"{tc.DEFAULT_MAX_AUTO_GB})")
    ap.add_argument("--quota-gb", type=float, default=tc.QUOTA_GB,
                    help=f"Crypto Lake monthly download cap GB (default {tc.QUOTA_GB:.0f})")
    ap.add_argument("--headroom-gb", type=float, default=tc.DEFAULT_HEADROOM_GB,
                    help=f"quota GB always left unused (default {tc.DEFAULT_HEADROOM_GB:.0f})")
    ap.add_argument("--allow-broad", action="store_true",
                    help="override the auto cap (still refused if it breaches quota headroom)")
    ap.add_argument("--out-dir", default="data/reports",
                    help=f"report dir (git-ignored); writes {REPORT_NAME} (default data/reports)")
    ap.add_argument("--strict", action="store_true",
                    help=f"exit {tc.VALIDATION_FAILED_EXIT} on any blocking fail (CI gate); default "
                         "writes the report and exits 0")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the resolved plan (days × venues, est GB, quota decision) and write "
                         "nothing — NO vendor calls")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return run(args)
    except ValueError as e:                          # bad --venues / malformed day → clean nonzero
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
