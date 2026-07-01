"""Deterministic quota-budgeted batch planner for the BROAD Coinbase quality map
(docs/data.md §5a-QualityMap staging / §6 quota constraint).

The full-window quality map over all present Coinbase Lake days is NOT a one-shot pull on the
individual plan (~652 days ≈ 313 GB at the conservative 0.48 GB/day estimate vs the 300 GB/month
cap — docs/data.md §5a-QualityMap / §6). This tool splits the usable-calendar's Coinbase-present
Lake days (`lake_all_days`) into deterministic, budget-sized `--days-file` batches for
`scripts/run_coinbase_quality_map.py`, so the broad map is staged across monthly quota windows
instead of run as one accidental pull.

PLANNING ONLY — reads one local calendar JSON and writes text/JSON plans under an ignored path.
It performs NO vendor I/O (no Crypto Lake session, no CoinAPI), starts no download, and does not
unlock the §5a CoinAPI backfill gate. Each generated batch still goes through the runner's own
quota gate (`lakeapi.used_data` + headroom) when actually executed.

Day categories (kept apart explicitly — the §5b calendar mixes them):
  * batch days              — `lake_all_days` (Coinbase book+trades present in Lake): the days the
                              broad map must actually download and reconstruct.
  * fill days (book gap)    — `coinbase_fill_days` with `book: true`: Lake book absent; they cost
                              ~0 GB and classify `missing_needs_coinapi` (map them separately via
                              the runner's `--include-gap-days`), so they are NOT batched here.
  * fill days (trade-only)  — `coinbase_fill_days` with `book: false`: outside `lake_all_days`
                              (trades gap) — recorded, not batched.
  * excluded days           — `excluded_days_by_reason` (non-Coinbase calendar exclusions, e.g. a
                              Binance gap): the runner skips them before any Lake load; batching
                              them would only waste quota.

Batch files are byte-deterministic for a given calendar + budget (sorted days, one per line); the
manifest additionally records a generation timestamp.

Usage:
  .venv/bin/python scripts/plan_coinbase_quality_map_batches.py                # plan + write files
  .venv/bin/python scripts/plan_coinbase_quality_map_batches.py --dry-run     # print plan only
  .venv/bin/python scripts/plan_coinbase_quality_map_batches.py \
      --max-gb-per-batch 250 --gb-per-day 0.48
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import sys

DEFAULT_CALENDAR = "data/usable_calendar.json"
DEFAULT_OUT_DIR = "data/tmp/coinbase_quality_map_batches"
# Current planning target: keep planned Lake downloads ~250 GB/month, i.e. one batch per monthly
# quota window with ~50 GB left for parity/smoke pulls (cap is 300 GB/month, docs §2.1/§8).
DEFAULT_MAX_GB_PER_BATCH = 250.0
# Conservative per-day estimate for Coinbase book_delta_v2 + book — matches the runner's
# LAKE_GB_PER_DAY sum (0.30 + 0.18, docs §6). Measured wire rate is lower (~0.26 GB/day,
# §5a-QualityMap); plan with the conservative number by default.
DEFAULT_GB_PER_DAY = 0.48

# How each planned batch is actually run (docs §5a-QualityMap staged workflow). --allow-broad is
# deliberate: a batch IS a broad pull; the runner's quota-headroom gate still applies.
COMMAND_TEMPLATE = (".venv/bin/python scripts/run_coinbase_quality_map.py "
                    "--engine native --no-cold-ab --days-file {days_file} --allow-broad")

MANIFEST_NAME = "manifest.json"
BATCH_GLOB = "batch_*_days.txt"
CALENDAR_ERROR_EXIT = 2


class PlanError(ValueError):
    """A clear, user-actionable planning failure (bad calendar file/fields or budget)."""


def batch_file_name(n: int) -> str:
    return f"batch_{n:03d}_days.txt"


# ----------------------------------------------------------------------------- calendar loading
def _require_days(field: str, days, *, allow_dict: bool) -> None:
    """Validate a calendar day collection: right container type, every key a real ISO date."""
    ok_type = isinstance(days, dict) if allow_dict else isinstance(days, list)
    if not ok_type:
        want = "a {day: ...} dict" if allow_dict else "a list of YYYY-MM-DD strings"
        raise PlanError(f"usable calendar field '{field}' must be {want}, got "
                        f"{type(days).__name__}")
    for d in days:
        try:
            dt.date.fromisoformat(d)
        except (TypeError, ValueError):
            raise PlanError(f"usable calendar field '{field}' has an invalid day {d!r} "
                            "(expected YYYY-MM-DD)") from None


def load_calendar(path: str) -> dict:
    """Load and validate the usable-calendar JSON (ingest/verify_trades_and_calendar.py output).
    Fails clearly (PlanError) on a missing file or missing/invalid required fields — a silent
    default here could plan a quota-sized pull off the wrong day set."""
    if not os.path.exists(path):
        raise PlanError(f"usable calendar not found: {path} "
                        "(run ingest/verify_trades_and_calendar.py to produce it)")
    with open(path) as f:
        try:
            cal = json.load(f)
        except json.JSONDecodeError as e:
            raise PlanError(f"usable calendar {path} is not valid JSON: {e}") from None
    if not isinstance(cal, dict):
        raise PlanError(f"usable calendar {path} must be a JSON object")
    for field, allow_dict in (("lake_all_days", False), ("coinbase_fill_days", True),
                              ("excluded_days_by_reason", True)):
        if field not in cal:
            raise PlanError(f"usable calendar {path} is missing required field '{field}'")
        _require_days(field, cal[field], allow_dict=allow_dict)
    # fill VALUES drive the book-gap/trade-only split — a corrupted entry must fail here, not
    # crash later in select_days
    for d, v in cal["coinbase_fill_days"].items():
        if not isinstance(v, dict):
            raise PlanError(f"usable calendar field 'coinbase_fill_days' entry {d} must be a "
                            f"{{'book': bool, 'trades': bool}} dict, got {type(v).__name__}")
    return cal


# ----------------------------------------------------------------------------- day selection
def select_days(cal: dict) -> dict:
    """Split the calendar into the day categories the plan keeps apart. Pure and deterministic:
    same calendar ⇒ same (sorted) day lists."""
    fill = cal["coinbase_fill_days"]
    excluded = cal["excluded_days_by_reason"]
    lake_all = set(cal["lake_all_days"])
    not_batchable = set(fill) | set(excluded)
    # Defensive: in a consistent calendar lake_all_days is disjoint from fill/excluded days; if a
    # contradictory calendar overlaps them, surface the overlap instead of silently batching it.
    dropped = sorted(lake_all & not_batchable)
    return {
        "batch_days": sorted(lake_all - not_batchable),
        "fill_days_book_gap": sorted(d for d, v in fill.items() if (v or {}).get("book") is True),
        "fill_days_trade_only": sorted(d for d, v in fill.items()
                                       if (v or {}).get("book") is not True),
        "excluded_days": {d: excluded[d] for d in sorted(excluded)},
        "lake_days_dropped_as_excluded_or_fill": dropped,
    }


# ----------------------------------------------------------------------------- batching
def plan_batches(days: list[str], *, max_gb_per_batch: float, gb_per_day: float) -> list[list[str]]:
    """Chunk sorted `days` into batches of floor(max_gb_per_batch / gb_per_day) days each.
    Deterministic; every batch estimate is ≤ the budget by construction."""
    if not (math.isfinite(gb_per_day) and gb_per_day > 0):
        raise PlanError(f"--gb-per-day must be a positive number, got {gb_per_day}")
    if not (math.isfinite(max_gb_per_batch) and max_gb_per_batch > 0):
        raise PlanError(f"--max-gb-per-batch must be a positive number, got {max_gb_per_batch}")
    # tiny epsilon so an exact-multiple budget (0.96/0.48) is not knocked down by float noise;
    # otherwise floor() errs toward FEWER days per batch (the safe direction for a quota).
    per_batch = int((max_gb_per_batch / gb_per_day) + 1e-9)
    if per_batch < 1:
        raise PlanError(f"--max-gb-per-batch {max_gb_per_batch} GB does not allow even one day at "
                        f"{gb_per_day} GB/day — the budget must allow at least one day per batch")
    return [days[i:i + per_batch] for i in range(0, len(days), per_batch)]


# ----------------------------------------------------------------------------- manifest
def build_manifest(sel: dict, batches: list[list[str]], *, calendar_path: str, out_dir: str,
                   max_gb_per_batch: float, gb_per_day: float,
                   generated_utc: str | None = None) -> dict:
    per_batch = int((max_gb_per_batch / gb_per_day) + 1e-9)
    batch_rows = []
    for i, days in enumerate(batches, start=1):
        fname = batch_file_name(i)
        batch_rows.append({
            "file": fname,
            "n_days": len(days),
            "first_day": days[0],
            "last_day": days[-1],
            "est_gb": round(len(days) * gb_per_day, 2),
            "command": COMMAND_TEMPLATE.format(days_file=os.path.join(out_dir, fname)),
        })
    if generated_utc is None:
        generated_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return {
        "meta": {
            "input_calendar": calendar_path,
            "generated_utc": generated_utc,
            "gb_per_day": gb_per_day,
            "max_gb_per_batch": max_gb_per_batch,
            "days_per_batch": per_batch,
            "out_dir": out_dir,
            "command_template": COMMAND_TEMPLATE,
            "note": "PLANNING ONLY (docs/data.md §5a-QualityMap staging). No vendor I/O: this "
                    "plans Lake batches, it does not download anything and does not unlock the "
                    "§5a CoinAPI backfill gate. Run at most one batch per monthly quota window "
                    "and re-check lakeapi.used_data before each run.",
        },
        "summary": {
            "n_batch_days": sum(len(b) for b in batches),
            "n_batches": len(batches),
            "est_gb_per_batch": [r["est_gb"] for r in batch_rows],
            "total_est_gb": round(sum(len(b) for b in batches) * gb_per_day, 2),
            "n_fill_days_book_gap": len(sel["fill_days_book_gap"]),
            "n_fill_days_trade_only": len(sel["fill_days_trade_only"]),
            "n_excluded_days": len(sel["excluded_days"]),
        },
        "batches": batch_rows,
        "skipped": {
            "fill_days_book_gap": sel["fill_days_book_gap"],
            "fill_days_trade_only": sel["fill_days_trade_only"],
            "excluded_days_by_reason": sel["excluded_days"],
            "lake_days_dropped_as_excluded_or_fill":
                sel["lake_days_dropped_as_excluded_or_fill"],
        },
    }


# ----------------------------------------------------------------------------- writing
def write_plan(batches: list[list[str]], manifest: dict, out_dir: str) -> list[str]:
    """Write batch files + manifest. Removes stale `batch_*_days.txt` from a previous plan first —
    a leftover higher-numbered batch would otherwise be runnable against the wrong day set."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for stale in out.glob(BATCH_GLOB):
        stale.unlink()
    written = []
    for i, days in enumerate(batches, start=1):
        path = out / batch_file_name(i)
        path.write_text("".join(f"{d}\n" for d in days))
        written.append(str(path))
    manifest_path = out / MANIFEST_NAME
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")
    written.append(str(manifest_path))
    return written


# ----------------------------------------------------------------------------- reporting (stdout)
def print_plan(manifest: dict, *, dry_run: bool) -> None:
    m, s = manifest["meta"], manifest["summary"]
    print("=" * 74)
    print("  COINBASE QUALITY-MAP BATCH PLAN — planning only, no vendor I/O"
          + ("  [DRY RUN]" if dry_run else ""))
    print("=" * 74)
    print(f"  calendar:  {m['input_calendar']}")
    print(f"  budget:    {m['max_gb_per_batch']:g} GB/batch at {m['gb_per_day']:g} GB/day "
          f"(conservative) -> {m['days_per_batch']} day(s)/batch")
    print(f"  days:      {s['n_batch_days']} present Lake day(s) to map in {s['n_batches']} "
          f"batch(es), ~{s['total_est_gb']:g} GB total estimate")
    print(f"  skipped:   {s['n_fill_days_book_gap']} book-gap fill day(s) (map via "
          f"--include-gap-days, ~0 GB), {s['n_fill_days_trade_only']} trade-only fill day(s), "
          f"{s['n_excluded_days']} excluded day(s)")
    dropped = manifest["skipped"]["lake_days_dropped_as_excluded_or_fill"]
    if dropped:
        print(f"  WARNING:   {len(dropped)} lake_all_days overlapped fill/excluded days and were "
              f"dropped: {', '.join(dropped[:6])}{' …' if len(dropped) > 6 else ''}")
    for r in manifest["batches"]:
        print(f"    {r['file']}  {r['n_days']:>4} day(s)  {r['first_day']} .. {r['last_day']}  "
              f"~{r['est_gb']:g} GB")
    print("  run one batch per monthly quota window, e.g.:")
    print(f"    {manifest['batches'][0]['command'] if manifest['batches'] else '(no batches)'}")
    if dry_run:
        print("  DRY RUN — no files written.")


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Plan deterministic quota-budgeted day batches for the broad Coinbase "
                    "quality map (planning only — no vendor I/O, no downloads; see "
                    "docs/data.md §5a-QualityMap)")
    ap.add_argument("--calendar", default=DEFAULT_CALENDAR,
                    help=f"usable-calendar JSON (default {DEFAULT_CALENDAR})")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help=f"output dir for batch files + manifest (git-ignored; default "
                         f"{DEFAULT_OUT_DIR})")
    ap.add_argument("--max-gb-per-batch", type=float, default=DEFAULT_MAX_GB_PER_BATCH,
                    help=f"planned Lake GB budget per batch (default {DEFAULT_MAX_GB_PER_BATCH:g}"
                         " — one batch per monthly quota window with headroom)")
    ap.add_argument("--gb-per-day", type=float, default=DEFAULT_GB_PER_DAY,
                    help=f"conservative per-day estimate, Coinbase book_delta_v2+book (default "
                         f"{DEFAULT_GB_PER_DAY} — docs §6; matches the runner's quota estimator)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the batch plan without writing any files")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        cal = load_calendar(args.calendar)
        sel = select_days(cal)
        if not sel["batch_days"]:
            raise PlanError(f"usable calendar {args.calendar} has no batchable present Lake days "
                            "(lake_all_days minus fill/excluded days is empty)")
        batches = plan_batches(sel["batch_days"], max_gb_per_batch=args.max_gb_per_batch,
                               gb_per_day=args.gb_per_day)
    except PlanError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return CALENDAR_ERROR_EXIT
    manifest = build_manifest(sel, batches, calendar_path=args.calendar, out_dir=args.out_dir,
                              max_gb_per_batch=args.max_gb_per_batch, gb_per_day=args.gb_per_day)
    if not args.dry_run:
        written = write_plan(batches, manifest, args.out_dir)
        print_plan(manifest, dry_run=False)
        print(f"  wrote {len(written)} file(s) under {args.out_dir}")
    else:
        print_plan(manifest, dry_run=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
