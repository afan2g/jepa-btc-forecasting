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
  * batch days              — `lake_all_days` (Coinbase book+trades present in Lake) PLUS the
                              trade-only fill days below: every day whose Lake `book_delta_v2` is
                              present and must be downloaded and reconstructed by the broad map.
  * fill days (book gap)    — `coinbase_fill_days` with `book: true`: Lake book absent; they cost
                              ~0 GB and classify `missing_needs_coinapi` (map them separately via
                              the runner's `--include-gap-days`), so they are NOT batched here.
  * fill days (trade-only)  — `coinbase_fill_days` with `book: false`: only trades are gapped, the
                              Lake BOOK is present — batched (its reconstruction quality still
                              needs validating before the backfill gate) and listed separately in
                              the manifest because the day still needs a CoinAPI TRADES fill.
  * excluded days           — `excluded_days_by_reason` (non-Coinbase calendar exclusions, e.g. a
                              Binance gap): the runner skips them before any Lake load; batching
                              them would only waste quota. Exclusion wins over every other
                              category.

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
# deliberate: a batch IS a broad pull; the runner's quota-headroom gate still applies. The command
# pins {calendar} (the runner would otherwise default to data/usable_calendar.json — a plan built
# from another snapshot must run against the SAME excluded/fill context) and a per-batch
# {report_dir} (the runner writes a fixed coinbase_quality_map.json under --out-dir, so staged
# batches would otherwise overwrite each other's artifact).
COMMAND_TEMPLATE = (".venv/bin/python scripts/run_coinbase_quality_map.py "
                    "--engine native --no-cold-ab --days-file {days_file} "
                    "--usable-calendar {calendar} --out-dir {report_dir} --allow-broad")
REPORT_ROOT = "data/reports/coinbase_quality_map_batches"

# The runner re-estimates every request at ITS fixed conservative rate and refuses, REGARDLESS of
# --allow-broad, anything estimated over quota − headroom — so an emitted command is only
# executable up to floor((300 − 10) / 0.48) = 604 days. Batch sizing is capped there even when
# planning with a lower (e.g. measured-wire-rate) --gb-per-day. The planner must stay stdlib-only
# (importing the runner pulls in pandas), so these are pinned copies of the runner's constants,
# kept aligned by a contract test.
RUNNER_GB_PER_DAY = 0.48   # == sum(run_coinbase_quality_map.LAKE_GB_PER_DAY.values())
RUNNER_QUOTA_GB = 300.0    # == run_coinbase_quality_map.QUOTA_GB
RUNNER_HEADROOM_GB = 10.0  # == run_coinbase_quality_map.DEFAULT_HEADROOM_GB

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
    # fill VALUES drive the book-gap vs trade-only (BATCHED) split off `book is True` — flags must
    # be strict bools, or a corrupted entry (e.g. "book": "true", or missing flags) would silently
    # route a book-gap day into a Lake download batch instead of failing here
    for d, v in cal["coinbase_fill_days"].items():
        if not isinstance(v, dict) or not all(isinstance(v.get(k), bool)
                                              for k in ("book", "trades")):
            raise PlanError(f"usable calendar field 'coinbase_fill_days' entry {d} must be a "
                            f"{{'book': bool, 'trades': bool}} dict, got {v!r}")
    return cal


# ----------------------------------------------------------------------------- day selection
def select_days(cal: dict) -> dict:
    """Split the calendar into the day categories the plan keeps apart. Pure and deterministic:
    same calendar ⇒ same (sorted) day lists.

    Batchable = every day with a PRESENT Lake book: `lake_all_days` plus the trade-only fill days
    (`book: false` = only trades gapped) — a trade-only day's book quality still needs validating
    before the backfill gate. Withheld: book-gap fill days (`book: true`, nothing to map) and
    excluded days (exclusion wins over every other category)."""
    fill = cal["coinbase_fill_days"]
    excluded = cal["excluded_days_by_reason"]
    lake_all = set(cal["lake_all_days"])
    book_gap = {d for d, v in fill.items() if (v or {}).get("book") is True}
    trade_only = set(fill) - book_gap
    excluded_set = set(excluded)
    # Defensive: in a consistent calendar lake_all_days is disjoint from book-gap/excluded days; if
    # a contradictory calendar overlaps them, surface the overlap instead of silently batching it.
    dropped = sorted(lake_all & (book_gap | excluded_set))
    return {
        "batch_days": sorted((lake_all | trade_only) - book_gap - excluded_set),
        "fill_days_book_gap": sorted(book_gap),
        "fill_days_trade_only_batched": sorted(trade_only - excluded_set),
        "excluded_days": {d: excluded[d] for d in sorted(excluded)},
        "days_dropped_as_excluded_or_book_gap": dropped,
    }


# ----------------------------------------------------------------------------- batching
def days_per_batch(max_gb_per_batch: float, gb_per_day: float) -> tuple[int, bool]:
    """Days per batch under the user budget, capped at what the RUNNER's quota gate can ever
    accept. Returns (days, capped). Tiny epsilon so an exact-multiple budget (0.96/0.48) is not
    knocked down by float noise; otherwise floor() errs toward FEWER days per batch (the safe
    direction for a quota)."""
    if not (math.isfinite(gb_per_day) and gb_per_day > 0):
        raise PlanError(f"--gb-per-day must be a positive number, got {gb_per_day}")
    if not (math.isfinite(max_gb_per_batch) and max_gb_per_batch > 0):
        raise PlanError(f"--max-gb-per-batch must be a positive number, got {max_gb_per_batch}")
    requested = int((max_gb_per_batch / gb_per_day) + 1e-9)
    runner_cap = int((RUNNER_QUOTA_GB - RUNNER_HEADROOM_GB) / RUNNER_GB_PER_DAY + 1e-9)
    if requested < 1:
        raise PlanError(f"--max-gb-per-batch {max_gb_per_batch} GB does not allow even one day at "
                        f"{gb_per_day} GB/day — the budget must allow at least one day per batch")
    return min(requested, runner_cap), requested > runner_cap


def plan_batches(days: list[str], *, max_gb_per_batch: float, gb_per_day: float) -> list[list[str]]:
    """Chunk sorted `days` into runner-executable batches (see days_per_batch).
    Deterministic; every batch estimate is ≤ the budget by construction."""
    per_batch, _ = days_per_batch(max_gb_per_batch, gb_per_day)
    return [days[i:i + per_batch] for i in range(0, len(days), per_batch)]


# ----------------------------------------------------------------------------- manifest
def build_manifest(sel: dict, batches: list[list[str]], *, calendar_path: str, out_dir: str,
                   max_gb_per_batch: float, gb_per_day: float,
                   generated_utc: str | None = None) -> dict:
    per_batch, capped = days_per_batch(max_gb_per_batch, gb_per_day)
    batch_rows = []
    for i, days in enumerate(batches, start=1):
        fname = batch_file_name(i)
        report_dir = os.path.join(REPORT_ROOT, fname.removesuffix("_days.txt"))
        batch_rows.append({
            "file": fname,
            "n_days": len(days),
            "first_day": days[0],
            "last_day": days[-1],
            "est_gb": round(len(days) * gb_per_day, 2),
            # what the RUNNER's own quota gate will estimate this batch at (its fixed rate)
            "runner_est_gb": round(len(days) * RUNNER_GB_PER_DAY, 2),
            "report_dir": report_dir,
            "command": COMMAND_TEMPLATE.format(days_file=os.path.join(out_dir, fname),
                                               calendar=calendar_path, report_dir=report_dir),
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
            "days_per_batch_capped_by_runner_gate": capped,
            "runner_gb_per_day": RUNNER_GB_PER_DAY,
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
            "n_fill_days_trade_only_batched": len(sel["fill_days_trade_only_batched"]),
            "n_excluded_days": len(sel["excluded_days"]),
        },
        "batches": batch_rows,
        # batched (book present) but still needing a CoinAPI TRADES fill — for fill planning
        "batched_trade_only_fill_days": sel["fill_days_trade_only_batched"],
        "skipped": {
            "fill_days_book_gap": sel["fill_days_book_gap"],
            "excluded_days_by_reason": sel["excluded_days"],
            "days_dropped_as_excluded_or_book_gap":
                sel["days_dropped_as_excluded_or_book_gap"],
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
          f"(conservative) -> {m['days_per_batch']} day(s)/batch"
          + (" (capped by the runner quota gate: it re-estimates at "
             f"{m['runner_gb_per_day']:g} GB/day)" if m["days_per_batch_capped_by_runner_gate"]
             else ""))
    print(f"  days:      {s['n_batch_days']} present-book day(s) to map in {s['n_batches']} "
          f"batch(es), ~{s['total_est_gb']:g} GB total estimate "
          f"(includes {s['n_fill_days_trade_only_batched']} trade-only fill day(s) — book "
          "present, trades still need CoinAPI fill)")
    print(f"  skipped:   {s['n_fill_days_book_gap']} book-gap fill day(s) (map via "
          f"--include-gap-days, ~0 GB), {s['n_excluded_days']} excluded day(s)")
    dropped = manifest["skipped"]["days_dropped_as_excluded_or_book_gap"]
    if dropped:
        print(f"  WARNING:   {len(dropped)} lake_all_days overlapped book-gap/excluded days and "
              f"were dropped: {', '.join(dropped[:6])}{' …' if len(dropped) > 6 else ''}")
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
