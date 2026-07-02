"""Deterministic quota-budgeted batch planner for the Binance Crypto Lake archive pull
(docs/superpowers/plans/2026-07-02-binance-downloader-plan.md, Requirement 7/8 / Task 6).

The 12-24 mo Binance-only pull (output feeds + the `book` seed) is ~671 GB (~1.23 GB/day x 547 d)
> two 300 GB/month quota windows, so it CANNOT be one pull — it must be staged across ~3 windows.
This tool splits a day source (a plain --start/--end range, or a --calendar day-list JSON) into
deterministic, budget-sized `batch_NNN_days.txt` files for `ingest/download_lake_binance.py`, one
batch per monthly quota window.

PLANNING ONLY — reads at most one local calendar JSON and writes text/JSON plans under an ignored
path. It performs NO vendor I/O (no Crypto Lake session, no lakeapi/boto3/pyarrow), starts no
download, and consumes no quota. Each generated batch still goes through the downloader's own
broad-pull / quota-headroom gate (ingest.lake_binance.check_broad_gate) when actually executed.

Stdlib-only on purpose (mirrors scripts/plan_coinbase_quality_map_batches.py): importing the pure
helper module ingest.lake_binance pulls pandas, so the downloader's quota-gate constants are pinned
here as copies, kept aligned to ingest.lake_binance.full_pull_gb_per_day() by a contract test
(tests/test_plan_lake_binance_batches.py::test_runner_gate_constants_match_lake_binance).

Usage:
  .venv/bin/python scripts/plan_lake_binance_batches.py --start 2025-01-01 --end 2026-06-30
  .venv/bin/python scripts/plan_lake_binance_batches.py --calendar data/usable_calendar.json --dry-run
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import pathlib
import sys

DEFAULT_OUT_DIR = "data/tmp/binance_download_batches"
# One batch per monthly quota window with ~50 GB left for probes/parity (cap 300 GB/month, docs §2.1).
DEFAULT_MAX_GB_PER_BATCH = 250.0

# Pinned COPIES of the downloader's quota-gate constants (see module docstring). The rate is a FULL
# pull of both instruments, all feeds + the `book` seed == ingest.lake_binance.full_pull_gb_per_day();
# a contract test keeps these aligned to that source of truth.
RUNNER_GB_PER_DAY = 1.2278   # == full_pull_gb_per_day() (docs §6 derived ~1.23 GB/day)
RUNNER_QUOTA_GB = 300.0      # == lake_binance.QUOTA_GB
RUNNER_HEADROOM_GB = 10.0    # == lake_binance.DEFAULT_HEADROOM_GB
DEFAULT_GB_PER_DAY = RUNNER_GB_PER_DAY

# Both in-scope instruments are pulled per day (pinned copy of ingest.lake_binance.INSTRUMENTS keys).
INSTRUMENT_KEYS = ("binance-perp", "binance-spot")
# ONE invocation per batch, both instruments, driven by the exact days-file (not a --start/--end
# range). See batch_command for why this is the quota-safe shape.
COMMAND_TEMPLATE = (".venv/bin/python ingest/download_lake_binance.py "
                    "--instrument {instruments} --days-file {days_file} --allow-broad")

DEFAULT_CALENDAR_FIELD = "binance_present_days"
MANIFEST_NAME = "manifest.json"
BATCH_GLOB = "batch_*_days.txt"
CALENDAR_ERROR_EXIT = 2


class PlanError(ValueError):
    """A clear, user-actionable planning failure (bad range/calendar/budget)."""


def batch_file_name(n: int) -> str:
    return f"batch_{n:03d}_days.txt"


# ----------------------------------------------------------------------------- day sources
def _parse_day(s: str, *, what: str) -> dt.date:
    try:
        return dt.date.fromisoformat(s)
    except (TypeError, ValueError):
        raise PlanError(f"{what} {s!r} is not a valid YYYY-MM-DD date") from None


def daterange(start_iso: str, end_iso: str) -> list[str]:
    """Inclusive sorted ISO day list for a [start, end] range. Fails clearly on bad/reversed dates."""
    start = _parse_day(start_iso, what="--start")
    end = _parse_day(end_iso, what="--end")
    if end < start:
        raise PlanError(f"--end {end_iso} is before --start {start_iso}")
    days, d = [], start
    while d <= end:
        days.append(d.isoformat())
        d += dt.timedelta(days=1)
    return days


def load_calendar_days(path: str, field: str) -> list[str]:
    """Sorted ISO day list from a calendar JSON's `field` (a list of YYYY-MM-DD). Fails clearly on a
    missing file, bad JSON, a missing/non-list field, or an invalid day — a silent default here could
    plan a quota-sized pull off the wrong day set."""
    if not os.path.exists(path):
        raise PlanError(f"calendar not found: {path}")
    with open(path) as f:
        try:
            cal = json.load(f)
        except json.JSONDecodeError as e:
            raise PlanError(f"calendar {path} is not valid JSON: {e}") from None
    if not isinstance(cal, dict) or field not in cal:
        raise PlanError(f"calendar {path} is missing required field {field!r} "
                        "(a list of YYYY-MM-DD Binance-present days)")
    days = cal[field]
    if not isinstance(days, list):
        raise PlanError(f"calendar field {field!r} must be a list of YYYY-MM-DD strings, "
                        f"got {type(days).__name__}")
    for d in days:
        _parse_day(d, what=f"calendar field {field!r} day")
    return sorted(days)


def resolve_days(args) -> list[str]:
    """Sorted day list from --calendar (if given) else --start/--end. Exactly one source required."""
    if args.calendar:
        return load_calendar_days(args.calendar, args.calendar_field)
    if args.start and args.end:
        return daterange(args.start, args.end)
    raise PlanError("provide either --start and --end (a day range) or --calendar (a day-list JSON)")


# ----------------------------------------------------------------------------- batching
def days_per_batch(max_gb_per_batch: float, gb_per_day: float) -> tuple[int, bool]:
    """Days per batch under the user budget, capped at what the downloader's quota gate can ever
    accept (floor((300 - 10) / 1.2278) = 236 days). Returns (days, capped). A tiny epsilon keeps an
    exact-multiple budget from being knocked down by float noise; otherwise floor() errs toward FEWER
    days (the safe direction for a quota)."""
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
    """Chunk sorted `days` into downloader-executable batches. Deterministic; every batch estimate is
    <= the budget and <= the runner-gate cap by construction."""
    per_batch, _ = days_per_batch(max_gb_per_batch, gb_per_day)
    return [days[i:i + per_batch] for i in range(0, len(days), per_batch)]


def batch_command(days_file: str) -> str:
    """The single downloader invocation for a batch: BOTH instruments together over the EXACT day
    list in `days_file`.

    Combining instruments in ONE invocation lets the downloader estimate + gate the COMBINED request
    once. Per-instrument commands (`perp && spot`) would each gate against only their own estimate;
    because `used_data` lags ~60 min, the second could miss the first's transfer and a ~250 GB batch
    could breach the monthly quota/headroom (Codex P1). Passing the authoritative days-file (not
    --start/--end) runs exactly the batch's days, so a sparse `--calendar` batch never executes the
    absent days its enclosing range would span and the run matches `est_gb` (Codex P2)."""
    return COMMAND_TEMPLATE.format(instruments=",".join(INSTRUMENT_KEYS), days_file=days_file)


# ----------------------------------------------------------------------------- manifest
def build_manifest(batches: list[list[str]], *, day_source: str, out_dir: str,
                   max_gb_per_batch: float, gb_per_day: float,
                   generated_utc: str | None = None) -> dict:
    per_batch, capped = days_per_batch(max_gb_per_batch, gb_per_day)
    batch_rows = []
    for i, days in enumerate(batches, start=1):
        fname = batch_file_name(i)
        batch_rows.append({
            "file": fname,
            "n_days": len(days),
            "first_day": days[0],
            "last_day": days[-1],
            "est_gb": round(len(days) * gb_per_day, 2),
            "runner_est_gb": round(len(days) * RUNNER_GB_PER_DAY, 2),
            "command": batch_command(os.path.join(out_dir, fname)),
        })
    if generated_utc is None:
        generated_utc = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    return {
        "meta": {
            "day_source": day_source,
            "generated_utc": generated_utc,
            "gb_per_day": gb_per_day,
            "max_gb_per_batch": max_gb_per_batch,
            "days_per_batch": per_batch,
            "days_per_batch_capped_by_runner_gate": capped,
            "runner_gb_per_day": RUNNER_GB_PER_DAY,
            "out_dir": out_dir,
            "note": "PLANNING ONLY. No vendor I/O: this plans Lake batches, it does not download "
                    "anything and consumes no quota. Run at most one batch per monthly quota window "
                    "and re-check lakeapi.used_data before each run. Each command runs BOTH "
                    "instruments in ONE download_lake_binance.py invocation over the exact "
                    "batch_NNN_days.txt list (--days-file), so the downloader gates the COMBINED "
                    "estimate once and executes exactly the listed days — never the absent days a "
                    "sparse-calendar batch's enclosing range would span.",
        },
        "summary": {
            "n_batch_days": sum(len(b) for b in batches),
            "n_batches": len(batches),
            "est_gb_per_batch": [r["est_gb"] for r in batch_rows],
            "total_est_gb": round(sum(len(b) for b in batches) * gb_per_day, 2),
        },
        "batches": batch_rows,
    }


# ----------------------------------------------------------------------------- writing
def write_plan(batches: list[list[str]], manifest: dict, out_dir: str) -> list[str]:
    """Write batch files + manifest. Removes stale `batch_*_days.txt` from a previous plan first — a
    leftover higher-numbered batch would otherwise be runnable against the wrong day set."""
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
    print("  BINANCE LAKE DOWNLOAD BATCH PLAN — planning only, no vendor I/O"
          + ("  [DRY RUN]" if dry_run else ""))
    print("=" * 74)
    print(f"  source:    {m['day_source']}")
    print(f"  budget:    {m['max_gb_per_batch']:g} GB/batch at {m['gb_per_day']:g} GB/day "
          f"-> {m['days_per_batch']} day(s)/batch"
          + (" (capped by the downloader quota gate at "
             f"{m['runner_gb_per_day']:g} GB/day)" if m["days_per_batch_capped_by_runner_gate"]
             else ""))
    print(f"  days:      {s['n_batch_days']} day(s) in {s['n_batches']} batch(es), "
          f"~{s['total_est_gb']:g} GB total estimate")
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
        description="Plan deterministic quota-budgeted day batches for the Binance Lake archive pull "
                    "(planning only — no vendor I/O, no downloads).")
    ap.add_argument("--start", help="YYYY-MM-DD inclusive (with --end; a plain day range)")
    ap.add_argument("--end", help="YYYY-MM-DD inclusive (with --start)")
    ap.add_argument("--calendar", help="calendar JSON with a Binance-present day-list field "
                                       "(overrides --start/--end)")
    ap.add_argument("--calendar-field", default=DEFAULT_CALENDAR_FIELD,
                    help=f"calendar field holding the day list (default {DEFAULT_CALENDAR_FIELD})")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR,
                    help=f"output dir for batch files + manifest (git-ignored; default "
                         f"{DEFAULT_OUT_DIR})")
    ap.add_argument("--max-gb-per-batch", type=float, default=DEFAULT_MAX_GB_PER_BATCH,
                    help=f"planned Lake GB budget per batch (default {DEFAULT_MAX_GB_PER_BATCH:g})")
    ap.add_argument("--gb-per-day", type=float, default=DEFAULT_GB_PER_DAY,
                    help=f"per-day GB estimate, both instruments + `book` seed (default "
                         f"~{DEFAULT_GB_PER_DAY:g}; matches the downloader's quota estimator)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the batch plan without writing any files")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        days = resolve_days(args)
        if not days:
            raise PlanError("day source resolved to zero days — nothing to plan")
        batches = plan_batches(days, max_gb_per_batch=args.max_gb_per_batch,
                               gb_per_day=args.gb_per_day)
    except PlanError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return CALENDAR_ERROR_EXIT
    day_source = (f"calendar:{args.calendar}#{args.calendar_field}"
                  if args.calendar else f"range:{args.start}..{args.end}")
    manifest = build_manifest(batches, day_source=day_source, out_dir=args.out_dir,
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
