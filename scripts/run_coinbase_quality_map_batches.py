"""Safe, resumable batch orchestrator for the BROAD Coinbase quality map
(docs/data.md §5a-QualityMap staging / §6 quota).

Consumes the plan manifest written by `scripts/plan_coinbase_quality_map_batches.py`
(`data/tmp/coinbase_quality_map_batches/manifest.json`) and drives the sibling per-batch runner
`scripts/run_coinbase_quality_map.py` ONE batch at a time, with resume/skip for already-completed
batches and a single status/aggregate artifact.

WHAT THIS DOES NOT DO — it does not, by itself, unlock the §5a CoinAPI backfill gate, and running it
without `--execute` performs NO vendor I/O at all. It never re-implements the runner's Lake/quota
logic: each batch is launched as the exact command the planner emitted (only the Python interpreter
is swapped to the running one), so every quota/headroom/`--allow-broad` safeguard the runner enforces
still applies to every batch. The broad full-window quality map remains the gate; this tool just
stages it safely across monthly quota windows.

Modes:
  * default (no flags)  — STATUS + AGGREGATE. Read the manifest and any local batch reports, classify
                          every batch (complete / pending / failed / blocked_quota), write
                          `summary.json`, and print the status. No vendor I/O, no ledger writes.
  * --dry-run           — PREVIEW. Print the exact command `--execute` would run for the next pending
                          batch(es) and exit. Writes nothing. Overrides --execute (a safety belt).
  * --execute           — RUN. Launch the next pending batch(es) as real subprocesses (LIVE Crypto
                          Lake I/O, gated by the runner's own quota gate). Bounded by --max-batches
                          (default 1: one batch per monthly quota window, docs §5a). Stops the moment
                          a batch is refused by the quota headroom. Appends to the status ledger and
                          rewrites `summary.json`. NOTE: the per-batch quota gate reads
                          `lakeapi.used_data`, which lags ~60 min, so it is only reliable at ONE broad
                          batch per window — running --max-batches > 1 back-to-back can breach the
                          300 GB/month cap because an earlier batch's pull is not yet reflected (the
                          tool warns loudly in that case; prefer the default of 1).

A batch is COMPLETE iff its runner wrote `coinbase_quality_map.json` under its report dir (the runner
writes that file only on a clean exit 0; a quota-refused run writes nothing), so completion is
detectable on disk and re-runs skip finished batches.

Usage:
  # 1) plan (separate tool, no vendor I/O):
  .venv/bin/python scripts/plan_coinbase_quality_map_batches.py
  # 2) inspect status (safe, no vendor I/O):
  .venv/bin/python scripts/run_coinbase_quality_map_batches.py
  # 3) preview the next batch command (safe):
  .venv/bin/python scripts/run_coinbase_quality_map_batches.py --dry-run
  # 4) run ONE batch this quota window (LIVE Lake download, still quota-gated):
  .venv/bin/python scripts/run_coinbase_quality_map_batches.py --execute
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shlex
import subprocess
import sys

# manifest + runner contract (kept in one place; a contract test pins the runner exit codes)
DEFAULT_MANIFEST = "data/tmp/coinbase_quality_map_batches/manifest.json"
STATUS_ROOT = "data/reports/coinbase_quality_map_batches"   # == planner REPORT_ROOT
RUNNER_SCRIPT = "scripts/run_coinbase_quality_map.py"
REPORT_NAME = "coinbase_quality_map.json"  # fixed filename the runner writes under its --out-dir
STATUS_LEDGER_NAME = "_runner_status.jsonl"  # append-only attempt ledger (cf. ingest _manifest.jsonl)
SUMMARY_NAME = "summary.json"

# orchestrator exit codes (small-int convention: planner calendar-err 2, parity 3, gate 4, quota 5)
MANIFEST_ERROR_EXIT = 2   # missing/invalid manifest
BATCH_FAILED_EXIT = 3     # a batch subprocess failed (non-quota) — rerun to resume the rest
QUOTA_BLOCKED_EXIT = 5    # execution stopped: a batch was refused by the Lake quota headroom

# recognized run_coinbase_quality_map.py subprocess exit codes (pinned by a contract test)
QUOTA_REFUSED_EXIT = 5      # == run_coinbase_quality_map.QUOTA_REFUSED_EXIT
NATIVE_UNAVAILABLE_EXIT = 6  # == run_coinbase_quality_map.NATIVE_UNAVAILABLE_EXIT

# per-batch status values
COMPLETE = "complete"
PENDING = "pending"
FAILED = "failed"
BLOCKED_QUOTA = "blocked_quota"
STALE = "stale"  # a report is present but does not cover this plan row's day set (re-plan drift)
STATUSES = (COMPLETE, PENDING, FAILED, BLOCKED_QUOTA, STALE)


class RunnerError(ValueError):
    """A clear, user-actionable orchestration failure (bad/missing manifest)."""


def _utcnow_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


# ----------------------------------------------------------------------------- manifest loading
def load_manifest(path: str) -> dict:
    """Load and validate the plan manifest (plan_coinbase_quality_map_batches.py output). Fails
    clearly (RunnerError) rather than silently orchestrating off a malformed/partial manifest."""
    if not os.path.exists(path):
        raise RunnerError(f"batch manifest not found: {path} "
                          "(run scripts/plan_coinbase_quality_map_batches.py to produce it)")
    with open(path) as f:
        try:
            manifest = json.load(f)
        except json.JSONDecodeError as e:
            raise RunnerError(f"batch manifest {path} is not valid JSON: {e}") from None
    if not isinstance(manifest, dict):
        raise RunnerError(f"batch manifest {path} must be a JSON object")
    batches = manifest.get("batches")
    if not isinstance(batches, list):
        raise RunnerError(f"batch manifest {path} is missing a 'batches' list")
    for i, b in enumerate(batches, start=1):
        if not isinstance(b, dict):
            raise RunnerError(f"batch manifest {path} batch #{i} must be a JSON object")
        for key in ("file", "report_dir", "command"):
            if not isinstance(b.get(key), str) or not b[key]:
                raise RunnerError(f"batch manifest {path} batch #{i} is missing a non-empty "
                                  f"'{key}' field")
        _runner_argv(b)  # fail fast here (exit 2) rather than mid-execute on a malformed command
    return manifest


# ----------------------------------------------------------------------------- runner command
def _runner_argv(batch: dict) -> list[str]:
    """Tokenize a batch command and confirm it invokes the KNOWN quota-gated runner. Raises
    RunnerError on a malformed command or an unexpected program. The manifest is a trusted, locally
    generated artifact, but a stale/hand-edited one must not launch some other tool that skips the
    Lake quota gate — so the program name is pinned to RUNNER_SCRIPT (defense-in-depth)."""
    argv = shlex.split(batch["command"])
    if len(argv) < 2:
        raise RunnerError(f"batch {batch.get('file')!r} has an unparseable command: "
                          f"{batch['command']!r}")
    if pathlib.Path(argv[1]).name != pathlib.Path(RUNNER_SCRIPT).name:
        raise RunnerError(f"batch {batch.get('file')!r} command does not invoke {RUNNER_SCRIPT} "
                          f"(got program {argv[1]!r}); refusing to run an unexpected tool")
    return argv


def build_command(batch: dict) -> list[str]:
    """The exact subprocess argv for one batch: the planner's emitted command, with the interpreter
    swapped for the CURRENT one (worktrees have no local .venv, so the literal '.venv/bin/python' is
    not runnable). Every planner-chosen flag (--engine native, --allow-broad, pinned --out-dir /
    --usable-calendar) is preserved verbatim — the planner owns the command, this only executes it."""
    return [sys.executable, *_runner_argv(batch)[1:]]


# ----------------------------------------------------------------------------- report / status
def report_path(batch: dict, base_dir: str) -> pathlib.Path:
    """Where the runner writes this batch's report: <base_dir>/<report_dir>/coinbase_quality_map.json
    (an absolute report_dir ignores base_dir, per pathlib join semantics)."""
    return pathlib.Path(base_dir) / batch["report_dir"] / REPORT_NAME


def read_report(path: pathlib.Path) -> dict | None:
    """Return the parsed batch report iff it exists AND has the runner's shape
    ({meta, summary(dict), days}); otherwise None (missing / truncated / wrong shape ⇒ NOT complete,
    so a half-written artifact never counts as a finished batch)."""
    try:
        report = json.loads(path.read_text())
    except (FileNotFoundError, NotADirectoryError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(report, dict):
        return None
    if "days" not in report or not isinstance(report.get("summary"), dict) or "meta" not in report:
        return None
    return report


def _planned_days(batch: dict, base_dir: str) -> set[str] | None:
    """The batch's planned day set, read from the days file its command points at (`--days-file`,
    resolved under base_dir the same way the runner reads it). None when the file is absent/
    unreadable — data/tmp is ephemeral — so the caller falls back to the endpoint check."""
    try:
        argv = shlex.split(batch.get("command", ""))
    except ValueError:
        return None
    if "--days-file" not in argv:
        return None
    i = argv.index("--days-file")
    if i + 1 >= len(argv):
        return None
    try:
        text = (pathlib.Path(base_dir) / argv[i + 1]).read_text()
    except (FileNotFoundError, NotADirectoryError, OSError):
        return None
    days = {tok.strip() for line in text.splitlines() for tok in line.split(",")}
    days.discard("")
    return days or None


def report_matches_batch(report: dict, batch: dict, *, base_dir: str) -> bool:
    """True iff a present report actually covers this batch's PLANNED day set. Guards against a stale
    report left under the index-derived report_dir (e.g. `batch_003/`) by an EARLIER plan whose day
    set differed — accepting it as complete would be a silent coverage gap in the very full-window
    map that gates the backfill.

    Primary check: the report's day set equals the batch's planned days (from the days file the
    command points at). This catches an interior-day swap that leaves n_days/first/last unchanged.
    Fallback (days file absent, or the report carries no per-day rows): the always-present n_days and,
    when present, the first/last day (`days` are {"day": ...} dicts)."""
    days = report.get("days") or []
    report_days = {(r.get("day") if isinstance(r, dict) else r) for r in days}
    report_days.discard(None)
    planned = _planned_days(batch, base_dir)
    if planned is not None and report_days:
        return report_days == planned
    summary = report.get("summary") or {}
    if batch.get("n_days") is not None and summary.get("n_days") != batch.get("n_days"):
        return False
    if days:
        first = days[0].get("day") if isinstance(days[0], dict) else days[0]
        last = days[-1].get("day") if isinstance(days[-1], dict) else days[-1]
        if batch.get("first_day") is not None and first != batch.get("first_day"):
            return False
        if batch.get("last_day") is not None and last != batch.get("last_day"):
            return False
    return True


def batch_status(batch: dict, *, base_dir: str, ledger_index: dict) -> str:
    """Authoritative per-batch status. A present report that MATCHES the plan row wins (disk is ground
    truth); a present report that does NOT match the row is `stale` (a re-run overwrites it); with no
    report, fall back to the last recorded attempt: quota-refused ⇒ blocked_quota, any other nonzero ⇒
    failed, else pending."""
    report = read_report(report_path(batch, base_dir))
    if report is not None:
        return COMPLETE if report_matches_batch(report, batch, base_dir=base_dir) else STALE
    last = ledger_index.get(batch["file"])
    if last is not None:
        code = last.get("exit_code")
        if code == QUOTA_REFUSED_EXIT:
            return BLOCKED_QUOTA
        if code not in (0, None):
            return FAILED
        # exit 0 recorded but no report on disk (anomalous) ⇒ pending, so a re-run retries it
    return PENDING


# ----------------------------------------------------------------------------- status ledger
def load_ledger(path: str) -> list[dict]:
    """Read the append-only attempt ledger (mirrors ingest/lake_binance.manifest_index): skips
    blank/malformed/keyless lines so a partially-written line never breaks status."""
    p = pathlib.Path(path)
    if not p.exists():
        return []
    records = []
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(rec, dict) and rec.get("file"):
            records.append(rec)
    return records


def ledger_index(records: list[dict]) -> dict:
    """Map batch file -> latest attempt record (last record wins)."""
    idx: dict = {}
    for rec in records:
        idx[rec["file"]] = rec
    return idx


def append_ledger(path: str, record: dict) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a") as f:
        f.write(json.dumps(record, allow_nan=False) + "\n")


# ----------------------------------------------------------------------------- execution
def _default_runner(argv: list[str], cwd: str) -> int:
    """Run one batch as a child process, streaming its output (so the runner's quota-gate prints and
    refusals are visible). Returns the child's exit code. shell=False — argv is already tokenized."""
    return subprocess.run(argv, cwd=cwd).returncode  # noqa: S603 (argv from our own planner)


def _run_status_from_exit(code: int) -> str:
    if code == 0:
        return COMPLETE
    if code == QUOTA_REFUSED_EXIT:
        return BLOCKED_QUOTA
    return FAILED


def execute(batches: list[dict], *, base_dir: str, status_dir: str, max_batches: int,
            runner, ledger_idx: dict) -> dict:
    """Run pending batches (skipping completed ones) up to `max_batches`, appending each attempt to
    the status ledger. Stops immediately when a batch is quota-blocked (the monthly window is spent);
    a non-quota failure is recorded and the loop continues within the budget. `ledger_idx` is updated
    in place so a follow-up summary reflects this run."""
    ledger_path = os.path.join(status_dir, STATUS_LEDGER_NAME)
    ran = 0
    outcomes: list[dict] = []
    for batch in batches:
        status = batch_status(batch, base_dir=base_dir, ledger_index=ledger_idx)
        if status == COMPLETE:
            print(f"  {batch['file']}  skip (report exists)")
            continue
        if ran >= max_batches:
            break
        argv = build_command(batch)
        print(f"\n  running {batch['file']} ({batch.get('n_days')} day(s), "
              f"{batch.get('first_day')}..{batch.get('last_day')}, "
              f"~{batch.get('runner_est_gb')} GB) …")
        started = _utcnow_iso()
        code = int(runner(argv, base_dir))
        record = {"file": batch["file"], "report_dir": batch["report_dir"], "exit_code": code,
                  "status": _run_status_from_exit(code), "n_days": batch.get("n_days"),
                  "runner_est_gb": batch.get("runner_est_gb"),
                  "started_utc": started, "finished_utc": _utcnow_iso()}
        append_ledger(ledger_path, record)
        ledger_idx[batch["file"]] = record
        outcomes.append(record)
        ran += 1
        if code == QUOTA_REFUSED_EXIT:
            print(f"  {batch['file']}  BLOCKED by Lake quota/headroom (exit {code}) — stopping. "
                  "Re-run after the next monthly quota window (re-check lakeapi.used_data first).")
            break
        if code != 0:
            print(f"  {batch['file']}  FAILED (exit {code}) — continuing with the next batch.",
                  file=sys.stderr)
        else:
            print(f"  {batch['file']}  complete.")
    return {"ran": ran, "outcomes": outcomes}


def execute_exit_code(result: dict) -> int:
    codes = [o["exit_code"] for o in result["outcomes"]]
    if any(c == QUOTA_REFUSED_EXIT for c in codes):
        return QUOTA_BLOCKED_EXIT
    if any(c != 0 for c in codes):
        return BATCH_FAILED_EXIT
    return 0


def preview_execute(batches: list[dict], *, base_dir: str, max_batches: int,
                    ledger_idx: dict) -> None:
    """DRY RUN: print the exact command `--execute` would run for the next pending batch(es). No
    vendor I/O, nothing written."""
    print("DRY RUN — no vendor I/O, no files written. Commands --execute WOULD run:")
    shown = 0
    for batch in batches:
        status = batch_status(batch, base_dir=base_dir, ledger_index=ledger_idx)
        if status == COMPLETE:
            print(f"  {batch['file']}  skip (report exists)")
            continue
        if shown >= max_batches:
            print(f"  … {batch['file']} and later pending batches deferred "
                  f"(--max-batches {max_batches}).")
            break
        print(f"  {batch['file']}  ({status}) would run:\n    {' '.join(build_command(batch))}")
        shown += 1
    if shown == 0:
        print("  (nothing pending — all batches already complete).")


# ----------------------------------------------------------------------------- aggregate summary
def aggregate_quality(batches: list[dict], *, base_dir: str) -> dict | None:
    """Combine the `summary` blocks of every COMPLETE batch report into one quality-map roll-up.
    Only reports that MATCH their plan row are aggregated (a stale report is excluded, so it never
    silently pads the coverage). Returns None when no batch has a matching local report yet."""
    reports = [r for b in batches
               if (r := read_report(report_path(b, base_dir))) is not None
               and report_matches_batch(r, b, base_dir=base_dir)]
    if not reports:
        return None
    n_days = 0
    counts: dict[str, int] = {}
    fill_counts: dict[str, int] = {}
    full_day_reason_counts: dict[str, int] = {}
    lists: dict[str, set] = {k: set() for k in
                             ("needs_fill", "no_fill", "no_verdict", "not_in_scope", "partial_fill")}
    for report in reports:
        summary = report.get("summary") or {}
        n_days += int(summary.get("n_days") or 0)
        for cls, n in (summary.get("counts") or {}).items():
            counts[cls] = counts.get(cls, 0) + int(n)
        fill = summary.get("coinapi_fill") or {}
        for key, n in (fill.get("fill_counts") or {}).items():
            fill_counts[key] = fill_counts.get(key, 0) + int(n)
        for reason, n in (fill.get("full_day_reason_counts") or {}).items():
            full_day_reason_counts[reason] = full_day_reason_counts.get(reason, 0) + int(n)
        for key in lists:
            lists[key].update(fill.get(key) or [])
    return {"n_batches_complete": len(reports), "n_batches_total": len(batches), "n_days": n_days,
            "counts": counts,
            "coinapi_fill": {"fill_counts": fill_counts,
                             "full_day_reason_counts": full_day_reason_counts,
                             **{key: sorted(vals) for key, vals in lists.items()}}}


def build_summary(manifest: dict, *, base_dir: str, ledger_idx: dict, manifest_path: str) -> dict:
    """The single status/aggregate artifact: per-batch status + counts, and the quality-map roll-up
    over completed batches."""
    batches = manifest["batches"]
    by_status: dict[str, list] = {s: [] for s in STATUSES}
    rows = []
    for batch in batches:
        status = batch_status(batch, base_dir=base_dir, ledger_index=ledger_idx)
        by_status[status].append(batch["file"])
        last = ledger_idx.get(batch["file"], {})
        rows.append({"file": batch["file"], "report_dir": batch["report_dir"], "status": status,
                     "n_days": batch.get("n_days"), "first_day": batch.get("first_day"),
                     "last_day": batch.get("last_day"), "runner_est_gb": batch.get("runner_est_gb"),
                     "last_exit_code": last.get("exit_code"),
                     "last_attempt_utc": last.get("finished_utc")})
    meta = manifest.get("meta") or {}
    return {
        "meta": {"manifest": manifest_path, "generated_utc": _utcnow_iso(),
                 "n_batches": len(batches),
                 "manifest_generated_utc": meta.get("generated_utc"),
                 "input_calendar": meta.get("input_calendar"),
                 "note": "Status/aggregate for the broad Coinbase quality map. Does NOT unlock the "
                         "§5a CoinAPI backfill gate; the broad full-window map remains the gate."},
        "status": {"counts": {s: len(by_status[s]) for s in STATUSES},
                   "by_status": by_status, "batches": rows},
        "quality_map": aggregate_quality(batches, base_dir=base_dir),
    }


def write_summary(summary: dict, status_dir: str) -> str:
    d = pathlib.Path(status_dir)
    d.mkdir(parents=True, exist_ok=True)
    path = d / SUMMARY_NAME
    with open(path, "w") as f:
        json.dump(summary, f, indent=2, allow_nan=False)
        f.write("\n")
    return str(path)


# ----------------------------------------------------------------------------- reporting (stdout)
def print_status(summary: dict) -> None:
    counts = summary["status"]["counts"]
    m = summary["meta"]
    print("=" * 78)
    print("  COINBASE QUALITY-MAP BATCH STATUS")
    print("=" * 78)
    print(f"  manifest:  {m['manifest']}  (planned {m['manifest_generated_utc']})")
    print(f"  batches:   {counts[COMPLETE]} complete, {counts[PENDING]} pending, "
          f"{counts[FAILED]} failed, {counts[BLOCKED_QUOTA]} blocked_quota, "
          f"{counts[STALE]} stale of {m['n_batches']}")
    for row in summary["status"]["batches"]:
        tail = "" if row["last_exit_code"] is None else f"  [last exit {row['last_exit_code']}]"
        print(f"    {row['file']}  {row['status']:<13}  {row.get('n_days')} day(s)  "
              f"{row.get('first_day')}..{row.get('last_day')}{tail}")
    q = summary["quality_map"]
    if q is None:
        print("  quality map: (no completed batch reports yet — nothing to aggregate)")
    else:
        print(f"  quality map: {q['n_batches_complete']}/{q['n_batches_total']} batches, "
              f"{q['n_days']} day(s) aggregated; class counts {q['counts']}; "
              f"needs_fill {q['coinapi_fill']['fill_counts'].get('needs_fill', 0)}")


# ----------------------------------------------------------------------------- main
def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Safe, resumable orchestrator for the broad Coinbase quality map. Default is a "
                    "read-only STATUS view (no vendor I/O); --execute runs one planned batch per "
                    "quota window (LIVE Lake download, still quota-gated). It does NOT unlock the "
                    "§5a CoinAPI backfill gate (docs/data.md §5a-QualityMap).")
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST,
                    help=f"plan manifest from plan_coinbase_quality_map_batches.py "
                         f"(default {DEFAULT_MANIFEST})")
    ap.add_argument("--base-dir", default=".",
                    help="directory the batch report_dir paths and the runner subprocess resolve "
                         "against (default: current dir = repo root)")
    ap.add_argument("--status-dir", default=None,
                    help=f"where to write summary.json + the status ledger (git-ignored; default "
                         f"<base-dir>/{STATUS_ROOT})")
    ap.add_argument("--execute", action="store_true",
                    help="actually run pending batches as subprocesses (LIVE Crypto Lake I/O, still "
                         "refused by the runner's own quota headroom). Without this the tool only "
                         "reports status.")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the exact command --execute would run for the next pending batch(es) "
                         "and exit; writes nothing (overrides --execute)")
    ap.add_argument("--max-batches", type=int, default=1,
                    help="max batches to run per --execute invocation (default 1 = one batch per "
                         "monthly quota window, docs §5a). N>1 runs batches back-to-back and can "
                         "breach the monthly cap because lakeapi.used_data lags ~60 min, so the "
                         "per-batch gate may not yet see an earlier pull — keep this at 1 for broad "
                         "batches.")
    return ap.parse_args(argv)


def main(argv=None, *, runner=None) -> int:
    args = parse_args(argv)
    if args.max_batches < 1:
        print(f"ERROR: --max-batches must be >= 1, got {args.max_batches}", file=sys.stderr)
        return MANIFEST_ERROR_EXIT
    try:
        manifest = load_manifest(args.manifest)
    except RunnerError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return MANIFEST_ERROR_EXIT

    status_dir = args.status_dir or os.path.join(args.base_dir, STATUS_ROOT)
    ledger_path = os.path.join(status_dir, STATUS_LEDGER_NAME)
    ledger_idx = ledger_index(load_ledger(ledger_path))

    if args.dry_run:
        preview_execute(manifest["batches"], base_dir=args.base_dir,
                        max_batches=args.max_batches, ledger_idx=ledger_idx)
        return 0

    exit_code = 0
    if args.execute:
        if args.max_batches > 1:
            print(f"WARNING: --max-batches {args.max_batches} runs broad batches back-to-back in ONE "
                  "quota window. lakeapi.used_data lags ~60 min, so the runner's per-batch quota gate "
                  "may not yet reflect an earlier ~250 GB batch and could allow a pull that breaches "
                  "the 300 GB/month cap. Prefer one batch per monthly quota window (--max-batches 1).",
                  file=sys.stderr)
        result = execute(manifest["batches"], base_dir=args.base_dir, status_dir=status_dir,
                         max_batches=args.max_batches, runner=runner or _default_runner,
                         ledger_idx=ledger_idx)
        exit_code = execute_exit_code(result)

    summary = build_summary(manifest, base_dir=args.base_dir, ledger_idx=ledger_idx,
                            manifest_path=args.manifest)
    write_summary(summary, status_dir)
    print_status(summary)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
