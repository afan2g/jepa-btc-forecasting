"""Offline tests for the resumable Coinbase quality-map BATCH ORCHESTRATOR
(`scripts/run_coinbase_quality_map_batches.py`, docs/data.md §5a-QualityMap staging).

Drives the orchestrator end-to-end on a SYNTHETIC plan manifest + fake local report files with an
INJECTED fake runner — no vendor I/O and the real `run_coinbase_quality_map.py` is NEVER invoked
here (no subprocess). Covers manifest loading/validation, resume/skip of completed batches, per-batch
status (complete/pending/failed/blocked_quota), the one-batch-per-quota-window execute loop with a
hard stop on a quota-blocked batch, dry-run command preview, the append-only status ledger, and the
aggregate summary artifact. Mirrors the module-load-by-path pattern of test_plan_quality_map_batches."""
import importlib.util
import json
import pathlib
import sys

import pytest

# scripts/ is not a package — load the script module by path (same pattern as the planner test).
_SPEC = importlib.util.spec_from_file_location(
    "run_coinbase_quality_map_batches",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_coinbase_quality_map_batches.py")
qmb = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qmb
_SPEC.loader.exec_module(qmb)

REPORT_NAME = "coinbase_quality_map.json"


# --------------------------------------------------------------------------- synthetic fixtures
def _command(report_dir, days_file="days.txt", calendar="cal.json") -> str:
    """A manifest `command` string exactly as the planner emits it (COMMAND_TEMPLATE)."""
    return (".venv/bin/python scripts/run_coinbase_quality_map.py --engine native --no-cold-ab "
            f"--days-file {days_file} --usable-calendar {calendar} --out-dir {report_dir} "
            "--allow-broad")


def _batch(n: int, report_dir, days_file, calendar) -> dict:
    return {"file": f"batch_{n:03d}_days.txt", "n_days": 2,
            "first_day": "2025-01-01", "last_day": "2025-01-02",
            "est_gb": 0.96, "runner_est_gb": 0.96, "report_dir": str(report_dir),
            "command": _command(report_dir, days_file, calendar)}


def _manifest_dict(tmp_path, n_batches=3, *, relative_report_dirs=False) -> dict:
    cal = str(tmp_path / "cal.json")
    batches = []
    for i in range(1, n_batches + 1):
        rd = f"reports/batch_{i:03d}" if relative_report_dirs else str(tmp_path / "reports"
                                                                        / f"batch_{i:03d}")
        batches.append(_batch(i, rd, str(tmp_path / f"batch_{i:03d}_days.txt"), cal))
    return {
        "meta": {"input_calendar": cal, "generated_utc": "2026-07-03T00:00:00+00:00",
                 "out_dir": str(tmp_path / "tmp"), "gb_per_day": 0.48, "max_gb_per_batch": 250.0},
        "summary": {"n_batches": n_batches, "n_batch_days": 2 * n_batches,
                    "total_est_gb": round(0.96 * n_batches, 2)},
        "batches": batches,
        "batched_trade_only_fill_days": [],
        "skipped": {"fill_days_book_gap": [], "excluded_days_by_reason": {},
                    "days_dropped_as_excluded_or_book_gap": []},
    }


def _write_manifest(tmp_path, manifest=None) -> str:
    manifest = manifest if manifest is not None else _manifest_dict(tmp_path)
    p = tmp_path / "manifest.json"
    p.write_text(json.dumps(manifest))
    return str(p)


def _write_report(report_dir, *, n_days=2, counts=None, coinapi_fill=None, days=None) -> pathlib.Path:
    """Write a minimal but SHAPE-VALID coinbase_quality_map.json (what the runner writes on exit 0),
    with per-day rows consistent with n_days (the runner always emits len(days) == summary.n_days)."""
    rd = pathlib.Path(report_dir)
    rd.mkdir(parents=True, exist_ok=True)
    counts = counts or {"lake_usable": 2}
    cf = coinapi_fill or {
        "needs_fill": [], "no_fill": ["2025-01-01", "2025-01-02"], "no_verdict": [],
        "not_in_scope": [], "partial_fill": [],
        "fill_counts": {"needs_fill": 0, "full_day_fill": 0, "no_fill": 2, "no_verdict": 0,
                        "not_in_scope": 0},
        "full_day_reason_counts": {}}
    if days is None:
        days = [{"day": f"2025-01-{i + 1:02d}"} for i in range(n_days)]
    report = {"meta": {"engine": "native"},
              "summary": {"n_days": n_days, "counts": counts,
                          "by_class": {c: [] for c in counts}, "coinapi_fill": cf},
              "days": days}
    (rd / REPORT_NAME).write_text(json.dumps(report))
    return rd / REPORT_NAME


class _FakeRunner:
    """Injected in place of the real subprocess. Returns a scripted exit code per call and, on a
    0 (success) code, emulates run_coinbase_quality_map.py writing its fixed report under --out-dir."""
    def __init__(self, exits):
        self.exits = list(exits)
        self.calls = []

    def __call__(self, argv, cwd):
        self.calls.append(list(argv))
        code = self.exits.pop(0)
        if code == 0:
            out_dir = argv[argv.index("--out-dir") + 1]
            _write_report(pathlib.Path(cwd) / out_dir)
        return code


def _boom_runner(argv, cwd):  # must never be called in dry-run / status mode
    raise AssertionError(f"runner invoked but should not have been: {argv}")


def _status_dir(tmp_path) -> str:
    return str(tmp_path / "status")


def _read_summary(tmp_path) -> dict:
    return json.loads((pathlib.Path(_status_dir(tmp_path)) / "summary.json").read_text())


# --------------------------------------------------------------------------- manifest loading
def test_missing_manifest_file_exits_2(tmp_path, capsys):
    rc = qmb.main(["--manifest", str(tmp_path / "nope.json"), "--status-dir", _status_dir(tmp_path)])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_load_manifest_requires_batches_field(tmp_path):
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"meta": {}}))
    with pytest.raises(ValueError, match="batches"):
        qmb.load_manifest(str(p))


def test_load_manifest_rejects_batch_without_command_or_report_dir(tmp_path):
    bad = _manifest_dict(tmp_path)
    del bad["batches"][0]["command"]
    p = tmp_path / "m.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="command"):
        qmb.load_manifest(str(p))


def test_load_manifest_rejects_non_json(tmp_path):
    p = tmp_path / "m.json"
    p.write_text("not json {{{")
    with pytest.raises(ValueError):
        qmb.load_manifest(str(p))


def test_load_manifest_rejects_a_command_that_is_not_the_known_runner(tmp_path):
    bad = _manifest_dict(tmp_path)
    bad["batches"][0]["command"] = ".venv/bin/python scripts/evil.py --out-dir x"
    p = tmp_path / "m.json"
    p.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="does not invoke"):
        qmb.load_manifest(str(p))


def test_main_exits_2_on_a_malformed_batch_command(tmp_path, capsys):
    # a bad command must fail fast at load with the clean exit 2, NOT a raw traceback mid-execute
    bad = _manifest_dict(tmp_path)
    bad["batches"][0]["command"] = "python"  # one token, unparseable as a runner invocation
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(bad))
    rc = qmb.main(["--manifest", str(mpath), "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute"], runner=_boom_runner)
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


# --------------------------------------------------------------------------- build_command (seam)
def test_build_command_swaps_interpreter_and_preserves_planner_flags(tmp_path):
    batch = _manifest_dict(tmp_path)["batches"][0]
    argv = qmb.build_command(batch)
    assert argv[0] == sys.executable  # NOT the literal ".venv/bin/python" (worktrees have no .venv)
    assert argv[1].endswith("run_coinbase_quality_map.py")
    for flag in ("--engine", "native", "--no-cold-ab", "--allow-broad", "--out-dir",
                 "--usable-calendar", "--days-file"):
        assert flag in argv
    # the pinned report dir and calendar survive verbatim
    assert argv[argv.index("--out-dir") + 1] == batch["report_dir"]


# --------------------------------------------------------------------------- report / status
def test_report_present_and_valid_is_complete(tmp_path):
    m = _manifest_dict(tmp_path)
    _write_report(m["batches"][0]["report_dir"])
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index={}) == qmb.COMPLETE


def test_report_absent_is_pending(tmp_path):
    m = _manifest_dict(tmp_path)
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index={}) == qmb.PENDING


def test_corrupt_report_json_is_not_complete(tmp_path):
    m = _manifest_dict(tmp_path)
    rd = pathlib.Path(m["batches"][0]["report_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    (rd / REPORT_NAME).write_text("{ truncated")
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index={}) == qmb.PENDING


def test_report_missing_required_keys_is_not_complete(tmp_path):
    m = _manifest_dict(tmp_path)
    rd = pathlib.Path(m["batches"][0]["report_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    (rd / REPORT_NAME).write_text(json.dumps({"meta": {}}))  # no summary/days
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index={}) == qmb.PENDING


def test_status_reflects_quota_block_from_ledger(tmp_path):
    # only exit_code drives classification (the ledger's own 'status' field is not read here)
    m = _manifest_dict(tmp_path)
    idx = {m["batches"][0]["file"]: {"file": m["batches"][0]["file"], "exit_code": 5}}
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index=idx) == qmb.BLOCKED_QUOTA


def test_status_reflects_failure_from_ledger(tmp_path):
    m = _manifest_dict(tmp_path)
    idx = {m["batches"][0]["file"]: {"file": m["batches"][0]["file"], "exit_code": 6}}
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index=idx) == qmb.FAILED


def test_report_path_resolves_relative_report_dir_under_base_dir(tmp_path):
    m = _manifest_dict(tmp_path, relative_report_dirs=True)
    _write_report(tmp_path / "reports" / "batch_001")
    assert qmb.batch_status(m["batches"][0], base_dir=str(tmp_path), ledger_index={}) == qmb.COMPLETE


def test_exit_zero_without_report_is_pending(tmp_path):
    # anomalous: a 0 recorded but the report is not on disk ⇒ pending, so a re-run retries it
    m = _manifest_dict(tmp_path)
    idx = {m["batches"][0]["file"]: {"file": m["batches"][0]["file"], "exit_code": 0}}
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index=idx) == qmb.PENDING


def test_present_report_overrides_conflicting_blocked_ledger(tmp_path):
    # disk is ground truth: a valid report wins even over a stale blocked_quota ledger entry
    m = _manifest_dict(tmp_path)
    _write_report(m["batches"][0]["report_dir"])
    idx = {m["batches"][0]["file"]: {"file": m["batches"][0]["file"], "exit_code": 5}}
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index=idx) == qmb.COMPLETE


def test_build_command_rejects_unparseable_command():
    with pytest.raises(ValueError, match="unparseable"):
        qmb.build_command({"file": "b", "report_dir": "r", "command": "python"})


def test_build_command_rejects_a_command_that_is_not_the_known_runner():
    # defense-in-depth: a manifest command that does not invoke the quota-gated runner is refused
    batch = {"file": "b", "report_dir": "r",
             "command": ".venv/bin/python scripts/some_other_tool.py --out-dir r"}
    with pytest.raises(ValueError, match="does not invoke"):
        qmb.build_command(batch)


def test_runner_pin_rejects_same_basename_at_a_different_path():
    # Codex P2: basename-only pinning would accept a same-named script ELSEWHERE (e.g. /tmp), which
    # --execute would then run instead of the repo's quota-gated runner. Pin the full relative path.
    batch = {"file": "b", "report_dir": "r",
             "command": "python /tmp/run_coinbase_quality_map.py --out-dir r"}
    with pytest.raises(ValueError, match="does not invoke"):
        qmb.build_command(batch)


def test_build_command_rejects_a_quota_override_flag():
    # Codex P2: a command that keeps the pinned runner but weakens the quota gate must be refused,
    # or the "runner quota gate preserved" guarantee is void. The planner never emits these flags.
    batch = {"file": "b", "report_dir": "r",
             "command": ("python scripts/run_coinbase_quality_map.py --allow-broad "
                         "--quota-gb 100000 --out-dir r")}
    with pytest.raises(ValueError, match="quota"):
        qmb.build_command(batch)


def test_build_command_rejects_a_headroom_override_equals_form():
    batch = {"file": "b", "report_dir": "r",
             "command": ("python scripts/run_coinbase_quality_map.py --headroom-gb=0 --out-dir r")}
    with pytest.raises(ValueError, match="quota"):
        qmb.build_command(batch)


def test_build_command_rejects_malformed_shell_quoting():
    # Codex P3: shlex.split raises a bare ValueError on malformed quoting — it must become a clean
    # RunnerError ("unparseable"), not a raw traceback, so all bad manifest commands fail uniformly.
    batch = {"file": "b", "report_dir": "r",
             "command": 'python scripts/run_coinbase_quality_map.py --out-dir "unterminated'}
    with pytest.raises(ValueError, match="unparseable"):
        qmb.build_command(batch)


def test_main_exits_2_on_malformed_command_quoting(tmp_path, capsys):
    bad = _manifest_dict(tmp_path)
    bad["batches"][0]["command"] = ('python scripts/run_coinbase_quality_map.py '
                                    '--out-dir "unterminated')
    mpath = tmp_path / "m.json"
    mpath.write_text(json.dumps(bad))
    rc = qmb.main(["--manifest", str(mpath), "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute"], runner=_boom_runner)
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_stale_report_not_matching_the_plan_row_is_not_complete(tmp_path):
    # a report left under batch_001's report_dir by a PRIOR plan covering a DIFFERENT day set
    # (n_days differs) must NOT count as complete — silently skipping it is a coverage gap in the
    # gate map. It is surfaced as a distinct STALE status so a re-run overwrites it.
    m = _manifest_dict(tmp_path)  # batch_001 row claims n_days=2
    _write_report(m["batches"][0]["report_dir"], n_days=5)  # stale: 5 != 2
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index={}) == qmb.STALE


def test_stale_report_is_excluded_from_the_aggregate(tmp_path):
    m = _manifest_dict(tmp_path, n_batches=2)
    _write_report(m["batches"][0]["report_dir"], n_days=2)  # matches the row → complete
    _write_report(m["batches"][1]["report_dir"], n_days=9)  # mismatched → stale, must not aggregate
    mpath = _write_manifest(tmp_path, m)
    qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path), "--base-dir", str(tmp_path)],
             runner=_boom_runner)
    s = _read_summary(tmp_path)
    assert s["status"]["counts"]["complete"] == 1
    assert s["status"]["counts"]["stale"] == 1
    assert s["quality_map"]["n_batches_complete"] == 1  # only the matching report contributes


def test_stale_report_with_swapped_interior_day_is_detected(tmp_path):
    # Codex P2: a re-plan can swap an INTERIOR day while keeping n_days/first_day/last_day identical.
    # The endpoint check alone would miss it, so the FULL planned day set (from the batch days file
    # the command points at) must be compared against the report's per-day rows.
    days_file = tmp_path / "batch_001_days.txt"
    days_file.write_text("2025-01-01\n2025-01-02\n2025-01-03\n")  # current plan
    report_dir = tmp_path / "reports" / "batch_001"
    report_dir.mkdir(parents=True, exist_ok=True)
    batch = {"file": "batch_001_days.txt", "report_dir": str(report_dir),
             "n_days": 3, "first_day": "2025-01-01", "last_day": "2025-01-03",
             "command": (".venv/bin/python scripts/run_coinbase_quality_map.py --engine native "
                         f"--no-cold-ab --days-file {days_file} --usable-calendar c "
                         f"--out-dir {report_dir} --allow-broad")}
    report = {"meta": {}, "summary": {"n_days": 3, "counts": {}, "coinapi_fill": {}},
              # same count + endpoints, but the interior day is 01-09 (an OLD plan), not 01-02
              "days": [{"day": "2025-01-01"}, {"day": "2025-01-09"}, {"day": "2025-01-03"}]}
    (report_dir / REPORT_NAME).write_text(json.dumps(report))
    assert qmb.batch_status(batch, base_dir=".", ledger_index={}) == qmb.STALE
    # correcting the interior day to match the current days file → complete
    report["days"][1]["day"] = "2025-01-02"
    (report_dir / REPORT_NAME).write_text(json.dumps(report))
    assert qmb.batch_status(batch, base_dir=".", ledger_index={}) == qmb.COMPLETE


def test_report_with_matching_days_but_wrong_summary_count_is_not_complete(tmp_path):
    # Codex P2: the day set matching the plan is not enough — a corrupt summary.n_days must not be
    # trusted (aggregate_quality reads summary counts), so the count has to agree too.
    days_file = tmp_path / "batch_001_days.txt"
    days_file.write_text("2025-01-01\n2025-01-02\n")
    report_dir = tmp_path / "reports" / "batch_001"
    report_dir.mkdir(parents=True, exist_ok=True)
    batch = {"file": "batch_001_days.txt", "report_dir": str(report_dir),
             "n_days": 2, "first_day": "2025-01-01", "last_day": "2025-01-02",
             "command": ("python scripts/run_coinbase_quality_map.py "
                         f"--days-file {days_file} --out-dir {report_dir}")}
    report = {"meta": {}, "summary": {"n_days": 99, "counts": {}, "coinapi_fill": {}},  # corrupt count
              "days": [{"day": "2025-01-01"}, {"day": "2025-01-02"}]}  # day set matches the plan
    (report_dir / REPORT_NAME).write_text(json.dumps(report))
    assert qmb.batch_status(batch, base_dir=".", ledger_index={}) != qmb.COMPLETE


def test_days_file_disagreeing_with_row_bounds_is_not_complete(tmp_path):
    # Codex P2: a stale manifest row + regenerated days file (same n_days, different endpoints) must
    # not mark the wrong coverage window complete — the days file must also agree with the row bounds.
    days_file = tmp_path / "batch_001_days.txt"
    days_file.write_text("2025-02-01\n2025-02-02\n")  # days file endpoints are in February …
    report_dir = tmp_path / "reports" / "batch_001"
    report_dir.mkdir(parents=True, exist_ok=True)
    batch = {"file": "batch_001_days.txt", "report_dir": str(report_dir),
             "n_days": 2, "first_day": "2025-01-01", "last_day": "2025-01-02",  # … row claims January
             "command": ("python scripts/run_coinbase_quality_map.py "
                         f"--days-file {days_file} --out-dir {report_dir}")}
    report = {"meta": {}, "summary": {"n_days": 2, "counts": {}, "coinapi_fill": {}},
              "days": [{"day": "2025-02-01"}, {"day": "2025-02-02"}]}  # matches days file, not the row
    (report_dir / REPORT_NAME).write_text(json.dumps(report))
    assert qmb.batch_status(batch, base_dir=".", ledger_index={}) != qmb.COMPLETE


def test_report_with_no_day_rows_is_not_complete(tmp_path):
    # Codex P2: an empty days[] with n_days>0 is an inconsistent/half-written report — not complete
    m = _manifest_dict(tmp_path)  # batch_001 row claims n_days=2 (no days file written → fallback)
    rd = pathlib.Path(m["batches"][0]["report_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    report = {"meta": {}, "summary": {"n_days": 2, "counts": {}, "coinapi_fill": {}}, "days": []}
    (rd / REPORT_NAME).write_text(json.dumps(report))
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index={}) != qmb.COMPLETE


def test_report_matching_day_boundaries_is_complete(tmp_path):
    # a report whose day list boundaries match the plan row is accepted (days are {day: ...} dicts)
    m = _manifest_dict(tmp_path)
    rd = pathlib.Path(m["batches"][0]["report_dir"])
    rd.mkdir(parents=True, exist_ok=True)
    report = {"meta": {}, "summary": {"n_days": 2, "counts": {}, "coinapi_fill": {}},
              "days": [{"day": "2025-01-01"}, {"day": "2025-01-02"}]}  # first/last match the row
    (rd / REPORT_NAME).write_text(json.dumps(report))
    assert qmb.batch_status(m["batches"][0], base_dir=".", ledger_index={}) == qmb.COMPLETE


# --------------------------------------------------------------------------- execute loop
def test_execute_runs_one_batch_by_default(tmp_path):
    mpath = _write_manifest(tmp_path)
    runner = _FakeRunner([0])
    rc = qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute"], runner=runner)
    assert rc == 0
    assert len(runner.calls) == 1  # ONLY the next pending batch, not all three
    assert (pathlib.Path(tmp_path) / "reports" / "batch_001" / REPORT_NAME).exists()
    assert not (pathlib.Path(tmp_path) / "reports" / "batch_002" / REPORT_NAME).exists()


def test_execute_skips_completed_batch_and_runs_the_next(tmp_path):
    m = _manifest_dict(tmp_path)
    _write_report(m["batches"][0]["report_dir"])  # batch_001 already done
    mpath = _write_manifest(tmp_path, m)
    runner = _FakeRunner([0])
    rc = qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute"], runner=runner)
    assert rc == 0
    assert len(runner.calls) == 1
    # the one call targets batch_002 (batch_001 was skipped as already complete)
    assert runner.calls[0][runner.calls[0].index("--out-dir") + 1] == m["batches"][1]["report_dir"]


def test_execute_stops_on_quota_blocked_batch(tmp_path):
    mpath = _write_manifest(tmp_path)
    runner = _FakeRunner([5])  # QUOTA_REFUSED_EXIT — runner writes no report
    rc = qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute", "--max-batches", "3"], runner=runner)
    assert rc == qmb.QUOTA_BLOCKED_EXIT
    assert len(runner.calls) == 1  # stopped immediately, did not try batch_002
    s = _read_summary(tmp_path)
    assert s["status"]["by_status"]["blocked_quota"] == ["batch_001_days.txt"]


def test_execute_records_failure_and_continues_within_budget(tmp_path):
    mpath = _write_manifest(tmp_path)
    runner = _FakeRunner([6, 0])  # batch_001 native-unavailable, batch_002 ok
    rc = qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute", "--max-batches", "2"], runner=runner)
    assert rc == qmb.BATCH_FAILED_EXIT
    assert len(runner.calls) == 2
    s = _read_summary(tmp_path)
    assert s["status"]["by_status"]["failed"] == ["batch_001_days.txt"]
    assert s["status"]["by_status"]["complete"] == ["batch_002_days.txt"]


def test_execute_appends_status_ledger(tmp_path):
    mpath = _write_manifest(tmp_path)
    qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
              "--base-dir", str(tmp_path), "--execute"], runner=_FakeRunner([0]))
    ledger = pathlib.Path(_status_dir(tmp_path)) / "_runner_status.jsonl"
    lines = [json.loads(x) for x in ledger.read_text().splitlines() if x.strip()]
    assert len(lines) == 1
    assert lines[0]["file"] == "batch_001_days.txt"
    assert lines[0]["exit_code"] == 0
    assert lines[0]["status"] == qmb.COMPLETE


def test_blocked_batch_is_retried_on_the_next_execute(tmp_path):
    """The resume promise: a batch refused by quota one window is retried (and can complete) the
    next --execute against the SAME status dir — nothing marks it permanently done."""
    mpath = _write_manifest(tmp_path)
    sdir = _status_dir(tmp_path)
    rc1 = qmb.main(["--manifest", mpath, "--status-dir", sdir, "--base-dir", str(tmp_path),
                    "--execute"], runner=_FakeRunner([5]))  # window 1: quota-blocked, no report
    assert rc1 == qmb.QUOTA_BLOCKED_EXIT
    assert not (pathlib.Path(tmp_path) / "reports" / "batch_001" / REPORT_NAME).exists()
    runner2 = _FakeRunner([0])
    rc2 = qmb.main(["--manifest", mpath, "--status-dir", sdir, "--base-dir", str(tmp_path),
                    "--execute"], runner=runner2)  # window 2: quota now available
    assert rc2 == 0
    assert runner2.calls[0][runner2.calls[0].index("--out-dir") + 1].endswith("batch_001")
    assert (pathlib.Path(tmp_path) / "reports" / "batch_001" / REPORT_NAME).exists()


def test_execute_warns_when_max_batches_exceeds_one(tmp_path, capsys):
    # used_data lags ~60 min, so >1 broad batch per window can breach the cap — warn loudly
    mpath = _write_manifest(tmp_path)
    qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path), "--base-dir", str(tmp_path),
              "--execute", "--max-batches", "2"], runner=_FakeRunner([0, 0]))
    err = capsys.readouterr().err
    assert "used_data lags" in err and "300 GB" in err


def test_execute_all_complete_is_a_noop(tmp_path):
    m = _manifest_dict(tmp_path, n_batches=2)
    _write_report(m["batches"][0]["report_dir"])
    _write_report(m["batches"][1]["report_dir"])
    mpath = _write_manifest(tmp_path, m)
    runner = _FakeRunner([])  # would IndexError if called
    rc = qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute"], runner=runner)
    assert rc == 0
    assert runner.calls == []


# --------------------------------------------------------------------------- dry-run / status modes
def test_dry_run_previews_command_and_writes_nothing(tmp_path, capsys):
    mpath = _write_manifest(tmp_path)
    rc = qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute", "--dry-run"], runner=_boom_runner)
    assert rc == 0
    out = capsys.readouterr().out
    assert "run_coinbase_quality_map.py" in out  # the command it WOULD run is printed
    assert "batch_001" in out
    assert not (pathlib.Path(_status_dir(tmp_path)) / "summary.json").exists()  # writes nothing
    assert not (pathlib.Path(_status_dir(tmp_path)) / "_runner_status.jsonl").exists()


def test_default_status_mode_writes_summary_and_runs_nothing(tmp_path):
    m = _manifest_dict(tmp_path)
    _write_report(m["batches"][0]["report_dir"])
    mpath = _write_manifest(tmp_path, m)
    rc = qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path)], runner=_boom_runner)  # no --execute
    assert rc == 0
    s = _read_summary(tmp_path)
    assert s["status"]["counts"]["complete"] == 1
    assert s["status"]["counts"]["pending"] == 2


# --------------------------------------------------------------------------- aggregate summary
def test_summary_status_counts_across_disk_and_ledger(tmp_path):
    m = _manifest_dict(tmp_path, n_batches=3)
    _write_report(m["batches"][0]["report_dir"])  # complete
    # batch_002 pending (no report, no ledger); batch_003 blocked via ledger
    mpath = _write_manifest(tmp_path, m)
    sdir = pathlib.Path(_status_dir(tmp_path))
    sdir.mkdir(parents=True, exist_ok=True)
    # ledger entry recorded under the CURRENT plan (its generated_utc) so it is honored
    (sdir / "_runner_status.jsonl").write_text(
        json.dumps({"file": "batch_003_days.txt", "exit_code": 5,
                    "plan_generated_utc": "2026-07-03T00:00:00+00:00"}) + "\n")
    qmb.main(["--manifest", mpath, "--status-dir", str(sdir), "--base-dir", str(tmp_path)],
             runner=_boom_runner)
    s = _read_summary(tmp_path)
    assert s["status"]["counts"] == {"complete": 1, "pending": 1, "failed": 0, "blocked_quota": 1,
                                     "stale": 0}
    assert s["status"]["by_status"]["complete"] == ["batch_001_days.txt"]
    assert s["status"]["by_status"]["blocked_quota"] == ["batch_003_days.txt"]


def test_ledger_entry_from_a_different_plan_is_ignored(tmp_path):
    # Codex P2: reused batch_NNN names + a persistent ledger must not let an OLD plan's exit 5/6
    # mark a never-attempted batch of the CURRENT plan as blocked/failed — scope by plan generation.
    m = _manifest_dict(tmp_path, n_batches=1)  # meta.generated_utc = 2026-07-03T00:00:00+00:00
    mpath = _write_manifest(tmp_path, m)
    sdir = pathlib.Path(_status_dir(tmp_path))
    sdir.mkdir(parents=True, exist_ok=True)
    (sdir / "_runner_status.jsonl").write_text(json.dumps(
        {"file": "batch_001_days.txt", "exit_code": 5,
         "plan_generated_utc": "2020-01-01T00:00:00+00:00"}) + "\n")  # an OLDER plan
    qmb.main(["--manifest", mpath, "--status-dir", str(sdir), "--base-dir", str(tmp_path)],
             runner=_boom_runner)
    s = _read_summary(tmp_path)
    assert s["status"]["by_status"]["pending"] == ["batch_001_days.txt"]  # NOT blocked_quota
    assert s["status"]["counts"]["blocked_quota"] == 0
    # the displayed attempt must also be scoped to the current plan (not the old exit 5)
    row = next(r for r in s["status"]["batches"] if r["file"] == "batch_001_days.txt")
    assert row["last_exit_code"] is None
    assert row["last_attempt_utc"] is None


def test_summary_aggregates_quality_over_complete_batches_only(tmp_path):
    m = _manifest_dict(tmp_path, n_batches=3)
    _write_report(m["batches"][0]["report_dir"], n_days=2, counts={"lake_usable": 2},
                  coinapi_fill={"needs_fill": ["2025-01-01"], "no_fill": ["2025-01-02"],
                                "no_verdict": [], "not_in_scope": [], "partial_fill": [],
                                "fill_counts": {"needs_fill": 1, "full_day_fill": 1, "no_fill": 1,
                                                "no_verdict": 0, "not_in_scope": 0},
                                "full_day_reason_counts": {"crossed_seed_source": 1}})
    _write_report(m["batches"][1]["report_dir"], n_days=2,
                  counts={"lake_usable": 1, "lake_present_degraded": 1},
                  coinapi_fill={"needs_fill": ["2025-01-03"], "no_fill": ["2025-01-04"],
                                "no_verdict": [], "not_in_scope": [], "partial_fill": ["2025-01-03"],
                                "fill_counts": {"needs_fill": 1, "full_day_fill": 0,
                                                "leading_partial_fill": 1, "no_fill": 1,
                                                "no_verdict": 0, "not_in_scope": 0},
                                "full_day_reason_counts": {}})
    # batch_003 left pending — must NOT contribute
    mpath = _write_manifest(tmp_path, m)
    qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path), "--base-dir", str(tmp_path)],
             runner=_boom_runner)
    q = _read_summary(tmp_path)["quality_map"]
    assert q["n_batches_complete"] == 2
    assert q["n_batches_total"] == 3
    assert q["n_days"] == 4
    assert q["counts"] == {"lake_usable": 3, "lake_present_degraded": 1}
    assert q["coinapi_fill"]["fill_counts"]["needs_fill"] == 2
    assert q["coinapi_fill"]["fill_counts"]["full_day_fill"] == 1
    assert q["coinapi_fill"]["fill_counts"]["leading_partial_fill"] == 1
    assert q["coinapi_fill"]["full_day_reason_counts"] == {"crossed_seed_source": 1}
    assert q["coinapi_fill"]["needs_fill"] == ["2025-01-01", "2025-01-03"]  # sorted union
    assert q["coinapi_fill"]["partial_fill"] == ["2025-01-03"]


def test_summary_quality_map_dedups_days_shared_across_batches(tmp_path):
    # two complete batches whose fill day-lists overlap: the roll-up must be a sorted UNION, not a
    # concatenation with duplicates (batch date ranges are contiguous, but a defensive dedup matters)
    m = _manifest_dict(tmp_path, n_batches=2)
    shared = {"needs_fill": ["2025-01-02", "2025-01-01"], "no_fill": [], "no_verdict": [],
              "not_in_scope": [], "partial_fill": [],
              "fill_counts": {"needs_fill": 2}, "full_day_reason_counts": {}}
    _write_report(m["batches"][0]["report_dir"], coinapi_fill=dict(shared))
    _write_report(m["batches"][1]["report_dir"],
                  coinapi_fill={**shared, "needs_fill": ["2025-01-02", "2025-01-03"]})
    mpath = _write_manifest(tmp_path, m)
    qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path), "--base-dir", str(tmp_path)],
             runner=_boom_runner)
    q = _read_summary(tmp_path)["quality_map"]
    assert q["coinapi_fill"]["needs_fill"] == ["2025-01-01", "2025-01-02", "2025-01-03"]


def test_summary_quality_map_null_when_no_complete_batches(tmp_path):
    mpath = _write_manifest(tmp_path)
    qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path), "--base-dir", str(tmp_path)],
             runner=_boom_runner)
    s = _read_summary(tmp_path)
    assert s["quality_map"] is None


def test_summary_records_manifest_provenance(tmp_path):
    mpath = _write_manifest(tmp_path)
    qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path), "--base-dir", str(tmp_path)],
             runner=_boom_runner)
    s = _read_summary(tmp_path)
    assert s["meta"]["manifest"].endswith("manifest.json")
    assert s["meta"]["manifest_generated_utc"] == "2026-07-03T00:00:00+00:00"
    assert s["meta"]["n_batches"] == 3


# --------------------------------------------------------------------------- argument / robustness
def test_max_batches_below_one_is_rejected(tmp_path, capsys):
    mpath = _write_manifest(tmp_path)
    rc = qmb.main(["--manifest", mpath, "--status-dir", _status_dir(tmp_path),
                   "--base-dir", str(tmp_path), "--execute", "--max-batches", "0"],
                  runner=_boom_runner)
    assert rc == 2
    assert "max-batches" in capsys.readouterr().err
    # nothing ran, nothing written
    assert not (pathlib.Path(_status_dir(tmp_path)) / "summary.json").exists()


def test_default_status_dir_resolves_under_base_dir(tmp_path):
    # omit --status-dir: summary + ledger must land under <base-dir>/STATUS_ROOT (the default branch)
    m = _manifest_dict(tmp_path)
    _write_report(m["batches"][0]["report_dir"])
    mpath = _write_manifest(tmp_path, m)
    rc = qmb.main(["--manifest", mpath, "--base-dir", str(tmp_path)], runner=_boom_runner)
    assert rc == 0
    assert (pathlib.Path(tmp_path) / qmb.STATUS_ROOT / "summary.json").exists()


def test_load_ledger_tolerates_blank_and_malformed_lines(tmp_path):
    p = tmp_path / "_runner_status.jsonl"
    p.write_text("\n"
                 "not json at all\n"
                 '{"no_file_key": true}\n'  # dict but missing 'file' → skipped
                 + json.dumps({"file": "batch_002_days.txt", "exit_code": 5,
                               "status": qmb.BLOCKED_QUOTA}) + "\n")
    records = qmb.load_ledger(str(p))
    assert [r["file"] for r in records] == ["batch_002_days.txt"]
    assert qmb.ledger_index(records)["batch_002_days.txt"]["exit_code"] == 5


# --------------------------------------------------------------------------- runner contract alignment
def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "run_coinbase_quality_map",
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_coinbase_quality_map.py")
    qm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = qm
    spec.loader.exec_module(qm)
    return qm


def test_recognized_runner_exit_codes_stay_aligned():
    """The orchestrator classifies a batch subprocess by the runner's own exit-code convention. If
    the runner renumbers QUOTA_REFUSED / NATIVE_UNAVAILABLE, this pins the orchestrator's copies."""
    qm = _load_runner()
    assert qmb.QUOTA_REFUSED_EXIT == qm.QUOTA_REFUSED_EXIT
    assert qmb.NATIVE_UNAVAILABLE_EXIT == qm.NATIVE_UNAVAILABLE_EXIT


def test_report_name_matches_the_runner_fixed_output(tmp_path):
    """The runner writes a FIXED coinbase_quality_map.json under --out-dir; the orchestrator's
    completion marker must be that exact filename."""
    assert qmb.REPORT_NAME == "coinbase_quality_map.json"


def test_status_root_matches_planner_report_root():
    """The orchestrator's summary/ledger root (STATUS_ROOT) must equal the planner's per-batch report
    root, or a default-path run would look for batch reports where the planner never wrote them."""
    spec = importlib.util.spec_from_file_location(
        "plan_coinbase_quality_map_batches",
        pathlib.Path(__file__).resolve().parents[1] / "scripts"
        / "plan_coinbase_quality_map_batches.py")
    pm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = pm
    spec.loader.exec_module(pm)
    assert qmb.STATUS_ROOT == pm.REPORT_ROOT
    # and the default manifest path is the planner's default out-dir + manifest name
    assert qmb.DEFAULT_MANIFEST == f"{pm.DEFAULT_OUT_DIR}/{pm.MANIFEST_NAME}"
