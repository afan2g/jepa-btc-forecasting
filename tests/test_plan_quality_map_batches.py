"""Offline tests for the quota-budgeted Coinbase quality-map batch planner
(`scripts/plan_coinbase_quality_map_batches.py`, docs/data.md §5a-QualityMap staging / §6 quota).

Drives the planner end-to-end on SYNTHETIC usable-calendar JSON — no vendor I/O anywhere (the
planner is stdlib-only and never opens a Lake/CoinAPI session). Covers deterministic day selection
and batching, the GB budget, explicit gap/fill and excluded-day handling, the manifest contract
(including the run_coinbase_quality_map.py command template), and clear failures on bad input."""
import datetime as dt
import importlib.util
import json
import pathlib
import sys
import types

import pytest

# scripts/ is not a package — load the script module by path (same pattern as test_quality_map).
_SPEC = importlib.util.spec_from_file_location(
    "plan_coinbase_quality_map_batches",
    pathlib.Path(__file__).resolve().parents[1] / "scripts"
    / "plan_coinbase_quality_map_batches.py",
)
pm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pm
_SPEC.loader.exec_module(pm)


# --------------------------------------------------------------------------- synthetic calendars
# 5 present Lake days deliberately UNSORTED (selection must sort), plus one book-gap fill day, one
# trade-only fill day, and one excluded (Binance-gap) day — the three categories the plan must keep
# apart. Mirrors the real data/usable_calendar.json field shapes (§5b).
LAKE_ALL = ["2025-01-03", "2025-01-01", "2025-01-02", "2025-01-05", "2025-01-04"]
FILL = {"2025-01-10": {"book": True, "trades": True},    # Lake book absent → CoinAPI book fill
        "2025-01-11": {"book": False, "trades": True}}   # trade-only gap → Lake book present
EXCLUDED = {"2025-01-20": ["missing:binF_book"]}


def _calendar_dict(**overrides) -> dict:
    cal = {"lake_all_days": list(LAKE_ALL),
           "coinbase_fill_days": {k: dict(v) for k, v in FILL.items()},
           "excluded_days_by_reason": {k: list(v) for k, v in EXCLUDED.items()}}
    cal.update(overrides)
    return cal


def _write_calendar(tmp_path, cal=None) -> str:
    path = tmp_path / "usable_calendar.json"
    path.write_text(json.dumps(cal if cal is not None else _calendar_dict()))
    return str(path)


def _main(tmp_path, *extra, cal=None) -> tuple[int, str]:
    """Run the planner CLI against a synthetic calendar; returns (exit_code, out_dir)."""
    cal_path = _write_calendar(tmp_path, cal)
    out_dir = str(tmp_path / "batches")
    rc = pm.main(["--calendar", cal_path, "--out-dir", out_dir,
                  "--max-gb-per-batch", "1.0", "--gb-per-day", "0.48", *extra])
    return rc, out_dir


# --------------------------------------------------------------------------- day selection
def test_batch_days_are_sorted_and_deterministic():
    sel = pm.select_days(_calendar_dict())
    assert sel["batch_days"] == sorted(LAKE_ALL)
    assert pm.select_days(_calendar_dict()) == sel  # same input ⇒ identical plan


def test_fill_days_split_book_gap_vs_trade_only_and_never_batched():
    sel = pm.select_days(_calendar_dict())
    assert sel["fill_days_book_gap"] == ["2025-01-10"]
    assert sel["fill_days_trade_only"] == ["2025-01-11"]
    assert not set(sel["batch_days"]) & set(FILL)


def test_excluded_days_are_recorded_and_never_batched():
    sel = pm.select_days(_calendar_dict())
    assert sel["excluded_days"] == EXCLUDED
    assert "2025-01-20" not in sel["batch_days"]


def test_lake_day_overlapping_excluded_or_fill_is_dropped_and_recorded():
    # Defensive: a contradictory calendar (a "present" Lake day that is also excluded / a fill day)
    # must not be silently batched — the overlap is dropped and surfaced in the selection.
    cal = _calendar_dict(lake_all_days=LAKE_ALL + ["2025-01-10", "2025-01-20"])
    sel = pm.select_days(cal)
    assert sel["batch_days"] == sorted(LAKE_ALL)
    assert sel["lake_days_dropped_as_excluded_or_fill"] == ["2025-01-10", "2025-01-20"]


# --------------------------------------------------------------------------- batching / budget
def test_batches_respect_gb_budget_and_preserve_order():
    days = sorted(LAKE_ALL)
    batches = pm.plan_batches(days, max_gb_per_batch=1.0, gb_per_day=0.48)
    assert batches == [days[0:2], days[2:4], days[4:5]]  # floor(1.0/0.48) = 2 days/batch
    for b in batches:
        assert len(b) * 0.48 <= 1.0


def test_budget_boundary_is_exact_days_per_batch():
    days = [f"2025-02-{i:02d}" for i in range(1, 7)]
    # 0.96/0.48 = exactly 2 days — float noise must not knock this down to 1
    batches = pm.plan_batches(days, max_gb_per_batch=0.96, gb_per_day=0.48)
    assert [len(b) for b in batches] == [2, 2, 2]


def test_budget_smaller_than_one_day_fails_clearly():
    with pytest.raises(ValueError, match="at least one day"):
        pm.plan_batches(sorted(LAKE_ALL), max_gb_per_batch=0.3, gb_per_day=0.48)


def test_non_positive_gb_per_day_fails_clearly():
    with pytest.raises(ValueError, match="gb.per.day"):
        pm.plan_batches(sorted(LAKE_ALL), max_gb_per_batch=1.0, gb_per_day=0.0)


# --------------------------------------------------------------------------- calendar validation
def test_missing_calendar_file_fails_clearly(tmp_path):
    missing = str(tmp_path / "nope.json")
    with pytest.raises(ValueError, match="nope.json"):
        pm.load_calendar(missing)


def test_missing_required_field_fails_clearly(tmp_path):
    cal = _calendar_dict()
    del cal["lake_all_days"]
    with pytest.raises(ValueError, match="lake_all_days"):
        pm.load_calendar(_write_calendar(tmp_path, cal))


def test_wrong_field_type_fails_clearly(tmp_path):
    cal = _calendar_dict(coinbase_fill_days=["2025-01-10"])  # list, not the {day: {...}} dict
    with pytest.raises(ValueError, match="coinbase_fill_days"):
        pm.load_calendar(_write_calendar(tmp_path, cal))


def test_invalid_day_string_fails_clearly(tmp_path):
    cal = _calendar_dict(lake_all_days=LAKE_ALL + ["2025-13-99"])
    with pytest.raises(ValueError, match="2025-13-99"):
        pm.load_calendar(_write_calendar(tmp_path, cal))


@pytest.mark.parametrize("bad_value", ["book", ["book"], None])
def test_malformed_fill_day_value_fails_clearly(tmp_path, bad_value):
    # a corrupted/hand-edited fill entry must fail at load, not crash later in select_days
    cal = _calendar_dict(coinbase_fill_days={"2025-01-10": bad_value})
    with pytest.raises(ValueError, match="coinbase_fill_days"):
        pm.load_calendar(_write_calendar(tmp_path, cal))


def test_main_reports_calendar_errors_as_exit_code_2(tmp_path, capsys):
    rc = pm.main(["--calendar", str(tmp_path / "nope.json")])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_default_budget_matches_the_documented_planning_target(tmp_path):
    """Pins the quota-safety defaults (250 GB/batch at 0.48 GB/day → 520 days/batch, docs §5a/§6)
    through both the argparse wiring and the defaults-path manifest — a regressed constant would
    otherwise plan wrong-sized batches on every default invocation and no test would notice."""
    args = pm.parse_args([])
    assert args.calendar == "data/usable_calendar.json"
    assert args.out_dir == "data/tmp/coinbase_quality_map_batches"
    assert args.max_gb_per_batch == 250.0
    assert args.gb_per_day == 0.48

    out_dir = str(tmp_path / "batches")
    rc = pm.main(["--calendar", _write_calendar(tmp_path), "--out-dir", out_dir])
    assert rc == 0
    m = json.loads((pathlib.Path(out_dir) / "manifest.json").read_text())
    assert m["meta"]["max_gb_per_batch"] == 250.0
    assert m["meta"]["gb_per_day"] == 0.48
    assert m["meta"]["days_per_batch"] == 520


# --------------------------------------------------------------------------- writing the plan
def test_write_plan_emits_batch_files_and_manifest(tmp_path):
    rc, out_dir = _main(tmp_path)
    assert rc == 0
    out = pathlib.Path(out_dir)
    files = sorted(p.name for p in out.glob("batch_*_days.txt"))
    assert files == ["batch_001_days.txt", "batch_002_days.txt", "batch_003_days.txt"]
    # one day per line, ascending across batches — the format resolve_days() accepts
    assert (out / "batch_001_days.txt").read_text() == "2025-01-01\n2025-01-02\n"
    assert (out / "batch_003_days.txt").read_text() == "2025-01-05\n"
    json.loads((out / "manifest.json").read_text())  # strict JSON


def test_manifest_summarizes_the_plan(tmp_path):
    rc, out_dir = _main(tmp_path)
    m = json.loads((pathlib.Path(out_dir) / "manifest.json").read_text())
    assert m["meta"]["input_calendar"].endswith("usable_calendar.json")
    assert m["meta"]["generated_utc"]  # timestamp present (batch FILES are the deterministic part)
    assert m["meta"]["gb_per_day"] == 0.48
    assert m["meta"]["max_gb_per_batch"] == 1.0
    assert m["summary"]["n_batch_days"] == 5
    assert m["summary"]["n_batches"] == 3
    assert m["summary"]["est_gb_per_batch"] == [0.96, 0.96, 0.48]
    assert m["summary"]["total_est_gb"] == 2.4
    assert m["summary"]["n_fill_days_book_gap"] == 1
    assert m["summary"]["n_fill_days_trade_only"] == 1
    assert m["summary"]["n_excluded_days"] == 1
    assert m["skipped"]["fill_days_book_gap"] == ["2025-01-10"]
    assert m["skipped"]["excluded_days_by_reason"] == EXCLUDED


def test_manifest_contains_the_runner_command_template(tmp_path):
    rc, out_dir = _main(tmp_path)
    m = json.loads((pathlib.Path(out_dir) / "manifest.json").read_text())
    tmpl = m["meta"]["command_template"]
    for part in ("scripts/run_coinbase_quality_map.py", "--engine native", "--no-cold-ab",
                 "--days-file {days_file}", "--usable-calendar {calendar}",
                 "--out-dir {report_dir}", "--allow-broad"):
        assert part in tmpl
    # each batch entry carries the concrete command for ITS days file
    cmd0 = m["batches"][0]["command"]
    assert cmd0 == tmpl.format(days_file=str(pathlib.Path(out_dir) / "batch_001_days.txt"),
                               calendar=m["meta"]["input_calendar"],
                               report_dir=m["batches"][0]["report_dir"])


def test_batch_commands_use_distinct_report_dirs_and_the_planned_calendar(tmp_path):
    """The runner writes a FIXED coinbase_quality_map.json under --out-dir, and defaults
    --usable-calendar to data/usable_calendar.json — so each generated command must pin its own
    report dir (staged batches must not overwrite each other's artifact) and the exact calendar
    the plan was built from (Codex review, PR #12)."""
    rc, out_dir = _main(tmp_path)
    m = json.loads((pathlib.Path(out_dir) / "manifest.json").read_text())
    report_dirs = [b["report_dir"] for b in m["batches"]]
    assert len(set(report_dirs)) == len(report_dirs)  # distinct per batch
    for b in m["batches"]:
        stem = b["file"].removesuffix("_days.txt")
        assert b["report_dir"].endswith(stem)
        assert f"--out-dir {b['report_dir']}" in b["command"]
        assert f"--usable-calendar {m['meta']['input_calendar']}" in b["command"]


def test_batch_files_are_byte_deterministic_across_runs(tmp_path):
    _, out_dir = _main(tmp_path)
    first = {p.name: p.read_bytes() for p in pathlib.Path(out_dir).glob("batch_*_days.txt")}
    rc, _ = _main(tmp_path)
    assert rc == 0
    second = {p.name: p.read_bytes() for p in pathlib.Path(out_dir).glob("batch_*_days.txt")}
    assert first == second


def test_stale_batch_files_from_a_previous_plan_are_removed(tmp_path):
    out = tmp_path / "batches"
    out.mkdir()
    stale = out / "batch_009_days.txt"
    stale.write_text("2020-01-01\n")
    rc, _ = _main(tmp_path)
    assert rc == 0
    assert not stale.exists()  # a re-plan must not leave runnable stale batches behind


def test_dry_run_prints_the_plan_and_writes_nothing(tmp_path, capsys):
    rc, out_dir = _main(tmp_path, "--dry-run")
    assert rc == 0
    assert not pathlib.Path(out_dir).exists()
    printed = capsys.readouterr().out
    assert "DRY RUN" in printed
    assert "batch_001_days.txt" in printed


# --------------------------------------------------------------------------- runner contract
def test_batch_files_are_loadable_by_the_quality_map_runner(tmp_path):
    """The generated days-file must parse through run_coinbase_quality_map.py::resolve_days —
    the actual consumer contract, not just 'one day per line'."""
    spec = importlib.util.spec_from_file_location(
        "run_coinbase_quality_map",
        pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_coinbase_quality_map.py")
    qm = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = qm
    spec.loader.exec_module(qm)

    _, out_dir = _main(tmp_path)
    args = types.SimpleNamespace(days_file=str(pathlib.Path(out_dir) / "batch_001_days.txt"),
                                 days=None, include_gap_days=0, usable_calendar="unused")
    assert qm.resolve_days(args) == [dt.date(2025, 1, 1), dt.date(2025, 1, 2)]
