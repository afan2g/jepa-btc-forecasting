"""Offline tests for the Binance Lake download batch planner (scripts/plan_lake_binance_batches.py).

Drives the planner end-to-end on a fixed day RANGE and on a SYNTHETIC calendar JSON — no vendor I/O
anywhere (the planner is stdlib-only and never opens a Lake session). Covers deterministic batching,
the GB budget + runner-gate cap, byte-determinism, the manifest/command contract, and clear failures.
Also pins the planner's quota-gate constant to ingest.lake_binance.full_pull_gb_per_day() (the planner
must stay stdlib-only, so it copies the rate — kept aligned by this contract test, mirroring the
Coinbase planner's RUNNER_* contract test)."""
import datetime as dt
import importlib.util
import json
import pathlib
import sys

import pytest

from ingest import lake_binance as lb

# scripts/ is not a package — load by path (same pattern as test_plan_quality_map_batches).
_SPEC = importlib.util.spec_from_file_location(
    "plan_lake_binance_batches",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "plan_lake_binance_batches.py")
pm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pm
_SPEC.loader.exec_module(pm)

RATE = 1.2278  # == full_pull_gb_per_day(): both instruments, all feeds + `book` seed


def _run(tmp_path, *extra):
    out_dir = str(tmp_path / "batches")
    rc = pm.main(["--start", "2026-04-01", "--end", "2026-04-06",
                  "--max-gb-per-batch", "2.5", "--gb-per-day", str(RATE),
                  "--out-dir", out_dir, *extra])
    return rc, out_dir


# --------------------------------------------------------------------------- day resolution
def test_daterange_is_sorted_inclusive():
    assert pm.daterange("2026-04-01", "2026-04-03") == ["2026-04-01", "2026-04-02", "2026-04-03"]


def test_daterange_end_before_start_fails():
    with pytest.raises(pm.PlanError):
        pm.daterange("2026-04-03", "2026-04-01")


def test_daterange_invalid_day_fails():
    with pytest.raises(pm.PlanError):
        pm.daterange("2026-13-99", "2026-13-99")


# --------------------------------------------------------------------------- batching / budget
def test_batches_respect_budget_and_preserve_order():
    days = pm.daterange("2026-04-01", "2026-04-06")
    batches = pm.plan_batches(days, max_gb_per_batch=2.5, gb_per_day=RATE)
    assert batches == [days[0:2], days[2:4], days[4:6]]  # floor(2.5/1.2278) = 2 days/batch
    for b in batches:
        assert len(b) * RATE <= 2.5


def test_budget_smaller_than_one_day_fails():
    with pytest.raises(pm.PlanError, match="at least one day"):
        pm.plan_batches(["2026-04-01"], max_gb_per_batch=0.5, gb_per_day=RATE)


def test_non_positive_gb_per_day_fails():
    with pytest.raises(pm.PlanError, match="gb.per.day"):
        pm.plan_batches(["2026-04-01"], max_gb_per_batch=2.5, gb_per_day=0.0)


def test_batch_size_capped_by_runner_gate():
    # a --gb-per-day below the runner's fixed rate must not plan a batch the downloader's own quota
    # gate cannot run: cap = floor((300 - 10) / 1.2278) = 236 days, REGARDLESS of --allow-broad.
    days = [(dt.date(2026, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(300)]
    batches = pm.plan_batches(days, max_gb_per_batch=1000.0, gb_per_day=0.5)  # requested 2000 days
    assert [len(b) for b in batches] == [236, 64]
    for b in batches:
        assert len(b) * pm.RUNNER_GB_PER_DAY <= pm.RUNNER_QUOTA_GB - pm.RUNNER_HEADROOM_GB


# --------------------------------------------------------------------------- writing the plan
def test_write_plan_emits_batch_files_and_manifest(tmp_path):
    rc, out_dir = _run(tmp_path)
    assert rc == 0
    out = pathlib.Path(out_dir)
    files = sorted(p.name for p in out.glob("batch_*_days.txt"))
    assert files == ["batch_001_days.txt", "batch_002_days.txt", "batch_003_days.txt"]
    assert (out / "batch_001_days.txt").read_text() == "2026-04-01\n2026-04-02\n"
    assert (out / "batch_003_days.txt").read_text() == "2026-04-05\n2026-04-06\n"
    json.loads((out / "manifest.json").read_text())  # strict JSON


def test_batch_files_are_byte_deterministic_across_runs(tmp_path):
    _, out_dir = _run(tmp_path)
    first = {p.name: p.read_bytes() for p in pathlib.Path(out_dir).glob("batch_*_days.txt")}
    rc, _ = _run(tmp_path)
    assert rc == 0
    second = {p.name: p.read_bytes() for p in pathlib.Path(out_dir).glob("batch_*_days.txt")}
    assert first == second


def test_dry_run_writes_nothing(tmp_path, capsys):
    rc, out_dir = _run(tmp_path, "--dry-run")
    assert rc == 0
    assert not pathlib.Path(out_dir).exists()
    assert "DRY RUN" in capsys.readouterr().out


def test_stale_batch_files_removed(tmp_path):
    out = tmp_path / "batches"
    out.mkdir()
    stale = out / "batch_009_days.txt"
    stale.write_text("2020-01-01\n")
    _run(tmp_path)
    assert not stale.exists()  # a re-plan must not leave runnable stale batches behind


# --------------------------------------------------------------------------- manifest contract
def test_manifest_summarizes_plan(tmp_path):
    _, out_dir = _run(tmp_path)
    m = json.loads((pathlib.Path(out_dir) / "manifest.json").read_text())
    assert m["summary"]["n_batch_days"] == 6
    assert m["summary"]["n_batches"] == 3
    assert m["meta"]["gb_per_day"] == RATE
    assert m["meta"]["max_gb_per_batch"] == 2.5
    assert m["meta"]["generated_utc"]  # timestamp present (batch FILES are the deterministic part)
    assert m["meta"]["runner_gb_per_day"] == pm.RUNNER_GB_PER_DAY


def test_manifest_command_targets_downloader_with_range_and_both_instruments(tmp_path):
    _, out_dir = _run(tmp_path)
    m = json.loads((pathlib.Path(out_dir) / "manifest.json").read_text())
    cmd0 = m["batches"][0]["command"]
    assert "ingest/download_lake_binance.py" in cmd0
    assert "--start 2026-04-01" in cmd0
    assert "--end 2026-04-02" in cmd0
    assert "--allow-broad" in cmd0
    assert "binance-perp" in cmd0 and "binance-spot" in cmd0  # both instruments per batch


# --------------------------------------------------------------------------- calendar day source
def test_calendar_day_source_is_sorted(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"binance_present_days": ["2026-04-03", "2026-04-01", "2026-04-02"]}))
    out_dir = str(tmp_path / "b")
    rc = pm.main(["--calendar", str(cal), "--out-dir", out_dir,
                  "--max-gb-per-batch", "2.5", "--gb-per-day", str(RATE)])
    assert rc == 0
    assert (pathlib.Path(out_dir) / "batch_001_days.txt").read_text() == "2026-04-01\n2026-04-02\n"


def test_missing_calendar_file_exit_2(tmp_path, capsys):
    rc = pm.main(["--calendar", str(tmp_path / "nope.json"), "--out-dir", str(tmp_path / "b")])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


def test_missing_calendar_field_fails(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps({"other": []}))
    with pytest.raises(pm.PlanError, match="binance_present_days"):
        pm.load_calendar_days(str(cal), "binance_present_days")


def test_neither_range_nor_calendar_exit_2(tmp_path, capsys):
    rc = pm.main(["--out-dir", str(tmp_path / "b")])
    assert rc == 2
    assert "ERROR" in capsys.readouterr().err


# --------------------------------------------------------------------------- defaults + contract
def test_default_constants():
    args = pm.parse_args([])
    assert args.max_gb_per_batch == 250.0
    assert args.out_dir == "data/tmp/binance_download_batches"
    assert args.gb_per_day == pytest.approx(pm.RUNNER_GB_PER_DAY)


def test_runner_gate_constants_match_lake_binance():
    """The stdlib-only planner copies the downloader's quota-gate rate/cap/headroom (it cannot import
    lake_binance, which pulls pandas). Pin the copies to the source of truth."""
    assert pm.RUNNER_GB_PER_DAY == pytest.approx(lb.full_pull_gb_per_day())
    assert pm.RUNNER_QUOTA_GB == lb.QUOTA_GB
    assert pm.RUNNER_HEADROOM_GB == lb.DEFAULT_HEADROOM_GB
    assert set(pm.INSTRUMENT_KEYS) == set(lb.INSTRUMENTS)  # both instruments pulled per day
