"""Offline tests for the thin Lake CLI wrapper (`ingest/validate_trade_feeds.py`,
docs/data.md §5b / §10 "trade validation breadth", plan
docs/superpowers/plans/2026-07-02-trade-validation-breadth-plan.md §9 case 13 / Phase-1b Task 2).

The wrapper loads Crypto Lake `trades` per `(venue, day)` and feeds the *pure* source-agnostic
`ingest.trade_checks.validate_trade_frame` unchanged. Every test here drives the wrapper with an
INJECTED load seam (a synthetic frame / None / a raiser) so CI never imports lakeapi/boto3 and never
touches a vendor — the same discipline as tests/test_trade_checks.py and tests/test_quality_map.py.
The one live-path guard (case 13) asserts the module import and the synthetic path stay vendor-free.
"""
import json
import pathlib
import random
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from ingest import trade_checks as tc
from ingest import validate_trade_feeds as vf

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
FIXED_UTC = "2026-07-02T00:00:00+00:00"


# --------------------------------------------------------------------------- synthetic loader frames
def _clean_full_day(start="2025-06-01T00:00:00", n=2400):
    """A clean full-day normalized `trades` frame (~100 rows/UTC-hour) → classifies `pass` for the
    matching day. Mirrors tests/test_trade_checks.py `_clean_full_day`."""
    base = pd.Timestamp(start)
    step = 86_400_000 // n
    origin = base + pd.to_timedelta(np.arange(n) * step, unit="ms")
    return pd.DataFrame({
        "origin_time": origin,
        "received_time": origin + pd.to_timedelta(160, unit="ms"),
        "price": np.full(n, 60000.0),
        "quantity": np.full(n, 0.01),
        "side": np.where(np.arange(n) % 2, "buy", "sell"),
        "trade_id": np.arange(n, dtype="int64"),
    })


def _raiser(*_a, **_k):
    raise AssertionError("vendor seam must not be called on this path")


def _fake_session():
    return object()


def _used(gb):
    return lambda _s: {"downloaded_gb": gb}


def _write_cal(tmp_path, cal):
    p = tmp_path / "usable_calendar.json"
    p.write_text(json.dumps(cal))
    return str(p)


# --------------------------------------------------------------------------- 1. import is vendor-free
def test_import_does_not_touch_lakeapi_or_boto3():
    # Importing the wrapper must NOT import lakeapi/boto3 (they load only inside the live seam).
    # Assert on the DELTA of a reload, not global sys.modules membership: another test module in a
    # full-suite run (e.g. test_verify_script → scripts.verify_book_delta_v2) imports the vendor
    # clients at module scope, so a bare `"lakeapi" not in sys.modules` is order-dependent. Re-running
    # the wrapper's module body must add NEITHER client. (The fresh-interpreter subprocess test below
    # is the authoritative, fully-isolated guarantee.)
    import importlib
    before = {m for m in ("lakeapi", "boto3") if m in sys.modules}
    importlib.reload(vf)
    after = {m for m in ("lakeapi", "boto3") if m in sys.modules}
    assert after == before                           # reloading the wrapper imported no vendor client
    assert callable(vf.main)


def test_import_is_vendor_free_in_a_fresh_interpreter():
    # Fully isolated: a fresh interpreter that imports the module must not pull in lakeapi/boto3,
    # regardless of what earlier in-process tests loaded (the strong form of case 13).
    code = (
        "import sys; import ingest.validate_trade_feeds as v; "
        "assert 'lakeapi' not in sys.modules, 'lakeapi imported at module load'; "
        "assert 'boto3' not in sys.modules, 'boto3 imported at module load'; "
        "assert callable(v.main); print('OK')"
    )
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert "OK" in r.stdout


# --------------------------------------------------------------------------- 2. day selection
def test_default_day_selection_is_the_safe_five_day_sample():
    days, mode = vf.resolve_days(vf.parse_args([]), cal=None)
    assert days == list(vf.DEFAULT_SAMPLE_DAYS)
    assert mode == "default_sample"
    assert len(days) == 5
    # 2024-08-06 (a full Coinbase gap) is deliberately included so the one bounded run exercises
    # the coinapi_fill route (plan §3.4).
    assert "2024-08-06" in days


def test_explicit_days_take_precedence_and_dedupe():
    days, mode = vf.resolve_days(vf.parse_args(["--days", "2025-06-01, 2024-08-05 ,2025-06-01"]),
                                 cal=None)
    assert days == ["2025-06-01", "2024-08-05"]      # order preserved, duplicate dropped
    assert mode == "explicit_days"


def test_malformed_explicit_day_is_rejected():
    with pytest.raises(ValueError):
        vf.resolve_days(vf.parse_args(["--days", "2025-13-99"]), cal=None)


def test_days_file_is_read(tmp_path):
    f = tmp_path / "days.txt"
    f.write_text("2025-06-01\n2024-08-05,2025-01-07\n\n")
    days, mode = vf.resolve_days(vf.parse_args(["--days-file", str(f)]), cal=None)
    assert days == ["2025-06-01", "2024-08-05", "2025-01-07"]
    assert mode == "days_file"


def test_explicit_but_empty_selector_is_not_silently_defaulted(tmp_path):
    # `--days ""` (e.g. `--days "$UNSET"`) is an EXPLICIT-but-empty selection: it must NOT fall
    # through to the 5-day default sample (which would pull data on a live run). It resolves to []
    # under the explicit branch and hits the empty-selection guard → main() exit 2.
    days, mode = vf.resolve_days(vf.parse_args(["--days", ""]), cal=None)
    assert days == [] and mode == "explicit_days"
    assert vf.main(["--days", "", "--calendar", str(tmp_path / "none.json")]) == 2
    # same for an explicitly-empty --days-file
    days2, mode2 = vf.resolve_days(vf.parse_args(["--days-file", ""]), cal=None)
    assert days2 == [] and mode2 == "days_file"
    # ...and an explicitly-empty --start (unset `$START`) rejects rather than defaulting
    with pytest.raises(ValueError):
        vf.resolve_days(vf.parse_args(["--start", "", "--end", "2026-06-22"]), cal=None)
    assert vf.main(["--start", "", "--calendar", str(tmp_path / "none.json")]) == 2


def test_day_tokens_are_canonicalized_to_iso_yyyy_mm_dd():
    # date.fromisoformat (≥3.11) accepts basic-format 20240806, but the usable-calendar keys are
    # YYYY-MM-DD; an un-canonicalized token would mis-route a known fill/excluded day as required.
    import datetime as _dt
    try:
        _dt.date.fromisoformat("20240806")
    except ValueError:
        pytest.skip("interpreter's date.fromisoformat rejects basic-format dates (<3.11)")
    days, _ = vf.resolve_days(vf.parse_args(["--days", "20240806,2025-06-01"]), cal=None)
    assert days == ["2024-08-06", "2025-06-01"]          # canonicalized to YYYY-MM-DD
    days2, _ = vf.resolve_days(vf.parse_args(["--days", "2024-08-06,20240806"]), cal=None)
    assert days2 == ["2024-08-06"]                       # two spellings dedupe to one canonical key


def test_range_selection_is_deterministic_and_reproducible():
    cal = {"usable_days": ([f"2024-{m:02d}-{d:02d}" for m in range(7, 13) for d in range(1, 28)]
                           + [f"2026-01-{d:02d}" for d in range(1, 28)]
                           + [f"2026-03-{d:02d}" for d in range(1, 28)])}
    a = vf.parse_args(["--start", "2024-07-01", "--end", "2026-03-31",
                       "--sample-n", "4", "--seed", "3"])
    d1, mode = vf.resolve_days(a, cal)
    d2, _ = vf.resolve_days(a, cal)
    assert d1 == d2                                  # same args → same days (byte-reproducible)
    assert mode == "range_sample"
    # in-range regime-cohort members are always present (incl. the deliberate 2024-08-06 gap)...
    assert {"2025-06-01", "2024-08-05", "2024-08-06", "2025-01-07"} <= set(d1)
    # ...out-of-range cohort members are dropped (end is 2026-03-31)...
    assert "2026-04-15" not in d1 and "2026-06-15" not in d1
    # ...and the seeded random sample adds usable days beyond the cohort
    assert len(d1) > len([d for d in vf.REGIME_COHORT if "2024-07-01" <= d <= "2026-03-31"])


def test_stratified_sample_is_deterministic_and_spans_splits():
    usable = ([f"2024-{m:02d}-{d:02d}" for m in range(7, 13) for d in range(1, 28)]
              + [f"2026-01-{d:02d}" for d in range(1, 28)]
              + [f"2026-04-{d:02d}" for d in range(1, 28)])
    s1 = vf.stratified_sample_days(usable, 6, seed=7)
    s2 = vf.stratified_sample_days(usable, 6, seed=7)
    assert s1 == s2 and len(s1) == 6
    assert set(s1) <= set(usable)                    # only usable days drawn
    # order-independence: a permuted input yields the SAME sample (inputs are sorted internally),
    # so a regression dropping the internal sorted(set(...)) would fail here.
    perm = usable[:]
    random.Random(99).shuffle(perm)
    assert vf.stratified_sample_days(perm, 6, seed=7) == s1
    # stratified: the draw is spread across more than one split span
    spans_hit = {i for d in s1 for i, (lo, hi) in enumerate(vf.SPLIT_SPANS) if lo <= d <= hi}
    assert len(spans_hit) >= 2
    assert vf.stratified_sample_days(usable, 0, seed=7) == []


def test_selection_flag_precedence(tmp_path):
    # §3 first-match precedence: --days > --days-file > --start/--end > default.
    f = tmp_path / "days.txt"
    f.write_text("2024-08-05\n")
    days, mode = vf.resolve_days(vf.parse_args(["--days", "2025-06-01", "--days-file", str(f)]), None)
    assert days == ["2025-06-01"] and mode == "explicit_days"      # --days beats --days-file
    days2, mode2 = vf.resolve_days(
        vf.parse_args(["--days-file", str(f), "--start", "2024-06-22"]), None)
    assert days2 == ["2024-08-05"] and mode2 == "days_file"        # --days-file beats --start


def test_empty_selection_is_rejected(tmp_path):
    # A selection that resolves to ZERO (venue, day) pairs must NOT emit a vacuously-green report
    # (n_days=0 → lake_required_pass/bars_ready True at exit 0, which Phase 4 would consume). run()
    # raises before any session/write; the injected seams raise if touched.
    out = tmp_path / "reports"
    with pytest.raises(ValueError):
        vf.run(vf.parse_args(["--days", ",", "--out-dir", str(out)]),
               load_fn=_raiser, session_factory=_raiser, used_data_fn=_raiser)
    assert not out.exists()
    with pytest.raises(ValueError):                               # whitespace-only venues → empty set
        vf.run(vf.parse_args(["--days", "2025-06-01", "--venues", " , ", "--out-dir", str(out)]),
               load_fn=_raiser, session_factory=_raiser, used_data_fn=_raiser)
    assert not out.exists()


def test_main_maps_bad_input_to_exit_2(tmp_path):
    # main() wires argv → run() and maps a bad-input ValueError (unknown venue / malformed day /
    # empty selection) to exit 2 — all before any vendor seam, so the real default seams are safe.
    nocal = str(tmp_path / "none.json")
    assert vf.main(["--venues", "dogecoin"]) == 2
    assert vf.main(["--days", "2025-13-99", "--calendar", nocal]) == 2
    assert vf.main(["--days", ",", "--calendar", nocal]) == 2


# --------------------------------------------------------------------------- 3. venue selection
def test_venue_selection_default_and_canonical_order():
    assert vf.resolve_venues(None) == list(tc.VENUES)          # default: all 3, canonical order
    # input order does not matter → deterministic canonical order out
    assert vf.resolve_venues("coinbase,binance_perp") == ["binance_perp", "coinbase"]


def test_unknown_venue_is_rejected():
    with pytest.raises(ValueError, match="venue"):
        vf.resolve_venues("binance_perp,dogecoin")


def test_explicit_but_empty_venues_is_rejected_not_expanded_to_all(tmp_path):
    # --venues "" (unset `$VENUES`) must resolve to [] and hit the empty-selection guard, NOT
    # silently expand to all three venues (which would inflate a live quota estimate). Only an
    # OMITTED --venues (None) defaults to all three.
    assert vf.resolve_venues("") == []
    assert vf.resolve_venues(None) == list(tc.VENUES)
    assert vf.main(["--days", "2025-06-01", "--venues", "",
                    "--calendar", str(tmp_path / "none.json")]) == 2


# --------------------------------------------------------------------------- 4. dry-run
def test_dry_run_makes_no_vendor_calls_and_writes_nothing(tmp_path):
    out = tmp_path / "reports"
    args = vf.parse_args(["--days", "2025-06-01", "--out-dir", str(out), "--dry-run"])
    rc = vf.run(args, load_fn=_raiser, session_factory=_raiser, used_data_fn=_raiser)
    assert rc == 0
    assert not (out / "trade_feed_validation.json").exists()
    assert not out.exists()                          # nothing created at all


# --------------------------------------------------------------------------- 5. quota gate → exit 5
def test_broad_plan_over_auto_cap_exits_5_with_no_vendor_calls(tmp_path):
    # 15 days × 3 venues × 0.27 GB/day ≈ 4.05 GB > the 3 GB auto cap, no --allow-broad → refuse.
    # The auto-cap decision is pure (no used_data read), so even --dry-run refuses with zero vendor
    # calls (the injected seams raise if touched).
    days = ",".join(f"2025-06-{d:02d}" for d in range(1, 16))
    # nonexistent calendar → every pair is a required load (no fill/excluded routing), so the
    # estimate is the full 15 × 3 grid regardless of the repo calendar's fill days.
    args = vf.parse_args(["--days", days, "--out-dir", str(tmp_path / "r"), "--dry-run",
                          "--calendar", str(tmp_path / "none.json")])
    rc = vf.run(args, load_fn=_raiser, session_factory=_raiser, used_data_fn=_raiser)
    assert rc == tc.QUOTA_REFUSED_EXIT == 5


def test_quota_headroom_breach_exits_5_before_any_load(tmp_path):
    # A small in-cap pull (required load), but current usage is ~at the monthly cap → the headroom
    # gate refuses regardless of --allow-broad, before any partition load (load seam raises if
    # touched). Nonexistent calendar → the day is a required load, so the session/usage read happens.
    args = vf.parse_args(["--days", "2025-06-01", "--allow-broad", "--out-dir", str(tmp_path / "r"),
                          "--calendar", str(tmp_path / "none.json")])
    rc = vf.run(args, load_fn=_raiser, session_factory=_fake_session,
                used_data_fn=_used(299.0), generated_utc=FIXED_UTC)
    assert rc == 5


# --------------------------------------------------------------------------- 6. fake loader → report
def test_fake_loader_produces_a_deterministic_pass_report(tmp_path):
    out = tmp_path / "reports"
    # calendar anchor (2026-06-20) is deliberately DIFFERENT from the run's --end default
    # (2026-06-22) to pin that source_artifacts records the calendar snapshot, not the CLI end.
    calp = _write_cal(tmp_path, {"usable_days": ["2025-06-01"], "anchor_end": "2026-06-20"})

    def fake_load(_s, _v, day):
        return _clean_full_day(start=f"{day}T00:00:00")

    argv = ["--days", "2025-06-01", "--venues", "binance_spot,coinbase",
            "--calendar", calp, "--out-dir", str(out)]
    rc = vf.run(vf.parse_args(argv), load_fn=fake_load, session_factory=_fake_session,
                used_data_fn=_used(42.0), generated_utc=FIXED_UTC)
    assert rc == 0
    p = out / "trade_feed_validation.json"
    txt = p.read_text()
    assert "NaN" not in txt and "Infinity" not in txt      # strict JSON (allow_nan=False)
    rep = json.loads(txt)
    assert {(r["venue"], r["day"]) for r in rep["days"]} == {("binance_spot", "2025-06-01"),
                                                             ("coinbase", "2025-06-01")}
    assert all(r["status"] == tc.PASS for r in rep["days"])
    assert all(r["vendor_source"] == "lake" for r in rep["days"])
    assert rep["summary"]["gate"]["lake_required_pass"] is True
    assert rep["meta"]["selection_mode"] == "explicit_days"
    # §6 meta schema: dry_run lives inside vendor_api; source_artifacts records the CALENDAR file's
    # anchor (2026-06-20), distinct from the run's --end anchor (meta.anchor_end == 2026-06-22).
    assert rep["meta"]["vendor_api"]["dry_run"] is False
    assert rep["meta"]["anchor_end"] == "2026-06-22"
    assert rep["meta"]["source_artifacts"]["usable_calendar_anchor_end"] == "2026-06-20"

    # byte-for-byte deterministic across a second identical run
    out2 = tmp_path / "reports2"
    argv2 = ["--days", "2025-06-01", "--venues", "binance_spot,coinbase",
             "--calendar", calp, "--out-dir", str(out2)]
    vf.run(vf.parse_args(argv2), load_fn=fake_load, session_factory=_fake_session,
           used_data_fn=_used(42.0), generated_utc=FIXED_UTC)
    assert (out2 / "trade_feed_validation.json").read_text() == txt


# --------------------------------------------------------------------------- 7. missing / fill routes
def test_missing_and_fill_routes_are_surfaced_per_trade_checks(tmp_path):
    out = tmp_path / "reports"
    calp = _write_cal(tmp_path, {
        "usable_days": ["2025-06-01"],
        "coinbase_fill_days": {"2024-08-06": {"book": True, "trades": True}},
        "excluded_days_by_reason": {},
    })
    called = []

    def fake_load(_s, venue, day):
        called.append((venue, day))
        return None                                  # every required partition is absent

    argv = ["--days", "2025-06-01,2024-08-06", "--venues", "binance_spot,coinbase",
            "--calendar", calp, "--out-dir", str(out)]
    rc = vf.run(vf.parse_args(argv), load_fn=fake_load, session_factory=_fake_session,
                used_data_fn=_used(0.0), generated_utc=FIXED_UTC)
    rep = json.loads((out / "trade_feed_validation.json").read_text())
    recs = {(r["venue"], r["day"]): r for r in rep["days"]}

    # Coinbase 2024-08-06 is a trades-fill day → coinapi_fill, and its Lake side is NOT loaded.
    assert recs[("coinbase", "2024-08-06")]["status"] == tc.COINAPI_FILL
    assert ("coinbase", "2024-08-06") not in called
    assert {"day": "2024-08-06", "venue": "coinbase"} in \
        rep["summary"]["gate"]["coinapi_fill_deferred"]
    # A Binance venue on the same day is unaffected → required → loaded → None → missing_partition.
    assert recs[("binance_spot", "2024-08-06")]["status"] == tc.FAIL
    assert tc.MISSING_PARTITION in recs[("binance_spot", "2024-08-06")]["reason_codes"]
    assert ("binance_spot", "2024-08-06") in called
    # A required day with an absent partition fails and lands in blocking_failures.
    assert recs[("binance_spot", "2025-06-01")]["status"] == tc.FAIL
    assert rep["summary"]["gate"]["lake_required_pass"] is False
    assert rc == 0                                   # not --strict → still exit 0


# --------------------------------------------------------------------------- 8. strict exit code
def test_strict_exits_7_but_still_writes_the_report(tmp_path):
    out = tmp_path / "reports"
    argv = ["--days", "2025-06-01", "--venues", "binance_spot", "--strict", "--out-dir", str(out)]
    rc = vf.run(vf.parse_args(argv), load_fn=lambda *_a: None, session_factory=_fake_session,
                used_data_fn=_used(0.0), generated_utc=FIXED_UTC)
    assert rc == tc.VALIDATION_FAILED_EXIT == 7
    assert (out / "trade_feed_validation.json").exists()  # report written BEFORE the nonzero exit


def test_default_run_with_a_blocking_fail_still_exits_0(tmp_path):
    out = tmp_path / "reports"
    argv = ["--days", "2025-06-01", "--venues", "binance_spot", "--out-dir", str(out)]
    rc = vf.run(vf.parse_args(argv), load_fn=lambda *_a: None, session_factory=_fake_session,
                used_data_fn=_used(0.0), generated_utc=FIXED_UTC)
    assert rc == 0


# --------------------------------------------------------------------------- 9. load-error handling
def test_no_files_found_routes_to_missing_partition_not_load_error(tmp_path):
    out = tmp_path / "reports"

    class NoFilesFound(Exception):
        pass

    def gap(_s, _v, _d):
        raise NoFilesFound("No files found for the requested partition")

    argv = ["--days", "2025-06-01", "--venues", "binance_spot", "--out-dir", str(out)]
    rc = vf.run(vf.parse_args(argv), load_fn=gap, session_factory=_fake_session,
                used_data_fn=_used(0.0), generated_utc=FIXED_UTC)
    rec = json.loads((out / "trade_feed_validation.json").read_text())["days"][0]
    assert rec["status"] == tc.FAIL
    assert tc.MISSING_PARTITION in rec["reason_codes"]
    assert tc.LOAD_ERROR not in rec["reason_codes"]
    assert rc == 0


def test_unexpected_load_error_is_surfaced_as_a_fail(tmp_path):
    out = tmp_path / "reports"

    def boom(_s, _v, _d):
        raise RuntimeError("connection reset")

    argv = ["--days", "2025-06-01", "--venues", "binance_spot", "--out-dir", str(out),
            "--calendar", str(tmp_path / "none.json")]
    rc = vf.run(vf.parse_args(argv), load_fn=boom, session_factory=_fake_session,
                used_data_fn=_used(0.0), generated_utc=FIXED_UTC)
    rep = json.loads((out / "trade_feed_validation.json").read_text())
    rec = rep["days"][0]
    assert rec["status"] == tc.FAIL and tc.LOAD_ERROR in rec["reason_codes"]
    # the failed load still made a lakeapi.load_data attempt → it must be counted in the audit meta
    assert rep["meta"]["vendor_api"]["lakeapi_calls"] == 1
    assert rc == 0


# --------------------------------------------------------------------------- 10. GB estimate reuse
def test_gb_estimate_is_over_loaded_pairs_using_trade_checks_footprints():
    # The estimate is over the (venue, day) pairs that actually LOAD, using the pure per-day
    # footprints; fill/excluded pairs (not passed here) cost 0 GB.
    pairs = [("binance_perp", "2025-06-01"), ("coinbase", "2025-06-01")]
    assert vf.estimate_plan_gb(pairs) == \
        tc.TRADES_GB_PER_DAY["binance_perp"] + tc.TRADES_GB_PER_DAY["coinbase"]
    assert vf.estimate_plan_gb([]) == 0.0            # a calendar-only selection loads nothing


def test_calendar_only_selection_needs_no_lake_session(tmp_path):
    # A selection whose pairs are ALL calendar-routed (coinapi_fill/excluded) loads zero Lake
    # partitions, so no session/credentials/usage read may happen — the injected seams all raise if
    # touched. The report is calendar-only (the fill record present, lakeapi_calls=0).
    out = tmp_path / "reports"
    calp = _write_cal(tmp_path, {
        "usable_days": ["2025-06-01"],
        "coinbase_fill_days": {"2024-08-06": {"book": True, "trades": True}},
    })
    args = vf.parse_args(["--days", "2024-08-06", "--venues", "coinbase",
                          "--calendar", calp, "--out-dir", str(out)])
    rc = vf.run(args, load_fn=_raiser, session_factory=_raiser, used_data_fn=_raiser,
                generated_utc=FIXED_UTC)
    assert rc == 0
    rep = json.loads((out / "trade_feed_validation.json").read_text())
    assert rep["days"][0]["status"] == tc.COINAPI_FILL
    assert rep["meta"]["vendor_api"]["lakeapi_calls"] == 0
    assert rep["summary"]["gate"]["coinapi_fill_deferred"] == \
        [{"day": "2024-08-06", "venue": "coinbase"}]
