"""Offline tests for the Coinbase backfill review-manifest tool
(`scripts/review_coinbase_backfill_manifest.py`, docs/data.md §5a-QualityMap;
design docs/superpowers/specs/2026-07-03-coinbase-backfill-manifest-review-design.md).

Drives the tool end-to-end on SYNTHETIC plan/report/calendar JSON — no vendor I/O anywhere
(the tool is stdlib-only and never opens a Lake/CoinAPI session). Covers the enum alignment
contract, calendar/cost helpers, validation, the per-day record builder, plan-driven
completeness, consistency/drift/fill-availability checks, manifest assembly, and the CLI
(readiness/inspection modes, fail-closed exit codes, deterministic output)."""
import datetime as _dt
import importlib.util as _ilu
import json
import os
import pathlib as _pl
import sys

import pytest

# scripts/ is not a package — load the script module by path (same pattern as test_quality_map).
_SPEC = _ilu.spec_from_file_location(
    "review_coinbase_backfill_manifest",
    _pl.Path(__file__).resolve().parents[1] / "scripts"
    / "review_coinbase_backfill_manifest.py")
rv = _ilu.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rv
_SPEC.loader.exec_module(rv)


def _load_runner():
    """Load scripts/run_coinbase_quality_map.py by path (needs pandas/numpy — present in the venv)."""
    spec = _ilu.spec_from_file_location(
        "run_coinbase_quality_map",
        _pl.Path(__file__).resolve().parents[1] / "scripts" / "run_coinbase_quality_map.py")
    qm = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = qm
    spec.loader.exec_module(qm)
    return qm


# =========================================================================== synthetic inputs
def _calendar(**overrides) -> dict:
    cal = {
        "anchor_end": "2026-06-22",
        "lake_all_days": ["2025-01-01", "2025-01-02"],
        "usable_days": ["2025-01-01", "2025-01-02", "2025-01-10", "2025-01-11"],
        "coinbase_fill_days": {
            "2025-01-10": {"book": True, "trades": True},    # book gap + trades
            "2025-01-11": {"book": False, "trades": True},   # trade-only (book present)
        },
        "excluded_days_by_reason": {"2025-01-20": ["missing:binF_book"]},
        "fill_status": {
            "2025-01-10": {"book": {"present": True, "mb": 1000.0, "ok": True},
                           "trades": {"present": True, "mb": 30.0, "ok": True},
                           "error": False, "reason": "", "ok": True},
            "2025-01-11": {"book": None,
                           "trades": {"present": True, "mb": 20.0, "ok": True},
                           "error": False, "reason": "", "ok": True},
        },
        "fill_days_unfillable": [],
        "fill_days_probe_error": [],
        "backfill_verified": True,
    }
    cal.update(overrides)
    return cal


def _fill_block(needs_fill, why, fill_profile=None, full_day_reason=None,
                fill_segments=None, seams=None, seam_policy=None) -> dict:
    return {"needs_fill": needs_fill, "why": why, "fill_profile": fill_profile,
            "full_day_reason": full_day_reason, "fill_segments": fill_segments,
            "seams": seams, "seam_policy": seam_policy}


def _seg_iso(ns):
    secs, rem = divmod(ns, 1_000_000_000)
    b = _dt.datetime.fromtimestamp(secs, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{b}.{rem:09d}Z" if rem else f"{b}Z"


def _day_bounds(day):
    d = _dt.date.fromisoformat(day)
    o = int(_dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc).timestamp()) * 1_000_000_000
    return o, o + 86_400 * 1_000_000_000


def _full_day_seg(day, source="coinapi", reason="quality_over_usable_bar"):
    """A single whole-day segment that genuinely partitions the day (independent of the module's
    own bounds math — a divergence would surface as a validation failure in the ready tests)."""
    o, e = _day_bounds(day)
    return {"source": source, "start_ts": o, "start_iso": _seg_iso(o), "end_ts": e,
            "end_iso": _seg_iso(e), "reason": reason}


def _day(day, classification, coinapi_fill, *, trusted=(None, None),
         calendar=None, reasons=None, fillable=None) -> dict:
    return {
        "day": day, "classification": classification, "reasons": reasons or [],
        "lake_book_delta_v2_present": classification != "missing_needs_coinapi",
        "quality": {"grid_ms": 1000, "trusted_lake_start_ts": trusted[0],
                    "trusted_lake_end_ts": trusted[1], "n_invalid_runs": 0, "invalid_runs": []},
        "coinapi": {"parquet_local": False, "parquet_path": None, "fillable": fillable},
        "calendar": calendar or {"in_usable_days": True, "in_lake_all_days": True,
                                 "is_coinbase_fill_day": False, "excluded_reason": None},
        "coinapi_fill": coinapi_fill,
    }


def _report(days, **meta_overrides) -> dict:
    meta = {"k": 10, "grid_ms": 1000, "exchange": "COINBASE", "symbol": "BTC-USD",
            "engine": "native", "policy": {"reseed": True, "cold_ab": False},
            "thresholds": {"crossed_usable_max": 0.01,
            "missing_usable_max": 0.02, "thin_usable_max": 0.1, "seed_crossed_frac_max": 0.05},
            "quota": {"ok": True, "reason": "ok", "used_gb_before": 0.26, "used_gb_after": 0.26},
            "generated_utc": "2026-07-01T00:00:00Z"}
    meta.update(meta_overrides)
    counts = {c: 0 for c in rv.CLASSES}
    for d in days:
        counts[d["classification"]] = counts.get(d["classification"], 0) + 1
    return {"meta": meta, "summary": {"n_days": len(days), "counts": counts, "by_class": {},
                                      "coinapi_fill": {"fill_counts": rv._recompute_fill_counts(days)}},
            "days": days}


# trade-only day 2025-01-11 (book present, trades gapped) IS a batch day — the clean report must
# map it; its report calendar context is in_lake_all_days=False, is_coinbase_fill_day=True.
_TRADE_ONLY_CTX = {"in_usable_days": True, "in_lake_all_days": False,
                   "is_coinbase_fill_day": True, "excluded_reason": None}


def _clean_reports():
    """One batch report covering the three present-book batch days of the default calendar:
    2025-01-01 lake_usable, 2025-01-02 degraded → full_day_fill, 2025-01-11 lake_usable
    (trade-only day: book present, so it is mapped like any present day)."""
    return [_report([
        _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[_full_day_seg("2025-01-02")],
                         seams=[], seam_policy={"seam_guard_s": 60.0}),
             fillable=True),   # report-driven fill: CoinAPI availability verified (Fix 1 gate)
        _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
             calendar=_TRADE_ONLY_CTX),
    ])]


def _write_tree(tmp_path, cal=None, reports=None):
    """Write calendar + plan manifest + per-batch reports into tmp_path; return
    (plan_path, calendar_path). One batch per report, days = the report's day list."""
    reports = reports if reports is not None else _clean_reports()
    cal = cal if cal is not None else _calendar()
    cal_path = tmp_path / "usable_calendar.json"
    cal_path.write_text(json.dumps(cal))
    out_dir = tmp_path / "batches"
    out_dir.mkdir()
    report_root = tmp_path / "reports"
    batches = []
    for i, rep in enumerate(reports, start=1):
        stem = f"batch_{i:03d}"
        days = [d["day"] for d in rep["days"]]
        (out_dir / f"{stem}_days.txt").write_text("".join(f"{d}\n" for d in days))
        rdir = report_root / stem
        rdir.mkdir(parents=True)
        (rdir / "coinbase_quality_map.json").write_text(json.dumps(rep))
        batches.append({"file": f"{stem}_days.txt", "n_days": len(days),
                        "first_day": days[0] if days else None,
                        "last_day": days[-1] if days else None, "report_dir": str(rdir)})
    plan = {"meta": {"input_calendar": str(cal_path), "out_dir": str(out_dir),
                     "generated_utc": "2026-07-02T00:00:00Z"},
            "summary": {"n_batches": len(batches)},
            "batches": batches,
            "batched_trade_only_fill_days": ["2025-01-11"],
            "skipped": {"fill_days_book_gap": ["2025-01-10"],
                        "excluded_days_by_reason": cal["excluded_days_by_reason"],
                        "days_dropped_as_excluded_or_book_gap": []}}
    plan_path = tmp_path / "plan_manifest.json"
    plan_path.write_text(json.dumps(plan))
    return str(plan_path), str(cal_path)


# =========================================================================== Task 1: enums
def test_class_enum_aligned_with_runner():
    qm = _load_runner()
    assert rv.CLASSES == qm.CLASSES


def test_fill_profile_enum_aligned_with_stitch_policy():
    import recon.stitch_policy as sp
    assert rv.FULL_DAY_FILL == sp.FULL_DAY_FILL
    assert rv.LAKE_ONLY == sp.LAKE_ONLY
    assert rv.PARTIAL_FILL_PROFILES == sp.PARTIAL_FILL_PROFILES


def test_default_seam_policy_aligned_with_stitch_policy():
    import recon.stitch_policy as sp
    assert rv.DEFAULT_SEAM_POLICY == sp.DEFAULT_SEAM_POLICY.as_dict()


def test_segment_sources_aligned_with_stitch_policy():
    import recon.stitch_policy as sp
    assert rv.SEGMENT_SOURCES == sp.SOURCES


def test_seed_source_unreliable_reason_aligned_with_runner():
    qm = _load_runner()
    assert rv._SEED_SOURCE_UNRELIABLE == qm.SEED_SOURCE_UNRELIABLE


def test_why_codes_cover_every_runner_fill_decision():
    qm = _load_runner()
    seen = {
        qm.coinapi_fill_decision(qm.MISSING_NEEDS_COINAPI, [])["why"],
        qm.coinapi_fill_decision(qm.LAKE_PRESENT_DEGRADED, ["seed_accepted"])["why"],
        qm.coinapi_fill_decision(qm.INCONCLUSIVE, [qm.SEED_SOURCE_UNRELIABLE])["why"],
        qm.coinapi_fill_decision(qm.LAKE_USABLE, ["seed_accepted"])["why"],
        qm.coinapi_fill_decision(qm.EXCLUDED, [])["why"],
        qm.coinapi_fill_decision(qm.INCONCLUSIVE, ["no_seed_snapshots"])["why"],
    }
    assert seen == set(rv.WHY_CODES)


# =========================================================================== Task 2: input helpers
def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib
    p = tmp_path / "x.json"
    p.write_bytes(b'{"a": 1}\n')
    assert rv.sha256_file(str(p)) == hashlib.sha256(b'{"a": 1}\n').hexdigest()


def test_load_json_object_ok(tmp_path):
    p = tmp_path / "x.json"
    p.write_text('{"a": 1}')
    assert rv.load_json_object(str(p), what="thing") == {"a": 1}


def test_load_json_object_missing_file_raises(tmp_path):
    with pytest.raises(rv.ReviewInputError, match="nope.json"):
        rv.load_json_object(str(tmp_path / "nope.json"), what="thing")


def test_load_json_object_bad_json_raises(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("{not json")
    with pytest.raises(rv.ReviewInputError, match="not valid JSON"):
        rv.load_json_object(str(p), what="thing")


def test_load_json_object_non_object_raises(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("[1, 2, 3]")
    with pytest.raises(rv.ReviewInputError, match="must be a JSON object"):
        rv.load_json_object(str(p), what="thing")


def test_load_json_object_rejects_nonfinite_constant(tmp_path):
    # Fix 4: a bare NaN/Infinity token (json.load accepts it by default) fails closed at the boundary
    p = tmp_path / "x.json"
    p.write_text('{"mb": NaN}')
    with pytest.raises(rv.ReviewInputError, match="non-finite"):
        rv.load_json_object(str(p), what="thing")


def test_load_json_object_rejects_overflowed_number(tmp_path):
    # a syntactically valid but overflowing literal (1e999 -> inf) must also fail closed at load
    p = tmp_path / "x.json"
    p.write_text('{"mb": 1e999}')
    with pytest.raises(rv.ReviewInputError, match="non-finite"):
        rv.load_json_object(str(p), what="thing")


def test_fill_status_helpers_tolerate_malformed_record():
    # a scalar/list fill_status[day] must not crash is_fillable/measured_mb (.get on a non-dict)
    cal = _calendar(fill_status={"2025-01-10": 5, "2025-01-11": ["bad"]})
    assert rv.is_fillable(cal, "2025-01-10", "book") is False   # non-dict → unavailable, no crash
    assert rv.measured_mb(cal, "2025-01-11", "trades") is None


def test_load_json_object_rejects_oversized_integer(tmp_path):
    # a huge integer literal (overflows float) must fail closed at load, not crash later
    p = tmp_path / "x.json"
    p.write_text('{"mb": %s}' % ("9" * 400))   # ~400-digit int -> OverflowError on float()
    with pytest.raises(rv.ReviewInputError, match="out-of-range or unparseable"):
        rv.load_json_object(str(p), what="thing")


def test_measured_mb_rejects_boolean():
    # a JSON true/false mb (bool subclasses int) must NOT price a fill at 1.0/0.0 GB
    cal = _calendar(fill_status={"2025-01-10": {"book": {"present": True, "mb": True, "ok": True},
                                                "trades": None, "error": False, "reason": "", "ok": True}})
    assert rv.measured_mb(cal, "2025-01-10", "book") is None
    # so the cost model falls back to the conservative estimate, not 0.001 GB
    assert rv.day_book_gb(cal, "2025-01-10") == (rv.EST_BOOK_GB_PER_DAY, "estimated")


def test_measured_mb_rejects_negative():
    # a negative mb must not lower the apparent cost — fall back to the conservative estimate
    cal = _calendar(fill_status={"2025-01-10": {"book": {"present": True, "mb": -500.0, "ok": True},
                                                "trades": None, "error": False, "reason": "", "ok": True}})
    assert rv.measured_mb(cal, "2025-01-10", "book") is None
    assert rv.day_book_gb(cal, "2025-01-10") == (rv.EST_BOOK_GB_PER_DAY, "estimated")


def test_cost_baseline_uses_estimate_for_invalid_mb(tmp_path):
    # the calendar-gap baseline must also fall back to the estimate for an invalid mb, not zero it
    cal = _calendar()
    cal["fill_status"]["2025-01-10"]["book"]["mb"] = -1.0   # invalid book size for the gap day
    plan_path, cal_path = _write_tree(tmp_path, cal=cal)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    c = m["cost_summary"]
    # baseline book gb for 2025-01-10 is now the estimate (2.27), not 0.0
    trades_gb = 0.03 + 0.02   # 2025-01-10 + 2025-01-11 trades, still measured
    assert c["calendar_gap_baseline_usd"] == round(rv.EST_BOOK_GB_PER_DAY * 1.0 + trades_gb * 3.0, 4)


# =========================================================================== Task 3: calendar
def test_book_gap_and_trade_fill_days():
    cal = _calendar()
    assert rv.book_gap_days(cal) == {"2025-01-10"}
    assert rv.trade_fill_days(cal) == {"2025-01-10", "2025-01-11"}


def test_calendar_batch_days_matches_planner():
    spec = _ilu.spec_from_file_location(
        "plan_coinbase_quality_map_batches",
        _pl.Path(__file__).resolve().parents[1] / "scripts"
        / "plan_coinbase_quality_map_batches.py")
    pm = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = pm
    spec.loader.exec_module(pm)
    cal = _calendar()
    assert rv.calendar_batch_days(cal) == pm.select_days(cal)["batch_days"]
    assert rv.calendar_batch_days(cal) == ["2025-01-01", "2025-01-02", "2025-01-11"]


def test_measured_mb_present_and_absent():
    cal = _calendar()
    assert rv.measured_mb(cal, "2025-01-10", "book") == 1000.0
    assert rv.measured_mb(cal, "2025-01-11", "book") is None
    assert rv.measured_mb(cal, "2025-01-99", "book") is None


def test_is_fillable():
    cal = _calendar()
    assert rv.is_fillable(cal, "2025-01-10", "book") is True
    assert rv.is_fillable(cal, "2025-01-11", "trades") is True
    assert rv.is_fillable(cal, "2025-01-11", "book") is False
    bad = _calendar(fill_days_unfillable=["2025-01-10"])
    assert rv.is_fillable(bad, "2025-01-10", "book") is False
    err = _calendar(fill_status={"2025-01-10": {"book": {"present": True, "mb": 1.0, "ok": True},
                                                "trades": None, "error": True, "reason": "x",
                                                "ok": False}})
    assert rv.is_fillable(err, "2025-01-10", "book") is False


def test_is_fillable_rejects_malformed_error():
    # a malformed error ("true"/1/absent) must be treated as a probe error → unavailable
    for bad_err in ("true", 1):
        cal = _calendar(fill_status={"2025-01-10": {"book": {"present": True, "mb": 1.0, "ok": True},
                                                    "trades": None, "error": bad_err, "reason": "",
                                                    "ok": True}})
        assert rv.is_fillable(cal, "2025-01-10", "book") is False


def test_is_fillable_requires_strict_true():
    # truthy non-bool present/ok (e.g. "ok":"false", a truthy string) must NOT count as fillable
    cal = _calendar(fill_status={"2025-01-10": {"book": {"present": "yes", "ok": "false"},
                                                "trades": {"present": True, "mb": 1.0, "ok": True},
                                                "error": False, "reason": "", "ok": True}})
    assert rv.is_fillable(cal, "2025-01-10", "book") is False
    assert rv.is_fillable(cal, "2025-01-10", "trades") is True


def test_validate_calendar():
    rv.validate_calendar(_calendar(), "cal")   # clean: no raise
    bad_flag = _calendar(coinbase_fill_days={"2025-01-10": {"book": "true", "trades": True}})
    with pytest.raises(rv.ReviewInputError, match="coinbase_fill_days"):
        rv.validate_calendar(bad_flag, "cal")
    # a MISSING fill-day registry must fail closed (else the calendar reads as gap-free and drops fills)
    missing = _calendar()
    del missing["coinbase_fill_days"]
    with pytest.raises(rv.ReviewInputError, match="coinbase_fill_days"):
        rv.validate_calendar(missing, "cal")
    # a missing lake_all_days / excluded_days_by_reason likewise fails closed
    no_lake = _calendar()
    del no_lake["lake_all_days"]
    with pytest.raises(rv.ReviewInputError, match="lake_all_days"):
        rv.validate_calendar(no_lake, "cal")
    # a non-ISO day key would crash _synth_full_day_plan later → reject at validation
    bad_key = _calendar(coinbase_fill_days={"bad-day": {"book": True, "trades": False}})
    with pytest.raises(rv.ReviewInputError, match="invalid day"):
        rv.validate_calendar(bad_key, "cal")
    bad_list = _calendar(lake_all_days=["2025-01-01", 20250102])   # non-string day
    with pytest.raises(rv.ReviewInputError, match="invalid day"):
        rv.validate_calendar(bad_list, "cal")


def test_validate_calendar_allows_null_fill_status():
    # verify_trades_and_calendar WITHOUT --verify-backfill writes fill_status: null (unverified) — a
    # valid snapshot that must pass validation. A present non-null, non-object value is still malformed.
    rv.validate_calendar(_calendar(fill_status=None), "cal")   # no raise
    with pytest.raises(rv.ReviewInputError, match="fill_status"):
        rv.validate_calendar(_calendar(fill_status=["x"]), "cal")


def test_readiness_null_fill_status_blocks_via_availability(tmp_path):
    # an unverified calendar (fill_status null) is NOT a structural error: the CLI still writes an
    # auditable manifest with the calendar book/trade gaps surfaced as availability blockers, and
    # --report-only can downgrade it for inspection.
    plan_path, cal_path = _write_tree(tmp_path, cal=_calendar(fill_status=None))
    out = tmp_path / "m.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.BLOCKING_EXIT
    m = json.loads(out.read_text())
    assert m["meta"]["status"] == "blocking"
    assert "2025-01-10" in m["blockers"]["book_fill_unavailable"]
    assert set(m["blockers"]["trade_fill_unavailable"]) == {"2025-01-10", "2025-01-11"}
    rc2 = rv.main(["--plan-manifest", plan_path, "--out", str(out), "--report-only",
                   "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc2 == 0


def test_readiness_rejects_malformed_calendar_fill_flag(tmp_path):
    cal = _calendar()
    cal["coinbase_fill_days"]["2025-01-11"]["trades"] = "true"   # stringly-typed flag
    plan_path, cal_path = _write_tree(tmp_path, cal=cal)
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(tmp_path / "m.json"),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.INPUT_ERROR_EXIT


# =========================================================================== Task 4: cost
def test_gb_from_mb():
    assert rv.gb_from_mb(1000.0) == 1.0
    assert rv.gb_from_mb(None) is None


def test_day_book_gb_measured_vs_estimated():
    cal = _calendar()
    assert rv.day_book_gb(cal, "2025-01-10") == (1.0, "measured")
    assert rv.day_book_gb(cal, "2025-01-02") == (rv.EST_BOOK_GB_PER_DAY, "estimated")


def test_day_trades_gb_measured_vs_estimated():
    cal = _calendar()
    assert rv.day_trades_gb(cal, "2025-01-11") == (0.02, "measured")
    assert rv.day_trades_gb(cal, "2025-01-02") == (rv.EST_TRADES_GB_PER_DAY, "estimated")


def test_cost_helpers():
    assert rv.book_usd(2.0) == 2.0
    assert rv.trades_usd(2.0) == 6.0


# =========================================================================== Task 5: validation
def test_report_missing_keys():
    assert rv.report_missing_keys({"meta": {}, "summary": {}, "days": []}) == []
    assert set(rv.report_missing_keys({"meta": {}})) == {"summary", "days"}


def test_day_record_issues_clean():
    rec = _day("2025-01-02", "lake_present_degraded",
               _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                           full_day_reason="quality_over_usable_bar",
                           fill_segments=[_full_day_seg("2025-01-02")],
                           seams=[], seam_policy={"seam_guard_s": 60.0}))
    assert rv.day_record_issues(rec) == []


def test_day_record_issues_unknown_enums():
    rec = _day("2025-01-02", "weird_class",
               _fill_block(True, "made_up_why", fill_profile="mystery_profile",
                           full_day_reason="x"))
    issues = rv.day_record_issues(rec)
    assert "unknown_classification:weird_class" in issues
    assert "unknown_why:made_up_why" in issues
    assert "unknown_fill_profile:mystery_profile" in issues


def test_day_record_issues_contradictions():
    r1 = _day("d", "lake_present_degraded", _fill_block(True, "quality_over_usable_bar"))
    assert "needs_fill_without_plan" in rv.day_record_issues(r1)
    r2 = _day("d", "lake_present_degraded",
              _fill_block(True, "quality_over_usable_bar", fill_profile="lake_only"))
    assert "needs_fill_without_plan" in rv.day_record_issues(r2)
    r3 = _day("d", "missing_needs_coinapi",
              _fill_block(True, "lake_book_delta_v2_absent", fill_profile="full_day_fill"))
    assert "full_day_without_reason" in rv.day_record_issues(r3)
    r4 = _day("d", "lake_present_degraded",
              _fill_block(True, "quality_over_usable_bar", fill_profile="leading_partial_fill",
                          full_day_reason="quality_over_usable_bar"))
    assert "full_day_reason_without_full_day_fill" in rv.day_record_issues(r4)
    # the Codex gap: fill_profile=null (the no-plan shape) but full_day_reason set — must be caught
    r4b = _day("d", "lake_usable",
               _fill_block(False, "lake_usable", fill_profile=None,
                           full_day_reason="quality_over_usable_bar"))
    assert "full_day_reason_without_full_day_fill" in rv.day_record_issues(r4b)
    r5 = _day("d", "missing_needs_coinapi",
              _fill_block(True, "lake_book_delta_v2_absent", fill_profile="full_day_fill",
                          full_day_reason="lake_book_delta_v2_absent"), trusted=(10, 20))
    assert "full_day_with_trusted_lake_span" in rv.day_record_issues(r5)
    r6 = _day("d", "lake_usable", _fill_block(False, "lake_usable", fill_profile="full_day_fill"))
    assert "plan_without_needs_fill" in rv.day_record_issues(r6)


def test_day_record_issues_missing_keys():
    rec = {"day": "d"}
    issues = rv.day_record_issues(rec)
    assert "missing_key:classification" in issues
    assert "missing_key:coinapi_fill" in issues


def test_day_record_issues_non_bool_needs_fill():
    # needs_fill=1 equals True under ==, but build_day_record's `is True` would drop the fill
    rec = _day("2025-01-02", "lake_present_degraded",
               {"needs_fill": 1, "why": "quality_over_usable_bar", "fill_profile": None,
                "full_day_reason": None, "fill_segments": None, "seams": None, "seam_policy": None})
    assert any("non_bool_needs_fill" in i for i in rv.day_record_issues(rec))


def test_readiness_blocks_non_bool_needs_fill(tmp_path):
    reports = _clean_reports()
    reports[0]["days"][1]["coinapi_fill"]["needs_fill"] = 1   # 2025-01-02 degraded, numeric needs_fill
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "blocking"
    assert any("non_bool_needs_fill" in x for x in m["blockers"]["inconsistencies"])


def test_day_record_issues_fill_day_requires_stitch_plan():
    # a fill day (needs_fill=True, real profile) missing the executable stitch plan must be flagged
    bad = _day("d", "lake_present_degraded",
               _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                           full_day_reason="quality_over_usable_bar"))  # segments/seams/seam_policy None
    issues = rv.day_record_issues(bad)
    assert "fill_day_missing_fill_segments" in issues
    assert "fill_day_missing_seams" in issues
    assert "fill_day_missing_seam_policy" in issues
    # a complete fill day (seams may be an empty list on a full-day route) is not flagged
    ok = _day("d", "lake_present_degraded",
              _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                          full_day_reason="quality_over_usable_bar",
                          fill_segments=[{"source": "coinapi", "start_ts": 1, "start_iso": "x",
                                          "end_ts": 2, "end_iso": "y", "reason": "r"}],
                          seams=[], seam_policy={"seam_guard_s": 60.0}))
    assert not any(i.startswith("fill_day_missing_") for i in rv.day_record_issues(ok))


def test_day_record_issues_malformed_fill_segments():
    day = "2025-01-02"
    o, e = _day_bounds(day)

    def _seg(st, en, src="coinapi"):
        return {"source": src, "start_ts": st, "start_iso": _seg_iso(st), "end_ts": en,
                "end_iso": _seg_iso(en), "reason": "r"}

    def _issues(segs, profile="full_day_fill", fdr="quality_over_usable_bar", seams=None):
        rec = _day(day, "lake_present_degraded",
                   _fill_block(True, "quality_over_usable_bar", fill_profile=profile,
                               full_day_reason=fdr, fill_segments=segs,
                               seams=seams if seams is not None else [],
                               seam_policy={"seam_guard_s": 60.0}))
        return rv.day_record_issues(rec)

    assert "fill_segments_start_ne_day_open" in _issues([_seg(o + 1000, e)])
    assert "fill_segments_end_ne_day_close" in _issues([_seg(o, e - 1000)])
    assert any("gap_or_overlap" in i for i in
               _issues([_seg(o, o + rv.DAY_NS // 2, "lake"), _seg(o + rv.DAY_NS // 2 + 5, e)],
                       profile="mixed_partial_fill", fdr=None, seams=[o + rv.DAY_NS // 2]))
    assert any("bad_source" in i for i in _issues([_seg(o, e, "binance")]))


def test_day_record_issues_seam_mismatch():
    day = "2025-01-02"
    o, e = _day_bounds(day)
    mid = o + rv.DAY_NS // 2
    segs = [{"source": "lake", "start_ts": o, "start_iso": _seg_iso(o), "end_ts": mid,
             "end_iso": _seg_iso(mid), "reason": "r"},
            {"source": "coinapi", "start_ts": mid, "start_iso": _seg_iso(mid), "end_ts": e,
             "end_iso": _seg_iso(e), "reason": "r"}]

    def _mk(seams):
        return _day(day, "lake_present_degraded",
                    _fill_block(True, "quality_over_usable_bar", fill_profile="trailing_partial_fill",
                                fill_segments=segs, seams=seams, seam_policy={"seam_guard_s": 60.0}))

    assert any("seams_mismatch" in i for i in rv.day_record_issues(_mk([])))      # stale/empty seam
    assert not any("seams_mismatch" in i for i in rv.day_record_issues(_mk([mid])))  # correct seam


def test_day_record_issues_fill_plan_needs_coinapi_segment():
    # Fix 2: a full_day_fill whose only whole-day segment is source=lake pulls nothing from CoinAPI
    day = "2025-01-02"
    bad = _day(day, "lake_present_degraded",
               _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                           full_day_reason="quality_over_usable_bar",
                           fill_segments=[_full_day_seg(day, source="lake")],
                           seams=[], seam_policy={"seam_guard_s": 60.0}))
    assert "fill_day_missing_coinapi_segment" in rv.day_record_issues(bad)
    # a full_day_fill mixing coinapi + a non-coinapi segment is also flagged
    o, e = _day_bounds(day)
    mid = o + rv.DAY_NS // 2
    mixed = _day(day, "lake_present_degraded",
                 _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                             full_day_reason="quality_over_usable_bar",
                             fill_segments=[{"source": "coinapi", "start_ts": o, "start_iso": _seg_iso(o),
                                             "end_ts": mid, "end_iso": _seg_iso(mid), "reason": "r"},
                                            {"source": "lake", "start_ts": mid, "start_iso": _seg_iso(mid),
                                             "end_ts": e, "end_iso": _seg_iso(e), "reason": "r"}],
                             seams=[mid], seam_policy={"seam_guard_s": 60.0}))
    assert "full_day_fill_non_coinapi_segment" in rv.day_record_issues(mixed)
    # a legit single-coinapi full-day plan stays clean
    ok = _day(day, "lake_present_degraded",
              _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                          full_day_reason="quality_over_usable_bar",
                          fill_segments=[_full_day_seg(day)], seams=[],
                          seam_policy={"seam_guard_s": 60.0}))
    assert not any("coinapi_segment" in i for i in rv.day_record_issues(ok))


def test_missing_needs_coinapi_partial_profile_fails_closed():
    # P1: a missing_needs_coinapi day has NO Lake book for the WHOLE day, so a partial profile that
    # preserves a Lake segment references nonexistent data. It must be flagged (require full-day) —
    # the existing full_day_fill_non_coinapi_segment check only covers the full_day profile.
    day = "2025-01-05"
    o, e = _day_bounds(day)
    mid = o + rv.DAY_NS // 2
    segs = [{"source": "coinapi", "start_ts": o, "start_iso": _seg_iso(o), "end_ts": mid,
             "end_iso": _seg_iso(mid), "reason": "gap"},
            {"source": "lake", "start_ts": mid, "start_iso": _seg_iso(mid), "end_ts": e,
             "end_iso": _seg_iso(e), "reason": "trusted"}]
    bad = _day(day, "missing_needs_coinapi",
               _fill_block(True, "lake_book_delta_v2_absent", fill_profile="leading_partial_fill",
                           fill_segments=segs, seams=[mid], seam_policy={"seam_guard_s": 60.0}))
    assert "missing_needs_coinapi_requires_full_day_fill" in rv.day_record_issues(bad)
    # a legit all-CoinAPI full-day replacement for the whole-day gap stays clean
    ok = _day(day, "missing_needs_coinapi",
              _fill_block(True, "lake_book_delta_v2_absent", fill_profile="full_day_fill",
                          full_day_reason="lake_book_delta_v2_absent",
                          fill_segments=[_full_day_seg(day, reason="lake_book_delta_v2_absent")],
                          seams=[], seam_policy={"seam_guard_s": 60.0}))
    assert not any("missing_needs_coinapi_requires_full_day_fill" in i
                   for i in rv.day_record_issues(ok))


def test_day_record_issues_scalar_seams_no_crash():
    # a corrupt non-list `seams` (e.g. a scalar) must fail closed, not crash on list(seams)
    day = "2025-01-02"
    rec = _day(day, "lake_present_degraded",
               _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                           full_day_reason="quality_over_usable_bar",
                           fill_segments=[_full_day_seg(day)], seams=5,
                           seam_policy={"seam_guard_s": 60.0}))
    issues = rv.day_record_issues(rec)   # must not raise
    assert "fill_day_missing_seams" in issues


def test_summary_fill_counts_cross_checked():
    rep = _clean_reports()[0]
    rep["summary"]["coinapi_fill"] = {"fill_counts": {
        "needs_fill": 99, "full_day_fill": 99, "leading_partial_fill": 0,
        "trailing_partial_fill": 0, "internal_gap_fill": 0, "mixed_partial_fill": 0,
        "crossed_source_full_day": 0, "no_verdict": 0, "no_fill": 0, "not_in_scope": 0}}
    assert any("fill_counts_mismatch" in i for i in rv.summary_count_issues(rep))
    rep2 = _clean_reports()[0]
    rep2["summary"]["coinapi_fill"] = {"fill_counts": rv._recompute_fill_counts(rep2["days"])}
    assert not any("fill_counts_mismatch" in i for i in rv.summary_count_issues(rep2))


def test_summary_fill_counts_required():
    rep = _clean_reports()[0]
    rep["summary"]["coinapi_fill"] = {}   # a report missing fill_counts must fail closed, not skip
    assert "summary_fill_counts_missing" in rv.summary_count_issues(rep)


def test_readiness_blocks_missing_fill_counts(tmp_path):
    reports = _clean_reports()
    reports[0]["summary"]["coinapi_fill"] = {}   # drop fill_counts entirely
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "blocking"
    assert any("fill_counts_missing" in x for x in m["blockers"]["inconsistencies"])


def test_day_record_issues_fill_decision_contradicts_classification():
    # stale report: a degraded/missing day mismarked as no-fill would silently DROP a required fill
    for cls in ("lake_present_degraded", "missing_needs_coinapi"):
        rec = _day("d", cls, _fill_block(False, "lake_usable"))
        assert any(i.startswith("fill_decision_contradicts_classification")
                   for i in rv.day_record_issues(rec))
    # a lake_usable day mismarked as needing a full-day fill is also a contradiction
    rec2 = _day("d", "lake_usable",
                _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                            full_day_reason="quality_over_usable_bar"))
    assert any(i.startswith("fill_decision_contradicts_classification")
               for i in rv.day_record_issues(rec2))
    # both legitimate inconclusive outcomes pass (the crossed-source fill needs its marking reason)
    ok_unresolved = _day("d", "inconclusive", _fill_block(None, "no_verdict"))
    ok_crossed = _day("d", "inconclusive",
                      _fill_block(True, "crossed_seed_source_cross_validated_2026-07-01",
                                  fill_profile="full_day_fill", full_day_reason="crossed_seed_source"),
                      reasons=["seed_accepted_but_source_unreliable"])
    for rec in (ok_unresolved, ok_crossed):
        assert not any(i.startswith("fill_decision_contradicts_classification")
                       for i in rv.day_record_issues(rec))
    # a crossed-source fill on an inconclusive day WITHOUT the marking reason is a contradiction
    # (a stale report must not convert a no_verdict blocker into an approved fill)
    bad_crossed = _day("d", "inconclusive",
                       _fill_block(True, "crossed_seed_source_cross_validated_2026-07-01",
                                   fill_profile="full_day_fill", full_day_reason="crossed_seed_source"),
                       reasons=["no_seed_snapshots"])
    assert any(i.startswith("fill_decision_contradicts_classification")
               for i in rv.day_record_issues(bad_crossed))
    # reasons as a STRING (substring-matches the marker) must NOT satisfy the crossed-source contract
    str_reasons = _day("d", "inconclusive",
                       _fill_block(True, "crossed_seed_source_cross_validated_2026-07-01",
                                   fill_profile="full_day_fill", full_day_reason="crossed_seed_source"),
                       reasons="seed_accepted_but_source_unreliable")
    assert any(i.startswith("fill_decision_contradicts_classification")
               for i in rv.day_record_issues(str_reasons))


# =========================================================================== Task 6: counts
def test_recompute_class_counts():
    days = [_day("a", "lake_usable", _fill_block(False, "lake_usable")),
            _day("b", "lake_usable", _fill_block(False, "lake_usable")),
            _day("c", "inconclusive", _fill_block(None, "no_verdict"))]
    counts = rv.recompute_class_counts(days)
    assert counts["lake_usable"] == 2 and counts["inconclusive"] == 1
    assert counts["excluded"] == 0


def test_summary_counts_consistent_ok_and_mismatch():
    rep = _clean_reports()[0]
    assert rv.summary_count_issues(rep) == []
    rep_bad = _clean_reports()[0]
    rep_bad["summary"]["counts"]["lake_usable"] = 99
    assert rv.summary_count_issues(rep_bad) != []


def test_summary_counts_non_numeric_is_mismatch_not_crash():
    # Fix 3: int() over an untrusted non-numeric count previously crashed; now it's a mismatch
    rep = _clean_reports()[0]
    rep["summary"]["counts"]["lake_usable"] = "many"
    rep["summary"]["coinapi_fill"]["fill_counts"]["needs_fill"] = [1]
    issues = rv.summary_count_issues(rep)   # must not raise
    assert any("summary_counts_mismatch:lake_usable" in i for i in issues)
    assert any("fill_counts_mismatch:needs_fill" in i for i in issues)


# =========================================================================== Task 7: day record
def test_build_day_record_quality_map_full_day_fill():
    cal = _calendar()
    rec = rv.build_day_record("2025-01-02", _clean_reports()[0]["days"][1], cal)
    assert rec["classification"] == "lake_present_degraded"
    assert rec["sources"] == ["quality_map"]
    bf = rec["book_fill"]
    assert bf["needed"] is True and bf["kind"] == "full_day" and bf["source"] == "quality_map"
    assert bf["fill_profile"] == "full_day_fill"
    assert bf["gb_basis"] == "estimated" and bf["gb"] == rv.EST_BOOK_GB_PER_DAY
    assert bf["fill_segments"] == [_full_day_seg("2025-01-02")]   # preserved verbatim
    assert rec["trade_fill"]["needed"] is False


def test_build_day_record_calendar_book_gap_measured():
    cal = _calendar()
    rec = rv.build_day_record("2025-01-10", None, cal)
    assert rec["classification"] is None
    assert "calendar_gap" in rec["sources"]
    bf = rec["book_fill"]
    assert bf["needed"] is True and bf["kind"] == "full_day" and bf["source"] == "calendar_gap"
    assert bf["why"] == "calendar_book_gap" and bf["gb"] == 1.0 and bf["gb_basis"] == "measured"
    # a calendar-gap fill must carry an executable full-day stitch plan (not null plan fields)
    assert bf["fill_segments"] and bf["fill_segments"][0]["source"] == "coinapi"
    assert bf["fill_segments"][0]["start_iso"] == "2025-01-10T00:00:00Z"
    assert bf["fill_segments"][0]["end_iso"] == "2025-01-11T00:00:00Z"
    assert bf["fill_segments"][0]["start_ts"] < bf["fill_segments"][0]["end_ts"]
    assert bf["seams"] == []
    assert isinstance(bf["seam_policy"], dict) and bf["seam_policy"]["seam_guard_s"] == 60.0
    tf = rec["trade_fill"]
    assert tf["needed"] is True and tf["gb"] == 0.03 and tf["gb_basis"] == "measured"


def test_build_day_record_book_gap_also_in_report_is_both():
    cal = _calendar()
    rep_day = _day("2025-01-10", "missing_needs_coinapi",
                   _fill_block(True, "lake_book_delta_v2_absent", fill_profile="full_day_fill",
                               full_day_reason="lake_book_delta_v2_absent"))
    rec = rv.build_day_record("2025-01-10", rep_day, cal)
    assert rec["book_fill"]["source"] == "both"
    assert rec["book_fill"]["gb"] == 1.0 and rec["book_fill"]["gb_basis"] == "measured"


def test_build_day_record_trade_only():
    cal = _calendar()
    rep_day = _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
                   calendar=_TRADE_ONLY_CTX)
    rec = rv.build_day_record("2025-01-11", rep_day, cal)
    assert rec["book_fill"]["needed"] is False
    assert rec["trade_fill"]["needed"] is True and rec["trade_fill"]["gb"] == 0.02


def test_excluded_overlap_day_wins_over_fill():
    # a day in BOTH coinbase_fill_days and excluded_days_by_reason: exclusion wins — no fill, no cost
    cal = _calendar(
        coinbase_fill_days={"2025-01-10": {"book": True, "trades": True},
                            "2025-01-15": {"book": True, "trades": True}},
        excluded_days_by_reason={"2025-01-20": ["missing:binF_book"],
                                 "2025-01-15": ["missing:binF_book"]},   # overlaps a fill day
        fill_status={"2025-01-15": {"book": {"present": True, "mb": 999.0, "ok": True},
                                    "trades": {"present": True, "mb": 30.0, "ok": True},
                                    "error": False, "reason": "", "ok": True}})
    # exclusion removes it from the fill-day sets (availability/completeness/cost)
    assert "2025-01-15" not in rv.book_gap_days(cal)
    assert "2025-01-15" not in rv.trade_fill_days(cal)
    # and the per-day record is excluded, not filled
    rec = rv.build_day_record("2025-01-15", None, cal)
    assert rec["book_fill"]["needed"] is False and rec["trade_fill"]["needed"] is False
    assert rec["excluded"] == {"reason": ["missing:binF_book"]}


def test_build_day_record_excluded():
    cal = _calendar()
    rec = rv.build_day_record("2025-01-20", None, cal)
    assert rec["excluded"] == {"reason": ["missing:binF_book"]}
    assert "calendar_excluded" in rec["sources"]
    assert rec["book_fill"]["needed"] is False and rec["trade_fill"]["needed"] is False


def test_build_day_record_unresolved():
    cal = _calendar()
    rep_day = _day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
                   reasons=["no_seed_snapshots"])
    rec = rv.build_day_record("2025-01-01", rep_day, cal)
    assert rec["unresolved"] == {"why": "no_verdict", "classification": "inconclusive",
                                 "reasons": ["no_seed_snapshots"]}
    assert rec["book_fill"]["needed"] is False


# =========================================================================== Task 8: completeness
def test_load_plan_and_reports_indexes_days(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reports, day_index = rv.load_batch_reports(plan)
    assert len(reports) == 1
    assert set(day_index) == {"2025-01-01", "2025-01-02", "2025-01-11"}


def test_completeness_clean(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reports, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports, day_index, cal, blockers)
    assert all(not blockers[k] for k in rv.BLOCKER_KEYS)


def test_completeness_missing_report_blocks(tmp_path):
    # A planned batch with no report yet (staged workflow) is a BLOCKING coverage gap, not a
    # structural exit-2 error: load_batch_reports skips it (no raise), missing_batch_reports records
    # it, and check_completeness emits planned_but_no_report. Its own days-file days are NOT also
    # double-reported as day_not_mapped.
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    os.remove(os.path.join(plan["batches"][0]["report_dir"], "coinbase_quality_map.json"))
    reports, day_index = rv.load_batch_reports(plan)   # no raise
    missing = rv.missing_batch_reports(plan)
    assert missing and not reports
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports, day_index, cal, blockers, missing)
    gaps = blockers["coverage_gaps"]
    assert any("planned_but_no_report" in x for x in gaps)
    assert not any(x.startswith("day_not_mapped:2025-01-01") for x in gaps)


def test_missing_report_is_blocking_not_input_error(tmp_path):
    # end-to-end: a not-yet-run batch still writes a manifest with status=blocking (exit 3, NOT the
    # exit-2 structural path), and --report-only downgrades the exit for inspection.
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    os.remove(os.path.join(plan["batches"][0]["report_dir"], "coinbase_quality_map.json"))
    out = tmp_path / "manifest.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.BLOCKING_EXIT
    m = json.loads(out.read_text())
    assert m["meta"]["status"] == "blocking"
    assert any("planned_but_no_report" in x for x in m["blockers"]["coverage_gaps"])
    rc2 = rv.main(["--plan-manifest", plan_path, "--out", str(out), "--report-only",
                   "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc2 == 0
    assert json.loads(out.read_text())["meta"]["status"] == "blocking"


def test_completeness_day_not_mapped_blocks(tmp_path):
    cal = _calendar(lake_all_days=["2025-01-01", "2025-01-02", "2025-01-03"])
    plan_path, cal_path = _write_tree(tmp_path, cal=cal)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reports, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports, day_index, cal, blockers)
    assert any("2025-01-03" in x for x in blockers["coverage_gaps"])


def test_completeness_mapped_gap_day_must_classify_missing(tmp_path):
    # a book-gap day mapped in a report must be missing_needs_coinapi, not lake_usable/excluded
    reports = [_report([
        _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[_full_day_seg("2025-01-02")], seams=[],
                         seam_policy={"seam_guard_s": 60.0}), fillable=True),
        _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"), calendar=_TRADE_ONLY_CTX),
        _day("2025-01-10", "lake_usable", _fill_block(False, "lake_usable"),   # gap day misclassified
             calendar={"in_usable_days": False, "in_lake_all_days": False,
                       "is_coinbase_fill_day": True, "excluded_reason": None}),
    ])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reps, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reps, day_index, cal, blockers)
    assert any("gap_day_misclassified:2025-01-10" in x for x in blockers["coverage_gaps"])


def test_completeness_stale_gap_report_on_present_day_blocks(tmp_path):
    # a report classifying a now-present (trade-only) day as missing_needs_coinapi is a stale gap
    # report — the current calendar says the Lake book is present, so it must block
    reports = [_report([
        _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[_full_day_seg("2025-01-02")], seams=[],
                         seam_policy={"seam_guard_s": 60.0}), fillable=True),
        _day("2025-01-11", "missing_needs_coinapi",   # trade-only day, but report says book gap
             _fill_block(True, "lake_book_delta_v2_absent", fill_profile="full_day_fill",
                         full_day_reason="lake_book_delta_v2_absent",
                         fill_segments=[_full_day_seg("2025-01-11")], seams=[],
                         seam_policy={"seam_guard_s": 60.0}), calendar=_TRADE_ONLY_CTX, fillable=True),
    ])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reps, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reps, day_index, cal, blockers)
    assert any("missing_needs_coinapi_not_current_gap:2025-01-11" in x
               for x in blockers["coverage_gaps"])


def test_completeness_stale_excluded_report_on_in_scope_day_blocks(tmp_path):
    # a report classifying an in-scope (trade-only) day as excluded, while the current calendar does
    # NOT exclude it, would drop the required verdict/fill and reach ready → must block
    reports = [_report([
        _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[_full_day_seg("2025-01-02")], seams=[],
                         seam_policy={"seam_guard_s": 60.0}), fillable=True),
        _day("2025-01-11", "excluded", _fill_block(None, "excluded_not_in_scope"),
             calendar=_TRADE_ONLY_CTX),
    ])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reps, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reps, day_index, cal, blockers)
    assert any("excluded_not_current_exclusion:2025-01-11" in x for x in blockers["coverage_gaps"])


def test_completeness_batch_incomplete_on_refused_quota(tmp_path):
    reports = _clean_reports()
    reports[0]["meta"]["quota"] = {"ok": False, "reason": "quota_headroom"}
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reps, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reps, day_index, cal, blockers)
    assert blockers["batch_incomplete"]


def test_completeness_batch_incomplete_on_missing_quota_reason(tmp_path):
    reports = _clean_reports()
    reports[0]["meta"]["quota"] = {}   # quota dict present but no completion evidence (no reason)
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reps, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reps, day_index, cal, blockers)
    assert any("quota" in x for x in blockers["batch_incomplete"])


def test_completeness_stale_report_vs_batch_file_blocks(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    # tamper the authoritative batch days-file so it no longer matches the report's day-set
    bf = os.path.join(plan["meta"]["out_dir"], plan["batches"][0]["file"])
    with open(bf, "w") as f:
        f.write("2025-01-01\n2025-01-02\n2025-01-03\n")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reports, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports, day_index, cal, blockers)
    assert any(("batch file" in x or "stale" in x) for x in blockers["batch_incomplete"])


def test_completeness_gap_day_unmapped_blocks(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    plan["skipped"]["fill_days_book_gap"] = []
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reports, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports, day_index, cal, blockers)
    assert any("2025-01-10" in x for x in blockers["coverage_gaps"])


def test_load_batch_reports_rejects_non_list_days(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    rpath = os.path.join(plan["batches"][0]["report_dir"], "coinbase_quality_map.json")
    rep = json.loads(_pl.Path(rpath).read_text())
    rep["days"] = ["2025-01-01", "2025-01-02"]   # list of non-objects → AttributeError previously
    _pl.Path(rpath).write_text(json.dumps(rep))
    with pytest.raises(rv.ReviewInputError, match="days"):
        rv.load_batch_reports(plan)


def test_cli_non_list_days_is_input_error(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    rpath = os.path.join(plan["batches"][0]["report_dir"], "coinbase_quality_map.json")
    rep = json.loads(_pl.Path(rpath).read_text())
    rep["days"] = {}                              # present but wrong type
    _pl.Path(rpath).write_text(json.dumps(rep))
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(tmp_path / "m.json"),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.INPUT_ERROR_EXIT


def test_completeness_missing_out_dir_fails_closed(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    del plan["meta"]["out_dir"]                   # cannot verify against the authoritative days-file
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reports, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports, day_index, cal, blockers)
    assert any("days-file" in x for x in blockers["batch_incomplete"])


def test_load_batch_reports_rejects_non_string_day(tmp_path):
    # Fix 5: a numeric report `day` would make _universe's sorted() crash on a mixed int/str set
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    rpath = os.path.join(plan["batches"][0]["report_dir"], "coinbase_quality_map.json")
    rep = json.loads(_pl.Path(rpath).read_text())
    rep["days"][0]["day"] = 20250101
    _pl.Path(rpath).write_text(json.dumps(rep))
    with pytest.raises(rv.ReviewInputError, match="day must be a string"):
        rv.load_batch_reports(plan)


def test_completeness_duplicate_day_across_batches_blocks(tmp_path):
    # a day in two batches is a coverage_gaps blocking VERDICT (duplicate_across_batches), not a
    # structural exit-2: load_batch_reports keeps the first mapping (no raise) and check_completeness
    # records the overlap so a manifest still assembles and --report-only can inspect it.
    reports = [_report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))]),
               _report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reports_loaded, day_index = rv.load_batch_reports(plan)   # no raise
    assert rv.duplicate_batch_days(reports_loaded) == ["2025-01-01"]
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports_loaded, day_index, cal, blockers)
    assert any("duplicate_across_batches:2025-01-01" in x for x in blockers["coverage_gaps"])


# =========================================================================== Task 9: consistency
def test_check_reports_validation_and_meta_drift(tmp_path):
    reports = [_report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))]),
               _report([_day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable"))],
                       symbol="ETH-USD")]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reps, _ = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_report_consistency(reps, blockers)
    assert any("symbol" in x for x in blockers["inconsistencies"])


def test_wrong_market_report_blocks(tmp_path):
    # reports all for another market (consistent across batches → no drift) must still fail closed
    rep = _report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))],
                  exchange="BINANCE", symbol="ETH-USDT")
    plan_path, cal_path = _write_tree(tmp_path, reports=[rep])
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reps, _ = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_report_consistency(reps, blockers)
    assert any("wrong_market" in x for x in blockers["inconsistencies"])


def test_missing_pinned_meta_blocks(tmp_path):
    # a pinned run-parameter omitted from every report defeats the drift pin → block, don't pass
    rep = _report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))])
    del rep["meta"]["engine"]
    plan_path, cal_path = _write_tree(tmp_path, reports=[rep])
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reps, _ = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_report_consistency(reps, blockers)
    assert any("missing_meta:engine" in x for x in blockers["inconsistencies"])


def test_meta_drift_pins_run_parameters(tmp_path):
    # two batches with different grid_ms (or k/engine/policy) must not combine into a ready manifest
    reports = [_report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))]),
               _report([_day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable"))],
                       grid_ms=500)]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reps, _ = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_report_consistency(reps, blockers)
    assert any("meta_drift:grid_ms" in x for x in blockers["inconsistencies"])


def test_non_dict_coinapi_fill_fails_closed_no_crash(tmp_path):
    # a truthy non-dict coinapi_fill (e.g. a scalar) previously crashed on .get; must fail closed
    reports = _clean_reports()
    reports[0]["days"][1]["coinapi_fill"] = 5
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)   # must not raise
    assert m["meta"]["status"] == "blocking"
    assert any("coinapi_fill" in x for x in m["blockers"]["missing_keys"])


def test_check_unresolved_days_block(tmp_path):
    reports = [_report([_day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
                             reasons=["no_seed_snapshots"]),
                        _day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable"))])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reps, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_report_consistency(reps, blockers)
    assert blockers["unresolved_days"] == ["2025-01-01"]


def test_calendar_drift_blocks():
    cal = _calendar()
    rep_day = _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"),
                   calendar={"in_usable_days": True, "in_lake_all_days": False,
                             "is_coinbase_fill_day": False, "excluded_reason": None})
    blockers = rv.new_blockers()
    rv.check_calendar_drift([{"report": {"days": [rep_day]}}], cal, blockers)
    assert any("2025-01-01" in x for x in blockers["calendar_drift"])


def test_calendar_drift_missing_context_blocks():
    cal = _calendar()
    # a report day with NO calendar context must block (can't verify it's the right calendar)
    rec = _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))
    del rec["calendar"]
    blockers = rv.new_blockers()
    rv.check_calendar_drift([{"report": {"days": [rec]}}], cal, blockers)
    assert any("missing_calendar_context" in x for x in blockers["calendar_drift"])
    # a report day missing one calendar field must block too
    rec2 = _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"),
                calendar={"in_usable_days": True, "is_coinbase_fill_day": False,
                          "excluded_reason": None})   # missing in_lake_all_days
    blockers2 = rv.new_blockers()
    rv.check_calendar_drift([{"report": {"days": [rec2]}}], cal, blockers2)
    assert any("missing_in_lake_all_days" in x for x in blockers2["calendar_drift"])


def test_report_fill_availability_ignores_report_fillable_without_measured_status():
    # The report's coinapi.fillable is NOT an independent block signal: the runner derives it as
    # bool(fill_status.book and book.present), so it is None for never-probed present-degraded fills
    # AND spuriously False for trade-only days (book=null). With no non-null MEASURED book status,
    # none of these block — blocking on absence of evidence would defeat pre-spend approval.
    day_index = {
        "d_false": {"coinapi_fill": {"needs_fill": True}, "coinapi": {"fillable": False}},
        "d_true": {"coinapi_fill": {"needs_fill": True}, "coinapi": {"fillable": True}},
        "d_none": {"coinapi_fill": {"needs_fill": True}, "coinapi": {"fillable": None}},
        "d_nofill": {"coinapi_fill": {"needs_fill": False}, "coinapi": {"fillable": False}},
    }
    blockers = rv.new_blockers()
    rv.check_report_fill_availability(day_index, {}, blockers)
    assert blockers["book_fill_unavailable"] == []


def test_report_fill_available_via_local_parquet(tmp_path):
    # a local CoinAPI parquet re-stat'd on disk makes a fill available even when the calendar's
    # MEASURED book status is present-but-not-ok (the data is already in hand) — flag re-verified
    cal = _calendar(fill_status={"2025-01-02": {"book": {"present": True, "mb": 100.0, "ok": False},
                                                "trades": None, "error": False, "reason": "", "ok": True}})
    pq = tmp_path / "data.parquet"
    pq.write_text("x")
    day_index = {"2025-01-02": {"coinapi_fill": {"needs_fill": True},
                                "coinapi": {"fillable": True, "parquet_local": True,
                                            "parquet_path": str(pq)}}}
    blockers = rv.new_blockers()
    rv.check_report_fill_availability(day_index, cal, blockers)
    assert blockers["book_fill_unavailable"] == []
    # once the parquet is gone the stale flag is not trusted, and the measured not-ok book blocks
    pq.unlink()
    blockers2 = rv.new_blockers()
    rv.check_report_fill_availability(day_index, cal, blockers2)
    assert blockers2["book_fill_unavailable"] == ["2025-01-02:calendar_book_not_ok"]


def test_fill_status_container_non_dict():
    # a top-level fill_status that is a truthy non-object must not crash _fill_status/is_fillable
    cal = _calendar(fill_status=["not", "an", "object"])
    assert rv.is_fillable(cal, "2025-01-10", "book") is False   # no crash
    assert rv.measured_mb(cal, "2025-01-10", "book") is None
    # and validate_calendar fails closed on it (exit 2)
    with pytest.raises(rv.ReviewInputError, match="fill_status"):
        rv.validate_calendar(cal, "cal")


def test_report_book_fill_on_trade_only_day_not_blocked():
    # P2: a trade-only day has fill_status[d].book == null (no book probe), and the runner derives
    # coinapi.fillable = bool(None and ...) == False SPURIOUSLY. If that day is later classified
    # lake_present_degraded and needs a quality-map book fill, the spurious False is NOT positive
    # evidence CoinAPI lacks the book, so it must NOT block — for any report fillable value.
    cal = _calendar(fill_status={"2025-01-11": {"book": None,
                                                "trades": {"present": True, "mb": 20.0, "ok": True},
                                                "error": False, "reason": "", "ok": True}})
    for fillable in (True, False, None):
        day_index = {"2025-01-11": {"coinapi_fill": {"needs_fill": True},
                                    "coinapi": {"fillable": fillable}}}
        blockers = rv.new_blockers()
        rv.check_report_fill_availability(day_index, cal, blockers)
        assert blockers["book_fill_unavailable"] == [], f"fillable={fillable} must not block"


def test_report_book_fill_malformed_book_status_fails_closed():
    # a MALFORMED non-null book status (string/list) is NOT the trade-only None case → fail closed,
    # do not fall back to a stale report's fillable=True
    cal = _calendar(fill_status={"2025-01-11": {"book": "bad",
                                                "trades": {"present": True, "mb": 20.0, "ok": True},
                                                "error": False, "reason": "", "ok": True}})
    day_index = {"2025-01-11": {"coinapi_fill": {"needs_fill": True}, "coinapi": {"fillable": True}}}
    blockers = rv.new_blockers()
    rv.check_report_fill_availability(day_index, cal, blockers)
    assert blockers["book_fill_unavailable"] == ["2025-01-11:calendar_book_not_ok"]


def test_check_report_fill_availability_rechecks_calendar_ok():
    # report `coinapi.fillable` only reflects book.present; if the calendar says present=true but
    # ok=false (unverifiable flat file), the stricter is_fillable cross-check must block.
    cal = _calendar(fill_status={"2025-01-02": {"book": {"present": True, "mb": 100.0, "ok": False},
                                                "trades": None, "error": False, "reason": "", "ok": True}})
    day_index = {"2025-01-02": {"coinapi_fill": {"needs_fill": True},
                                "coinapi": {"fillable": True}}}
    blockers = rv.new_blockers()
    rv.check_report_fill_availability(day_index, cal, blockers)
    assert blockers["book_fill_unavailable"] == ["2025-01-02:calendar_book_not_ok"]


def test_readiness_blocks_report_fill_measured_book_not_ok(tmp_path):
    # a report-driven fill on a day whose calendar MEASURED book status is present-but-not-ok
    # (unverifiable flat file) blocks readiness — the fill must not be priced `ready` from bad MB
    reports = _clean_reports()   # 2025-01-02 is a degraded full_day_fill (needs_fill True)
    cal = _calendar(fill_status={
        "2025-01-02": {"book": {"present": True, "mb": 100.0, "ok": False},
                       "trades": None, "error": False, "reason": "", "ok": True},
        "2025-01-10": {"book": {"present": True, "mb": 1000.0, "ok": True},
                       "trades": {"present": True, "mb": 30.0, "ok": True},
                       "error": False, "reason": "", "ok": True},
        "2025-01-11": {"book": None, "trades": {"present": True, "mb": 20.0, "ok": True},
                       "error": False, "reason": "", "ok": True}})
    plan_path, cal_path = _write_tree(tmp_path, cal=cal, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "blocking"
    assert any("2025-01-02:calendar_book_not_ok" in x
               for x in m["blockers"]["book_fill_unavailable"])


def test_readiness_quality_map_present_fill_without_evidence_is_normal(tmp_path):
    # the pre-spend workflow: a report-driven degraded/crossed-source present fill (coinapi.fillable
    # None, no fill_status entry, no local parquet) is a NORMAL quality-map-added fill priced by the
    # nominal estimate and must NOT block readiness — requiring pre-downloaded proof would be circular.
    reports = _clean_reports()
    reports[0]["days"][1]["coinapi"]["fillable"] = None   # 2025-01-02 degraded full_day_fill
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["blockers"]["book_fill_unavailable"] == []
    assert m["meta"]["status"] == "ready"
    assert "2025-01-02" in m["sections"]["full_day_book_fills"]


def test_fill_availability_blocks_unfillable_book_and_trade():
    cal = _calendar(fill_status={
        "2025-01-10": {"book": {"present": False, "mb": None, "ok": False},
                       "trades": {"present": True, "mb": 30.0, "ok": True},
                       "error": False, "reason": "", "ok": True},
        "2025-01-11": {"book": None,
                       "trades": {"present": False, "mb": None, "ok": False},
                       "error": False, "reason": "", "ok": True}})
    blockers = rv.new_blockers()
    rv.check_fill_availability(cal, blockers)
    assert blockers["book_fill_unavailable"] == ["2025-01-10"]
    assert blockers["trade_fill_unavailable"] == ["2025-01-11"]


# =========================================================================== Task 10: assembly
def test_build_manifest_ready(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "ready"
    assert m["meta"]["scope_complete"] is True
    s = m["sections"]
    assert "2025-01-02" in s["full_day_book_fills"]
    assert "2025-01-10" in s["full_day_book_fills"]
    assert set(s["trade_fills"]) == {"2025-01-10", "2025-01-11"}
    assert s["lake_usable_days"] == ["2025-01-01", "2025-01-11"]
    assert s["lake_present_degraded_days"] == ["2025-01-02"]
    assert s["excluded_days"] == ["2025-01-20"]
    assert s["unresolved_days"] == []
    assert len(m["meta"]["inputs"]["usable_calendar"]["sha256"]) == 64
    br0 = m["meta"]["inputs"]["batch_reports"][0]
    assert len(br0["sha256"]) == 64
    # the batch days-file the stale-report guard reads is sha-pinned for a reproducible audit trail
    assert len(br0["batch_days_file"]["sha256"]) == 64
    assert br0["batch_days_file"]["path"].endswith("batch_001_days.txt")


def test_build_manifest_cost_summary(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    c = m["cost_summary"]
    assert c["book_gb_measured"] == 1.0
    assert c["book_gb_estimated"] == rv.EST_BOOK_GB_PER_DAY
    assert round(c["trades_gb_measured"], 4) == 0.05
    assert c["book_usd"] == round(1.0 + rv.EST_BOOK_GB_PER_DAY, 4)
    assert c["trades_usd"] == round(0.05 * 3.0, 4)
    assert c["calendar_gap_baseline_usd"] == round(1.0 * 1.0 + 0.05 * 3.0, 4)
    assert c["docs_reference_usd"] == 92.0
    assert c["net_usd"] == round(c["gross_usd"] - 25.0, 4)


def test_build_manifest_blocking_unresolved(tmp_path):
    reports = [_report([_day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
                             reasons=["no_seed_snapshots"]),
                        _day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable")),
                        _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
                             calendar=_TRADE_ONLY_CTX)])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "blocking"
    assert m["meta"]["scope_complete"] is False
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]
    assert m["sections"]["unresolved_days"] == ["2025-01-01"]


def test_malformed_report_day_missing_classification_fails_closed(tmp_path):
    # a report day missing `classification` must fail closed (status=blocking, missing_keys),
    # NOT crash build_day_record with a KeyError (Codex P2).
    reports = _clean_reports()
    del reports[0]["days"][1]["classification"]   # 2025-01-02 now missing classification
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "blocking"
    assert any("classification" in x for x in m["blockers"]["missing_keys"])


def test_readiness_blocks_fill_day_without_stitch_plan(tmp_path):
    # a fill day stripped of its stitch plan must not reach a ready manifest with an unexecutable fill
    reports = _clean_reports()
    cf = reports[0]["days"][1]["coinapi_fill"]   # 2025-01-02 degraded full_day_fill
    cf["fill_segments"], cf["seams"], cf["seam_policy"] = None, None, None
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "blocking"
    assert any("fill_day_missing" in x for x in m["blockers"]["inconsistencies"])


def test_readiness_blocks_fill_contradicting_classification(tmp_path):
    # a degraded day corrupted to claim no fill must NOT silently drop the fill into a ready manifest
    reports = _clean_reports()
    reports[0]["days"][1]["coinapi_fill"] = _fill_block(False, "lake_usable")   # 2025-01-02 degraded
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "blocking"
    assert any("contradicts_classification" in x for x in m["blockers"]["inconsistencies"])


def test_build_manifest_inspection_report_only(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    rpath = os.path.join(rv.load_json_object(plan_path, what="p")["batches"][0]["report_dir"],
                         "coinbase_quality_map.json")
    m = rv.build_manifest_inspection([rpath], cal_path, generated_utc="2026-07-03T00:00:00Z")
    assert m["meta"]["status"] == "report_only"
    assert m["meta"]["scope_complete"] is False


# =========================================================================== Task 11: CLI
def test_cli_ready_exit_0_and_writes(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    out = tmp_path / "manifest.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == 0
    m = json.loads(out.read_text())
    assert m["meta"]["status"] == "ready"


def test_cli_blocking_exit_3(tmp_path):
    reports = [_report([_day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict")),
                        _day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable")),
                        _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
                             calendar=_TRADE_ONLY_CTX)])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    out = tmp_path / "manifest.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.BLOCKING_EXIT


def test_cli_report_only_downgrades_to_0(tmp_path):
    reports = [_report([_day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict")),
                        _day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable")),
                        _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
                             calendar=_TRADE_ONLY_CTX)])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    out = tmp_path / "manifest.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out), "--report-only",
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == 0
    assert json.loads(out.read_text())["meta"]["status"] == "blocking"


def test_cli_input_error_exit_2(tmp_path):
    rc = rv.main(["--plan-manifest", str(tmp_path / "nope.json"),
                  "--out", str(tmp_path / "m.json")])
    assert rc == rv.INPUT_ERROR_EXIT


def test_cli_nonfinite_input_fails_closed_without_touching_out(tmp_path):
    # Fix 4: a NaN size must fail closed at load (exit 2) BEFORE write_manifest truncates --out
    cal = _calendar()
    cal["fill_status"]["2025-01-10"]["book"]["mb"] = float("nan")
    plan_path, cal_path = _write_tree(tmp_path, cal=cal)
    out = tmp_path / "existing_manifest.json"
    out.write_text('{"prior": "good manifest"}')   # a prior good artifact that must NOT be destroyed
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.INPUT_ERROR_EXIT
    assert json.loads(out.read_text()) == {"prior": "good manifest"}   # untouched


def test_cli_requires_exactly_one_mode():
    with pytest.raises(SystemExit):
        rv.parse_args([])
    with pytest.raises(SystemExit):
        rv.parse_args(["--plan-manifest", "p", "--report", "r"])


def test_cli_inspection_mode_report_only(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    rpath = os.path.join(rv.load_json_object(plan_path, what="p")["batches"][0]["report_dir"],
                         "coinbase_quality_map.json")
    out = tmp_path / "m.json"
    rc = rv.main(["--report", rpath, "--usable-calendar", cal_path, "--out", str(out),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == 0
    assert json.loads(out.read_text())["meta"]["status"] == "report_only"


def test_cli_inspection_mode_rejects_malformed_days(tmp_path):
    # inspection mode must fail closed (exit 2) on a malformed report, not crash on rec.get(...)
    plan_path, cal_path = _write_tree(tmp_path)
    rpath = os.path.join(rv.load_json_object(plan_path, what="p")["batches"][0]["report_dir"],
                         "coinbase_quality_map.json")
    rep = json.loads(_pl.Path(rpath).read_text())
    rep["days"] = ["2025-01-01"]   # list of non-objects
    _pl.Path(rpath).write_text(json.dumps(rep))
    rc = rv.main(["--report", rpath, "--usable-calendar", cal_path, "--out", str(tmp_path / "m.json"),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.INPUT_ERROR_EXIT


def test_cli_inspection_mode_rejects_non_string_day(tmp_path):
    # P3: a numeric report `day` in inspection mode must fail closed (exit 2) like readiness, not
    # crash _universe's sorted() over a mixed int/str set
    plan_path, cal_path = _write_tree(tmp_path)
    rpath = os.path.join(rv.load_json_object(plan_path, what="p")["batches"][0]["report_dir"],
                         "coinbase_quality_map.json")
    rep = json.loads(_pl.Path(rpath).read_text())
    rep["days"][0]["day"] = 20250101
    _pl.Path(rpath).write_text(json.dumps(rep))
    rc = rv.main(["--report", rpath, "--usable-calendar", cal_path, "--out", str(tmp_path / "m.json"),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.INPUT_ERROR_EXIT


def test_cli_deterministic_bytes(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    rv.main(["--plan-manifest", plan_path, "--out", str(out1),
             "--generated-utc", "2026-07-03T00:00:00Z"])
    rv.main(["--plan-manifest", plan_path, "--out", str(out2),
             "--generated-utc", "2026-07-03T00:00:00Z"])
    assert out1.read_bytes() == out2.read_bytes()


# =========================================================================== Task 12: resolutions
# Targeted per-day resolutions (docs/data.md §5a-QualityMap "Reviewed backfill manifest"):
# a hand-authored decisions file lets a TARGETED rerun report or an EXPLICIT policy decision
# supersede an original unresolved (no_verdict) day — without rerunning a whole batch and without
# hand-editing generated JSON. Fail-closed: only currently-unresolved days may be superseded, every
# entry pins the original (classification, reasons) it resolves, and any invalid entry blocks.
_SR_REASONS = ["seed_rejected:crossed", "crossed_rate_after=0.0845"]


def _reports_with_unresolved(reasons=None):
    """The _clean_reports() tree with 2025-01-01 unresolved (inconclusive / no_verdict)."""
    return [_report([
        _day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
             reasons=list(reasons) if reasons is not None else list(_SR_REASONS)),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[_full_day_seg("2025-01-02")],
                         seams=[], seam_policy={"seam_guard_s": 60.0}),
             fillable=True),
        _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
             calendar=_TRADE_ONLY_CTX),
    ])]


def _entry(day, *, action, expect_classification="inconclusive", expect_reasons=_SR_REASONS,
           report=None, decision=None, reason="targeted follow-up", decided_by="tester",
           decided_utc="2026-07-10T00:00:00Z"):
    e = {"day": day, "action": action,
         "expect": {"classification": expect_classification, "reasons": list(expect_reasons)},
         "reason": reason, "decided_by": decided_by, "decided_utc": decided_utc}
    if report is not None:
        e["report"] = report
    if decision is not None:
        e["decision"] = decision
    return e


def _resolutions_file(tmp_path, entries, *, version=1, name="resolutions.json"):
    path = tmp_path / name
    path.write_text(json.dumps({"resolutions_version": version, "resolutions": entries}))
    return str(path)


def _write_rerun_report(tmp_path, days, *, subdir="rerun_001", **meta_overrides):
    rdir = tmp_path / subdir
    rdir.mkdir(parents=True, exist_ok=True)
    path = rdir / "coinbase_quality_map.json"
    path.write_text(json.dumps(_report(days, **meta_overrides)))
    return str(path)


def _build(plan_path, cal_path, res_path):
    return rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                       report_only=False, resolutions_path=res_path)


def _day_rec(m, day):
    return next(r for r in m["days"] if r["day"] == day)


def test_resolution_rerun_supersedes_unresolved_day(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))])
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    m = _build(plan_path, cal_path, res)
    assert m["blockers"]["resolution_issues"] == []
    assert m["blockers"]["unresolved_days"] == []
    assert m["meta"]["status"] == "ready" and m["meta"]["scope_complete"] is True
    rec = _day_rec(m, "2025-01-01")
    assert rec["classification"] == "lake_usable"   # the rerun record supersedes the original
    assert rec["unresolved"] is None
    assert rec["resolution"]["action"] == "rerun"
    assert rec["resolution"]["applied"] is True and rec["resolution"]["resolved"] is True
    assert rec["resolution"]["report"]["path"] == rerun
    assert len(rec["resolution"]["report"]["sha256"]) == 64
    assert "2025-01-01" in m["sections"]["lake_usable_days"]
    assert "2025-01-01" not in m["sections"]["unresolved_days"]
    # provenance: the resolutions file and every rerun report are pinned in meta.inputs
    ri = m["meta"]["inputs"]["resolutions"]
    assert ri["path"] == res and len(ri["sha256"]) == 64 and ri["n_entries"] == 1
    assert ri["rerun_reports"] == [{"path": rerun, "sha256": rv.sha256_file(rerun)}]


def test_resolution_policy_fill_supersedes_unresolved_day(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="policy",
                                              decision="coinapi_fill_full_day")])
    m = _build(plan_path, cal_path, res)
    assert m["blockers"]["resolution_issues"] == []
    assert m["blockers"]["unresolved_days"] == []
    assert m["meta"]["status"] == "ready"
    rec = _day_rec(m, "2025-01-01")
    assert rec["classification"] == "inconclusive"   # the Lake verdict is preserved as-is
    assert rec["unresolved"] is None
    bf = rec["book_fill"]
    assert bf["needed"] is True and bf["kind"] == "full_day" and bf["source"] == "policy"
    assert bf["fill_profile"] == "full_day_fill"
    o, e = _day_bounds("2025-01-01")
    assert [s["source"] for s in bf["fill_segments"]] == ["coinapi"]
    assert bf["fill_segments"][0]["start_ts"] == o and bf["fill_segments"][-1]["end_ts"] == e
    assert bf["gb"] == rv.EST_BOOK_GB_PER_DAY and bf["gb_basis"] == "estimated"
    assert "2025-01-01" in m["sections"]["full_day_book_fills"]
    # 2025-01-02 (report) + 2025-01-10 (calendar gap) + the policy fill
    assert m["cost_summary"]["full_book_fill_days"] == 3
    assert rec["resolution"]["decision"] == "coinapi_fill_full_day"
    assert rec["resolution"]["applied"] is True and rec["resolution"]["resolved"] is True


def test_resolution_policy_exclude_supersedes_unresolved_day(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="policy", decision="exclude",
                                              reason="unrecoverable: no seed source")])
    m = _build(plan_path, cal_path, res)
    assert m["blockers"]["resolution_issues"] == []
    assert m["blockers"]["unresolved_days"] == []
    assert m["meta"]["status"] == "ready"
    rec = _day_rec(m, "2025-01-01")
    assert rec["unresolved"] is None
    assert rec["excluded"] == {"reason": "policy_decision:unrecoverable: no seed source"}
    assert rec["book_fill"]["needed"] is False and rec["trade_fill"]["needed"] is False
    assert "2025-01-01" in m["sections"]["excluded_days"]


def test_resolution_stale_expect_blocks(tmp_path):
    # the entry pins the original (classification, reasons); a regenerated batch report that no
    # longer matches makes the decision STALE — it must not apply. Both pins are checked.
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="policy",
                                              decision="coinapi_fill_full_day",
                                              expect_reasons=["no_seed_snapshots"])])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any(x.startswith("stale_expect:2025-01-01:reasons:")
               for x in m["blockers"]["resolution_issues"])
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]   # NOT resolved
    rec = _day_rec(m, "2025-01-01")
    assert rec["unresolved"] is not None and rec["book_fill"]["needed"] is False
    assert rec["resolution"]["applied"] is False and rec["resolution"]["issues"]
    # the classification pin is enforced independently of the reasons pin
    res2 = _resolutions_file(tmp_path, [_entry("2025-01-01", action="policy",
                                               decision="coinapi_fill_full_day",
                                               expect_classification="lake_usable")],
                             name="resolutions2.json")
    m2 = _build(plan_path, cal_path, res2)
    assert m2["meta"]["status"] == "blocking"
    assert any(x.startswith("stale_expect:2025-01-01:classification:")
               for x in m2["blockers"]["resolution_issues"])
    assert m2["blockers"]["unresolved_days"] == ["2025-01-01"]


def test_resolution_conflicting_duplicate_entries_block(tmp_path):
    # two decisions for the same day conflict — NEITHER applies (fail-closed)
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    res = _resolutions_file(tmp_path, [
        _entry("2025-01-01", action="policy", decision="coinapi_fill_full_day"),
        _entry("2025-01-01", action="policy", decision="exclude")])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("duplicate_entry:2025-01-01" in x for x in m["blockers"]["resolution_issues"])
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]
    rec = _day_rec(m, "2025-01-01")
    assert rec["unresolved"] is not None
    assert rec["book_fill"]["needed"] is False and rec["excluded"] is None
    assert rec["resolution"]["applied"] is False


def test_resolution_target_with_verdict_blocks(tmp_path):
    # 2025-01-02 already carries a deterministic fill verdict — an override must not touch it
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    res = _resolutions_file(tmp_path, [_entry("2025-01-02", action="policy", decision="exclude",
                                              expect_classification="lake_present_degraded",
                                              expect_reasons=[])])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("target_not_unresolved:2025-01-02" in x
               for x in m["blockers"]["resolution_issues"])
    rec = _day_rec(m, "2025-01-02")
    assert rec["book_fill"]["needed"] is True and rec["excluded"] is None   # verdict preserved
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]   # real unresolved day untouched


def test_resolution_unknown_day_blocks(tmp_path):
    # a resolution for a day no batch report maps cannot invent coverage (plan-driven scope)
    plan_path, cal_path = _write_tree(tmp_path)   # clean tree: no unresolved days at all
    res = _resolutions_file(tmp_path, [_entry("2025-03-03", action="policy", decision="exclude")])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("unknown_day:2025-03-03" in x for x in m["blockers"]["resolution_issues"])
    # the entry's day is outside the manifest universe (no invented day record), so its audit
    # trail must survive under meta.inputs.resolutions instead of vanishing
    assert all(rec["day"] != "2025-03-03" for rec in m["days"])
    unattached = m["meta"]["inputs"]["resolutions"]["unattached"]
    assert unattached["2025-03-03"]["applied"] is False
    assert unattached["2025-03-03"]["issues"] == ["unknown_day:2025-03-03"]


def test_resolution_policy_exclude_wins_over_calendar_trade_fill(tmp_path):
    # exclude must behave like a calendar exclusion ("exclusion wins over fill routing"): the
    # excluded day leaves the fill-availability day-set AND the calendar-gap cost baseline —
    # otherwise excluding an irrecoverable trade-gap day whose probe is not-ok can NEVER unblock
    # the gate, and a ready manifest ships a baseline that still prices the excluded day.
    reports = [_report([
        _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[_full_day_seg("2025-01-02")],
                         seams=[], seam_policy={"seam_guard_s": 60.0}),
             fillable=True),
        _day("2025-01-11", "inconclusive", _fill_block(None, "no_verdict"),
             calendar=_TRADE_ONLY_CTX, reasons=list(_SR_REASONS)),
    ])]
    cal = _calendar()
    cal["fill_status"]["2025-01-11"]["trades"]["ok"] = False   # probe not-ok: unfillable trades
    plan_path, cal_path = _write_tree(tmp_path, cal=cal, reports=reports)
    # without a resolution: the unresolved day AND its unfillable trade fill both block
    m0 = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                     report_only=False)
    assert m0["blockers"]["trade_fill_unavailable"] == ["2025-01-11"]
    res = _resolutions_file(tmp_path, [_entry("2025-01-11", action="policy", decision="exclude",
                                              reason="irrecoverable")])
    m = _build(plan_path, cal_path, res)
    assert m["blockers"]["resolution_issues"] == []
    assert m["blockers"]["trade_fill_unavailable"] == []   # exclusion wins
    assert m["blockers"]["unresolved_days"] == []
    assert m["meta"]["status"] == "ready"
    assert "2025-01-11" not in m["sections"]["trade_fills"]
    # baseline covers 2025-01-10 only: 1.0 GB book ($1.00) + 0.03 GB trades ($0.09) — the
    # excluded day's $0.06 trade cost must NOT remain in the baseline
    assert m["cost_summary"]["calendar_gap_baseline_usd"] == 1.09


def test_resolution_rerun_report_cannot_be_a_batch_report(tmp_path):
    # ORIGINAL batch evidence must not be re-presented as a "targeted rerun": that would stamp
    # applied-rerun provenance without any new data having been produced
    reports = [_report([_day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
                             reasons=list(_SR_REASONS))]),
               _report([
                   _day("2025-01-02", "lake_present_degraded",
                        _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                                    full_day_reason="quality_over_usable_bar",
                                    fill_segments=[_full_day_seg("2025-01-02")],
                                    seams=[], seam_policy={"seam_guard_s": 60.0}),
                        fillable=True),
                   _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
                        calendar=_TRADE_ONLY_CTX)])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    batch1 = rv.load_json_object(plan_path, what="p")["batches"][0]["report_dir"]
    res = _resolutions_file(tmp_path, [_entry(
        "2025-01-01", action="rerun",
        report=os.path.join(batch1, "coinbase_quality_map.json"))])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("rerun_is_batch_report" in x for x in m["blockers"]["resolution_issues"])
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]   # not applied
    assert _day_rec(m, "2025-01-01")["resolution"]["applied"] is False


def test_resolution_policy_fill_blocks_when_calendar_book_not_ok(tmp_path):
    # a policy fill faces the same positive-unavailability evidence bar as report-driven fills:
    # a MEASURED present-but-not-ok calendar book status must block the spend
    cal = _calendar()
    cal["fill_status"]["2025-01-01"] = {"book": {"present": True, "mb": 900.0, "ok": False},
                                        "trades": None, "error": False, "reason": "", "ok": True}
    plan_path, cal_path = _write_tree(tmp_path, cal=cal, reports=_reports_with_unresolved())
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="policy",
                                              decision="coinapi_fill_full_day")])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert m["blockers"]["book_fill_unavailable"] == ["2025-01-01:calendar_book_not_ok"]
    assert m["blockers"]["resolution_issues"] == []   # the decision itself is valid
    assert m["blockers"]["unresolved_days"] == []     # ...and applied
    assert _day_rec(m, "2025-01-01")["resolution"]["applied"] is True


def test_resolution_policy_fill_blocks_on_malformed_fill_status(tmp_path):
    # deep-review P2: a MALFORMED per-day fill_status record (non-dict) is CORRUPT availability
    # evidence, not absent evidence — a policy fill must fail closed on it via is_fillable like
    # every other fill-availability path, not clear the blocker and approve spend
    cal = _calendar()
    cal["fill_status"]["2025-01-01"] = "garbage"
    plan_path, cal_path = _write_tree(tmp_path, cal=cal, reports=_reports_with_unresolved())
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="policy",
                                              decision="coinapi_fill_full_day")])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert m["blockers"]["book_fill_unavailable"] == ["2025-01-01:calendar_book_not_ok"]
    assert m["blockers"]["resolution_issues"] == []   # the decision itself is valid + applied
    assert m["blockers"]["unresolved_days"] == []


def test_resolution_policy_fill_blocks_on_probe_error_evidence(tmp_path):
    # deep-review P2: a day-level probe ERROR record ({'book': None, 'error': true}) or
    # membership in fill_days_unfillable / fill_days_probe_error is positive unavailability
    # evidence — the policy fill must route it through is_fillable, not read it as "never probed"
    def _variant(cal, i):
        if i == 0:
            cal["fill_status"]["2025-01-01"] = {"book": None, "trades": None, "error": True,
                                                "reason": "probe failed", "ok": False}
        elif i == 1:
            cal["fill_days_unfillable"] = ["2025-01-01"]
        else:
            cal["fill_days_probe_error"] = ["2025-01-01"]
        return cal
    for i in range(3):
        d = tmp_path / f"v{i}"
        d.mkdir()
        plan_path, cal_path = _write_tree(d, cal=_variant(_calendar(), i),
                                          reports=_reports_with_unresolved())
        res = _resolutions_file(d, [_entry("2025-01-01", action="policy",
                                           decision="coinapi_fill_full_day")])
        m = _build(plan_path, cal_path, res)
        assert m["meta"]["status"] == "blocking", f"variant {i}"
        assert m["blockers"]["book_fill_unavailable"] == ["2025-01-01:calendar_book_not_ok"], \
            f"variant {i}"
        assert m["blockers"]["resolution_issues"] == [] and \
            m["blockers"]["unresolved_days"] == [], f"variant {i}"


def test_resolution_rerun_fill_judged_by_availability_checks(tmp_path):
    # pins the apply-BEFORE-checks ordering: check_report_fill_availability must judge the
    # SUPERSEDING rerun record — a rerun-introduced fill on a day whose calendar book status is
    # measured present-but-not-ok blocks, which can only happen if the rerun record replaced the
    # original in day_index before the availability checks ran
    cal = _calendar()
    cal["fill_status"]["2025-01-01"] = {"book": {"present": True, "mb": 900.0, "ok": False},
                                        "trades": None, "error": False, "reason": "", "ok": True}
    plan_path, cal_path = _write_tree(tmp_path, cal=cal, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_present_degraded",
                        _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                                    full_day_reason="quality_over_usable_bar",
                                    fill_segments=[_full_day_seg("2025-01-01")],
                                    seams=[], seam_policy={"seam_guard_s": 60.0}))])
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    m = _build(plan_path, cal_path, res)
    assert m["blockers"]["resolution_issues"] == []
    assert m["blockers"]["unresolved_days"] == []
    assert m["meta"]["status"] == "blocking"
    assert m["blockers"]["book_fill_unavailable"] == ["2025-01-01:calendar_book_not_ok"]
    rec = _day_rec(m, "2025-01-01")
    assert rec["resolution"]["applied"] is True and rec["resolution"]["resolved"] is True


def test_resolution_one_rerun_report_resolves_two_days(tmp_path):
    # the intended production shape: one targeted rerun report covering several unresolved days,
    # each claimed by its own entry
    reports = [_report([
        _day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
             reasons=list(_SR_REASONS)),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[_full_day_seg("2025-01-02")],
                         seams=[], seam_policy={"seam_guard_s": 60.0}),
             fillable=True),
        _day("2025-01-11", "inconclusive", _fill_block(None, "no_verdict"),
             calendar=_TRADE_ONLY_CTX, reasons=["no_seed_snapshots"]),
    ])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
                   _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
                        calendar=_TRADE_ONLY_CTX)])
    res = _resolutions_file(tmp_path, [
        _entry("2025-01-01", action="rerun", report=rerun),
        _entry("2025-01-11", action="rerun", report=rerun,
               expect_reasons=["no_seed_snapshots"])])
    m = _build(plan_path, cal_path, res)
    assert m["blockers"]["resolution_issues"] == []
    assert m["blockers"]["unresolved_days"] == []
    assert m["meta"]["status"] == "ready"
    for d in ("2025-01-01", "2025-01-11"):
        assert _day_rec(m, d)["resolution"]["applied"] is True
    assert len(m["meta"]["inputs"]["resolutions"]["rerun_reports"]) == 1


def test_resolution_rerun_report_path_spellings_are_canonicalized(tmp_path):
    # Codex P2: equivalent spellings of ONE rerun report (e.g. one normalized, one with
    # redundant `..` components — or relative vs absolute) must group claims under a single
    # canonical file; split claims would see each other's days as unclaimed and wrongly block
    # a fully valid multi-day rerun
    reports = [_report([
        _day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
             reasons=list(_SR_REASONS)),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[_full_day_seg("2025-01-02")],
                         seams=[], seam_policy={"seam_guard_s": 60.0}),
             fillable=True),
        _day("2025-01-11", "inconclusive", _fill_block(None, "no_verdict"),
             calendar=_TRADE_ONLY_CTX, reasons=["no_seed_snapshots"]),
    ])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
                   _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"),
                        calendar=_TRADE_ONLY_CTX)])
    alias = str(tmp_path / "rerun_001" / ".." / "rerun_001" / "coinbase_quality_map.json")
    assert alias != rerun and os.path.realpath(alias) == os.path.realpath(rerun)
    res = _resolutions_file(tmp_path, [
        _entry("2025-01-01", action="rerun", report=rerun),
        _entry("2025-01-11", action="rerun", report=alias,
               expect_reasons=["no_seed_snapshots"])])
    m = _build(plan_path, cal_path, res)
    assert m["blockers"]["resolution_issues"] == []
    assert m["blockers"]["unresolved_days"] == []
    assert m["meta"]["status"] == "ready"
    for d in ("2025-01-01", "2025-01-11"):
        assert _day_rec(m, d)["resolution"]["applied"] is True


def test_cli_report_only_with_resolutions_keeps_honest_status(tmp_path, capsys):
    # --report-only still downgrades the exit while resolution_issues keep the status honest;
    # the terminal summary reports the entry/applied/resolved accounting
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))])
    res = _resolutions_file(tmp_path, [
        _entry("2025-01-01", action="rerun", report=rerun),
        _entry("2025-01-02", action="policy", decision="exclude",
               expect_classification="lake_present_degraded", expect_reasons=[])])
    out = tmp_path / "m.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out), "--resolutions", res,
                  "--report-only", "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == 0
    m = json.loads(out.read_text())
    assert m["meta"]["status"] == "blocking"   # target_not_unresolved:2025-01-02
    assert "resolutions: 2 entries, 1 applied, 1 resolved" in capsys.readouterr().out


def test_resolution_rerun_still_no_verdict_is_preserved(tmp_path):
    # a truthful rerun that STILL reaches no verdict is not an error — but the day must remain an
    # unresolved blocker (never silently cleared), now carrying the rerun's evidence
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
                        reasons=["seed_rejected:crossed", "crossed_rate_after=0.5000"])])
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    m = _build(plan_path, cal_path, res)
    assert m["blockers"]["resolution_issues"] == []
    assert m["meta"]["status"] == "blocking"
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]
    rec = _day_rec(m, "2025-01-01")
    assert rec["resolution"]["applied"] is True and rec["resolution"]["resolved"] is False
    assert rec["unresolved"]["reasons"] == ["seed_rejected:crossed", "crossed_rate_after=0.5000"]


def test_resolution_rerun_meta_drift_blocks(tmp_path):
    # a rerun under different pinned run parameters (here: thresholds) is not comparable evidence
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))],
        thresholds={"crossed_usable_max": 0.5, "missing_usable_max": 0.02,
                    "thin_usable_max": 0.1, "seed_crossed_frac_max": 0.05})
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("meta_drift:thresholds" in x for x in m["blockers"]["resolution_issues"])
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]   # not applied
    assert _day_rec(m, "2025-01-01")["classification"] == "inconclusive"


def test_resolution_rerun_unclaimed_day_blocks(tmp_path):
    # every day a rerun report covers must be explicitly claimed — no silent cherry-picking
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
                   _day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable"))])
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("unclaimed_day:2025-01-02" in x for x in m["blockers"]["resolution_issues"])
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]   # not applied


def test_resolution_rerun_duplicate_day_in_report_blocks(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    dup = _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))
    rerun = _write_rerun_report(tmp_path, [dup, dict(dup)])
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("rerun_day_duplicated:2025-01-01" in x
               for x in m["blockers"]["resolution_issues"])
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]   # not applied


def test_resolution_rerun_report_missing_blocks(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun",
                                              report=str(tmp_path / "nope.json"))])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("rerun_report_missing" in x for x in m["blockers"]["resolution_issues"])
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]


def test_resolution_rerun_corrupt_record_blocks(tmp_path):
    # the superseding record passes the SAME per-day validation as batch report days
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_present_degraded",
                        _fill_block(True, "quality_over_usable_bar"))])   # fill without a plan
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    m = _build(plan_path, cal_path, res)
    assert m["meta"]["status"] == "blocking"
    assert any("needs_fill_without_plan" in x for x in m["blockers"]["resolution_issues"])
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]   # not applied
    assert _day_rec(m, "2025-01-01")["classification"] == "inconclusive"


def test_resolutions_file_structural_errors_exit_2(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    bad_files = [
        {"resolutions_version": 99, "resolutions": []},                        # unknown version
        {"resolutions_version": 1, "resolutions": {}},                         # not a list
        {"resolutions_version": 1,
         "resolutions": [_entry("2025-01-01", action="frobnicate")]},          # unknown action
        {"resolutions_version": 1,
         "resolutions": [_entry("2025-01-01", action="policy", decision="accept_as_usable")]},
        {"resolutions_version": 1,
         "resolutions": [_entry("2025-01-01", action="rerun")]},               # rerun w/o report
        {"resolutions_version": 1,
         "resolutions": [{k: v for k, v in
                          _entry("2025-01-01", action="policy", decision="exclude").items()
                          if k != "expect"}]},                                 # missing expect pin
    ]
    for i, obj in enumerate(bad_files):
        p = tmp_path / f"res_{i}.json"
        p.write_text(json.dumps(obj))
        with pytest.raises(rv.ReviewInputError):
            _build(plan_path, cal_path, str(p))
    # via the CLI: exit 2 and a pre-existing --out is never truncated
    out = tmp_path / "m.json"
    out.write_text('{"prior": "good manifest"}')
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out),
                  "--resolutions", str(tmp_path / "res_0.json"),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.INPUT_ERROR_EXIT
    assert json.loads(out.read_text()) == {"prior": "good manifest"}


def test_manifest_without_resolutions_has_null_provenance(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["inputs"]["resolutions"] is None
    assert all(rec["resolution"] is None for rec in m["days"])


def test_cli_resolutions_end_to_end_ready(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))])
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    out = tmp_path / "m.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out), "--resolutions", res,
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == 0
    m = json.loads(out.read_text())
    assert m["meta"]["status"] == "ready"
    assert m["meta"]["inputs"]["resolutions"]["path"] == res


def test_cli_resolutions_rejected_in_inspection_mode(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    rpath = os.path.join(rv.load_json_object(plan_path, what="p")["batches"][0]["report_dir"],
                         "coinbase_quality_map.json")
    res = _resolutions_file(tmp_path, [])
    with pytest.raises(SystemExit):
        rv.main(["--report", rpath, "--usable-calendar", cal_path,
                 "--out", str(tmp_path / "m.json"), "--resolutions", res])


def test_cli_deterministic_bytes_with_resolutions(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path, reports=_reports_with_unresolved())
    rerun = _write_rerun_report(
        tmp_path, [_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))])
    res = _resolutions_file(tmp_path, [_entry("2025-01-01", action="rerun", report=rerun)])
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    for out in (out1, out2):
        rv.main(["--plan-manifest", plan_path, "--out", str(out), "--resolutions", res,
                 "--generated-utc", "2026-07-03T00:00:00Z"])
    assert out1.read_bytes() == out2.read_bytes()


def test_blocker_summary_truncation_is_explicit(tmp_path, capsys):
    # >8 blockers of one kind: the terminal summary must COUNT the omitted entries, not silently
    # print only the first eight (the 2026-07-09 production run hid the 9th unresolved day)
    days10 = [f"2025-02-{i:02d}" for i in range(1, 11)]
    cal = _calendar(lake_all_days=list(days10), usable_days=list(days10), coinbase_fill_days={},
                    excluded_days_by_reason={}, fill_status={})
    reports = [_report([_day(d, "inconclusive", _fill_block(None, "no_verdict"),
                             reasons=["seed_rejected:crossed"]) for d in days10])]
    plan_path, cal_path = _write_tree(tmp_path, cal=cal, reports=reports)
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(tmp_path / "m.json"),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.BLOCKING_EXIT
    out = capsys.readouterr().out
    lines = [ln for ln in out.splitlines() if ln.strip().startswith("- unresolved_days:")]
    assert len(lines) == 9   # 8 entries + 1 explicit remainder line
    assert "2025-02-08" in out
    assert "+2 more (10 total)" in out
