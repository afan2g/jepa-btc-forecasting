"""Offline tests for the Coinbase backfill review-manifest tool
(`scripts/review_coinbase_backfill_manifest.py`, docs/data.md §5a-QualityMap;
design docs/superpowers/specs/2026-07-03-coinbase-backfill-manifest-review-design.md).

Drives the tool end-to-end on SYNTHETIC plan/report/calendar JSON — no vendor I/O anywhere
(the tool is stdlib-only and never opens a Lake/CoinAPI session). Covers the enum alignment
contract, calendar/cost helpers, validation, the per-day record builder, plan-driven
completeness, consistency/drift/fill-availability checks, manifest assembly, and the CLI
(readiness/inspection modes, fail-closed exit codes, deterministic output)."""
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


def _day(day, classification, coinapi_fill, *, trusted=(None, None),
         calendar=None, reasons=None) -> dict:
    return {
        "day": day, "classification": classification, "reasons": reasons or [],
        "lake_book_delta_v2_present": classification != "missing_needs_coinapi",
        "quality": {"grid_ms": 1000, "trusted_lake_start_ts": trusted[0],
                    "trusted_lake_end_ts": trusted[1], "n_invalid_runs": 0, "invalid_runs": []},
        "coinapi": {"parquet_local": False, "parquet_path": None, "fillable": None},
        "calendar": calendar or {"in_usable_days": True, "in_lake_all_days": True,
                                 "is_coinbase_fill_day": False, "excluded_reason": None},
        "coinapi_fill": coinapi_fill,
    }


def _report(days, **meta_overrides) -> dict:
    meta = {"k": 10, "grid_ms": 1000, "exchange": "COINBASE", "symbol": "BTC-USD",
            "engine": "native", "thresholds": {"crossed_usable_max": 0.01,
            "missing_usable_max": 0.02, "thin_usable_max": 0.1, "seed_crossed_frac_max": 0.05},
            "quota": {"ok": True, "reason": "ok", "used_gb_before": 0.26, "used_gb_after": 0.26},
            "generated_utc": "2026-07-01T00:00:00Z"}
    meta.update(meta_overrides)
    counts = {c: 0 for c in rv.CLASSES}
    for d in days:
        counts[d["classification"]] = counts.get(d["classification"], 0) + 1
    return {"meta": meta, "summary": {"n_days": len(days), "counts": counts,
                                      "by_class": {}, "coinapi_fill": {}}, "days": days}


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
                         fill_segments=[{"source": "coinapi", "start_ts": 1, "start_iso": "x",
                                         "end_ts": 2, "end_iso": "y", "reason": "r"}],
                         seams=[], seam_policy={"seam_guard_s": 60.0})),
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
                           full_day_reason="quality_over_usable_bar"))
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
    assert "partial_with_full_day_reason" in rv.day_record_issues(r4)
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
    assert bf["fill_segments"] == [{"source": "coinapi", "start_ts": 1, "start_iso": "x",
                                    "end_ts": 2, "end_iso": "y", "reason": "r"}]
    assert rec["trade_fill"]["needed"] is False


def test_build_day_record_calendar_book_gap_measured():
    cal = _calendar()
    rec = rv.build_day_record("2025-01-10", None, cal)
    assert rec["classification"] is None
    assert "calendar_gap" in rec["sources"]
    bf = rec["book_fill"]
    assert bf["needed"] is True and bf["kind"] == "full_day" and bf["source"] == "calendar_gap"
    assert bf["why"] == "calendar_book_gap" and bf["gb"] == 1.0 and bf["gb_basis"] == "measured"
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
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    os.remove(os.path.join(plan["batches"][0]["report_dir"], "coinbase_quality_map.json"))
    with pytest.raises(rv.ReviewInputError, match="report"):
        rv.load_batch_reports(plan)


def test_completeness_day_not_mapped_blocks(tmp_path):
    cal = _calendar(lake_all_days=["2025-01-01", "2025-01-02", "2025-01-03"])
    plan_path, cal_path = _write_tree(tmp_path, cal=cal)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reports, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports, day_index, cal, blockers)
    assert any("2025-01-03" in x for x in blockers["coverage_gaps"])


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


def test_completeness_gap_day_unmapped_blocks(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    plan["skipped"]["fill_days_book_gap"] = []
    cal = rv.load_json_object(cal_path, what="usable calendar")
    reports, day_index = rv.load_batch_reports(plan)
    blockers = rv.new_blockers()
    rv.check_completeness(plan, reports, day_index, cal, blockers)
    assert any("2025-01-10" in x for x in blockers["coverage_gaps"])


def test_completeness_duplicate_day_across_batches_blocks(tmp_path):
    reports = [_report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))]),
               _report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    with pytest.raises(rv.ReviewInputError, match="duplicate"):
        rv.load_batch_reports(plan)


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
    assert len(m["meta"]["inputs"]["batch_reports"][0]["sha256"]) == 64


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


def test_cli_deterministic_bytes(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    out1, out2 = tmp_path / "a.json", tmp_path / "b.json"
    rv.main(["--plan-manifest", plan_path, "--out", str(out1),
             "--generated-utc", "2026-07-03T00:00:00Z"])
    rv.main(["--plan-manifest", plan_path, "--out", str(out2),
             "--generated-utc", "2026-07-03T00:00:00Z"])
    assert out1.read_bytes() == out2.read_bytes()
