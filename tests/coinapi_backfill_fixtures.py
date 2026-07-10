"""Shared synthetic fixtures for the CoinAPI reviewed-manifest backfill executor tests.

Builds tiny plan/report/calendar trees (mirroring tests/test_review_backfill_manifest.py) and
runs the REAL reviewer (scripts/review_coinbase_backfill_manifest.py) over them, so the executor
tests consume a genuine `status=ready` reviewed manifest — the executor's acceptance gate is then
tested against the actual upstream contract, not a hand-approximation of it. No vendor I/O.

The clean tree yields a sparse fill scope with every unit shape the executor must handle:
  2025-01-01  lake_usable                          -> no unit
  2025-01-02  degraded, full-day book fill          -> book unit (estimated GB)
  2025-01-03  degraded, LEADING PARTIAL book fill   -> book unit (whole-day file; stitch verbatim)
  2025-01-10  calendar book gap + trades gap        -> book unit (measured GB) + trades unit
  2025-01-11  trade-only gap (book present)         -> trades unit
  2025-01-20  calendar-excluded                     -> no unit
Days 2025-01-04..2025-01-09 lie between fill days and must never become units.
"""
import datetime as _dt
import importlib.util as _ilu
import json
import pathlib as _pl
import sys

_ROOT = _pl.Path(__file__).resolve().parents[1]


def load_by_path(name, rel):
    spec = _ilu.spec_from_file_location(name, _ROOT / rel)
    mod = _ilu.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rv = load_by_path("review_coinbase_backfill_manifest", "scripts/review_coinbase_backfill_manifest.py")

NS = 1_000_000_000
GENERATED_UTC = "2026-07-10T00:00:00Z"


def day_bounds(day):
    d = _dt.date.fromisoformat(day)
    o = int(_dt.datetime(d.year, d.month, d.day, tzinfo=_dt.timezone.utc).timestamp()) * NS
    return o, o + 86_400 * NS


def _seg_iso(ns):
    secs, rem = divmod(ns, NS)
    b = _dt.datetime.fromtimestamp(secs, tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{b}.{rem:09d}Z" if rem else f"{b}Z"


def _seg(source, start, end, reason):
    return {"source": source, "start_ts": start, "start_iso": _seg_iso(start),
            "end_ts": end, "end_iso": _seg_iso(end), "reason": reason}


def full_day_seg(day, reason="quality_over_usable_bar"):
    o, e = day_bounds(day)
    return _seg("coinapi", o, e, reason)


def partial_plan(day):
    """Leading partial fill: coinapi [open, open+6h) then lake [open+6h, close); one seam."""
    o, e = day_bounds(day)
    seam = o + 6 * 3600 * NS
    segs = [_seg("coinapi", o, seam, "leading_gap"), _seg("lake", seam, e, "trusted_lake")]
    return segs, [seam], dict(rv.DEFAULT_SEAM_POLICY), seam


def fill_block(needs_fill, why, fill_profile=None, full_day_reason=None,
               fill_segments=None, seams=None, seam_policy=None):
    return {"needs_fill": needs_fill, "why": why, "fill_profile": fill_profile,
            "full_day_reason": full_day_reason, "fill_segments": fill_segments,
            "seams": seams, "seam_policy": seam_policy}


def day_rec(day, classification, coinapi_fill, *, trusted=(None, None), calendar=None,
            reasons=None, fillable=None):
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


def make_report(days, **meta_overrides):
    meta = {"k": 10, "grid_ms": 1000, "exchange": "COINBASE", "symbol": "BTC-USD",
            "engine": "native", "policy": {"reseed": True, "cold_ab": False},
            "thresholds": {"crossed_usable_max": 0.01, "missing_usable_max": 0.02,
                           "thin_usable_max": 0.1, "seed_crossed_frac_max": 0.05},
            "quota": {"ok": True, "reason": "ok", "used_gb_before": 0.26, "used_gb_after": 0.26},
            "generated_utc": "2026-07-01T00:00:00Z"}
    meta.update(meta_overrides)
    counts = {c: 0 for c in rv.CLASSES}
    for d in days:
        counts[d["classification"]] = counts.get(d["classification"], 0) + 1
    return {"meta": meta,
            "summary": {"n_days": len(days), "counts": counts, "by_class": {},
                        "coinapi_fill": {"fill_counts": rv._recompute_fill_counts(days)}},
            "days": days}


_TRADE_ONLY_CTX = {"in_usable_days": True, "in_lake_all_days": False,
                   "is_coinbase_fill_day": True, "excluded_reason": None}


def make_calendar(**overrides):
    cal = {
        "anchor_end": "2026-06-22",
        "lake_all_days": ["2025-01-01", "2025-01-02", "2025-01-03"],
        "usable_days": ["2025-01-01", "2025-01-02", "2025-01-03", "2025-01-10", "2025-01-11"],
        "coinbase_fill_days": {
            "2025-01-10": {"book": True, "trades": True},
            "2025-01-11": {"book": False, "trades": True},
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


def clean_report_days(extra_days=()):
    segs3, seams3, policy3, seam3 = partial_plan("2025-01-03")
    _, e3 = day_bounds("2025-01-03")
    days = [
        day_rec("2025-01-01", "lake_usable", fill_block(False, "lake_usable")),
        day_rec("2025-01-02", "lake_present_degraded",
                fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                           full_day_reason="quality_over_usable_bar",
                           fill_segments=[full_day_seg("2025-01-02")], seams=[],
                           seam_policy=dict(rv.DEFAULT_SEAM_POLICY)),
                fillable=True),
        day_rec("2025-01-03", "lake_present_degraded",
                fill_block(True, "quality_over_usable_bar", fill_profile="leading_partial_fill",
                           fill_segments=segs3, seams=seams3, seam_policy=policy3),
                trusted=(seam3, e3), fillable=True),
        day_rec("2025-01-11", "lake_usable", fill_block(False, "lake_usable"),
                calendar=_TRADE_ONLY_CTX),
    ]
    days.extend(extra_days)
    return days


def unresolved_day(day):
    """An unresolved inconclusive day: blocks readiness (status=blocking)."""
    return day_rec(day, "inconclusive", fill_block(None, "no_verdict"),
                   reasons=["seed_rejected:crossed"])


def write_tree(tmp_path, cal=None, report_days=None):
    """Write calendar + plan manifest + one batch report; return (plan_path, cal_path)."""
    cal = cal if cal is not None else make_calendar()
    report = make_report(sorted(report_days if report_days is not None else clean_report_days(),
                                key=lambda r: r["day"]))
    cal_path = tmp_path / "usable_calendar.json"
    cal_path.write_text(json.dumps(cal))
    out_dir = tmp_path / "batches"
    out_dir.mkdir(exist_ok=True)
    days = [d["day"] for d in report["days"]]
    (out_dir / "batch_001_days.txt").write_text("".join(f"{d}\n" for d in days))
    rdir = tmp_path / "reports" / "batch_001"
    rdir.mkdir(parents=True, exist_ok=True)
    (rdir / "coinbase_quality_map.json").write_text(json.dumps(report))
    plan = {"meta": {"input_calendar": str(cal_path), "out_dir": str(out_dir),
                     "generated_utc": "2026-07-02T00:00:00Z"},
            "summary": {"n_batches": 1},
            "batches": [{"file": "batch_001_days.txt", "n_days": len(days),
                         "first_day": days[0], "last_day": days[-1],
                         "report_dir": str(rdir)}],
            "batched_trade_only_fill_days": ["2025-01-11"],
            "skipped": {"fill_days_book_gap": ["2025-01-10"],
                        "excluded_days_by_reason": cal["excluded_days_by_reason"],
                        "days_dropped_as_excluded_or_book_gap": []}}
    plan_path = tmp_path / "plan_manifest.json"
    plan_path.write_text(json.dumps(plan))
    return str(plan_path), str(cal_path)


def ready_manifest(tmp_path, *, cal=None, report_days=None, expect_status="ready"):
    """Run the real reviewer over the synthetic tree; write + return the reviewed manifest."""
    plan_path, cal_path = write_tree(tmp_path, cal=cal, report_days=report_days)
    manifest = rv.build_manifest_readiness(plan_path, cal_path, generated_utc=GENERATED_UTC,
                                           report_only=False)
    assert manifest["meta"]["status"] == expect_status, manifest["blockers"]
    out = tmp_path / "coinbase_backfill_manifest.json"
    rv.write_manifest(manifest, str(out))
    return str(out), manifest


def blocking_manifest(tmp_path):
    cal = make_calendar()
    cal["lake_all_days"] = cal["lake_all_days"] + ["2025-01-04"]
    days = clean_report_days(extra_days=[unresolved_day("2025-01-04")])
    return ready_manifest(tmp_path, cal=cal, report_days=days, expect_status="blocking")


def mutate_manifest(path, mutator):
    """Load, mutate in place via `mutator(manifest)`, rewrite. Returns the path."""
    with open(path) as f:
        m = json.load(f)
    mutator(m)
    with open(path, "w") as f:
        json.dump(m, f, indent=2)
        f.write("\n")
    return path
