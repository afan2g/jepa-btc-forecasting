"""Review completed Coinbase quality-map reports into a human-auditable CoinAPI backfill
manifest (docs/data.md §5a-QualityMap; design
docs/superpowers/specs/2026-07-03-coinbase-backfill-manifest-review-design.md).

GATEKEEPING ONLY — no vendor I/O, no downloads, no live API calls. It reads the batch-plan
manifest (scripts/plan_coinbase_quality_map_batches.py), the per-batch quality-map reports it
registers (scripts/run_coinbase_quality_map.py), and the usable calendar
(ingest/verify_trades_and_calendar.py), and emits a deterministic backfill manifest (JSON +
terminal summary). It does NOT unlock or run the backfill — the §5a gate stays in
ingest/download_coinapi.py / ingest/_common.py.

Stdlib-only on purpose (mirrors scripts/plan_coinbase_quality_map_batches.py): a CI-safe
offline test drives it without pandas/numpy/boto3. The quality-map enum strings are pinned
copies here (importing the runner pulls pandas); a contract test keeps them aligned.

Two mutually-exclusive modes:
  * readiness  (--plan-manifest): the gate; can reach status=ready. Fail-closed by default.
  * inspection (--report ...):     eyeball one or more reports; status is always report_only.

Usage:
  .venv/bin/python scripts/review_coinbase_backfill_manifest.py \
      --plan-manifest data/tmp/coinbase_quality_map_batches/manifest.json
  .venv/bin/python scripts/review_coinbase_backfill_manifest.py --report data/reports/…/…json
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys

# ----------------------------------------------------------- pinned quality-map enums
# Aligned to scripts/run_coinbase_quality_map.py + recon/stitch_policy.py by a contract test.
LAKE_USABLE = "lake_usable"
LAKE_PRESENT_DEGRADED = "lake_present_degraded"
MISSING_NEEDS_COINAPI = "missing_needs_coinapi"
EXCLUDED = "excluded"
INCONCLUSIVE = "inconclusive"
CLASSES = (LAKE_USABLE, LAKE_PRESENT_DEGRADED, MISSING_NEEDS_COINAPI, EXCLUDED, INCONCLUSIVE)

WHY_CODES = ("lake_usable", "quality_over_usable_bar", "lake_book_delta_v2_absent",
             "crossed_seed_source_cross_validated_2026-07-01", "no_verdict",
             "excluded_not_in_scope")

FULL_DAY_FILL = "full_day_fill"
LAKE_ONLY = "lake_only"
PARTIAL_FILL_PROFILES = ("leading_partial_fill", "trailing_partial_fill",
                         "internal_gap_fill", "mixed_partial_fill")
FILL_PROFILES = (LAKE_ONLY, FULL_DAY_FILL, *PARTIAL_FILL_PROFILES)

SEGMENT_SOURCES = ("lake", "coinapi", "excluded")   # pinned from recon.stitch_policy.SOURCES

NS_PER_S = 1_000_000_000
DAY_NS = 86_400 * NS_PER_S
# Pinned copy of recon.stitch_policy.DEFAULT_SEAM_POLICY.as_dict() (contract-tested) — stamped on
# synthesized calendar-gap full-day fills so they carry the same seam policy as report-derived fills.
DEFAULT_SEAM_POLICY = {"seam_guard_s": 60.0, "warmup_consecutive": 3, "fill_min_s": 300.0,
                       "min_lake_segment_s": 3600.0, "span_invalid_max": 0.01,
                       "exclude_labels_crossing_seam": True, "exclude_features_crossing_seam": True}

# ----------------------------------------------------------- cost model (docs §2.2/§6/§8)
BOOK_USD_PER_GB = 1.0
TRADES_USD_PER_GB = 3.0
EST_BOOK_GB_PER_DAY = 2.27          # §2.2/§6 nominal L3, conservative
EST_TRADES_GB_PER_DAY = 0.05        # §8 2.6 GB / 52 days
CREDIT_USD = 25.0                   # §8 flat-files trial pool
DOCS_REFERENCE_USD = 92.0           # §8 calendar-gap figure, 2026-06-22 snapshot (reference only)
MB_PER_GB = 1000.0                  # docs use decimal GB (84.6 GB == 84_600 MB)

# ----------------------------------------------------------- control
MANIFEST_VERSION = 1
INPUT_ERROR_EXIT = 2                # structural/input error (missing/invalid files) — matches planner
BLOCKING_EXIT = 3                   # fail-closed blocking verdict
REPORT_NAME = "coinbase_quality_map.json"
DEFAULT_OUT = "data/reports/backfill/coinbase_backfill_manifest.json"
BLOCKER_KEYS = ("structural", "missing_keys", "coverage_gaps", "inconsistencies",
                "unresolved_days", "batch_incomplete", "book_fill_unavailable",
                "trade_fill_unavailable", "calendar_drift")


class ReviewInputError(ValueError):
    """A clear, user-actionable input failure (missing/invalid plan, report, or calendar)."""


# ----------------------------------------------------------- input helpers
def sha256_file(path: str) -> str:
    """Hex SHA-256 of a file's raw bytes — pins each input's identity in meta.inputs."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json_object(path: str, *, what: str) -> dict:
    """Load a JSON object or raise a clear ReviewInputError. Fail-closed: a missing/invalid/
    non-object input is a structural error (exit 2), never a silent default."""
    if not os.path.exists(path):
        raise ReviewInputError(f"{what} not found: {path}")
    try:
        with open(path) as f:
            obj = json.load(f)
    except json.JSONDecodeError as e:
        raise ReviewInputError(f"{what} {path} is not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise ReviewInputError(f"{what} {path} must be a JSON object")
    return obj


# ----------------------------------------------------------- calendar accessors
def _fill_days(cal: dict) -> dict:
    return cal.get("coinbase_fill_days") or {}


def book_gap_days(cal: dict) -> set:
    """Calendar days whose Lake `book_delta_v2` is absent (coinbase_fill_days[d].book==true)."""
    return {d for d, v in _fill_days(cal).items() if (v or {}).get("book") is True}


def trade_fill_days(cal: dict) -> set:
    """Calendar days whose Coinbase trades are gapped (coinbase_fill_days[d].trades==true)."""
    return {d for d, v in _fill_days(cal).items() if (v or {}).get("trades") is True}


def calendar_batch_days(cal: dict) -> list:
    """The present-book day-set the batch planner would map: lake_all_days ∪ trade-only fill days,
    minus book-gap and excluded days. Stdlib reimplementation of
    plan_coinbase_quality_map_batches.select_days()['batch_days'] (pinned by contract test)."""
    lake_all = set(cal.get("lake_all_days") or [])
    fill = _fill_days(cal)
    excluded = set(cal.get("excluded_days_by_reason") or {})
    bg = {d for d, v in fill.items() if (v or {}).get("book") is True}
    trade_only = set(fill) - bg
    return sorted((lake_all | trade_only) - bg - excluded)


def _fill_status(cal: dict, day: str):
    return (cal.get("fill_status") or {}).get(day)


def measured_mb(cal: dict, day: str, product: str):
    """Measured per-day size (MB) for `product` in ("book","trades") from fill_status, or None."""
    fs = _fill_status(cal, day)
    if not fs:
        return None
    p = fs.get(product)
    if isinstance(p, dict) and p.get("present"):
        mb = p.get("mb")
        return float(mb) if isinstance(mb, (int, float)) else None
    return None


def is_fillable(cal: dict, day: str, product: str) -> bool:
    """True iff CoinAPI `product` for `day` is verifiably available (present+ok, no error, not in
    the unfillable/probe-error lists) per the calendar verifier's fill_status."""
    fs = _fill_status(cal, day)
    if not fs or fs.get("error") is True:
        return False
    if day in set(cal.get("fill_days_unfillable") or []):
        return False
    if day in set(cal.get("fill_days_probe_error") or []):
        return False
    p = fs.get(product)
    return bool(isinstance(p, dict) and p.get("present") and p.get("ok"))


# ----------------------------------------------------------- cost model
def gb_from_mb(mb):
    return None if mb is None else float(mb) / MB_PER_GB


def day_book_gb(cal: dict, day: str) -> tuple:
    """(gb, basis) for a book fill on `day`: measured fill_status where available, else the
    conservative nominal per-day estimate. Partial fills are charged as a full day-file, so the
    per-day figure is used regardless of profile."""
    mb = measured_mb(cal, day, "book")
    return (gb_from_mb(mb), "measured") if mb is not None else (EST_BOOK_GB_PER_DAY, "estimated")


def day_trades_gb(cal: dict, day: str) -> tuple:
    mb = measured_mb(cal, day, "trades")
    return (gb_from_mb(mb), "measured") if mb is not None else (EST_TRADES_GB_PER_DAY, "estimated")


def book_usd(gb: float) -> float:
    return round(float(gb) * BOOK_USD_PER_GB, 4)


def trades_usd(gb: float) -> float:
    return round(float(gb) * TRADES_USD_PER_GB, 4)


# ----------------------------------------------------------- validation
_REQUIRED_REPORT_KEYS = ("meta", "summary", "days")
_REQUIRED_DAY_KEYS = ("day", "classification", "coinapi_fill")
_REQUIRED_FILL_KEYS = ("needs_fill", "why", "fill_profile")


def report_missing_keys(report: dict) -> list:
    return [k for k in _REQUIRED_REPORT_KEYS if k not in report]


# The runner's deterministic classification -> (needs_fill, why) contract
# (run_coinbase_quality_map.coinapi_fill_decision). A report whose coinapi_fill disagrees with its
# own classification is stale/corrupt — trusting it would silently DROP a required fill (a degraded
# or missing day mismarked needs_fill=false) or invent one, so it must fail closed.
_CLASS_FILL_CONTRACT = {
    MISSING_NEEDS_COINAPI: (True, "lake_book_delta_v2_absent"),
    LAKE_PRESENT_DEGRADED: (True, "quality_over_usable_bar"),
    LAKE_USABLE: (False, "lake_usable"),
    EXCLUDED: (None, "excluded_not_in_scope"),
}
# `inconclusive` has two legitimate outcomes: unresolved (no_verdict), or the crossed-seed-source fill
# — but the latter is valid ONLY when the reason marks it (the runner routes it there iff
# SEED_SOURCE_UNRELIABLE is in reasons; the provisional 2026-07-01 cross-validation policy).
_CROSSED_SOURCE_FILL = (True, "crossed_seed_source_cross_validated_2026-07-01")
_SEED_SOURCE_UNRELIABLE = "seed_accepted_but_source_unreliable"  # run_coinbase_quality_map.SEED_SOURCE_UNRELIABLE


def _fill_contract_issue(cls, needs_fill, why, reasons):
    """None if (needs_fill, why[, reasons]) matches the classification's runner contract, else a code."""
    if cls in _CLASS_FILL_CONTRACT:
        if (needs_fill, why) != _CLASS_FILL_CONTRACT[cls]:
            return (f"fill_decision_contradicts_classification:{cls}:got=({needs_fill},{why}):"
                    f"expected={_CLASS_FILL_CONTRACT[cls]}")
    elif cls == INCONCLUSIVE:
        if (needs_fill, why) == _CROSSED_SOURCE_FILL:
            # the crossed-source fill path is only legitimate when the reason marks it — otherwise a
            # stale report could convert a no_verdict blocker into an approved fill.
            if _SEED_SOURCE_UNRELIABLE not in (reasons or ()):
                return (f"fill_decision_contradicts_classification:{cls}:crossed_source_fill_without_"
                        f"{_SEED_SOURCE_UNRELIABLE}")
        elif (needs_fill, why) != (None, "no_verdict"):
            return f"fill_decision_contradicts_classification:{cls}:got=({needs_fill},{why})"
    return None


def _fill_segments_issues(day, segments, seams) -> list:
    """A fill day's segments must PARTITION [day_open, day_close) as ordered, contiguous half-open
    [start, end) spans with a known source (spec §9 #3) — a manifest is an executable stitch plan, so
    a segment that starts after day-open, ends before day-close, overlaps/gaps a neighbour, or uses an
    unexpected source must fail review. `seams` must be EXACTLY the source-change boundaries between
    adjacent segments. Structural checks always run; the day-open/close coverage check runs only when
    `day` is a parseable ISO date."""
    issues = []
    try:
        day_open, day_end = _day_bounds_ns(day)
    except (TypeError, ValueError):
        day_open = day_end = None
    prev_end = None
    for i, s in enumerate(segments):
        if not isinstance(s, dict):
            issues.append(f"fill_segment[{i}]_not_object")
            continue
        st, en, src = s.get("start_ts"), s.get("end_ts"), s.get("source")
        if not isinstance(st, int) or not isinstance(en, int):
            issues.append(f"fill_segment[{i}]_non_int_bounds")
            continue
        if en <= st:
            issues.append(f"fill_segment[{i}]_non_positive_span")
        if src not in SEGMENT_SOURCES:
            issues.append(f"fill_segment[{i}]_bad_source:{src}")
        if prev_end is not None:
            if st != prev_end:
                issues.append(f"fill_segments_gap_or_overlap_at[{i}]")
        elif day_open is not None and st != day_open:
            issues.append("fill_segments_start_ne_day_open")
        prev_end = en
    if day_end is not None and prev_end is not None and prev_end != day_end:
        issues.append("fill_segments_end_ne_day_close")
    # seams must be exactly the source-change boundaries (stitch_policy: the RIGHT segment's start_ts
    # wherever adjacent sources differ). An empty/stale seams list drops the guard bands a consumer
    # needs at a real Lake<->CoinAPI switch.
    expected_seams = [cur.get("start_ts") for prev, cur in zip(segments, segments[1:])
                      if isinstance(prev, dict) and isinstance(cur, dict)
                      and prev.get("source") != cur.get("source")]
    # only derive a mismatch when seams is a list — a non-list seams (e.g. a scalar) is caught by the
    # fill_day_missing_seams check in day_record_issues; `list(seams)` here would crash on a scalar.
    if isinstance(seams, list) and seams != expected_seams:
        issues.append(f"seams_mismatch:expected={expected_seams}:got={seams}")
    return issues


def day_record_issues(rec: dict) -> list:
    """Structural + enum + contradiction issues for ONE report-backed day record. Never applied
    to calendar-only days (they carry classification=None by construction)."""
    issues = []
    for k in _REQUIRED_DAY_KEYS:
        if k not in rec:
            issues.append(f"missing_key:{k}")
    cf = rec.get("coinapi_fill")
    if not isinstance(cf, dict):
        issues.append("missing_key:coinapi_fill")
        return issues
    for k in _REQUIRED_FILL_KEYS:
        if k not in cf:
            issues.append(f"missing_key:coinapi_fill.{k}")

    cls = rec.get("classification")
    if "classification" in rec and cls not in CLASSES:
        issues.append(f"unknown_classification:{cls}")
    nf, why = cf.get("needs_fill"), cf.get("why")
    prof, fdr = cf.get("fill_profile"), cf.get("full_day_reason")
    if why is not None and why not in WHY_CODES:
        issues.append(f"unknown_why:{why}")
    if prof is not None and prof not in FILL_PROFILES:
        issues.append(f"unknown_fill_profile:{prof}")

    # contradictions (spec §4 / §9 invariants 4-5)
    if nf is True and prof in (None, LAKE_ONLY):
        issues.append("needs_fill_without_plan")
    if nf is not True and prof is not None:
        issues.append("plan_without_needs_fill")
    # full_day_reason non-null IFF fill_profile == full_day_fill (spec §9 #5). Enforce BOTH
    # directions: the second check covers partial profiles AND the null/lake_only no-plan shapes
    # (a stale/corrupt report could otherwise carry full_day_reason with fill_profile=null).
    if prof == FULL_DAY_FILL and fdr is None:
        issues.append("full_day_without_reason")
    if prof != FULL_DAY_FILL and fdr is not None:
        issues.append("full_day_reason_without_full_day_fill")
    if prof == FULL_DAY_FILL:
        q = rec.get("quality") or {}
        if q.get("trusted_lake_start_ts") is not None or q.get("trusted_lake_end_ts") is not None:
            issues.append("full_day_with_trusted_lake_span")
    # a fill day must carry an EXECUTABLE stitch plan (segments partition the day; spec §9 #3).
    # The runner always emits these for a needs_fill day; their absence is a corrupt report that
    # would otherwise pass an approved fill with no plan for the backfill runner to execute.
    if nf is True and prof not in (None, LAKE_ONLY):
        segs = cf.get("fill_segments")
        if not isinstance(segs, list) or not segs:
            issues.append("fill_day_missing_fill_segments")
        else:
            # segments must partition the day AND seams must match their source-change boundaries
            issues.extend(_fill_segments_issues(rec.get("day"), segs, cf.get("seams")))
        if not isinstance(cf.get("seams"), list):        # may be empty (full_day) but never null
            issues.append("fill_day_missing_seams")
        if not isinstance(cf.get("seam_policy"), dict):
            issues.append("fill_day_missing_seam_policy")
    # the fill decision must match the classification (+ reasons) it was derived from (never
    # drop/invent a fill, and only route crossed-source inconclusive days to a fill)
    contract = _fill_contract_issue(cls, nf, why, rec.get("reasons"))
    if contract is not None:
        issues.append(contract)
    return issues


def recompute_class_counts(days: list) -> dict:
    counts = {c: 0 for c in CLASSES}
    for r in days:
        c = r.get("classification")
        if c in counts:
            counts[c] += 1
    return counts


def _recompute_fill_counts(days: list) -> dict:
    """Recompute the runner's summary.coinapi_fill.fill_counts from days[] (build_report's logic)."""
    profiles = (FULL_DAY_FILL, *PARTIAL_FILL_PROFILES)
    counts = {"needs_fill": 0, **{p: 0 for p in profiles},
              "crossed_source_full_day": 0, "no_verdict": 0, "no_fill": 0, "not_in_scope": 0}
    for r in days:
        cf = r.get("coinapi_fill") or {}
        nf = cf.get("needs_fill")
        if nf:
            counts["needs_fill"] += 1
            prof = cf.get("fill_profile")
            if prof in counts:
                counts[prof] += 1
            if cf.get("full_day_reason") == "crossed_seed_source":   # stitch_policy.REASON_CROSSED_SOURCE
                counts["crossed_source_full_day"] += 1
        elif nf is False:
            counts["no_fill"] += 1
        else:
            counts["not_in_scope" if cf.get("why") == "excluded_not_in_scope" else "no_verdict"] += 1
    return counts


def summary_count_issues(report: dict) -> list:
    """The report's summary.counts AND summary.coinapi_fill.fill_counts must each equal a
    recomputation over days[] (days[] is primary; both summary blocks are cross-checks)."""
    days = report.get("days") or []
    got = (report.get("summary") or {}).get("counts") or {}
    want = recompute_class_counts(days)
    issues = []
    for c in CLASSES:
        if int(got.get(c, 0)) != want[c]:
            issues.append(f"summary_counts_mismatch:{c}:summary={got.get(c)}:recomputed={want[c]}")
    fc_got = ((report.get("summary") or {}).get("coinapi_fill") or {}).get("fill_counts")
    if not isinstance(fc_got, dict):
        # a report without fill_counts (pre-extension/stale) can't be cross-checked → fail closed
        issues.append("summary_fill_counts_missing")
    else:   # a stale fill_counts from a prior run must not pass review
        for k, v in _recompute_fill_counts(days).items():
            if int(fc_got.get(k, 0)) != v:
                issues.append(f"fill_counts_mismatch:{k}:summary={fc_got.get(k)}:recomputed={v}")
    return issues


# ----------------------------------------------------------- per-day canonical record
def _calendar_context(cal: dict, day: str) -> dict:
    fill = _fill_days(cal).get(day) or {}
    return {"in_lake_all_days": day in set(cal.get("lake_all_days") or []),
            "in_usable_days": day in set(cal.get("usable_days") or []),
            "is_coinbase_fill_day": day in _fill_days(cal),
            "book_gap": (fill.get("book") is True),
            "trades_gap": (fill.get("trades") is True),
            "excluded_reason": (cal.get("excluded_days_by_reason") or {}).get(day)}


def _empty_book_fill() -> dict:
    return {"needed": False, "source": None, "kind": None, "why": None, "fill_profile": None,
            "full_day_reason": None, "fill_segments": None, "seams": None, "seam_policy": None,
            "trusted_lake_start_ts": None, "trusted_lake_end_ts": None,
            "gb": 0.0, "gb_basis": "measured", "usd": 0.0}


def _empty_trade_fill() -> dict:
    return {"needed": False, "source": None, "measured_mb": None,
            "gb": 0.0, "gb_basis": "measured", "usd": 0.0}


def _iso_ns(ts_ns: int) -> str:
    """UTC ISO stamp for an int-ns timestamp — mirrors recon.stitch_policy._iso_utc's format."""
    secs, rem = divmod(int(ts_ns), NS_PER_S)
    base = dt.datetime.fromtimestamp(secs, tz=dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    return f"{base}.{rem:09d}Z" if rem else f"{base}Z"


def _day_bounds_ns(day: str) -> tuple:
    """(day_open_ts, day_end_ts) in int ns for a YYYY-MM-DD partition day (midnight UTC bounds)."""
    d = dt.date.fromisoformat(day)
    day_open = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()) * NS_PER_S
    return day_open, day_open + DAY_NS


def _synth_full_day_plan(day: str, reason: str) -> tuple:
    """Full-day CoinAPI stitch plan for a calendar-derived fill (no report grid available): one
    whole-day coinapi segment, no seams, the default seam policy — so a calendar-gap fill carries
    the same executable plan shape as a report-derived full_day_fill, not null plan fields."""
    day_open, day_end = _day_bounds_ns(day)
    seg = {"source": "coinapi", "start_ts": day_open, "start_iso": _iso_ns(day_open),
           "end_ts": day_end, "end_iso": _iso_ns(day_end), "reason": reason}
    return [seg], [], dict(DEFAULT_SEAM_POLICY)


def build_day_record(day: str, report_rec: dict | None, cal: dict) -> dict:
    """Merge a report day record (or None for a calendar-only day) with the calendar into the
    canonical per-day manifest record. Stitch decisions are copied VERBATIM."""
    cctx = _calendar_context(cal, day)
    sources: list = []
    if report_rec is not None:
        sources.append("quality_map")
    if cctx["book_gap"]:
        sources.append("calendar_gap")
    if cctx["trades_gap"]:
        sources.append("calendar_trade")
    if cctx["excluded_reason"] is not None:
        sources.append("calendar_excluded")

    # .get, not [...]: a report day missing `classification` is a missing_keys blocker
    # (check_report_consistency), and readiness still builds records — a hard subscript here would
    # crash instead of failing closed with status=blocking / exit 3.
    classification = report_rec.get("classification") if report_rec else None
    cf = (report_rec or {}).get("coinapi_fill") or {}
    q = (report_rec or {}).get("quality") or {}

    book = _empty_book_fill()
    if cf.get("needs_fill") is True:
        prof = cf.get("fill_profile")
        gb, basis = day_book_gb(cal, day)
        book.update(
            needed=True,
            source="both" if cctx["book_gap"] else "quality_map",
            kind="partial" if prof in PARTIAL_FILL_PROFILES else "full_day",
            why=cf.get("why"), fill_profile=prof, full_day_reason=cf.get("full_day_reason"),
            fill_segments=cf.get("fill_segments"), seams=cf.get("seams"),
            seam_policy=cf.get("seam_policy"),
            trusted_lake_start_ts=q.get("trusted_lake_start_ts"),
            trusted_lake_end_ts=q.get("trusted_lake_end_ts"),
            gb=gb, gb_basis=basis, usd=book_usd(gb))
    elif cctx["book_gap"]:
        # calendar book-gap day not carried as a report fill → synthesize a full-day book fill WITH
        # an executable stitch plan (whole-day coinapi segment), never null plan fields.
        gb, basis = day_book_gb(cal, day)
        segs, seams_, policy = _synth_full_day_plan(day, "calendar_book_gap")
        book.update(needed=True, source="calendar_gap", kind="full_day", why="calendar_book_gap",
                    fill_profile=FULL_DAY_FILL, full_day_reason="calendar_book_gap",
                    fill_segments=segs, seams=seams_, seam_policy=policy,
                    gb=gb, gb_basis=basis, usd=book_usd(gb))

    trade = _empty_trade_fill()
    if cctx["trades_gap"]:
        gb, basis = day_trades_gb(cal, day)
        trade.update(needed=True, source="calendar", measured_mb=measured_mb(cal, day, "trades"),
                     gb=gb, gb_basis=basis, usd=trades_usd(gb))

    excluded = None
    if cctx["excluded_reason"] is not None:
        excluded = {"reason": cctx["excluded_reason"]}
    elif classification == EXCLUDED:
        excluded = {"reason": (report_rec or {}).get("reasons") or []}

    unresolved = None
    if cf.get("needs_fill") is None and cf.get("why") == "no_verdict":
        unresolved = {"why": "no_verdict", "classification": classification,
                      "reasons": (report_rec or {}).get("reasons") or []}

    return {"day": day, "classification": classification, "sources": sources,
            "calendar": cctx, "book_fill": book, "trade_fill": trade,
            "excluded": excluded, "unresolved": unresolved, "notes": []}


# ----------------------------------------------------------- blockers container
def new_blockers() -> dict:
    return {k: [] for k in BLOCKER_KEYS}


def any_blockers(blockers: dict) -> bool:
    return any(blockers[k] for k in BLOCKER_KEYS)


# ----------------------------------------------------------- plan + report loading
def load_batch_reports(plan: dict) -> tuple:
    """Load every report the plan registers (report_dir/coinbase_quality_map.json) and index
    day → report record. A missing/invalid report or a day appearing in two batches is a
    structural error (fail-closed, exit 2)."""
    batches = plan.get("batches")
    if not isinstance(batches, list):
        raise ReviewInputError("plan manifest 'batches' must be a list")
    reports, day_index = [], {}
    for b in batches:
        if not isinstance(b, dict):
            raise ReviewInputError(f"plan 'batches' entries must be objects, got {type(b).__name__}")
        rdir = b.get("report_dir")
        if not rdir:
            raise ReviewInputError(f"plan batch {b.get('file')!r} has no report_dir")
        rpath = os.path.join(rdir, REPORT_NAME)
        report = load_json_object(rpath, what="quality-map report")
        reports.append({"path": rpath, "report_dir": rdir, "batch": b, "report": report})
        days = report.get("days")
        if not isinstance(days, list):
            # An ABSENT 'days' key is a missing_keys blocker (report_missing_keys, exit 3); a
            # present-but-wrong-type 'days' is a structural malformation — fail closed here (exit 2)
            # rather than crash on rec.get(...) below.
            if "days" in report:
                raise ReviewInputError(f"{rpath}: report 'days' must be a list of objects")
            continue
        for rec in days:
            if not isinstance(rec, dict):
                raise ReviewInputError(f"{rpath}: report 'days' must contain objects, got "
                                       f"{type(rec).__name__}")
            d = rec.get("day")
            if d in day_index:
                raise ReviewInputError(f"day {d} appears in more than one batch report "
                                       "(duplicate_across_batches); batch day-sets must be disjoint")
            day_index[d] = rec
    return reports, day_index


def _batch_ran(report: dict) -> list:
    """Issues indicating the batch did NOT actually run to completion (spec §fix-5). Requires the
    quota completion field `reason` to be present and prove the run happened — an empty `quota: {}`
    or a missing/refusal `reason` fails closed rather than passing as a ran batch."""
    issues = []
    meta = report.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("quota"), dict):
        return ["missing meta.quota"]
    reason = meta["quota"].get("reason")
    if reason is None:
        issues.append("missing meta.quota.reason")
    elif reason not in ("ok", "no_days_to_load"):   # quota_headroom / exceeds_auto_cap / unknown
        issues.append(f"quota not ok: {reason}")
    n_days = (report.get("summary") or {}).get("n_days")
    if n_days != len(report.get("days") or []):
        issues.append(f"summary.n_days {n_days} != len(days) {len(report.get('days') or [])}")
    return issues


def _check_batch_matches_plan(r: dict, out_dir, blockers: dict) -> None:
    """The report a batch points at must have been produced FOR that batch: its day-set must match
    the batch's authoritative days-file (and the plan's n_days/first/last). A stale report whose
    internal summary.n_days is self-consistent but was built for a different day-set would otherwise
    slip past `_batch_ran` (spec §4 completeness)."""
    b = r["batch"] or {}
    rep_days = sorted(rec.get("day") for rec in r["report"].get("days") or [] if rec.get("day"))
    if b.get("n_days") is not None and b["n_days"] != len(rep_days):
        blockers["batch_incomplete"].append(
            f"{r['report_dir']}: plan n_days {b['n_days']} != report days {len(rep_days)}")
    if rep_days:
        if b.get("first_day") is not None and b["first_day"] != rep_days[0]:
            blockers["batch_incomplete"].append(
                f"{r['report_dir']}: plan first_day {b['first_day']} != report {rep_days[0]}")
        if b.get("last_day") is not None and b["last_day"] != rep_days[-1]:
            blockers["batch_incomplete"].append(
                f"{r['report_dir']}: plan last_day {b['last_day']} != report {rep_days[-1]}")
    # authoritative check: the batch days-file the runner consumed must equal the report's day-set.
    # n_days/first/last alone can't catch a stale report with the same count+endpoints but different
    # middle days, so a missing out_dir/file must FAIL CLOSED rather than bypass this guard.
    bf = b.get("file")
    if not out_dir or not bf:
        blockers["batch_incomplete"].append(
            f"{r['report_dir']}: cannot verify against the authoritative batch days-file "
            "(plan meta.out_dir or batch 'file' missing)")
    else:
        path = os.path.join(out_dir, bf)
        if not os.path.exists(path):
            blockers["batch_incomplete"].append(f"{r['report_dir']}: batch file {bf} missing under {out_dir}")
        else:
            with open(path) as f:
                planned = sorted(x.strip() for x in f.read().splitlines() if x.strip())
            if planned != rep_days:
                blockers["batch_incomplete"].append(
                    f"{r['report_dir']}: report day-set != batch file {bf} (stale report?)")


def check_completeness(plan: dict, reports: list, day_index: dict, cal: dict,
                       blockers: dict) -> None:
    """Plan-driven completeness (spec §4). Populates blockers in place."""
    out_dir = (plan.get("meta") or {}).get("out_dir")
    for r in reports:
        for msg in _batch_ran(r["report"]):
            blockers["batch_incomplete"].append(f"{r['report_dir']}: {msg}")
        _check_batch_matches_plan(r, out_dir, blockers)

    expected = set(calendar_batch_days(cal))
    mapped = set(day_index)
    bg = book_gap_days(cal)
    for d in sorted(expected - mapped):
        blockers["coverage_gaps"].append(f"day_not_mapped:{d}")
    # a mapped book-gap day (via --include-gap-days) is legitimate, not "unexpected"
    for d in sorted(mapped - expected - bg):
        blockers["coverage_gaps"].append(f"unexpected_day:{d}")

    withheld = set((plan.get("skipped") or {}).get("fill_days_book_gap") or [])
    for d in sorted(bg):
        rec = day_index.get(d)
        mapped_missing = rec is not None and rec.get("classification") == MISSING_NEEDS_COINAPI
        if not mapped_missing and d not in withheld:
            blockers["coverage_gaps"].append(f"gap_day_unmapped:{d}")

    dropped = (plan.get("skipped") or {}).get("days_dropped_as_excluded_or_book_gap") or []
    if dropped:
        blockers["coverage_gaps"].append(f"contradictory_calendar_dropped:{sorted(dropped)}")


# ----------------------------------------------------------- consistency / drift / availability
def check_report_consistency(reports: list, blockers: dict) -> None:
    """Per-report structural/enum/contradiction issues, summary count cross-check, cross-report
    meta drift, and the global unresolved-day block. Populates blockers in place."""
    ref_meta = None
    for r in reports:
        report = r["report"]
        for k in report_missing_keys(report):
            blockers["missing_keys"].append(f"{r['report_dir']}: missing {k}")
        for rec in report.get("days") or []:
            for issue in day_record_issues(rec):
                if issue.startswith("missing_key:"):
                    blockers["missing_keys"].append(f"{rec.get('day')}: {issue}")
                else:
                    blockers["inconsistencies"].append(f"{rec.get('day')}: {issue}")
            cf = rec.get("coinapi_fill") or {}
            if cf.get("needs_fill") is None and cf.get("why") == "no_verdict":
                blockers["unresolved_days"].append(rec.get("day"))
        for issue in summary_count_issues(report):
            blockers["inconsistencies"].append(f"{r['report_dir']}: {issue}")
        meta = report.get("meta") or {}
        pin = {"exchange": meta.get("exchange"), "symbol": meta.get("symbol"),
               "thresholds": meta.get("thresholds")}
        if ref_meta is None:
            ref_meta = pin
        elif pin != ref_meta:
            for key in ("exchange", "symbol", "thresholds"):
                if pin[key] != ref_meta[key]:
                    blockers["inconsistencies"].append(
                        f"meta_drift:{key}:{ref_meta[key]!r}!={pin[key]!r}")
    blockers["unresolved_days"] = sorted(set(blockers["unresolved_days"]))


def check_calendar_drift(reports: list, cal: dict, blockers: dict) -> None:
    """A mapped day's report `calendar` context must agree with the loaded usable calendar."""
    lake_all = set(cal.get("lake_all_days") or [])
    fill = _fill_days(cal)
    excluded = cal.get("excluded_days_by_reason") or {}
    for r in reports:
        for rec in r["report"].get("days") or []:
            d = rec.get("day")
            rc = rec.get("calendar") or {}
            if "in_lake_all_days" in rc and rc["in_lake_all_days"] != (d in lake_all):
                blockers["calendar_drift"].append(f"{d}:in_lake_all_days")
            if "is_coinbase_fill_day" in rc and rc["is_coinbase_fill_day"] != (d in fill):
                blockers["calendar_drift"].append(f"{d}:is_coinbase_fill_day")
            if "excluded_reason" in rc and rc["excluded_reason"] != excluded.get(d):
                blockers["calendar_drift"].append(f"{d}:excluded_reason")


def check_fill_availability(cal: dict, blockers: dict) -> None:
    """Every calendar book-gap / trade-fill day must be verifiably fillable (fill_status)."""
    for d in sorted(book_gap_days(cal)):
        if not is_fillable(cal, d, "book"):
            blockers["book_fill_unavailable"].append(d)
    for d in sorted(trade_fill_days(cal)):
        if not is_fillable(cal, d, "trades"):
            blockers["trade_fill_unavailable"].append(d)


# ----------------------------------------------------------- sections + cost summary + assembly
def _sections(days: list) -> dict:
    sec = {"full_day_book_fills": [], "partial_day_book_fills": [], "trade_fills": [],
           "lake_usable_days": [], "lake_present_degraded_days": [], "excluded_days": [],
           "unresolved_days": []}
    for r in days:
        d = r["day"]
        bf = r["book_fill"]
        if bf["needed"] and bf["kind"] == "full_day":
            sec["full_day_book_fills"].append(d)
        elif bf["needed"] and bf["kind"] == "partial":
            sec["partial_day_book_fills"].append(d)
        if r["trade_fill"]["needed"]:
            sec["trade_fills"].append(d)
        if r["classification"] == LAKE_USABLE:
            sec["lake_usable_days"].append(d)
        if r["classification"] == LAKE_PRESENT_DEGRADED:
            sec["lake_present_degraded_days"].append(d)
        if r["excluded"] is not None:
            sec["excluded_days"].append(d)
        if r["unresolved"] is not None:
            sec["unresolved_days"].append(d)
    return {k: sorted(v) for k, v in sec.items()}


def _cost_summary(days: list, cal: dict) -> dict:
    book_m = book_e = trades_m = trades_e = 0.0
    full_n = partial_n = trade_n = 0
    for r in days:
        bf, tf = r["book_fill"], r["trade_fill"]
        if bf["needed"]:
            if bf["gb_basis"] == "measured":
                book_m += bf["gb"]
            else:
                book_e += bf["gb"]
            full_n += bf["kind"] == "full_day"
            partial_n += bf["kind"] == "partial"
        if tf["needed"]:
            trade_n += 1
            if tf["gb_basis"] == "measured":
                trades_m += tf["gb"]
            else:
                trades_e += tf["gb"]
    book_usd_total = book_usd(book_m + book_e)
    trades_usd_total = trades_usd(trades_m + trades_e)
    gross = round(book_usd_total + trades_usd_total, 4)
    # calendar-gap baseline: measured fill_status over calendar book-gap + trade-fill days only
    base_book_gb = sum((measured_mb(cal, d, "book") or 0.0) / MB_PER_GB
                       for d in sorted(book_gap_days(cal)))
    base_trade_gb = sum((measured_mb(cal, d, "trades") or 0.0) / MB_PER_GB
                        for d in sorted(trade_fill_days(cal)))
    baseline = round(book_usd(base_book_gb) + trades_usd(base_trade_gb), 4)
    low = round(book_usd(book_m) + trades_usd(trades_m), 4)   # measured-only
    return {
        "book_fill_days": full_n + partial_n, "full_book_fill_days": full_n,
        "partial_book_fill_days": partial_n, "trade_fill_days": trade_n,
        "book_gb_measured": round(book_m, 4), "book_gb_estimated": round(book_e, 4),
        "book_gb_total": round(book_m + book_e, 4),
        "trades_gb_measured": round(trades_m, 4), "trades_gb_estimated": round(trades_e, 4),
        "trades_gb_total": round(trades_m + trades_e, 4),
        "book_usd": book_usd_total, "trades_usd": trades_usd_total, "gross_usd": gross,
        "credit_usd": CREDIT_USD, "net_usd": round(gross - CREDIT_USD, 4),
        "calendar_gap_baseline_usd": baseline,
        "quality_map_addition_usd": round(gross - baseline, 4),
        "docs_reference_usd": DOCS_REFERENCE_USD,
        "band": {"low_usd": low, "high_usd": gross},
    }


def _cost_model_block() -> dict:
    return {"book_usd_per_gb": BOOK_USD_PER_GB, "trades_usd_per_gb": TRADES_USD_PER_GB,
            "est_book_gb_per_day": EST_BOOK_GB_PER_DAY,
            "est_trades_gb_per_day": EST_TRADES_GB_PER_DAY, "credit_usd": CREDIT_USD,
            "partial_day_charged_as_full_day": True, "tiered_discount_applied": False}


def _universe(reports: list, cal: dict) -> list:
    days = set()
    for r in reports:
        for rec in r["report"].get("days") or []:
            days.add(rec.get("day"))
    days |= book_gap_days(cal) | trade_fill_days(cal)
    days |= set(cal.get("excluded_days_by_reason") or {})
    return sorted(d for d in days if d)


def _thresholds(reports: list):
    for r in reports:
        t = (r["report"].get("meta") or {}).get("thresholds")
        if t is not None:
            return t
    return None


def _assemble(days, sections, cost, blockers, *, status, scope_complete, generated_utc,
              inputs, thresholds, exchange, symbol) -> dict:
    return {
        "manifest_version": MANIFEST_VERSION,
        "meta": {"kind": "coinbase_backfill_review",
                 "tool": "scripts/review_coinbase_backfill_manifest.py",
                 "generated_utc": generated_utc, "status": status,
                 "scope_complete": scope_complete, "exchange": exchange, "symbol": symbol,
                 "thresholds": thresholds, "inputs": inputs, "cost_model": _cost_model_block()},
        "days": days, "sections": sections, "cost_summary": cost, "blockers": blockers,
    }


def build_manifest_readiness(plan_path, cal_path, *, generated_utc, report_only) -> dict:
    plan = load_json_object(plan_path, what="plan manifest")
    resolved_cal = cal_path or (plan.get("meta") or {}).get("input_calendar")
    if not resolved_cal:
        raise ReviewInputError("no usable calendar: pass --usable-calendar or set "
                               "plan meta.input_calendar")
    cal = load_json_object(resolved_cal, what="usable calendar")
    reports, day_index = load_batch_reports(plan)

    blockers = new_blockers()
    check_completeness(plan, reports, day_index, cal, blockers)
    check_report_consistency(reports, blockers)
    check_calendar_drift(reports, cal, blockers)
    check_fill_availability(cal, blockers)

    days = [build_day_record(d, day_index.get(d), cal) for d in _universe(reports, cal)]
    sections, cost = _sections(days), _cost_summary(days, cal)

    blocking = any_blockers(blockers)
    status = "blocking" if blocking else "ready"
    inputs = {
        "plan_manifest": {"path": plan_path, "sha256": sha256_file(plan_path)},
        "usable_calendar": {"path": resolved_cal, "sha256": sha256_file(resolved_cal),
                            "anchor_end": cal.get("anchor_end")},
        "batch_reports": [{"report_dir": r["report_dir"], "path": r["path"],
                           "sha256": sha256_file(r["path"]),
                           "batch_file": (r["batch"] or {}).get("file"),
                           "n_days": len(r["report"].get("days") or [])} for r in reports],
        "n_batches": len(reports),
        "plan_generated_utc": (plan.get("meta") or {}).get("generated_utc"),
    }
    m0 = (reports[0]["report"].get("meta") if reports else {}) or {}
    return _assemble(days, sections, cost, blockers, status=status,
                     scope_complete=(not blocking), generated_utc=generated_utc, inputs=inputs,
                     thresholds=_thresholds(reports), exchange=m0.get("exchange"),
                     symbol=m0.get("symbol"))


def build_manifest_inspection(report_paths, cal_path, *, generated_utc) -> dict:
    if not report_paths:
        raise ReviewInputError("inspection mode needs at least one --report")
    reports = []
    for p in report_paths:
        reports.append({"path": p, "report_dir": os.path.dirname(p), "batch": None,
                        "report": load_json_object(p, what="quality-map report")})
    cal = load_json_object(cal_path, what="usable calendar") if cal_path else {}
    day_index = {}
    for r in reports:
        for rec in r["report"].get("days") or []:
            day_index.setdefault(rec.get("day"), rec)
    days = [build_day_record(d, day_index.get(d), cal) for d in _universe(reports, cal)]
    inputs = {"batch_reports": [{"path": r["path"], "sha256": sha256_file(r["path"])}
                                for r in reports],
              "usable_calendar": ({"path": cal_path, "sha256": sha256_file(cal_path)}
                                  if cal_path else None)}
    m0 = reports[0]["report"].get("meta") or {}
    return _assemble(days, _sections(days), _cost_summary(days, cal), new_blockers(),
                     status="report_only", scope_complete=False, generated_utc=generated_utc,
                     inputs=inputs, thresholds=_thresholds(reports), exchange=m0.get("exchange"),
                     symbol=m0.get("symbol"))


# ----------------------------------------------------------- output + CLI
def write_manifest(manifest: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
        f.write("\n")
    return path


def print_summary(manifest: dict) -> None:
    meta, s, c = manifest["meta"], manifest["sections"], manifest["cost_summary"]
    print("\n" + "=" * 74)
    print(f"  COINBASE BACKFILL REVIEW — status: {meta['status'].upper()} "
          f"(scope_complete={meta['scope_complete']})")
    print("=" * 74)
    for key in ("full_day_book_fills", "partial_day_book_fills", "trade_fills",
                "lake_usable_days", "lake_present_degraded_days", "excluded_days",
                "unresolved_days"):
        print(f"  {key:<28} {len(s[key]):>4}")
    print(f"  cost: gross ${c['gross_usd']:.2f} (net ${c['net_usd']:.2f} after "
          f"${c['credit_usd']:.0f} credit); band ${c['band']['low_usd']:.2f}"
          f"–${c['band']['high_usd']:.2f}; calendar-gap baseline "
          f"${c['calendar_gap_baseline_usd']:.2f}")
    if meta["status"] == "blocking":
        print("  BLOCKERS:")
        for k in BLOCKER_KEYS:
            for item in manifest["blockers"][k][:8]:
                print(f"    - {k}: {item}")
    print("  NOTE: review only — does NOT unlock or run the backfill (§5a gate stays enforced "
          "in ingest/download_coinapi.py). A multi-day pull needs --allow-backfill + CoinAPI "
          "Spend Management (docs/data.md §8).")


def parse_args(argv=None):
    ap = argparse.ArgumentParser(
        description="Review completed Coinbase quality-map reports into a human-auditable CoinAPI "
                    "backfill manifest (gatekeeping only — no vendor I/O, does NOT unlock the "
                    "§5a backfill gate).")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--plan-manifest", default=None,
                      help="READINESS mode: batch-plan manifest.json (authoritative batch registry)")
    mode.add_argument("--report", nargs="+", default=None,
                      help="INSPECTION mode: one or more quality-map report JSONs (status=report_only)")
    ap.add_argument("--usable-calendar", default=None,
                    help="usable calendar JSON (default: plan meta.input_calendar in readiness mode)")
    ap.add_argument("--out", default=DEFAULT_OUT, help=f"manifest output path (default {DEFAULT_OUT})")
    ap.add_argument("--report-only", action="store_true",
                    help="readiness mode: downgrade a blocking verdict to exit 0 (keeps honest "
                         "status + blockers)")
    ap.add_argument("--generated-utc", default=None,
                    help="override the manifest timestamp (for deterministic tests)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    generated = args.generated_utc or dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    try:
        if args.plan_manifest:
            manifest = build_manifest_readiness(args.plan_manifest, args.usable_calendar,
                                                generated_utc=generated,
                                                report_only=args.report_only)
        else:
            manifest = build_manifest_inspection(args.report, args.usable_calendar,
                                                 generated_utc=generated)
    except ReviewInputError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return INPUT_ERROR_EXIT
    write_manifest(manifest, args.out)
    print_summary(manifest)
    print(f"\n  wrote {args.out}")
    if manifest["meta"]["status"] == "blocking" and not args.report_only:
        return BLOCKING_EXIT
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
