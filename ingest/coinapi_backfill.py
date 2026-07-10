"""Planning core for the reviewed-manifest CoinAPI backfill executor (issue #53).

Turns the canonical reviewed backfill manifest (scripts/review_coinbase_backfill_manifest.py
output, docs/data.md §5a-QualityMap) into a deterministic, fail-closed download plan of EXACT
sparse fill units — never a contiguous date range — plus the resume-state keying and the
reconciled execution report the downloader (ingest/download_coinapi.py --manifest) consumes.

Stdlib-only on purpose (mirrors ingest/_common.py): a CI-safe offline test drives every
acceptance/refusal path without pandas/boto3/pyarrow, and NO code in this module performs vendor
I/O — all S3 access stays in download_coinapi.py behind its --execute + spend-authorization +
§5a --allow-backfill gates.

Fail-closed acceptance (BackfillRefusal, exit 3 at the CLI): the manifest must be
status=ready + scope_complete with every blocker list empty; match an operator-pinned sha256 when
one is given (the spend-approval pin); have every pinned input (meta.inputs sha256s) still intact
on disk (else it is STALE — the quality map / calendar were regenerated since review); and its
days[]/sections/cost_summary must mutually reconcile (a tampered or drifted manifest never plans
a download). Structural problems reading a file are BackfillInputError (exit 2) instead.

Resume state is keyed on (source, product, day, manifest fingerprint): an output produced under a
different manifest — or with no state record at all — is a CONFLICT, never silently counted done.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os

SOURCE = "coinapi_flatfiles"
PRODUCT_BOOK = "limitbook_full"
PRODUCT_TRADES = "trades"
PRODUCTS = (PRODUCT_BOOK, PRODUCT_TRADES)

PLAN_VERSION = 1
REPORT_VERSION = 1
MANIFEST_VERSION = 1                       # reviewed-manifest schema this executor understands
MANIFEST_KIND = "coinbase_backfill_review"
EXPECTED_EXCHANGE = "COINBASE"
EXPECTED_SYMBOL = "BTC-USD"
BOOK_KINDS = ("full_day", "partial")
GB_BASES = ("measured", "estimated")
# Pinned copy of recon.stitch_policy.SOURCES (via the reviewer's SEGMENT_SOURCES; contract test).
SEGMENT_SOURCES = ("lake", "coinapi", "excluded")
# Pinned key set of recon.stitch_policy.DEFAULT_SEAM_POLICY (via the reviewer; contract test).
SEAM_POLICY_KEYS = ("seam_guard_s", "warmup_consecutive", "fill_min_s", "min_lake_segment_s",
                    "span_invalid_max", "exclude_labels_crossing_seam",
                    "exclude_features_crossing_seam")
NS_PER_S = 1_000_000_000
DAY_NS = 86_400 * NS_PER_S
SECTION_KEYS = ("full_day_book_fills", "partial_day_book_fills", "trade_fills",
                "lake_usable_days", "lake_present_degraded_days", "excluded_days",
                "unresolved_days")
# Pinned copy of the reviewer's BLOCKER_KEYS (scripts/review_coinbase_backfill_manifest.py) —
# kept aligned by a contract test. The acceptance gate requires this EXACT key set.
BLOCKER_KEYS = ("structural", "missing_keys", "coverage_gaps", "inconsistencies",
                "resolution_issues", "unresolved_days", "batch_incomplete",
                "book_fill_unavailable", "trade_fill_unavailable", "calendar_drift")

INPUT_ERROR_EXIT = 2                       # structural/input error — matches the reviewer
REFUSAL_EXIT = 3                           # fail-closed refusal verdict — matches the reviewer
STATE_DIRNAME = "_backfill_state"

# float reconciliation bound: manifest figures are round(x, 4); recomputing the same sums in the
# same order reproduces them exactly, so anything beyond double-rounding noise is real drift.
_RECON_TOL = 1e-9


class BackfillInputError(ValueError):
    """A user-actionable structural input failure (missing/unreadable/invalid file) — exit 2."""


class BackfillRefusal(ValueError):
    """A fail-closed refusal of a parsed manifest / window / spend request — exit 3."""


# ----------------------------------------------------------- input helpers
def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_reviewed_manifest(path: str) -> tuple:
    """Load the reviewed manifest as a strict JSON object. Returns (manifest, sha256hex) where the
    hash is computed over the SAME bytes that were parsed (single read) — so the operator pin, the
    plan's recorded fingerprint, and the resume-state keying all attest exactly what was executed,
    with no parse-then-rehash window. Missing/invalid/non-object input is a structural
    BackfillInputError (exit 2); NaN/Infinity and out-of-range integer literals are rejected at
    the boundary — a non-finite GB/cost figure must never reach the spend math."""
    if not os.path.exists(path):
        raise BackfillInputError(f"reviewed manifest not found: {path}")

    def _reject(token):
        raise BackfillInputError(f"reviewed manifest {path} contains a non-finite JSON constant "
                                 f"({token})")

    def _finite(s):
        v = float(s)
        if not math.isfinite(v):
            raise BackfillInputError(f"reviewed manifest {path} contains a non-finite number ({s})")
        return v

    def _bounded_int(s):
        # a huge integer literal either exceeds the CPython str->int digit limit (ValueError) or
        # overflows when later used as a float (OverflowError) — both become a clean input error.
        try:
            v = int(s)
            float(v)
        except (ValueError, OverflowError):
            raise BackfillInputError(f"reviewed manifest {path} contains an out-of-range or "
                                     f"unparseable integer ({s[:32]})") from None
        return v
    with open(path, "rb") as f:
        data = f.read()
    try:
        obj = json.loads(data, parse_constant=_reject, parse_float=_finite,
                         parse_int=_bounded_int)
    except json.JSONDecodeError as e:
        raise BackfillInputError(f"reviewed manifest {path} is not valid JSON: {e}") from None
    if not isinstance(obj, dict):
        raise BackfillInputError(f"reviewed manifest {path} must be a JSON object")
    return obj, hashlib.sha256(data).hexdigest()


def _as_dict(v):
    return v if isinstance(v, dict) else {}


def _is_num(v) -> bool:
    if not isinstance(v, (int, float)) or isinstance(v, bool):
        return False
    try:   # defense in depth: an int too large for float() must read as "not a usable number"
        return math.isfinite(float(v))
    except OverflowError:
        return False


def write_json_atomic(obj: dict, path: str) -> str:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.write("\n")
    os.replace(tmp, path)
    return path


# ----------------------------------------------------------- acceptance gate
def _acceptance_issues(manifest: dict) -> list:
    """Everything that disqualifies the manifest BEFORE any unit is derived. The gate accepts only
    a ready, scope-complete review manifest for the market this executor serves."""
    issues = []
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        issues.append(f"manifest_version:{manifest.get('manifest_version')!r} "
                      f"(expected {MANIFEST_VERSION})")
    meta = manifest.get("meta")
    if not isinstance(meta, dict):
        issues.append("meta:missing_or_not_object")
        meta = {}
    if meta.get("kind") != MANIFEST_KIND:
        issues.append(f"kind:{meta.get('kind')!r} (expected {MANIFEST_KIND!r})")
    if meta.get("status") != "ready":
        issues.append(f"status:{meta.get('status')!r} (only a ready manifest is executable)")
    if meta.get("scope_complete") is not True:
        issues.append(f"scope_complete:{meta.get('scope_complete')!r}")
    if meta.get("exchange") != EXPECTED_EXCHANGE or meta.get("symbol") != EXPECTED_SYMBOL:
        issues.append(f"exchange/symbol:{meta.get('exchange')!r}/{meta.get('symbol')!r} "
                      f"(expected {EXPECTED_EXCHANGE}/{EXPECTED_SYMBOL})")
    cm = meta.get("cost_model")
    # rates must be positive: a zero/negative rate would price the whole backfill at <= $0 and
    # hollow out the --approve-usd spend cap even though every reconciliation still balances
    if not (isinstance(cm, dict) and _is_num(cm.get("book_usd_per_gb"))
            and _is_num(cm.get("trades_usd_per_gb"))
            and float(cm.get("book_usd_per_gb")) > 0
            and float(cm.get("trades_usd_per_gb")) > 0):
        issues.append("cost_model:missing_malformed_or_nonpositive_rates")
    blockers = manifest.get("blockers")
    if not isinstance(blockers, dict) or not blockers:
        issues.append("blockers:missing_or_not_object")
    else:
        # exact key set: coverage_gaps/batch_incomplete/calendar_drift &c. leave no days[]-level
        # trace, so a deleted blocker key would silently unblock — refuse anything but the
        # reviewer's full v1 key set (pinned as BLOCKER_KEYS, contract-tested below).
        if set(blockers) != set(BLOCKER_KEYS):
            issues.append(f"blockers:key_set_mismatch:missing={sorted(set(BLOCKER_KEYS) - set(blockers))}:"
                          f"extra={sorted(set(blockers) - set(BLOCKER_KEYS))}")
        for k, v in blockers.items():
            if not isinstance(v, list):
                issues.append(f"blockers:{k}:not_a_list")
            elif v:
                issues.append(f"blockers:{k}:non-empty ({len(v)} entries)")
    days = manifest.get("days")
    if not (isinstance(days, list) and days and all(isinstance(r, dict) for r in days)):
        issues.append("days:missing_empty_or_malformed")
    sections = manifest.get("sections")
    if not isinstance(sections, dict):
        issues.append("sections:missing_or_not_object")
    else:
        for k in SECTION_KEYS:
            if not isinstance(sections.get(k), list):
                issues.append(f"sections:missing_or_malformed_key:{k}")
    if not isinstance(manifest.get("cost_summary"), dict):
        issues.append("cost_summary:missing_or_not_object")
    return issues


def verify_pinned_inputs(manifest: dict) -> list:
    """Re-hash every input the review pinned in meta.inputs. A missing file or a sha256 mismatch
    means the manifest is STALE — its quality map / calendar / plan / resolutions were regenerated
    (or removed) after the review, so the reviewed fill scope can no longer be trusted."""
    inputs = _as_dict(manifest.get("meta")).get("inputs")
    if not isinstance(inputs, dict):
        return ["stale_inputs:meta.inputs:missing_or_not_object"]
    pinned = []

    def _pin(what, entry):
        e = _as_dict(entry)
        pinned.append((what, e.get("path"), e.get("sha256")))

    _pin("plan_manifest", inputs.get("plan_manifest"))
    _pin("usable_calendar", inputs.get("usable_calendar"))
    reports = inputs.get("batch_reports")
    if not isinstance(reports, list) or not reports:
        return ["stale_inputs:batch_reports:missing_or_empty"]
    for i, r in enumerate(reports):
        r = _as_dict(r)
        _pin(f"batch_reports[{i}]", r)
        if r.get("batch_days_file") is not None:
            _pin(f"batch_reports[{i}].batch_days_file", r.get("batch_days_file"))
    res = inputs.get("resolutions")
    if res is not None:
        _pin("resolutions", res)
        for i, rr in enumerate(_as_dict(res).get("rerun_reports") or []):
            _pin(f"resolutions.rerun_reports[{i}]", rr)

    issues = []
    for what, path, sha in pinned:
        if not isinstance(path, str) or not isinstance(sha, str):
            issues.append(f"stale_inputs:unpinned:{what}")
        elif not os.path.exists(path):
            issues.append(f"stale_inputs:missing_file:{what}:{path}")
        elif sha256_file(path) != sha:
            issues.append(f"stale_inputs:sha256_mismatch:{what}:{path}")
    return issues


# ----------------------------------------------------------- unit derivation
def _day_bounds_ns(day: str) -> tuple:
    """(day_open_ts, day_end_ts) in int ns for a YYYY-MM-DD partition day (midnight UTC)."""
    d = dt.date.fromisoformat(day)
    open_ = int(dt.datetime(d.year, d.month, d.day, tzinfo=dt.timezone.utc).timestamp()) * NS_PER_S
    return open_, open_ + DAY_NS


def _fill_segments_issues(day: str, segments: list, seams) -> list:
    """Ported from the reviewer's stitch-plan validation (contract-pinned enums): segments must
    PARTITION [day_open, day_close) as ordered, contiguous, half-open int-ns spans with known
    sources, and `seams` must be exactly the source-change boundaries. The executor re-validates
    because it is the SPEND gate: sections/cost_summary do not cover these fields, so a
    pinned-but-tampered/stale stitch plan would otherwise be executed and carried verbatim into
    downstream seam masking."""
    issues = []
    try:
        day_open, day_end = _day_bounds_ns(day)
    except (TypeError, ValueError):
        day_open = day_end = None
    prev_end = None
    for i, s in enumerate(segments):
        if not isinstance(s, dict):
            issues.append(f"fill_segment[{i}]_not_object:{day}")
            continue
        st, en, src = s.get("start_ts"), s.get("end_ts"), s.get("source")
        if (not isinstance(st, int) or not isinstance(en, int)
                or isinstance(st, bool) or isinstance(en, bool)):
            issues.append(f"fill_segment[{i}]_non_int_bounds:{day}")
            continue
        if en <= st:
            issues.append(f"fill_segment[{i}]_non_positive_span:{day}")
        if src not in SEGMENT_SOURCES:
            issues.append(f"fill_segment[{i}]_bad_source:{day}:{src!r}")
        if prev_end is not None:
            if st != prev_end:
                issues.append(f"fill_segments_gap_or_overlap_at[{i}]:{day}")
        elif day_open is not None and st != day_open:
            issues.append(f"fill_segments_start_ne_day_open:{day}")
        prev_end = en
    if day_end is not None and prev_end is not None and prev_end != day_end:
        issues.append(f"fill_segments_end_ne_day_close:{day}")
    # seams must be exactly the source-change boundaries (right segment's start_ts); a stale or
    # emptied seams list would drop the guard bands a consumer applies at a Lake<->CoinAPI switch
    expected_seams = [cur.get("start_ts") for prev, cur in zip(segments, segments[1:])
                      if isinstance(prev, dict) and isinstance(cur, dict)
                      and prev.get("source") != cur.get("source")]
    if isinstance(seams, list) and seams != expected_seams:
        issues.append(f"seams_mismatch:{day}:expected={expected_seams}:got={seams}")
    return issues


def _seam_policy_issues(day: str, policy: dict) -> list:
    """The seam policy rides verbatim into downstream stitch masking, so the spend gate refuses
    unsafe VALUES, not just non-dicts: a zeroed guard band, a non-positive warmup/fill floor, or
    a disabled exclude_*_crossing_seam switch would let labels/features cross a vendor seam
    (leakage) despite the segment re-validation."""
    if set(policy) != set(SEAM_POLICY_KEYS):
        return [f"seam_policy_key_mismatch:{day}:"
                f"missing={sorted(set(SEAM_POLICY_KEYS) - set(policy))}:"
                f"extra={sorted(set(policy) - set(SEAM_POLICY_KEYS))}"]
    issues = []
    for k in ("seam_guard_s", "fill_min_s", "min_lake_segment_s"):
        if not _is_num(policy[k]) or float(policy[k]) <= 0:
            issues.append(f"seam_policy_bad_value:{day}:{k}:{policy[k]!r}")
    if not _is_num(policy["warmup_consecutive"]) or float(policy["warmup_consecutive"]) < 1:
        issues.append(f"seam_policy_bad_value:{day}:warmup_consecutive:"
                      f"{policy['warmup_consecutive']!r}")
    if (not _is_num(policy["span_invalid_max"])
            or not 0 <= float(policy["span_invalid_max"]) < 1):
        issues.append(f"seam_policy_bad_value:{day}:span_invalid_max:"
                      f"{policy['span_invalid_max']!r}")
    for k in ("exclude_labels_crossing_seam", "exclude_features_crossing_seam"):
        if policy[k] is not True:   # the leakage guards must be hard-on, never relaxed here
            issues.append(f"seam_policy_bad_value:{day}:{k}:{policy[k]!r}")
    return issues


def _fill_unit_issues(day: str, fill: dict, product: str) -> list:
    """A fill unit must be executable and costable: a valid kind, a verbatim stitch plan that
    actually partitions the day and pulls from CoinAPI (book), and finite non-negative GB/$
    figures."""
    issues = []
    kind = fill.get("kind") if product == PRODUCT_BOOK else "full_day"
    if product == PRODUCT_BOOK:
        if kind not in BOOK_KINDS:
            issues.append(f"book_fill_bad_kind:{day}:{kind!r}")
        segs = fill.get("fill_segments")
        if not isinstance(segs, list) or not segs:
            issues.append(f"book_fill_missing_fill_segments:{day}")
        else:
            issues.extend(_fill_segments_issues(day, segs, fill.get("seams")))
            srcs = [s.get("source") for s in segs if isinstance(s, dict)]
            if "coinapi" not in srcs:
                issues.append(f"book_fill_no_coinapi_segment:{day}")
            elif kind == "full_day" and any(src != "coinapi" for src in srcs):
                # a full-day fill is an all-CoinAPI replacement; a lake segment inside it would
                # preserve data the fill decision says cannot be trusted (or does not exist)
                issues.append(f"full_day_fill_non_coinapi_segment:{day}")
        if not isinstance(fill.get("seams"), list):
            issues.append(f"book_fill_missing_seams:{day}")
        if not isinstance(fill.get("seam_policy"), dict):
            issues.append(f"book_fill_missing_seam_policy:{day}")
        else:
            issues.extend(_seam_policy_issues(day, fill["seam_policy"]))
    if not _is_num(fill.get("gb")) or float(fill.get("gb")) < 0:
        issues.append(f"fill_bad_gb:{product}:{day}:{fill.get('gb')!r}")
    if fill.get("gb_basis") not in GB_BASES:
        issues.append(f"fill_bad_gb_basis:{product}:{day}:{fill.get('gb_basis')!r}")
    if not _is_num(fill.get("usd")) or float(fill.get("usd")) < 0:
        issues.append(f"fill_bad_usd:{product}:{day}:{fill.get('usd')!r}")
    return issues


def derive_units(manifest: dict) -> tuple:
    """(units, issues): the exact sparse fill units the manifest authorizes, in deterministic
    (day, product) order. Only days[] records with an affirmative fill produce units — an excluded
    or unresolved day, or any day between fills, never does."""
    units, issues, seen = [], [], set()
    for rec in manifest.get("days") or []:
        day = rec.get("day")
        try:
            # canonical YYYY-MM-DD only: fromisoformat also accepts compact forms (20250102) on
            # 3.11+, which would defeat duplicate detection and produce non-canonical partitions
            if dt.date.fromisoformat(day).isoformat() != day:
                raise ValueError(day)
        except (TypeError, ValueError):
            issues.append(f"bad_day:{day!r} (canonical YYYY-MM-DD required)")
            continue
        if day in seen:
            issues.append(f"duplicate_day_record:{day}")
            continue
        seen.add(day)
        book, trade = _as_dict(rec.get("book_fill")), _as_dict(rec.get("trade_fill"))
        excluded, unresolved = rec.get("excluded"), rec.get("unresolved")
        if unresolved is not None:
            issues.append(f"unresolved_day:{day}")
            continue
        if excluded is not None:
            if book.get("needed") is True or trade.get("needed") is True:
                issues.append(f"excluded_day_with_fill:{day}")
            continue                       # exclusion wins: an excluded day is never a unit
        prov_base = {"classification": rec.get("classification"),
                     "sources": rec.get("sources"), "resolution": rec.get("resolution")}
        if book.get("needed") is True:
            issues.extend(_fill_unit_issues(day, book, PRODUCT_BOOK))
            units.append({"source": SOURCE, "product": PRODUCT_BOOK, "day": day,
                          "kind": book.get("kind"), "gb": book.get("gb"),
                          "gb_basis": book.get("gb_basis"), "usd": book.get("usd"),
                          "provenance": {**prov_base, "fill": book}})
        if trade.get("needed") is True:
            issues.extend(_fill_unit_issues(day, trade, PRODUCT_TRADES))
            units.append({"source": SOURCE, "product": PRODUCT_TRADES, "day": day,
                          "kind": "full_day", "gb": trade.get("gb"),
                          "gb_basis": trade.get("gb_basis"), "usd": trade.get("usd"),
                          "provenance": {**prov_base, "fill": trade}})
    units.sort(key=lambda u: (u["day"], u["product"]))
    return units, issues


def _sections_issues(manifest: dict, units: list) -> list:
    """Re-derive every section from days[] + units and require exact agreement with the manifest's
    sections views — a tampered/drifted manifest whose sections disagree with its records must
    fail closed, not silently follow one of the two."""
    sections = _as_dict(manifest.get("sections"))
    derived = {k: [] for k in SECTION_KEYS}
    for u in units:
        if u["product"] == PRODUCT_BOOK:
            key = "full_day_book_fills" if u["kind"] == "full_day" else "partial_day_book_fills"
            derived[key].append(u["day"])
        else:
            derived["trade_fills"].append(u["day"])
    for rec in manifest.get("days") or []:
        d = rec.get("day")
        if rec.get("classification") == "lake_usable":
            derived["lake_usable_days"].append(d)
        if rec.get("classification") == "lake_present_degraded":
            derived["lake_present_degraded_days"].append(d)
        if rec.get("excluded") is not None:
            derived["excluded_days"].append(d)
        if rec.get("unresolved") is not None:
            derived["unresolved_days"].append(d)
    issues = []
    for k in SECTION_KEYS:
        if sections.get(k) != sorted(derived[k]):
            issues.append(f"sections_mismatch:{k}:manifest={sections.get(k)!r}:"
                          f"derived={sorted(derived[k])!r}")
    return issues


# ----------------------------------------------------------- totals + reconciliation
def unit_totals(units: list, cost_model: dict) -> dict:
    """Aggregate unit GB/$ with the SAME math as the reviewer's cost summary (rates from the
    manifest's own cost_model), so the full-scope totals reconcile bit-for-bit."""
    book_rate = float(cost_model["book_usd_per_gb"])
    trades_rate = float(cost_model["trades_usd_per_gb"])
    book_m = book_e = tr_m = tr_e = 0.0
    full_n = partial_n = trade_n = 0
    for u in units:
        gb = float(u["gb"])
        measured = u["gb_basis"] == "measured"
        if u["product"] == PRODUCT_BOOK:
            full_n += u["kind"] == "full_day"
            partial_n += u["kind"] == "partial"
            book_m, book_e = (book_m + gb, book_e) if measured else (book_m, book_e + gb)
        else:
            trade_n += 1
            tr_m, tr_e = (tr_m + gb, tr_e) if measured else (tr_m, tr_e + gb)
    book_usd = round((book_m + book_e) * book_rate, 4)
    trades_usd = round((tr_m + tr_e) * trades_rate, 4)
    gross = round(book_usd + trades_usd, 4)
    low = round(round(book_m * book_rate, 4) + round(tr_m * trades_rate, 4), 4)
    return {"n_units": len(units), "book_units": full_n + partial_n,
            "book_full_days": full_n, "book_partial_days": partial_n, "trade_units": trade_n,
            "book_gb_measured": round(book_m, 4), "book_gb_estimated": round(book_e, 4),
            "book_gb_total": round(book_m + book_e, 4),
            "trades_gb_measured": round(tr_m, 4), "trades_gb_estimated": round(tr_e, 4),
            "trades_gb_total": round(tr_m + tr_e, 4),
            "book_usd": book_usd, "trades_usd": trades_usd, "gross_usd": gross,
            "usd_low": low, "usd_high": gross}


_COST_SUMMARY_FIELDS = (("book_fill_days", "book_units"),
                        ("full_book_fill_days", "book_full_days"),
                        ("partial_book_fill_days", "book_partial_days"),
                        ("trade_fill_days", "trade_units"),
                        ("book_gb_measured",) * 2, ("book_gb_estimated",) * 2,
                        ("book_gb_total",) * 2, ("trades_gb_measured",) * 2,
                        ("trades_gb_estimated",) * 2, ("trades_gb_total",) * 2,
                        ("book_usd",) * 2, ("trades_usd",) * 2, ("gross_usd",) * 2)


def _cost_reconciliation_issues(manifest: dict, full_totals: dict) -> list:
    """The units re-derived from days[] must price to EXACTLY the reviewed cost_summary the human
    approved — any drift (tampering, partial edit, contract change) refuses before spend."""
    cs = _as_dict(manifest.get("cost_summary"))
    issues = []
    for cs_key, t_key in _COST_SUMMARY_FIELDS:
        got, want = cs.get(cs_key), full_totals.get(t_key)
        if not _is_num(got) or abs(float(got) - float(want)) > _RECON_TOL:
            issues.append(f"cost_reconciliation_failed:{cs_key}:manifest={got!r}:derived={want!r}")
    band = _as_dict(cs.get("band"))
    for band_key, t_key in (("high_usd", "usd_high"), ("low_usd", "usd_low")):
        got = band.get(band_key)
        if not _is_num(got) or abs(float(got) - float(full_totals[t_key])) > _RECON_TOL:
            issues.append(f"cost_reconciliation_failed:band.{band_key}:manifest={got!r}:"
                          f"derived={full_totals[t_key]!r}")
    return issues


# ----------------------------------------------------------- pilot window
def select_window(units: list, start, end) -> list:
    """Deterministic pilot-window subset: units whose day falls in [start, end], selected from the
    already-derived units — the fill policy and per-unit cost rows are never recomputed."""
    if (start is None) != (end is None):
        raise BackfillInputError("pilot window needs BOTH --pilot-start and --pilot-end")
    if start is None:
        return list(units)
    try:
        d0, d1 = dt.date.fromisoformat(start), dt.date.fromisoformat(end)
    except (TypeError, ValueError):
        raise BackfillInputError(f"pilot window bounds must be YYYY-MM-DD, got "
                                 f"{start!r}..{end!r}") from None
    if d1 < d0:
        raise BackfillInputError(f"pilot window end {end} is before start {start}")
    sel = [u for u in units if d0 <= dt.date.fromisoformat(u["day"]) <= d1]
    if not sel:
        raise BackfillInputError(f"pilot window {start}..{end} selects zero fill units "
                                 "(nothing to do — check the window against the manifest)")
    return sel


# ----------------------------------------------------------- plan assembly
def build_plan(manifest_path: str, *, generated_utc: str, expected_sha256: str | None = None,
               window_start: str | None = None, window_end: str | None = None,
               mode: str = "dry_run", verify_inputs: bool = True) -> dict:
    """Load + fail-closed-validate the reviewed manifest and emit the deterministic download plan.
    Raises BackfillInputError (exit 2) for structural problems and BackfillRefusal (exit 3) for
    any acceptance/staleness/reconciliation failure. The emitted plan always reconciles."""
    manifest, actual_sha = load_reviewed_manifest(manifest_path)
    if expected_sha256 is not None and expected_sha256 != actual_sha:
        raise BackfillRefusal(f"manifest sha256 mismatch: pinned {expected_sha256} != actual "
                              f"{actual_sha} — refusing (approve the exact manifest by hash)")
    issues = _acceptance_issues(manifest)
    if issues:
        raise BackfillRefusal(f"refusing reviewed manifest {manifest_path}: " + "; ".join(issues))
    if verify_inputs:
        stale = verify_pinned_inputs(manifest)
        if stale:
            raise BackfillRefusal(f"refusing reviewed manifest {manifest_path}: "
                                  + "; ".join(stale))
    # refuse malformed units BEFORE any section/cost math touches them: a non-numeric or
    # non-canonical field must surface as a clean refusal, never a ValueError/TypeError from
    # sorting or the cost aggregation
    units, issues = derive_units(manifest)
    if issues:
        raise BackfillRefusal(f"refusing reviewed manifest {manifest_path}: " + "; ".join(issues))
    issues += _sections_issues(manifest, units)
    meta = manifest["meta"]
    full_totals = unit_totals(units, meta["cost_model"])
    issues += _cost_reconciliation_issues(manifest, full_totals)
    if issues:
        raise BackfillRefusal(f"refusing reviewed manifest {manifest_path}: " + "; ".join(issues))
    selected = select_window(units, window_start, window_end)
    window = None if window_start is None else {"start": window_start, "end": window_end}
    return {
        "plan_version": PLAN_VERSION,
        "meta": {"kind": "coinapi_backfill_plan", "tool": "ingest/download_coinapi.py",
                 "generated_utc": generated_utc, "mode": mode, "source": SOURCE,
                 "exchange": meta.get("exchange"), "symbol": meta.get("symbol"),
                 "manifest": {"path": manifest_path, "sha256": actual_sha,
                              "generated_utc": meta.get("generated_utc"),
                              "status": meta.get("status"),
                              "scope_complete": meta.get("scope_complete")},
                 "window": window,
                 "cost_model": meta["cost_model"],
                 "input_verification": {"verified": bool(verify_inputs)}},
        "scope": {"manifest_days": len(manifest["days"]),
                  "excluded_days": manifest["sections"]["excluded_days"],
                  "unresolved_days": manifest["sections"]["unresolved_days"],
                  "full_totals": full_totals,
                  "skipped_by_window": len(units) - len(selected)},
        "units": selected,
        "totals": unit_totals(selected, meta["cost_model"]),
        # derived, not asserted: `issues` is provably empty here (any entry raised above), so an
        # emitted plan always reconciles — the field documents that the check ran
        "reconciliation": {"matches_manifest_cost_summary": not issues, "issues": list(issues)},
    }


# ----------------------------------------------------------- resume state
def state_path(out_root: str, manifest_sha256: str, product: str, day: str) -> str:
    """Resume-state record path: keyed on source + manifest fingerprint + product + day, so an
    output downloaded under a different (or no) manifest can never satisfy this run's resume."""
    return os.path.join(out_root, STATE_DIRNAME, SOURCE, manifest_sha256, product, f"{day}.json")


def write_state(out_root: str, manifest_sha256: str, unit: dict, record: dict) -> str:
    path = state_path(out_root, manifest_sha256, unit["product"], unit["day"])
    rec = {"source": SOURCE, "product": unit["product"], "day": unit["day"],
           "manifest_sha256": manifest_sha256, **record}
    return write_json_atomic(rec, path)


def load_state(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            obj = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return obj if isinstance(obj, dict) else None


def unit_resume_status(out_root: str, manifest_sha256: str, unit: dict, final_path: str,
                       overwrite: bool = False) -> str:
    """'done' | 'todo' | 'conflict' for one unit.
    done      — final output exists AND this manifest's state record vouches for those exact bytes
                (size and sha256 both re-verified: same-size drift/bit-rot must not count done).
    todo      — no final output (a dangling state record is stale and does not count).
    conflict  — final output exists without a state record for THIS manifest fingerprint that
                matches the actual bytes (foreign/stale/drifted output): fail closed, never adopt
                or overwrite it.
    --overwrite is the explicit operator override: everything is re-downloaded."""
    if overwrite:
        return "todo"
    if not os.path.exists(final_path):
        return "todo"
    st = load_state(state_path(out_root, manifest_sha256, unit["product"], unit["day"]))
    if (st is not None and st.get("status") == "ok"
            and st.get("out_bytes") == os.path.getsize(final_path)
            and isinstance(st.get("out_sha256"), str)
            and st.get("out_sha256") == sha256_file(final_path)):
        return "done"
    return "conflict"


# ----------------------------------------------------------- execution report
_RESULT_STATUSES = ("ok", "done", "missing", "conflict", "error", "refused_budget")


def build_execution_report(plan: dict, results: list, *, spend: dict, generated_utc: str) -> dict:
    """Reconcile per-unit results against the plan BY IDENTITY, not just by count: every planned
    (product, day) unit must appear exactly once, nothing unplanned may appear at all, and each
    reported unit carries its manifest provenance (verbatim stitch metadata) plus bytes/rows/
    hashes and the spend evidence — the run is auditable from the report alone. `complete` is
    strict: planned == ok + done_prior, nothing missing/conflicted/errored/budget-refused,
    nothing unaccounted, duplicated, or unplanned."""
    counts = {s: 0 for s in _RESULT_STATUSES}
    bytes_dl = rows = 0
    prod_bytes = {PRODUCT_BOOK: 0, PRODUCT_TRADES: 0}
    seen: dict = {}
    for r in results:
        s = r.get("status")
        counts[s] = counts.get(s, 0) + 1
        key = (r.get("product"), r.get("day"))
        seen[key] = seen.get(key, 0) + 1
        if s == "ok":
            rows += int(r.get("rows") or 0)
        # spend accounting uses BILLED bytes: a unit whose GET ran but failed afterwards
        # (header drift, parse error) was still delivered — and charged — by the vendor
        billed = int(r.get("billed_bytes") or 0)
        if not billed and s == "ok":
            billed = int(r.get("src_bytes") or 0)
        if billed:
            bytes_dl += billed
            prod_bytes[r.get("product")] = prod_bytes.get(r.get("product"), 0) + billed
    provenance = {(u["product"], u["day"]): u["provenance"] for u in plan["units"]}
    planned_keys = set(provenance)
    unaccounted = sorted(f"{p}/{d}" for p, d in planned_keys - set(seen))
    unplanned = sorted(f"{p}/{d}" for p, d in set(seen) - planned_keys)
    duplicates = sorted(f"{p}/{d}" for (p, d), n in seen.items() if n > 1)
    planned = len(plan["units"])
    accounted = len(results)
    complete = (accounted == planned
                and counts["ok"] + counts["done"] == planned
                and not (counts["missing"] or counts["conflict"] or counts["error"]
                         or counts["refused_budget"])
                and not (unaccounted or unplanned or duplicates))
    cm = plan["meta"]["cost_model"]
    measured_gb = round(bytes_dl / 1e9, 4)
    measured_usd = round(prod_bytes[PRODUCT_BOOK] / 1e9 * float(cm["book_usd_per_gb"])
                         + prod_bytes[PRODUCT_TRADES] / 1e9 * float(cm["trades_usd_per_gb"]), 4)
    totals = plan["totals"]
    return {
        "report_version": REPORT_VERSION,
        "meta": {"kind": "coinapi_backfill_execution_report",
                 "tool": "ingest/download_coinapi.py", "generated_utc": generated_utc,
                 "source": SOURCE, "manifest": plan["meta"]["manifest"],
                 "window": plan["meta"]["window"]},
        "spend": {"approve_usd": spend.get("approve_usd"),
                  "spend_evidence": spend.get("spend_evidence"),
                  "allow_backfill": spend.get("allow_backfill"),
                  "overwrite": spend.get("overwrite"),
                  "planned_usd_high": totals["usd_high"],
                  "planned_usd_low": totals["usd_low"],
                  "measured_gb_downloaded": measured_gb,
                  "measured_usd_at_model_rates": measured_usd},
        "units": [{**r, "provenance": provenance.get((r.get("product"), r.get("day")))}
                  for r in results],
        "reconciliation": {"planned": planned, "accounted": accounted,
                           "ok": counts["ok"], "done_prior": counts["done"],
                           "missing": counts["missing"], "conflict": counts["conflict"],
                           "error": counts["error"],
                           "refused_budget": counts["refused_budget"],
                           "unaccounted_units": unaccounted,
                           "unplanned_results": unplanned,
                           "duplicate_results": duplicates,
                           "complete": complete,
                           "bytes_downloaded": bytes_dl, "rows_written": rows,
                           "planned_gb_high": round(totals["book_gb_total"]
                                                    + totals["trades_gb_total"], 4),
                           "measured_gb": measured_gb},
    }
