# Coinbase Backfill Review Manifest — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build `scripts/review_coinbase_backfill_manifest.py` — a gatekeeping tool that joins the batch-plan manifest, the per-batch quality-map reports, and the usable calendar into a deterministic, human-auditable CoinAPI backfill manifest, failing closed on incomplete/inconsistent inputs. It does **not** unlock or run the backfill.

**Architecture:** A single stdlib-only script (mirrors `scripts/plan_coinbase_quality_map_batches.py` — no pandas/numpy/boto3, so offline tests are cheap). Pure helpers (calendar accessors, cost model, validation) → per-day canonical record builder → plan-driven completeness/consistency checks → manifest assembly → CLI with two mutually-exclusive modes (readiness / inspection). Quality-map enum strings are pinned copies here (importing the runner would pull pandas); a contract test keeps them aligned with `run_coinbase_quality_map.py` and `recon/stitch_policy.py`.

**Tech Stack:** Python 3, stdlib only (`argparse`, `json`, `hashlib`, `pathlib`, `datetime`, `os`, `sys`). pytest for tests (loads the script by path, `tmp_path` fixtures, synthetic JSON — no vendor I/O).

**Design reference:** `docs/superpowers/specs/2026-07-03-coinbase-backfill-manifest-review-design.md`

---

## Shared definitions (used across all tasks — keep names consistent)

**Enum / cost / control constants** (defined in Task 1):

```python
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

BOOK_USD_PER_GB = 1.0
TRADES_USD_PER_GB = 3.0
EST_BOOK_GB_PER_DAY = 2.27          # §2.2/§6 nominal L3, conservative
EST_TRADES_GB_PER_DAY = 0.05        # §8 2.6 GB / 52 days
CREDIT_USD = 25.0                   # §8 flat-files trial pool
DOCS_REFERENCE_USD = 92.0           # §8 calendar-gap figure, 2026-06-22 snapshot (reconciliation only)
MB_PER_GB = 1000.0                  # docs use decimal GB (84.6 GB == 84_600 MB)

MANIFEST_VERSION = 1
INPUT_ERROR_EXIT = 2                # structural/input error (missing/invalid files) — matches planner's exit 2
BLOCKING_EXIT = 3                   # fail-closed blocking verdict
REPORT_NAME = "coinbase_quality_map.json"
DEFAULT_OUT = "data/reports/backfill/coinbase_backfill_manifest.json"

BLOCKER_KEYS = ("structural", "missing_keys", "coverage_gaps", "inconsistencies",
                "unresolved_days", "batch_incomplete", "book_fill_unavailable",
                "trade_fill_unavailable", "calendar_drift")
```

**Canonical per-day record** (built by `build_day_record`, Task 7):

```python
{
  "day": "2025-01-02",
  "classification": "lake_present_degraded" | ... | None,   # None for calendar-only days
  "sources": ["quality_map", "calendar_gap"],               # provenance
  "calendar": {"in_lake_all_days": bool, "in_usable_days": bool, "is_coinbase_fill_day": bool,
               "book_gap": bool, "trades_gap": bool, "excluded_reason": None | list},
  "book_fill": {"needed": bool, "source": "calendar_gap"|"quality_map"|"both"|None,
                "kind": "full_day"|"partial"|None, "why": str|None,
                "fill_profile": str|None, "full_day_reason": str|None,
                "fill_segments": list|None, "seams": list|None, "seam_policy": dict|None,
                "trusted_lake_start_ts": int|None, "trusted_lake_end_ts": int|None,
                "gb": float, "gb_basis": "measured"|"estimated", "usd": float},
  "trade_fill": {"needed": bool, "source": "calendar"|None, "measured_mb": float|None,
                 "gb": float, "gb_basis": "measured"|"estimated", "usd": float},
  "excluded": None | {"reason": list},
  "unresolved": None | {"why": "no_verdict", "classification": str, "reasons": list},
  "notes": list,
}
```

**Test module:** all tests live in `tests/test_review_backfill_manifest.py`. It loads the script by path (scripts/ is not a package), exactly like `tests/test_plan_quality_map_batches.py`:

```python
import importlib.util, json, pathlib, sys
_SPEC = importlib.util.spec_from_file_location(
    "review_coinbase_backfill_manifest",
    pathlib.Path(__file__).resolve().parents[1] / "scripts"
    / "review_coinbase_backfill_manifest.py")
rv = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = rv
_SPEC.loader.exec_module(rv)
```

**Shared synthetic fixtures** (added to the test module in Task 3, reused after): a calendar builder + report/plan builders. Defined once (see Task 3 Step 1) as module-level helpers `_calendar()`, `_report()`, `_plan()`, `_write_tree()`.

---

## Task 1: Module skeleton + pinned constants + alignment contract test

**Files:**
- Create: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write the module skeleton with the shared constants**

Create `scripts/review_coinbase_backfill_manifest.py`:

```python
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
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import pathlib
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

# ----------------------------------------------------------- cost model (docs §2.2/§6/§8)
BOOK_USD_PER_GB = 1.0
TRADES_USD_PER_GB = 3.0
EST_BOOK_GB_PER_DAY = 2.27
EST_TRADES_GB_PER_DAY = 0.05
CREDIT_USD = 25.0
DOCS_REFERENCE_USD = 92.0
MB_PER_GB = 1000.0

# ----------------------------------------------------------- control
MANIFEST_VERSION = 1
INPUT_ERROR_EXIT = 2
BLOCKING_EXIT = 3
REPORT_NAME = "coinbase_quality_map.json"
DEFAULT_OUT = "data/reports/backfill/coinbase_backfill_manifest.json"
BLOCKER_KEYS = ("structural", "missing_keys", "coverage_gaps", "inconsistencies",
                "unresolved_days", "batch_incomplete", "book_fill_unavailable",
                "trade_fill_unavailable", "calendar_drift")


class ReviewInputError(ValueError):
    """A clear, user-actionable input failure (missing/invalid plan, report, or calendar)."""


if __name__ == "__main__":
    raise SystemExit(main())  # noqa: F821  (main defined in Task 11)
```

- [ ] **Step 2: Write the alignment contract test**

Add the loader block (see Shared definitions) to `tests/test_review_backfill_manifest.py`, then:

```python
import importlib.util as _ilu
import pathlib as _pl


def _load_runner():
    """Load scripts/run_coinbase_quality_map.py by path (needs pandas/numpy — present in the venv)."""
    spec = _ilu.spec_from_file_location(
        "run_coinbase_quality_map",
        _pl.Path(__file__).resolve().parents[1] / "scripts" / "run_coinbase_quality_map.py")
    qm = _ilu.module_from_spec(spec)
    sys.modules[spec.name] = qm
    spec.loader.exec_module(qm)
    return qm


def test_class_enum_aligned_with_runner():
    qm = _load_runner()
    assert rv.CLASSES == qm.CLASSES


def test_fill_profile_enum_aligned_with_stitch_policy():
    import recon.stitch_policy as sp
    assert rv.FULL_DAY_FILL == sp.FULL_DAY_FILL
    assert rv.LAKE_ONLY == sp.LAKE_ONLY
    assert rv.PARTIAL_FILL_PROFILES == sp.PARTIAL_FILL_PROFILES


def test_why_codes_cover_every_runner_fill_decision():
    """Exercise the runner's real coinapi_fill_decision for each class → its why string must be
    a code this tool knows, so a runner why-code change breaks the build here."""
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
```

- [ ] **Step 3: Run the tests to verify they pass**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q`
Expected: 3 passed (the module imports; enums align). If `test_why_codes...` fails, reconcile `WHY_CODES` with the runner's actual strings.

- [ ] **Step 4: Verify the script compiles**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m py_compile scripts/review_coinbase_backfill_manifest.py`
Expected: no output (exit 0).

- [ ] **Step 5: Commit**

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: backfill-manifest review skeleton + pinned quality-map enums"
```

---

## Task 2: Input helpers — sha256 + strict JSON load

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
import pytest


def test_sha256_file_matches_hashlib(tmp_path):
    p = tmp_path / "x.json"
    p.write_bytes(b'{"a": 1}\n')
    import hashlib
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "sha256 or load_json"`
Expected: FAIL (`module 'review_...' has no attribute 'sha256_file'`).

- [ ] **Step 3: Implement the helpers**

Add to the script (after `ReviewInputError`):

```python
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
```

- [ ] **Step 4: Run to verify they pass**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "sha256 or load_json"`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: sha256 + strict JSON-object input loader (fail-closed)"
```

---

## Task 3: Calendar accessors + shared synthetic fixtures

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Add the shared synthetic fixtures to the test module**

Add near the top of the test module (after the loader). These are reused by every later task.

```python
# --- synthetic inputs (mirror real artifact shapes; no vendor I/O) --------------------------
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
            "engine": "native", "thresholds": {"crossed_usable_max": 0.01, "missing_usable_max": 0.02,
            "thin_usable_max": 0.1, "seed_crossed_frac_max": 0.05},
            "quota": {"ok": True, "reason": "ok", "used_gb_before": 0.26, "used_gb_after": 0.26},
            "generated_utc": "2026-07-01T00:00:00Z"}
    meta.update(meta_overrides)
    counts = {c: 0 for c in rv.CLASSES}
    for d in days:
        counts[d["classification"]] = counts.get(d["classification"], 0) + 1
    return {"meta": meta, "summary": {"n_days": len(days), "counts": counts,
                                      "by_class": {}, "coinapi_fill": {}}, "days": days}


def _clean_reports():
    """One batch report: 2025-01-01 lake_usable (no fill), 2025-01-02 degraded → full_day_fill."""
    return [_report([
        _day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable")),
        _day("2025-01-02", "lake_present_degraded",
             _fill_block(True, "quality_over_usable_bar", fill_profile="full_day_fill",
                         full_day_reason="quality_over_usable_bar",
                         fill_segments=[{"source": "coinapi", "start_ts": 1, "start_iso": "x",
                                         "end_ts": 2, "end_iso": "y", "reason": "r"}],
                         seams=[], seam_policy={"seam_guard_s": 60.0})),
    ])]


def _write_tree(tmp_path, cal=None, reports=None):
    """Write calendar + a plan manifest + per-batch reports into tmp_path; return
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
                        "first_day": days[0], "last_day": days[-1],
                        "report_dir": str(rdir)})
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
```

- [ ] **Step 2: Write failing tests for the calendar accessors**

```python
def test_book_gap_and_trade_fill_days():
    cal = _calendar()
    assert rv.book_gap_days(cal) == {"2025-01-10"}
    assert rv.trade_fill_days(cal) == {"2025-01-10", "2025-01-11"}


def test_calendar_batch_days_matches_planner():
    """Reimplements plan_coinbase_quality_map_batches.select_days()['batch_days'] stdlib-only."""
    spec = _ilu.spec_from_file_location(
        "plan_coinbase_quality_map_batches",
        _pl.Path(__file__).resolve().parents[1] / "scripts"
        / "plan_coinbase_quality_map_batches.py")
    pm = _ilu.module_from_spec(spec); sys.modules[spec.name] = pm; spec.loader.exec_module(pm)
    cal = _calendar()
    assert rv.calendar_batch_days(cal) == pm.select_days(cal)["batch_days"]
    assert rv.calendar_batch_days(cal) == ["2025-01-01", "2025-01-02", "2025-01-11"]


def test_measured_mb_present_and_absent():
    cal = _calendar()
    assert rv.measured_mb(cal, "2025-01-10", "book") == 1000.0
    assert rv.measured_mb(cal, "2025-01-11", "book") is None      # book is None (trade-only)
    assert rv.measured_mb(cal, "2025-01-99", "book") is None      # no fill_status entry


def test_is_fillable():
    cal = _calendar()
    assert rv.is_fillable(cal, "2025-01-10", "book") is True
    assert rv.is_fillable(cal, "2025-01-11", "trades") is True
    assert rv.is_fillable(cal, "2025-01-11", "book") is False     # book None
    bad = _calendar(fill_days_unfillable=["2025-01-10"])
    assert rv.is_fillable(bad, "2025-01-10", "book") is False
    err = _calendar(fill_status={"2025-01-10": {"book": {"present": True, "mb": 1.0, "ok": True},
                                                "trades": None, "error": True, "reason": "x", "ok": False}})
    assert rv.is_fillable(err, "2025-01-10", "book") is False
```

- [ ] **Step 3: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "book_gap or batch_days or measured_mb or is_fillable"`
Expected: FAIL (no attribute `book_gap_days`).

- [ ] **Step 4: Implement the calendar accessors**

Add to the script:

```python
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
    """True iff CoinAPI `product` for `day` is verifiably available (present+ok, no error,
    not in the unfillable/probe-error lists) per the calendar verifier's fill_status."""
    fs = _fill_status(cal, day)
    if not fs or fs.get("error") is True:
        return False
    if day in set(cal.get("fill_days_unfillable") or []):
        return False
    if day in set(cal.get("fill_days_probe_error") or []):
        return False
    p = fs.get(product)
    return bool(isinstance(p, dict) and p.get("present") and p.get("ok"))
```

- [ ] **Step 5: Run tests, then commit**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q`
Expected: all passing so far.

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: calendar accessors (book/trade gaps, batch-days, fill_status) + fixtures"
```

---

## Task 4: Cost model helpers

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
def test_gb_from_mb():
    assert rv.gb_from_mb(1000.0) == 1.0
    assert rv.gb_from_mb(None) is None


def test_day_book_gb_measured_vs_estimated():
    cal = _calendar()
    assert rv.day_book_gb(cal, "2025-01-10") == (1.0, "measured")     # 1000 MB / 1000
    assert rv.day_book_gb(cal, "2025-01-02") == (rv.EST_BOOK_GB_PER_DAY, "estimated")


def test_day_trades_gb_measured_vs_estimated():
    cal = _calendar()
    assert rv.day_trades_gb(cal, "2025-01-11") == (0.02, "measured")  # 20 MB / 1000
    assert rv.day_trades_gb(cal, "2025-01-02") == (rv.EST_TRADES_GB_PER_DAY, "estimated")


def test_cost_helpers():
    assert rv.book_usd(2.0) == 2.0
    assert rv.trades_usd(2.0) == 6.0
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "gb or cost"`
Expected: FAIL (no attribute `gb_from_mb`).

- [ ] **Step 3: Implement**

```python
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
```

- [ ] **Step 4: Run, then commit**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "gb or cost"`
Expected: 4 passed.

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: hybrid measured/estimated CoinAPI cost model helpers"
```

---

## Task 5: Report & day-record validation

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
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
    # needs_fill True but no plan (fill_profile null) — a degraded/fill day with no fill policy
    r1 = _day("d", "lake_present_degraded", _fill_block(True, "quality_over_usable_bar"))
    assert "needs_fill_without_plan" in rv.day_record_issues(r1)
    # needs_fill True but lake_only profile
    r2 = _day("d", "lake_present_degraded",
              _fill_block(True, "quality_over_usable_bar", fill_profile="lake_only"))
    assert "needs_fill_without_plan" in rv.day_record_issues(r2)
    # full_day_fill without full_day_reason
    r3 = _day("d", "missing_needs_coinapi",
              _fill_block(True, "lake_book_delta_v2_absent", fill_profile="full_day_fill"))
    assert "full_day_without_reason" in rv.day_record_issues(r3)
    # partial profile with a full_day_reason set
    r4 = _day("d", "lake_present_degraded",
              _fill_block(True, "quality_over_usable_bar", fill_profile="leading_partial_fill",
                          full_day_reason="quality_over_usable_bar"))
    assert "partial_with_full_day_reason" in rv.day_record_issues(r4)
    # full-day route but trusted_lake_* not nulled
    r5 = _day("d", "missing_needs_coinapi",
              _fill_block(True, "lake_book_delta_v2_absent", fill_profile="full_day_fill",
                          full_day_reason="lake_book_delta_v2_absent"), trusted=(10, 20))
    assert "full_day_with_trusted_lake_span" in rv.day_record_issues(r5)
    # needs_fill False but a plan present
    r6 = _day("d", "lake_usable", _fill_block(False, "lake_usable", fill_profile="full_day_fill"))
    assert "plan_without_needs_fill" in rv.day_record_issues(r6)


def test_day_record_issues_missing_keys():
    rec = {"day": "d"}  # no classification, no coinapi_fill
    issues = rv.day_record_issues(rec)
    assert "missing_key:classification" in issues
    assert "missing_key:coinapi_fill" in issues
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "missing_keys or day_record_issues"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ----------------------------------------------------------- validation
_REQUIRED_REPORT_KEYS = ("meta", "summary", "days")
_REQUIRED_DAY_KEYS = ("day", "classification", "coinapi_fill")
_REQUIRED_FILL_KEYS = ("needs_fill", "why", "fill_profile")


def report_missing_keys(report: dict) -> list:
    return [k for k in _REQUIRED_REPORT_KEYS if k not in report]


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
    if prof == FULL_DAY_FILL and fdr is None:
        issues.append("full_day_without_reason")
    if prof in PARTIAL_FILL_PROFILES and fdr is not None:
        issues.append("partial_with_full_day_reason")
    if prof == FULL_DAY_FILL:
        q = rec.get("quality") or {}
        if q.get("trusted_lake_start_ts") is not None or q.get("trusted_lake_end_ts") is not None:
            issues.append("full_day_with_trusted_lake_span")
    return issues
```

- [ ] **Step 4: Run, then commit**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "missing_keys or day_record_issues"`
Expected: all passed.

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: report/day-record validation (missing keys, enums, contradictions)"
```

---

## Task 6: Count recomputation + summary cross-check

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "recompute or summary_counts"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
def recompute_class_counts(days: list) -> dict:
    counts = {c: 0 for c in CLASSES}
    for r in days:
        c = r.get("classification")
        if c in counts:
            counts[c] += 1
    return counts


def summary_count_issues(report: dict) -> list:
    """The report's summary.counts must equal a recomputation over days[] (days[] is primary)."""
    got = (report.get("summary") or {}).get("counts") or {}
    want = recompute_class_counts(report.get("days") or [])
    issues = []
    for c in CLASSES:
        if int(got.get(c, 0)) != want[c]:
            issues.append(f"summary_counts_mismatch:{c}:summary={got.get(c)}:recomputed={want[c]}")
    return issues
```

- [ ] **Step 4: Run, then commit**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "recompute or summary_counts"`
Expected: passed.

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: recompute per-class counts and cross-check report summary"
```

---

## Task 7: Per-day canonical record builder

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
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
                                    "end_ts": 2, "end_iso": "y", "reason": "r"}]  # verbatim
    assert rec["trade_fill"]["needed"] is False


def test_build_day_record_calendar_book_gap_measured():
    cal = _calendar()
    rec = rv.build_day_record("2025-01-10", None, cal)   # book-gap day, not in any report
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
    rep_day = _day("2025-01-11", "lake_usable", _fill_block(False, "lake_usable"))
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "build_day_record"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
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

    classification = report_rec["classification"] if report_rec else None
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
        # calendar book-gap day not carried as a report fill → synthesize a full-day book fill
        gb, basis = day_book_gb(cal, day)
        book.update(needed=True, source="calendar_gap", kind="full_day", why="calendar_book_gap",
                    fill_profile=FULL_DAY_FILL, full_day_reason="calendar_book_gap",
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
```

- [ ] **Step 4: Run, then commit**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "build_day_record"`
Expected: 6 passed.

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: canonical per-day record builder (report+calendar merge, verbatim stitch)"
```

---

## Task 8: Plan loading + batch-completeness checks

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
def test_load_plan_and_reports_indexes_days(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    plan = rv.load_json_object(plan_path, what="plan manifest")
    reports, day_index = rv.load_batch_reports(plan)
    assert len(reports) == 1
    assert set(day_index) == {"2025-01-01", "2025-01-02"}


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
    # delete the batch's report file
    import os as _os
    _os.remove(_os.path.join(plan["batches"][0]["report_dir"], "coinbase_quality_map.json"))
    with pytest.raises(rv.ReviewInputError, match="report"):
        rv.load_batch_reports(plan)


def test_completeness_day_not_mapped_blocks(tmp_path):
    # calendar has an extra present day (2025-01-03) that no report covers
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
    # book-gap day 2025-01-10 removed from skipped list and not in any report
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "completeness or load_plan"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
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
        rdir = b.get("report_dir")
        if not rdir:
            raise ReviewInputError(f"plan batch {b.get('file')!r} has no report_dir")
        rpath = os.path.join(rdir, REPORT_NAME)
        report = load_json_object(rpath, what="quality-map report")
        reports.append({"path": rpath, "report_dir": rdir, "batch": b, "report": report})
        for rec in report.get("days") or []:
            d = rec.get("day")
            if d in day_index:
                raise ReviewInputError(f"day {d} appears in more than one batch report "
                                       "(duplicate_across_batches); batch day-sets must be disjoint")
            day_index[d] = rec
    return reports, day_index


def _batch_ran(report: dict) -> list:
    """Issues indicating the batch did NOT actually run to completion (spec §fix-5)."""
    issues = []
    meta = report.get("meta")
    if not isinstance(meta, dict) or not isinstance(meta.get("quota"), dict):
        return ["missing meta.quota"]
    reason = meta["quota"].get("reason")
    if reason in ("quota_headroom", "exceeds_auto_cap"):
        issues.append(f"quota refused: {reason}")
    n_days = (report.get("summary") or {}).get("n_days")
    if n_days != len(report.get("days") or []):
        issues.append(f"summary.n_days {n_days} != len(days) {len(report.get('days') or [])}")
    return issues


def check_completeness(plan: dict, reports: list, day_index: dict, cal: dict,
                       blockers: dict) -> None:
    """Plan-driven completeness (spec §4). Populates blockers in place."""
    # each batch must have run to completion
    for r in reports:
        for msg in _batch_ran(r["report"]):
            blockers["batch_incomplete"].append(f"{r['report_dir']}: {msg}")

    expected = set(calendar_batch_days(cal))
    mapped = set(day_index)
    for d in sorted(expected - mapped):
        blockers["coverage_gaps"].append(f"day_not_mapped:{d}")
    for d in sorted(mapped - expected):
        blockers["coverage_gaps"].append(f"unexpected_day:{d}")

    # book-gap days must be mapped as missing_needs_coinapi or acknowledged as withheld
    withheld = set((plan.get("skipped") or {}).get("fill_days_book_gap") or [])
    for d in sorted(book_gap_days(cal)):
        rec = day_index.get(d)
        mapped_missing = rec is not None and rec.get("classification") == MISSING_NEEDS_COINAPI
        if not mapped_missing and d not in withheld:
            blockers["coverage_gaps"].append(f"gap_day_unmapped:{d}")

    dropped = (plan.get("skipped") or {}).get("days_dropped_as_excluded_or_book_gap") or []
    if dropped:
        blockers["coverage_gaps"].append(f"contradictory_calendar_dropped:{sorted(dropped)}")
```

- [ ] **Step 4: Run, then commit**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "completeness or load_plan"`
Expected: all passed.

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: plan-driven batch loading + completeness checks (fail-closed)"
```

---

## Task 9: Consistency checks — report validation, meta drift, calendar drift, fill availability

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
def test_check_reports_validation_and_meta_drift(tmp_path):
    reports = [_report([_day("2025-01-01", "lake_usable", _fill_block(False, "lake_usable"))]),
               _report([_day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable"))],
                       symbol="ETH-USD")]  # symbol drift vs batch 1
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
    # report says day is NOT a lake_all_day, calendar says it is
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
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "consistency or unresolved or drift or availability"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
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
        # cross-batch meta drift
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
```

- [ ] **Step 4: Run, then commit**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "consistency or unresolved or drift or availability"`
Expected: all passed.

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: report consistency, meta/calendar drift, fill-availability checks"
```

---

## Task 10: Manifest assembly — sections, cost summary, status

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
def test_build_manifest_ready(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "ready"
    assert m["meta"]["scope_complete"] is True
    # sections
    s = m["sections"]
    assert "2025-01-02" in s["full_day_book_fills"]           # degraded → full-day
    assert "2025-01-10" in s["full_day_book_fills"]           # calendar book gap
    assert set(s["trade_fills"]) == {"2025-01-10", "2025-01-11"}
    assert s["lake_usable_days"] == ["2025-01-01"]
    assert s["lake_present_degraded_days"] == ["2025-01-02"]  # NORMAL section, not blocking
    assert s["excluded_days"] == ["2025-01-20"]
    assert s["unresolved_days"] == []
    # input identity pinned
    assert len(m["meta"]["inputs"]["usable_calendar"]["sha256"]) == 64
    assert len(m["meta"]["inputs"]["batch_reports"][0]["sha256"]) == 64


def test_build_manifest_cost_summary(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    c = m["cost_summary"]
    # book: 2025-01-10 measured 1.0 GB + 2025-01-02 estimated 2.27 GB
    assert c["book_gb_measured"] == 1.0
    assert c["book_gb_estimated"] == rv.EST_BOOK_GB_PER_DAY
    # trades: 2025-01-10 measured 0.03 + 2025-01-11 measured 0.02 = 0.05 GB
    assert round(c["trades_gb_measured"], 4) == 0.05
    assert c["book_usd"] == round(1.0 + rv.EST_BOOK_GB_PER_DAY, 4)
    assert c["trades_usd"] == round(0.05 * 3.0, 4)
    # calendar-gap baseline computed from measured fill_status only (book 1.0 + trades 0.05)
    assert c["calendar_gap_baseline_usd"] == round(1.0 * 1.0 + 0.05 * 3.0, 4)
    assert c["docs_reference_usd"] == 92.0
    assert c["net_usd"] == round(c["gross_usd"] - 25.0, 4)


def test_build_manifest_blocking_unresolved(tmp_path):
    reports = [_report([_day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict"),
                             reasons=["no_seed_snapshots"]),
                        _day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable"))])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    m = rv.build_manifest_readiness(plan_path, cal_path, generated_utc="2026-07-03T00:00:00Z",
                                    report_only=False)
    assert m["meta"]["status"] == "blocking"
    assert m["meta"]["scope_complete"] is False
    assert m["blockers"]["unresolved_days"] == ["2025-01-01"]
    assert m["sections"]["unresolved_days"] == ["2025-01-01"]


def test_build_manifest_inspection_report_only(tmp_path):
    plan_path, cal_path = _write_tree(tmp_path)
    reports = [os.path.join(rv.load_json_object(plan_path, what="p")["batches"][0]["report_dir"],
                            "coinbase_quality_map.json")]
    m = rv.build_manifest_inspection(reports, cal_path, generated_utc="2026-07-03T00:00:00Z")
    assert m["meta"]["status"] == "report_only"
    assert m["meta"]["scope_complete"] is False
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "build_manifest"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
# ----------------------------------------------------------- sections + cost summary
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
    base_book_gb = sum((measured_mb(cal, d, "book") or 0.0) / MB_PER_GB for d in book_gap_days(cal))
    base_trade_gb = sum((measured_mb(cal, d, "trades") or 0.0) / MB_PER_GB
                        for d in trade_fill_days(cal))
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
        "calendar_gap_baseline_usd": baseline, "quality_map_addition_usd": round(gross - baseline, 4),
        "docs_reference_usd": DOCS_REFERENCE_USD,
        "band": {"low_usd": low, "high_usd": gross},
    }


def _cost_model_block() -> dict:
    return {"book_usd_per_gb": BOOK_USD_PER_GB, "trades_usd_per_gb": TRADES_USD_PER_GB,
            "est_book_gb_per_day": EST_BOOK_GB_PER_DAY, "est_trades_gb_per_day": EST_TRADES_GB_PER_DAY,
            "credit_usd": CREDIT_USD, "partial_day_charged_as_full_day": True,
            "tiered_discount_applied": False}


def _universe(reports: list, cal: dict) -> list:
    days = set()
    for r in reports:
        for rec in r["report"].get("days") or []:
            days.add(rec.get("day"))
    days |= book_gap_days(cal) | trade_fill_days(cal)
    days |= set(cal.get("excluded_days_by_reason") or {})
    return sorted(d for d in days if d)


def _thresholds(reports: list) -> dict | None:
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
```

- [ ] **Step 4: Run, then commit**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "build_manifest"`
Expected: all passed.

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: assemble backfill manifest (sections, cost summary, status, input pins)"
```

---

## Task 11: CLI — argparse (mode XOR), main, write, print, exit codes

**Files:**
- Modify: `scripts/review_coinbase_backfill_manifest.py`
- Test: `tests/test_review_backfill_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
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
                        _day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable"))])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    out = tmp_path / "manifest.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out),
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == rv.BLOCKING_EXIT


def test_cli_report_only_downgrades_to_0(tmp_path):
    reports = [_report([_day("2025-01-01", "inconclusive", _fill_block(None, "no_verdict")),
                        _day("2025-01-02", "lake_usable", _fill_block(False, "lake_usable"))])]
    plan_path, cal_path = _write_tree(tmp_path, reports=reports)
    out = tmp_path / "manifest.json"
    rc = rv.main(["--plan-manifest", plan_path, "--out", str(out), "--report-only",
                  "--generated-utc", "2026-07-03T00:00:00Z"])
    assert rc == 0
    assert json.loads(out.read_text())["meta"]["status"] == "blocking"  # honest status


def test_cli_input_error_exit_2(tmp_path):
    rc = rv.main(["--plan-manifest", str(tmp_path / "nope.json"),
                  "--out", str(tmp_path / "m.json")])
    assert rc == rv.INPUT_ERROR_EXIT


def test_cli_requires_exactly_one_mode(tmp_path):
    with pytest.raises(SystemExit):
        rv.parse_args([])                        # neither mode
    with pytest.raises(SystemExit):
        rv.parse_args(["--plan-manifest", "p", "--report", "r"])  # both


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
    rv.main(["--plan-manifest", plan_path, "--out", str(out1), "--generated-utc", "2026-07-03T00:00:00Z"])
    rv.main(["--plan-manifest", plan_path, "--out", str(out2), "--generated-utc", "2026-07-03T00:00:00Z"])
    assert out1.read_bytes() == out2.read_bytes()
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q -k "cli"`
Expected: FAIL.

- [ ] **Step 3: Implement**

```python
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
    print(f"  cost: gross ${c['gross_usd']:.2f} (net ${c['net_usd']:.2f} after ${c['credit_usd']:.0f} "
          f"credit); band ${c['band']['low_usd']:.2f}–${c['band']['high_usd']:.2f}; "
          f"calendar-gap baseline ${c['calendar_gap_baseline_usd']:.2f}")
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
                                                generated_utc=generated, report_only=args.report_only)
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
```

- [ ] **Step 4: Run the full test module**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/review_coinbase_backfill_manifest.py tests/test_review_backfill_manifest.py
git commit -m "feat: CLI (readiness/inspection modes), fail-closed exit codes, deterministic output"
```

---

## Task 12: Docs + full-suite verification

**Files:**
- Modify: `docs/data.md`
- Test: (run full suite + py_compile)

- [ ] **Step 1: Add the review-step subsection to docs/data.md**

Insert after the §5a-QualityMap "CoinAPI cross-validation" material (search for the
`### 5a-QualityMap` block; add this subsection at its end, before `### 5a-Recon` or the
next `###`):

```markdown
#### Reviewed backfill manifest (gate before spend)

`scripts/review_coinbase_backfill_manifest.py` is the **review / decision layer** between the
quality map and any CoinAPI backfill. It joins three artifacts — the batch-plan `manifest.json`
(authoritative batch registry), the per-batch quality-map reports it registers, and the usable
calendar (measured `fill_status` sizes) — into a deterministic, human-auditable backfill
manifest (`data/reports/backfill/coinbase_backfill_manifest.json`, git-ignored):

```
quality map (per-batch reports) + batch plan + usable calendar
   -> reviewed backfill manifest   (this tool: no vendor I/O, no downloads)
   -> human approval + CoinAPI Spend Management (§8)
   -> CoinAPI backfill             (ingest/download_coinapi.py --allow-backfill)
```

The manifest separates full-day book fills, partial-day book fills (stitch segments preserved
verbatim from `coinapi_fill`), trade fills (calendar-sourced), Lake-usable days, Lake-present
degraded days, calendar-excluded days, and unresolved/blocking days, with a hybrid GB/cost
estimate (measured `fill_status` where available, nominal §6 per-day rate for quality-map-added
present days; $1/GB book, $3/GB trades; partial fills charged as a full day-file). It **fails
closed** (`status=blocking`, exit 3): any missing/refused batch report, unmapped planned day,
unresolved `no_verdict` day, contradictory `coinapi_fill`, calendar drift, or unverifiable
`fill_status` blocks a `ready` verdict. `--report-only` downgrades the exit but keeps the honest
status; an inspection run (`--report ...`, no plan) is always `report_only`.

**This tool does NOT unlock or run the backfill.** The §5a gate stays enforced in
`ingest/download_coinapi.py` / `ingest/_common.py`; a multi-day pull still needs
`--allow-backfill` with Spend Management on (§8). The manifest is the auditable spend/approval
input a future backfill runner consumes.
```

- [ ] **Step 2: Verify docs render (no broken fences) and mention the new script in §9 repro list (optional but consistent)**

Add one line to the §9 `## 9. Reproducing the verification` bash block, after the
`plan_coinbase_quality_map_batches.py` line:

```bash
.venv/bin/python scripts/review_coinbase_backfill_manifest.py --plan-manifest data/tmp/coinbase_quality_map_batches/manifest.json  # review batches -> data/reports/backfill/ (gatekeeping; NO download, does not unlock backfill)
```

- [ ] **Step 3: Run py_compile + the full targeted suite**

Run:
```bash
/home/aaron/jepa-btc-forecasting/.venv/bin/python -m py_compile scripts/review_coinbase_backfill_manifest.py
/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_review_backfill_manifest.py -q
```
Expected: py_compile silent (exit 0); pytest all passed.

- [ ] **Step 4: Run the pre-existing related suites to confirm no regressions**

Run:
```bash
/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_plan_quality_map_batches.py tests/test_quality_map.py tests/test_stitch_policy.py -q
```
Expected: all passed (this PR adds a new script + test; it must not touch these).

- [ ] **Step 5: Commit**

```bash
git add docs/data.md
git commit -m "docs: document quality-map -> reviewed backfill manifest -> approval -> backfill"
```

---

## Self-Review

**1. Spec coverage:**
- Inputs (plan manifest, per-batch reports, calendar) → Tasks 2, 3, 8, 10. ✓
- Manifest schema (days[], sections, cost_summary, blockers, meta.inputs sha256) → Tasks 7, 10. ✓
- Fail-closed rules (structural exit 2; missing_keys, coverage_gaps, batch_incomplete,
  inconsistencies, unresolved, calendar_drift, book/trade fill unavailable) → Tasks 5, 6, 8, 9. ✓
- Degraded ≠ blocking; unresolved global block → Tasks 7 (record), 9 (unresolved), 10 (sections). ✓
- Cost model (measured+estimated hybrid, partial=full, computed baseline, docs_reference) → Tasks 4, 10. ✓
- CLI modes XOR + report-only + deterministic output + exit codes → Task 11. ✓
- Docs subsection + "does not unlock backfill" → Task 12. ✓
- Contract-pin enums to the runner/stitch policy → Task 1. ✓

**2. Placeholder scan:** No TBD/TODO; every code step shows complete code; every run step shows the exact command + expected result. ✓

**3. Type/name consistency:** `build_day_record`, `_sections`, `_cost_summary`, `build_manifest_readiness`,
`build_manifest_inspection`, `new_blockers`, `BLOCKER_KEYS`, `load_batch_reports`,
`check_completeness`, `check_report_consistency`, `check_calendar_drift`, `check_fill_availability`,
`book_usd`/`trades_usd`, `day_book_gb`/`day_trades_gb` are used identically across tasks. Record
keys (`book_fill`, `trade_fill`, `sources`, `unresolved`) match the Shared-definitions block. ✓

**Note on a known cross-check (spec §10):** the `crossed_source_full_day` vs
`full_day_reason_counts['crossed_seed_source']` divergence is intentionally NOT hard-failed; it is
not among the blocker checks. Native-fallback over-routing is surfaced only (empty `notes` by
default) — acceptable per the design.
