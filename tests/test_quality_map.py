"""Offline synthetic tests for the quota-aware Coinbase quality-map runner
(`scripts/run_coinbase_quality_map.py`, docs/data.md §5a-Recon / §10 quality-map TODO).

Drives the runner's PURE core (no vendor I/O) with in-memory Lake `book_delta_v2` frames and
`book` snapshots, plus the quota-estimate / quota-gate / classification / report helpers — so the
full classification and quota logic is exercised without any Crypto Lake or CoinAPI access. CI never
touches a vendor here (mirrors tests/test_parity_script.py)."""
import datetime as dt
import importlib.util
import json
import math
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from recon import native as _qm_native
from recon.reseed import book_snapshot

native = pytest.mark.skipif(not _qm_native.native_available(),
                            reason="recon_native extension not built (maturin develop)")

# scripts/ is not a package — load the script module by path (same pattern as test_parity_script).
# Register it in sys.modules before exec so the module-level @dataclass (Thresholds) can resolve its
# own module under `from __future__ import annotations` (the documented importlib idiom).
_SPEC = importlib.util.spec_from_file_location(
    "run_coinbase_quality_map",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_coinbase_quality_map.py",
)
qm = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = qm
_SPEC.loader.exec_module(qm)

DAY = dt.date(2025, 6, 1)
DAY_OPEN = pd.Timestamp("2025-06-01").value
S = 1_000_000_000


# --------------------------------------------------------------------------- Lake fixtures
def _lake_df(rows):
    """Real-Lake-schema book_delta_v2 frame from (ts_ns, seq, is_bid, price, size) tuples."""
    df = pd.DataFrame(rows, columns=["origin_time", "sequence_number", "side_is_bid",
                                     "price", "size"])
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    return df


def _clean_lake_df():
    """A clean, never-crossing day: seeded book stays two-sided/uncrossed all day."""
    return _lake_df([
        (DAY_OPEN + 1 * S, 1, True, 100.0, 1.0),
        (DAY_OPEN + 1 * S, 2, False, 101.0, 1.0),
        (DAY_OPEN + 2 * S, 3, True, 100.0, 2.0),
    ])


def _stranded_lake_df():
    """Cold-started this strands ask101 and crosses ~all day (the 2025-06-01 67% failure mode)."""
    return _lake_df([
        (DAY_OPEN + 1 * S, 1, True, 100.0, 1.0),
        (DAY_OPEN + 1 * S, 2, False, 101.0, 1.0),
        (DAY_OPEN + 2 * S, 3, True, 102.0, 1.0),   # strands ask101 → crossed from +2s onward
    ])


def _valid_seed():
    return [book_snapshot(DAY_OPEN + 1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]


def _fixing_seed():
    """Day-open seed + a later snapshot of the true uncrossed book (drives a reseed)."""
    return [book_snapshot(DAY_OPEN + 1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
            book_snapshot(DAY_OPEN + 3 * S, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]


def _crossed_seed():
    """All-crossed `book` product → EVERY seed candidate rejected → no seed accepted at all."""
    return [book_snapshot(DAY_OPEN + 1, bids=[(101.0, 1.0)], asks=[(100.0, 1.0)])]


def _partial_crossed_seed():
    """The REAL 2026-04-01 case: a mostly-valid but partly-crossed `book` product. A valid day-open
    snapshot DOES seed (seed_accepted=True), but the source has >5% crossed candidates → the accepted
    seed is untrustworthy → inconclusive. (Distinct from `_crossed_seed`, where NO seed is accepted.)"""
    return [book_snapshot(DAY_OPEN + 1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),     # valid → seeds
            book_snapshot(DAY_OPEN + 2 * S, bids=[(105.0, 1.0)], asks=[(104.0, 1.0)]),  # crossed
            book_snapshot(DAY_OPEN + 3 * S, bids=[(106.0, 1.0)], asks=[(105.0, 1.0)])]  # crossed


# --------------------------------------------------------------------------- classify_day (pure)
def test_classify_missing_lake_day_is_missing_needs_coinapi():
    cls, reasons = qm.classify_day(have_lake=False, meta=None, lake_q=None,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.MISSING_NEEDS_COINAPI
    assert "lake_book_delta_v2_absent" in reasons  # exact reason code


def test_classify_seeded_clean_day_is_lake_usable():
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.0}
    lake_q = {"crossed_rate": 0.0001, "missing_book_fraction": 0.0001}
    cls, _ = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q, thresholds=qm.THRESHOLDS)
    assert cls == qm.LAKE_USABLE


def test_classify_seeded_but_crossed_day_is_present_degraded():
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.0}
    lake_q = {"crossed_rate": 0.40, "missing_book_fraction": 0.0}   # >1% crossed
    cls, reasons = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.LAKE_PRESENT_DEGRADED
    assert any("crossed" in r for r in reasons)


def test_classify_missing_book_fraction_over_threshold_is_present_degraded():
    # The missing-book threshold (one of three usable bars) must independently demote a day.
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.0}
    lake_q = {"crossed_rate": 0.0, "missing_book_fraction": 0.05}   # >2% missing
    cls, reasons = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.LAKE_PRESENT_DEGRADED
    assert any("missing_book_fraction" in r for r in reasons)


def test_classify_rejected_seed_day_is_inconclusive():
    # An all-crossed `book` product → NO seed accepted → reconstruction cannot be validated.
    meta = {"seed_accepted": False, "seed_reason": "crossed", "thin_depth_fraction": 0.0}
    lake_q = {"crossed_rate": 0.67, "missing_book_fraction": 0.0}
    cls, reasons = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.INCONCLUSIVE
    assert "seed_rejected:crossed" in reasons      # exact sub-cause (not no_seed_snapshots)


def test_classify_no_snapshots_uses_distinct_reason_code():
    # No seed source at all → inconclusive, but a DIFFERENT sub-cause than a rejected seed.
    meta = {"seed_accepted": False, "seed_reason": "no_snapshots", "thin_depth_fraction": 0.0}
    lake_q = {"crossed_rate": 0.0, "missing_book_fraction": 0.0}
    cls, reasons = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.INCONCLUSIVE
    assert "no_seed_snapshots" in reasons and not any("seed_rejected" in r for r in reasons)


def test_classify_unreliable_seed_source_is_inconclusive():
    # 2026-04-01 reality: a valid seed IS accepted, but the `book` source is >5% crossed → the
    # accepted seed cannot be trusted → inconclusive (NOT a silently-usable verdict).
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.0,
            "snapshot_reason_codes": {"ok": 68, "crossed": 32}}   # 32% crossed source
    lake_q = {"crossed_rate": 0.0, "missing_book_fraction": 0.0}  # reconstruction looks clean...
    cls, reasons = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.INCONCLUSIVE
    assert any("seed_source_crossed_frac" in r for r in reasons)


def test_classify_small_crossed_seed_fraction_stays_usable():
    # A trace of crossed candidates (the 2025-06-01 case ~0%) must NOT demote an otherwise-clean day.
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.0,
            "snapshot_reason_codes": {"ok": 65466, "one_sided": 1}}
    lake_q = {"crossed_rate": 0.0002, "missing_book_fraction": 0.0001}
    cls, _ = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q, thresholds=qm.THRESHOLDS)
    assert cls == qm.LAKE_USABLE


def test_classify_thin_depth_over_threshold_is_present_degraded():
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.50}
    lake_q = {"crossed_rate": 0.0, "missing_book_fraction": 0.0}
    cls, reasons = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.LAKE_PRESENT_DEGRADED
    assert any("thin" in r for r in reasons)


def test_classify_threshold_boundary_is_inclusive_usable():
    # The bars use strict `>`, so a metric exactly AT the threshold is still usable.
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.10}
    lake_q = {"crossed_rate": 0.01, "missing_book_fraction": 0.02}  # all == their max
    cls, _ = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q, thresholds=qm.THRESHOLDS)
    assert cls == qm.LAKE_USABLE


# --------------------------------------------------------------------------- assess_lake_day (real recon path)
def test_assess_missing_day_classifies_missing_needs_coinapi():
    res = qm.assess_lake_day(pd.DataFrame(), None, day=DAY, k=1)
    assert res["classification"] == qm.MISSING_NEEDS_COINAPI
    assert res["lake_book_delta_v2_present"] is False
    assert res["lake_delta_rows"] == 0


def test_assess_valid_seeded_clean_day_is_lake_usable():
    res = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), day=DAY, k=1, seed_min_levels=1)
    assert res["classification"] == qm.LAKE_USABLE
    assert res["seed"]["seed_accepted"] is True
    assert res["quality"]["crossed_rate_after"] < 0.01
    assert res["quality"]["missing_book_fraction"] < 0.02


def test_assess_reseed_repairs_stranded_day_to_lake_usable():
    # Stranded book + a fixing snapshot → reseed clears the crossing → usable (the 2025-06-01 fix).
    res = qm.assess_lake_day(_stranded_lake_df(), _fixing_seed(), day=DAY, k=1,
                             reseed=True, reseed_after_crossed_s=0.0, seed_min_levels=1)
    assert res["classification"] == qm.LAKE_USABLE
    assert res["seed"]["reseed_count"] >= 1
    assert res["quality"]["crossed_rate_after"] < 0.01


def test_assess_seed_only_no_repair_day_is_present_degraded():
    # Seed accepted but reseed disabled → the stranded cross persists → present but degraded.
    res = qm.assess_lake_day(_stranded_lake_df(), _valid_seed(), day=DAY, k=1,
                             reseed=False, reseed_after_crossed_s=0.0, seed_min_levels=1)
    assert res["classification"] == qm.LAKE_PRESENT_DEGRADED
    assert res["seed"]["seed_accepted"] is True
    assert res["quality"]["crossed_rate_after"] > 0.5


def test_assess_all_crossed_book_product_day_is_inconclusive():
    # An all-crossed `book` product → NO seed accepted at all → inconclusive.
    res = qm.assess_lake_day(_stranded_lake_df(), _crossed_seed(), day=DAY, k=1,
                             reseed=True, reseed_after_crossed_s=0.0, seed_min_levels=1)
    assert res["classification"] == qm.INCONCLUSIVE
    assert res["seed"]["seed_accepted"] is False
    assert res["seed"]["seed_reason"] == "crossed"


def test_assess_partially_crossed_book_product_day_is_inconclusive():
    # The REAL 2026-04-01 case (the PR's default demonstration day): a valid seed IS accepted, but the
    # `book` source is ~67% crossed here (>5%) → the accepted seed is untrustworthy → inconclusive.
    # A clean reconstruction must NOT mask an unreliable seed source as lake_usable.
    res = qm.assess_lake_day(_clean_lake_df(), _partial_crossed_seed(), day=DAY, k=1,
                             reseed=True, reseed_after_crossed_s=0.0, seed_min_levels=1)
    assert res["classification"] == qm.INCONCLUSIVE
    assert res["seed"]["seed_accepted"] is True            # a valid seed WAS found...
    assert res["seed"]["snapshot_reason_codes"].get("crossed", 0) >= 1
    assert any("seed_source_crossed_frac" in r for r in res["reasons"])  # ...but source unreliable


def test_assess_no_snapshots_is_inconclusive_not_usable():
    # book_delta_v2 present but NO seed source → cannot validate the reconstruction → inconclusive.
    res = qm.assess_lake_day(_clean_lake_df(), None, day=DAY, k=1)
    assert res["classification"] == qm.INCONCLUSIVE
    assert res["seed"]["snapshots_present"] is False
    assert "no_seed_snapshots" in res["reasons"]


# --------------------------------------------------------------------------- quota estimate / gate
def test_estimate_lake_gb_scales_with_days_and_products():
    one = qm.estimate_lake_gb(1)
    assert one == qm.LAKE_GB_PER_DAY["book_delta_v2"] + qm.LAKE_GB_PER_DAY["book"]
    assert qm.estimate_lake_gb(3) == 3 * one
    assert qm.estimate_lake_gb(0) == 0.0
    # restricting products lowers the estimate
    assert qm.estimate_lake_gb(2, products=("book_delta_v2",)) == \
        2 * qm.LAKE_GB_PER_DAY["book_delta_v2"]


def test_quota_gate_allows_a_small_default_pull():
    d = qm.quota_decision(est_gb=1.0, used_gb=0.26, quota_gb=300.0, max_auto_gb=5.0,
                          allow_broad=False)
    assert d["ok"] is True


def test_quota_gate_refuses_broad_pull_without_override():
    d = qm.quota_decision(est_gb=50.0, used_gb=0.0, quota_gb=300.0, max_auto_gb=5.0,
                          allow_broad=False)
    assert d["ok"] is False
    assert d["reason"] == "exceeds_auto_cap"


def test_quota_gate_allows_broad_pull_with_override():
    d = qm.quota_decision(est_gb=50.0, used_gb=0.0, quota_gb=300.0, max_auto_gb=5.0,
                          allow_broad=True)
    assert d["ok"] is True


def test_quota_gate_refuses_when_it_would_breach_monthly_quota_even_with_override():
    # The 300 GB/month cap is a HARD external limit — --allow-broad overrides the auto cap, NOT the
    # quota headroom. A pull that would exceed the remaining quota is refused regardless.
    d = qm.quota_decision(est_gb=295.0, used_gb=20.0, quota_gb=300.0, max_auto_gb=5.0,
                          allow_broad=True, headroom_gb=10.0)
    assert d["ok"] is False
    assert d["reason"] == "quota_headroom"


def test_quota_gate_allows_zero_gb_request_even_at_the_cap():
    # All days excluded ⇒ est_gb == 0 ⇒ nothing is loaded ⇒ allowed even with no headroom left.
    d = qm.quota_decision(est_gb=0.0, used_gb=299.9, quota_gb=300.0, max_auto_gb=5.0,
                          allow_broad=False, headroom_gb=10.0)
    assert d["ok"] is True


def test_quota_gate_boundary_at_safe_remaining_is_allowed():
    # est_gb exactly == safe_remaining (strict `>`): allowed.
    d = qm.quota_decision(est_gb=290.0, used_gb=0.0, quota_gb=300.0, max_auto_gb=1000.0,
                          allow_broad=False, headroom_gb=10.0)
    assert d["ok"] is True


# --------------------------------------------------------------------------- coinapi fill mapping
def test_fill_decision_crossed_seed_source_inconclusive_needs_fill():
    # The 2026-07-01 cross-validation policy: inconclusive VIA the crossed-seed-source bar → fill
    # (provisional, 2 of 4 days — docs/data.md §5a-QualityMap "CoinAPI cross-validation").
    d = qm.coinapi_fill_decision(qm.INCONCLUSIVE,
                                 [qm.SEED_SOURCE_UNRELIABLE, "seed_source_crossed_frac=0.3751>0.05"])
    assert d["needs_fill"] is True
    assert d["why"] == "crossed_seed_source_cross_validated_2026-07-01"


def test_fill_decision_other_inconclusive_is_no_verdict_not_fill_and_not_clean():
    # No-seed / rejected-seed / load-failure inconclusives have NO measured fill policy: they must
    # surface as needs_fill=None (unresolved), never silently drop out of a fill manifest.
    for reasons in (["no_seed_snapshots"], ["seed_rejected:crossed"], ["lake_load_failed:boom"], None):
        d = qm.coinapi_fill_decision(qm.INCONCLUSIVE, reasons)
        assert d["needs_fill"] is None
        assert d["why"] == "no_verdict"


def test_fill_decision_maps_the_remaining_classes():
    assert qm.coinapi_fill_decision(qm.MISSING_NEEDS_COINAPI,
                                    ["lake_book_delta_v2_absent"])["needs_fill"] is True
    assert qm.coinapi_fill_decision(qm.LAKE_PRESENT_DEGRADED,
                                    ["seed_accepted", "missing_book_fraction=0.6146>0.02"]
                                    )["needs_fill"] is True
    assert qm.coinapi_fill_decision(qm.LAKE_USABLE, ["seed_accepted"])["needs_fill"] is False
    assert qm.coinapi_fill_decision(qm.EXCLUDED, ["binance_gap"])["needs_fill"] is None


def test_classify_day_emits_the_shared_seed_source_reason_constant():
    # classify_day and coinapi_fill_decision must agree on the reason code — via the shared constant.
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.0,
            "snapshot_reason_codes": {"ok": 68, "crossed": 32}}   # 32% crossed source
    lake_q = {"crossed_rate": 0.0, "missing_book_fraction": 0.0}
    cls, reasons = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.INCONCLUSIVE and qm.SEED_SOURCE_UNRELIABLE in reasons
    assert qm.coinapi_fill_decision(cls, reasons)["needs_fill"] is True


def test_build_report_stamps_coinapi_fill_per_day_and_summary():
    days = [
        {"day": "2025-06-01", "classification": qm.LAKE_USABLE, "reasons": ["seed_accepted"]},
        {"day": "2026-04-01", "classification": qm.INCONCLUSIVE,
         "reasons": [qm.SEED_SOURCE_UNRELIABLE, "seed_source_crossed_frac=0.3751>0.05"]},
        {"day": "2025-03-03", "classification": qm.INCONCLUSIVE, "reasons": ["no_seed_snapshots"]},
        {"day": "2024-12-05", "classification": qm.MISSING_NEEDS_COINAPI,
         "reasons": ["lake_book_delta_v2_absent"]},
    ]
    rep = qm.build_report(days, meta={"k": 10})
    stamped = {r["day"]: r["coinapi_fill"] for r in rep["days"]}
    assert stamped["2026-04-01"]["needs_fill"] is True   # crossed-seed-source inconclusive → fill
    assert stamped["2025-03-03"]["needs_fill"] is None   # other inconclusive → unresolved
    assert stamped["2024-12-05"]["needs_fill"] is True
    assert stamped["2025-06-01"]["needs_fill"] is False
    fill = rep["summary"]["coinapi_fill"]
    assert fill["needs_fill"] == ["2026-04-01", "2024-12-05"]
    assert fill["no_verdict"] == ["2025-03-03"]
    assert fill["no_fill"] == ["2025-06-01"]
    # input records are not mutated (build_report stamps copies)
    assert "coinapi_fill" not in days[0]


# --------------------------------------------------------------------------- report aggregation + JSON
def test_build_report_counts_each_classification():
    days = [
        {"day": "2025-06-01", "classification": qm.LAKE_USABLE},
        {"day": "2026-04-01", "classification": qm.INCONCLUSIVE},
        {"day": "2024-12-05", "classification": qm.MISSING_NEEDS_COINAPI},
    ]
    rep = qm.build_report(days, meta={"k": 10})
    assert rep["summary"]["n_days"] == 3
    assert rep["summary"]["counts"][qm.LAKE_USABLE] == 1
    assert rep["summary"]["counts"][qm.MISSING_NEEDS_COINAPI] == 1
    assert rep["summary"]["by_class"][qm.INCONCLUSIVE] == ["2026-04-01"]
    # every canonical class is present in the counts (even at zero) for a stable schema
    for c in qm.CLASSES:
        assert c in rep["summary"]["counts"]


def test_report_is_strict_json_serializable(tmp_path):
    # Real assess output (which embeds numpy scalars + possible NaN) must survive allow_nan=False.
    res = qm.assess_lake_day(_stranded_lake_df(), _fixing_seed(), day=DAY, k=1,
                             reseed=True, reseed_after_crossed_s=0.0, seed_min_levels=1)
    rep = qm.build_report([res], meta={"k": 1, "thresholds": qm.THRESHOLDS.as_dict()})
    out = tmp_path / "quality_map.json"
    qm.write_report(rep, str(out))
    txt = out.read_text()
    loaded = json.loads(txt)                       # round-trips through strict JSON
    assert loaded["days"][0]["day"] == "2025-06-01"
    assert "NaN" not in txt and "Infinity" not in txt


def test_json_safe_sanitizes_non_finite_and_numpy():
    out = qm._json_safe({"a": float("nan"), "b": np.float64(1.5),
                         "c": [np.int64(3), float("inf")], "d": "x"})
    assert out == {"a": None, "b": 1.5, "c": [3, None], "d": "x"}
    assert math.isfinite(out["b"])


def test_thresholds_are_documented_and_stable():
    t = qm.THRESHOLDS
    assert t.crossed_usable_max == 0.01
    assert t.missing_usable_max == 0.02
    assert t.thin_usable_max == 0.10
    assert t.seed_crossed_frac_max == 0.05
    assert set(t.as_dict()) == {"crossed_usable_max", "missing_usable_max", "thin_usable_max",
                                "seed_crossed_frac_max"}


# --------------------------------------------------------------------------- schema-consistent records
def test_excluded_result_is_schema_consistent():
    r = qm.excluded_result(DAY, ["missing:binF_book"], k=10, grid_ms=1000)
    assert r["classification"] == qm.EXCLUDED
    assert r["reasons"] == ["missing:binF_book"]
    assert r["lake_book_delta_v2_present"] is None and r["lake_delta_rows"] is None
    assert set(r) >= {"day", "classification", "reasons", "seed", "quality", "coinapi", "calendar"}


def test_inconclusive_load_failure_record():
    r = qm.inconclusive_load_failure(DAY, "RuntimeError('boom')", k=10, grid_ms=1000)
    assert r["classification"] == qm.INCONCLUSIVE
    assert r["reasons"][0].startswith("lake_load_failed:")
    assert set(r) >= {"day", "classification", "reasons", "seed", "quality", "coinapi", "calendar"}


# --------------------------------------------------------------------------- calendar / coinapi / day helpers
_CAL = {
    "usable_days": ["2025-06-01", "2026-04-01"],
    "lake_all_days": ["2025-06-01", "2026-04-01"],
    # coinbase_fill_days: {"book"/"trades": which product Lake is MISSING that day}. book=True is a
    # book gap (this runner's target); book=False (e.g. 2024-11-13) is a trade-only gap (book present).
    "coinbase_fill_days": {
        "2024-12-06": {"book": True, "trades": True},
        "2024-12-05": {"book": True, "trades": True},
        "2024-11-13": {"book": False, "trades": True},   # trade-only → NOT a book gap
        "2024-07-03": {"book": False, "trades": True},   # trade-only → NOT a book gap
    },
    "excluded_days_by_reason": {"2026-02-05": ["missing:binF_book", "missing:binF_trade"]},
    "fill_status": {"2024-12-05": {"book": {"present": True}}, "2024-12-06": {"book": None}},
}


def test_calendar_context_flags():
    assert qm.calendar_context(_CAL, "2025-06-01")["in_usable_days"] is True
    assert qm.calendar_context(_CAL, "2024-12-05")["is_coinbase_fill_day"] is True
    assert qm.calendar_context(_CAL, "2026-02-05")["excluded_reason"] == \
        ["missing:binF_book", "missing:binF_trade"]
    assert qm.calendar_context(None, "2025-06-01") == qm._default_calendar_block()


def test_coinapi_context_reflects_fillability_and_local_parquet(tmp_path):
    c = qm.coinapi_context(_CAL, dt.date(2024, 12, 5), str(tmp_path), "COINBASE", "BTC-USD")
    assert c["parquet_local"] is False and c["fillable"] is True
    assert qm.coinapi_context(_CAL, dt.date(2024, 12, 6), str(tmp_path),
                              "COINBASE", "BTC-USD")["fillable"] is False
    p = pathlib.Path(qm.coinapi_parquet_path(str(tmp_path), dt.date(2024, 12, 5),
                                             "COINBASE", "BTC-USD"))
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("x")
    assert qm.coinapi_context(_CAL, dt.date(2024, 12, 5), str(tmp_path),
                              "COINBASE", "BTC-USD")["parquet_local"] is True


def test_gap_days_from_calendar_returns_book_gaps_only_sorted():
    # Only BOOK-gap days (coinbase_fill_days[*].book == True), sorted; trade-only days are skipped
    # (this runner maps book_delta_v2, so a trade-only day's Lake book is present — Codex P2).
    assert qm.gap_days_from_calendar(_CAL, 5) == ["2024-12-05", "2024-12-06"]  # 2 book gaps only
    assert qm.gap_days_from_calendar(_CAL, 1) == ["2024-12-05"]
    assert "2024-11-13" not in qm.gap_days_from_calendar(_CAL, 5)   # trade-only excluded
    assert qm.gap_days_from_calendar(None, 2) == []
    assert qm.gap_days_from_calendar(_CAL, 0) == []


def test_resolve_days_include_gap_days_picks_book_gaps_only(tmp_path):
    cal = tmp_path / "cal.json"
    cal.write_text(json.dumps(_CAL))
    args = qm.parse_args(["--days", "2025-06-01", "--include-gap-days", "5",
                          "--usable-calendar", str(cal)])
    days = qm.resolve_days(args)
    assert dt.date(2024, 12, 5) in days and dt.date(2024, 12, 6) in days   # book gaps added
    assert dt.date(2024, 11, 13) not in days and dt.date(2024, 7, 3) not in days  # trade-only skipped


def test_resolve_days_dedupes_and_honors_explicit_days():
    args = qm.parse_args(["--days", "2025-06-01,2026-04-01,2025-06-01"])
    assert qm.resolve_days(args) == [dt.date(2025, 6, 1), dt.date(2026, 4, 1)]


def test_resolve_days_reads_a_days_file(tmp_path):
    f = tmp_path / "days.txt"
    f.write_text("2025-06-01, 2026-04-01\n2024-12-05\n")
    args = qm.parse_args(["--days-file", str(f)])
    assert qm.resolve_days(args) == [dt.date(2025, 6, 1), dt.date(2026, 4, 1), dt.date(2024, 12, 5)]


# --------------------------------------------------------------------------- main() wiring (monkeypatched vendor)
def _patch_vendor(monkeypatch, *, used_gb=0.26, delta_for=None, raise_for=None, snaps=None):
    """Stub the vendor seam so main() runs fully offline; returns the list of days actually loaded."""
    loaded: list = []
    monkeypatch.setattr(qm, "lake_session", lambda: object())
    monkeypatch.setattr(qm, "lake_used_data",
                        lambda sess: {"downloaded_gb": used_gb, "timeframe_days": 31})

    def fake_delta(sess, day, ex, sym):
        loaded.append(day)
        if raise_for and day in raise_for:
            raise raise_for[day]
        return (delta_for or {}).get(day, pd.DataFrame())

    monkeypatch.setattr(qm, "load_lake_book_delta_v2", fake_delta)
    monkeypatch.setattr(qm, "load_lake_book_snapshots", lambda *a, **k: list(snaps or []))
    return loaded


def _write_cal(tmp_path, cal):
    p = tmp_path / "cal.json"
    p.write_text(json.dumps(cal))
    return str(p)


def _run_report(tmp_path, argv):
    p = pathlib.Path(argv[argv.index("--out-dir") + 1]) / "coinbase_quality_map.json"
    return json.loads(p.read_text()) if p.exists() else None


def test_main_excluded_day_is_classified_without_loading(monkeypatch, tmp_path):
    loaded = _patch_vendor(monkeypatch)
    cal = _write_cal(tmp_path, _CAL)
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2026-02-05", "--usable-calendar", cal, "--out-dir", out])
    assert rc == 0
    assert loaded == []                                  # excluded BEFORE any Lake load (saves quota)
    rep = _run_report(tmp_path, ["--out-dir", out])
    assert rep["summary"]["counts"][qm.EXCLUDED] == 1
    assert rep["days"][0]["classification"] == qm.EXCLUDED


def test_main_all_excluded_runs_without_a_lake_session(monkeypatch, tmp_path):
    # Calendar-only run (every day excluded) must NOT require Lake keys / create a session (Codex P3).
    def _explode_session():
        raise SystemExit("Crypto Lake AWS key not found in .env or environment")

    def _explode(*a, **k):
        raise AssertionError("Lake vendor call made on an all-excluded run")

    monkeypatch.setattr(qm, "lake_session", _explode_session)
    monkeypatch.setattr(qm, "lake_used_data", _explode)
    monkeypatch.setattr(qm, "load_lake_book_delta_v2", _explode)
    cal = _write_cal(tmp_path, _CAL)
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2026-02-05", "--usable-calendar", cal, "--out-dir", out])
    assert rc == 0                                        # succeeds despite no Lake keys
    rep = _run_report(tmp_path, ["--out-dir", out])
    assert rep["summary"]["counts"][qm.EXCLUDED] == 1
    assert rep["meta"]["quota"]["reason"] == "no_days_to_load"
    assert rep["meta"]["quota"]["used_gb_before"] is None


def test_main_gap_day_raising_nofiles_is_missing_needs_coinapi(monkeypatch, tmp_path):
    # lakeapi raises NoFilesFound for an absent partition; it must map to missing_needs_coinapi.
    class NoFilesFound(Exception):
        pass
    gap = dt.date(2024, 12, 5)
    loaded = _patch_vendor(monkeypatch, raise_for={gap: NoFilesFound("No files found for the period")})
    cal = _write_cal(tmp_path, _CAL)
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2024-12-05", "--usable-calendar", cal, "--out-dir", out])
    assert rc == 0 and loaded == [gap]                   # it DID attempt the load, then caught NoFiles
    rep = _run_report(tmp_path, ["--out-dir", out])
    assert rep["days"][0]["classification"] == qm.MISSING_NEEDS_COINAPI


def test_main_generic_load_error_is_inconclusive(monkeypatch, tmp_path):
    bad = dt.date(2025, 6, 1)
    _patch_vendor(monkeypatch, raise_for={bad: RuntimeError("transient s3 error")})
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2025-06-01", "--out-dir", out, "--max-auto-gb", "10"])
    assert rc == 0
    rep = _run_report(tmp_path, ["--out-dir", out])
    assert rep["days"][0]["classification"] == qm.INCONCLUSIVE
    assert rep["days"][0]["reasons"][0].startswith("lake_load_failed:")


def test_main_refuses_broad_request_without_override(monkeypatch, tmp_path):
    loaded = _patch_vendor(monkeypatch, used_gb=0.0)
    out = str(tmp_path / "rep")
    # 2 days × 0.48 = 0.96 GB > a 0.5 GB auto cap, no --allow-broad → refuse before any load.
    rc = qm.main(["--days", "2025-06-01,2026-04-01", "--out-dir", out, "--max-auto-gb", "0.5"])
    assert rc == qm.QUOTA_REFUSED_EXIT
    assert loaded == []                                  # refused BEFORE loading
    assert not (tmp_path / "rep" / "coinbase_quality_map.json").exists()


def test_main_unreadable_used_data_refuses_run(monkeypatch, tmp_path):
    _patch_vendor(monkeypatch)

    def boom(sess):
        raise RuntimeError("used_data network down")

    monkeypatch.setattr(qm, "lake_used_data", boom)
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2025-06-01", "--out-dir", out, "--allow-broad"])
    assert rc == qm.QUOTA_REFUSED_EXIT                   # fail-safe: refuse even with --allow-broad


# --------------------------------------------------------------------------- engine selection (§5a native)
def test_main_engine_python_records_python_engine(monkeypatch, tmp_path):
    _patch_vendor(monkeypatch, delta_for={dt.date(2025, 6, 1): _clean_lake_df()}, snaps=_valid_seed())
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2025-06-01", "--engine", "python", "--k", "1", "--seed-min-levels", "1",
                  "--out-dir", out, "--max-auto-gb", "10"])
    assert rc == 0
    rep = _run_report(tmp_path, ["--out-dir", out])
    assert rep["meta"]["engine"] == "python" and rep["meta"]["engine_requested"] == "python"
    assert rep["days"][0]["classification"] == qm.LAKE_USABLE


@native
def test_main_engine_native_runs_and_matches_python(monkeypatch, tmp_path):
    # --engine native must actually use the native core AND land on the same classification as Python
    # (the conformance guarantee). BTC-USD has a verified tick scale, so native is selected.
    argv = ["--days", "2025-06-01", "--k", "1", "--seed-min-levels", "1", "--max-auto-gb", "10"]

    def _run(engine, sub):
        _patch_vendor(monkeypatch, delta_for={dt.date(2025, 6, 1): _clean_lake_df()},
                      snaps=_valid_seed())
        out = str(tmp_path / sub)
        rc = qm.main(argv + ["--engine", engine, "--out-dir", out])
        assert rc == 0
        return _run_report(tmp_path, ["--out-dir", out])

    nat = _run("native", "nat")
    py = _run("python", "py")
    assert nat["meta"]["engine"] == "native"
    assert nat["meta"]["engine_price_scale"] == 100
    assert nat["days"][0]["classification"] == py["days"][0]["classification"]
    assert nat["days"][0]["quality"]["crossed_rate_after"] == py["days"][0]["quality"]["crossed_rate_after"]
    assert nat["days"][0]["quality"]["missing_book_fraction"] == py["days"][0]["quality"]["missing_book_fraction"]


def test_main_engine_native_unavailable_fails_before_load(monkeypatch, tmp_path):
    # Simulate the extension being absent: an explicit --engine native must exit nonzero BEFORE any
    # Lake load (it must never silently fall back — plan Review Checklist).
    loaded = _patch_vendor(monkeypatch, delta_for={dt.date(2025, 6, 1): _clean_lake_df()},
                           snaps=_valid_seed())
    monkeypatch.setattr(qm._native, "native_available", lambda: False)
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2025-06-01", "--engine", "native", "--out-dir", out, "--max-auto-gb", "10"])
    assert rc == qm.NATIVE_UNAVAILABLE_EXIT
    assert loaded == []                                          # aborted before any Lake load
    assert not (tmp_path / "rep" / "coinbase_quality_map.json").exists()


def test_main_engine_native_unverified_symbol_fails_before_load(monkeypatch, tmp_path):
    # Native extension present but the symbol has NO verified tick scale => explicit native must abort.
    loaded = _patch_vendor(monkeypatch, delta_for={dt.date(2025, 6, 1): _clean_lake_df()})
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2025-06-01", "--engine", "native", "--symbol", "FOO-BAR",
                  "--out-dir", out, "--max-auto-gb", "10"])
    assert rc == qm.NATIVE_UNAVAILABLE_EXIT
    assert loaded == []


def test_main_engine_auto_falls_back_cleanly_for_unverified_symbol(monkeypatch, tmp_path):
    # auto + a symbol without a verified tick scale => Python, run completes, engine recorded.
    _patch_vendor(monkeypatch, delta_for={dt.date(2025, 6, 1): _clean_lake_df()}, snaps=_valid_seed())
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2025-06-01", "--engine", "auto", "--symbol", "FOO-BAR", "--k", "1",
                  "--seed-min-levels", "1", "--out-dir", out, "--max-auto-gb", "10"])
    assert rc == 0
    rep = _run_report(tmp_path, ["--out-dir", out])
    assert rep["meta"]["engine"] == "python" and rep["meta"]["engine_requested"] == "auto"


def test_main_no_lake_seed_estimate_excludes_book_product(monkeypatch, tmp_path):
    # --no-lake-seed skips the `book` snapshot pull, so the quota estimate must drop the `book` size
    # (else a cold-start run is over-estimated and can be wrongly refused — Codex P2).
    df = _clean_lake_df()
    _patch_vendor(monkeypatch, delta_for={dt.date(2025, 6, 1): df})
    out = str(tmp_path / "rep")
    rc = qm.main(["--days", "2025-06-01", "--no-lake-seed", "--out-dir", out])
    assert rc == 0
    rep = _run_report(tmp_path, ["--out-dir", out])
    assert rep["meta"]["quota"]["est_gb"] == qm.LAKE_GB_PER_DAY["book_delta_v2"]   # book NOT counted
    assert rep["days"][0]["classification"] == qm.INCONCLUSIVE   # no seed source → can't validate
