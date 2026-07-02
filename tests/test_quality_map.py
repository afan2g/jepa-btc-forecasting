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
        {"day": "2025-02-02", "classification": qm.EXCLUDED, "reasons": ["binance_gap"]},
    ]
    rep = qm.build_report(days, meta={"k": 10})
    stamped = {r["day"]: r["coinapi_fill"] for r in rep["days"]}
    assert stamped["2026-04-01"]["needs_fill"] is True   # crossed-seed-source inconclusive → fill
    assert stamped["2025-03-03"]["needs_fill"] is None   # other inconclusive → unresolved
    assert stamped["2024-12-05"]["needs_fill"] is True
    assert stamped["2025-06-01"]["needs_fill"] is False
    fill = rep["summary"]["coinapi_fill"]
    assert fill["needs_fill"] == ["2026-04-01", "2024-12-05"]
    assert fill["no_verdict"] == ["2025-03-03"]          # unresolved ONLY — no excluded days here
    assert fill["no_fill"] == ["2025-06-01"]
    # calendar-excluded days are out of Coinbase-fill scope, NOT unresolved: separate bucket
    assert fill["not_in_scope"] == ["2025-02-02"]
    # input records are not mutated (build_report stamps copies)
    assert "coinapi_fill" not in days[0]


# --------------------------------------------------------------------------- partial-day fill wiring
# (docs/superpowers/plans/2026-07-02-partial-day-fill-policy.md Q7 / Task 2: the per-day
# `coinapi_fill` block gains the stitch plan, `quality` gains coverage timestamps + invalid runs,
# and the summary gains partial_fill + fill_counts. No vendor I/O; the backfill gate is untouched.)
from recon.stitch_policy import plan_day_stitch  # noqa: E402


def _leading_partial_lake_df():
    """The 2025-01-07 shape at full-day scale: Lake silent until +50,000 s, clean afterwards."""
    t0 = DAY_OPEN + 50_000 * S
    return _lake_df([
        (t0, 1, True, 100.0, 1.0),
        (t0, 2, False, 101.0, 1.0),
        (t0 + 1 * S, 3, True, 100.0, 2.0),
    ])


def _late_seed():
    return [book_snapshot(DAY_OPEN + 50_000 * S, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]


def _day_grid_plan(valid_from_s, *, trusted=True, day="2025-01-07"):
    """A real plan_day_stitch dict over the full 86,400-sample day grid."""
    ts = DAY_OPEN + np.arange(86_400, dtype=np.int64) * S
    valid = ts >= DAY_OPEN + valid_from_s * S
    return plan_day_stitch(ts, valid, grid_ns=S, seed_accepted=True,
                           seed_ts=DAY_OPEN + valid_from_s * S, seed_source_trusted=trusted,
                           day=day).as_dict()


def test_assess_clean_day_emits_lake_only_stitch_plan_and_coverage():
    res = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), day=DAY, k=1, seed_min_levels=1)
    assert res["classification"] == qm.LAKE_USABLE
    plan = res["stitch_plan"]
    assert plan["fill_profile"] == "lake_only"
    q = res["quality"]
    # boundary = 3rd consecutive valid grid sample at/after the seed (+1 ns): +1s, +2s, +3s
    assert q["trusted_lake_start_ts"] == plan["trusted_lake_start_ts"] == DAY_OPEN + 3 * S
    assert q["trusted_lake_end_ts"] == DAY_OPEN + 86_400 * S
    assert q["lake_present_start_ts"] == DAY_OPEN + 1 * S
    assert q["lake_present_end_ts"] == DAY_OPEN + 86_400 * S
    assert q["n_invalid_runs"] == 1
    assert q["invalid_runs"] == [[DAY_OPEN, DAY_OPEN + 1 * S]]  # the pre-seed day-open sample


def test_assess_leading_partial_day_plans_leading_fill():
    res = qm.assess_lake_day(_leading_partial_lake_df(), _late_seed(), day=DAY, k=1,
                             seed_min_levels=1)
    assert res["classification"] == qm.LAKE_PRESENT_DEGRADED     # missing ~58% > 2%, Lake-only view
    plan = res["stitch_plan"]
    assert plan["fill_profile"] == "leading_partial_fill"
    segs = plan["fill_segments"]
    boundary = DAY_OPEN + 50_002 * S                             # warmup requalifies at the 3rd sample
    assert [s["source"] for s in segs] == ["coinapi", "lake"]
    assert segs[0]["reason"] == "lake_missing_leading_segment"
    assert segs[0]["end_ts"] == boundary == segs[1]["start_ts"]
    assert plan["seams"] == [boundary]
    q = res["quality"]
    assert q["trusted_lake_start_ts"] == boundary
    assert q["lake_present_start_ts"] == DAY_OPEN + 50_000 * S
    assert q["invalid_runs"] == [[DAY_OPEN, DAY_OPEN + 50_000 * S]]


def test_assess_crossed_seed_source_day_plans_full_day_crossed():
    # 2024-08-05 shape: classification stays inconclusive (Lake-only view), but the plan routes
    # full-day CoinAPI with the crossed-source reason — crossed dominates even a partial day.
    res = qm.assess_lake_day(_clean_lake_df(), _partial_crossed_seed(), day=DAY, k=1,
                             reseed=True, reseed_after_crossed_s=0.0, seed_min_levels=1)
    assert res["classification"] == qm.INCONCLUSIVE
    plan = res["stitch_plan"]
    assert plan["fill_profile"] == "full_day_fill"
    assert plan["full_day_reason"] == "crossed_seed_source"
    q = res["quality"]
    assert q["trusted_lake_start_ts"] is None                    # full-day route: no trusted span
    assert q["lake_present_start_ts"] is not None                # presence recorded, trust not implied


def test_assess_missing_day_carries_no_stitch_plan():
    res = qm.assess_lake_day(pd.DataFrame(), None, day=DAY, k=1)
    assert res["stitch_plan"] is None
    assert res["quality"]["trusted_lake_start_ts"] is None
    assert res["quality"]["invalid_runs"] is None


def test_assess_native_meta_without_coverage_falls_back_full_day(monkeypatch):
    # A native meta WITHOUT the `coverage` block (a stale pre-coverage extension would be rejected
    # at import, but the script-level fallback stays defensive): no mask plan, all-None coverage,
    # and a degraded native fill day gets the conservative synthesized full-day plan at report
    # build. Offline: the native reconstruction is stubbed with coverage-less metrics-only meta,
    # the exact shape `_lake_quality_from_meta` consumes.
    meta = {"seed_accepted": True, "seed_reason": "ok", "seed_ts": DAY_OPEN + 1,
            "reseed_count": 0, "reseed_blocked_invalid_snapshot": 0,
            "snapshot_reason_codes": {"ok": 10}, "thin_depth_fraction": 0.0,
            "crossed_duration_s": 5.0, "n_samples": 86_400, "crossed_samples": 34_560,
            "crossed_rate": 0.40, "missing_book_samples": 0, "missing_book_fraction": 0.0}
    monkeypatch.setattr(qm, "_seeded_reconstruct", lambda *a, **k: (None, dict(meta)))
    res = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), day=DAY, k=1, seed_min_levels=1,
                             cold_ab=False, engine="native", price_scale=100)
    assert res["classification"] == qm.LAKE_PRESENT_DEGRADED    # crossed 40% > 1%
    assert res["stitch_plan"] is None
    assert all(res["quality"][key] is None for key in qm._EMPTY_COVERAGE)
    rep = qm.build_report([res], meta={})
    day_rec = rep["days"][0]
    assert "stitch_plan" not in day_rec
    cf = day_rec["coinapi_fill"]
    assert cf["needs_fill"] is True
    assert cf["fill_profile"] == "full_day_fill"
    assert cf["full_day_reason"] == "quality_over_usable_bar"
    seg, = cf["fill_segments"]
    assert seg["start_iso"] == "2025-06-01T00:00:00Z" and seg["end_iso"] == "2025-06-02T00:00:00Z"


# --------------------------------------------------- native coverage → partial plans (plan Task 3)
def _native_cov_meta(*, n=86_400, invalid_runs_idx, present, seed_ts, crossed=0, missing=0):
    """Metrics-only native-shaped meta WITH the compact `coverage` block (the recon/native.py
    `_assemble` contract): half-open [i0, i1) sample-index invalid runs + presence bound indices."""
    pfi, pli = present
    return {"seed_accepted": True, "seed_reason": "ok", "seed_ts": seed_ts,
            "reseed_count": 0, "reseed_blocked_invalid_snapshot": 0,
            "snapshot_reason_codes": {"ok": 10}, "thin_depth_fraction": 0.0,
            "crossed_duration_s": 0.0, "n_samples": n, "crossed_samples": crossed,
            "crossed_rate": crossed / n, "missing_book_samples": missing,
            "missing_book_fraction": missing / n,
            "coverage": {"present_first_idx": pfi, "present_last_idx": pli,
                         "n_invalid_runs": len(invalid_runs_idx),
                         "invalid_runs_idx": [list(r) for r in invalid_runs_idx]}}


def test_assess_native_coverage_meta_plans_leading_partial_fill(monkeypatch):
    # The native compact coverage block must reconstruct the SAME mask-derived stitch plan + Q7
    # coverage keys the Python frame path emits for the 2025-01-07 shape — a real partial plan,
    # not the full-day fallback. Offline: native reconstruction stubbed with coverage meta.
    meta = _native_cov_meta(invalid_runs_idx=[[0, 50_000]], present=(50_000, 86_399),
                            seed_ts=DAY_OPEN + 50_000 * S, missing=50_000)
    monkeypatch.setattr(qm, "_seeded_reconstruct", lambda *a, **k: (None, dict(meta)))
    res = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), day=DAY, k=1, seed_min_levels=1,
                             cold_ab=False, engine="native", price_scale=100)
    assert res["classification"] == qm.LAKE_PRESENT_DEGRADED     # missing ~58% > 2%
    plan = res["stitch_plan"]
    boundary = DAY_OPEN + 50_002 * S                             # 3rd consecutive valid sample
    assert plan["fill_profile"] == "leading_partial_fill"
    assert [s["source"] for s in plan["fill_segments"]] == ["coinapi", "lake"]
    assert plan["fill_segments"][0]["end_ts"] == boundary
    assert plan["seams"] == [boundary]
    q = res["quality"]
    assert q["trusted_lake_start_ts"] == boundary
    assert q["trusted_lake_end_ts"] == DAY_OPEN + 86_400 * S
    assert q["lake_present_start_ts"] == DAY_OPEN + 50_000 * S
    assert q["lake_present_end_ts"] == DAY_OPEN + 86_400 * S
    assert q["n_invalid_runs"] == 1
    assert q["invalid_runs"] == [[DAY_OPEN, DAY_OPEN + 50_000 * S]]
    rep = qm.build_report([res], meta={})                        # the partial plan survives stamping
    assert rep["days"][0]["coinapi_fill"]["fill_profile"] == "leading_partial_fill"


def test_assess_native_coverage_invalid_runs_capped_with_full_count(monkeypatch):
    # Q7 cap at the report boundary, native path: invalid_runs[:100], n_invalid_runs keeps 150.
    runs = [[i, i + 1] for i in range(0, 1200, 8)]               # 150 isolated invalid samples
    meta = _native_cov_meta(invalid_runs_idx=runs, present=(0, 86_399), seed_ts=DAY_OPEN,
                            missing=150)
    monkeypatch.setattr(qm, "_seeded_reconstruct", lambda *a, **k: (None, dict(meta)))
    res = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), day=DAY, k=1, seed_min_levels=1,
                             cold_ab=False, engine="native", price_scale=100)
    q = res["quality"]
    assert q["n_invalid_runs"] == 150
    assert len(q["invalid_runs"]) == qm.INVALID_RUNS_CAP == 100
    assert q["invalid_runs"][0] == [DAY_OPEN, DAY_OPEN + 1 * S]
    # present_first_idx=0 while valid[0] is False: lake_present_start_ts must come from the
    # PRESENCE bounds, not the validity complement (pins _masks_from_native_coverage's
    # presence-bounds wiring — a valid-mask-derived presence would start one sample late).
    assert q["lake_present_start_ts"] == DAY_OPEN


@native
def test_assess_native_engine_matches_python_stitch_plan_and_coverage():
    # End-to-end engine conformance at the assess level on a leading-partial day: the native
    # coverage path must yield the IDENTICAL stitch plan and quality block (incl. the Q7 coverage
    # keys) as the Python frame path — the load-bearing Task-3 guarantee for the broad map.
    kw = dict(day=DAY, k=1, seed_min_levels=1, cold_ab=False)
    py = qm.assess_lake_day(_leading_partial_lake_df(), _late_seed(), engine="python", **kw)
    nat = qm.assess_lake_day(_leading_partial_lake_df(), _late_seed(), engine="native",
                             price_scale=100, **kw)
    assert nat["classification"] == py["classification"] == qm.LAKE_PRESENT_DEGRADED
    assert nat["stitch_plan"] == py["stitch_plan"]
    assert nat["stitch_plan"]["fill_profile"] == "leading_partial_fill"
    assert nat["quality"] == py["quality"]


@native
def test_assess_native_engine_clean_day_matches_python_lake_only():
    kw = dict(day=DAY, k=1, seed_min_levels=1, cold_ab=False)
    py = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), engine="python", **kw)
    nat = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), engine="native", price_scale=100,
                             **kw)
    assert nat["classification"] == py["classification"] == qm.LAKE_USABLE
    assert nat["stitch_plan"] == py["stitch_plan"]
    assert nat["stitch_plan"]["fill_profile"] == "lake_only"
    assert nat["quality"] == py["quality"]


def _trailing_crossed_lake_df():
    """Clean from day open, then a stranding bid at +50,000 s crosses the book to day end —
    the trailing-partial shape whose PRESENCE (both tops still printed) outlives its validity."""
    return _lake_df([
        (DAY_OPEN + 1 * S, 1, True, 100.0, 1.0),
        (DAY_OPEN + 1 * S, 2, False, 101.0, 1.0),
        (DAY_OPEN + 50_000 * S, 3, True, 102.0, 1.0),   # strands ask101 → crossed to day end
    ])


@native
def test_assess_native_trailing_crossed_day_presence_outlives_trust():
    # present != valid at the day edge: crossed samples are PRESENT but invalid, so
    # lake_present_end_ts must reach day end while trusted_lake_end_ts stops at the crossing.
    # Pins the native presence-bounds wiring — a valid-mask-derived presence would end early —
    # and the trailing-partial plan shape end-to-end on both engines.
    kw = dict(day=DAY, k=1, seed_min_levels=1, cold_ab=False)
    py = qm.assess_lake_day(_trailing_crossed_lake_df(), _valid_seed(), engine="python", **kw)
    nat = qm.assess_lake_day(_trailing_crossed_lake_df(), _valid_seed(), engine="native",
                             price_scale=100, **kw)
    assert nat["classification"] == py["classification"] == qm.LAKE_PRESENT_DEGRADED
    assert nat["stitch_plan"] == py["stitch_plan"]
    assert nat["stitch_plan"]["fill_profile"] == "trailing_partial_fill"
    assert nat["quality"] == py["quality"]
    q = nat["quality"]
    assert q["lake_present_end_ts"] == DAY_OPEN + 86_400 * S    # presence reaches day end…
    assert q["trusted_lake_end_ts"] == DAY_OPEN + 50_000 * S    # …trust stops at the crossing


# ------------------------------------------------------------------- coinapi_fill_block (composer)
def test_fill_block_no_fill_and_no_verdict_days_carry_no_plan():
    lake_only = _day_grid_plan(0, day="2025-06-01")
    b = qm.coinapi_fill_block(qm.LAKE_USABLE, ["seed_accepted"], day="2025-06-01",
                              stitch_plan=lake_only)
    assert b["needs_fill"] is False and b["why"] == "lake_usable"
    assert b["fill_profile"] is None and b["fill_segments"] is None
    assert b["seams"] is None and b["seam_policy"] is None and b["full_day_reason"] is None
    for cls, reasons in ((qm.INCONCLUSIVE, ["no_seed_snapshots"]),
                         (qm.EXCLUDED, ["binance_gap"])):
        b = qm.coinapi_fill_block(cls, reasons, day="2025-06-01")
        assert b["needs_fill"] is None and b["fill_profile"] is None


def test_fill_block_missing_day_synthesizes_full_day_plan():
    b = qm.coinapi_fill_block(qm.MISSING_NEEDS_COINAPI, ["lake_book_delta_v2_absent"],
                              day="2024-12-05")
    assert b["needs_fill"] is True
    assert b["fill_profile"] == "full_day_fill"
    assert b["full_day_reason"] == "lake_book_delta_v2_absent"
    seg, = b["fill_segments"]
    assert seg["source"] == "coinapi" and seg["reason"] == "lake_book_delta_v2_absent"
    assert seg["start_iso"] == "2024-12-05T00:00:00Z" and seg["end_iso"] == "2024-12-06T00:00:00Z"
    assert b["seams"] == [] and b["seam_policy"]["seam_guard_s"] == 60.0


def test_fill_block_degraded_partial_day_keeps_the_mask_plan():
    plan = _day_grid_plan(50_000)
    b = qm.coinapi_fill_block(qm.LAKE_PRESENT_DEGRADED,
                              ["seed_accepted", "missing_book_fraction=0.5787>0.02"],
                              day="2025-01-07", stitch_plan=plan)
    assert b["needs_fill"] is True and b["why"] == "quality_over_usable_bar"
    assert b["fill_profile"] == "leading_partial_fill"
    assert b["full_day_reason"] is None
    assert b["fill_segments"] == plan["fill_segments"] and b["seams"] == plan["seams"]


def test_fill_block_degraded_day_with_lake_only_plan_routes_full_day():
    # A day-level bar failed (e.g. thin depth) but the top-of-book validity mask shows no fillable
    # window: no mask-supported narrower fill exists, so route the WHOLE day to CoinAPI.
    lake_only = _day_grid_plan(0, day="2025-06-01")
    b = qm.coinapi_fill_block(qm.LAKE_PRESENT_DEGRADED,
                              ["seed_accepted", "thin_depth_fraction=0.5000>0.1"],
                              day="2025-06-01", stitch_plan=lake_only)
    assert b["needs_fill"] is True
    assert b["fill_profile"] == "full_day_fill"
    assert b["full_day_reason"] == "quality_over_usable_bar"
    seg, = b["fill_segments"]
    assert (seg["start_ts"], seg["end_ts"]) == (lake_only["day_open_ts"], lake_only["day_end_ts"])


def test_fill_block_keeps_mask_full_day_reasons_verbatim():
    # A mask-derived FULL-DAY plan (Q2 rules 4-6) must pass through the composer untouched — its
    # own reason code (here lake_never_warmup_qualified), NOT the generic fallback code.
    ts = DAY_OPEN + np.arange(86_400, dtype=np.int64) * S
    valid = np.tile(np.array([True, True, False]), 28_800)   # runs of 2 < warmup_consecutive=3
    plan = plan_day_stitch(ts, valid, grid_ns=S, seed_accepted=True, seed_ts=DAY_OPEN,
                           seed_source_trusted=True, day="2025-06-01").as_dict()
    assert plan["full_day_reason"] == "lake_never_warmup_qualified"
    b = qm.coinapi_fill_block(qm.LAKE_PRESENT_DEGRADED,
                              ["seed_accepted", "crossed_rate_after=0.4000>0.01"],
                              day="2025-06-01", stitch_plan=plan)
    assert b["full_day_reason"] == "lake_never_warmup_qualified"
    seg, = b["fill_segments"]
    assert seg["reason"] == "lake_never_warmup_qualified"
    rep = qm.build_report([{"day": "2025-06-01", "classification": qm.LAKE_PRESENT_DEGRADED,
                            "reasons": ["seed_accepted", "crossed_rate_after=0.4000>0.01"],
                            "stitch_plan": plan}], meta={})
    assert rep["summary"]["coinapi_fill"]["full_day_reason_counts"] == {
        "lake_never_warmup_qualified": 1}


def test_stitch_coverage_invalid_runs_list_is_capped_with_full_count():
    # Q7: invalid_runs is capped like reseed_ts[:100]; n_invalid_runs keeps the full count.
    n = 1200
    ts = DAY_OPEN + np.arange(n, dtype=np.int64) * S
    bad = np.zeros(n, dtype=bool)
    bad[np.arange(0, n, 8)] = True                           # 150 isolated invalid samples
    frame = pd.DataFrame({"sample_ts": ts,
                          "bid_0_price": np.where(bad, np.nan, 100.0),
                          "ask_0_price": np.full(n, 101.0)})
    meta = {"seed_accepted": True, "seed_ts": int(ts[0])}
    _, cov = qm._stitch_and_coverage(frame, meta=meta, reasons=[], grid_ms=1000, day=DAY)
    assert cov["n_invalid_runs"] == 150
    assert len(cov["invalid_runs"]) == qm.INVALID_RUNS_CAP == 100
    assert cov["invalid_runs"][0] == [DAY_OPEN + 0 * S, DAY_OPEN + 1 * S]


def test_classify_thin_over_bar_emits_stable_reason_code():
    # Thin depth is the one degraded dimension the top-of-book validity mask cannot see, so the
    # classifier emits a stable code (the SEED_SOURCE_UNRELIABLE pattern) for the fill composer.
    meta = {"seed_accepted": True, "seed_reason": "ok", "thin_depth_fraction": 0.50,
            "snapshot_reason_codes": {"ok": 10}}
    lake_q = {"crossed_rate": 0.0, "missing_book_fraction": 0.0}
    cls, reasons = qm.classify_day(have_lake=True, meta=meta, lake_q=lake_q,
                                   thresholds=qm.THRESHOLDS)
    assert cls == qm.LAKE_PRESENT_DEGRADED
    assert qm.THIN_DEPTH_OVER_BAR in reasons
    # ...and a thin-clean degraded day does not carry the code
    meta2 = {**meta, "thin_depth_fraction": 0.0}
    lake_q2 = {"crossed_rate": 0.40, "missing_book_fraction": 0.0}
    _, reasons2 = qm.classify_day(have_lake=True, meta=meta2, lake_q=lake_q2,
                                  thresholds=qm.THRESHOLDS)
    assert qm.THIN_DEPTH_OVER_BAR not in reasons2


def test_fill_block_thin_failure_discards_a_partial_mask_plan():
    # Codex P2: a thin-degraded day that ALSO has a real gap must not keep its mask-planned Lake
    # span — the mask can't vouch for depth, so the whole day routes to CoinAPI.
    plan = _day_grid_plan(50_000)
    b = qm.coinapi_fill_block(
        qm.LAKE_PRESENT_DEGRADED,
        ["seed_accepted", "missing_book_fraction=0.5787>0.02", qm.THIN_DEPTH_OVER_BAR,
         "thin_depth_fraction=0.5000>0.1"],
        day="2025-01-07", stitch_plan=plan)
    assert b["fill_profile"] == "full_day_fill"
    assert b["full_day_reason"] == "quality_over_usable_bar"
    seg, = b["fill_segments"]
    assert (seg["start_ts"], seg["end_ts"]) == (plan["day_open_ts"], plan["day_end_ts"])


def test_thin_partial_day_end_to_end_routes_full_day_and_nulls_trust():
    # k=2 with a one-level book AND a leading gap: mask plans a leading partial fill, but the
    # thin-depth bar failed → report routes full-day and nulls trusted_lake_*.
    res = qm.assess_lake_day(_leading_partial_lake_df(), _late_seed(), day=DAY, k=2,
                             seed_min_levels=1)
    assert res["classification"] == qm.LAKE_PRESENT_DEGRADED
    assert qm.THIN_DEPTH_OVER_BAR in res["reasons"]
    assert res["stitch_plan"]["fill_profile"] == "leading_partial_fill"   # mask-level view intact
    rep = qm.build_report([res], meta={})
    cf = rep["days"][0]["coinapi_fill"]
    assert cf["fill_profile"] == "full_day_fill"
    assert cf["full_day_reason"] == "quality_over_usable_bar"
    assert rep["days"][0]["quality"]["trusted_lake_start_ts"] is None
    assert res["quality"]["trusted_lake_start_ts"] is not None            # caller record unmutated


def test_build_grid_and_cli_reject_non_divisor_grid_ms():
    # Codex P3: a grid_ms that does not divide the 24 h day truncates the grid, so mask-plan day
    # bounds would stop short of midnight while synthesized full-day plans use 24:00. Fail fast.
    with pytest.raises(ValueError, match="divide"):
        qm.build_grid(DAY, 7000)
    with pytest.raises(SystemExit):
        qm.parse_args(["--grid-ms", "7000"])
    assert len(qm.build_grid(DAY, 1000)) == 86_400
    assert len(qm.build_grid(DAY, 500)) == 172_800


def test_fill_block_crossed_source_without_a_mask_plan_falls_back_full_day():
    # Metrics-only (native-engine) records carry no stitch plan; the crossed-source fill decision
    # still gets a conservative full-day plan with the stable crossed-source reason.
    b = qm.coinapi_fill_block(qm.INCONCLUSIVE,
                              [qm.SEED_SOURCE_UNRELIABLE, "seed_source_crossed_frac=0.3751>0.05"],
                              day="2026-04-01")
    assert b["needs_fill"] is True
    assert b["fill_profile"] == "full_day_fill"
    assert b["full_day_reason"] == "crossed_seed_source"


def test_degraded_thin_day_end_to_end_routes_full_day_fill():
    # k=2 with a one-level book: thin-degraded classification, LAKE_ONLY mask plan, conservative
    # full-day fill in the stamped report block.
    res = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), day=DAY, k=2, seed_min_levels=1)
    assert res["classification"] == qm.LAKE_PRESENT_DEGRADED
    assert any("thin" in r for r in res["reasons"])
    assert res["stitch_plan"]["fill_profile"] == "lake_only"
    rep = qm.build_report([res], meta={})
    cf = rep["days"][0]["coinapi_fill"]
    assert cf["needs_fill"] is True
    assert cf["fill_profile"] == "full_day_fill"
    assert cf["full_day_reason"] == "quality_over_usable_bar"
    # No Lake coverage survives a full-day route (plan-doc definitions table): the override must
    # null the report's trusted_lake_* — presence/invalid-run facts stay, trust does not...
    q = rep["days"][0]["quality"]
    assert q["trusted_lake_start_ts"] is None and q["trusted_lake_end_ts"] is None
    assert q["lake_present_start_ts"] is not None and q["n_invalid_runs"] is not None
    # ...and the caller's record (the mask-level view, consistent with its lake_only plan) is
    # not mutated.
    assert res["quality"]["trusted_lake_start_ts"] is not None


# ------------------------------------------------------------------- report stamping + summary
def test_build_report_stamps_fill_plans_and_summary_counts():
    partial_plan = _day_grid_plan(50_000)
    days = [
        {"day": "2025-06-01", "classification": qm.LAKE_USABLE, "reasons": ["seed_accepted"]},
        {"day": "2024-12-05", "classification": qm.MISSING_NEEDS_COINAPI,
         "reasons": ["lake_book_delta_v2_absent"]},
        {"day": "2025-01-07", "classification": qm.LAKE_PRESENT_DEGRADED,
         "reasons": ["seed_accepted", "missing_book_fraction=0.5787>0.02"],
         "stitch_plan": partial_plan},
        {"day": "2024-08-05", "classification": qm.INCONCLUSIVE,
         "reasons": [qm.SEED_SOURCE_UNRELIABLE, "seed_source_crossed_frac=0.2878>0.05"]},
        {"day": "2025-03-03", "classification": qm.INCONCLUSIVE, "reasons": ["no_seed_snapshots"]},
        {"day": "2025-02-02", "classification": qm.EXCLUDED, "reasons": ["binance_gap"]},
    ]
    rep = qm.build_report(days, meta={})
    by = {r["day"]: r["coinapi_fill"] for r in rep["days"]}
    assert by["2025-01-07"]["fill_profile"] == "leading_partial_fill"
    assert by["2024-12-05"]["full_day_reason"] == "lake_book_delta_v2_absent"
    assert by["2024-08-05"]["full_day_reason"] == "crossed_seed_source"
    assert by["2025-06-01"]["fill_profile"] is None
    assert by["2025-03-03"]["fill_profile"] is None
    f = rep["summary"]["coinapi_fill"]
    assert f["needs_fill"] == ["2024-12-05", "2025-01-07", "2024-08-05"]   # existing list intact
    assert f["partial_fill"] == ["2025-01-07"]                             # Q7: ⊆ needs_fill
    assert f["fill_counts"] == {
        "needs_fill": 3, "full_day_fill": 2, "leading_partial_fill": 1,
        "trailing_partial_fill": 0, "internal_gap_fill": 0, "mixed_partial_fill": 0,
        "crossed_source_full_day": 1, "no_verdict": 1, "no_fill": 1, "not_in_scope": 1}
    assert f["full_day_reason_counts"] == {"lake_book_delta_v2_absent": 1,
                                           "crossed_seed_source": 1}
    # the internal stitch_plan key is consumed into coinapi_fill, never emitted per-day...
    assert all("stitch_plan" not in r for r in rep["days"])
    # ...and the caller's records are not mutated
    assert days[2]["stitch_plan"] is partial_plan and "coinapi_fill" not in days[2]


def test_report_with_fill_plans_is_strict_json(tmp_path):
    res = qm.assess_lake_day(_leading_partial_lake_df(), _late_seed(), day=DAY, k=1,
                             seed_min_levels=1)
    gap = qm.assess_lake_day(pd.DataFrame(), None, day=dt.date(2024, 12, 5), k=1)
    rep = qm.build_report([res, gap], meta={"k": 1, "thresholds": qm.THRESHOLDS.as_dict()})
    out = tmp_path / "quality_map_fill.json"
    qm.write_report(rep, str(out))
    txt = out.read_text()
    loaded = json.loads(txt)                                     # strict-JSON round trip
    assert "NaN" not in txt and "Infinity" not in txt
    cf = loaded["days"][0]["coinapi_fill"]
    assert cf["fill_profile"] == "leading_partial_fill"
    assert loaded["days"][0]["quality"]["invalid_runs"] == [[DAY_OPEN, DAY_OPEN + 50_000 * S]]
    assert loaded["days"][1]["coinapi_fill"]["fill_profile"] == "full_day_fill"
    assert loaded["summary"]["coinapi_fill"]["fill_counts"]["leading_partial_fill"] == 1


def test_quality_block_coverage_keys_are_schema_consistent():
    # Every quality block — assessed, missing, excluded, load-failed — carries the Q7 coverage keys.
    keys = {"lake_present_start_ts", "lake_present_end_ts", "trusted_lake_start_ts",
            "trusted_lake_end_ts", "n_invalid_runs", "invalid_runs"}
    assert keys <= set(qm._empty_quality_block(10, 1000))
    assert keys <= set(qm.excluded_result(DAY, ["x"])["quality"])
    assert keys <= set(qm.inconclusive_load_failure(DAY, "boom")["quality"])
    res = qm.assess_lake_day(_clean_lake_df(), _valid_seed(), day=DAY, k=1, seed_min_levels=1)
    assert keys <= set(res["quality"])


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
    # The whole quality block — incl. the Q7 coverage keys the native path now derives from its
    # compact coverage meta — and the stamped fill block must match the Python frame path exactly.
    assert nat["days"][0]["quality"] == py["days"][0]["quality"]
    assert nat["days"][0]["coinapi_fill"] == py["days"][0]["coinapi_fill"]


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
