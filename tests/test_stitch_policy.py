"""Synthetic tests for the partial-day / vendor-seam fill policy helpers
(docs/superpowers/plans/2026-07-02-partial-day-fill-policy.md).

These pin the normative segment-derivation algorithm: the vendor-switch boundary is the first
post-seed warmup-qualified sample (the parity gate's `max(lake_warmup_cutoff, seed_ts)` clamp,
generalized per span), crossed-seed-source days route to full-day CoinAPI fill even when partial
(the 2024-08-05 shape), sustained invalid runs become CoinAPI fill segments while short blips stay
masked samples, and label/feature windows never train across a seam or inside its guard band.
No vendor I/O anywhere — the backfill gate is untouched.
"""
import json

import numpy as np
import pandas as pd
import pytest

from recon.parity import lake_warmup_cutoff
from recon.stitch_policy import (
    COINAPI,
    DEFAULT_SEAM_POLICY,
    EXCLUDED,
    FULL_DAY_FILL,
    INTERNAL_GAP_FILL,
    LAKE,
    LAKE_ONLY,
    LEADING_PARTIAL_FILL,
    MIXED_PARTIAL_FILL,
    TRAILING_PARTIAL_FILL,
    SeamPolicy,
    feature_valid_mask,
    label_valid_mask,
    plan_day_stitch,
    seam_guard_mask,
    valid_mask_from_frame,
    vendor_source_at,
    warmup_qualified_ts,
    window_crosses_seam,
    window_vendor_sources,
)

S = 1_000_000_000  # ns per second; tests use a 1 s grid like the production 86,400-sample day

# Hand-checkable knobs: fill windows from ≥300 s outages, Lake islands must span ≥600 s.
POLICY = SeamPolicy(seam_guard_s=60.0, warmup_consecutive=3, fill_min_s=300.0,
                    min_lake_segment_s=600.0, span_invalid_max=0.01)


def _grid(n):
    return np.arange(n, dtype=np.int64) * S


def _plan(valid, *, seed_ts=0, seed_accepted=True, trusted=True, policy=POLICY, present=None,
          day=None):
    valid = np.asarray(valid, dtype=bool)
    return plan_day_stitch(_grid(len(valid)), valid, grid_ns=S, seed_accepted=seed_accepted,
                           seed_ts=seed_ts, seed_source_trusted=trusted, policy=policy,
                           present=present, day=day)


def _assert_partition(plan):
    """Segments are ordered, contiguous, and exactly cover [day_open, day_end)."""
    segs = plan.segments
    assert segs[0].start_ts == plan.day_open_ts
    assert segs[-1].end_ts == plan.day_end_ts
    for a, b in zip(segs, segs[1:]):
        assert a.end_ts == b.start_ts
        assert a.end_ts > a.start_ts
    assert segs[-1].end_ts > segs[-1].start_ts
    assert all(s.source in (LAKE, COINAPI, EXCLUDED) for s in segs)


# ------------------------------------------------------------------- 1. leading missing segment
def test_leading_missing_segment_yields_coinapi_then_lake():
    # 2025-01-07 shape: Lake resumes at 3600 s, seed lands at the resume, clean afterwards.
    valid = _grid(7200) >= 3600 * S
    plan = _plan(valid, seed_ts=3600 * S)
    _assert_partition(plan)
    assert [(s.source, s.start_ts, s.end_ts) for s in plan.segments] == [
        (COINAPI, 0, 3602 * S), (LAKE, 3602 * S, 7200 * S)]
    assert plan.segments[0].reason == "lake_missing_leading_segment"
    assert plan.segments[1].reason == "trusted_seeded_lake_reconstruction"
    assert plan.fill_profile == LEADING_PARTIAL_FILL
    assert plan.seams == (3602 * S,)
    assert plan.trusted_lake_start_ts == 3602 * S
    assert plan.trusted_lake_end_ts == 7200 * S


def test_boundary_is_the_warmup_qualified_sample_not_the_seed():
    # The 3rd consecutive valid sample at/after the seed IS the boundary sample (Lake side).
    valid = _grid(7200) >= 3600 * S
    plan = _plan(valid, seed_ts=3600 * S)
    assert plan.trusted_lake_start_ts == 3602 * S  # samples 3600,3601 are warm-up → CoinAPI side


# ------------------------------------------------------------------ 2. trailing missing segment
def test_trailing_missing_segment_yields_lake_then_coinapi():
    valid = _grid(7200) < 5400 * S  # Lake dies at 5400 s and stays dead
    plan = _plan(valid, seed_ts=0)
    _assert_partition(plan)
    assert [(s.source, s.start_ts, s.end_ts) for s in plan.segments] == [
        (EXCLUDED, 0, 2 * S), (LAKE, 2 * S, 5400 * S), (COINAPI, 5400 * S, 7200 * S)]
    assert plan.segments[0].reason == "leading_warmup_excluded"
    assert plan.segments[2].reason == "lake_missing_trailing_segment"
    assert plan.fill_profile == TRAILING_PARTIAL_FILL
    assert plan.seams == (2 * S, 5400 * S)


# ------------------------------------------------------- 3. internal gap and sub-threshold blip
def test_internal_gap_splits_lake_and_requalifies_after_the_gap():
    valid = np.ones(7200, dtype=bool)
    valid[1000:1400] = False        # 400 s outage ≥ fill_min_s → CoinAPI fill window
    valid[2000:2003] = False        # 3 s blip < fill_min_s → stays masked inside the Lake segment
    plan = _plan(valid, seed_ts=0)
    _assert_partition(plan)
    assert [(s.source, s.start_ts, s.end_ts) for s in plan.segments] == [
        (EXCLUDED, 0, 2 * S), (LAKE, 2 * S, 1000 * S),
        (COINAPI, 1000 * S, 1402 * S),  # gap + the 2-sample post-gap requalification prefix
        (LAKE, 1402 * S, 7200 * S)]
    assert plan.segments[2].reason == "lake_missing_internal_segment"
    assert plan.fill_profile == INTERNAL_GAP_FILL
    assert plan.seams == (2 * S, 1000 * S, 1402 * S)
    assert plan.trusted_lake_start_ts == 2 * S
    assert plan.trusted_lake_end_ts == 7200 * S


def test_short_blip_does_not_create_a_fill_segment():
    valid = np.ones(7200, dtype=bool)
    valid[3000:3003] = False
    plan = _plan(valid, seed_ts=0)
    assert [s.source for s in plan.segments] == [EXCLUDED, LAKE]
    assert plan.fill_profile == LAKE_ONLY


# ----------------------------------------------------------- 4. crossed seed source (2024-08-05)
def test_crossed_seed_source_routes_full_day_even_when_partial():
    valid = _grid(7200) >= 3600 * S  # partial AND crossed-source: crossed dominates
    plan = _plan(valid, seed_ts=3600 * S, trusted=False)
    _assert_partition(plan)
    assert [(s.source, s.start_ts, s.end_ts) for s in plan.segments] == [(COINAPI, 0, 7200 * S)]
    assert plan.full_day_reason == "crossed_seed_source"
    assert plan.fill_profile == FULL_DAY_FILL
    assert plan.seams == ()
    assert plan.trusted_lake_start_ts is None


# ------------------------------------------------------------------ 5. other full-day routings
def test_no_accepted_seed_routes_full_day():
    plan = _plan(np.ones(7200, dtype=bool), seed_accepted=False, seed_ts=None)
    assert plan.full_day_reason == "no_accepted_seed"
    assert plan.fill_profile == FULL_DAY_FILL


def test_never_warmup_qualified_routes_full_day():
    plan = _plan(np.zeros(7200, dtype=bool), seed_ts=0)
    assert plan.full_day_reason == "lake_never_warmup_qualified"
    assert plan.fill_profile == FULL_DAY_FILL


def test_trusted_span_too_short_routes_full_day():
    valid = np.zeros(7200, dtype=bool)
    valid[3600:3700] = True  # 98 s trusted island < min_lake_segment_s=600
    plan = _plan(valid, seed_ts=3600 * S)
    assert plan.full_day_reason == "lake_trusted_span_too_short"
    assert plan.fill_profile == FULL_DAY_FILL


def test_scattered_invalid_over_span_bar_routes_full_day():
    valid = np.ones(7200, dtype=bool)
    valid[np.arange(10, 7200, 50)] = False  # ~2% scattered blips > span_invalid_max=1%
    plan = _plan(valid, seed_ts=0)
    assert plan.full_day_reason == "quality_over_trusted_span"
    assert plan.fill_profile == FULL_DAY_FILL


# ------------------------------------------------------------------------------- 6. seed clamp
def test_valid_samples_before_the_seed_never_count_toward_warmup():
    plan = _plan(np.ones(7200, dtype=bool), seed_ts=1000 * S)
    assert [(s.source, s.start_ts, s.end_ts) for s in plan.segments] == [
        (COINAPI, 0, 1002 * S), (LAKE, 1002 * S, 7200 * S)]
    assert plan.fill_profile == LEADING_PARTIAL_FILL


# -------------------------------------------------------------------------------- 7. guard band
def test_seam_guard_mask_covers_both_sides_of_the_seam():
    ts = _grid(2000)
    mask = seam_guard_mask(ts, (1000 * S,), guard_ns=60 * S)
    assert not mask[939]
    assert mask[940] and mask[1059]  # [seam-60s, seam+60s)
    assert not mask[1060]
    assert mask.sum() == 120


def test_guard_mask_with_no_seams_masks_nothing():
    assert not seam_guard_mask(_grid(100), (), guard_ns=60 * S).any()


# ------------------------------------------------------- 8. windows crossing seams are excluded
def test_window_crosses_seam_boundary_semantics():
    seams = (1000 * S,)
    assert not window_crosses_seam(900 * S, 999 * S, seams)
    assert window_crosses_seam(900 * S, 1000 * S, seams)   # target lands on the far side
    assert window_crosses_seam(999 * S, 1100 * S, seams)
    assert not window_crosses_seam(1000 * S, 1100 * S, seams)  # boundary sample is right-side


def test_label_valid_mask_excludes_seam_crossing_and_guard_touching_labels():
    ts = _grid(2000)
    ok = label_valid_mask(ts, (1000 * S,), horizon_ns=60 * S, guard_ns=60 * S)
    # valid iff t+60s < 940s (clear of the guard) or t >= 1060s (past it)
    assert ok[879] and not ok[880]
    assert not ok[1059] and ok[1060]


def test_feature_valid_mask_excludes_lookback_windows_crossing_the_seam():
    ts = _grid(2000)
    ok = feature_valid_mask(ts, (1000 * S,), lookback_ns=120 * S, guard_ns=60 * S)
    # valid iff t < 940s or t-120s >= 1060s
    assert ok[939] and not ok[940]
    assert not ok[1179] and ok[1180]


def test_masks_are_all_true_when_there_is_no_seam():
    ts = _grid(100)
    assert label_valid_mask(ts, (), horizon_ns=60 * S, guard_ns=60 * S).all()
    assert feature_valid_mask(ts, (), lookback_ns=60 * S, guard_ns=60 * S).all()


# ------------------------------------------------------------------------------ 9. no-lookahead
def test_boundary_uses_only_information_at_or_before_itself():
    ts, valid = _grid(7200), _grid(7200) >= 3600 * S
    b = warmup_qualified_ts(ts, valid, seed_ts=3600 * S, warmup_consecutive=3)
    assert b == 3602 * S
    # Truncating everything after the boundary sample leaves the boundary unchanged.
    i = int(np.searchsorted(ts, b)) + 1
    assert warmup_qualified_ts(ts[:i], valid[:i], seed_ts=3600 * S, warmup_consecutive=3) == b
    # Corrupting everything after the boundary sample leaves the boundary unchanged.
    corrupted = valid.copy()
    corrupted[i:] = False
    assert warmup_qualified_ts(ts, corrupted, seed_ts=3600 * S, warmup_consecutive=3) == b


def test_trusted_start_is_none_when_the_book_never_sustains():
    valid = np.tile([True, True, False], 100)  # runs of 2 < warmup_consecutive=3
    assert warmup_qualified_ts(_grid(300), valid, seed_ts=0, warmup_consecutive=3) is None


def test_boundary_is_a_conservative_refinement_of_the_parity_clamp():
    # A book valid BEFORE the seed: the parity clamp lets the pre-seed warmup run stand and
    # clamps to the seed (1000 s); the policy boundary restarts the run at the seed (1002 s).
    # The divergence is deliberate and always conservative: boundary >= max(cutoff, seed_ts).
    n, seed = 1100, 1000 * S
    ts, valid = _grid(n), np.ones(n, dtype=bool)
    frame = pd.DataFrame({"sample_ts": ts,
                          "bid_0_price": np.full(n, 100.0), "ask_0_price": np.full(n, 101.0)})
    clamp = max(lake_warmup_cutoff(frame, min_consecutive=3, min_levels_per_side=1), seed)
    b = warmup_qualified_ts(ts, valid, seed_ts=seed, warmup_consecutive=3)
    assert clamp == 1000 * S
    assert b == 1002 * S
    assert b >= clamp


# ---------------------------------------------------- 10. partition, JSON, and grid invariants
def test_segments_partition_every_scenario_and_plan_json_round_trips():
    scenarios = [
        _plan(_grid(7200) >= 3600 * S, seed_ts=3600 * S),                   # leading
        _plan(_grid(7200) < 5400 * S, seed_ts=0),                           # trailing
        _plan(_grid(7200) >= 3600 * S, seed_ts=3600 * S, trusted=False),    # full-day
        _plan(np.ones(7200, dtype=bool), seed_ts=0, day="2025-01-07"),      # lake_only
    ]
    for plan in scenarios:
        _assert_partition(plan)
        d = plan.as_dict()
        assert json.loads(json.dumps(d, allow_nan=False)) == d
        assert d["seam_policy"]["seam_guard_s"] == 60.0
        assert {"fill_profile", "fill_segments", "seams", "trusted_lake_start_ts"} <= set(d)


def test_mixed_profile_when_leading_and_internal_fills_coexist():
    valid = np.ones(7200, dtype=bool)
    valid[:1000] = False
    valid[3000:3400] = False
    plan = _plan(valid, seed_ts=1000 * S)
    assert plan.fill_profile == MIXED_PARTIAL_FILL
    assert [s.source for s in plan.segments] == [COINAPI, LAKE, COINAPI, LAKE]


def test_clean_day_is_lake_only_with_a_tiny_excluded_warmup_window():
    plan = _plan(np.ones(7200, dtype=bool), seed_ts=0)
    assert [(s.source, s.start_ts, s.end_ts) for s in plan.segments] == [
        (EXCLUDED, 0, 2 * S), (LAKE, 2 * S, 7200 * S)]
    assert plan.fill_profile == LAKE_ONLY


def test_irregular_grid_is_rejected():
    ts = np.array([0, S, 3 * S], dtype=np.int64)  # never silently compact a grid
    with pytest.raises(ValueError, match="regular full-day grid"):
        plan_day_stitch(ts, np.ones(3, dtype=bool), grid_ns=S, seed_accepted=True, seed_ts=0,
                        seed_source_trusted=True)


def test_degenerate_grid_step_is_rejected():
    flat = np.array([5 * S, 5 * S, 5 * S], dtype=np.int64)
    with pytest.raises(ValueError, match="positive step"):
        plan_day_stitch(flat, np.ones(3, dtype=bool), grid_ns=0, seed_accepted=True, seed_ts=0,
                        seed_source_trusted=True)
    with pytest.raises(ValueError, match="positive step"):  # n==1 must not skip the check
        plan_day_stitch(np.array([0], dtype=np.int64), np.ones(1, dtype=bool), grid_ns=-1,
                        seed_accepted=True, seed_ts=0, seed_source_trusted=True)


def test_present_mask_length_mismatch_is_rejected():
    with pytest.raises(ValueError, match="present mask"):
        plan_day_stitch(_grid(7200), np.ones(7200, dtype=bool), grid_ns=S, seed_accepted=True,
                        seed_ts=0, seed_source_trusted=True,
                        present=np.ones(10, dtype=bool))


def test_present_mask_populates_lake_present_span():
    valid = _grid(7200) >= 3600 * S
    plan = _plan(valid, seed_ts=3600 * S, present=valid)
    assert plan.lake_present_start_ts == 3600 * S
    assert plan.lake_present_end_ts == 7200 * S


def test_segment_dicts_carry_iso_timestamps():
    plan = _plan(np.ones(7200, dtype=bool), seed_ts=0)
    seg0 = plan.as_dict()["fill_segments"][0]
    assert seg0["start_iso"] == "1970-01-01T00:00:00Z"
    assert seg0["start_ts"] == 0


# ---------------------------------------------------------------------- 11. shared valid predicate
def test_valid_mask_from_frame_matches_lake_warmup_cutoff():
    nan = float("nan")
    frame = pd.DataFrame({
        "sample_ts": _grid(7),
        "bid_0_price": [100.0, 101.0, 100.0, 100.0, 100.0, 100.0, 100.0],
        "bid_1_price": [99.0, 100.0, 99.0, nan, 99.0, 99.0, 99.0],
        "ask_0_price": [nan, 100.5, 101.0, 101.0, 101.0, 101.0, 101.0],
        "ask_1_price": [nan, 101.5, 102.0, 102.0, 102.0, 102.0, 102.0],
    })
    # row0 one-sided, row1 crossed, row3 thin at min_levels_per_side=2
    for min_levels in (1, 2):
        mask = valid_mask_from_frame(frame, min_levels_per_side=min_levels)
        run, cutoff = 0, None
        for i, good in enumerate(mask):
            run = run + 1 if good else 0
            if run >= 3:
                cutoff = int(frame["sample_ts"].iloc[i])
                break
        assert cutoff == lake_warmup_cutoff(frame, min_consecutive=3,
                                            min_levels_per_side=min_levels)
    assert list(valid_mask_from_frame(frame, min_levels_per_side=2)) == [
        False, False, True, False, True, True, True]


# ------------------------------------------------------------------------- per-sample provenance
def test_vendor_source_at_maps_samples_to_their_segment():
    valid = np.ones(7200, dtype=bool)
    valid[1000:1400] = False
    plan = _plan(valid, seed_ts=0)
    ts = np.array([0, 2 * S, 999 * S, 1000 * S, 1401 * S, 1402 * S, 7199 * S], dtype=np.int64)
    assert vendor_source_at(ts, plan.segments) == [
        EXCLUDED, LAKE, LAKE, COINAPI, COINAPI, LAKE, LAKE]


def test_vendor_source_at_rejects_out_of_day_timestamps():
    plan = _plan(np.ones(7200, dtype=bool), seed_ts=0)
    for bad in (-1, 7200 * S):
        with pytest.raises(ValueError, match="outside the day"):
            vendor_source_at(np.array([bad], dtype=np.int64), plan.segments)


def test_window_vendor_sources_pins_the_single_vendor_rule():
    valid = np.ones(7200, dtype=bool)
    valid[1000:1400] = False
    segs = _plan(valid, seed_ts=0).segments  # excluded|lake|coinapi|lake
    assert window_vendor_sources(100 * S, 900 * S, segs) == {LAKE}
    assert window_vendor_sources(1100 * S, 1300 * S, segs) == {COINAPI}
    assert window_vendor_sources(900 * S, 1100 * S, segs) == {LAKE, COINAPI}  # crosses the seam
    assert window_vendor_sources(0, 1 * S, segs) == {EXCLUDED}  # no vendor behind it
    assert window_vendor_sources(1 * S, 3 * S, segs) == {EXCLUDED, LAKE}
    # Touching a segment's half-open end does NOT pull in the next segment.
    assert window_vendor_sources(100 * S, 999 * S, segs) == {LAKE}


# -------------------------------------------------------------- defaults and threshold semantics
def test_default_seam_policy_matches_the_documented_defaults():
    assert DEFAULT_SEAM_POLICY.as_dict() == {
        "seam_guard_s": 60.0, "warmup_consecutive": 3, "fill_min_s": 300.0,
        "min_lake_segment_s": 3600.0, "span_invalid_max": 0.01,
        "exclude_labels_crossing_seam": True, "exclude_features_crossing_seam": True}
    # warmup_consecutive must track the parity gate's --warmup-consecutive default (3).
    assert DEFAULT_SEAM_POLICY.warmup_consecutive == 3


def test_fill_min_is_inclusive_a_run_exactly_at_the_bar_fills():
    lenient = SeamPolicy(seam_guard_s=60.0, warmup_consecutive=3, fill_min_s=300.0,
                         min_lake_segment_s=600.0, span_invalid_max=0.20)
    at_bar = np.ones(7200, dtype=bool)
    at_bar[1000:1300] = False  # exactly 300 s >= fill_min_s → CoinAPI window
    plan = _plan(at_bar, seed_ts=0, policy=lenient)
    assert (COINAPI, 1000 * S) in [(s.source, s.start_ts) for s in plan.segments]
    below = np.ones(7200, dtype=bool)
    below[1000:1299] = False   # 299 s < fill_min_s → masked blip, no fill window
    plan = _plan(below, seed_ts=0, policy=lenient)
    assert [s.source for s in plan.segments] == [EXCLUDED, LAKE]


def test_min_lake_segment_is_inclusive_an_island_exactly_at_the_bar_survives():
    valid = np.zeros(7200, dtype=bool)
    valid[1000:1602] = True  # trusted portion [1002, 1602) is exactly 600 s
    plan = _plan(valid, seed_ts=1000 * S)
    assert (LAKE, 1002 * S, 1602 * S) in [(s.source, s.start_ts, s.end_ts)
                                          for s in plan.segments]
    shorter = np.zeros(7200, dtype=bool)
    shorter[1000:1601] = True  # 599 s < min_lake_segment_s → dropped → full-day
    plan = _plan(shorter, seed_ts=1000 * S)
    assert plan.full_day_reason == "lake_trusted_span_too_short"


def test_span_invalid_bar_is_strict_exactly_at_threshold_stays_partial():
    # Lake span [2 s, 7202 s) has 7,200 samples; 72 scattered blips = exactly 1% — strict >
    # keeps the day (the quality-map inclusive-usable convention), one more blip routes it.
    valid = np.ones(7202, dtype=bool)
    blips = np.arange(100, 7300, 100)[:72]
    valid[blips] = False
    assert _plan(valid, seed_ts=0).fill_profile == LAKE_ONLY
    valid[50] = False  # 73/7200 > 1%
    assert _plan(valid, seed_ts=0).full_day_reason == "quality_over_trusted_span"


def test_dropped_island_reports_the_surviving_segment_not_the_first_qualification():
    # A 98 s qualified island (dropped by min_lake_segment_s=600) followed by a long span:
    # plan.trusted_lake_start_ts is the SURVIVING segment's start, while the raw boundary
    # primitive still returns the island's qualification ts — two different, documented values.
    valid = np.zeros(7200, dtype=bool)
    valid[1000:1100] = True
    valid[2000:7200] = True
    plan = _plan(valid, seed_ts=1000 * S)
    assert [(s.source, s.start_ts, s.end_ts) for s in plan.segments] == [
        (COINAPI, 0, 2002 * S), (LAKE, 2002 * S, 7200 * S)]
    assert plan.trusted_lake_start_ts == 2002 * S
    assert warmup_qualified_ts(_grid(7200), valid, seed_ts=1000 * S,
                               warmup_consecutive=3) == 1002 * S
