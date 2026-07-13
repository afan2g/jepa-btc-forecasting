"""Causal forward-return + triple-barrier labels (issue #79 / plan §B, §D, §E, T5).

Covers: the P0 = true-mid-at-t_event anchor discipline (never the observable
target_read_ts read), hand-computed positive/negative/no-hit resolutions off the
causal true-mid path, exact price-touch and horizon-endpoint boundaries, actual
first-hit t_barrier vs the nominal horizon, per-horizon 2s/10s/60s isolation,
trailing/as-of EWMA barrier widths (hand-computed, parameter-persisted, immune to
post-t_event returns), same-time tie coalescing, fail-closed contract violations
(non-monotone paths/anchors, duplicate decision keys, invalid or mismatching P0,
end-of-partition refusal), per-row rejections (insufficient history, degenerate
width), deterministic ordering/rebuilds, and streaming laziness."""
import math
from typing import NamedTuple

import pytest

from data.labels import (
    DEFAULT_HORIZONS,
    ESTIMATOR,
    REJECT_DEGENERATE_WIDTH,
    REJECT_INSUFFICIENT_HISTORY,
    BarrierParams,
    LabelAnchor,
    LabelRejection,
    LabelRow,
    anchor_from_bar_reads,
    triple_barrier_labels,
    validate_barrier_params,
)

G = 1_000_000_000  # 1 second in integer nanoseconds
T0 = 100 * G       # canonical anchor decision time for the fixtures


def flat_warmup(t_event=T0, *, n=4, step=G, mid=100.0):
    """n+1 true-mid path points ending exactly at (t_event, mid): n zero returns,
    so the EWMA vol is exactly 0 and the barrier width is exactly
    width_mult * vol_floor_bps — float-exact barrier prices for boundary tests."""
    return [(t_event - (n - i) * step, mid) for i in range(n + 1)]


def params(**over):
    kw = dict(halflife_ns=10 * G, min_returns=2, width_mult=1.0, vol_floor_bps=50.0)
    kw.update(over)
    return BarrierParams(**kw)


def run(path, anchors, *, coverage_end=T0 + 60 * G, **over):
    return list(triple_barrier_labels(path, anchors, params=params(**over),
                                      coverage_end_ns=coverage_end))


# ------------------------------------------------------------ core resolutions

def test_default_horizon_ladder_is_2s_10s_60s():
    assert dict(DEFAULT_HORIZONS) == {"2s": 2 * G, "10s": 10 * G, "60s": 60 * G}


def test_up_move_labels_plus_one_at_the_actual_first_hit():
    path = flat_warmup() + [(T0 + 1 * G, 100.2), (T0 + 3 * G, 100.6),
                            (T0 + 4 * G, 101.5)]
    r2, r10, r60 = run(path, [(T0, 100.0)])
    # 2s: only the +20 bps point is inside (T0, T0+2s] -> vertical, return kept
    assert isinstance(r2, LabelRow)
    assert (r2.horizon, r2.label, r2.t_barrier) == ("2s", 0, T0 + 2 * G)
    assert r2.y_fwd_bps == pytest.approx(20.0)
    # 10s/60s: the +60 bps point at T0+3s is the FIRST hit; the larger move at
    # T0+4s must not steal t_barrier
    for r, tag in ((r10, "10s"), (r60, "60s")):
        assert (r.horizon, r.label, r.t_barrier) == (tag, 1, T0 + 3 * G)
        assert r.y_fwd_bps == pytest.approx(60.0)
        assert r.p0 == 100.0
        assert r.width_bps == 50.0            # vol 0, floor 50, mult 1: exact
    assert all(isinstance(r.label, int) and isinstance(r.t_barrier, int)
               for r in (r2, r10, r60))


def test_down_move_labels_minus_one():
    path = flat_warmup() + [(T0 + 2 * G, 99.4)]
    r2, r10, r60 = run(path, [(T0, 100.0)])
    # the -60 bps point sits exactly ON the 2s horizon end: window is inclusive
    for r in (r2, r10, r60):
        assert (r.label, r.t_barrier) == (-1, T0 + 2 * G)
        assert r.y_fwd_bps == pytest.approx(-60.0)


def test_no_hit_keeps_the_realized_vertical_return_and_labels_zero():
    path = flat_warmup() + [(T0 + 1 * G, 100.2), (T0 + 5 * G, 99.8),
                            (T0 + 30 * G, 100.3)]
    r2, r10, r60 = run(path, [(T0, 100.0)])
    assert [r.label for r in (r2, r10, r60)] == [0, 0, 0]
    assert [r.t_barrier for r in (r2, r10, r60)] == [T0 + 2 * G, T0 + 10 * G,
                                                     T0 + 60 * G]
    assert r2.y_fwd_bps == pytest.approx(20.0)    # as-of mid at T0+2s = 100.2
    assert r10.y_fwd_bps == pytest.approx(-20.0)  # as-of mid at T0+10s = 99.8
    assert r60.y_fwd_bps == pytest.approx(30.0)   # as-of mid at T0+60s = 100.3


def test_quiet_window_resolves_vertical_with_exactly_zero_return():
    r2, r10, r60 = run(flat_warmup(), [(T0, 100.0)])
    for r, h in ((r2, 2 * G), (r10, 10 * G), (r60, 60 * G)):
        assert (r.y_fwd_bps, r.label, r.t_barrier) == (0.0, 0, T0 + h)


# ------------------------------------------------- exact price/time boundaries

def test_exact_price_touch_is_a_hit_and_just_inside_is_not():
    # width_mult * vol_floor = 50 bps of P0=100.0: barriers exactly 100.5/99.5
    up = flat_warmup() + [(T0 + 3 * G, 100.5)]
    (_, r10, _) = run(up, [(T0, 100.0)])
    assert (r10.label, r10.t_barrier) == (1, T0 + 3 * G)
    assert r10.y_fwd_bps == 50.0                  # float-exact at this fixture
    down = flat_warmup() + [(T0 + 3 * G, 99.5)]
    (_, r10, _) = run(down, [(T0, 100.0)])
    assert (r10.label, r10.y_fwd_bps) == (-1, -50.0)
    inside = flat_warmup() + [(T0 + 3 * G, 100.4)]
    (_, r10, _) = run(inside, [(T0, 100.0)])
    assert (r10.label, r10.t_barrier) == (0, T0 + 10 * G)
    assert r10.y_fwd_bps == pytest.approx(40.0)


def test_hit_exactly_at_the_horizon_end_is_horizontal_not_vertical():
    path = flat_warmup() + [(T0 + 2 * G, 100.6)]
    (r2, _, _) = run(path, [(T0, 100.0)])
    assert (r2.label, r2.t_barrier) == (1, T0 + 2 * G)
    assert r2.y_fwd_bps == pytest.approx(60.0)


def test_breach_one_ns_after_the_horizon_end_resolves_vertical():
    path = flat_warmup() + [(T0 + 2 * G + 1, 100.6)]
    (r2, r10, _) = run(path, [(T0, 100.0)])
    assert (r2.label, r2.t_barrier, r2.y_fwd_bps) == (0, T0 + 2 * G, 0.0)
    assert (r10.label, r10.t_barrier) == (1, T0 + 2 * G + 1)


# --------------------------------------------------------- horizon isolation

def test_horizons_resolve_independently_on_one_path():
    path = flat_warmup() + [(T0 + 1 * G, 100.2), (T0 + 5 * G, 100.7),
                            (T0 + 20 * G, 99.2)]
    r2, r10, r60 = run(path, [(T0, 100.0)])
    assert (r2.label, r2.t_barrier) == (0, T0 + 2 * G)      # no 2s breach
    assert (r10.label, r10.t_barrier) == (1, T0 + 5 * G)    # +70 bps first
    assert (r60.label, r60.t_barrier) == (1, T0 + 5 * G)    # NOT the later -80


def test_first_hit_is_by_time_not_magnitude_or_direction():
    path = flat_warmup() + [(T0 + 3 * G, 99.4), (T0 + 6 * G, 102.0)]
    (_, r10, r60) = run(path, [(T0, 100.0)])
    for r in (r10, r60):
        assert (r.label, r.t_barrier) == (-1, T0 + 3 * G)
        assert r.y_fwd_bps == pytest.approx(-60.0)


# ------------------------------------------------------- rows/ordering per key

def test_one_row_per_t_event_horizon_in_deterministic_ladder_order():
    t1 = T0 + 2 * G
    path = flat_warmup() + [(T0 + 1 * G, 100.2), (t1, 100.3),
                            (t1 + 30 * G, 100.35)]
    rows = run(path, [(T0, 100.0), (t1, 100.3)], coverage_end=t1 + 60 * G)
    keys = [(r.t_event, r.horizon) for r in rows]
    assert keys == [(T0, "2s"), (T0, "10s"), (T0, "60s"),
                    (t1, "2s"), (t1, "10s"), (t1, "60s")]
    assert len(set(keys)) == len(keys)


def test_horizon_emission_order_is_by_duration_not_dict_insertion():
    shuffled = {"60s": 60 * G, "2s": 2 * G, "10s": 10 * G}
    rows = run(flat_warmup(), [(T0, 100.0)], horizons=shuffled)
    assert [r.horizon for r in rows] == ["2s", "10s", "60s"]


# ----------------------------------------------------------- EWMA barrier width

def test_ewma_width_matches_the_hand_computed_time_decay_form():
    # returns: +50 bps at T0-10s (float-exact), then (101.505-100.5)/100.5*1e4
    # ~ +100 bps at T0; the decay for a 10s gap at halflife 10s is exactly 0.5:
    #   S = 0.5*50^2 + r2^2,  W = 0.5 + 1,  width = width_mult * sqrt(S/W)
    path = [(T0 - 20 * G, 100.0), (T0 - 10 * G, 100.5), (T0, 101.505)]
    (row,) = run(path, [(T0, 101.505)], horizons={"2s": 2 * G},
                 min_returns=2, vol_floor_bps=0.0, width_mult=2.0)
    r2 = (101.505 - 100.5) / 100.5 * 1e4
    expected = 2.0 * math.sqrt((0.5 * 50.0 ** 2 + r2 ** 2) / 1.5)
    assert row.width_bps == pytest.approx(expected, rel=1e-12)
    # a plain unweighted mean of squares is measurably different: decay bites
    unweighted = 2.0 * math.sqrt((50.0 ** 2 + r2 ** 2) / 2.0)
    assert abs(row.width_bps - unweighted) > 1.0


# -------------------------------------------------------- parameter persistence

def test_params_as_dict_persists_every_estimator_parameter():
    p = params()
    d = p.as_dict()
    assert d == {
        "estimator": ESTIMATOR,
        "horizons": {"2s": 2 * G, "10s": 10 * G, "60s": 60 * G},
        "halflife_ns": 10 * G,
        "min_returns": 2,
        "width_mult": 1.0,
        "vol_floor_bps": 50.0,
    }
    d["horizons"]["2s"] = 1            # the dict is a copy: params stay frozen
    assert DEFAULT_HORIZONS["2s"] == 2 * G


def test_invalid_params_fail_closed():
    good = dict(halflife_ns=10, min_returns=1, width_mult=1.0, vol_floor_bps=0.0,
                horizons={"h": 10})
    validate_barrier_params(BarrierParams(**good))
    bad = [
        dict(good, horizons={}),
        dict(good, horizons={"": 10}),
        dict(good, horizons={1: 10}),
        dict(good, horizons={"h": 0}),
        dict(good, horizons={"h": -5}),
        dict(good, horizons={"h": 1.5}),
        dict(good, horizons={"a": 10, "b": 10}),  # two tags, one physical rung
        dict(good, halflife_ns=0),
        dict(good, halflife_ns=-1),
        dict(good, halflife_ns=1.5),
        dict(good, halflife_ns=True),
        dict(good, min_returns=0),
        dict(good, min_returns=1.5),
        dict(good, width_mult=0.0),
        dict(good, width_mult=-1.0),
        dict(good, width_mult=float("nan")),
        dict(good, vol_floor_bps=-1.0),
        dict(good, vol_floor_bps=float("inf")),
    ]
    for kw in bad:
        with pytest.raises(ValueError):
            validate_barrier_params(BarrierParams(**kw))


def test_labeler_validates_params_and_coverage_eagerly_before_iteration():
    with pytest.raises(ValueError):
        triple_barrier_labels([], [], params=params(width_mult=0.0),
                              coverage_end_ns=100)
    with pytest.raises(ValueError):
        triple_barrier_labels([], [], params=params(), coverage_end_ns=1.5)


# ---------------------------------------------- trailing-vol no-lookahead (§J)

def wavy_warmup():
    """Warm-up with real (non-zero) returns so the EWMA vol is exercised."""
    return [(T0 - 30 * G, 100.0), (T0 - 20 * G, 100.5), (T0 - 10 * G, 99.9),
            (T0, 100.2)]


def test_mutating_returns_strictly_after_t_event_cannot_change_the_width():
    future = [(T0 + 3 * G, 100.9), (T0 + 8 * G, 99.0), (T0 + 30 * G, 101.5)]
    mutated = [(T0 + 3 * G, 99.1), (T0 + 8 * G, 103.0), (T0 + 30 * G, 97.5)]
    kw = dict(vol_floor_bps=0.0, min_returns=2)
    base = run(wavy_warmup() + future, [(T0, 100.2)], **kw)
    mut = run(wavy_warmup() + mutated, [(T0, 100.2)], **kw)
    # the as-of barrier width is EXACTLY invariant to post-t_event returns...
    assert [r.width_bps for r in base] == [r.width_bps for r in mut]
    # ...while the label legitimately depends on the future path (plan round 19:
    # assert on width, never on label invariance)
    assert base[1].label == 1 and mut[1].label == -1


def test_the_return_ending_exactly_at_t_event_feeds_the_trailing_ewma():
    # plan round 20: the mutation exclusion window is STRICTLY after t_event —
    # changing the as-of return that ends AT t_event must move the width
    kw = dict(vol_floor_bps=0.0, min_returns=2)
    (r_base, *_) = run(wavy_warmup(), [(T0, 100.2)], **kw)
    moved = wavy_warmup()[:-1] + [(T0, 100.4)]
    (r_moved, *_) = run(moved, [(T0, 100.4)], **kw)
    assert r_base.width_bps != r_moved.width_bps


def test_mutation_strictly_after_the_horizon_leaves_rows_identical():
    base = flat_warmup() + [(T0 + 5 * G, 100.2), (T0 + 61 * G, 100.3)]
    mutated = flat_warmup() + [(T0 + 5 * G, 100.2), (T0 + 61 * G, 55.5)]
    cov = T0 + 61 * G
    assert run(base, [(T0, 100.0)], coverage_end=cov) == \
        run(mutated, [(T0, 100.0)], coverage_end=cov)


# ------------------------------------------------------- per-row rejections

def test_insufficient_trailing_history_rejects_each_horizon():
    rows = run(flat_warmup(n=1), [(T0, 100.0)], min_returns=2)
    assert [(type(r), r.t_event, r.horizon, r.reason) for r in rows] == [
        (LabelRejection, T0, "2s", REJECT_INSUFFICIENT_HISTORY),
        (LabelRejection, T0, "10s", REJECT_INSUFFICIENT_HISTORY),
        (LabelRejection, T0, "60s", REJECT_INSUFFICIENT_HISTORY),
    ]
    # history accumulates: a later anchor with enough trailing returns labels
    path = flat_warmup(n=1) + [(T0 + 1 * G, 100.0)]
    t1 = T0 + 2 * G
    rows = run(path, [(T0, 100.0), (t1, 100.0)], min_returns=2,
               coverage_end=t1 + 60 * G)
    assert [type(r) for r in rows] == [LabelRejection] * 3 + [LabelRow] * 3


def test_flat_history_with_zero_vol_floor_rejects_degenerate_width():
    rows = run(flat_warmup(), [(T0, 100.0)], vol_floor_bps=0.0)
    assert [(r.horizon, r.reason) for r in rows] == [
        ("2s", REJECT_DEGENERATE_WIDTH), ("10s", REJECT_DEGENERATE_WIDTH),
        ("60s", REJECT_DEGENERATE_WIDTH)]
    # the SAME world with a positive floor emits usable rows (the floor is the
    # explicit escape hatch for flat warm-up regimes)
    assert all(isinstance(r, LabelRow)
               for r in run(flat_warmup(), [(T0, 100.0)], vol_floor_bps=50.0))


# ------------------------------------------------- end-of-partition refusal

def test_end_of_partition_refusal():
    with pytest.raises(ValueError, match="future support"):
        run(flat_warmup(), [(T0, 100.0)], coverage_end=T0 + 60 * G - 1)
    # the exact boundary is decidable: window end == coverage_end passes
    assert len(run(flat_warmup(), [(T0, 100.0)], coverage_end=T0 + 60 * G)) == 3


def test_refusal_happens_before_opening_the_path_stream_at_all():
    # mirrors the §J partition fixture: the unsafe row must be refused WITHOUT
    # consuming a single path element (deep-review P2: T9 may hand over a lazy
    # chained iterator whose next unread element opens the adjacent partition's
    # source — the refusal must fire before the stream is touched)
    consumed = []

    def tracked():
        for p in flat_warmup() + [(T0 + 5 * G, 100.2)]:
            consumed.append(p)
            yield p

    gen = triple_barrier_labels(tracked(), [(T0, 100.0)], params=params(),
                                coverage_end_ns=T0 + 2 * G)  # 60s cannot fit
    with pytest.raises(ValueError, match="future support"):
        next(gen)
    assert consumed == []                      # the stream was never opened


# ------------------------------------------------- fail-closed contract rails

def test_non_monotone_path_fails_closed():
    path = [(T0 - 2 * G, 100.0), (T0 - 3 * G, 100.0), (T0, 100.0)]
    with pytest.raises(ValueError, match="order"):
        run(path, [(T0, 100.0)])


def test_invalid_path_mids_fail_closed():
    for bad in (0.0, -5.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="mid"):
            run(flat_warmup()[:-1] + [(T0, bad)], [(T0, 100.0)])


def test_malformed_path_points_fail_closed():
    with pytest.raises(ValueError, match="path point"):
        run([(T0 - G,)], [(T0, 100.0)])
    with pytest.raises(ValueError, match="ts"):
        run([(1.5, 100.0), (T0, 100.0)], [(T0, 100.0)])


def test_duplicate_or_regressing_decision_keys_fail_closed():
    with pytest.raises(ValueError, match="increas"):
        run(flat_warmup(), [(T0, 100.0), (T0, 100.0)],
            coverage_end=T0 + 61 * G)
    with pytest.raises(ValueError, match="increas"):
        run(flat_warmup(), [(T0, 100.0), (T0 - G, 100.0)],
            coverage_end=T0 + 61 * G)


def test_invalid_p0_fails_closed():
    for bad in (0.0, -100.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="p0"):
            run(flat_warmup(), [(T0, bad)])


def test_p0_must_match_the_true_path_mid_at_t_event():
    with pytest.raises(ValueError, match="does not equal the true path mid"):
        run(flat_warmup(), [(T0, 100.3)])


def test_anchor_before_any_path_point_fails_closed():
    with pytest.raises(ValueError, match="no path point"):
        run([(T0 + G, 100.0), (T0 + 2 * G, 100.0)], [(T0, 100.0)])


# ------------------------------------------------------- same-time tie pinning

def test_same_time_ties_coalesce_to_the_final_state():
    # an intermediate breach at ts with the final same-ts state back inside is
    # NOT a hit: the path state AT an instant is the last point in input order
    path = flat_warmup() + [(T0 + 3 * G, 100.9), (T0 + 3 * G, 100.2)]
    (_, r10, _) = run(path, [(T0, 100.0)])
    assert (r10.label, r10.t_barrier) == (0, T0 + 10 * G)
    assert r10.y_fwd_bps == pytest.approx(20.0)
    # reverse order: the final same-ts state IS the breach -> hit at that ts
    path = flat_warmup() + [(T0 + 3 * G, 100.2), (T0 + 3 * G, 100.9)]
    (_, r10, _) = run(path, [(T0, 100.0)])
    assert (r10.label, r10.t_barrier) == (1, T0 + 3 * G)
    assert r10.y_fwd_bps == pytest.approx(90.0)
    # a same-ts group ending exactly at t_event pins the anchor state too
    path = flat_warmup()[:-1] + [(T0, 100.9), (T0, 100.0)]
    (r2, _, _) = run(path, [(T0, 100.0)])
    assert (r2.label, r2.y_fwd_bps) == (0, 0.0)


# ------------------------------------------------ T2 anchor-read wiring (P0)

def test_anchor_from_bar_reads_uses_the_label_read_never_the_observable():
    class Obs(NamedTuple):
        mid: float
        target_read_ts: int

    class Lab(NamedTuple):
        mid: float

    class Reads(NamedTuple):
        t_event: int
        observable: Obs
        label: Lab

    a = anchor_from_bar_reads(Reads(T0, Obs(999.0, T0 - 5), Lab(100.25)))
    assert a == LabelAnchor(t_event=T0, p0=100.25)


def test_t2_dual_reads_wire_into_labels_and_an_observable_p0_fails_closed():
    from bars.snapshot import BarBookReads, BookDelta, dual_book_reads

    # bid 100 / ask 101 at origin 10; a DELAYED ask improvement (origin 15,
    # received 90) puts the TRUE mid at 100.3 while the box still sees 100.5
    events = [
        BookDelta(10, 10, 1, "bid", 100.0, 2.0),
        BookDelta(10, 10, 2, "ask", 101.0, 3.0),
        BookDelta(15, 90, 3, "ask", 100.6, 3.0),
    ]
    (reads,) = dual_book_reads(events, [20], staleness_cap_ns=1_000)
    assert isinstance(reads, BarBookReads)
    assert (reads.observable.mid, reads.label.mid) == (100.5, 100.3)
    anchor = anchor_from_bar_reads(reads)
    assert anchor == LabelAnchor(t_event=20, p0=100.3)
    # the true-mid path is the label fold's mid at each origin time
    path = [(10, 100.5), (15, 100.3), (25, 101.0)]
    p = params(min_returns=1, horizons={"10ns": 10})
    (row,) = triple_barrier_labels(path, [anchor], params=p, coverage_end_ns=30)
    assert row.p0 == 100.3
    assert (row.label, row.t_barrier) == (1, 25)  # +69.8 bps >= 50 bps floor
    # wiring the OBSERVABLE read in as P0 fails closed against the true path
    with pytest.raises(ValueError, match="does not equal the true path mid"):
        list(triple_barrier_labels(path, [LabelAnchor(20, reads.observable.mid)],
                                   params=p, coverage_end_ns=30))


# ------------------------------------------------ determinism and streaming

def _random_world(seed):
    import random

    rng = random.Random(seed)
    ts, mid = 0, 100.0
    path = []
    for _ in range(400):
        ts += rng.randint(1, 3) * G // 2
        mid = round(mid * (1 + rng.choice([-30, -10, 0, 10, 30]) / 1e4), 6)
        path.append((ts, mid))
    # anchors ride the path itself, so p0 == the true as-of mid by construction
    anchors = [(path[i][0], path[i][1]) for i in range(50, 250, 20)]
    return path, anchors, path[-1][0]


def test_rebuilds_are_deterministic_across_input_forms():
    from data.labels import MidPoint

    for seed in (1, 2, 3):
        path, anchors, cov = _random_world(seed)
        first = run(path, anchors, coverage_end=cov)
        assert first == run(path, anchors, coverage_end=cov)
        assert first == run(iter(path), iter(anchors), coverage_end=cov)
        assert first == run([MidPoint(*p) for p in path],
                            [LabelAnchor(*a) for a in anchors],
                            coverage_end=cov)
        keys = [(r.t_event, DEFAULT_HORIZONS[r.horizon]) for r in first]
        assert keys == sorted(keys)                # (t_event, horizon_ns) order
        assert len(set(keys)) == len(keys)         # one row per decision key


def test_long_pre_anchor_prefix_is_folded_not_buffered():
    # Codex P2: a long warm-up/filtered prefix (T9 can pass a full-day path
    # whose anchors start late) must be folded into the trailing EWMA point by
    # point and discarded — never enqueued wholesale before the anchor. 200k
    # prefix points buffered as deque entries would allocate tens of MB; the
    # streaming fold stays within a small constant.
    import tracemalloc

    n = 200_000
    t_event = n + 60

    def path():
        for i in range(n):                     # varying mids: real returns
            yield (i, 100.0 + (i % 7) * 0.01)
        yield (t_event, 100.0)                 # the as-of anchor state

    p = params(horizons={"10ns": 10}, min_returns=2)
    tracemalloc.start()
    rows = list(triple_barrier_labels(path(), [(t_event, 100.0)], params=p,
                                      coverage_end_ns=t_event + 10))
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert len(rows) == 1 and isinstance(rows[0], LabelRow)
    assert peak < 4_000_000                    # bytes; buffering would be ~25MB


def test_generator_is_lazy_and_consumes_only_the_needed_prefix():
    consumed = []

    def stream():
        for p in flat_warmup() + [(T0 + 30 * G, 100.2), (T0 + 200 * G, 100.4),
                                  (T0 + 400 * G, 100.6)]:
            consumed.append(p)
            yield p

    t1 = T0 + 300 * G
    gen = triple_barrier_labels(stream(), [(T0, 100.0), (t1, 100.4)],
                                params=params(), coverage_end_ns=t1 + 60 * G)
    first3 = [next(gen) for _ in range(3)]
    assert all(r.t_event == T0 for r in first3)
    # only points with ts <= T0+60s plus ONE validated lookahead are consumed
    assert len(consumed) == len(flat_warmup()) + 2
    rest = list(gen)
    assert len(rest) == 3 and len(consumed) == len(flat_warmup()) + 3
