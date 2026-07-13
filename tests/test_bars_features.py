"""Causal stationarized single-venue bar features (issue #78 / plan §H, §C.2, T3).

Covers: the minimum top-K depth exposure on T2's observable read (received-gated,
label read stays bare), the pinned §H feature registry with hand-computed formula
fixtures, exact bar-member trade-flow discipline, zero-trade cap-bar policy,
fail-closed insufficient-depth / missing-prior-read handling, the true
`t_feature_start`, delayed-event and post-`t_event` mutation guards, stationarity
under price/size/tick rescaling, deterministic rebuilds, and value-level
no-lookahead over the full T1->T2->T3 pipeline."""
import itertools

import pytest

from bars.snapshot import (
    BarBookReads,
    BookDelta,
    LabelBookRead,
    ObservableBookRead,
    dual_book_reads,
)

_SEQ = itertools.count()


def delta(origin, received=None, *, side="bid", price=100.0, size=1.0, seq=None):
    """Book-event fixture: received defaults to origin (an undelayed event)."""
    return BookDelta(
        origin_time=origin,
        received_time=origin if received is None else received,
        seq=next(_SEQ) if seq is None else seq,
        side=side,
        price=price,
        size=size,
    )


def ladder(origin, *, bids=((100.0, 2.0),), asks=((101.0, 3.0),)):
    """A whole book ladder at one origin instant: [(price, size), ...] per side."""
    return ([delta(origin, side="bid", price=p, size=s) for p, s in bids]
            + [delta(origin, side="ask", price=p, size=s) for p, s in asks])


def reads(events, t_events, *, cap=10_000, **kw):
    return list(dual_book_reads(events, t_events, staleness_cap_ns=cap, **kw))


# --------------------- top-K depth exposure on the observable read (T2 ext.)

def test_observable_read_carries_topk_depth_in_canonical_order():
    events = ladder(10,
                    bids=((100.0, 2.0), (99.5, 5.0), (99.0, 1.0)),
                    asks=((101.0, 3.0), (101.5, 4.0), (102.0, 6.0)))
    (r,) = reads(events, [20], top_k=2)
    assert isinstance(r, BarBookReads)
    # bids descending from the best, asks ascending from the best
    assert r.observable.bid_prices == (100.0, 99.5)
    assert r.observable.bid_sizes == (2.0, 5.0)
    assert r.observable.ask_prices == (101.0, 101.5)
    assert r.observable.ask_sizes == (3.0, 4.0)
    # level 0 is exactly the already-exposed top of book
    assert r.observable.bid_prices[0] == r.observable.best_bid
    assert r.observable.bid_sizes[0] == r.observable.best_bid_size
    assert r.observable.ask_prices[0] == r.observable.best_ask
    assert r.observable.ask_sizes[0] == r.observable.best_ask_size


def test_topk_depth_is_received_gated_like_the_top():
    # a delayed second bid level must not surface in the depth tuples until
    # received; afterwards it must
    events = (ladder(10, bids=((100.0, 2.0),), asks=((101.0, 3.0),))
              + [delta(15, 60, side="bid", price=99.5, size=7.0)])
    before, after = reads(events, [20, 70], top_k=2)
    assert before.observable.bid_prices == (100.0,)
    assert after.observable.bid_prices == (100.0, 99.5)
    assert after.observable.bid_sizes == (2.0, 7.0)


def test_thin_book_yields_short_tuples_never_padding():
    # fewer levels than top_k: expose exactly what exists — no NaN/zero padding
    # (the feature layer owns the fail-closed insufficient-depth policy)
    events = ladder(10, bids=((100.0, 2.0),), asks=((101.0, 3.0), (102.0, 1.0)))
    (r,) = reads(events, [20], top_k=3)
    assert r.observable.bid_prices == (100.0,)
    assert r.observable.bid_sizes == (2.0,)
    assert r.observable.ask_prices == (101.0, 102.0)
    assert r.observable.ask_sizes == (3.0, 1.0)


def test_default_top_k_is_one_and_matches_the_top_of_book():
    (r,) = reads(ladder(10), [20])
    assert r.observable.bid_prices == (r.observable.best_bid,)
    assert r.observable.bid_sizes == (r.observable.best_bid_size,)
    assert r.observable.ask_prices == (r.observable.best_ask,)
    assert r.observable.ask_sizes == (r.observable.best_ask_size,)


def test_label_read_stays_a_bare_price_anchor_without_depth():
    for f in ("bid_prices", "bid_sizes", "ask_prices", "ask_sizes"):
        assert f in ObservableBookRead._fields
        assert f not in LabelBookRead._fields


def test_depth_reflects_removals_and_tombstones():
    # a removal empties a level out of the depth tuples; an older straggler to
    # the same level cannot resurrect it
    events = (ladder(10, bids=((100.0, 2.0), (99.5, 5.0)))
              + [delta(12, 40, side="bid", price=99.5, size=9.0),
                 delta(20, side="bid", price=99.5, size=0.0)])
    r_removed, r_after_straggler = reads(events, [30, 50], top_k=2)
    assert r_removed.observable.bid_prices == (100.0,)
    assert r_after_straggler.observable.bid_prices == (100.0,)


def test_non_positive_top_k_fails_closed():
    with pytest.raises(ValueError, match="top_k"):
        reads(ladder(10), [20], top_k=0)
    with pytest.raises(ValueError, match="top_k"):
        reads(ladder(10), [20], top_k=-3)


# ---------------------------------------------------------- feature registry

import math

from bars.clock import CLOSE_DAY_END, CLOSE_THRESHOLD, CLOSE_TIME_CAP, Bar
from bars.events import ClockTrade
from bars.features import (
    FEATURE_COLS,
    REJECT_INSUFFICIENT_DEPTH,
    REJECT_NO_PRIOR_READ,
    BarFeatureBuilder,
    FeatureConfig,
    FeatureRejection,
    FeatureRow,
)

_TSEQ = itertools.count(1)


def trade(origin, received=None, *, side="buy", price=100.0, amount=1.0, seq=None):
    return ClockTrade(origin_time=origin,
                      received_time=origin if received is None else received,
                      seq=next(_TSEQ) if seq is None else seq,
                      side=side, price=price, amount=amount)


def mkbar(members=(), *, t_event=100, interval_start=50, threshold=10_000.0,
          close_reason=None, cap_fire=None, index=0, is_warmup=False):
    members = tuple(members)
    if close_reason is None:
        close_reason = CLOSE_THRESHOLD if members else CLOSE_TIME_CAP
    if close_reason != CLOSE_THRESHOLD and cap_fire is None:
        cap_fire = t_event
    notional = sum(m.price * m.amount for m in members)
    return Bar(index=index, interval_start_ns=interval_start, close_reason=close_reason,
               cap_fire_ns=cap_fire, t_event=t_event, threshold=threshold,
               is_warmup=is_warmup, notional=notional, members=members)


def obs(*, ts=40, bids=((100.0, 2.0), (99.5, 6.0)), asks=((100.5, 4.0), (101.5, 8.0))):
    """Hand-built observable read; mid/microprice/top derived from the ladders."""
    bb, bbs = bids[0]
    ba, bas = asks[0]
    return ObservableBookRead(
        target_read_ts=ts, mid=(bb + ba) / 2.0,
        microprice=(bas * bb + bbs * ba) / (bbs + bas),
        best_bid=bb, best_ask=ba, best_bid_size=bbs, best_ask_size=bas,
        bid_prices=tuple(p for p, _ in bids), bid_sizes=tuple(s for _, s in bids),
        ask_prices=tuple(p for p, _ in asks), ask_sizes=tuple(s for _, s in asks))


CFG = FeatureConfig(top_k=2, tick_size=0.5)


def build1(bar, read, *, prev=obs(ts=30), config=CFG):
    """One-shot build with a seeded prior read (the T9 cross-bar chain)."""
    return BarFeatureBuilder(config, initial_read=prev).build(bar, read)


def test_feature_registry_is_exactly_the_pinned_g0_list_in_manifest_order():
    # plan §H feature registry — explicit ordered names, pinned; T8 writes these
    # into `feature_cols` verbatim
    assert FEATURE_COLS == (
        "ofi_integrated", "microprice_dev", "queue_imb", "spread_tick", "cvd",
        "depth_imbalance", "book_slope", "vwap_minus_mid", "trade_count",
        "signed_vol", "aggressor_imb", "largest_print", "event_intensity",
        "rv_intrabar", "mae_intrabar", "elapsed_ns", "tod_sin", "tod_cos",
    )
    assert FeatureRow._fields == ("t_event", "t_feature_start") + FEATURE_COLS


def test_no_feature_name_trips_the_manifest_leak_screen_or_reserved_set():
    from eval.manifest import leaky_feature_names
    from eval.matrix import RESERVED
    assert leaky_feature_names(list(FEATURE_COLS)) == []
    assert not set(FEATURE_COLS) & set(RESERVED)
    # the registry itself is pinned exactly above; the "no raw price level can
    # silently enter feature_cols" criterion is value-level and is proven by the
    # rescaling-invariance test below


# --------------------------------------------- hand-computed formula fixtures

def test_book_shape_features_match_hand_computed_values():
    read = obs(ts=95)  # bids (100.0,2.0),(99.5,6.0); asks (100.5,4.0),(101.5,8.0)
    # the PRIOR read differs on every ladder value, so a book-shape feature that
    # wrongly consumed the stale prior state cannot reproduce these numbers
    prev = obs(ts=30, bids=((99.8, 1.0), (99.2, 3.0)), asks=((100.7, 5.0), (101.8, 5.0)))
    row = build1(mkbar([trade(60, price=100.0, amount=30.0)], t_event=100), read,
                 prev=prev)
    assert isinstance(row, FeatureRow)
    mid, mp = 100.25, (4.0 * 100.0 + 2.0 * 100.5) / 6.0
    assert row.microprice_dev == pytest.approx(1e4 * (mp - mid) / mid)
    assert row.queue_imb == pytest.approx((2.0 - 4.0) / (2.0 + 4.0))
    assert row.spread_tick == pytest.approx((100.5 - 100.0) / 0.5)
    assert row.depth_imbalance == pytest.approx(((2 + 6) - (4 + 8)) / ((2 + 6) + (4 + 8)))
    # book_slope: K-level depth in bar-threshold units per bps of ladder span
    depth_units = (2 + 6 + 4 + 8) * mid / 10_000.0
    span_bps = 1e4 * (101.5 - 99.5) / mid
    assert row.book_slope == pytest.approx(depth_units / span_bps)


def test_trade_flow_features_match_hand_computed_values():
    members = [trade(60, price=100.0, amount=30.0, side="buy"),
               trade(70, price=99.5, amount=20.0, side="sell"),
               trade(80, price=101.0, amount=50.0, side="buy")]
    row = build1(mkbar(members, t_event=100, interval_start=50), obs(ts=95))
    mid = 100.25
    assert row.trade_count == 3.0
    assert row.cvd == pytest.approx((3000.0 - 1990.0 + 5050.0) / 10_000.0)
    assert row.signed_vol == pytest.approx((30.0 - 20.0 + 50.0) / (30.0 + 20.0 + 50.0))
    assert row.aggressor_imb == pytest.approx((2 - 1) / 3)
    assert row.largest_print == pytest.approx(5050.0 / 10_000.0)
    vwap = (3000.0 + 1990.0 + 5050.0) / 100.0
    assert row.vwap_minus_mid == pytest.approx(1e4 * (vwap - mid) / mid)


def test_intra_bar_path_features_match_hand_computed_values():
    members = [trade(60, price=100.0, amount=30.0, side="buy"),
               trade(70, price=99.5, amount=20.0, side="sell"),
               trade(80, price=101.0, amount=50.0, side="buy")]
    row = build1(mkbar(members, t_event=100, interval_start=50), obs(ts=95))
    r1, r2 = math.log(99.5 / 100.0), math.log(101.0 / 99.5)
    assert row.rv_intrabar == pytest.approx((1e4 * r1) ** 2 + (1e4 * r2) ** 2)
    # close 101 >= open 100 -> long direction; worst adverse = the dip to 99.5
    assert row.mae_intrabar == pytest.approx(-1e4 * math.log(99.5 / 100.0))
    # falling path: close < open -> short direction; adverse = the pop to 101
    falling = [trade(60, price=100.0, amount=30.0),
               trade(70, price=101.0, amount=20.0),
               trade(80, price=99.0, amount=50.0)]
    row = build1(mkbar(falling, t_event=100), obs(ts=95))
    assert row.mae_intrabar == pytest.approx(1e4 * math.log(101.0 / 100.0))


def test_time_and_intensity_features_match_hand_computed_values():
    members = [trade(60, price=100.0, amount=30.0),
               trade(70, price=99.5, amount=20.0),
               trade(80, price=101.0, amount=50.0)]
    row = build1(mkbar(members, t_event=100, interval_start=50), obs(ts=95))
    assert row.elapsed_ns == 30.0  # threshold close: crossing origin 80 - start 50
    assert row.event_intensity == pytest.approx(math.log1p(3 * 1e9 / 30.0))
    day_ns = 86_400 * 10**9
    frac = 100 / day_ns
    assert row.tod_sin == pytest.approx(math.sin(2 * math.pi * frac))
    assert row.tod_cos == pytest.approx(math.cos(2 * math.pi * frac))
    # a same-instant burst (elapsed 0) stays finite via the 1 ns floor
    burst = [trade(50, price=100.0, amount=200.0, seq=901),
             trade(50, price=100.0, amount=200.0, seq=902)]
    row = build1(mkbar(burst, t_event=90, interval_start=50), obs(ts=85))
    assert row.elapsed_ns == 0.0
    assert row.event_intensity == pytest.approx(math.log1p(2 * 1e9 / 1.0))
    assert all(math.isfinite(v) for v in row[2:])


def test_ofi_integrated_matches_the_hand_computed_level_indexed_formula():
    prev = obs(ts=30)   # bids (100.0,2.0),(99.5,6.0); asks (100.5,4.0),(101.5,8.0)
    cur = obs(ts=95, bids=((100.5, 3.0), (100.0, 2.0)),
              asks=((101.0, 5.0), (101.5, 6.0)))
    row = build1(mkbar([trade(60, price=100.0, amount=30.0)], t_event=100),
                 cur, prev=prev)
    # bid L0: 100.5>100.0 -> +3.0 ; bid L1: 100.0>99.5 -> +2.0
    # ask L0: 101.0>100.5 -> -4.0 ; ask L1: 101.5==101.5 -> 6.0-8.0 = -2.0
    ofi_raw = (3.0 + 2.0) - (-4.0 - 2.0)
    cur_mid = (100.5 + 101.0) / 2.0
    assert row.ofi_integrated == pytest.approx(ofi_raw * cur_mid / 10_000.0)


def test_ofi_sign_conventions_bid_and_ask_sides():
    # improving bid adds +size; retreating bid removes the prior size; the ask
    # side mirrors with opposite sign (Cont/CKS level convention)
    base = obs(ts=30)
    grow_bid = obs(ts=95, bids=((100.25, 5.0), (100.0, 2.0)))
    row = build1(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100),
                 grow_bid, prev=base)
    # bids: L0 100.25>100.0 -> +5 ; L1 100.0>99.5 -> +2  | asks unchanged -> 0
    assert row.ofi_integrated == pytest.approx((5.0 + 2.0) * grow_bid.mid / 10_000.0)
    retreat_ask = obs(ts=95, asks=((100.75, 1.0), (101.5, 8.0)))
    row = build1(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100),
                 retreat_ask, prev=base)
    # asks: L0 100.75>100.5 wait -- a LOWER ask price is an improvement:
    # L0 100.75 vs 100.5: price rose -> liquidity pulled -> e_ask = -4.0 (prev size)
    # L1 101.5 == 101.5 -> 8.0-8.0 = 0 ; bids unchanged -> 0
    assert row.ofi_integrated == pytest.approx(-(-4.0) * retreat_ask.mid / 10_000.0)


# ----------------------------------------- pinned edge policies (plan §E, #4)

def test_zero_trade_cap_bar_emits_the_pinned_zero_trade_flow_policy():
    bar = mkbar([], t_event=100, interval_start=50, close_reason=CLOSE_TIME_CAP,
                cap_fire=100)
    row = build1(bar, obs(ts=95))
    assert isinstance(row, FeatureRow)
    for name in ("cvd", "signed_vol", "aggressor_imb", "largest_print",
                 "vwap_minus_mid", "trade_count", "event_intensity",
                 "rv_intrabar", "mae_intrabar"):
        assert getattr(row, name) == 0.0
    # book-shape features still come from the observable book
    assert row.queue_imb != 0.0 and row.depth_imbalance != 0.0
    assert row.elapsed_ns == 50.0
    assert all(math.isfinite(v) for v in row[2:])


def test_single_trade_bar_has_zero_path_features_but_real_flow():
    row = build1(mkbar([trade(60, price=100.0, amount=30.0, side="sell")],
                       t_event=100), obs(ts=95))
    assert row.rv_intrabar == 0.0 and row.mae_intrabar == 0.0
    assert row.cvd == pytest.approx(-3000.0 / 10_000.0)
    assert row.aggressor_imb == -1.0 and row.signed_vol == -1.0


def test_two_trade_bar_pins_the_path_boundary():
    # n == 2 is exactly where the intra-bar path switches on: rv must be the
    # single squared log return, and mae is structurally 0 (with two points the
    # only non-open print IS the close, which defines the direction)
    two = [trade(60, price=100.0, amount=30.0), trade(70, price=101.0, amount=20.0)]
    row = build1(mkbar(two, t_event=100), obs(ts=95))
    assert row.rv_intrabar == pytest.approx((1e4 * math.log(101.0 / 100.0)) ** 2)
    assert row.rv_intrabar > 0.0
    assert row.mae_intrabar == 0.0


def test_day_end_bar_is_buildable():
    bar = mkbar([trade(60, price=100.0, amount=1.0)], t_event=100,
                close_reason=CLOSE_DAY_END, cap_fire=100)
    assert isinstance(build1(bar, obs(ts=95)), FeatureRow)


# ------------------------------------------------- fail-closed feature gates

def test_insufficient_depth_fails_closed_never_zero_fills():
    thin = obs(ts=95, bids=((100.0, 2.0),))  # 1 bid level < top_k=2
    r = build1(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100), thin)
    assert isinstance(r, FeatureRejection)
    assert (r.t_event, r.reason) == (100, REJECT_INSUFFICIENT_DEPTH)
    # ...and a thin PRIOR read also fails closed (the OFI window needs K on both)
    r = build1(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100),
               obs(ts=95), prev=obs(ts=30, asks=((100.5, 4.0),)))
    assert isinstance(r, FeatureRejection)
    assert r.reason == REJECT_INSUFFICIENT_DEPTH


def test_first_bar_without_a_prior_read_fails_closed_then_recovers():
    b = BarFeatureBuilder(CFG)
    r1 = b.build(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100), obs(ts=95))
    assert isinstance(r1, FeatureRejection)
    assert (r1.t_event, r1.reason) == (100, REJECT_NO_PRIOR_READ)
    # the rejected bar's read still advances the chain: the next bar computes
    r2 = b.build(mkbar([trade(160, price=100.0, amount=1.0)], t_event=200,
                       interval_start=150, index=1), obs(ts=195))
    assert isinstance(r2, FeatureRow)


def test_a_thin_read_poisons_exactly_the_next_bar_then_the_chain_recovers():
    b = BarFeatureBuilder(CFG, initial_read=obs(ts=30))
    thin = obs(ts=95, bids=((100.0, 2.0),))
    r1 = b.build(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100), thin)
    assert isinstance(r1, FeatureRejection)
    r2 = b.build(mkbar([trade(160, price=100.0, amount=1.0)], t_event=200,
                       interval_start=150, index=1), obs(ts=195))
    assert isinstance(r2, FeatureRejection)          # prior read is the thin one
    assert r2.reason == REJECT_INSUFFICIENT_DEPTH
    r3 = b.build(mkbar([trade(260, price=100.0, amount=1.0)], t_event=300,
                       interval_start=250, index=2), obs(ts=295))
    assert isinstance(r3, FeatureRow)


def test_label_book_read_is_rejected_as_a_feature_input():
    # match the role-specific message, not the generic wrong-type guard: the
    # dedicated P0-leak branch must exist, not just isinstance-typing
    lbl = LabelBookRead(label_cut_ts=95, mid=100.25, microprice=100.2)
    with pytest.raises(ValueError, match="label-anchor"):
        build1(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100), lbl)
    with pytest.raises(ValueError, match="label-anchor"):
        BarFeatureBuilder(CFG, initial_read=lbl)


# ------------------------------------------------------------ t_feature_start

def test_t_feature_start_is_the_oldest_consumed_origin_event():
    # the OFI window reaches back to the PRIOR observable read; the bar interval
    # starts later -> the prior read's origin is the true feature-window start
    row = build1(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100,
                       interval_start=50), obs(ts=95), prev=obs(ts=30))
    assert (row.t_event, row.t_feature_start) == (100, 30)
    # symmetric case: a fresh prior read but an interval opened earlier
    row = build1(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100,
                       interval_start=20), obs(ts=95), prev=obs(ts=45))
    assert row.t_feature_start == 20
    assert row.t_feature_start <= row.t_event


# ----------------------------------------------------- contract guard rails

def test_config_is_validated_fail_closed():
    with pytest.raises(ValueError, match="top_k"):
        BarFeatureBuilder(FeatureConfig(top_k=0, tick_size=0.5))
    for bad_tick in (0.0, -1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="tick_size"):
            BarFeatureBuilder(FeatureConfig(top_k=2, tick_size=bad_tick))


def test_bars_must_arrive_in_strictly_increasing_t_event_order():
    b = BarFeatureBuilder(CFG, initial_read=obs(ts=30))
    b.build(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100), obs(ts=95))
    # the dup bar is otherwise perfectly valid, so only the strictly-increasing
    # guard can raise — and the match pins that guard's own message
    dup = mkbar([ClockTrade(70, 70, 999, "buy", 100.0, 1.0)], t_event=100, index=1)
    with pytest.raises(ValueError, match="does not increase"):
        b.build(dup, obs(ts=99))


def test_a_read_from_after_t_event_is_rejected_as_mispaired():
    with pytest.raises(ValueError, match="target_read_ts"):
        build1(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100), obs(ts=101))


def test_reads_must_not_regress_across_bars():
    b = BarFeatureBuilder(CFG, initial_read=obs(ts=95))
    with pytest.raises(ValueError, match="target_read_ts"):
        b.build(mkbar([trade(60, price=100.0, amount=1.0)], t_event=100), obs(ts=40))


def test_member_contract_violations_raise_not_reject():
    # membership the box could not have known by t_event = a broken T1 pipeline
    late = [trade(60, 120, price=100.0, amount=1.0)]
    with pytest.raises(ValueError, match="received"):
        build1(mkbar(late, t_event=100), obs(ts=95))
    # capture before the exchange event = broken timestamp contract
    with pytest.raises(ValueError, match="received"):
        build1(mkbar([ClockTrade(60, 50, 1, "buy", 100.0, 1.0)], t_event=100),
               obs(ts=95))
    # out-of-order members = the clock's canonical order was not preserved
    unordered = [trade(70, price=100.0, amount=1.0), trade(60, price=100.0, amount=1.0)]
    with pytest.raises(ValueError, match="order"):
        build1(mkbar(unordered, t_event=100), obs(ts=95))
    # malformed prices/amounts fail loudly
    with pytest.raises(ValueError, match="price"):
        build1(mkbar([ClockTrade(60, 60, 1, "buy", -1.0, 1.0)], t_event=100),
               obs(ts=95))
    with pytest.raises(ValueError, match="amount"):
        build1(mkbar([ClockTrade(60, 60, 1, "buy", 100.0, 0.0)], t_event=100),
               obs(ts=95))
    with pytest.raises(ValueError, match="side"):
        build1(mkbar([ClockTrade(60, 60, 1, "bid", 100.0, 1.0)], t_event=100),
               obs(ts=95))
    # a member from before the bar's accumulation interval
    with pytest.raises(ValueError, match="precedes"):
        build1(mkbar([ClockTrade(40, 40, 1, "buy", 100.0, 1.0)], t_event=100,
                     interval_start=50), obs(ts=95))
    # duplicate (origin_time, seq) member keys: the canonical order is ambiguous
    dup_key = [ClockTrade(60, 60, 7, "buy", 100.0, 1.0),
               ClockTrade(60, 60, 7, "buy", 100.0, 2.0)]
    with pytest.raises(ValueError, match="order"):
        build1(mkbar(dup_key, t_event=100), obs(ts=95))


def test_malformed_depth_tuples_raise_as_contract_violations():
    good = obs(ts=95)
    bar = mkbar([trade(60, price=100.0, amount=1.0)], t_event=100)
    ragged = good._replace(bid_sizes=(2.0,))            # len != bid_prices
    with pytest.raises(ValueError, match="depth"):
        build1(bar, ragged)
    unsorted_bids = good._replace(bid_prices=(99.5, 100.0), bid_sizes=(6.0, 2.0))
    with pytest.raises(ValueError, match="depth"):
        build1(bar, unsorted_bids)
    bad_size = good._replace(bid_sizes=(2.0, 0.0))
    with pytest.raises(ValueError, match="depth"):
        build1(bar, bad_size)
    crossed = good._replace(ask_prices=(99.75, 101.5), ask_sizes=(4.0, 8.0))
    with pytest.raises(ValueError, match="depth"):
        build1(bar, crossed)


# ---------------------------------------- full T1 -> T2 -> T3 pipeline worlds

import datetime as _dt
import random

from bars.clock import ThresholdConfig, ThresholdSchedule, bars_for_day, coalesce_decision_bars
from bars.snapshot import SnapshotRejection

DAY = "2026-01-05"
DAY_OPEN = int(_dt.datetime(2026, 1, 5, tzinfo=_dt.timezone.utc).timestamp()) * 10**9


def at(offset_ns):
    return DAY_OPEN + int(offset_ns)


def _schedule(threshold):
    return ThresholdSchedule(ThresholdConfig(target_bars_per_day=100, window_days=7,
                                             warmup_days=1, seed_threshold=threshold))


def run_pipeline(trades, book_events, *, config=CFG, threshold=10_000.0,
                 time_cap_ns=2 * 10**9, staleness_cap_ns=60 * 10**9,
                 initial_read=None, until=None):
    """T1 clock -> coalesce -> T2 dual reads -> T3 features, exactly as the T9
    orchestrator will wire them (snapshot rejections skip the feature builder).
    Returns [(bar, read_or_snapshot_rejection, row_or_rejection), ...]."""
    bars = coalesce_decision_bars(bars_for_day(trades, day=DAY, schedule=_schedule(threshold),
                                               time_cap_ns=time_cap_ns))
    if until is not None:
        bars = itertools.takewhile(lambda b: b.t_event <= until, bars)
    bars = list(bars)
    read_stream = dual_book_reads(book_events, [b.t_event for b in bars],
                                  staleness_cap_ns=staleness_cap_ns, top_k=config.top_k)
    builder = BarFeatureBuilder(config, initial_read=initial_read)
    out = []
    for bar, rr in zip(bars, read_stream):
        if isinstance(rr, SnapshotRejection):
            out.append((bar, rr, rr))
        else:
            out.append((bar, rr, builder.build(bar, rr.observable)))
    return out


def _ladder_events(offset_ns, *, bids, asks, received_lag=0, seq_start):
    seq = itertools.count(seq_start)
    return [BookDelta(at(offset_ns), at(offset_ns) + received_lag, next(seq),
                      side, p, s)
            for side, levels in (("bid", bids), ("ask", asks))
            for p, s in levels]


L1_BIDS, L1_ASKS = ((100.0, 2.0), (99.5, 6.0)), ((100.5, 4.0), (101.5, 8.0))


def test_chain_survives_rejected_bars_and_reports_the_honest_feature_window():
    # bars: threshold @1.4s, quiet caps @3.4/5.4s, STALE cap @7.4s (skipped),
    # cap @9.4s after the book refreshes, threshold @9.8s — the builder chain
    # never raises and t_feature_start reaches back to the truly consumed reads
    books = (_ladder_events(5 * 10**8, bids=L1_BIDS, asks=L1_ASKS, seq_start=1)
             + _ladder_events(8 * 10**9, bids=((100.2, 3.0), (99.9, 1.0)),
                              asks=((100.6, 2.0), (101.0, 5.0)), seq_start=11))
    trades = [trade(at(o), seq=i + 1, price=100.0, amount=40.0)
              for i, o in enumerate((10**9, 12 * 10**8, 14 * 10**8))]
    trades += [trade(at(o), seq=i + 11, price=100.0, amount=40.0)
               for i, o in enumerate((96 * 10**8, 97 * 10**8, 98 * 10**8))]
    out = run_pipeline(trades, books, staleness_cap_ns=5 * 10**9,
                       initial_read=obs(ts=at(2 * 10**8)), until=at(10**10))
    rows = [r for _, _, r in out]
    assert [type(r).__name__ for r in rows] == [
        "FeatureRow", "FeatureRow", "FeatureRow", "SnapshotRejection",
        "FeatureRow", "FeatureRow"]
    assert rows[3].reason == "stale_book"
    assert rows[0].trade_count == 3.0 and rows[1].trade_count == 0.0
    # the skipped stale bar leaves the chain intact: the 9.4s cap bar's prior
    # read is the 5.4s bar's (the 0.5s ladder), and the interval opened at 7.4s
    assert rows[4].t_feature_start == at(5 * 10**8)
    # the 9.8s bar reaches back to its prior read = the 8.0s ladder
    assert rows[5].t_feature_start == at(8 * 10**9)
    for r in (rows[0], rows[1], rows[2], rows[4], rows[5]):
        assert r.t_feature_start <= r.t_event
        assert all(math.isfinite(v) for v in r[2:])


def test_trade_flow_uses_exactly_the_members_and_cvd_is_bar_additive():
    # T4 arrives EARLY (received 1.25s < bar A's t_event 1.35s) but its origin is
    # after A's crossing trade: it must count in bar B only (§J membership)
    books = _ladder_events(5 * 10**8, bids=L1_BIDS, asks=L1_ASKS, seq_start=1)
    trades = [
        ClockTrade(at(10 * 10**8), at(10 * 10**8), 1, "buy", 100.0, 40.0),
        ClockTrade(at(11 * 10**8), at(13 * 10**8), 2, "buy", 100.0, 40.0),
        ClockTrade(at(12 * 10**8), at(135 * 10**7), 3, "buy", 100.0, 80.0),  # crosses A
        ClockTrade(at(125 * 10**7), at(13 * 10**8), 4, "sell", 100.0, 10.0),  # early B member
        ClockTrade(at(20 * 10**8), at(20 * 10**8), 5, "buy", 100.0, 95.0),   # crosses B
    ]
    out = run_pipeline(trades, books, initial_read=obs(ts=at(2 * 10**8)),
                       until=at(25 * 10**8))
    (bar_a, _, row_a), (bar_b, _, row_b) = out
    assert bar_a.t_event == at(135 * 10**7) and bar_b.t_event == at(20 * 10**8)
    assert row_a.trade_count == 3.0 and row_b.trade_count == 2.0
    assert row_a.cvd == pytest.approx((4000.0 + 4000.0 + 8000.0) / 10_000.0)
    assert row_b.cvd == pytest.approx((-1000.0 + 9500.0) / 10_000.0)
    # bar-additive, no print double-counted across the boundary
    total = sum(r.cvd for _, _, r in out) * 10_000.0
    assert total == pytest.approx(4000.0 + 4000.0 + 8000.0 - 1000.0 + 9500.0)


def test_delayed_book_event_cannot_change_features_until_received():
    # D (origin 11.5s, received 20s) straddles bar A (t_event 12s): bar A's
    # features must be byte-identical with or without D; bar B (t_event 23s,
    # after receipt) must see it
    base_books = _ladder_events(10**9, bids=L1_BIDS, asks=L1_ASKS, seq_start=1)
    d_event = BookDelta(at(115 * 10**8), at(20 * 10**9), 99, "bid", 100.0, 9.0)
    trades = [trade(at(o * 10**8), seq=i + 1, price=100.0, amount=40.0)
              for i, o in enumerate((100, 110, 120, 210, 220, 230))]
    kw = dict(initial_read=obs(ts=at(5 * 10**8)), until=at(24 * 10**9),
              time_cap_ns=15 * 10**9)
    with_d = run_pipeline(trades, sorted(base_books + [d_event],
                                         key=lambda e: (e.origin_time, e.seq)), **kw)
    without_d = run_pipeline(trades, base_books, **kw)
    rows_with = [r for _, _, r in with_d]
    rows_without = [r for _, _, r in without_d]
    assert rows_with[0] == rows_without[0]              # A: identical pre-receipt
    assert isinstance(rows_with[1], FeatureRow)
    assert rows_with[1] != rows_without[1]              # B: receipt changes state
    assert rows_with[1].queue_imb != rows_without[1].queue_imb


def test_post_t_event_data_cannot_mutate_already_emitted_rows():
    # value-level no-lookahead (§E/T3): extending the day with trades and book
    # events entirely after the emitted decisions leaves every emitted row
    # byte-identical — mirrors tests/test_reconstruct_no_lookahead.py
    books = _ladder_events(5 * 10**8, bids=L1_BIDS, asks=L1_ASKS, seq_start=1)
    trades = [
        ClockTrade(at(10 * 10**8), at(10 * 10**8), 1, "buy", 100.0, 40.0),
        ClockTrade(at(11 * 10**8), at(13 * 10**8), 2, "buy", 100.0, 40.0),
        ClockTrade(at(12 * 10**8), at(135 * 10**7), 3, "buy", 100.0, 80.0),
        ClockTrade(at(125 * 10**7), at(13 * 10**8), 4, "sell", 100.0, 10.0),
        ClockTrade(at(20 * 10**8), at(20 * 10**8), 5, "buy", 100.0, 95.0),
    ]
    tail_books = _ladder_events(10 * 10**9, bids=((100.1, 1.0), (99.8, 2.0)),
                                asks=((100.4, 2.0), (100.9, 1.0)), seq_start=51)
    tail_trades = [trade(at((100 + i) * 10**8), seq=50 + i, price=100.2, amount=33.0)
                   for i in range(6)]
    kw = dict(initial_read=obs(ts=at(2 * 10**8)), until=at(25 * 10**8))
    base = run_pipeline(trades, books, **kw)
    extended = run_pipeline(trades + tail_trades, books + tail_books, **kw)
    assert [r for _, _, r in base] == [r for _, _, r in extended]
    assert [b for b, _, _ in base] == [b for b, _, _ in extended]


# --------------------------------------------- seeded random pipeline worlds

def _random_pipeline_world(seed, *, n_trades=120):
    """Seeded synthetic day: a rounded random-walk mid, full-ladder book refresh
    (with removals of stale levels) before each trade, mixed receipt lags on
    both channels, occasional quiet gaps that fire the time cap."""
    rng = random.Random(seed)
    trades, books = [], []
    price = 100.0
    t_ns = 10**9
    bseq = itertools.count(1)
    prev_levels: dict[str, set] = {"bid": set(), "ask": set()}
    for i in range(n_trades):
        t_ns += rng.randint(10 * 10**6, 800 * 10**6)
        if rng.random() < 0.04:
            t_ns += rng.randint(2 * 10**9, 5 * 10**9)   # quiet stretch -> cap bars
        price *= math.exp(rng.gauss(0.0, 1e-4))
        mid = round(price, 2)
        book_origin = t_ns - 5 * 10**6
        lag = rng.choice([0, 0, 0, 10**6, 50 * 10**6, 600 * 10**6])
        new_levels = {"bid": {(round(mid - 0.05 * (l + 1), 2), rng.uniform(0.5, 9.0))
                              for l in range(3)},
                      "ask": {(round(mid + 0.05 * (l + 1), 2), rng.uniform(0.5, 9.0))
                              for l in range(3)}}
        for side in ("bid", "ask"):
            new_prices = {p for p, _ in new_levels[side]}
            for stale in prev_levels[side] - new_prices:
                books.append(BookDelta(at(book_origin), at(book_origin) + lag,
                                       next(bseq), side, stale, 0.0))
            for p, s in sorted(new_levels[side]):
                books.append(BookDelta(at(book_origin), at(book_origin) + lag,
                                       next(bseq), side, p, s))
            prev_levels[side] = new_prices
        trades.append(ClockTrade(at(t_ns), at(t_ns) + rng.choice([0, 10**6, 40 * 10**6]),
                                 i + 1, rng.choice(["buy", "sell"]), mid,
                                 rng.uniform(0.5, 30.0)))
    return trades, books, t_ns


def test_random_worlds_emit_finite_causal_rows_and_rebuild_deterministically():
    for seed in (1, 2, 3):
        trades, books, last_ns = _random_pipeline_world(seed)
        kw = dict(threshold=8_000.0, staleness_cap_ns=2 * 10**9,
                  until=at(last_ns + 25 * 10**8))
        out = run_pipeline(trades, books, **kw)
        rows = [r for _, _, r in out if isinstance(r, FeatureRow)]
        assert len(rows) > 20     # the world is meaningful, not rejection soup
        for bar, _, r in out:
            if isinstance(r, FeatureRow):
                assert r.t_event == bar.t_event
                assert r.t_feature_start <= r.t_event
                assert all(math.isfinite(v) for v in r[2:])
        again = run_pipeline(trades, books, **kw)
        assert [r for _, _, r in again] == [r for _, _, r in out]
        # input-order scrambling of the trade stream is irrelevant (the clock
        # restores the canonical (origin_time, seq) order)
        shuffled = list(trades)
        random.Random(seed + 100).shuffle(shuffled)
        assert [r for _, _, r in run_pipeline(shuffled, books, **kw)] == \
               [r for _, _, r in out]


def test_features_are_invariant_under_price_size_tick_rescaling():
    # value-level stationarity (issue #78): price x4, size /4, tick x4 leaves
    # every feature unchanged — no raw price/size level leaks into the vector
    lam = 4.0
    trades, books, last_ns = _random_pipeline_world(5)
    scaled_trades = [t._replace(price=t.price * lam, amount=t.amount / lam)
                     for t in trades]
    scaled_books = [b._replace(price=b.price * lam,
                               size=(b.size / lam if b.size else 0.0)) for b in books]
    kw = dict(threshold=8_000.0, staleness_cap_ns=2 * 10**9,
              until=at(last_ns + 25 * 10**8))
    base = run_pipeline(trades, books, config=FeatureConfig(top_k=2, tick_size=0.5), **kw)
    scaled = run_pipeline(scaled_trades, scaled_books,
                          config=FeatureConfig(top_k=2, tick_size=0.5 * lam), **kw)
    assert len(base) == len(scaled) and len(base) > 20
    for (_, _, r0), (_, _, r1) in zip(base, scaled):
        assert type(r0) is type(r1)
        if isinstance(r0, FeatureRow):
            assert (r0.t_event, r0.t_feature_start) == (r1.t_event, r1.t_feature_start)
            for name, v0, v1 in zip(FEATURE_COLS, r0[2:], r1[2:]):
                assert v1 == pytest.approx(v0, rel=1e-9, abs=1e-12), name
        else:
            assert (r0.t_event, r0.reason) == (r1.t_event, r1.reason)


def test_streaming_rows_match_an_independent_oracle_recomputation():
    # independent (list-comprehension, stateless) recomputation of the
    # state-bearing features from each (bar, prior read, current read) triple —
    # catches prior-read chain bugs the formula fixtures cannot see
    trades, books, last_ns = _random_pipeline_world(9)
    out = run_pipeline(trades, books, threshold=8_000.0, staleness_cap_ns=2 * 10**9,
                       until=at(last_ns + 25 * 10**8))
    prev = None
    k = CFG.top_k
    for bar, rr, row in out:
        if isinstance(rr, SnapshotRejection):
            continue
        read = rr.observable
        if prev is None:
            assert isinstance(row, FeatureRejection) and row.reason == "no_prior_read"
        elif min(len(prev.bid_prices), len(prev.ask_prices),
                 len(read.bid_prices), len(read.ask_prices)) < k:
            assert isinstance(row, FeatureRejection)
            assert row.reason == "insufficient_depth"
        else:
            assert isinstance(row, FeatureRow)
            flows = []
            for prices0, sizes0, prices1, sizes1, better in (
                (prev.bid_prices, prev.bid_sizes, read.bid_prices, read.bid_sizes,
                 lambda a, b: a > b),
                (prev.ask_prices, prev.ask_sizes, read.ask_prices, read.ask_sizes,
                 lambda a, b: a < b),
            ):
                flows.append(sum(
                    sizes1[l] if better(prices1[l], prices0[l])
                    else (-sizes0[l] if better(prices0[l], prices1[l])
                          else sizes1[l] - sizes0[l])
                    for l in range(k)))
            expect_ofi = (flows[0] - flows[1]) * read.mid / bar.threshold
            assert row.ofi_integrated == pytest.approx(expect_ofi)
            # single-read book-shape features must come from the CURRENT read
            # (prev != read in these worlds, so a stale-state swap dies here)
            assert row.microprice_dev == pytest.approx(
                1e4 * (read.microprice - read.mid) / read.mid)
            assert row.queue_imb == pytest.approx(
                (read.bid_sizes[0] - read.ask_sizes[0])
                / (read.bid_sizes[0] + read.ask_sizes[0]))
            assert row.spread_tick == pytest.approx(
                (read.ask_prices[0] - read.bid_prices[0]) / CFG.tick_size)
            bd, ad = sum(read.bid_sizes[:k]), sum(read.ask_sizes[:k])
            assert row.depth_imbalance == pytest.approx((bd - ad) / (bd + ad))
            span_bps = 1e4 * (read.ask_prices[k - 1] - read.bid_prices[k - 1]) / read.mid
            assert row.book_slope == pytest.approx(
                (bd + ad) * read.mid / bar.threshold / span_bps)
            signed = [(1.0 if m.side == "buy" else -1.0, m.price * m.amount)
                      for m in bar.members]
            assert row.cvd == pytest.approx(
                sum(s * n for s, n in signed) / bar.threshold)
            assert row.trade_count == float(len(bar.members))
            if bar.members:
                n_buy = sum(1 for m in bar.members if m.side == "buy")
                assert row.aggressor_imb == pytest.approx(
                    (n_buy - (len(bar.members) - n_buy)) / len(bar.members))
            assert row.t_feature_start == min(bar.interval_start_ns,
                                              prev.target_read_ts)
        prev = read
