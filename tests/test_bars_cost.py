"""Causal per-row execution-cost inputs (issue #82 / plan §G, T7).

Covers: hand-computed spread/drift/slippage/fee/total-cost pins, zero drift and
target_read_ts == t_event, divergent observable vs true t_event mids (drift is
charged forward, the label anchor never moves), fees doubled exactly once with
spread kept OUT of cost_bps, fail-closed invalid/crossed/one-sided/negative/
nonfinite inputs and timestamps, the explicit immutable serializable cost
assumption with venue/product/source separation (no silent Binance/Coinbase
aliasing, no baked-in production fee tier), a post-t_event mutation causality
fixture through T2's dual_book_reads, and integration with eval.cost.net_pnl's
no-trade band (two spread crossings applied by the evaluator, never here).
"""
import itertools
import json
import math

import pytest

from bars.cost import (
    DRIFT_POLICY,
    CostAssumption,
    CostRow,
    cost_row,
    require_assumption_identity,
    validate_cost_assumption,
)
from bars.snapshot import (
    BarBookReads,
    BookDelta,
    LabelBookRead,
    ObservableBookRead,
    dual_book_reads,
)
from eval.cost import net_pnl

_SEQ = itertools.count()

PRODUCT = "BTC-USDT-PERP"
SOURCE = "lake/book_delta_v2@cert-2026-04-01"


def assumption(**kw) -> CostAssumption:
    """Synthetic assumption fixture. The values are EXPLICIT test constants —
    T10 selects and freezes the real Binance tier (plan Q1), never this file."""
    base = dict(venue="binance", product=PRODUCT, source=SOURCE,
                version="test-v1", taker_fee_bps=1.75, base_slippage_bps=0.25)
    base.update(kw)
    return CostAssumption(**base)


def obs_read(*, ts=20, bb=99.0, ba=101.0, bb_size=2.0, ba_size=3.0, mid=None):
    if mid is None:
        # a placeholder for non-numeric price fixtures: the price check fires
        # before the mid check, so this mid is never reached in those cases
        mid = (bb + ba) / 2.0 if all(
            isinstance(v, (int, float)) for v in (bb, ba)) else 100.0
    return ObservableBookRead(target_read_ts=ts, mid=mid, microprice=mid,
                              best_bid=bb, best_ask=ba,
                              best_bid_size=bb_size, best_ask_size=ba_size)


def label_read(*, ts=20, mid=100.0):
    return LabelBookRead(label_cut_ts=ts, mid=mid, microprice=mid)


def bar_reads(*, t_event=20, obs=None, lab=None) -> BarBookReads:
    return BarBookReads(t_event=t_event,
                        observable=obs if obs is not None else obs_read(),
                        label=lab if lab is not None else label_read())


def delta(origin, received=None, *, side="bid", price=100.0, size=1.0):
    """Book-event fixture: received defaults to origin (an undelayed event)."""
    return BookDelta(origin_time=origin,
                     received_time=origin if received is None else received,
                     seq=next(_SEQ), side=side, price=price, size=size)


def two_sided(origin, *, bid=100.0, bid_size=2.0, ask=101.0, ask_size=3.0):
    return [delta(origin, side="bid", price=bid, size=bid_size),
            delta(origin, side="ask", price=ask, size=ask_size)]


def one_read(events, t_event, *, cap=10_000) -> BarBookReads:
    (r,) = list(dual_book_reads(events, [t_event], staleness_cap_ns=cap))
    assert isinstance(r, BarBookReads), r
    return r


# ------------------------------------------------------------ hand-computed math

def test_hand_computed_spread_drift_slippage_fee_and_total_cost():
    # obs: bid 99 / ask 101 -> mid 100, spread 2 -> half_spread = 100 bps exactly
    # true t_event mid 100.5 -> drift = |100.5/100 - 1| * 1e4 ~= 50 bps
    a = assumption()  # taker 1.75, base slippage 0.25
    row = cost_row(bar_reads(lab=label_read(mid=100.5)), assumption=a)
    assert isinstance(row, CostRow)
    assert row.t_event == 20
    assert row.half_spread_bps == 0.5 * 2.0 / 100.0 * 1e4 == 100.0
    assert row.latency_drift_bps == pytest.approx(50.0, rel=1e-12)
    assert row.slippage_bps == 0.25 + row.latency_drift_bps
    assert row.cost_bps == 2.0 * 1.75 + row.slippage_bps
    assert row.cost_bps == pytest.approx(3.5 + 0.25 + 50.0, rel=1e-12)


def test_downward_drift_is_charged_the_same_as_upward():
    up = cost_row(bar_reads(lab=label_read(mid=100.5)), assumption=assumption())
    dn = cost_row(bar_reads(lab=label_read(mid=99.5)), assumption=assumption())
    assert dn.latency_drift_bps == pytest.approx(up.latency_drift_bps, rel=1e-12)
    assert dn.latency_drift_bps > 0.0


def test_zero_drift_when_true_mid_equals_observable_mid():
    a = assumption(taker_fee_bps=2.0, base_slippage_bps=0.5)
    row = cost_row(bar_reads(lab=label_read(mid=100.0)), assumption=a)
    assert row.latency_drift_bps == 0.0
    assert row.slippage_bps == 0.5
    assert row.cost_bps == 2.0 * 2.0 + 0.5


def test_zero_drift_and_exact_timestamp_through_the_t2_stream():
    # undelayed events at exactly t_event: target_read_ts == t_event == label cut,
    # both reads coincide -> drift is exactly zero
    r = one_read(two_sided(20), 20)
    assert r.observable.target_read_ts == r.t_event == r.label.label_cut_ts == 20
    row = cost_row(r, assumption=assumption(taker_fee_bps=1.0,
                                            base_slippage_bps=0.0))
    assert row.latency_drift_bps == 0.0
    assert row.slippage_bps == 0.0
    assert row.cost_bps == 2.0
    assert row.half_spread_bps == 0.5 * 1.0 / 100.5 * 1e4


def test_divergent_observable_and_true_mids_charge_drift_not_the_label():
    # an in-flight straggler (origin <= t_event < received) moves the TRUE book
    # only: the observable spread must not see it (§C.2), the label anchor is
    # not shifted, and the realized [target_read_ts, t_event] move is charged
    # forward as slippage (§G / plan #1)
    quiet = one_read(two_sided(10), 20)
    r = one_read(two_sided(10) + [delta(15, 40, side="ask", price=100.4,
                                        size=9.0)], 20)
    assert r.observable == quiet.observable          # straggler invisible to obs
    assert r.label.mid == (100.0 + 100.4) / 2.0      # but in the true t_event book
    a = assumption()
    row, quiet_row = (cost_row(x, assumption=a) for x in (r, quiet))
    assert row.half_spread_bps == quiet_row.half_spread_bps
    assert quiet_row.latency_drift_bps == 0.0
    assert row.latency_drift_bps == pytest.approx(
        abs(100.2 / 100.5 - 1.0) * 1e4, rel=1e-12)
    assert row.cost_bps == quiet_row.cost_bps + row.latency_drift_bps


def test_fees_doubled_exactly_once_and_spread_not_in_cost_bps():
    # huge spread, zero drift/slippage: cost_bps must be exactly 2 fees and the
    # spread must surface ONLY in half_spread_bps (net_pnl owns the crossings)
    a = assumption(taker_fee_bps=3.0, base_slippage_bps=0.0)
    row = cost_row(bar_reads(obs=obs_read(bb=90.0, ba=110.0),
                             lab=label_read(mid=100.0)), assumption=a)
    assert row.cost_bps == 6.0                       # 2 * 3.0, doubled once
    assert row.half_spread_bps == 0.5 * 20.0 / 100.0 * 1e4 == 1000.0
    zero_fee = cost_row(bar_reads(obs=obs_read(bb=90.0, ba=110.0),
                                  lab=label_read(mid=100.0)),
                        assumption=assumption(taker_fee_bps=0.0,
                                              base_slippage_bps=0.0))
    assert zero_fee.cost_bps == 0.0                  # no hidden spread term
    assert zero_fee.half_spread_bps == 1000.0


def test_rows_are_deterministic_and_finite():
    a = assumption()
    events = two_sided(10) + [delta(12, 18, side="bid", price=100.5, size=1.0),
                              delta(15, 40, side="ask", price=100.8, size=2.0)]
    rows = [cost_row(one_read(list(events), 20), assumption=a)
            for _ in range(2)]
    assert rows[0] == rows[1]
    assert all(math.isfinite(v) for v in rows[0][1:])
    assert rows[0].half_spread_bps > 0.0 and rows[0].cost_bps >= 0.0


# ----------------------------------------------------------- fail-closed inputs

def test_rejects_crossed_and_locked_observable_books():
    with pytest.raises(ValueError, match="crossed"):
        cost_row(bar_reads(obs=obs_read(bb=101.0, ba=99.0)),
                 assumption=assumption())
    with pytest.raises(ValueError, match="crossed"):
        cost_row(bar_reads(obs=obs_read(bb=100.0, ba=100.0)),
                 assumption=assumption())


@pytest.mark.parametrize("field,value", [
    ("bb", float("nan")), ("bb", float("inf")), ("bb", 0.0), ("bb", -1.0),
    ("ba", float("nan")), ("ba", 0.0), ("ba", -5.0), ("ba", None),
    ("bb", True), ("ba", True),        # bool-as-1.0 must not price a row
])
def test_rejects_nonfinite_nonpositive_or_missing_best_prices(field, value):
    # a one-sided book cannot reach T7 as a valid BarBookReads (T2 drops it);
    # here it surfaces as a missing/invalid side and must raise, not price a row
    with pytest.raises(ValueError, match="best_"):
        cost_row(bar_reads(obs=obs_read(**{field: value})),
                 assumption=assumption())


@pytest.mark.parametrize("field,value", [
    ("bb_size", 0.0), ("bb_size", -2.0), ("bb_size", float("nan")),
    ("ba_size", 0.0), ("ba_size", float("inf")), ("ba_size", True),
])
def test_rejects_nonpositive_or_nonfinite_sizes(field, value):
    with pytest.raises(ValueError, match="size"):
        cost_row(bar_reads(obs=obs_read(**{field: value})),
                 assumption=assumption())


def test_rejects_inconsistent_or_nonfinite_observable_mid():
    with pytest.raises(ValueError, match="mid"):
        cost_row(bar_reads(obs=obs_read(mid=float("nan"))),
                 assumption=assumption())
    # the read's own mid is the half-spread denominator; a mid that is not the
    # arithmetic top-of-book mid is a corrupted/incompatible read
    with pytest.raises(ValueError, match="mid"):
        cost_row(bar_reads(obs=obs_read(mid=99.0)), assumption=assumption())


@pytest.mark.parametrize("mid", [0.0, -100.0, float("nan"), float("inf"), None,
                                 True])
def test_rejects_invalid_true_t_event_mid(mid):
    # the match pins the INPUT check specifically: the nonfinite-OUTPUT guard's
    # message does not contain this phrase, so nan/inf cannot pass by merely
    # overflowing downstream (they must be refused at the read itself)
    with pytest.raises(ValueError, match=r"label \(true t_event\) mid"):
        cost_row(bar_reads(lab=label_read(mid=mid)), assumption=assumption())


def test_rejects_target_read_ts_after_t_event():
    # an observable read newer than the decision is broken causality upstream
    with pytest.raises(ValueError, match="target_read_ts"):
        cost_row(bar_reads(t_event=20, obs=obs_read(ts=21)),
                 assumption=assumption())


def test_rejects_label_cut_not_equal_to_t_event():
    for bad_ts in (19, 21):
        with pytest.raises(ValueError, match="label"):
            cost_row(bar_reads(t_event=20, lab=label_read(ts=bad_ts)),
                     assumption=assumption())


def test_rejects_a_drift_ratio_that_overflows_to_nonfinite():
    # pathological but individually-finite inputs: the mid ratio overflows to
    # inf, and a nonfinite row must never be emitted (it would ride until
    # eval.matrix.validate_matrix instead of failing at the source)
    obs = obs_read(bb=1e-320, ba=1e-308)
    with pytest.raises(ValueError, match="finite"):
        cost_row(bar_reads(obs=obs, lab=label_read(mid=1e308)),
                 assumption=assumption())


def test_rejects_non_integer_timestamps():
    with pytest.raises(ValueError, match="t_event"):
        cost_row(bar_reads(t_event=20.0), assumption=assumption())
    with pytest.raises(ValueError, match="t_event"):
        cost_row(bar_reads(t_event=True), assumption=assumption())
    # the nested read timestamps are guarded independently of reads.t_event
    with pytest.raises(ValueError, match="target_read_ts"):
        cost_row(bar_reads(obs=obs_read(ts=20.0)), assumption=assumption())
    with pytest.raises(ValueError, match="label_cut_ts"):
        cost_row(bar_reads(lab=label_read(ts=20.0)), assumption=assumption())


# ----------------------------------------------- assumption contract / identity

def test_assumption_is_immutable_and_json_serializable():
    a = assumption()
    with pytest.raises(AttributeError):
        a.taker_fee_bps = 9.9
    d = a.as_dict()
    assert d == json.loads(json.dumps(d))
    assert d == {"venue": "binance", "product": PRODUCT, "source": SOURCE,
                 "version": "test-v1", "taker_fee_bps": 1.75,
                 "base_slippage_bps": 0.25, "drift_policy": DRIFT_POLICY}
    assert CostAssumption(**d) == a                  # manifest round-trip (T8)


def test_as_dict_validates_before_serializing():
    # the T8 manifest-persistence path: an invalid assumption must never
    # serialize, or a bad venue/fee could reach the manifest sources block
    with pytest.raises(ValueError, match="venue"):
        assumption(venue="okx").as_dict()
    with pytest.raises(ValueError, match="taker_fee_bps"):
        assumption(taker_fee_bps=-1.0).as_dict()


def test_no_production_fee_tier_is_baked_in():
    # every identity and fee field is required: T10 selects and freezes the
    # real Binance tier; T7 must not invent one via defaults
    with pytest.raises(TypeError):
        CostAssumption()
    with pytest.raises(TypeError):
        CostAssumption(venue="binance", product=PRODUCT, source=SOURCE,
                       version="v1")


@pytest.mark.parametrize("venue", ["", "BINANCE", "okx", None, 7])
def test_rejects_unknown_or_missing_venue(venue):
    with pytest.raises(ValueError, match="venue"):
        validate_cost_assumption(assumption(venue=venue))


@pytest.mark.parametrize("field", ["product", "source", "version"])
@pytest.mark.parametrize("value", ["", None, 3])
def test_rejects_blank_or_non_string_identity(field, value):
    with pytest.raises(ValueError, match=field):
        validate_cost_assumption(assumption(**{field: value}))


@pytest.mark.parametrize("field", ["taker_fee_bps", "base_slippage_bps"])
@pytest.mark.parametrize("value", [-0.1, float("nan"), float("inf"), None,
                                   True, "1.0"])
def test_rejects_negative_or_nonfinite_fee_parameters(field, value):
    with pytest.raises(ValueError, match=field):
        validate_cost_assumption(assumption(**{field: value}))


def test_zero_fee_and_zero_slippage_are_legal_synthetic_values():
    validate_cost_assumption(assumption(taker_fee_bps=0.0,
                                        base_slippage_bps=0.0))


def test_rejects_unknown_drift_policy():
    with pytest.raises(ValueError, match="drift"):
        validate_cost_assumption(assumption(drift_policy="mystery_v9"))


def test_cost_row_validates_the_assumption_before_pricing():
    with pytest.raises(ValueError, match="venue"):
        cost_row(bar_reads(), assumption=assumption(venue="okx"))


def test_identity_binding_rejects_cross_venue_aliasing():
    bn = assumption(venue="binance")
    require_assumption_identity(bn, venue="binance", product=PRODUCT,
                                source=SOURCE)          # exact match passes
    with pytest.raises(ValueError, match="venue"):
        require_assumption_identity(bn, venue="coinbase", product=PRODUCT,
                                    source=SOURCE)
    cb = assumption(venue="coinbase", product="BTC-USD",
                    source="coinapi/coinbase_l2@stitch-2025")
    with pytest.raises(ValueError, match="venue"):
        require_assumption_identity(cb, venue="binance", product=PRODUCT,
                                    source=SOURCE)


def test_identity_binding_rejects_product_and_source_mismatch():
    bn = assumption()
    with pytest.raises(ValueError, match="product"):
        require_assumption_identity(bn, venue="binance", product="BTC-USD",
                                    source=SOURCE)
    with pytest.raises(ValueError, match="source"):
        require_assumption_identity(bn, venue="binance", product=PRODUCT,
                                    source="somewhere/else")


def test_identity_binding_validates_the_assumption_itself():
    bad = assumption(taker_fee_bps=float("nan"))
    with pytest.raises(ValueError, match="taker_fee_bps"):
        require_assumption_identity(bad, venue="binance", product=PRODUCT,
                                    source=SOURCE)


def test_binance_and_coinbase_serializations_are_distinct():
    bn = assumption(venue="binance").as_dict()
    cb = assumption(venue="coinbase", product="BTC-USD",
                    source="coinapi/coinbase_l2@stitch-2025").as_dict()
    assert bn["venue"] != cb["venue"]
    assert bn["product"] != cb["product"]
    assert bn["source"] != cb["source"]


# ------------------------------------------------------------------- causality

def test_post_t_event_mutation_cannot_change_the_cost_row():
    a = assumption()
    base = two_sided(10) + [delta(15, 40, side="ask", price=100.4, size=9.0)]
    tail = [delta(30, side="ask", price=100.2, size=9.0)]
    row = cost_row(one_read(base + tail, 20), assumption=a)
    # mutate everything strictly after t_event on the origin axis
    mutated = [delta(30, side="ask", price=50.0, size=999.0),
               delta(31, side="bid", price=99.0, size=0.0),
               delta(35, side="ask", price=200.0, size=1.0)]
    assert cost_row(one_read(base + mutated, 20), assumption=a) == row
    # and removing the tail entirely changes nothing either
    assert cost_row(one_read(list(base), 20), assumption=a) == row


def test_late_receipt_can_only_charge_drift_never_move_the_spread():
    # information the box had not received by t_event must never enter the
    # OBSERVABLE spread; it may only change the charged (label-side) drift
    a = assumption()
    quiet_row = cost_row(one_read(two_sided(10), 20), assumption=a)
    loud_row = cost_row(one_read(
        two_sided(10) + [delta(15, 40, side="ask", price=100.2, size=5.0)], 20),
        assumption=a)
    assert loud_row.half_spread_bps == quiet_row.half_spread_bps
    assert loud_row.latency_drift_bps > quiet_row.latency_drift_bps == 0.0


# ---------------------------------------------------- eval.cost.net_pnl wiring

def test_net_pnl_no_trade_band_uses_row_cost_plus_two_spread_crossings():
    # zero drift, taker 1.0, base slippage 0.5 -> cost_bps = 2.5;
    # bid 99.99 / ask 100.01 -> half_spread ~= 1 bp; band = cost + 2 * hs ~= 4.5
    a = assumption(taker_fee_bps=1.0, base_slippage_bps=0.5)
    bb, ba = 99.99, 100.01
    mid = (bb + ba) / 2.0
    row = cost_row(bar_reads(obs=obs_read(bb=bb, ba=ba),
                             lab=label_read(mid=mid)), assumption=a)
    band = row.cost_bps + 2.0 * row.half_spread_bps
    pnl, traded, gross = net_pnl([band * 0.999, band * 1.001], [10.0, 10.0],
                                 cost_bps=[row.cost_bps] * 2,
                                 half_spread_bps=[row.half_spread_bps] * 2)
    assert traded.tolist() == [False, True]          # band gates exactly there
    assert pnl[0] == 0.0
    assert pnl[1] == 10.0 - band                     # spread charged by net_pnl,
    assert gross[1] == 10.0                          # exactly twice, never inside
    assert row.cost_bps == 2.5                       # ...cost_bps itself
