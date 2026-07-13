"""Dual-cut target-book snapshots + staleness gate (issue #74 / plan §B, §C.2, T2).

Covers: the two distinct per-bar target-book reads (observable = received-gated fold
in origin order; label anchor = plain origin cut at t_event), delayed-receipt
exclusion and observable/label divergence, target_read_ts semantics, the exact
staleness boundary, fail-closed missing/one-sided/crossed/invalid/stale books with
stable reasons, equal-timestamp apply-before-read, equal-key input-order folding,
post-t_event mutation no-lookahead, deterministic rebuilds, and equivalence with the
existing materializing reconstruction helpers used as small fixture oracles."""
import itertools

import pytest

from bars.snapshot import (
    BarBookReads,
    BookDelta,
    LabelBookRead,
    ObservableBookRead,
    SnapshotRejection,
    dual_book_reads,
)
from recon.events import Delta
from recon.orderbook import OrderBook

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


def two_sided(origin, *, bid=100.0, bid_size=2.0, ask=101.0, ask_size=3.0):
    """A minimal valid two-sided book: one bid + one ask delta at `origin`."""
    return [delta(origin, side="bid", price=bid, size=bid_size),
            delta(origin, side="ask", price=ask, size=ask_size)]


def reads(events, t_events, *, cap=10_000):
    return list(dual_book_reads(events, t_events, staleness_cap_ns=cap))


# --------------------------------------------------------------- core dual read

def test_dual_read_without_delays_matches_the_hand_computed_book():
    events = (two_sided(10)
              + [delta(30, side="bid", price=100.0, size=0.0),
                 delta(30, side="bid", price=99.0, size=1.0)])
    r0, r1 = reads(events, [20, 40])
    assert isinstance(r0, BarBookReads) and isinstance(r1, BarBookReads)
    assert r0.t_event == 20
    assert r0.observable.target_read_ts == 10
    assert r0.observable.mid == 100.5
    assert r0.observable.microprice == (3.0 * 100.0 + 2.0 * 101.0) / 5.0
    assert r0.label.label_cut_ts == 20
    # no delayed events: the observable and label-anchor reads coincide
    assert (r0.label.mid, r0.label.microprice) == (r0.observable.mid,
                                                   r0.observable.microprice)
    assert r1.observable.target_read_ts == 30
    assert r1.observable.mid == 100.0  # bid 99 / ask 101 after the ts=30 moves
    assert r1.label.label_cut_ts == 40


def test_mid_and_microprice_use_the_existing_orderbook_formulas():
    events = [delta(10, side="bid", price=99.5, size=4.0),
              delta(11, side="ask", price=100.5, size=1.0)]
    ob = OrderBook()
    ob.apply(Delta(10, 1, "bid", 99.5, 4.0))
    ob.apply(Delta(11, 2, "ask", 100.5, 1.0))
    (r,) = reads(events, [15])
    assert r.observable.mid == ob.mid()
    assert r.observable.microprice == ob.microprice()
    assert r.label.mid == ob.mid()
    assert r.label.microprice == ob.microprice()


def test_event_at_exactly_t_event_is_applied_before_the_read():
    # apply-before-read is inclusive on both axes: origin == t_event enters the
    # label cut, received == t_event passes the observability gate
    events = two_sided(10) + [delta(20, side="ask", price=101.0, size=9.0)]
    (r,) = reads(events, [20])
    assert r.observable.target_read_ts == 20
    assert r.observable.microprice == (9.0 * 100.0 + 2.0 * 101.0) / 11.0
    assert r.label.microprice == r.observable.microprice


def test_observable_and_label_read_types_cannot_be_confused():
    # the two roles expose deliberately distinct timestamp field names; a caller
    # cannot swap them without an AttributeError
    assert "target_read_ts" in ObservableBookRead._fields
    assert "target_read_ts" not in LabelBookRead._fields
    assert "label_cut_ts" in LabelBookRead._fields
    assert "label_cut_ts" not in ObservableBookRead._fields


# ------------------------------------------- delayed receipt / no-lookahead

def test_delayed_event_is_excluded_from_the_observable_read_but_not_the_label():
    # a delta with origin <= t_event but received > t_event: the box could not
    # see it, so the observable read must not fold it — but the TRUE book at the
    # origin cut t_event did contain it, so the label anchor must (§C.2/#1/#2)
    events = two_sided(10) + [delta(15, 40, side="ask", price=101.0, size=9.0)]
    (r,) = reads(events, [20])
    assert isinstance(r, BarBookReads)
    assert r.observable.target_read_ts == 10          # the straggler is invisible
    assert r.observable.microprice == (3.0 * 100.0 + 2.0 * 101.0) / 5.0
    assert r.label.microprice == (9.0 * 100.0 + 2.0 * 101.0) / 11.0
    assert r.label.mid == r.observable.mid == 100.5   # mid untouched, only size moved


def test_observable_and_label_reads_diverge_exactly_when_delayed_events_exist():
    # regression pin for the role distinction: same world, one bar before the
    # delayed price move is observable, one after — divergence only in between
    events = two_sided(10) + [delta(15, 40, side="ask", price=100.6, size=3.0)]
    r_during, r_after = reads(events, [20, 45])
    assert r_during.observable.mid == 100.5   # old ask 101
    assert r_during.label.mid == 100.3        # true book already has ask 100.6
    assert r_during.observable.mid != r_during.label.mid
    # once received (40 <= 45), the two reads coincide again
    assert r_after.observable.mid == r_after.label.mid == 100.3
    assert r_after.observable.target_read_ts == 15


def test_events_received_after_t_event_cannot_mutate_observable_outputs():
    # byte-identical observable outputs whether the delayed events exist or not:
    # post-t_event receipts must be unable to change what the box observed
    base = two_sided(10) + [delta(12, side="bid", price=99.5, size=1.0)]
    delayed = [delta(11, 30, side="bid", price=100.2, size=5.0),
               delta(14, 50, side="ask", price=100.9, size=7.0)]
    merged = sorted(base + delayed, key=lambda e: (e.origin_time, e.seq))
    (with_delayed,) = reads(merged, [20])
    (without,) = reads(base, [20])
    assert with_delayed.observable == without.observable
    # while the label anchor DOES see them — that difference is the whole point
    assert with_delayed.label != without.label


def test_straggler_becomes_observable_later_and_folds_in_origin_order():
    # the straggler (origin 11) is received AFTER a later-origin event (origin 30)
    # was already folded; on promotion it must fold AS IF replayed in origin
    # order — the later-origin write to the same level wins, the straggler is a
    # no-op there, but it still lands on a level nothing overwrote
    events = (two_sided(10)
              + [delta(11, 45, side="ask", price=101.0, size=9.0),   # overwritten level
                 delta(11, 45, side="bid", price=99.0, size=4.0),    # untouched level
                 delta(30, 30, side="ask", price=101.0, size=6.0)])  # newer same-level
    r_before, r_after = reads(events, [40, 50])
    # before receipt: only the undelayed events are observable
    assert r_before.observable.microprice == (6.0 * 100.0 + 2.0 * 101.0) / 8.0
    # after receipt: ask@101 keeps the NEWER origin-30 size (6.0), not the
    # straggler's 9.0; bid@99 now exists (size 4.0) but best bid is still 100
    assert r_after.observable.microprice == (6.0 * 100.0 + 2.0 * 101.0) / 8.0
    assert r_after.observable.mid == 100.5
    # target_read_ts is the origin of the LAST observable event on the origin
    # axis — origin 30 (not the straggler's 11, not its receipt time 45)
    assert r_after.observable.target_read_ts == 30


def test_delayed_removal_is_invisible_until_received_then_empties_the_level():
    # a removal (size 0) received late: the observable book keeps the level until
    # receipt; afterwards the level is gone and an OLDER straggler cannot
    # resurrect it (tombstone semantics of the origin-order fold)
    events = (two_sided(10)
              + [delta(10, side="bid", price=99.5, size=1.0),        # surviving level
                 delta(12, 70, side="bid", price=100.0, size=8.0),   # older straggler
                 delta(20, 60, side="bid", price=100.0, size=0.0)])  # delayed removal
    r1, r2, r3 = reads(events, [30, 65, 75])
    assert r1.observable.mid == 100.5     # removal not yet visible: best bid 100
    assert r2.observable.mid == 100.25    # removal received at 60: best bid 99.5
    assert r3.observable.mid == 100.25    # origin-12 straggler < origin-20 removal:
    assert r3.observable.target_read_ts == 20  # the level stays gone (tombstone)


# ------------------------------------------------------------- staleness gate

def test_staleness_exactly_at_the_cap_passes_and_one_ns_over_fails():
    events = two_sided(10)
    at_cap, over_cap = reads(events, [110, 111], cap=100)
    assert isinstance(at_cap, BarBookReads)          # age 100 == cap: usable
    assert at_cap.observable.target_read_ts == 10
    assert isinstance(over_cap, SnapshotRejection)   # age 101 > cap: stale
    assert (over_cap.role, over_cap.reason) == ("observable", "stale_book")
    assert over_cap.t_event == 111


def test_zero_staleness_cap_requires_a_book_event_at_t_event_itself():
    events = two_sided(10) + two_sided(20)
    fresh, stale = reads(events, [20, 21], cap=0)
    assert isinstance(fresh, BarBookReads)
    assert isinstance(stale, SnapshotRejection)
    assert stale.reason == "stale_book"


def test_a_delayed_fresher_event_does_not_reset_staleness_until_received():
    # the only recent book event is received after t_event: the observable book
    # is still the stale origin-10 state and must be dropped, even though the
    # true (label) book has fresh data — staleness is an OBSERVABLE property
    events = two_sided(10) + [delta(200, 400, side="ask", price=101.0, size=5.0)]
    (r,) = reads(events, [250], cap=100)
    assert isinstance(r, SnapshotRejection)
    assert (r.role, r.reason) == ("observable", "stale_book")


# ------------------------------------------------- fail-closed book rejections

def test_no_observable_events_rejects_as_missing_book():
    (r,) = reads([], [20])
    assert isinstance(r, SnapshotRejection)
    assert (r.t_event, r.role, r.reason) == (20, "observable", "missing_book")


def test_events_exist_but_none_received_yet_rejects_as_missing_book():
    events = [e._replace(received_time=90) for e in two_sided(10)]
    (r,) = reads(events, [20])
    assert (r.role, r.reason) == ("observable", "missing_book")


def test_one_sided_observable_book_rejects_with_stable_reason():
    (only_bid,) = reads([delta(10, side="bid")], [20])
    assert (only_bid.role, only_bid.reason) == ("observable", "one_sided_book")
    (only_ask,) = reads([delta(10, side="ask")], [20])
    assert (only_ask.role, only_ask.reason) == ("observable", "one_sided_book")


def test_crossed_and_locked_observable_books_reject_with_stable_reason():
    crossed = [delta(10, side="bid", price=101.5), delta(10, side="ask", price=101.0)]
    (r,) = reads(crossed, [20])
    assert (r.role, r.reason) == ("observable", "crossed_book")
    locked = [delta(10, side="bid", price=101.0), delta(10, side="ask", price=101.0)]
    (r,) = reads(locked, [20])
    assert (r.role, r.reason) == ("observable", "crossed_book")


def test_label_book_rejections_carry_the_label_role():
    # observable book is clean; a DELAYED removal empties the true book's ask
    # side at the origin cut, so the label role fails closed
    events = two_sided(10) + [delta(15, 90, side="ask", price=101.0, size=0.0)]
    (r,) = reads(events, [20])
    assert isinstance(r, SnapshotRejection)
    assert (r.role, r.reason) == ("label", "one_sided_book")
    # a delayed crossing bid does the same with reason crossed_book
    events = two_sided(10) + [delta(15, 90, side="bid", price=102.0, size=1.0)]
    (r,) = reads(events, [20])
    assert (r.role, r.reason) == ("label", "crossed_book")


def test_validate_book_top_rejects_invalid_tops_directly():
    from bars.snapshot import validate_book_top
    ob = OrderBook()
    assert validate_book_top(ob) == ("missing_book",
                                     "book has no levels on either side")
    ob.bids[100.0] = 2.0
    reason, _ = validate_book_top(ob)
    assert reason == "one_sided_book"
    ob.asks[101.0] = float("nan")           # defensive: a corrupted size
    reason, _ = validate_book_top(ob)
    assert reason == "invalid_book"
    ob.asks[101.0] = 3.0
    assert validate_book_top(ob) is None
    ob.asks.pop(101.0)
    ob.asks[99.0] = 3.0                      # bid 100 >= ask 99
    reason, _ = validate_book_top(ob)
    assert reason == "crossed_book"


# ------------------------------------------------------ contract guard rails

def test_out_of_order_events_fail_closed():
    events = [delta(20, seq=5), delta(10, seq=6, side="ask")]
    with pytest.raises(ValueError, match="order"):
        reads(events, [30])
    same_origin_seq_regress = [delta(10, seq=7), delta(10, seq=6, side="ask")]
    with pytest.raises(ValueError, match="order"):
        reads(same_origin_seq_regress, [30])


def test_received_before_origin_fails_closed():
    with pytest.raises(ValueError, match="received_time"):
        reads([delta(10, 9)], [20])


def test_malformed_events_fail_closed():
    with pytest.raises(ValueError, match="side"):
        reads([delta(10, side="buy")], [20])
    with pytest.raises(ValueError, match="price"):
        reads([delta(10, price=0.0)], [20])
    with pytest.raises(ValueError, match="price"):
        reads([delta(10, price=float("nan"))], [20])
    with pytest.raises(ValueError, match="size"):
        reads([delta(10, size=-1.0)], [20])
    with pytest.raises(ValueError, match="size"):
        reads([delta(10, size=float("inf"))], [20])


def test_non_increasing_t_events_fail_closed():
    with pytest.raises(ValueError, match="increasing"):
        reads(two_sided(10), [20, 20])
    with pytest.raises(ValueError, match="increasing"):
        reads(two_sided(10), [20, 15])


def test_negative_staleness_cap_fails_closed():
    with pytest.raises(ValueError, match="staleness_cap_ns"):
        reads(two_sided(10), [20], cap=-1)


def test_equal_order_keys_fold_in_original_row_order():
    # two events with the SAME (origin_time, seq) on the same level: the input
    # position is the final tie-break, so the LATER row wins deterministically
    events = [delta(10, side="bid", price=100.0, size=2.0, seq=1),
              delta(10, side="bid", price=100.0, size=7.0, seq=1),
              delta(10, side="ask", price=101.0, size=3.0, seq=2)]
    (r,) = reads(events, [20])
    assert r.observable.microprice == (3.0 * 100.0 + 7.0 * 101.0) / 10.0
    assert r.label.microprice == r.observable.microprice


# ------------------------- determinism + materializing fixture oracles (§C.1)

def _recon_deltas(events):
    """BookDelta -> recon.events.Delta (drops received_time; origin becomes
    ts_engine). Only valid for a set already gated/ordered by the caller."""
    return [Delta(e.origin_time, e.seq, e.side, e.price, e.size) for e in events]


def _oracle_book(events, gate):
    """Materializing fixture oracle: filter by `gate`, fold in (origin, seq)
    order (stable sort keeps input order for equal keys) on a fresh OrderBook."""
    ob = OrderBook()
    for e in sorted((e for e in events if gate(e)), key=lambda e: (e.origin_time, e.seq)):
        ob.apply(Delta(e.origin_time, e.seq, e.side, e.price, e.size))
    return ob


def _random_world(rng, n_events=200):
    """Seeded synthetic day: random walkless two-sided-ish book traffic with
    random receipt lags (some large), removals included; unique (origin, seq)."""
    events, origin = [], 0
    for seq in range(n_events):
        origin += rng.randint(1, 40)
        lag = rng.choice([0, 1, 5, 30, 250, 900])
        side = rng.choice(["bid", "ask"])
        base = 100.0 if side == "bid" else 101.0
        price = base + rng.choice([-0.4, -0.2, 0.0, 0.2, 0.4])
        size = rng.choice([0.0, 1.0, 2.0, 5.0])  # 0.0 = removal
        events.append(BookDelta(origin, origin + lag, seq, side, price, size))
    t_events = sorted(rng.sample(range(50, origin + 400), 25))
    return events, t_events


def test_streaming_reads_match_the_materializing_oracle_on_random_worlds():
    import random

    from bars.snapshot import validate_book_top
    cap = 300
    for seed in (1, 2, 3, 4):
        rng = random.Random(seed)
        events, t_events = _random_world(rng)
        got = reads(events, t_events, cap=cap)
        assert len(got) == len(t_events)
        for t_event, r in zip(t_events, got):
            obs = _oracle_book(events, lambda e: e.received_time <= t_event)
            lab = _oracle_book(events, lambda e: e.origin_time <= t_event)
            observable_keys = [(e.origin_time, e.seq) for e in events
                               if e.received_time <= t_event]
            expect = validate_book_top(obs)
            if expect is None and observable_keys:
                age = t_event - max(observable_keys)[0]
                if age > cap:
                    expect = ("stale_book", None)
            if expect is not None:
                assert isinstance(r, SnapshotRejection)
                assert (r.role, r.reason) == ("observable", expect[0])
                continue
            expect = validate_book_top(lab)
            if expect is not None:
                assert isinstance(r, SnapshotRejection)
                assert (r.role, r.reason) == ("label", expect[0])
                continue
            assert isinstance(r, BarBookReads)
            assert r.observable.target_read_ts == max(observable_keys)[0]
            assert (r.observable.mid, r.observable.microprice) == (obs.mid(),
                                                                   obs.microprice())
            assert (r.label.mid, r.label.microprice) == (lab.mid(), lab.microprice())


def test_rebuilds_are_deterministic():
    import random
    events, t_events = _random_world(random.Random(7))
    assert reads(events, t_events) == reads(events, t_events)
    assert reads(iter(events), iter(t_events)) == reads(events, t_events)


def test_observable_read_matches_sample_topk_as_of_after_the_received_gate():
    # the mandated pattern (§C.1/#2): PRE-FILTER received <= t_event, then reuse
    # the existing shared origin-order sampler — must equal the streaming read
    from recon.reconstruct import sample_topk_as_of
    t_event = 40
    events = (two_sided(10)
              + [delta(15, 90, side="ask", price=100.6, size=3.0),   # delayed
                 delta(30, 35, side="bid", price=100.2, size=4.0)])
    gated = [e for e in events if e.received_time <= t_event]
    df = sample_topk_as_of([t_event], _recon_deltas(gated), k=1, book=OrderBook(),
                           apply=lambda b, d: b.apply(d), time_of=lambda d: d.ts_engine)
    (r,) = reads(events, [t_event])
    assert r.observable.mid == df["mid"].iloc[0]
    assert r.observable.microprice == df["microprice"].iloc[0]


def test_naive_sample_topk_as_of_without_the_gate_is_the_label_read_not_observable():
    # regression pin for the issue's "do not naively use sample_topk_as_of"
    # criterion: the plain origin cut over the RAW stream folds the delayed event
    # and reproduces the LABEL anchor — never the observable read
    from recon.reconstruct import sample_topk_as_of
    t_event = 40
    events = (two_sided(10)
              + [delta(15, 90, side="ask", price=100.6, size=3.0),   # delayed
                 delta(30, 35, side="bid", price=100.2, size=4.0)])
    df = sample_topk_as_of([t_event], _recon_deltas(events), k=1, book=OrderBook(),
                           apply=lambda b, d: b.apply(d), time_of=lambda d: d.ts_engine)
    (r,) = reads(events, [t_event])
    assert df["mid"].iloc[0] == r.label.mid
    assert df["mid"].iloc[0] != r.observable.mid


def test_no_delay_world_matches_reconstruct_book_at_samples():
    # compatibility with existing reconstruction behavior: when nothing is
    # delayed both roles equal the existing grid sampler's book
    from recon.reconstruct import reconstruct_book_at_samples
    events = (two_sided(10)
              + [delta(30, side="bid", price=100.0, size=0.0),
                 delta(30, side="bid", price=99.0, size=1.0),
                 delta(50, side="ask", price=102.0, size=4.0)])
    grid = [20, 40, 60]
    df = reconstruct_book_at_samples(_recon_deltas(events), grid, k=1).set_index("sample_ts")
    for r, g in zip(reads(events, grid), grid):
        assert isinstance(r, BarBookReads)
        assert r.observable.mid == r.label.mid == df.loc[g, "mid"]
        assert r.observable.microprice == r.label.microprice == df.loc[g, "microprice"]


# ------------------------------------------------------------------- streaming

def test_sampler_is_lazy_and_consumes_only_the_needed_event_prefix():
    consumed = []

    def event_stream():
        for e in (two_sided(10) + two_sided(1_000) + two_sided(2_000)):
            consumed.append(e)
            yield e

    out = dual_book_reads(event_stream(), iter([15, 2_500]), staleness_cap_ns=10_000)
    first = next(out)
    assert isinstance(first, BarBookReads)
    # only the origin <= 15 prefix plus ONE lookahead event may be consumed
    assert len(consumed) == 3
    rest = list(out)
    assert len(rest) == 1 and len(consumed) == 6
