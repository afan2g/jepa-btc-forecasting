"""Synthetic unit tests for the CoinAPI L3 → top-K L2 replay (recon/coinapi.py).

No vendor access: every stream is hand-built with the downloader Parquet schema
(docs/data.md §4.3) so the replay semantics are pinned exactly."""
import datetime as dt
import math

import pandas as pd
import pytest

from recon.coinapi import L3Book, coinapi_frame_from_rows, reconstruct_coinapi_l2_at_samples

DAY = dt.date(2025, 6, 1)
DAY_OPEN = pd.Timestamp("2025-06-01").value
S = 1_000_000_000  # one second in ns
PRIOR_CLOSE = 86_399_999_000_000  # 23:59:59.999 — the prior-day-close stamp on the SNAPSHOT


def at(*offsets_s):
    """Grid timestamps at the given second offsets after the partition-day open."""
    return [DAY_OPEN + int(o * S) for o in offsets_s]


def reco(rows, *, k=2, sample_s=(100,), **kw):
    df = coinapi_frame_from_rows(rows)
    return reconstruct_coinapi_l2_at_samples(df, k=k, day=DAY, sample_ts=at(*sample_s), **kw)


# --------------------------------------------------------------------------- snapshot-first
def test_snapshot_applied_first_even_when_its_timestamp_looks_like_prior_day_close():
    # SNAPSHOT rows carry 23:59:59.999 (prior-day close); a non-snapshot ADD arrives at +5s.
    # Sampling at +1s must already see the FULL snapshot book (clamp pushes the snapshot to
    # day-open), and must NOT yet see the +5s ADD (apply-before-read, no look-ahead).
    rows = [
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="b1", time_exchange_ns=PRIOR_CLOSE),
        dict(update_type="SNAPSHOT", is_buy=False, entry_px=101.0, entry_sx=3.0,
             order_id="a1", time_exchange_ns=PRIOR_CLOSE),
        dict(update_type="ADD", is_buy=True, entry_px=99.0, entry_sx=1.0,
             order_id="b2", time_exchange_ns=5 * S),
    ]
    frame, q = reco(rows, k=2, sample_s=(1, 10))
    f = frame.set_index("sample_ts")
    assert f.loc[at(1)[0], "bid_0_price"] == 100.0  # snapshot present at +1s
    assert f.loc[at(1)[0], "ask_0_price"] == 101.0
    assert math.isnan(f.loc[at(1)[0], "bid_1_price"])  # ADD@+5s not yet visible at +1s
    assert f.loc[at(10)[0], "bid_1_price"] == 99.0     # ADD visible by +10s
    assert q["snapshot_rows"] == 2


# --------------------------------------------------------------------------- seq, not time
def test_replay_uses_seq_order_not_timestamp_order():
    # In seq order: bid102 is ADDed then removed (SUB→0), so best bid is 100. If the engine
    # sorted by time_exchange instead, the SUB (t=5s) would precede the ADD (t=10s), hit a
    # missing order, be skipped, and leave bid102 standing → best bid 102. Asserting 100
    # proves seq order drives apply; the backward stamp is counted, not obeyed.
    rows = [
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="b0", time_exchange_ns=PRIOR_CLOSE),
        dict(update_type="ADD", is_buy=True, entry_px=102.0, entry_sx=1.0,
             order_id="O1", time_exchange_ns=10 * S),
        dict(update_type="SUB", is_buy=True, entry_px=102.0, entry_sx=0.0,
             order_id="O1", time_exchange_ns=5 * S),  # EARLIER stamp than its own ADD
    ]
    frame, q = reco(rows, k=2)
    assert frame.iloc[-1]["bid_0_price"] == 100.0
    assert q["time_regressions"] >= 1
    assert q.get("missing_order", 0) == 0  # SUB found O1 (seq order), so no missing-order


# --------------------------------------------------------------------------- update-type L2
def test_add_set_sub_match_delete_aggregate_to_expected_l2_absolute():
    rows = [
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=100.0, entry_sx=5.0, order_id="O1"),
        dict(update_type="SNAPSHOT", is_buy=False, entry_px=101.0, entry_sx=4.0, order_id="O2"),
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=3.0,
             order_id="O3", time_exchange_ns=1 * S),  # bid100 -> 8
        dict(update_type="ADD", is_buy=False, entry_px=102.0, entry_sx=2.0,
             order_id="O4", time_exchange_ns=2 * S),  # asks: 101@4, 102@2
        dict(update_type="SET", is_buy=False, entry_px=101.0, entry_sx=1.0,
             order_id="O2", time_exchange_ns=3 * S),  # O2 4 -> 1 ; ask101 = 1
        dict(update_type="SUB", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="O3", time_exchange_ns=4 * S),  # absolute: O3 -> 2 ; bid100 = 5+2 = 7
        dict(update_type="MATCH", is_buy=True, entry_px=100.0, entry_sx=0.0,
             order_id="O1", time_exchange_ns=5 * S),  # absolute: O1 removed ; bid100 = 2
        dict(update_type="DELETE", is_buy=False, entry_px=102.0, entry_sx=0.0,
             order_id="O4", time_exchange_ns=6 * S),  # ask102 gone ; asks: 101@1
    ]
    frame, q = reco(rows, k=2)
    r = frame.iloc[-1]
    assert (r["bid_0_price"], r["bid_0_size"]) == (100.0, 2.0)  # only O3 @ 2 left
    assert math.isnan(r["bid_1_price"])
    assert (r["ask_0_price"], r["ask_0_size"]) == (101.0, 1.0)  # only O2 @ 1 left
    assert math.isnan(r["ask_1_price"])
    assert r["mid"] == 100.5
    assert (q["snapshot_rows"], q["add"], q["set"], q["sub"], q["match"], q["delete"]) \
        == (2, 2, 1, 1, 1, 1)
    assert q.get("missing_order", 0) == 0


def test_decrement_policy_subtracts_matched_size():
    rows = [
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=100.0, entry_sx=5.0, order_id="O1"),
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=3.0,
             order_id="O2", time_exchange_ns=1 * S),                       # bid100 = 8
        dict(update_type="SUB", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="O2", time_exchange_ns=2 * S),                       # O2 3-2=1 ; bid100 = 6
        dict(update_type="MATCH", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="O1", time_exchange_ns=3 * S),                       # O1 5-5=0 remove ; bid100 = 1
    ]
    frame, _ = reco(rows, k=2, size_policy="decrement")
    r = frame.iloc[-1]
    assert (r["bid_0_price"], r["bid_0_size"]) == (100.0, 1.0)  # only O2 @ 1 remains


def test_absolute_and_decrement_diverge_on_reducing_events():
    # Same byte-stream, two policies: under absolute, SUB entry_sx=2 means "remaining 2";
    # under decrement it means "remove 2". They must NOT agree → proves the policy is live.
    rows = [
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=5.0, order_id="O1"),
        dict(update_type="SUB", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="O1", time_exchange_ns=1 * S),
    ]
    abs_frame, _ = reco(rows, k=1, size_policy="absolute")
    dec_frame, _ = reco(rows, k=1, size_policy="decrement")
    assert abs_frame.iloc[-1]["bid_0_size"] == 2.0  # remaining size
    assert dec_frame.iloc[-1]["bid_0_size"] == 3.0  # 5 - 2


# --------------------------------------------------------------------------- unknown types
def test_unknown_update_type_is_counted_and_skipped_by_default():
    rows = [
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=1.0, order_id="O1"),
        dict(update_type="WAT", is_buy=True, entry_px=100.0, entry_sx=9.0,
             order_id="O1", time_exchange_ns=1 * S),  # must not touch the book
    ]
    frame, q = reco(rows, k=1)
    assert frame.iloc[-1]["bid_0_size"] == 1.0  # unchanged
    assert q["unknown_total"] == 1 and q["unknown:WAT"] == 1


def test_unknown_update_type_raises_under_strict_policy():
    df = coinapi_frame_from_rows([
        dict(update_type="WAT", is_buy=True, entry_px=100.0, entry_sx=1.0, order_id="O1"),
    ])
    with pytest.raises(ValueError, match="unknown CoinAPI update_type"):
        reconstruct_coinapi_l2_at_samples(df, k=1, day=DAY, sample_ts=at(1), on_unknown="raise")


# --------------------------------------------------------------------------- quality signals
def test_missing_seed_is_surfaced_when_reducing_an_unseen_order():
    # No snapshot, and a DELETE/SUB referencing orders we never ADDed → missing-seed signal.
    rows = [
        dict(update_type="DELETE", is_buy=True, entry_px=100.0, entry_sx=0.0, order_id="ghost1"),
        dict(update_type="SUB", is_buy=False, entry_px=101.0, entry_sx=1.0,
             order_id="ghost2", time_exchange_ns=1 * S),
    ]
    frame, q = reco(rows, k=1)
    assert q["missing_order"] == 2
    # book never seeded → top-of-book missing on every sample
    assert q["missing_book_fraction"] == 1.0


def test_crossed_book_is_surfaced_in_quality_metrics():
    # A crossed seed (bid 101 ≥ ask 100) must show up as a non-zero crossed_rate, not be hidden.
    rows = [
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=101.0, entry_sx=1.0, order_id="b"),
        dict(update_type="SNAPSHOT", is_buy=False, entry_px=100.0, entry_sx=1.0, order_id="a"),
    ]
    frame, q = reco(rows, k=1, sample_s=(1, 2, 3))
    assert q["crossed_rate"] == 1.0 and q["crossed_samples"] == 3


def test_invalid_policy_args_raise():
    with pytest.raises(ValueError, match="size_policy"):
        L3Book(size_policy="nope")
    with pytest.raises(ValueError, match="on_unknown"):
        L3Book(on_unknown="nope")
