"""Within-timestamp ordering for the CoinAPI L3 replay (docs/data.md §10 open item).

CoinAPI `limitbook_full` has NO exchange `sequence_number`; many L3 events share one
`time_exchange_ns` (an ADD and its immediate MATCH routinely carry the same stamp). The
canonical ordering these tests pin (docs/superpowers/plans/2026-07-02-coinapi-within-
timestamp-ordering.md):

  1. File/stream order is canonical — the downloader writes `seq` = CSV row order, and the
     replay applies rows exactly as delivered (chunk by chunk), never re-sorting.
  2. `seq` is a *check*, not a sort key: a strict regression counts `seq_disorder`, a
     duplicate counts `seq_duplicate`; both keep file order.
  3. Ties (same timestamp, and even same `seq`) break by original row index — a stable
     no-op, because file order already IS row-index order.
  4. `order_id` is NEVER an ordering key (lexicographic UUID order is meaningless).

All streams are synthetic (downloader Parquet schema, docs/data.md §4.3) — no vendor access.
"""
import datetime as dt
import math

import pandas as pd

from recon.coinapi import (
    L3Book,
    _iter_actions,
    coinapi_frame_from_rows,
    reconstruct_coinapi_l2_at_samples,
)

DAY = dt.date(2025, 6, 1)
DAY_OPEN = pd.Timestamp("2025-06-01").value
S = 1_000_000_000  # one second in ns
T = 10 * S         # the shared within-timestamp instant used across these tests


def at(*offsets_s):
    """Grid timestamps at the given second offsets after the partition-day open."""
    return [DAY_OPEN + int(o * S) for o in offsets_s]


def reco(rows_or_chunks, *, k=2, sample_s=(100,), **kw):
    """Replay dict rows (built into a downloader-schema frame) or pre-built chunk(s)."""
    kw.setdefault("size_policy", "decrement")  # the VERIFIED Coinbase policy (§5a)
    data = rows_or_chunks
    if isinstance(data, list) and data and isinstance(data[0], dict):
        data = coinapi_frame_from_rows(data)
    return reconstruct_coinapi_l2_at_samples(data, k=k, day=DAY, sample_ts=at(*sample_s), **kw)


# --------------------------------------------------------------- same order, same timestamp
def test_same_ts_add_then_match_on_same_order_applies_in_file_order():
    # ADD posts O1@5, the SAME-timestamp MATCH fully fills it (decrement 5) → order gone,
    # level gone. Under the REVERSED order the MATCH would hit an unseen order (counted
    # `missing_order`, skipped) and the ADD would leave bid 100@5 standing — so this
    # assertion discriminates file order from any same-timestamp reordering.
    rows = [
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="O1", time_exchange_ns=T),
        dict(update_type="MATCH", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="O1", time_exchange_ns=T),
    ]
    frame, q = reco(rows)
    assert math.isnan(frame.iloc[-1]["bid_0_price"])  # order added AND fully filled
    assert q.get("missing_order", 0) == 0             # MATCH found the ADD → order held
    assert (q["add"], q["match"]) == (1, 1)


def test_same_ts_last_write_wins_on_same_order():
    # State-defining events on one order at one timestamp: the LATER row (file order) wins.
    # Reversed, the SET would seed O1@2 (a `set_missing` create) and the ADD would overwrite
    # it back to 5 — final size discriminates the orders.
    rows = [
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="O1", time_exchange_ns=T),
        dict(update_type="SET", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="O1", time_exchange_ns=T),
    ]
    frame, q = reco(rows)
    assert (frame.iloc[-1]["bid_0_price"], frame.iloc[-1]["bid_0_size"]) == (100.0, 2.0)
    assert q.get("set_missing", 0) == 0  # SET saw the ADD, it did not create the order


# ------------------------------------------------------- ADD/MATCH/SUB/DELETE interactions
def test_same_ts_add_match_sub_delete_cluster_yields_exact_final_book():
    # A full lifecycle cluster at ONE timestamp. Every reducing event targets an order
    # created earlier IN FILE ORDER at the same instant, so any permutation that moves a
    # reducer before its creator flips `missing_order` and leaves a different book.
    rows = [
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=4.0,
             order_id="b1", time_exchange_ns=T),
        dict(update_type="ADD", is_buy=True, entry_px=99.0, entry_sx=2.0,
             order_id="b2", time_exchange_ns=T),
        dict(update_type="ADD", is_buy=False, entry_px=101.0, entry_sx=3.0,
             order_id="a1", time_exchange_ns=T),
        dict(update_type="SUB", is_buy=True, entry_px=100.0, entry_sx=1.0,
             order_id="b1", time_exchange_ns=T),    # b1 4→3
        dict(update_type="MATCH", is_buy=True, entry_px=100.0, entry_sx=3.0,
             order_id="b1", time_exchange_ns=T),    # b1 3→0 → level 100 gone
        dict(update_type="DELETE", is_buy=True, entry_px=99.0, entry_sx=0.0,
             order_id="b2", time_exchange_ns=T),    # level 99 gone
        dict(update_type="ADD", is_buy=True, entry_px=98.0, entry_sx=1.5,
             order_id="b3", time_exchange_ns=T),
    ]
    frame, q = reco(rows, k=2)
    r = frame.iloc[-1]
    assert (r["bid_0_price"], r["bid_0_size"]) == (98.0, 1.5)  # only b3 survives
    assert math.isnan(r["bid_1_price"])
    assert (r["ask_0_price"], r["ask_0_size"]) == (101.0, 3.0)
    assert r["mid"] == 99.5
    assert (q["add"], q["sub"], q["match"], q["delete"]) == (4, 1, 1, 1)
    assert q.get("missing_order", 0) == 0
    assert q.get("seq_disorder", 0) == 0 and q.get("seq_duplicate", 0) == 0


# ------------------------------------------------------------------ stable tie-break rules
def test_same_ts_events_yield_in_input_row_order_not_order_id_order():
    # order_ids are ANTI-lexicographic ("z", "m", "a"): any (timestamp, order_id) sort would
    # emit a-m-z. The replay must emit exactly the input row order.
    rows = [
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=1.0,
             order_id="z", time_exchange_ns=T),
        dict(update_type="ADD", is_buy=True, entry_px=99.0, entry_sx=1.0,
             order_id="m", time_exchange_ns=T),
        dict(update_type="ADD", is_buy=True, entry_px=98.0, entry_sx=1.0,
             order_id="a", time_exchange_ns=T),
    ]
    df = coinapi_frame_from_rows(rows)
    book = L3Book(size_policy="decrement")
    actions = list(_iter_actions([df], book, DAY_OPEN))
    assert [a[5] for a in actions] == ["z", "m", "a"]
    assert len({a[0] for a in actions}) == 1  # one shared label time — a genuine tie
    assert book.q.get("seq_disorder", 0) == 0 and book.q.get("seq_duplicate", 0) == 0


def test_duplicate_seq_ties_break_by_row_order_and_count_as_duplicate_not_disorder():
    # Same timestamp AND same `seq` (upstream anomaly): the tie breaks by original row
    # index (file order still wins), and it is surfaced as `seq_duplicate` — NOT as
    # `seq_disorder`, which is reserved for strict regressions (a re-sorted stream).
    rows = [
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="O1", time_exchange_ns=T, seq=7),
        dict(update_type="SUB", is_buy=True, entry_px=100.0, entry_sx=3.0,
             order_id="O1", time_exchange_ns=T, seq=7),   # O1 5→2
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=1.0,
             order_id="O2", time_exchange_ns=T, seq=7),   # level 100 = 2+1
    ]
    frame, q = reco(rows)
    assert (frame.iloc[-1]["bid_0_price"], frame.iloc[-1]["bid_0_size"]) == (100.0, 3.0)
    assert q.get("missing_order", 0) == 0    # SUB followed its ADD despite the seq tie
    assert q.get("seq_duplicate", 0) == 2    # two rows repeated the previous seq
    assert q.get("seq_disorder", 0) == 0     # ties are NOT disorder
    # `last_seq` carries ACROSS chunks: splitting the duplicate-seq stream at any boundary
    # must not reset the counter state (a per-chunk reset would undercount seq_duplicate
    # while leaving the frame identical — only this q equality can catch it).
    df = coinapi_frame_from_rows(rows)
    for cut in (1, 2):
        split_frame, split_q = reco([df.iloc[:cut], df.iloc[cut:]])
        pd.testing.assert_frame_equal(split_frame, frame)
        assert split_q == q


def test_file_order_wins_over_decreasing_seq_and_disorder_is_counted():
    # A stream whose `seq` claims the MATCH came first (seq 3 < 5) but whose FILE order has
    # the ADD first. Policy: never re-sort by `seq` — apply file order (ADD then MATCH →
    # empty book, no missing_order) and surface the corruption as `seq_disorder`. A replay
    # that re-sorted by `seq` would instead skip the MATCH and leave bid 100@5.
    rows = [
        dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="O1", time_exchange_ns=T, seq=5),
        dict(update_type="MATCH", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="O1", time_exchange_ns=T, seq=3),
    ]
    frame, q = reco(rows)
    assert math.isnan(frame.iloc[-1]["bid_0_price"])
    assert q.get("missing_order", 0) == 0
    assert q["seq_disorder"] == 1
    assert q.get("seq_duplicate", 0) == 0


# ------------------------------------------------------------------------------ determinism
CLUSTER = [
    dict(update_type="ADD", is_buy=True, entry_px=100.0, entry_sx=4.0,
         order_id="b1", time_exchange_ns=T),
    dict(update_type="ADD", is_buy=False, entry_px=101.0, entry_sx=3.0,
         order_id="a1", time_exchange_ns=T),
    dict(update_type="SUB", is_buy=True, entry_px=100.0, entry_sx=1.0,
         order_id="b1", time_exchange_ns=T),
    dict(update_type="MATCH", is_buy=False, entry_px=101.0, entry_sx=3.0,
         order_id="a1", time_exchange_ns=T),
    dict(update_type="ADD", is_buy=False, entry_px=102.0, entry_sx=2.0,
         order_id="a2", time_exchange_ns=T),
]


def test_repeated_runs_are_byte_identical():
    runs = [reco(list(CLUSTER), k=2, sample_s=(1, 50, 100)) for _ in range(3)]
    for frame, q in runs[1:]:
        pd.testing.assert_frame_equal(frame, runs[0][0])
        assert q == runs[0][1]


def test_chunk_split_inside_a_same_ts_group_changes_nothing():
    # Stream the SAME rows as one chunk vs. split MID-GROUP (the boundary lands between two
    # same-timestamp events). Chunking is a transport detail: watermark + seq state carry
    # across chunks, so frames AND quality counters must be identical.
    df = coinapi_frame_from_rows(CLUSTER)
    whole_frame, whole_q = reco(df, k=2, sample_s=(1, 50, 100))
    for cut in (1, 2, 3, 4):
        split_frame, split_q = reco([df.iloc[:cut], df.iloc[cut:]], k=2, sample_s=(1, 50, 100))
        pd.testing.assert_frame_equal(split_frame, whole_frame)
        assert split_q == whole_q
