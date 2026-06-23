import math
from recon.events import Delta
from recon.orderbook import OrderBook


def apply_all(ob, deltas):
    for d in deltas:
        ob.apply(d)


def test_apply_add_and_best_prices():
    ob = OrderBook()
    apply_all(ob, [Delta(10, 1, "bid", 100.0, 2.0), Delta(10, 2, "ask", 101.0, 3.0)])
    assert ob.best_bid() == 100.0
    assert ob.best_ask() == 101.0
    assert ob.mid() == 100.5


def test_size_zero_removes_level():
    ob = OrderBook()
    apply_all(ob, [Delta(10, 1, "bid", 100.0, 2.0), Delta(20, 2, "bid", 100.0, 0.0),
                   Delta(20, 3, "bid", 99.0, 1.0)])
    assert ob.best_bid() == 99.0


def test_microprice_weights_by_opposite_size():
    ob = OrderBook()
    apply_all(ob, [Delta(10, 1, "bid", 100.0, 1.0), Delta(10, 2, "ask", 101.0, 3.0)])
    # microprice = (ask_sz*bid_px + bid_sz*ask_px)/(bid_sz+ask_sz)
    assert ob.microprice() == (3.0 * 100.0 + 1.0 * 101.0) / 4.0


def test_snapshot_top_k_sorted_and_padded():
    ob = OrderBook()
    apply_all(ob, [Delta(10, 1, "bid", 100.0, 2.0), Delta(10, 2, "bid", 99.0, 1.0),
                   Delta(10, 3, "ask", 101.0, 3.0)])
    snap = ob.snapshot(k=2)
    assert snap["bid_0_price"] == 100.0 and snap["bid_1_price"] == 99.0
    assert snap["ask_0_price"] == 101.0
    assert math.isnan(snap["ask_1_price"])  # padded with NaN when fewer than k levels


def test_gap_detection_on_nonmonotonic_sequence():
    ob = OrderBook()
    ob.apply(Delta(10, 5, "bid", 100.0, 2.0))
    assert ob.apply(Delta(10, 5, "bid", 100.0, 1.0)) is False  # seq not increasing => gap flag


def test_snapshot_topk_matches_full_sort_on_a_deep_book():
    # Guards the heapq.nlargest/nsmallest top-K against a full-sort reference on a deep,
    # non-monotonically-inserted book — the bounded-selection optimization must stay
    # byte-identical to sorted(...)[:k] (incl. best bid/ask, mid, microprice).
    ob = OrderBook()
    seq = 0
    for p in (137, 5, 991, 42, 600, 88, 250, 7, 333, 410, 159, 26, 845, 73, 512):
        seq += 1
        ob.apply(Delta(1, seq, "bid", float(p), float(p % 9 + 1)))      # bids below 1000
        seq += 1
        ob.apply(Delta(1, seq, "ask", float(p + 1000), float(p % 7 + 1)))  # asks above 1000
    k = 5
    snap = ob.snapshot(k)
    ref_bids = sorted(ob.bids, reverse=True)[:k]
    ref_asks = sorted(ob.asks)[:k]
    for i in range(k):
        assert snap[f"bid_{i}_price"] == ref_bids[i]
        assert snap[f"bid_{i}_size"] == ob.bids[ref_bids[i]]
        assert snap[f"ask_{i}_price"] == ref_asks[i]
        assert snap[f"ask_{i}_size"] == ob.asks[ref_asks[i]]
    assert snap["mid"] == (ref_bids[0] + ref_asks[0]) / 2.0
    bs, as_ = ob.bids[ref_bids[0]], ob.asks[ref_asks[0]]
    assert snap["microprice"] == (as_ * ref_bids[0] + bs * ref_asks[0]) / (bs + as_)
