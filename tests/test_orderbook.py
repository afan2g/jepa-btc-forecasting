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
