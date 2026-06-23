from recon.events import Delta, Trade, order_key

def test_delta_sorts_before_trade_at_equal_ts():
    d = Delta(ts_engine=10, seq=2, side="bid", price=100.0, size=2.0)
    t = Trade(ts_engine=10, seq=1001, side="buy", price=101.0, amount=0.5)
    assert order_key(d) < order_key(t)

def test_order_within_kind_is_by_seq():
    d1 = Delta(ts_engine=10, seq=1, side="bid", price=100.0, size=2.0)
    d2 = Delta(ts_engine=10, seq=2, side="ask", price=101.0, size=3.0)
    assert order_key(d1) < order_key(d2)
