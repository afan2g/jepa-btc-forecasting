import pandas as pd
from recon.events import Delta, Trade, order_key
from recon.merge import merge_sorted
from recon.orderbook import OrderBook
from recon.reconstruct import reconstruct_book_at_trades
from recon.synthetic import simple_world
from recon.ingest import deltas_from_df, trades_from_df


def _events():
    draw, traw = simple_world()
    # Rename normalized synthetic columns to RAW Lake names before ingest:
    # deltas use sequence_number, trades use id (see recon/ingest.py).
    d = deltas_from_df(pd.DataFrame(draw).rename(columns={"ts_engine": "origin_time", "seq": "sequence_number"}),
                       engine_time_col="origin_time")
    t = trades_from_df(pd.DataFrame(traw).rename(columns={"ts_engine": "timestamp", "seq": "id"}),
                       engine_time_col="timestamp")
    return d, t


def test_reconstruct_matches_hand_computed_snapshots():
    d, t = _events()
    out = reconstruct_book_at_trades(d, t, k=1)
    # trade @20 sees mid 100.5 ; @40 sees bid 99/ask 101 -> mid 100.0 ;
    # @60 sees bid 99/ask 102 -> mid 100.5
    assert list(out["trade_ts"]) == [20, 40, 60]
    assert list(out["mid"]) == [100.5, 100.0, 100.5]
    assert list(out["bid_0_price"]) == [100.0, 99.0, 99.0]
    assert list(out["ask_0_price"]) == [101.0, 101.0, 102.0]


def test_no_lookahead_dropping_future_deltas_is_invariant():
    """For each trade, deleting every delta with order key >= the trade's key must
    not change that trade's reconstructed snapshot (spec §5.3 lookahead guard)."""
    d, t = _events()
    full = reconstruct_book_at_trades(d, t, k=1)
    for i, tr in enumerate(t):
        kept = [x for x in d if order_key(x) < order_key(tr)]
        one = reconstruct_book_at_trades(kept, [tr], k=1)
        assert one["mid"].iloc[0] == full["mid"].iloc[i]
        assert one["bid_0_price"].iloc[0] == full["bid_0_price"].iloc[i]
        assert one["ask_0_price"].iloc[0] == full["ask_0_price"].iloc[i]


def test_trade_does_not_see_same_ts_later_kind_or_its_own_impact():
    # A delta with the SAME ts but kind=trade ordering must be excluded; a same-ts
    # delta (kind=0) must be included.
    d = [Delta(10, 1, "bid", 100.0, 2.0), Delta(10, 2, "ask", 101.0, 3.0),
         Delta(20, 3, "ask", 101.0, 0.0)]  # removes ask AT ts=20, same as the trade
    tr = [Trade(20, 1001, "buy", 101.0, 0.5)]
    out = reconstruct_book_at_trades(d, tr, k=1)
    # delta(20,kind0,seq3) < trade(20,kind1,seq1001) => the ask removal IS applied,
    # so the trade sees NO ask at 101 (best_ask absent -> NaN-padded).
    assert pd.isna(out["ask_0_price"].iloc[0])
