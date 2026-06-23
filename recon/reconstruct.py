"""Offline event-time reconstruction: book-state-at-trade with strict-< apply-before-read."""
from __future__ import annotations
import pandas as pd
from recon.events import Delta, Trade
from recon.merge import merge_sorted
from recon.orderbook import OrderBook


def reconstruct_book_at_trades(deltas, trades, *, k: int, seed: OrderBook | None = None) -> pd.DataFrame:
    """Replay the merged stream; at each trade, emit the book snapshot AS OF that trade
    (all deltas with order key < the trade's key already applied; the trade's own impact
    and later events excluded). Returns one row per trade."""
    ob = seed if seed is not None else OrderBook()
    rows: list[dict] = []
    for ev in merge_sorted(deltas, trades):
        if isinstance(ev, Delta):
            ob.apply(ev)
        else:  # Trade -> snapshot the book it saw
            snap = ob.snapshot(k)
            snap["trade_ts"] = ev.ts_engine
            snap["trade_seq"] = ev.seq
            snap["trade_side"] = ev.side
            snap["trade_price"] = ev.price
            snap["trade_amount"] = ev.amount
            rows.append(snap)
    return pd.DataFrame(rows)
