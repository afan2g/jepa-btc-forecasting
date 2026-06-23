"""Deterministic synthetic delta/trade streams for tests (no RNG state leakage).

A 'world' is a list of (kind, ts_engine, seq, side, price, size_or_amount) tuples
in true causal order. We build small, hand-verifiable books so tests can assert
exact reconstructed snapshots.
"""
from __future__ import annotations


def simple_world():
    """A tiny, fully hand-checkable stream.

    Timeline (ts in ns), deltas (kind d) seed/modify a 2-level book, trades (kind t)
    occur between updates. Returns (deltas, trades) as raw dict rows matching the
    NORMALIZED event field names used by recon.events.

    Book evolution:
      ts=10 d bid 100@2 ; ts=10 d ask 101@3       -> mid 100.5
      ts=20 t buy 101@0.5  (sees mid 100.5)
      ts=30 d bid 100@0    (remove) ; d bid  99@1  -> best bid 99
      ts=40 t sell 99@0.7  (sees bid 99 / ask 101)
      ts=50 d ask 101@0 (remove) ; d ask 102@4     -> best ask 102
      ts=60 t buy 102@0.2  (sees bid 99 / ask 102)
    """
    deltas = [
        dict(ts_engine=10, seq=1, side="bid", price=100.0, size=2.0),
        dict(ts_engine=10, seq=2, side="ask", price=101.0, size=3.0),
        dict(ts_engine=30, seq=3, side="bid", price=100.0, size=0.0),
        dict(ts_engine=30, seq=4, side="bid", price=99.0, size=1.0),
        dict(ts_engine=50, seq=5, side="ask", price=101.0, size=0.0),
        dict(ts_engine=50, seq=6, side="ask", price=102.0, size=4.0),
    ]
    trades = [
        dict(ts_engine=20, seq=1001, side="buy", price=101.0, amount=0.5),
        dict(ts_engine=40, seq=1002, side="sell", price=99.0, amount=0.7),
        dict(ts_engine=60, seq=1003, side="buy", price=102.0, amount=0.2),
    ]
    return deltas, trades


def same_ts_world():
    """A stream where EVERY trade shares its ts_engine with deltas, so the §5.3
    same-ts convention (deltas (kind=0) included before trades (kind=1) at equal
    ts_engine) is exercised by the generic no-lookahead invariant — simple_world has
    no delta/trade ts overlap, so it cannot catch a same-ts-ordering regression.

    Book evolution (each trade sees the SAME-ts deltas already applied):
      ts=10 d bid 100@2 ; d ask 101@3 ; t buy 101@0.5     -> sees bid100/ask101, mid100.5
      ts=20 d bid 100@0(remove) ; d bid 99@1 ; t sell 99@0.3 -> sees bid99/ask101, mid100.0
      ts=30 d ask 101@5(modify) ; t buy 101@0.2           -> sees bid99/ask101(sz5), mid100.0
      ts=40 d ask 101@0(remove) ; d ask 102@4 ; t buy 102@0.1 -> sees bid99/ask102, mid100.5
    """
    deltas = [
        dict(ts_engine=10, seq=1, side="bid", price=100.0, size=2.0),
        dict(ts_engine=10, seq=2, side="ask", price=101.0, size=3.0),
        dict(ts_engine=20, seq=3, side="bid", price=100.0, size=0.0),
        dict(ts_engine=20, seq=4, side="bid", price=99.0, size=1.0),
        dict(ts_engine=30, seq=5, side="ask", price=101.0, size=5.0),
        dict(ts_engine=40, seq=6, side="ask", price=101.0, size=0.0),
        dict(ts_engine=40, seq=7, side="ask", price=102.0, size=4.0),
    ]
    trades = [
        dict(ts_engine=10, seq=1001, side="buy", price=101.0, amount=0.5),
        dict(ts_engine=20, seq=1002, side="sell", price=99.0, amount=0.3),
        dict(ts_engine=30, seq=1003, side="buy", price=101.0, amount=0.2),
        dict(ts_engine=40, seq=1004, side="buy", price=102.0, amount=0.1),
    ]
    return deltas, trades
