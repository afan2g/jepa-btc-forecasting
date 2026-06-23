"""Normalized events + the single total-order convention (spec §5.3)."""
from __future__ import annotations
from typing import NamedTuple, Union


class Delta(NamedTuple):
    ts_engine: int
    seq: int
    side: str      # "bid" | "ask"
    price: float
    size: float    # absolute size at this level; 0.0 => remove the level


class Trade(NamedTuple):
    ts_engine: int
    seq: int
    side: str      # "buy" | "sell"
    price: float
    amount: float


Event = Union[Delta, Trade]


def order_key(ev: Event) -> tuple[int, int, int]:
    """Total order: (ts_engine, kind, seq). Deltas (kind=0) precede trades (kind=1)
    at equal ts_engine. book_state_at(trade) applies deltas with key < trade's key."""
    kind = 0 if isinstance(ev, Delta) else 1
    return (ev.ts_engine, kind, ev.seq)
