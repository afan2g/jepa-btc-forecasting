"""Received-time-bearing clock event record + the one normalized-trades adapter.

`recon.events.Trade` carries a single `ts_engine`, so the per-event
`received_time <= t_event` observability gate (plan §C.2, Codex P1) is
unimplementable on it. The clock consumes `ClockTrade`, which carries BOTH axes:

  origin_time    ns since epoch UTC — exchange time, the ORDERING axis (§A)
  received_time  ns since epoch UTC — capture time, the GATING axis (§C.2)
  seq            int tie-break within equal origin_time (mirrors order_key's seq)

`clock_trades_from_df` reads the single normalized trade contract both vendors
already emit — Lake trades (docs/data.md §4.1: origin_time, received_time, price,
quantity, side, trade_id) and the PR #59 CoinAPI normalized parquet
(ingest/download_coinapi.py TRADES_SCHEMA: same columns plus file-order `seq` and a
string-guid `trade_id`). Vendor schema knowledge stays at the ingestion boundary
(AGENTS.md); this adapter only checks the contract and fails closed on anything that
could corrupt ordering or gating. recon.events / recon.merge are untouched — the
global event total order (`order_key`) is not redefined here.
"""
from __future__ import annotations

import math
from typing import NamedTuple

import pandas as pd

# Fail-closed floor for "this is an absolute UTC timestamp": CoinAPI vendor times are
# time-of-day offsets (< 86_400e9 ns) until the ingestion normalizer adds the day open
# (plan §C.1, Codex P1); an unconverted offset would pass EVERY received_time <= t_event
# gate. 1e18 ns ~ 2001-09-09 — far below any market data we ingest, far above any
# offset or a seconds/ms/us-unit timestamp, so unit mistakes fail loudly too.
MIN_ABSOLUTE_NS = 1_000_000_000_000_000_000

_SIDES = ("buy", "sell")


class ClockTrade(NamedTuple):
    origin_time: int    # ns since epoch UTC (exchange time; ordering axis)
    received_time: int  # ns since epoch UTC (capture time; observability axis)
    seq: int            # deterministic tie-break within equal origin_time
    side: str           # "buy" | "sell"
    price: float
    amount: float


def clock_order_key(t: ClockTrade) -> tuple[int, int]:
    """Deterministic accumulation order: (origin_time, seq). received_time never
    participates in ordering — it gates observability only (§C.2). Matches the trade
    ordering inside recon.events.order_key ((ts, kind, seq) at fixed kind)."""
    return (t.origin_time, t.seq)


def _ns_array(df: pd.DataFrame, col: str):
    """int64-ns view of a time column (datetime64[ns] or int64), validated absolute."""
    vals = df[col].astype("int64").to_numpy()
    if len(vals) and int(vals.min()) < MIN_ABSOLUTE_NS:
        raise ValueError(
            f"{col} has values below {MIN_ABSOLUTE_NS} ns — not absolute UTC epoch "
            "timestamps (a CoinAPI time-of-day offset must be converted to "
            "day_open + offset BEFORE it reaches the clock; plan §C.1)"
        )
    return vals


def clock_trades_from_df(df: pd.DataFrame) -> list[ClockTrade]:
    """Normalized trades DataFrame -> list[ClockTrade], faithful to input row order
    (defensive sorting is the clock's job). Fails closed on schema drift."""
    for col in ("origin_time", "received_time", "price", "quantity", "side"):
        if col not in df.columns:
            raise ValueError(f"no {col} column in normalized trades; got {list(df.columns)}")
    origin = _ns_array(df, "origin_time")
    received = _ns_array(df, "received_time")
    if "seq" in df.columns:
        seq = df["seq"].astype("int64").to_numpy()
    elif "trade_id" in df.columns:
        try:
            seq = df["trade_id"].astype("int64").to_numpy()
        except (ValueError, TypeError) as e:
            raise ValueError(
                "trade_id is not integer-castable and no seq column is present — "
                "no deterministic within-origin tie-break available (CoinAPI "
                "normalized trades carry a file-order seq; Lake trades an int trade_id)"
            ) from e
    else:
        raise ValueError(f"no seq or trade_id column; got {list(df.columns)}")
    side = df["side"].astype(str).to_numpy()
    price = df["price"].astype("float64").to_numpy()
    amount = df["quantity"].astype("float64").to_numpy()
    out: list[ClockTrade] = []
    for i in range(len(df)):
        s = str(side[i])
        if s not in _SIDES:
            raise ValueError(f"unrecognized trade side {s!r} at row {i}; expected buy/sell "
                             "(vendor forms must be normalized at ingestion)")
        p, a = float(price[i]), float(amount[i])
        if not (math.isfinite(p) and p > 0.0):
            raise ValueError(f"non-positive or non-finite price {p!r} at row {i}")
        if not (math.isfinite(a) and a > 0.0):
            raise ValueError(f"non-positive or non-finite amount/quantity {a!r} at row {i}")
        out.append(ClockTrade(int(origin[i]), int(received[i]), int(seq[i]), s, p, a))
    return out
