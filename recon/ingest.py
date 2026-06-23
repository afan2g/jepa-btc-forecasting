"""Raw Crypto Lake DataFrame -> normalized event lists.

This adapter is the SINGLE schema-dependent seam. The post-`lakeapi` column names are
documented in docs/data.md §4.1 (and re-confirmed by scripts/verify_book_delta_v2.py).
After `lakeapi.load_data`:

  book_delta_v2: origin_time, received_time, sequence_number, side_is_bid (bool),
                 price, size              (origin_time/received_time are datetime64[ns])
  trades:        origin_time, received_time, price, quantity, side (buy/sell), trade_id

We resolve each field against a small set of known aliases (the real lakeapi name first,
then raw-`qnt`/synthetic alternates) and fail loudly — listing the columns we DID see — if
none match, so a vendor-schema drift surfaces as a clear error instead of a silent
miscolumn. Engine-time may be int64 ns (synthetic) or datetime64[ns] (real Lake);
`astype('int64')` normalizes both to ns (NaT -> negative, caught by the populated guard).
"""
from __future__ import annotations
import pandas as pd
from recon.events import Delta, Trade

# Side values meaning the bid/buy side. Covers bool `side_is_bid`, ints, and strings.
_BID_VALUES = ("bid", "b", "buy", True, 1, "1")


def _pick(df: pd.DataFrame, candidates: tuple[str, ...], *, field: str) -> str:
    """Return the first present column among `candidates`, else raise a clear error."""
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(
        f"no {field} column found; tried {candidates!r} but DataFrame has {list(df.columns)}"
    )


def _side_str(v) -> str:
    return "bid" if v in _BID_VALUES else "ask"


def _ns(s: pd.Series) -> pd.Series:
    """int64-ns view of an engine-time column (int64 ns or datetime64[ns])."""
    return s.astype("int64")


def _require_populated(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"engine-time column {col!r} not in {list(df.columns)}")
    if not (_ns(df[col]) > 0).all():
        raise ValueError(f"engine-time column {col!r} has non-populated (<=0) rows")


def deltas_from_df(df: pd.DataFrame, *, engine_time_col: str) -> list[Delta]:
    _require_populated(df, engine_time_col)
    seq_col = _pick(df, ("sequence_number", "seq"), field="delta sequence")
    side_col = _pick(df, ("side_is_bid", "side", "is_bid"), field="delta side")
    size_col = _pick(df, ("size", "amount"), field="delta size")
    ts = _ns(df[engine_time_col]).to_numpy()
    seq = df[seq_col].astype("int64").to_numpy()
    side = df[side_col].to_numpy()
    price = df["price"].astype("float64").to_numpy()
    size = df[size_col].astype("float64").to_numpy()
    return [Delta(int(ts[i]), int(seq[i]), _side_str(side[i]),
                  float(price[i]), float(size[i])) for i in range(len(df))]


def trades_from_df(df: pd.DataFrame, *, engine_time_col: str) -> list[Trade]:
    _require_populated(df, engine_time_col)
    id_col = _pick(df, ("trade_id", "id"), field="trade id")
    amt_col = _pick(df, ("quantity", "amount"), field="trade size")
    side_col = _pick(df, ("side",), field="trade side")
    ts = _ns(df[engine_time_col]).to_numpy()
    seq = df[id_col].astype("int64").to_numpy()
    side = df[side_col].astype(str).to_numpy()
    price = df["price"].astype("float64").to_numpy()
    amount = df[amt_col].astype("float64").to_numpy()
    return [Trade(int(ts[i]), int(seq[i]), str(side[i]),
                  float(price[i]), float(amount[i])) for i in range(len(df))]
