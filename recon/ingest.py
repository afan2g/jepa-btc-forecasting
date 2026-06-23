"""Raw Crypto Lake DataFrame -> normalized event lists.

The exact source column names for book_delta_v2 are confirmed by Task 1
(scripts/verify_book_delta_v2.py). This adapter is the SINGLE schema-dependent
seam: update the SIDE_COL / SIZE_COL fallbacks below if Lake differs.
"""
from __future__ import annotations
import pandas as pd
from recon.events import Delta, Trade


def _side_str(v, *, bid_set=("bid", "b", "buy", True, 1, "1")) -> str:
    return "bid" if v in bid_set else "ask"


def _require_populated(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise ValueError(f"engine-time column {col!r} not in {list(df.columns)}")
    if not (df[col].astype("int64") > 0).all():
        raise ValueError(f"engine-time column {col!r} has non-populated (<=0) rows")


def deltas_from_df(df: pd.DataFrame, *, engine_time_col: str) -> list[Delta]:
    _require_populated(df, engine_time_col)
    side_col = "side" if "side" in df.columns else "is_bid"
    size_col = "size" if "size" in df.columns else "amount"
    ts = df[engine_time_col].astype("int64").to_numpy()
    seq = df["sequence_number"].astype("int64").to_numpy()
    side = df[side_col].to_numpy()
    price = df["price"].astype("float64").to_numpy()
    size = df[size_col].astype("float64").to_numpy()
    return [Delta(int(ts[i]), int(seq[i]), _side_str(side[i]),
                  float(price[i]), float(size[i])) for i in range(len(df))]


def trades_from_df(df: pd.DataFrame, *, engine_time_col: str) -> list[Trade]:
    _require_populated(df, engine_time_col)
    ts = df[engine_time_col].astype("int64").to_numpy()
    seq = df["id"].astype("int64").to_numpy()
    side = df["side"].astype(str).to_numpy()
    price = df["price"].astype("float64").to_numpy()
    amount = df["amount"].astype("float64").to_numpy()
    return [Trade(int(ts[i]), int(seq[i]), str(side[i]),
                  float(price[i]), float(amount[i])) for i in range(len(df))]
