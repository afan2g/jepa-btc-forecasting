"""Offline tests: explicit, fail-loud passthrough normalizers (plan Task 3 / Requirement 6).

Pure pandas, no vendor I/O, no pyarrow. Stage-2 normalizes the raw-store trades/funding/
open_interest/liquidations partitions into fixed internal contracts (AGENTS.md "Downstream code
should consume normalized internal contracts"): canonical column names/order, int64-ns engine-time
columns, validated side values. Vendor schema drift (renamed/missing column, unknown side value)
must raise a clear error naming the columns/value seen — never silently mis-column
(the `recon.ingest._pick`/`_side_str` fail-loud pattern).
"""
import numpy as np
import pandas as pd
import pytest

from ingest import lake_binance as lb

NS = 1_000_000_000


def _t(*secs):
    """datetime64[ns] engine-time column from second offsets (the real-Lake dtype)."""
    return pd.to_datetime([s * NS for s in secs])


# --------------------------------------------------------------------------- registry coverage
def test_normalizers_cover_every_passthrough_feed():
    # book_delta_v2 reconstructs (recon owns its aliases); every OTHER scoped feed must normalize.
    assert set(lb.NORMALIZERS) == set(lb.FEEDS) - {"book_delta_v2"}


# --------------------------------------------------------------------------- time canonicalization
def test_canonicalize_renames_raw_vendor_time_columns():
    df = pd.DataFrame({"timestamp": _t(1, 2), "receipt_timestamp": _t(1, 2), "price": [1.0, 2.0]})
    out = lb.canonicalize_time_columns(df)
    assert "origin_time" in out.columns and "received_time" in out.columns
    assert "timestamp" not in out.columns and "receipt_timestamp" not in out.columns


def test_canonicalize_never_clobbers_existing_canonical_columns():
    df = pd.DataFrame({"origin_time": _t(1), "timestamp": _t(9), "price": [1.0]})
    out = lb.canonicalize_time_columns(df)
    assert list(out["origin_time"]) == list(_t(1))   # canonical kept
    assert "timestamp" in out.columns                 # alias untouched, not silently merged


# --------------------------------------------------------------------------- trades
def _raw_trades(**over):
    base = {"origin_time": _t(1, 2, 3), "received_time": _t(1, 2, 3),
            "price": [100.0, 101.0, 102.0], "quantity": [0.5, 1.0, 1.5],
            "side": ["buy", "sell", "buy"], "trade_id": [7, 8, 9]}
    base.update(over)
    return pd.DataFrame(base)


def test_normalize_trades_canonical_schema_and_order():
    out = lb.normalize_trades(_raw_trades())
    assert list(out.columns) == ["origin_time", "received_time", "price", "quantity",
                                 "side", "trade_id"]
    assert out["origin_time"].dtype == "int64" and out["received_time"].dtype == "int64"
    assert out["price"].dtype == "float64" and out["quantity"].dtype == "float64"
    assert out["trade_id"].dtype == "int64"
    assert list(out["side"]) == ["buy", "sell", "buy"]
    assert list(out["origin_time"]) == [NS, 2 * NS, 3 * NS]   # int64 ns, values preserved
    assert len(out) == 3                                       # never drops/reorders rows


def test_normalize_trades_resolves_documented_aliases():
    df = _raw_trades().rename(columns={"origin_time": "timestamp",
                                       "received_time": "receipt_timestamp",
                                       "quantity": "amount", "trade_id": "id"})
    out = lb.normalize_trades(df)
    assert list(out.columns) == ["origin_time", "received_time", "price", "quantity",
                                 "side", "trade_id"]
    assert list(out["quantity"]) == [0.5, 1.0, 1.5]
    assert list(out["trade_id"]) == [7, 8, 9]


def test_normalize_trades_missing_column_raises_listing_seen_columns():
    df = _raw_trades().drop(columns=["price"])
    with pytest.raises(ValueError, match="price"):
        lb.normalize_trades(df)
    df2 = _raw_trades().drop(columns=["received_time"])
    with pytest.raises(ValueError, match="received_time"):
        lb.normalize_trades(df2)


def test_normalize_trades_unknown_side_value_raises():
    with pytest.raises(ValueError, match="short"):
        lb.normalize_trades(_raw_trades(side=["buy", "short", "sell"]))


def test_normalize_trades_side_case_insensitive():
    out = lb.normalize_trades(_raw_trades(side=["BUY", "Sell", "b"]))
    assert list(out["side"]) == ["buy", "sell", "buy"]


def test_normalize_trades_non_numeric_price_raises():
    with pytest.raises((ValueError, TypeError)):
        lb.normalize_trades(_raw_trades(price=["a", "b", "c"]))


# --------------------------------------------------------------------------- funding
def test_normalize_funding_canonical_schema():
    df = pd.DataFrame({"origin_time": _t(1), "received_time": _t(1),
                       "funding_rate": [0.0001]})
    out = lb.normalize_funding(df)
    assert list(out.columns) == ["origin_time", "received_time", "funding_rate"]
    assert out["funding_rate"].dtype == "float64"


def test_normalize_funding_keeps_optional_next_funding_time():
    df = pd.DataFrame({"origin_time": _t(1), "received_time": _t(1),
                       "funding_rate": [0.0001], "next_funding_time": _t(9)})
    out = lb.normalize_funding(df)
    assert list(out.columns) == ["origin_time", "received_time", "funding_rate",
                                 "next_funding_time"]
    assert out["next_funding_time"].dtype == "int64"


def test_normalize_funding_resolves_rate_alias_and_fails_loud():
    df = pd.DataFrame({"origin_time": _t(1), "received_time": _t(1), "rate": [0.0001]})
    assert list(lb.normalize_funding(df)["funding_rate"]) == [0.0001]
    bad = pd.DataFrame({"origin_time": _t(1), "received_time": _t(1), "px": [1.0]})
    with pytest.raises(ValueError, match="funding_rate"):
        lb.normalize_funding(bad)


# --------------------------------------------------------------------------- open interest
def test_normalize_open_interest_canonical_schema_and_drift():
    df = pd.DataFrame({"origin_time": _t(1, 2), "received_time": _t(1, 2),
                       "open_interest": [1000.0, 1001.0]})
    out = lb.normalize_open_interest(df)
    assert list(out.columns) == ["origin_time", "received_time", "open_interest"]
    assert out["open_interest"].dtype == "float64"
    with pytest.raises(ValueError, match="open_interest"):
        lb.normalize_open_interest(df.rename(columns={"open_interest": "value"}))


# --------------------------------------------------------------------------- liquidations
def test_normalize_liquidations_canonical_schema():
    df = pd.DataFrame({"origin_time": _t(1, 2), "received_time": _t(1, 2),
                       "price": [100.0, 99.0], "quantity": [0.3, 0.4],
                       "side": ["sell", "buy"]})
    out = lb.normalize_liquidations(df)
    assert list(out.columns) == ["origin_time", "received_time", "price", "quantity", "side"]
    assert list(out["side"]) == ["sell", "buy"]


def test_normalize_liquidations_alias_and_unknown_side():
    df = pd.DataFrame({"timestamp": _t(1), "receipt_timestamp": _t(1),
                       "price": [100.0], "amount": [0.3], "side": ["sell"]})
    out = lb.normalize_liquidations(df)
    assert list(out.columns) == ["origin_time", "received_time", "price", "quantity", "side"]
    bad = pd.DataFrame({"origin_time": _t(1), "received_time": _t(1),
                        "price": [100.0], "quantity": [0.3], "side": [1.5]})
    with pytest.raises(ValueError):
        lb.normalize_liquidations(bad)


# --------------------------------------------------------------------------- empty frames
def test_normalizers_accept_zero_row_frames():
    df = _raw_trades().iloc[0:0]
    out = lb.normalize_trades(df)
    assert len(out) == 0
    assert list(out.columns) == ["origin_time", "received_time", "price", "quantity",
                                 "side", "trade_id"]
