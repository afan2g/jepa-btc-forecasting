"""Received-time-bearing clock event contract (issue #61 / plan §C.2 prerequisite).

The normalized `recon.events.Trade` carries a single `ts_engine`, so the
`received_time <= t_event` observability gate is unimplementable on it; the clock input
record must carry BOTH origin (ordering axis) and received (gating axis). The adapter
consumes the one normalized trade contract both vendors already emit — Lake trades
(`origin_time, received_time, price, quantity, side, trade_id`) and PR #59's CoinAPI
normalized parquet (same columns plus file-order `seq`, string-guid `trade_id`) — and
FAILS CLOSED on anything that could corrupt ordering or gating, most importantly a
CoinAPI time-of-day offset that was never converted to an absolute UTC timestamp."""
import pathlib
import sys

import pandas as pd
import pytest

from bars.events import MIN_ABSOLUTE_NS, ClockTrade, clock_order_key, clock_trades_from_df

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

DAY = "2025-01-07"
DAY_OPEN_NS = int(pd.Timestamp(DAY, tz="UTC").value)


def _lake_df():
    # Lake-shaped normalized trades: datetime64[ns] times, int trade_id, no seq column.
    return pd.DataFrame({
        "origin_time": pd.to_datetime([DAY_OPEN_NS + 1_000, DAY_OPEN_NS + 2_000], unit="ns"),
        "received_time": pd.to_datetime([DAY_OPEN_NS + 1_500, DAY_OPEN_NS + 2_700], unit="ns"),
        "price": [100.0, 101.0],
        "quantity": [0.5, 0.25],
        "side": ["buy", "sell"],
        "trade_id": [7, 3],
    })


def _coinapi_df():
    # PR #59 normalized CoinAPI trades: absolute times, string guid trade_id, seq column.
    return pd.DataFrame({
        "seq": [0, 1],
        "origin_time": pd.to_datetime([DAY_OPEN_NS + 10, DAY_OPEN_NS + 20], unit="ns"),
        "received_time": pd.to_datetime([DAY_OPEN_NS + 15, DAY_OPEN_NS + 12], unit="ns"),
        "price": [99.5, 99.6],
        "quantity": [1.0, 2.0],
        "side": ["sell", "buy"],
        "trade_id": ["guid-a", "guid-b"],
    })


def test_record_carries_both_timestamps_and_orders_on_origin_then_seq():
    a = ClockTrade(origin_time=10, received_time=99, seq=2, side="buy", price=1.0, amount=1.0)
    b = ClockTrade(origin_time=10, received_time=1, seq=5, side="sell", price=1.0, amount=1.0)
    c = ClockTrade(origin_time=11, received_time=0, seq=0, side="buy", price=1.0, amount=1.0)
    # ordering is origin-then-seq: received_time NEVER participates in ordering (it is
    # the gating axis), matching recon.events.order_key's (ts, kind, seq) trade ordering.
    assert sorted([c, b, a], key=clock_order_key) == [a, b, c]


def test_adapter_reads_lake_shape_and_uses_trade_id_as_seq():
    out = clock_trades_from_df(_lake_df())
    assert out == [
        ClockTrade(DAY_OPEN_NS + 1_000, DAY_OPEN_NS + 1_500, 7, "buy", 100.0, 0.5),
        ClockTrade(DAY_OPEN_NS + 2_000, DAY_OPEN_NS + 2_700, 3, "sell", 101.0, 0.25),
    ]


def test_adapter_reads_coinapi_shape_and_prefers_seq_over_guid_trade_id():
    out = clock_trades_from_df(_coinapi_df())
    assert [t.seq for t in out] == [0, 1]
    assert out[0].origin_time == DAY_OPEN_NS + 10
    assert out[0].received_time == DAY_OPEN_NS + 15


def test_adapter_accepts_int64_ns_columns():
    df = _lake_df()
    df["origin_time"] = df["origin_time"].astype("int64")
    df["received_time"] = df["received_time"].astype("int64")
    out = clock_trades_from_df(df)
    assert out[0].origin_time == DAY_OPEN_NS + 1_000


def test_adapter_preserves_input_row_order_without_sorting():
    # Ordering is the clock's job (defensive sort); the adapter is a faithful reader.
    df = _lake_df().iloc[::-1].reset_index(drop=True)
    out = clock_trades_from_df(df)
    assert [t.origin_time for t in out] == [DAY_OPEN_NS + 2_000, DAY_OPEN_NS + 1_000]


def test_adapter_rejects_unconverted_time_of_day_timestamps():
    # A raw CoinAPI time-of-day offset (ns since midnight, < 1 day) would pass every
    # received_time <= t_event gate; the adapter must fail closed before gating.
    df = _lake_df()
    df["origin_time"] = [13 * 3600 * 10**9, 14 * 3600 * 10**9]  # 13:00 / 14:00 offsets
    with pytest.raises(ValueError, match="absolute"):
        clock_trades_from_df(df)
    assert MIN_ABSOLUTE_NS > 86_400 * 10**9  # any time-of-day offset is below the floor


def test_adapter_rejects_unknown_side():
    df = _lake_df()
    df.loc[0, "side"] = "BUY_ESTIMATED"  # vendor raw form must be normalized upstream
    with pytest.raises(ValueError, match="side"):
        clock_trades_from_df(df)


def test_adapter_rejects_nonpositive_price_or_amount():
    df = _lake_df()
    df.loc[0, "price"] = 0.0
    with pytest.raises(ValueError, match="price"):
        clock_trades_from_df(df)
    df = _lake_df()
    df.loc[1, "quantity"] = -0.5
    with pytest.raises(ValueError, match="amount|quantity"):
        clock_trades_from_df(df)


def test_adapter_rejects_guid_trade_id_without_seq_column():
    df = _lake_df()
    df["trade_id"] = ["guid-a", "guid-b"]  # no deterministic int tie-break available
    with pytest.raises(ValueError, match="seq"):
        clock_trades_from_df(df)


def test_adapter_rejects_missing_required_columns():
    df = _lake_df().drop(columns=["received_time"])
    with pytest.raises(ValueError, match="received_time"):
        clock_trades_from_df(df)


def test_coinapi_vendor_offsets_reach_the_clock_as_absolute_utc():
    # End-to-end through the PR #59 normalizer: vendor time-of-day offsets ->
    # trades_to_table -> adapter must yield day_open + offset ABSOLUTE ns, so the
    # received_time <= t_event gate compares real instants (plan §C.1 Codex P1).
    pytest.importorskip("pyarrow")
    import coinapi_backfill_fixtures as fx
    dl = fx.load_by_path("download_coinapi", "ingest/download_coinapi.py")
    vendor = pd.DataFrame({
        "time_exchange": ["13:30:03.5851480", "13:30:04.0000000"],
        "time_coinapi": ["13:30:03.6851480", "13:30:04.2000000"],
        "guid": ["g-1", "g-2"],
        "price": [100.0, 100.5],
        "base_amount": [0.1, 0.2],
        "taker_side": ["BUY", "SELL_ESTIMATED"],
    })
    table = dl.trades_to_table(vendor, seq_start=0, day=DAY)
    out = clock_trades_from_df(table.to_pandas())
    off = (13 * 3600 + 30 * 60 + 3) * 10**9 + 585_148_000
    assert out[0].origin_time == DAY_OPEN_NS + off
    assert out[0].received_time == DAY_OPEN_NS + off + 100_000_000
    assert out[0].side == "buy" and out[1].side == "sell"
    assert [t.seq for t in out] == [0, 1]
