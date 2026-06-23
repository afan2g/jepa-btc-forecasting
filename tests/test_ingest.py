import pandas as pd
import pytest
from recon.events import Delta, Trade
from recon.ingest import deltas_from_df, trades_from_df
from recon.synthetic import simple_world
from tests.conftest import FIXTURES


def test_deltas_from_df_normalizes_synthetic():
    draw, _ = simple_world()
    # Simulate RAW Lake book_delta_v2 columns: normalized seq -> sequence_number,
    # ts_engine -> origin_time (mirrors the trades test renaming seq -> id).
    df = pd.DataFrame(draw).rename(columns={"ts_engine": "origin_time", "seq": "sequence_number"})
    out = deltas_from_df(df, engine_time_col="origin_time")
    assert out[0] == Delta(ts_engine=10, seq=1, side="bid", price=100.0, size=2.0)
    assert all(isinstance(d, Delta) for d in out)
    assert [d.ts_engine for d in out] == [10, 10, 30, 30, 50, 50]


def test_trades_from_df_normalizes_synthetic():
    _, traw = simple_world()
    df = pd.DataFrame(traw).rename(columns={"ts_engine": "timestamp", "seq": "id"})
    out = trades_from_df(df, engine_time_col="timestamp")
    assert out[0] == Trade(ts_engine=20, seq=1001, side="buy", price=101.0, amount=0.5)


def test_ingest_rejects_unpopulated_engine_time():
    df = pd.DataFrame([dict(origin_time=0, seq=1, side="bid", price=1.0, size=1.0)])
    with pytest.raises(ValueError, match="engine-time"):
        deltas_from_df(df, engine_time_col="origin_time")


def test_deltas_from_df_real_lake_schema():
    # Mimic the REAL post-lakeapi book_delta_v2 view (docs/data.md §4.1):
    # origin_time datetime64[ns], sequence_number, side_is_bid (bool), price, size.
    base = 1668470400000000000  # 2022-11-15T00:00:00Z in ns since epoch
    df = pd.DataFrame({
        "origin_time": pd.to_datetime([base, base + 1]),  # ns ints -> datetime64[ns]
        "received_time": pd.to_datetime([base, base + 2]),
        "sequence_number": pd.array([10, 11], dtype="int64"),
        "side_is_bid": [True, False],
        "price": [100.0, 101.0],
        "size": [2.0, 3.0],
    })
    assert df["origin_time"].dtype == "datetime64[ns]"  # exercise the real dtype path
    out = deltas_from_df(df, engine_time_col="origin_time")
    assert out[0] == Delta(ts_engine=1668470400000000000, seq=10, side="bid", price=100.0, size=2.0)
    assert out[1].side == "ask"  # side_is_bid False -> ask
    assert all(isinstance(d, Delta) for d in out)


def test_trades_from_df_real_lake_schema():
    # Real post-lakeapi trades view (docs/data.md §4.1): quantity + trade_id + side.
    df = pd.DataFrame({
        "origin_time": pd.to_datetime([1668470400000000000]),  # ns int -> datetime64[ns]
        "price": [101.0],
        "quantity": [0.5],
        "side": ["buy"],
        "trade_id": pd.array([777], dtype="int64"),
    })
    out = trades_from_df(df, engine_time_col="origin_time")
    assert out[0] == Trade(ts_engine=1668470400000000000, seq=777, side="buy", price=101.0, amount=0.5)


def test_ingest_raises_clear_error_when_side_column_missing():
    df = pd.DataFrame({"origin_time": [1], "sequence_number": [1], "price": [1.0], "size": [1.0]})
    with pytest.raises(ValueError, match="delta side"):
        deltas_from_df(df, engine_time_col="origin_time")


@pytest.mark.skipif(not (FIXTURES / "book_delta_v2_sample.parquet").exists(),
                    reason="needs Task 1 fixture")
def test_ingest_real_fixture_smoke():
    df = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    col = "origin_time" if (df.get("origin_time", pd.Series([0])).astype("int64") > 0).mean() > 0.99 else "received_time"
    out = deltas_from_df(df, engine_time_col=col)
    assert len(out) == len(df)
    assert all(d.ts_engine > 0 for d in out[:100])
