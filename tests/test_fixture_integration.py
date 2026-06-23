import pandas as pd
import pytest
from tests.conftest import FIXTURES
from recon.ingest import deltas_from_df, trades_from_df
from recon.reconstruct import reconstruct_book_at_trades

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "book_delta_v2_sample.parquet").exists(),
    reason="run scripts/verify_book_delta_v2.py first (needs Lake access)",
)


def _engine_col(df):
    for c in ("origin_time", "received_time", "timestamp", "receipt_timestamp"):
        if c in df.columns and (df[c].astype("int64") > 0).mean() > 0.99:
            return c
    raise AssertionError("no populated engine-time column")


def test_reconstruct_real_sample_is_sane():
    dd = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    tt = pd.read_parquet(FIXTURES / "trades_sample.parquet")
    deltas = deltas_from_df(dd, engine_time_col=_engine_col(dd))
    trades = trades_from_df(tt, engine_time_col=_engine_col(tt))
    out = reconstruct_book_at_trades(deltas, trades, k=10)
    assert len(out) > 0
    valid = out.dropna(subset=["bid_0_price", "ask_0_price"])
    # No crossed book once both sides exist.
    assert (valid["ask_0_price"] > valid["bid_0_price"]).all()
    # Trade timestamps are non-decreasing (total order preserved).
    assert out["trade_ts"].is_monotonic_increasing
