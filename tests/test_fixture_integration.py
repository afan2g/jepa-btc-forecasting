import pandas as pd
import pytest
from tests.conftest import FIXTURES
from recon.ingest import deltas_from_df, trades_from_df
from recon.reconstruct import reconstruct_book_at_trades

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "book_delta_v2_sample.parquet").exists(),
    reason="run scripts/verify_book_delta_v2.py first (needs Lake access)",
)


def _shared_engine_col(*dfs):
    """First engine-time column populated (>0 for ~all rows) in EVERY stream.

    The recon convention (plan §5.3) requires ONE engine-time axis used identically for
    deltas and trades. Selecting per-stream would, in the §4 fallback case (book
    origin_time empty but trades' populated), put deltas on received_time and trades on
    origin_time — mixing exchange and capture clocks and shifting trade/book order by
    capture latency. Picking a column populated in both (origin_time preferred; docs/data.md
    §5 'recon must align on origin_time') keeps a single clock, and fails loudly otherwise."""
    for c in ("origin_time", "received_time", "timestamp", "receipt_timestamp"):
        if all(c in df.columns and (df[c].astype("int64") > 0).mean() > 0.99 for df in dfs):
            return c
    raise AssertionError("no engine-time column populated in all streams")


def test_reconstruct_real_sample_is_sane():
    """SMOKE test only — NOT proof of top-K replay correctness.

    book_delta_v2 is a mid-stream incremental feed with NO per-day snapshot block
    (docs/data.md §4.1, §5a-Recon): a captured slice starts after the real book already
    exists, so reconstructing it from a COLD-START empty OrderBook leaves early top-K
    levels partial/untrustworthy. We therefore assert only pipeline-level sanity (it runs
    end-to-end; the total order is preserved; top-of-book is uncrossed where both sides
    exist) — validating top-K against the real book requires the deferred snapshot seed /
    warm-up gate (docs/data.md §5a-Recon)."""
    dd = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    tt = pd.read_parquet(FIXTURES / "trades_sample.parquet")
    col = _shared_engine_col(dd, tt)  # ONE clock for both streams (plan §5.3)
    deltas = deltas_from_df(dd, engine_time_col=col)
    trades = trades_from_df(tt, engine_time_col=col)
    out = reconstruct_book_at_trades(deltas, trades, k=10)
    assert len(out) > 0
    # Total order preserved (valid regardless of the cold start).
    assert out["trade_ts"].is_monotonic_increasing
    # Top-of-book sanity where both sides exist. Weak by design: a partial cold-start book
    # can still be uncrossed at the top, so this does NOT certify the top-K levels.
    valid = out.dropna(subset=["bid_0_price", "ask_0_price"])
    assert (valid["ask_0_price"] > valid["bid_0_price"]).all()
