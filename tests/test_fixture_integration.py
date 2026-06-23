import pandas as pd
import pytest
from tests.conftest import FIXTURES
from recon.ingest import deltas_from_df, trades_from_df, shared_engine_time_col
from recon.reconstruct import reconstruct_book_at_trades

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "book_delta_v2_sample.parquet").exists(),
    reason="run scripts/verify_book_delta_v2.py first (needs Lake access)",
)


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
    col = shared_engine_time_col(dd, tt)  # ONE clock for both streams (plan §5.3)
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
