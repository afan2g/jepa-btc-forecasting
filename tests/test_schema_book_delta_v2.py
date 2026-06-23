import pytest
import pandas as pd
from tests.conftest import FIXTURES

pytestmark = pytest.mark.skipif(
    not (FIXTURES / "book_delta_v2_sample.parquet").exists(),
    reason="run scripts/verify_book_delta_v2.py first (needs Lake access)",
)

def test_book_delta_v2_has_required_fields():
    df = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    # A usable incremental-L2 stream MUST give us: a sequence, a side, a price,
    # a size, and at least one engine-time column. Exact names confirmed in Task 1;
    # update this set to the observed names if Crypto Lake differs.
    have = set(df.columns)
    assert "sequence_number" in have
    # Real lakeapi book_delta_v2 uses `side_is_bid` (bool); accept legacy names too.
    assert {"side_is_bid", "side", "is_bid"} & have, "no side_is_bid/side/is_bid column"
    assert "price" in have
    assert "size" in have or "amount" in have, "no size/amount column"
    assert {"origin_time", "received_time", "timestamp", "receipt_timestamp"} & have

def test_engine_time_column_is_populated():
    df = pd.read_parquet(FIXTURES / "book_delta_v2_sample.parquet")
    # Pick the first engine-time column that is actually populated; the recon
    # adapter (Task 2) must use the SAME choice. Fails loudly if none are usable.
    candidates = [c for c in ("origin_time", "received_time", "timestamp",
                              "receipt_timestamp") if c in df.columns]
    usable = [c for c in candidates if (df[c].astype("int64") > 0).mean() > 0.99]
    assert usable, f"no populated engine-time column among {candidates}"
