"""E0.2: pull a minimal book_delta_v2 + trades sample, verify schema, capture fixtures.

Spec §4 check #1: is origin_time populated for Binance book_delta_v2? If empty
(0/-1), reconstruction falls back to received_time. Writes small parquet fixtures.
"""
import datetime as dt
import pathlib
import sys

import lakeapi

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"
OUT.mkdir(parents=True, exist_ok=True)

# Use the SAME explicit Crypto Lake session as the other ingest scripts: the
# subscriber keys come from .env, NOT the personal ~/.aws default profile (which would
# auth into the wrong account and fail with AccessDenied even when valid keys exist).
sys.path.insert(0, str(ROOT / "ingest"))
from verify_lake import lake_session  # noqa: E402

sess = lake_session()

# A 2-minute window AFTER Binance-futures book history start (2022-11-14, spec §4).
start = dt.datetime(2022, 11, 15, 0, 0, 0)
end = dt.datetime(2022, 11, 15, 0, 2, 0)

print("used_data BEFORE:", lakeapi.used_data(sess))

deltas = lakeapi.load_data(
    table="book_delta_v2", start=start, end=end,
    symbols=["BTC-USDT-PERP"], exchanges=["BINANCE_FUTURES"], boto3_session=sess,
)
trades = lakeapi.load_data(
    table="trades", start=start, end=end,
    symbols=["BTC-USDT-PERP"], exchanges=["BINANCE_FUTURES"], boto3_session=sess,
)

print("delta rows:", len(deltas), "cols:", list(deltas.columns))
print("delta dtypes:\n", deltas.dtypes)
print("delta head:\n", deltas.head(5).to_string())

# §4 origin_time population check.
for col in ("origin_time", "received_time", "timestamp", "receipt_timestamp"):
    if col in deltas.columns:
        empty = (deltas[col].astype("int64") <= 0).mean()
        print(f"  {col}: present, fraction<=0 = {empty:.3%}")

# Capture small fixtures (first ~5k delta rows, all trades in window).
deltas.head(5000).to_parquet(OUT / "book_delta_v2_sample.parquet")
trades.to_parquet(OUT / "trades_sample.parquet")
print("WROTE fixtures to", OUT)
print("used_data AFTER:", lakeapi.used_data(sess))
