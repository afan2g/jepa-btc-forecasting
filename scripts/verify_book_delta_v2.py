"""E0.2: pull a minimal book_delta_v2 + trades sample, verify schema, capture fixtures.

Spec §4 check #1: is origin_time populated for Binance book_delta_v2? If empty
(0/-1), reconstruction falls back to received_time. Writes small parquet fixtures.

The live, billable Lake work lives in main() under `if __name__ == "__main__"`, so
importing this module (e.g. to reuse/test engine_col() or lake_session()) does NOT touch
vendor APIs — only an explicit `python scripts/verify_book_delta_v2.py` run does
(AGENTS.md: separate cheap local checks from live vendor calls).
"""
import datetime as dt
import os
import pathlib

import boto3
import lakeapi

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "tests" / "fixtures"


def lake_session():
    """Crypto Lake boto3 session from the .env subscriber keys.

    Mirrors ingest/verify_lake.py::lake_session() credential semantics (keys from .env,
    NOT the personal ~/.aws default profile, which would auth into the wrong account and
    fail with AccessDenied), but does NOT require the unrelated COINAPI_KEY that
    ingest._common.load_env() demands — this E0.2 capture is Lake-only.
    """
    env = {}
    envpath = ROOT / ".env"
    if envpath.exists():
        for line in envpath.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip().strip('"').strip("'")
    env = {**env, **os.environ}
    try:
        return boto3.Session(
            aws_access_key_id=env["aws_access_key_id"],
            aws_secret_access_key=env["aws_secret_access_key"],
            region_name=env.get("region", "eu-west-1"),
        )
    except KeyError as e:
        raise SystemExit(
            f"Crypto Lake AWS key {e} not found in .env or environment "
            "(need aws_access_key_id and aws_secret_access_key)."
        ) from None


def engine_col(df):
    """First engine-time column that is present AND populated (>0 for ~all rows).

    origin_time may be PRESENT but empty — the spec §4 fallback case this script exists to
    detect — so presence alone is not enough: an unpopulated origin_time would yield a
    garbage cutoff and filter the trade fixture to the wrong span. This mirrors the
    populated-column selection in tests/test_fixture_integration.py::_engine_col, so the
    cutoff uses the SAME column reconstruction will use. astype('int64') normalizes both
    int64 ns and datetime64[ns] (NaT/epoch-0 -> <=0, correctly rejected)."""
    for c in ("origin_time", "received_time", "timestamp", "receipt_timestamp"):
        if c in df.columns and (df[c].astype("int64") > 0).mean() > 0.99:
            return c
    raise SystemExit(f"no populated engine-time column in {list(df.columns)}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
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

    # Capture a SMALL but INTERNALLY CONSISTENT fixture: keep the first ~5k deltas, and if
    # that truncates the window, keep only trades fully covered by those deltas (engine-time
    # strictly before the last retained delta) so reconstruction never snapshots a trade
    # against a book missing later deltas dropped by the row cap.
    MAX_DELTAS = 5000
    ts_col = engine_col(deltas)
    deltas_kept = deltas.head(MAX_DELTAS)
    if len(deltas_kept) < len(deltas):
        cutoff = deltas_kept[ts_col].max()
        trades_kept = trades[trades[ts_col] < cutoff]
        print(f"  truncated to {len(deltas_kept)} deltas; trades filtered to {ts_col} < {cutoff} "
              f"-> {len(trades_kept)}/{len(trades)} trades")
    else:
        trades_kept = trades

    deltas_kept.to_parquet(OUT / "book_delta_v2_sample.parquet")
    trades_kept.to_parquet(OUT / "trades_sample.parquet")
    print(f"WROTE fixtures to {OUT}: {len(deltas_kept)} deltas, {len(trades_kept)} trades")
    print("used_data AFTER:", lakeapi.used_data(sess))


if __name__ == "__main__":
    main()
