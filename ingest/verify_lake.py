"""
Crypto Lake access verification (spec §4 ⚠️ "verify on samples before committing").

Uses the Crypto Lake AWS keys from .env (NOT the personal ~/.aws default) via an
explicit boto3 session passed to lakeapi. Cheap checks first (auth, quota, table
existence/coverage via metadata listing — no parquet download), then the one
load-bearing download check: is `origin_time` populated in Binance book_delta_v2?
"""
from __future__ import annotations
import os
import sys
import datetime as dt

import boto3
import pandas as pd
import lakeapi

sys.path.insert(0, os.path.dirname(__file__))
from _common import load_env  # noqa: E402

BIN_FUT = "BINANCE_FUTURES"
PERP = "BTC-USDT-PERP"
# Anchor "today" for coverage windows. Defaults to the original verification snapshot;
# override to refresh, e.g. END=2026-09-01 python ingest/verify_lake.py
END = dt.date.fromisoformat(os.environ.get("END", "2026-06-22"))


def hr(t): print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def lake_session():
    env = load_env()
    return boto3.Session(
        aws_access_key_id=env["aws_access_key_id"],
        aws_secret_access_key=env["aws_secret_access_key"],
        region_name=env.get("region", "eu-west-1"),
    )


def coverage(sess, table, exchange, symbol, days=120):
    """Day-level coverage/gaps from metadata listing (no parquet download)."""
    end = dt.datetime.combine(END, dt.time())
    start = end - dt.timedelta(days=days)
    objs = lakeapi.list_data(table=table, start=start, end=end,
                             exchanges=[exchange], symbols=[symbol], boto3_session=sess)
    have = sorted({o["dt"] for o in objs})
    if not have:
        return 0, days, []
    present = {dt.date.fromisoformat(d) for d in have}
    cal = {(start.date() + dt.timedelta(days=i)) for i in range((end.date() - start.date()).days)}
    missing = sorted(d.isoformat() for d in (cal - present))
    return len(present), len(cal), missing


def main():
    sess = lake_session()
    cid = sess.client("sts").get_caller_identity()
    print(f"AWS identity: acct={cid['Account']} arn={cid['Arn']}")

    hr("0. LAKE LOGIN / QUOTA")
    from lakeapi.main import _login
    user, method = _login(sess, "trades")
    print(f"  lake user: {user} | method: {method}"
          + ("  (routing-lambda fallback; data reads go direct-S3 — fine)" if user == "unknown" else ""))
    try:
        print(f"  used_data: {lakeapi.used_data(sess)}")
    except Exception as e:
        print(f"  used_data error: {e!r}")

    hr("1. BINANCE FUTURES — which tables exist for BTC-USDT-PERP? (metadata only)")
    for table in ["book", "book_delta", "book_delta_v2", "trades", "funding",
                  "open_interest", "liquidations"]:
        try:
            n, tot, miss = coverage(sess, table, BIN_FUT, PERP, days=120)
            print(f"  {table:15}: {n}/{tot} days (last 120d), gaps {len(miss)}")
        except Exception as e:
            print(f"  {table:15}: — not found ({type(e).__name__})")

    hr("2. COINBASE — coverage of BTC-USD (confirm the ~80%/gaps story)")
    for table in ["book", "book_delta_v2", "trades"]:
        try:
            n, tot, miss = coverage(sess, table, "COINBASE", "BTC-USD", days=120)
            pct = 100 * n / tot if tot else 0
            print(f"  {table:15}: {n}/{tot} days ({pct:.0f}%), gaps {len(miss)}"
                  f"{'  e.g. ' + str(miss[:6]) if miss else ''}")
        except Exception as e:
            print(f"  {table:15}: — not found ({type(e).__name__})")

    hr("3. spec §4 #1 — is origin_time POPULATED in Binance book_delta_v2? (1-day download)")
    table = "book_delta_v2"
    # a day ~80d before the anchor (consolidated region); may land on a gap —
    # verify_lake2.py picks a guaranteed-present date instead.
    start = dt.datetime.combine(END - dt.timedelta(days=82), dt.time())
    end = start + dt.timedelta(days=1)
    df = None
    for cols in (["origin_time", "received_time"], ["timestamp", "receipt_timestamp"], None):
        try:
            df = lakeapi.load_data(table=table, start=start, end=end, symbols=[PERP],
                                   exchanges=[BIN_FUT], columns=cols, boto3_session=sess,
                                   drop_partition_cols=True)
            print(f"  loaded with columns={cols}")
            break
        except Exception as e:
            print(f"  columns={cols} failed: {type(e).__name__}: {str(e)[:80]}")
    if df is None or df.empty:
        print("  no book_delta_v2 rows for that day."); return
    print(f"  rows: {len(df):,} | columns: {list(df.columns)}")
    if "origin_time" in df.columns:
        empty = (df["origin_time"] < pd.Timestamp("2015-01-01")).mean()
        print(f"  origin_time: empty/invalid fraction = {empty:.4%}")
        print(f"  origin_time range: {df['origin_time'].min()} -> {df['origin_time'].max()}")
        print(f"  >>> origin_time {'USABLE (reconstruct on exchange time)' if empty < 0.01 else 'MOSTLY EMPTY -> fall back to received_time'}")
    if "received_time" in df.columns:
        emptyr = (df["received_time"] < pd.Timestamp("2015-01-01")).mean()
        print(f"  received_time: empty fraction = {emptyr:.4%} (Tokyo capture proxy)")


if __name__ == "__main__":
    main()
