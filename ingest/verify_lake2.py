"""Deeper Crypto Lake check: 2-yr coverage + gap STRUCTURE for book_delta_v2,
and origin_time population on dates that actually exist (Binance + Coinbase)."""
from __future__ import annotations
import os, sys, datetime as dt
import boto3, pandas as pd, lakeapi
sys.path.insert(0, os.path.dirname(__file__))
from _common import load_env  # noqa
from verify_lake import lake_session  # noqa

def hr(t): print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)

def present_dates(sess, table, exchange, symbol, days):
    end = dt.datetime(2026, 6, 22); start = end - dt.timedelta(days=days)
    objs = lakeapi.list_data(table=table, start=start, end=end, exchanges=[exchange],
                             symbols=[symbol], boto3_session=sess)
    return sorted({dt.date.fromisoformat(o["dt"]) for o in objs}), start.date(), end.date()

def gap_runs(present, start, end):
    """Return list of (gap_start, gap_end, length) contiguous missing runs."""
    cal = [start + dt.timedelta(days=i) for i in range((end - start).days)]
    pset = set(present); runs = []; cur = None
    for d in cal:
        if d not in pset:
            cur = [d, d] if cur is None else [cur[0], d]
        else:
            if cur: runs.append((cur[0], cur[1], (cur[1]-cur[0]).days + 1)); cur = None
    if cur: runs.append((cur[0], cur[1], (cur[1]-cur[0]).days + 1))
    return runs

def report(sess, exchange, symbol, days=730):
    hr(f"{exchange} {symbol} book_delta_v2 — {days}d coverage + gap structure")
    pres, s, e = present_dates(sess, "book_delta_v2", exchange, symbol, days)
    if not pres:
        print("  none found"); return None
    runs = gap_runs(pres, s, e)
    total = (e - s).days
    print(f"  present {len(pres)}/{total} days ({100*len(pres)/total:.1f}%) | {s}..{e}")
    print(f"  first present: {pres[0]} | last present: {pres[-1]}")
    big = sorted(runs, key=lambda r: -r[2])[:8]
    print(f"  gap runs: {len(runs)} | total missing {total-len(pres)} days")
    for gs, ge, n in big:
        print(f"    {gs} .. {ge}  ({n}d)")
    return pres

def origin_check(sess, exchange, symbol, day):
    hr(f"origin_time in {exchange} {symbol} book_delta_v2 on {day} (1-day, projected)")
    start = dt.datetime.combine(day, dt.time()); end = start + dt.timedelta(days=1)
    df = None
    for cols in (["origin_time","received_time"], ["timestamp","receipt_timestamp"], None):
        try:
            df = lakeapi.load_data(table="book_delta_v2", start=start, end=end, symbols=[symbol],
                                   exchanges=[exchange], columns=cols, boto3_session=sess,
                                   drop_partition_cols=True)
            print(f"  loaded columns={cols}"); break
        except Exception as ex:
            print(f"  columns={cols}: {type(ex).__name__} {str(ex)[:70]}")
    if df is None or df.empty:
        print("  no rows"); return
    print(f"  rows {len(df):,} | cols {list(df.columns)}")
    for c in ("origin_time","received_time"):
        if c in df.columns:
            empty = (df[c] < pd.Timestamp("2015-01-01")).mean()
            print(f"  {c}: empty {empty:.4%} | {df[c].min()} -> {df[c].max()}")

def main():
    sess = lake_session()
    binp = report(sess, "BINANCE_FUTURES", "BTC-USDT-PERP", 730)
    cbp  = report(sess, "COINBASE", "BTC-USD", 730)
    # origin_time on a solidly-present date (5th from the end avoids the publish-lag edge)
    if binp: origin_check(sess, "BINANCE_FUTURES", "BTC-USDT-PERP", binp[-5])
    if cbp:  origin_check(sess, "COINBASE", "BTC-USD", cbp[-5])

if __name__ == "__main__":
    main()
