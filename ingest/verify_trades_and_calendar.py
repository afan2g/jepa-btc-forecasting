"""
Addresses reviewer findings (4) usable all-feed calendar and (5) trade-feed validation.
Trades drive the bar clock (spec §5), so validate them directly; and OOS/usable spans must be
the intersection of all required feeds after gaps.

Anchor date is parameterized (END env / argv) — defaults to the 2026-06-22 verification snapshot.
"""
import os, sys, datetime as dt
sys.path.insert(0, "ingest")
import pandas as pd, lakeapi
from verify_lake import lake_session

END = dt.date.fromisoformat(os.environ.get("END", "2026-06-22"))

def hr(t): print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)

# ---------- (5) trade-feed validation ----------------------------------------
def check_trades(sess, exch, sym, day):
    s = dt.datetime.combine(day, dt.time()); e = s + dt.timedelta(days=1)
    df = lakeapi.load_data(table="trades", start=s, end=e, symbols=[sym], exchanges=[exch],
                           boto3_session=sess, drop_partition_cols=True)
    print(f"\n  {exch} {sym} trades {day}: rows {len(df):,}  cols {list(df.columns)}")
    if df.empty: print("    EMPTY"); return
    for c in ("origin_time", "received_time"):
        if c in df.columns:
            emp = (df[c] < pd.Timestamp("2015-01-01")).mean()
            print(f"    {c}: empty {emp:.4%}")
    if {"origin_time", "received_time"} <= set(df.columns):
        lag = (df["received_time"] - df["origin_time"]).dt.total_seconds() * 1e3
        print(f"    received-origin lag ms: median {lag.median():.1f} | p95 {lag.quantile(.95):.1f} "
              f"| negative {(lag < 0).mean():.3%}")
    if "side" in df.columns:
        print(f"    side values: {df['side'].value_counts(dropna=False).to_dict()}")
    idc = "trade_id" if "trade_id" in df.columns else ("id" if "id" in df.columns else None)
    if idc:
        u = df[idc].nunique(); mono = df[idc].is_monotonic_increasing
        print(f"    {idc}: unique {u:,}/{len(df):,} ({'UNIQUE' if u==len(df) else 'DUPES'}) | "
              f"monotonic-in-file {mono} | dtype {df[idc].dtype}")
    if "origin_time" in df.columns:
        print(f"    ordered by origin_time in file: {df['origin_time'].is_monotonic_increasing}")

# ---------- (4) usable all-feed calendar -------------------------------------
def present(sess, table, exch, sym, days):
    start = dt.datetime.combine(END, dt.time()) - dt.timedelta(days=days)
    objs = lakeapi.list_data(table=table, start=start, end=dt.datetime.combine(END, dt.time()),
                             exchanges=[exch], symbols=[sym], boto3_session=sess)
    return {dt.date.fromisoformat(o["dt"]) for o in objs}

def runs(daysset, start, end):
    cal = [start + dt.timedelta(i) for i in range((end - start).days)]
    out, cur = [], None
    for d in cal:
        if d in daysset:
            cur = [d, d] if cur is None else [cur[0], d]
        elif cur:
            out.append((cur[0], cur[1], (cur[1]-cur[0]).days+1)); cur = None
    if cur: out.append((cur[0], cur[1], (cur[1]-cur[0]).days+1))
    return out

def calendar(sess, days=730, require_funding_oi=True):
    hr(f"(4) USABLE ALL-FEED CALENDAR — {days}d ending {END}")
    start = END - dt.timedelta(days=days)
    feeds = {
        "binF_book":  ("book_delta_v2", "BINANCE_FUTURES", "BTC-USDT-PERP"),
        "binF_trade": ("trades",        "BINANCE_FUTURES", "BTC-USDT-PERP"),
        "binS_book":  ("book_delta_v2", "BINANCE",         "BTC-USDT"),
        "binS_trade": ("trades",        "BINANCE",         "BTC-USDT"),
        "cb_book":    ("book_delta_v2", "COINBASE",        "BTC-USD"),
        "cb_trade":   ("trades",        "COINBASE",        "BTC-USD"),
        "funding":    ("funding",       "BINANCE_FUTURES", "BTC-USDT-PERP"),
        "oi":         ("open_interest", "BINANCE_FUTURES", "BTC-USDT-PERP"),
    }
    P = {k: present(sess, *v, days) for k, v in feeds.items()}
    cal_n = (END - start).days
    for k in feeds: print(f"  {k:11}: {len(P[k])}/{cal_n} days")

    binance = P["binF_book"] & P["binF_trade"] & P["binS_book"] & P["binS_trade"]
    if require_funding_oi: binance &= P["funding"] & P["oi"]
    lake_all = binance & P["cb_book"] & P["cb_trade"]          # everything from Lake
    usable   = binance                                        # Coinbase is CoinAPI-backfillable
    print(f"\n  Binance-side intersection (the binding constraint): {len(binance)}/{cal_n}")
    print(f"  Lake-only all-feed intersection (incl. Coinbase):   {len(lake_all)}/{cal_n}")
    print(f"  USABLE with Coinbase backfill:                      {len(usable)}/{cal_n} "
          f"({100*len(usable)/cal_n:.1f}%)")
    # Coinbase days that must be CoinAPI-filled (usable-Binance days where Lake Coinbase missing)
    fill = sorted(d for d in usable if d not in (P["cb_book"] & P["cb_trade"]))
    print(f"  Coinbase days needing CoinAPI fill (within usable): {len(fill)}")
    # candidate OOS month = most recent contiguous usable run >= 21 days
    r = [x for x in runs(usable, start, END) if x[2] >= 21]
    print(f"\n  contiguous usable runs >=21d (OOS candidates):")
    for a, b, n in sorted(r, key=lambda x: -x[1].toordinal())[:6]:
        print(f"    {a} .. {b}  ({n}d)")

if __name__ == "__main__":
    sess = lake_session()
    day = dt.date(2025, 6, 1)
    hr(f"(5) TRADE-FEED VALIDATION on {day}")
    for exch, sym in [("BINANCE_FUTURES","BTC-USDT-PERP"), ("BINANCE","BTC-USDT"), ("COINBASE","BTC-USD")]:
        check_trades(sess, exch, sym, day)
    calendar(sess, days=730, require_funding_oi=True)
