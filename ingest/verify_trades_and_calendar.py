"""
(4) usable all-feed calendar and (5) trade-feed validation.
Trades drive the bar clock (spec §5), so validate them directly; OOS/usable spans must be the
intersection of all required feeds after gaps.

The "usable with Coinbase backfill" set is only an ASSUMPTION until we confirm CoinAPI actually has a
consolidated flat file for each Coinbase fill day. Pass --verify-backfill to probe CoinAPI per fill day
(throttled ~8/min) and report usable_after_verified_backfill. The exact fill-day list is written to JSON.

Anchor date: --end YYYY-MM-DD, else $END, else 2026-06-22 (the verification snapshot).
Run from anywhere: paths resolve via __file__.
"""
import os, sys, json, argparse, datetime as dt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd, lakeapi
from verify_lake import lake_session

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
def present(sess, table, exch, sym, end, days):
    start = dt.datetime.combine(end, dt.time()) - dt.timedelta(days=days)
    objs = lakeapi.list_data(table=table, start=start, end=dt.datetime.combine(end, dt.time()),
                             exchanges=[exch], symbols=[sym], boto3_session=sess)
    return {dt.date.fromisoformat(o["dt"]) for o in objs}

def coinapi_fill_status(fill_days, min_mb=50):
    """For each Coinbase fill day, does CoinAPI have a consolidated daily limitbook_full file?
    Returns {day: {'ok': bool, 'mb': float, 'reason': str}}. Throttled (~8/min)."""
    import coinapi_flatfiles as ff
    from _common import load_env
    s3 = ff.make_client(load_env()["COINAPI_KEY"])
    out = {}
    for d in fill_days:
        try:
            res = ff.coinbase_btc_file(s3, "coinapi", d.strftime("%Y%m%d"), "LIMITBOOK_FULL")
        except Exception as e:
            out[d] = {"ok": False, "mb": 0.0, "reason": f"error:{type(e).__name__}"}; continue
        if not res:
            out[d] = {"ok": False, "mb": 0.0, "reason": "missing/hourly-tail-only"}
        else:
            mb = res[1] / 1e6
            out[d] = {"ok": mb >= min_mb, "mb": round(mb, 1),
                      "reason": "ok" if mb >= min_mb else f"suspiciously small ({mb:.0f}MB)"}
    return out

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

def calendar(sess, end, days=730, require_funding_oi=True, verify_backfill=False,
             out="data/usable_calendar.json"):
    hr(f"(4) USABLE ALL-FEED CALENDAR — {days}d ending {end}")
    start = end - dt.timedelta(days=days)
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
    P = {k: present(sess, *v, end, days) for k, v in feeds.items()}
    cal_n = (end - start).days
    for k in feeds: print(f"  {k:11}: {len(P[k])}/{cal_n} days")

    binance = P["binF_book"] & P["binF_trade"] & P["binS_book"] & P["binS_trade"]
    if require_funding_oi: binance &= P["funding"] & P["oi"]
    lake_all = binance & P["cb_book"] & P["cb_trade"]          # everything from Lake
    usable_assumed = binance                                  # assumes Coinbase is CoinAPI-backfillable
    fill = sorted(d for d in usable_assumed if d not in (P["cb_book"] & P["cb_trade"]))
    print(f"\n  Binance-side intersection (the binding constraint): {len(binance)}/{cal_n}")
    print(f"  Lake-only all-feed intersection (incl. Coinbase):   {len(lake_all)}/{cal_n}")
    print(f"  USABLE assuming Coinbase backfill:                  {len(usable_assumed)}/{cal_n} "
          f"({100*len(usable_assumed)/cal_n:.1f}%)  [ASSUMPTION until --verify-backfill]")
    print(f"  Coinbase days needing CoinAPI fill (within usable): {len(fill)}")

    # (High finding) verify CoinAPI actually has each fill day; don't assume.
    status, usable_verified = None, None
    if verify_backfill and fill:
        hr(f"  Verifying CoinAPI flat-file availability for {len(fill)} fill days (throttled)")
        status = coinapi_fill_status(fill)
        bad = [d for d in fill if not status[d]["ok"]]
        usable_verified = sorted(usable_assumed - set(bad))
        for d in fill:
            s = status[d]; print(f"    {d}  {'OK ' if s['ok'] else 'BAD'}  {s['mb']:7.0f}MB  {s['reason']}")
        print(f"\n  fill days CoinAPI CANNOT cover: {len(bad)} -> {[d.isoformat() for d in bad]}")
        print(f"  USABLE after verified backfill: {len(usable_verified)}/{cal_n} "
              f"({100*len(usable_verified)/cal_n:.1f}%)")
    elif fill:
        print("  (skipped CoinAPI verification — pass --verify-backfill to confirm the fill set)")

    base = usable_verified if usable_verified is not None else sorted(usable_assumed)
    r = [x for x in runs(set(base), start, end) if x[2] >= 21]
    print(f"\n  contiguous usable runs >=21d (OOS candidates):")
    for a, b, n in sorted(r, key=lambda x: -x[1].toordinal())[:6]:
        print(f"    {a} .. {b}  ({n}d)")

    # (finding 3) emit the actual fill-day list + the calendar, not just counts
    rec = {
        "anchor_end": end.isoformat(), "days": days, "require_funding_oi": require_funding_oi,
        "feed_present_counts": {k: len(P[k]) for k in feeds},
        "binance_intersection": len(binance),
        "usable_assumed_backfill": len(usable_assumed),
        "coinbase_fill_days": [d.isoformat() for d in fill],
        "backfill_verified": bool(verify_backfill and fill),
        "fill_status": ({d.isoformat(): status[d] for d in fill} if status else None),
        "usable_after_verified_backfill": (len(usable_verified) if usable_verified is not None else None),
        "fill_days_unfillable": ([d.isoformat() for d in fill if status and not status[d]["ok"]]
                                 if status else None),
        "oos_candidate_runs": [[a.isoformat(), b.isoformat(), n] for a, b, n in r],
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w") as f: json.dump(rec, f, indent=2)
    print(f"\n  wrote calendar + fill-day list -> {out}")

def parse_args():
    p = argparse.ArgumentParser(description="Trade-feed validation + usable all-feed calendar")
    p.add_argument("--end", default=os.environ.get("END", "2026-06-22"), help="anchor 'today' YYYY-MM-DD")
    p.add_argument("--days", type=int, default=730)
    p.add_argument("--no-funding-oi", action="store_true", help="don't require funding/OI in the intersection")
    p.add_argument("--verify-backfill", action="store_true",
                   help="probe CoinAPI for each Coinbase fill day (throttled ~8/min)")
    p.add_argument("--trade-day", default="2025-06-01", help="day for trade-feed validation")
    p.add_argument("--out", default="data/usable_calendar.json")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    end = dt.date.fromisoformat(a.end)
    sess = lake_session()
    day = dt.date.fromisoformat(a.trade_day)
    hr(f"(5) TRADE-FEED VALIDATION on {day}")
    for exch, sym in [("BINANCE_FUTURES","BTC-USDT-PERP"), ("BINANCE","BTC-USDT"), ("COINBASE","BTC-USD")]:
        check_trades(sess, exch, sym, day)
    calendar(sess, end, days=a.days, require_funding_oi=not a.no_funding_oi,
             verify_backfill=a.verify_backfill, out=a.out)
