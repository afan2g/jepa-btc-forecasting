"""
CoinAPI Market Data REST verifier for Coinbase BTC-USD (the product the $25 free
credit funds). Cheap validation, ~a handful of credits total:

  1. /v1/symbols          -> authoritative coverage dates (trade/orderbook/quote start+end)
  2. /v1/trades/.../history  -> tick-trade schema + exchange timestamps   (1 credit / 100 pts)
  3. /v1/orderbooks/.../history -> L2 snapshot schema + depth              (<=10 credits)

NOTE on scope: REST order book is top-N levels (orderbook), NOT the full-depth
incremental `limitbook_full` — that lives in Flat Files (separate credit pool,
see coinapi_flatfiles.py). REST is enough to confirm existence, span, schema and
timestamp population for the spec §4 verification; Flat Files is the production pull.
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from _common import load_env, rest_get, QuotaExceeded, QUOTA_HINT  # noqa: E402

SYMBOL = "COINBASE_SPOT_BTC_USD"


def hr(t): print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def run():
    key = load_env()["COINAPI_KEY"]
    print(f"Using COINAPI_KEY ...{key[-4:]} (len={len(key)})  REST host rest.coinapi.io")

    hr("1. SYMBOL COVERAGE  (/v1/symbols)")
    syms = rest_get(key, f"/v1/symbols?filter_symbol_id={SYMBOL}")
    rec = next((s for s in syms if s.get("symbol_id") == SYMBOL), syms[0] if syms else None)
    if not rec:
        raise SystemExit(f"{SYMBOL} not found in /v1/symbols.")
    print(f"  symbol_id : {rec.get('symbol_id')}  type={rec.get('symbol_type')}")
    for k in ("data_trade_start", "data_trade_end", "data_orderbook_start",
              "data_orderbook_end", "data_quote_start", "data_quote_end"):
        if rec.get(k):
            print(f"  {k:22}: {rec[k]}")

    hr("2. TRADES schema  (/v1/trades/{symbol}/history)")
    start = (rec.get("data_trade_end") or "")[:10] + "T00:00:00"
    trades = rest_get(key, f"/v1/trades/{SYMBOL}/history?time_start={start}&limit=100")
    print(f"  rows: {len(trades)}")
    if trades:
        print(f"  columns: {sorted(trades[0].keys())}")
        t = trades[0]
        for k in ("time_exchange", "time_coinapi", "price", "size", "taker_side", "uuid"):
            if k in t: print(f"    {k}: {t[k]}")

    hr("3. ORDER BOOK schema/depth  (/v1/orderbooks/{symbol}/history, L2 top-N)")
    obs = rest_get(key, f"/v1/orderbooks/{SYMBOL}/history?time_start={start}&limit=1")
    if obs:
        ob = obs[0]
        print(f"  columns: {sorted(ob.keys())}")
        print(f"  time_exchange: {ob.get('time_exchange')}  time_coinapi: {ob.get('time_coinapi')}")
        print(f"  depth this snapshot: {len(ob.get('bids', []))} bids / {len(ob.get('asks', []))} asks")
        if ob.get("bids"):
            print(f"  top bid: {ob['bids'][0]} | top ask: {ob['asks'][0]}")
    else:
        print("  no order book rows returned.")

    hr("VERDICT (REST)")
    print(f"  Coinbase BTC-USD trades:   {rec.get('data_trade_start')} -> {rec.get('data_trade_end')}")
    print(f"  Coinbase BTC-USD orderbook:{rec.get('data_orderbook_start')} -> {rec.get('data_orderbook_end')}")
    print("  Full-depth incremental book = Flat Files (run coinapi_flatfiles.py once that pool is funded).")


if __name__ == "__main__":
    try:
        run()
    except QuotaExceeded:
        print(QUOTA_HINT)
        sys.exit(2)
