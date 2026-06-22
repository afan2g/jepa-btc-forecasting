"""Probe Crypto Lake: auth, what Coinbase/Binance data exists, and a real Coinbase pull."""
import datetime as dt
import sys
import boto3
import lakeapi
from lakeapi.main import _login

def hr(t): print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)

sess = boto3.Session(region_name="eu-west-1")

hr("0. WHO AM I (lake login)")
try:
    user, method = _login(sess, "trades")
    print("login user :", user)
    print("method     :", method)
except Exception as e:
    print("LOGIN FAILED:", repr(e)); sys.exit(1)

hr("0b. USED DATA / QUOTA")
try:
    print(lakeapi.used_data(sess))
except Exception as e:
    print("used_data error:", repr(e))

# Which tables actually exist for COINBASE, and how many symbol-days each.
for table in ["trades", "book", "book_delta", "candles", "level_1"]:
    hr(f"1. COINBASE availability — table={table}")
    try:
        s = lakeapi.available_symbols(table=table, exchanges=["COINBASE"], boto3_session=sess)
        # show BTC-USD + a few top symbols
        print("total coinbase symbol entries:", len(s))
        if "BTC-USD" in s.index.get_level_values("symbol"):
            print("BTC-USD days_available:",
                  int(s.xs("BTC-USD", level="symbol").iloc[0]))
        print(s.head(8).to_string())
    except Exception as e:
        print(f"  no {table} for COINBASE / error:", repr(e))

# Confirm book_delta_v2 truly absent (spec relies on it for Binance).
hr("2. Does 'book_delta_v2' exist at all? (BINANCE_FUTURES)")
for t in ["book_delta", "book_delta_v2"]:
    try:
        s = lakeapi.available_symbols(table=t, exchanges=["BINANCE_FUTURES"], boto3_session=sess)
        print(f"  {t}: FOUND, {len(s)} symbol entries; BTC-USDT-PERP in index:",
              "BTC-USDT-PERP" in s.index.get_level_values("symbol"))
    except Exception as e:
        print(f"  {t}: NOT FOUND ->", type(e).__name__, str(e)[:120])
