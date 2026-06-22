"""What does the FREE sample lake contain? Can we pull Coinbase + inspect book schema?"""
import boto3, lakeapi
def hr(t): print("\n"+"="*70+f"\n{t}\n"+"="*70)
lakeapi.use_sample_data(anonymous_access=True)
sess = boto3.Session(region_name="eu-west-1")

for table in ["trades","book","book_delta","level_1","candles","funding","open_interest","liquidations"]:
    hr(f"sample table={table}: exchange/symbol -> days")
    try:
        s = lakeapi.available_symbols(table=table, boto3_session=sess)
        print(s.to_string())
    except Exception as e:
        print("  none/error:", type(e).__name__, str(e)[:100])

# If Coinbase is present, pull BTC-USD trades+book and inspect.
for table in ["trades","book","book_delta"]:
    hr(f"PULL Coinbase BTC-USD {table}")
    try:
        df = lakeapi.load_data(table=table, start=None, end=None,
                               symbols=["BTC-USD"], exchanges=["COINBASE"])
        print("rows:", len(df), "cols:", list(df.columns))
        print(df.head(2).to_string())
    except Exception as e:
        print("  not in sample/error:", type(e).__name__, str(e)[:120])

# Inspect a Binance book to confirm level count + origin_time population (spec §4 check #1)
hr("Binance BTC-USDT 'book' schema — level count + origin_time populated?")
try:
    df = lakeapi.load_data(table="book", symbols=["BTC-USDT"], exchanges=["BINANCE"])
    print("rows:", len(df), "ncols:", len(df.columns))
    print("cols:", list(df.columns)[:12], "...")
    bidcols = [c for c in df.columns if c.startswith("bid_") and "price" in c]
    print("bid price levels:", len(bidcols))
    if "origin_time" in df.columns:
        z = (df["origin_time"].astype("int64") <= 0).mean()
        print(f"origin_time present; fraction <=0 (empty): {z:.3%}")
        print("origin_time sample:", df["origin_time"].iloc[0])
except Exception as e:
    print("  error:", type(e).__name__, str(e)[:150])
