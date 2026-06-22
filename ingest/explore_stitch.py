"""
Stitch feasibility: (A) does CoinAPI cover Crypto Lake's Coinbase gap days, and
(B) do the two vendors agree on a clean overlap day (so a stitched series is seamless)?

Note: Crypto Lake `book_delta_v2` has NO per-day snapshot seed and uses absolute-size /
0=remove updates — reconstructing it standalone-per-day is invalid (recon must carry book
state across days). So for vendor agreement we use Crypto Lake's `book` snapshot product on a
day where it is clean (0% crossed) vs CoinAPI L1 `quotes`.
"""
import os, sys, datetime as dt, tempfile
sys.path.insert(0, "ingest")
import pandas as pd, numpy as np, lakeapi
from verify_lake import lake_session
import coinapi_flatfiles as ff
from _common import load_env

CB = "+SC-COINBASE_SPOT_BTC_USD+"

def hr(t): print("\n" + "=" * 72 + f"\n{t}\n" + "=" * 72)

# ---- (A) CoinAPI coverage of Crypto Lake's gap days -------------------------
def coinapi_has(s3, day, table="LIMITBOOK_FULL"):
    objs, _ = ff.list_prefix(s3, "coinapi", f"T-{table}/D-{day:%Y%m%d}/E-COINBASE/")
    hit = [o for o in objs if CB in o["Key"]]
    return hit[0]["Size"] if hit else 0

def part_A(s3):
    hr("A. CoinAPI coverage of Crypto Lake Coinbase gap days")
    # sample the big 33-day gap + the other holes (throttled; ~14 list calls)
    days = ([dt.date(2024,12,5)+dt.timedelta(d) for d in range(0,33,5)] +
            [dt.date(2024,11,26), dt.date(2024,11,29),
             dt.date(2025,3,4), dt.date(2025,1,20), dt.date(2024,7,14)])
    ok = 0
    for d in sorted(days):
        lb = coinapi_has(s3, d, "LIMITBOOK_FULL"); q = coinapi_has(s3, d, "QUOTES")
        present = lb > 0
        ok += present
        print(f"  {d}  limitbook_full {lb/1e6:7.0f}MB | quotes {q/1e6:6.0f}MB  {'OK' if present else 'MISSING'}")
    print(f"\n  CoinAPI covers {ok}/{len(days)} sampled gap days "
          f"-> {'can fill the gaps' if ok==len(days) else 'PARTIAL — investigate'}")

# ---- (B) vendor agreement on a clean overlap day ----------------------------
def load_lake_book(sess, day):
    s = dt.datetime.combine(day, dt.time()); e = s + dt.timedelta(days=1)
    df = lakeapi.load_data(table="book", start=s, end=e, symbols=["BTC-USD"], exchanges=["COINBASE"],
        columns=["timestamp", "bid_0_price", "ask_0_price"], boto3_session=sess, drop_partition_cols=True)
    df = df.rename(columns={"origin_time": "t"}).set_index("t").sort_index()
    df["mid"] = (df["bid_0_price"] + df["ask_0_price"]) / 2
    crossed = (df["ask_0_price"] <= df["bid_0_price"]).mean()
    return df[["bid_0_price", "ask_0_price", "mid"]], crossed

def load_coinapi_quotes(s3, day):
    objs, _ = ff.list_prefix(s3, "coinapi", f"T-QUOTES/D-{day:%Y%m%d}/E-COINBASE/")
    key = [o["Key"] for o in objs if CB in o["Key"]][0]
    tmp = os.path.join(tempfile.gettempdir(), "cbquotes.csv.gz")
    ff.RL.wait()
    body = s3.get_object(Bucket="coinapi", Key=key)["Body"]
    with open(tmp, "wb") as f:
        while (b := body.read(8 << 20)): f.write(b)
    df = pd.read_csv(tmp, sep=";", usecols=["time_exchange", "ask_px", "bid_px"])
    os.remove(tmp)
    df["t"] = pd.to_datetime(df["time_exchange"], format="ISO8601")
    df = df.set_index("t").sort_index()
    df["mid"] = (df["bid_px"] + df["ask_px"]) / 2
    return df[["bid_px", "ask_px", "mid"]]

def part_B(sess, s3, day):
    hr(f"B. Vendor agreement on clean overlap day {day} (exchange-time, 1s grid)")
    lake, crossed = load_lake_book(sess, day)
    print(f"  Crypto Lake book rows {len(lake):,} | crossed {crossed:.2%}")
    cap = load_coinapi_quotes(s3, day)
    print(f"  CoinAPI quotes rows {len(cap):,}")
    grid = pd.date_range(day.isoformat(), periods=24*3600, freq="1s", tz=None)
    L = pd.merge_asof(pd.DataFrame(index=grid), lake, left_index=True, right_index=True)
    C = pd.merge_asof(pd.DataFrame(index=grid), cap, left_index=True, right_index=True)
    m = pd.DataFrame({"lake": L["mid"], "cap": C["mid"]}).dropna()
    d = (m["lake"] - m["cap"]).abs()
    print(f"  aligned grid points: {len(m):,}")
    print(f"  |Δmid| $:  median {d.median():.3f} | mean {d.mean():.3f} | p95 {d.quantile(.95):.3f} | max {d.max():.2f}")
    print(f"  |Δmid| as bps of price: median {1e4*d.median()/m['cap'].median():.3f} bps")
    print(f"  fraction |Δmid|<$1: {(d<1).mean():.2%} | <$5: {(d<5).mean():.2%}")
    print(f"  mid correlation: {m['lake'].corr(m['cap']):.6f}")
    print(f"  >>> {'AGREE — stitch is seamless' if d.median()<2 and m['lake'].corr(m['cap'])>0.999 else 'DISAGREE — investigate'}")

if __name__ == "__main__":
    sess = lake_session()
    s3 = ff.make_client(load_env()["COINAPI_KEY"])
    part_A(s3)
    part_B(sess, s3, dt.date(2025, 6, 1))   # Lake book is 0% crossed here
