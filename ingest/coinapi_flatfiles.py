"""
CoinAPI Flat Files S3 client + Coinbase BTC-USD verification report.

Purpose (spec §4 / §12.1): confirm CoinAPI can supply the Coinbase order-book +
trades data the project needs, and verify what docs can't:
  1. day-level coverage / gaps in a consolidated window  (the reason we left Crypto Lake)
  2. limitbook_full schema + exchange-timestamp population (feeds §5.3 event-time recon)
  3. per-day compressed size -> cost projection for the 12-24mo SSL span

Constraints learned from the live bucket (2026-06-22):
  * Free tier = **10 requests/minute** -> every S3 call is throttled (RateLimiter).
  * Partitions are **daily** (D-YYYYMMDD) once consolidated, but the most recent
    ~2-3 weeks live as an **hourly tail** (D-YYYYMMDDHH) and/or in the separate
    `coinapi-daily-tail` bucket. So coverage scans target the consolidated region
    (default end = today-25d). REST /v1/symbols gives the authoritative full span.
  * Symbol match must be exact: `+SC-COINBASE_SPOT_BTC_USD+` (else BTC_USDT leaks in).

Cost discipline: coverage via cheap throttled LIST calls; schema via a bounded
8 MB HTTP Range GET of one file (not the multi-GB full day).

S3 access: endpoint https://s3.flatfiles.coinapi.io, region us-east-1,
access-key-id = CoinAPI key, secret = "coinapi", path-style addressing.
Layout: T-<TYPE>/D-<date>/E-<EXCHANGE>/IDDI-..+SC-<symbol_id>+S-..csv.gz
"""
from __future__ import annotations
import os
import sys
import time
import zlib
import datetime as dt
from io import StringIO
from collections import Counter, defaultdict

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from _common import load_env, is_quota_error, QUOTA_HINT  # noqa: E402

ENDPOINT = "https://s3.flatfiles.coinapi.io"
BUCKET = "coinapi"
EXCHANGE = "COINBASE"
SYMBOL_MATCH = "+SC-COINBASE_SPOT_BTC_USD+"     # exact (trailing + excludes BTC_USDT)
SAMPLE_BYTES = 8 * 1024 * 1024                  # 8 MB Range GET for schema sampling
REQ_PER_MIN = 8                                 # stay under the 10/min tier limit


# ----------------------------------------------------------------------------- rate limiting
class RateLimiter:
    def __init__(self, per_min):
        self.interval = 60.0 / per_min
        self._last = 0.0
        self.calls = 0

    def wait(self):
        gap = time.monotonic() - self._last
        if gap < self.interval:
            time.sleep(self.interval - gap)
        self._last = time.monotonic()
        self.calls += 1


RL = RateLimiter(REQ_PER_MIN)


# ----------------------------------------------------------------------------- client
def make_client(api_key: str):
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT,
        region_name="us-east-1",
        aws_access_key_id=api_key,
        aws_secret_access_key="coinapi",
        config=Config(
            signature_version="s3v4",
            s3={"addressing_style": "path"},
            retries={"max_attempts": 5, "mode": "adaptive"},   # also backs off on SlowDown
        ),
    )


def discover_bucket(s3) -> str:
    RL.wait()
    names = [b["Name"] for b in s3.list_buckets().get("Buckets", [])]
    print(f"  buckets visible: {names}")
    return BUCKET if BUCKET in names else (names[0] if names else BUCKET)


# ----------------------------------------------------------------------------- listing
def list_prefix(s3, bucket, prefix, delimiter=None):
    """(objects, common_prefixes), fully paginated, throttled per page."""
    objs, cps = [], []
    kwargs = {"Bucket": bucket, "Prefix": prefix}
    if delimiter:
        kwargs["Delimiter"] = delimiter
    token = None
    while True:
        RL.wait()
        if token:
            kwargs["ContinuationToken"] = token
        page = s3.list_objects_v2(**kwargs)
        objs += page.get("Contents", [])
        cps += [c["Prefix"] for c in page.get("CommonPrefixes", [])]
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return objs, cps


def coinbase_btc_file(s3, bucket, day, data_type="LIMITBOOK_FULL"):
    """Exact COINBASE_SPOT_BTC_USD daily file for a day -> (key, size) or None."""
    objs, _ = list_prefix(s3, bucket, f"T-{data_type}/D-{day}/E-{EXCHANGE}/")
    hits = [o for o in objs if SYMBOL_MATCH in o["Key"]]
    if not hits:
        return None
    o = max(hits, key=lambda x: x["Size"])
    return o["Key"], o["Size"]


# ----------------------------------------------------------------------------- schema sampling
def gunzip_partial(b: bytes) -> str:
    """Decompress a truncated gzip stream (Range GET); stops cleanly at the cut."""
    d = zlib.decompressobj(16 + zlib.MAX_WBITS)
    return d.decompress(b).decode("utf-8", "ignore")


def sample_schema(s3, bucket, key, n_bytes=SAMPLE_BYTES):
    RL.wait()
    resp = s3.get_object(Bucket=bucket, Key=key, Range=f"bytes=0-{n_bytes - 1}")
    raw = resp["Body"].read()
    lines = gunzip_partial(raw).split("\n")
    if len(lines) < 3:
        raise RuntimeError(f"Too little decompressed ({len(lines)} lines) from {len(raw)} B.")
    df = pd.read_csv(StringIO("\n".join(lines[:-1])), sep=";")   # CoinAPI flat files are ';'-delimited
    return df, len(raw)


# ----------------------------------------------------------------------------- report
def hr(t): print("\n" + "=" * 74 + f"\n{t}\n" + "=" * 74)


def run(window_days=14, end_offset_days=25):
    api_key = load_env()["COINAPI_KEY"]
    print(f"Using COINAPI_KEY ...{api_key[-4:]}  | throttle {REQ_PER_MIN} req/min "
          f"(tier limit 10) | window {window_days}d ending today-{end_offset_days}d")
    s3 = make_client(api_key)

    hr("0. AUTH / bucket")
    bucket = discover_bucket(s3)
    print(f"  using bucket: {bucket}")

    # End the scan inside the consolidated (daily) region, past the hourly tail.
    today = dt.date(2026, 6, 22)                          # harness clock (no Date.now in this env)
    end = today - dt.timedelta(days=end_offset_days)

    hr(f"1. COINBASE BTC-USD limitbook_full coverage — {window_days} days ending {end}")
    present, sizes, missing = [], {}, []
    for i in range(window_days):
        ds = (end - dt.timedelta(days=i)).strftime("%Y%m%d")
        res = coinbase_btc_file(s3, bucket, ds, "LIMITBOOK_FULL")
        if res:
            present.append(ds); sizes[ds] = res[1]
        else:
            missing.append(ds)
        print(f"  {ds}: {'—missing—' if not res else f'{res[1]/1e6:7.1f} MB'}")
    cov = 100 * len(present) / window_days
    print(f"\n  coverage: {len(present)}/{window_days} days ({cov:.1f}%)  | missing: {len(missing)}")
    if sizes:
        mb = sorted(v / 1e6 for v in sizes.values())
        avg = sum(mb) / len(mb)
        print(f"  per-day size: min {mb[0]:.0f} | median {mb[len(mb)//2]:.0f} | "
              f"max {mb[-1]:.0f} | avg {avg:.0f} MB")
        print(f"  projection: 12mo ~{avg*365/1000:.1f} GB | 18mo ~{avg*547/1000:.1f} GB "
              f"| 24mo ~{avg*730/1000:.1f} GB (compressed, BTC-USD only)")
    if missing:
        print(f"  MISSING: {missing}")

    if not present:
        print("  No consolidated daily files in window — try a larger --offset.")
        return

    hr("2. limitbook_full SCHEMA + timestamps (8 MB Range GET)")
    sday = max(present)
    key, fsize = coinbase_btc_file(s3, bucket, sday, "LIMITBOOK_FULL")
    print(f"  file: {key.split('/')[-1]}  ({fsize/1e6:.0f} MB; sampling {SAMPLE_BYTES/1e6:.0f} MB)")
    df, got = sample_schema(s3, bucket, key)
    print(f"  sample rows: {len(df)} (from {got/1e6:.1f} MB)")
    print(f"  columns: {list(df.columns)}")
    if "update_type" in df.columns:
        print(f"  update_type: {dict(Counter(df['update_type']))}")
    for tcol in ("time_exchange", "time_coinapi"):
        if tcol in df.columns:
            print(f"  {tcol}: non-null {df[tcol].notna().mean():.3%} | e.g. {df[tcol].iloc[0]}")
    if {"update_type", "is_buy"} <= set(df.columns):
        snap = df[df["update_type"].astype(str).str.upper().str.contains("SNAP")]
        if len(snap):
            nb = snap["is_buy"].isin([1, True, "1", "true"]).sum()
            print(f"  opening snapshot: {nb} bid / {len(snap)-nb} ask levels (full-depth)")
    if "order_id" in df.columns:
        print(f"  order_id present -> L3 (order-by-order). distinct in sample: {df['order_id'].nunique()}")

    hr("VERDICT (Flat Files)")
    print(f"  Coinbase BTC-USD full-depth book: coverage {cov:.0f}% over {window_days}d, "
          f"{len(missing)} gaps | double-timestamped: "
          f"{ {'time_exchange','time_coinapi'} <= set(df.columns) } | "
          f"incremental: {'update_type' in df.columns}")
    print(f"  vs Crypto Lake's ~80% Coinbase. Total S3 requests used: {RL.calls}")


if __name__ == "__main__":
    win = int(sys.argv[1]) if len(sys.argv) > 1 else 14
    off = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    try:
        run(window_days=win, end_offset_days=off)
    except ClientError as e:
        if is_quota_error(str(e)):
            print(QUOTA_HINT); sys.exit(2)
        raise
