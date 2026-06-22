# Data — Sources, Coverage & Methodology

**Status:** access live and verified 2026-06-22. Covers the `ingest/` layer (spec §3, §4, §12.1).
Numbers here are **measured**, not vendor-quoted, unless marked *(vendor)*.

Companion docs: [`jepa_btc_forecasting_spec.md`](../jepa_btc_forecasting_spec.md) §4–6,
[`ingest/README.md`](../ingest/README.md) (how to run the scripts).

---

## 1. Scope & instruments

We forecast short-horizon **Coinbase BTC-USD** mid moves using **Binance** as the primary
information source (spec §1). Data needed:

| Role | Venue / instrument | Vendor | Feeds |
|---|---|---|---|
| Signal (primary) | Binance **BTC-USDT-PERP** (futures) | Crypto Lake | `book_delta_v2`, `trades`, `funding`, `open_interest`, `liquidations` |
| Signal (secondary) | Binance **BTC-USDT** (spot) | Crypto Lake | `book_delta_v2`, `trades` |
| Target / label venue | Coinbase **BTC-USD** (spot) | Crypto Lake + CoinAPI (hybrid) | `book_delta_v2`, `trades` |

History span: **12–24 months** for SSL pretrain; recent 3–6 mo for head finetune; clean recent
~1 mo held out (spec §4). Planning figures below use **18 months** (547 days) unless noted.

---

## 2. Sources

### 2.1 Crypto Lake — Binance (and most of Coinbase)
- **Access:** S3 via `lakeapi` 0.22.3. Subscriber IAM keys in `.env`
  (`aws_access_key_id` / `aws_secret_access_key` / `region`), STS arn
  `…:user/subscribers/stripe/<email>`. Pass them through an **explicit `boto3.Session`**
  (`ingest/verify_lake.py:lake_session()`) — do **not** rely on the default chain, since
  `~/.aws` holds a different (personal) account that gets `AccessDenied`.
- **Login quirk:** `lakeapi._login` returns `unknown/s3` (subscribers can't invoke the routing
  lambda). That's the normal fallback — **data reads go direct-S3 and work**.
- **Cache caveat:** clear `.lake_cache/` if keys ever change; stale botocache from a bad-key run
  makes `load_data` silently return 0 rows.
- **Bucket:** `qnt.data`, prefix `market-data/cryptofeed/{table}/exchange={E}/symbol={S}/dt={date}/`.
  Not requester-pays (reads succeed without `RequestPayer`), so **we pay no S3 egress** — Crypto
  Lake bears it. Capture is in **AWS Tokyo** *(vendor; consistent with the 4.4 ms Binance feed lag
  measured below)*.
- **Cost:** flat ~$64/mo individual plan, effectively unlimited download.

### 2.2 CoinAPI — Coinbase gap backfill (and L3 option)
- **Two separate products / credit pools:**
  - **Flat Files** (S3-compatible bulk): endpoint `https://s3.flatfiles.coinapi.io`, region
    `us-east-1`, access-key-id = CoinAPI key, secret = literal `"coinapi"`, **path-style**.
    Buckets `coinapi` (history) + `coinapi-daily-tail`. Billed **per GB downloaded**:
    **$1.00/GB** limit-book & quotes, **$3.00/GB** trades.
  - **Market Data REST/WS** (`rest.coinapi.io`, header `X-CoinAPI-Key`): where the **$25 free
    credit** lives; metered 100 data points = 1 credit. Order book here is top-N L2 **snapshots**
    (≤100 levels), not incremental — validation only, not production.
- **Rate limit (free tier): 10 requests/min** — all clients throttle to 8/min.
- **Layout:** `T-{TYPE}/D-{date}/E-COINBASE/…+SC-COINBASE_SPOT_BTC_USD+…csv.gz`.
  Recent ~3 weeks are an **hourly tail** (`D-YYYYMMDDHH`) / in `coinapi-daily-tail`; consolidated
  history is daily (`D-YYYYMMDD`). We target consolidated daily.

---

## 3. Vendor decision & rationale

**Binance → Crypto Lake.** Structurally equivalent to Tardis, far cheaper, Tokyo-captured. Coverage
and timestamps verified excellent (§5).

**Coinbase → hybrid (Crypto Lake + CoinAPI backfill).** Crypto Lake's Coinbase is 92.9% over 2 yr
with one **33-day hole (2024-12-05 → 2025-01-06)** — the "large gaps" the spec warned about. Rather
than buy the full Coinbase span from CoinAPI L3 (~$1,240/18 mo), we take Coinbase from Crypto Lake
(flat-rate, ~93%) and backfill only the gaps from CoinAPI (~$82 for the 33-day hole). Trade-off: the
label venue is stitched from two vendors — recon must align on `origin_time` (both populate it; §5).

> Single-vendor alternatives if contiguity is preferred: **all-CoinAPI L3** (~$1,240, contiguous,
> full L3) or **all-Crypto-Lake** (free, but accept the 33-day hole via purge/embargo).

CoinAPI has **no mid-tier L2 product** — only L3 `limitbook_full` (2.27 GB/day) or L1 `quotes`
(74 MB/day). Crypto Lake's `book_delta_v2` *is* incremental L2 with `sequence_number`, which is what
§6 actually needs, so the hybrid keeps storage small and only pays CoinAPI's L3 premium on gap days.

---

## 4. Schemas

### 4.1 Crypto Lake `book_delta_v2` (incremental L2)
Raw parquet columns: `timestamp`, `receipt_timestamp`, `sequence_number`, `side`, `price`, `size`
(+ partition `exchange`/`symbol`/`dt`). `lakeapi` renames `timestamp→origin_time`,
`receipt_timestamp→received_time` (datetime64[ns]) on load. **To column-project in `load_data`,
pass the RAW names.** `book` (snapshot) variant is 2×20 levels (85 cols). `trades` has
`origin_time`, `received_time`, `price`, `quantity`, `side`, `id`.

### 4.2 CoinAPI Flat Files `limitbook_full` (L3, order-by-order)
**Semicolon-delimited** CSV.gz: `time_exchange;time_coinapi;update_type;is_buy;entry_px;entry_sx;order_id`.
- `update_type ∈ {SNAPSHOT, ADD, DELETE, MATCH, SET, SUB}` — open set; **MATCH = trades against the
  book**. Store as string, never an enum.
- `order_id` 100% populated → true L3. Opening `SNAPSHOT` ≈ 62 k bid + 51 k ask levels = full book.
- **Timestamps are time-only** `HH:MM:SS.fffffff` (no date) — date is implicit from the partition;
  the SNAPSHOT block carries the prior-day close time (23:59:59.999) as the opening-book stamp.

### 4.3 Our downloader output (CoinAPI → Parquet), `download_coinapi.py`
Hive layout `data/raw/limitbook_full/exchange=/symbol=/dt=/data.parquet`. **Lossless, no date
assumption** (that's recon's job):

| col | type | meaning |
|---|---|---|
| `seq` | int64 | row order in the file = canonical event order (there is no `sequence_number`) |
| `time_exchange_ns` | int64 | ns since midnight UTC |
| `time_coinapi_ns` | int64 | ns since midnight UTC (receive time) |
| `update_type` | string | SNAPSHOT/ADD/DELETE/MATCH/SET/SUB |
| `is_buy` | bool | |
| `entry_px`, `entry_sx` | float64 | price, size |
| `order_id` | string | UUID (L3) |

ZSTD compression, dictionary on `update_type`. Recon combines partition `dt` + `*_ns`.

---

## 5. Coverage & timestamp quality (measured 2026-06-22)

**2-year window (2024-06-22 → 2026-06-22), `book_delta_v2`:**

| Feed | Coverage | Gap structure |
|---|---|---|
| Binance fut `book_delta_v2` | **96.4%** (704/730) | gaps only recent (May–Jun '26), max 6 d |
| Binance fut `trades`, `funding` | ~100% | — |
| Binance fut `open_interest` | 119/120 (recent) | 1 d |
| Binance fut `liquidations` | ~66% | sparse, event-driven (expected) |
| Coinbase `book_delta_v2`/`book`/`trades` | **92.9%** (678/730) | **33-day hole 2024-12-05→2025-01-06** + a 6-day + singletons |

**`origin_time` (the load-bearing §4 #1 check) — 100% populated for `book_delta_v2`, both venues**
(0.0000% empty), contradicting the docs' generic "order book often lacks origin_time" warning. So
event-time reconstruction (§5.3) runs on **exchange time** directly. Feed lag (received − origin):

| Venue | median | p95 | note |
|---|---|---|---|
| Binance perp | **4.4 ms** | 149 ms | Tokyo co-location confirmed |
| Coinbase | ~90 ms | 137 ms | cross-region capture |

CoinAPI Coinbase (REST `/v1/symbols`): trades from 2015-01-14, order book from 2015-05-17, both
through T+1; double-timestamped (`time_exchange`+`time_coinapi`), `taker_side` present.

---

## 5a. Vendor stitching (Coinbase gap-fill) — validated 2026-06-22

The hybrid plan needs CoinAPI to fill Crypto Lake's Coinbase holes seamlessly. Both conditions hold:

- **Coverage:** CoinAPI has **12/12 sampled gap days** (entire 33-day hole + the Nov gap + singletons),
  each with substantial `limitbook_full` (0.9–3.5 GB) and `quotes`. CoinAPI can fill every gap.
- **Agreement (clean overlap day 2025-06-01, exchange-time 1 s grid):** Crypto Lake `book` vs CoinAPI
  `quotes` mid — **median |Δmid| = $0.000 (0.000 bps)**, correlation **0.999982**, 89% of seconds
  within $1, 96% within $5. No systematic offset or unit mismatch → **the stitch is seamless.**
  (Rare transient spikes to ~$249 at isolated seconds = momentary one-sided/stale top-of-book during
  fast moves; they wash out in bars — QC at the recon layer.)

**Two caveats surfaced while validating:**
1. **`book_delta_v2` needs continuous reconstruction.** The daily file has **no snapshot seed** and uses
   absolute-size / `0`=remove updates. You **cannot** reconstruct one day in isolation — recon must carry
   book state across day boundaries from a cold-start snapshot. (A naive per-day replay-from-empty looks
   ~99% crossed; that's the method, not the data.)
2. **The derived `book` (20-level snapshot) product is intermittently crossed on some days**
   (e.g. 2026-04-01: 31.75% crossed, spreads to −$1188; 2025-06-01: 0%). We don't use this product in
   production (we reconstruct from `book_delta_v2`), so it likely doesn't affect us — **confirm once recon
   exists** by checking the reconstructed book is uncrossed on a 2026-04-01-type day. If `book_delta_v2`
   itself proves degraded on some present days, those days also get CoinAPI fill (treat like gaps).

---

## 6. Per-day sizes & storage budget (measured)

Crypto Lake parquet, BTC, 2026-04-01:

| Feed | MB/day | rows/day |
|---|---|---|
| Binance perp `book_delta_v2` | 573.8 | 109.3 M |
| Binance perp `trades`/`funding`/`OI`/`liq` | ~33 | — |
| Binance spot `book_delta_v2` + `trades` | 261 | — |
| Coinbase `book_delta_v2` + `trades` | 303 | 34.7 M (book) |
| **All feeds** | **~1.17 GB/day** | |

CoinAPI Coinbase (csv.gz): `limitbook_full` (L3) **2266 MB/day**, `quotes` 74, `trades` 68. Note the
L3 csv.gz is ~8× Crypto Lake's L2 parquet — another reason the hybrid keeps Coinbase on Crypto Lake.

| Span | Crypto Lake (all feeds) | + CoinAPI 33-day backfill | **Total raw** |
|---|---|---|---|
| 12 mo | 427 GB | ~75 GB | **~0.50 TB** |
| 18 mo | 643 GB | ~75 GB | **~0.72 TB** |
| 24 mo | 858 GB | ~75 GB | **~0.93 TB** |

After `recon` + bar building, the **training set collapses to GB-scale** (§7).

---

## 7. Compute, storage & RAM plan

**Target box:** local, RTX 3070 (8 GB), **32 GB RAM**, **2 TB SATA SSD**. Decision: **local-first**
(spec §11). No AWS egress is charged to us (Crypto Lake not requester-pays; CoinAPI on Cloudflare R2 =
no egress), and the model is small — so cloud buys only optional *speed* on the one-time recon pass
(a single same-region Tokyo spot instance), never a persistent cluster.

- **Disk (2 TB):** comfortable — ~0.7–0.9 TB raw + ≤60 GB processed leaves ~1 TB free. Keep raw
  compressed; never decompress `book_delta_v2` to disk.
- **RAM (32 GB) is the binding constraint** and drives bar design. The SSL feature matrix =
  bars × features/bar × dtype:
  - bars (§5.1, ~1 bar/0.5–2 s) over 18 mo ≈ 20–47 M; features/bar (§6, ~3 venues) ≈ 150–250.
  - **Defaults to stay RAM-resident (~10–15 GB):** store features **fp16** and tune the dollar-bar
    threshold to **~20–30 M bars**. fp32 + fine bars (→ 30–60 GB) would not fit; fall back to
    **mmap from the SSD** (OK for a small, GPU-bound model) only if a config pushes past RAM.
- **Recon must stream per day** — one day of Binance perp `book_delta_v2` is 109 M rows (~4 GB if
  loaded whole); process events sequentially (Rust, parallel per (day, instrument)).
- **Training** is GPU-bound: data loaded once into RAM, SATA never in the hot path.

---

## 8. Cost summary

| Item | Cost |
|---|---|
| Crypto Lake subscription | ~$64/mo (flat, unlimited) |
| CoinAPI 33-day Coinbase backfill (L3 book + trades) | ~$82 (−$25 free credit ≈ $57 out of pocket) |
| CoinAPI all historical Coinbase gaps (~46 d) | ~$113 |
| Compute | $0 (local); optional one-off Tokyo spot for recon |
| Storage | $0 (existing 2 TB SSD) |

---

## 9. Reproducing the verification

```bash
.venv/bin/python ingest/verify_lake.py      # Lake auth + table existence/coverage (metadata only)
.venv/bin/python ingest/verify_lake2.py     # 2-yr gap structure + origin_time (downloads ~2 GB)
.venv/bin/python ingest/coinapi_rest.py     # CoinAPI coverage dates via the $25 REST credit
.venv/bin/python ingest/coinapi_flatfiles.py 14   # CoinAPI flat-files coverage + 8 MB schema sample
.venv/bin/python ingest/download_coinapi.py --start 2025-01-01 --end 2025-01-31  # CoinAPI → Parquet
```
Scripts are throttled, resumable, and exit cleanly on quota/billing gates. Secrets live in `.env`
(git-ignored).

---

## 10. Open items / verification TODOs

- [x] **Cross-vendor Coinbase stitching** — VALIDATED (§5a): CoinAPI covers all sampled gap days;
      vendor mids agree to $0.000 median / 0.999982 corr on a clean overlap day.
- [x] **Crypto Lake bucket region** — `eu-west-1` (confirmed via pyarrow S3 read; head_bucket 403s).
- [ ] **`book_delta_v2` continuous reconstruction** — recon must seed from a snapshot and carry book
      state across days (no per-day seed; absolute-size/0=remove). Then verify the reconstructed book is
      uncrossed, incl. on a day where the `book` snapshot product is crossed (e.g. 2026-04-01).
- [ ] **Crypto Lake Coinbase quality map** — scan how many *present* days have a degraded `book_delta_v2`
      reconstruction (not just the `book` snapshot product); those get CoinAPI fill like gaps.
- [ ] **Within-timestamp ordering for CoinAPI** (no `sequence_number`; rely on `seq` row order + L3 `order_id`).
- [ ] **Binance downloader** — not yet built; same throttled/resumable/partitioned pattern as
      `download_coinapi.py`, streaming per day (109 M rows). Read direct via pyarrow S3 (`eu-west-1`)
      or lakeapi.
- [ ] **Liquidations sparsity** — confirm low coverage is genuine (no liquidations) vs missing files.

---

*Last verified 2026-06-22. All coverage/size/timestamp figures measured against live vendor data on
that date; re-run §9 to refresh.*
