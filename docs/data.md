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

History span: **12–24 months** for SSL pretrain; recent 3–6 mo for head finetune; clean held-out OOS
~1 mo (spec §4). **The OOS month must be chosen from the usable all-feed calendar (§5b), not simply
"most recent"** — recent May–June 2026 has Binance gaps; the most-recent usable run ends 2026-05-05
(OOS ≈ April 2026). Planning figures below use **18 months** (547 days) unless noted.

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
- **One shared credit balance, two access products** (the $25 trial works across both; it unlocks only
  after a **verified payment method** is added):
  - **Flat Files** (S3-compatible bulk): endpoint `https://s3.flatfiles.coinapi.io`, region
    `us-east-1`, access-key-id = CoinAPI key, secret = literal `"coinapi"`, **path-style**.
    Buckets `coinapi` (history) + `coinapi-daily-tail`. Billed **per GB downloaded**:
    **$1.00/GB** limit-book & quotes, **$3.00/GB** trades *(flat-files pricing page)*. You also pay for
    **requests** (LIST/GET ops) — small, but our coverage scans are LIST-heavy, so cache and reuse.
    *(Open question: whether bulk flat-files downloads get the tiered GB discount that WebSocket "Tier-1
    data" gets — $1/GB→$0.10/GB above 512 GB — which would cut the full L3 pull from ~$1k toward
    ~$200. Confirm against measured spend on the first backfill day; assume flat $1/GB until then.)*
  - **Market Data REST/WS** (`rest.coinapi.io`, header `X-CoinAPI-Key`): metered **100 data points = 1
    credit** (date-bounded queries capped at 10 credits; no-`limit` call = 1 credit). REST credit price
    ≈ $5.26/1,000 for the first 1,000/day, cheaper at volume. Order book here is top-N L2 **snapshots**
    (≤100 levels), not incremental — validation only, not production.
- **Limits (spend-tiered):** global **max RPM / concurrency** rise with cumulative spend —
  T0 10/1 (cliff $8) · **T1 40/2 (current, cliff $32)** · T2 160/4 ($128) · T3 640/8 ($512) · T4 none.
  Exceeding RPM → `SlowDown`/429. Clients throttle via `COINAPI_RPM` (default **32**, safe under T1's 40;
  bump when the tier rises). Rate-limit headers are **not reliable**; monitor via **Customer Portal →
  Usage Explorer / Traces** (free). Credits **never expire**; top up $5–$5000 (PAYG).
- **⚠️ Before any bulk download, enable Spend Management** (Billing → Spend Management, **OFF by
  default**): set a daily credit cap + hard-stop + alerts (50/80/95%). This is the guardrail against a
  runaway backfill/full-pull bill. See §8.
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
Raw parquet columns: `timestamp`, `receipt_timestamp`, `sequence_number`, **`side_is_bid` (bool)**,
`price`, `size` (+ partition `exchange`/`symbol`/`dt`). `lakeapi` renames `timestamp→origin_time`,
`receipt_timestamp→received_time` (datetime64[ns]) on load. **To column-project in `load_data`, pass
the RAW names.** Update semantics: **`size` is the absolute size at that price; `size==0` removes the
level.** There is **no per-day snapshot block** — the daily file starts mid-stream (see §5a-Recon).
`book` (snapshot) variant is 2×20 levels (85 cols) — *not used in production* (see §5a). `trades` has
`origin_time`, `received_time`, `price`, `quantity`, `side`∈{buy,sell}, `trade_id` (int64).

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

ZSTD compression, dictionary on `update_type`.

**Reconstruction ordering rule (mandatory).** Replay strictly in **`seq` order** (file/row order), *not*
by a reconstructed timestamp. The opening `SNAPSHOT` block (lowest `seq`) is the **initial book state for
the partition day and is applied before all non-snapshot events**, even though its `time_exchange`
carries the *prior-day close* time (e.g. 23:59:59.999). Do **not** build a wall-clock timestamp as
`dt + time_exchange_ns` for ordering — that would sort the opening snapshot to the end of the day.
Use `dt + time_exchange_ns` only as a *display/label* time for non-snapshot events (and clamp the
snapshot block to the partition-day open). `seq` is the canonical order; there is no `sequence_number`.

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

## 5b. Trade-feed validation & usable all-feed calendar

**Trades drive the bar clock (§5.1), so validated directly** (Crypto Lake, 2025-06-01):

| Venue | rows | origin/recv empty | recv−origin lag (median/p95) | `side` | `trade_id` | file order |
|---|---|---|---|---|---|---|
| Binance perp | 812,701 | 0% / 0% | 57 ms / 200 ms | buy/sell | int64, **unique**, monotonic | **sorted by origin_time** |
| Binance spot | 645,930 | 0% / 0% | 5 ms / 63 ms | buy/sell | int64, **unique**, monotonic | **sorted by origin_time** |
| Coinbase | 274,489 | 0% / 0% | 164 ms / 238 ms | buy/sell | int64, **unique**, *not* monotonic | **NOT sorted by origin_time** |

- `side` = taker/aggressor side (drives CVD / aggressor imbalance, §6). Lag is always ≥0 (0% negative).
- **Coinbase trades are not stored in `origin_time` order and `trade_id` is not monotonic** — the clock
  **must sort Coinbase trades by `origin_time`** (Binance feeds are already ordered). One day, one
  symbol each — extend to multi-day before relying on it.

**Usable all-feed calendar (730 d to 2026-06-22)** — OOS/usable spans must be the *intersection* of all
required feeds after gaps. Binance is the binding constraint (no backfill vendor); Coinbase gaps are
CoinAPI-fillable:

| feed | days | | feed | days |
|---|---|---|---|---|
| Binance perp `book_delta_v2` | 704/730 | | Coinbase `book_delta_v2` | 678/730 |
| Binance perp `trades` | 729/730 | | Coinbase `trades` | 674/730 |
| Binance spot `book`/`trades` | 730/730 | | `funding` / `open_interest` | 730 / 729 |

- **Binance-side intersection = 704/730**; **usable with Coinbase backfill = 704/730 (96.4%)**;
  Lake-only all-feed intersection (no backfill) = 652/730.
- **52 Coinbase days need CoinAPI fill** within the usable set, split by product: **47 need book**
  (84.6 GB L3) and **all 52 need trades** (2.6 GB) — both **verified present in CoinAPI flat files
  (0 unfillable, 0 probe-error → `backfill_verified=True`)**. So `usable_after_verified_backfill =
  704/730 (96.4%)` is *measured*, not assumed. The full artifact — `usable_days` (704), `lake_all_days`
  (652), `excluded_days_by_reason` (26, e.g. `missing:binF_book`), the fill-day book/trades status, and
  OOS runs — is written to **`data/usable_calendar.json`** (auditable without re-listing vendors).
- **OOS month must come from the usable calendar, not "most recent."** Most-recent contiguous usable run
  ≥21 d = **2026-02-06 → 2026-05-05**; the prior run is 2024-06-22 → 2026-02-04 (split by 1-day Binance
  gaps). **Recent May–June 2026 is NOT usable** (Binance `book_delta_v2` gaps). Pick OOS ≈ **April 2026**.

Reproduce: `ingest/verify_trades_and_calendar.py --verify-backfill` (anchor via `--end`/`END`); writes
`data/usable_calendar.json`.

---

## 5a. Vendor stitching (Coinbase gap-fill) — unit/timestamp sanity PASSED; recon parity PENDING

**Status: NOT production-validated.** The hybrid is promising but two hard gates remain (recon-level
parity, snapshot/day-boundary semantics). What we have shown so far:

- **Coverage (done):** CoinAPI has **12/12 sampled Crypto Lake gap days** (entire 33-day hole + the Nov
  gap + singletons), each with substantial `limitbook_full` (0.9–3.5 GB) and `quotes`. CoinAPI *can*
  supply every gap day.
- **Unit/timestamp sanity (done, but a weaker test than production):** on a clean overlap day
  (2025-06-01, exchange-time 1 s grid), Crypto Lake's **derived `book`** vs CoinAPI **`quotes` (L1)** mid
  agree — median |Δmid| = $0.000, correlation 0.999982, 89% of seconds within $1. This proves
  **prices, units, and timestamp conventions line up** with no systematic offset. It does **not** prove
  production parity: production uses Crypto Lake **reconstructed `book_delta_v2`** vs CoinAPI **L3
  aggregated to L2** — neither of which was exercised here. And the rare ~$249 second-scale spikes
  **cannot be assumed to "wash out"** for second-scale labels; they must be characterized, not dismissed.

**Hard gates before treating the hybrid as production-validated:**
1. **Recon-level parity:** reconstruct Crypto Lake `book_delta_v2` → top-K L2 and CoinAPI `limitbook_full`
   (L3) → top-K L2 for the **same overlap day**, and compare per-level price/size and the resulting
   labels (not just L1 mid). Quantify the spike population at the exact bar/label horizons.
2. **Snapshot/day-boundary semantics:** apply the §4.3 / §5a-Recon ordering rules and confirm the
   reconstructed book is uncrossed across the day boundary.

**Tooling status (parity gate) — added, live run PENDING.** The one-day parity gate is now
implemented and **synthetic-unit-validated** (no measured vendor results yet): `recon/coinapi.py`
replays CoinAPI `limitbook_full` L3 → top-K L2 (seq-order, snapshot-first day-open clamp,
defensive `SNAPSHOT/ADD/DELETE/MATCH/SET/SUB` with `order_id` state and quality counters);
`recon/reconstruct.py::reconstruct_lake_l2_at_samples` reconstructs Lake `book_delta_v2` → top-K
L2 on the same exchange-time grid (memory-safe, no per-row object list); `recon/parity.py`
compares per-level price/size, mid, crossed/missing rates, the |Δmid| spike population, and
directional label agreement at the 2 s/10 s/60 s horizons; `scripts/run_coinbase_parity.py` wires
it on real data. Because `book_delta_v2` cold-starts with no per-day snapshot (§5a-Recon), the gate
applies a **seed-established warm-up cutoff** (best bid/ask present, uncrossed, sustained) and
**excludes the Lake warm-up window** from the comparison so warm-up artifacts don't drive the
decision (`--no-warmup-gate` to disable); it also reports **per-level both-present coverage** so
thin/one-sided top-K depths are marked, not silently dropped. The full validated seed from Lake's
`book` snapshot product stays the deferred §5a-Recon follow-up. The CoinAPI **SUB/MATCH size
convention** was an A/B assumption (absolute-size vs `--size-policy decrement`); the live run below
**resolved the MATCH path: `decrement` is correct for Coinbase `limitbook_full` MATCH events**.
⚠️ **SUB is NOT yet verified** — 2025-06-01 had **0 SUB events**, so the `decrement` default also
applies to SUB by family analogy only; a future day with partial-fill `SUB` rows must confirm it
(see "Measured results"). Run (after enabling CoinAPI Spend Management, §8):

```bash
.venv/bin/python ingest/download_coinapi.py --start 2025-06-01 --end 2025-06-01            # one overlap day
.venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --k 10 --size-policy decrement   # -> data/reports/
```

**Measured results — first live run, 2025-06-01 (2026-06-29). Gate NOT yet passed (Lake side).**
Pulled the full CoinAPI day (26.3M L3 events, 800MB→588MB parquet) and loaded the live Crypto Lake
Coinbase `book_delta_v2` day (16.5M delta rows). Two reconstruction issues surfaced — **neither is
true vendor disagreement**:

1. **CoinAPI `MATCH` size convention = `decrement` (RESOLVED).** A `MATCH` event's `entry_sx` is the
   *traded* quantity (amount removed), not the resting remainder — confirmed by tracing order
   histories (e.g. ADD `sx=9.64e-06` then MATCH `sx=9.64e-06` ⇒ fully filled). Under the old default
   `size_policy="absolute"`, MATCH re-set the order to the traded size and left stale residue at the
   touch → CoinAPI book **crossed 99.99%** of samples (−$708 deep by mid-morning, ~3.8k stale ask
   levels under the best bid). Under `--size-policy decrement` the CoinAPI book is **0.00% crossed**,
   clean (spread +$0.01, no stale levels). DELETEs (cancels, 12.9M/day) were always fine; only the
   MATCH path (275k fills, all at top-of-book) was affected. ⇒ **`decrement` is the Coinbase default**
   (the `absolute` path stays available for other venues / A/B). **Scope of the evidence:** this day
   had **0 `SUB` events** (event mix: SNAPSHOT 102,694 · ADD 13.0M · DELETE 12.9M · MATCH 275,247 ·
   SET 8,091), so only the MATCH size convention is *verified*. `decrement` decrements SUB under the
   same policy by analogy; that must be re-checked on a day that actually contains `SUB` rows before
   trusting partial-fill reconstruction there.

2. **Lake `book_delta_v2` crosses 67% — intraday level-stranding, NOT cold-start (the blocker).**
   The seed-established warm-up gate excluded only 2 pre-seed samples, yet the Lake book is crossed
   67% of the day. By hour: h00 7% (genuine warm-up), several whole hours **0% (clean)**, but
   h01–h13 / h16–h17 run **80–100% crossed with median spread −$60 to −$695** (mean −$306). Signature:
   a price level is stranded (its `size=0` clearing update never lands), best bid/ask freeze and
   cross, then recover when a later delta hits that exact price. A single day-open seed would **not**
   fix this — it needs the gap-aware seed/reseed policy in §5a-Recon. (Caveat for that work:
   `sequence_number` is **per-event, not per-row** — ~91% of consecutive rows duplicate it, max 6.2M
   vs 16.5M rows — and Coinbase's channel sequence also counts trades, so naive `seq`-diff ≠ dropped
   book data; the exact increment semantics must be confirmed there. Note the Lake `book` snapshot is
   **0% crossed on 2025-06-01**, so it is a valid seed candidate for this day.)

**Parity after the CoinAPI fix, with Lake still crossing (the residual gap is entirely the Lake side):**
`|Δmid|` median **$55.96** / p95 $345 / corr **0.977**; directional label agreement **0.90 / 0.93 /
0.95** at 2 s / 10 s / 60 s; per-level both-present coverage 100% to L9. Decision: **do not backfill**;
the gate cannot pass until the Lake `book_delta_v2` reseed policy (§5a-Recon) lands. This decision is
**enforced in code**: `ingest/download_coinapi.py` refuses a multi-day full pull (exit 4) until the
gate passes — single-day parity pulls and `--sample-mb` smoke tests are allowed, `--allow-backfill`
overrides. Report artifacts: `data/reports/parity_coinbase_2025-06-01_k10*.{json,csv}` (the on-disk
JSON is the `decrement` run).

### 5a-Recon. `book_delta_v2` reconstruction & reseed policy
`book_delta_v2` is a **mid-stream incremental feed** (no per-day snapshot, absolute-size/`0`=remove), so
recon cannot naively carry state across *every* boundary — `book_delta_v2` has gaps and Coinbase has
large vendor-filled holes. Required policy:
- **Initial seed:** cold-start from a known full state. Candidate seed = the first dense multi-level
  cluster after a gap, or Crypto Lake's `book` snapshot **only on days verified uncrossed** (the `book`
  product is intermittently crossed — see below — so validate the seed before trusting it).
- **Gap/reseed detection:** reseed whenever (a) a partition day is missing, (b) `sequence_number`
  discontinuity within a day, (c) a vendor switch (Lake↔CoinAPI) at a fill boundary, or (d) the
  reconstructed book goes/stays crossed beyond a tolerance. Never carry state *through* a gap.
- **Seed-quality gate:** after seeding, require N consecutive uncrossed, plausibly-deep snapshots before
  emitting bars; otherwise reseed from the next candidate.
- **Vendor-switch seams:** at each Lake→CoinAPI(fill)→Lake transition, reseed from the incoming vendor's
  first full state; do not assume continuity across the seam.

**Why the seed must be validated:** Crypto Lake's derived `book` (20-level snapshot) product is
**intermittently crossed on some days** (2026-04-01: 31.75% crossed, spreads to −$1188; 2025-06-01: 0%).
We don't use that product for features, but if it's used as a reseed source it must be checked first.
Whether the underlying `book_delta_v2` *reconstruction* is also degraded on such days is **unknown until
recon exists** — that feeds the quality-map TODO (§10); degraded present-days get CoinAPI fill like gaps.

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

| Span | Crypto Lake (all feeds) | + CoinAPI backfill (~52 fill days, L3) | **Total raw** |
|---|---|---|---|
| 12 mo | 427 GB | ~120 GB | **~0.55 TB** |
| 18 mo | 643 GB | ~120 GB | **~0.76 TB** |
| 24 mo | 858 GB | ~120 GB | **~0.98 TB** |

(Backfilled L3 is transient — it aggregates to top-K L2 at ~10 GB after recon. Fill-day count scales
with the chosen span; ~52 is the full 2-yr usable-calendar set, §5b.)

After `recon` + bar building, the **training set collapses to GB-scale** (§7).

---

## 7. Compute, storage & RAM plan

**Target box:** local, RTX 3070 (8 GB), **32 GB RAM**, **2 TB SATA SSD**. Decision: **local-first**
(spec §11). No AWS egress is charged to us (Crypto Lake not requester-pays; CoinAPI on Cloudflare R2 =
no egress), and the model is small — so cloud buys only optional *speed* on the one-time recon pass
(a single same-region **eu-west-1** spot instance — that's where the Lake bucket lives, even though
capture is in Tokyo), never a persistent cluster.

- **Disk (2 TB):** comfortable — ~0.7–0.9 TB raw + ≤60 GB processed leaves ~1 TB free. Keep raw
  compressed; never decompress `book_delta_v2` to disk.
- **RAM (32 GB) is the binding constraint** and drives bar design. The SSL feature matrix =
  bars × features/bar × dtype:
  - bars (§5.1, ~1 bar/0.5–2 s) over 18 mo ≈ 20–47 M; features/bar (§6, ~3 venues) ≈ 150–250.
  - **Canonical processed features are `float32`** (or typed scaled int columns), **not fp16.** The
    milestone-0 baseline is **LightGBM** and labels are **bps-level returns** (§10) — fp16's ~3-decimal
    mantissa corrupts both. `fp16` is allowed **only as a GPU tensor export format** for the JEPA
    encoder, derived from the float32 canonical store.
  - At float32, ~25 M bars × ~200 feat ≈ **~20 GB** — fits 32 GB but with little headroom. Keep it
    resident by tuning the dollar-bar threshold toward **~20–25 M bars** and trimming feature count
    (e.g. K=10). If a config pushes past ~24 GB, **mmap the float32 store from the SSD** (fine for a
    small, GPU-bound model) rather than dropping to fp16. Targets (returns/triple-barrier) stay
    `float32`/`float64` always.
- **Recon must stream per day** — one day of Binance perp `book_delta_v2` is 109 M rows (~4 GB if
  loaded whole); process events sequentially (Rust, parallel per (day, instrument)).
- **Training** is GPU-bound: data loaded once into RAM, SATA never in the hot path.

---

## 8. Cost summary

| Item | Cost |
|---|---|
| Crypto Lake subscription | ~$64/mo (flat, unlimited) |
| CoinAPI Coinbase backfill (§5b, all verified), book 47 d / trades 52 d | **~$92** (book 84.6 GB≈$85 + trades 2.6 GB≈$8) at $1/$3 per GB; −$25 ≈ **$67 OOP** |
| ↳ if only the single 33-day hole is filled | ~$82 |
| Compute | $0 (local); optional one-off same-region (**eu-west-1**) spot for recon |
| Storage | $0 (existing 2 TB SSD) |

- Backfill GB is **measured** (split by product, summed from `data/usable_calendar.json`): 47 book days
  = **84.6 GB** L3 (~$85), 52 trade days = **2.6 GB** (~$8 at $3/GB) → **~$92**, mostly the Dec'24 cluster.
  If the WebSocket "Tier-1" tiered GB discount applies to flat files (unconfirmed), it'd be **less**.
- **⚠️ Enable CoinAPI Spend Management (daily cap + hard-stop) before running the backfill or any full
  L3 pull** — it's OFF by default (§2.2). Set a daily budget, watch Billing → Overview and Usage
  Explorer. Credits never expire, so top up only what a bounded run needs.
- The **full 18-mo CoinAPI-L3 alternative** (if abandoning the hybrid) is ~1 TB → ~$1k at flat $1/GB,
  or possibly ~$200 if the tiered discount applies — confirm before committing.

---

## 9. Reproducing the verification

```bash
# Coverage/calendar scripts anchor on $END (default 2026-06-22 = the snapshot below).
# Set END=YYYY-MM-DD to refresh against a later "today".
END=2026-06-22 .venv/bin/python ingest/verify_lake.py              # auth + table coverage (metadata)
END=2026-06-22 .venv/bin/python ingest/verify_lake2.py            # 2-yr gap structure + origin_time (~2 GB)
.venv/bin/python ingest/verify_trades_and_calendar.py --end 2026-06-22 --verify-backfill  # trades + usable calendar + CoinAPI fill check -> data/usable_calendar.json
.venv/bin/python ingest/coinapi_rest.py                          # CoinAPI coverage dates (REST $25 credit)
.venv/bin/python ingest/coinapi_flatfiles.py 14                  # CoinAPI flat-files coverage + 8 MB schema
.venv/bin/python ingest/download_coinapi.py --start 2025-06-01 --end 2025-06-01   # CoinAPI → Parquet (ONE day)
# NOTE: multi-day BULK pulls are the backfill and are GATED — download_coinapi.py refuses a >1-day full
# pull (exit 4) until the §5a parity + reseed gates pass. Single days + --sample-mb smoke always allowed;
# --allow-backfill overrides once the gate passes (with CoinAPI Spend Management on, §8).
```
Coverage windows are **anchored on the `END` env var** (default `2026-06-22`); without it they reproduce
the original snapshot. (The `coinapi_flatfiles.py`/`download_coinapi.py` day arguments are explicit
already.) Scripts are throttled, resumable, and exit cleanly on quota/billing gates. Secrets in `.env`
(git-ignored). **Figures in this doc are the 2026-06-22 snapshot** — re-run with a new `END` to refresh.

---

## 10. Open items / verification TODOs

Done:
- [x] **Coinbase backfill coverage by CoinAPI** — all 52 fill days verified for the **needed product(s)**:
      47 book (84.6 GB) + 52 trades (2.6 GB), 0 unfillable, 0 probe-error → `backfill_verified=True`.
      Auditable `usable_days`/`lake_all_days`/`excluded_days_by_reason` in `data/usable_calendar.json` (§5b).
- [x] **Unit/timestamp sanity** (L1 mid, clean day) — $0.000 median / 0.999982 corr (§5a). *Not* parity.
- [x] **Crypto Lake bucket region** — `eu-west-1` (pyarrow S3 read; head_bucket 403s).
- [x] **Trade-feed validation** (1 day, 3 venues) — §5b; Coinbase needs origin_time sort.
- [x] **Usable all-feed calendar** — §5b; OOS ≈ April 2026, 52 Coinbase fill days.
- [x] **Coverage scripts de-hard-coded** — anchor on `END`/`--end` (§9).
- [x] **CoinAPI billing/limits understood** — $1/GB flat-files, REST credits, ~10 req/min, shared
      balance (§2.2). **Pre-download action:** enable Spend Management (daily cap + hard-stop).

Hard gates before the hybrid Coinbase plan is production-validated:
- [ ] **Recon-level L3→L2 / L2 parity** — reconstruct Lake `book_delta_v2`→top-K and CoinAPI
      `limitbook_full`→top-K on the same overlap day; compare per-level price/size **and labels** at the
      bar/label horizons. Characterize the ~$249 second-scale spike population (do **not** assume wash-out).
      *(Tooling added & synthetic-validated — `recon/coinapi.py`, `recon/parity.py`,
      `scripts/run_coinbase_parity.py`. **Live run done 2025-06-01 (see §5a "Measured results"):**
      CoinAPI side RESOLVED (`MATCH`=`decrement`, 0% crossed); gate still blocked by Lake
      `book_delta_v2` 67% intraday crossing → needs the reseed policy below.)*
- [ ] **`book_delta_v2` continuous reconstruction + reseed policy** (§5a-Recon) — apply `seq`-order +
      snapshot-first rules; confirm reconstructed book uncrossed across day boundaries and on a day where
      the `book` snapshot product is crossed (e.g. 2026-04-01).
- [ ] **Crypto Lake Coinbase quality map** — how many *present* days have a degraded `book_delta_v2`
      *reconstruction* (not just the `book` snapshot product)? Degraded present-days get CoinAPI fill.

Other open items:
- [ ] **Trade validation breadth** — extend §5b checks to multiple days/regimes per venue.
- [ ] **Within-timestamp ordering for CoinAPI** (no `sequence_number`; rely on `seq` + L3 `order_id`).
- [ ] **Binance downloader** — not yet built; same throttled/resumable/partitioned pattern as
      `download_coinapi.py`, streaming per day (109 M rows). Read direct via pyarrow S3 (`eu-west-1`) or lakeapi.
- [ ] **Liquidations sparsity** — confirm low coverage is genuine (no liquidations) vs missing files.

---

*Last verified 2026-06-22. All coverage/size/timestamp figures measured against live vendor data on
that date; re-run §9 to refresh.*
