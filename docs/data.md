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
- **Cost/quota:** flat ~$64/mo individual plan with a **300 GB/month download limit** (pricing
  page checked 2026-06-30). We still pay no separate AWS egress, but broad raw Lake pulls consume
  the subscription quota. Check `lakeapi.used_data(sess)` before live Lake runs; usage at the
  2026-06-30 check was **0.26 GB / 31 days**.

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

## 5a. Vendor stitching (Coinbase gap-fill) — unit/timestamp sanity PASSED; recon parity RUN 2025-06-01 (CoinAPI + Lake seed/reseed RESOLVED on this day; multi-day validation pending before backfill unlock)

**Status: one-day recon parity PASSES (2025-06-01); NOT yet multi-day/production-validated.** The two
hard gates below (recon-level parity, snapshot/day-boundary semantics) are now **met on the 2025-06-01
pilot** — Lake reconstructed `book_delta_v2` (seed/reseed) ↔ CoinAPI L3→L2 agree to a $0.00 median mid
(see Measured results) — but production needs the **multi-day** validation in §10 (gap days, vendor
seams, a crossed-`book` day, a `SUB` day) before backfill unlock. What we have shown so far:

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

**Hard gates before treating the hybrid as production-validated** (both ✅ MET on 2025-06-01; multi-day
pending — §10):
1. **Recon-level parity ✅ (2025-06-01):** reconstruct Crypto Lake `book_delta_v2` → top-K L2 and CoinAPI
   `limitbook_full` (L3) → top-K L2 for the **same overlap day**, and compare per-level price/size and
   the resulting labels (not just L1 mid). Quantify the spike population at the exact bar/label horizons.
   *Result: |Δmid| median $0.00, corr 0.99999778, label agreement 0.951/0.983/0.995 (see Measured results).*
2. **Snapshot/day-boundary semantics ✅ (2025-06-01):** apply the §4.3 / §5a-Recon ordering rules and
   confirm the reconstructed book is uncrossed across the day boundary. *Result: seed/reseed brings the
   Lake book from 67% crossed to 0.015%.*

**Tooling status (parity gate) — implemented, synthetic-unit-validated, and RUN LIVE on 2025-06-01
(measured results below; Lake side initially FAILED at 67% crossed on the first run, then RESOLVED by
the seed/reseed policy).** The one-day parity gate: `recon/coinapi.py`
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
thin/one-sided top-K depths are marked, not silently dropped. The full validated seed/reseed from
Lake's `book` snapshot product is now **implemented** (`recon/reseed.py`, wired into the parity
script; synthetic-unit-validated — see §5a-Recon "Implementation") and **live-validated on 2025-06-01
(2026-06-30): cold-start 67% crossed → reseed 0.015%, `|Δmid|` median $55.96 → $0.00** (see Measured
results). The CoinAPI **SUB/MATCH size
convention** was an A/B assumption (absolute-size vs `--size-policy decrement`); the live run below
**resolved the MATCH path: `decrement` is correct for Coinbase `limitbook_full` MATCH events**.
⚠️ **SUB is NOT yet verified** — 2025-06-01 had **0 SUB events**, so the `decrement` default also
applies to SUB by family analogy only; a future day with partial-fill `SUB` rows must confirm it
(see "Measured results"). Run (after enabling CoinAPI Spend Management, §8):

```bash
.venv/bin/python ingest/download_coinapi.py --start 2025-06-01 --end 2025-06-01            # one overlap day
.venv/bin/python scripts/run_coinbase_parity.py --day 2025-06-01 --k 10 --size-policy decrement   # -> data/reports/
```

**Measured results — first live run, 2025-06-01 (2026-06-29). Lake side FAILED here (67% crossed) —
RESOLVED by the seed/reseed A/B below (2026-06-30).**
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
   **➜ RESOLVED (2026-06-30)** by the §5a-Recon seed/reseed policy (`recon/reseed.py`) — see the
   reseed A/B below.

**Seed/reseed A/B — live re-run, 2025-06-01, k=10 (2026-06-30).** Same day, same CoinAPI parquet
(`decrement`), now with the §5a-Recon Lake seed/reseed policy. Seeded from the Lake `book` product
(**65,466/65,467 candidates valid** at min-5-levels; 1 one-sided skipped), seed accepted at
00:00:03.18, **3 intraday reseeds** fired (crossed-beyond-2 s episodes), 0 blocked:

| metric (k=10) | **before** (cold-start) | **after** (seed + reseed) |
|---|---|---|
| Lake crossed-book rate (full grid) | **67.04 %** | **0.015 %** (13 samples) |
| `\|Δmid\|` median / p95 / p99 / max | $55.96 / $345 / — / — | **$0.00** / $0.48 / $4.35 / $66.59 |
| mid correlation | 0.977 | **0.99999778** |
| label agreement 2 s / 10 s / 60 s | 0.90 / 0.93 / 0.95 | **0.951 / 0.983 / 0.995** |
| `\|Δmid\|` spikes >$1 / >$10 / >$50 / >$100 | — | 3127 / 199 / 2 / 0 |

The reseed clears the stranded levels: the cold-start 67 % crossing collapses to **0.015 %** (12.3 s
total residual crossed time across the 3 episodes), and the Lake mid now matches CoinAPI to a **$0.00
median** with **0.99999778** correlation. Parity ran on **86,383 / 86,400** grid points (4 pre-seed
warm-up samples — the cutoff is clamped to the accepted seed at 00:00:03.18 — plus 13 residual-crossed
samples excluded; `n_grid_full` stays the true 86,400). The A/B confirms
the fix is the reseed, not a code-path change — the cold arm is the byte-identical reconstruction.
The known rare second-scale spikes survive as a small, *characterized* tail (2 samples >$50, max
$66.59), not assumed to wash out. Report artifacts (git-ignored):
`data/reports/parity_coinbase_2025-06-01_k10*.{json,csv}`.

**Gate status & backfill.** The Lake-side blocker is **resolved on 2025-06-01**; with the CoinAPI
`decrement` fix and Lake seed/reseed, the day's recon-level parity is clean. Backfill stays **gated**
pending multi-day validation — other days (gaps, vendor seams), a day where the Lake `book` product is
itself crossed (e.g. 2026-04-01 — measured 2026-07-01: a seed IS accepted on such days, but the crossed
source fails the §5a-QualityMap reliability bar, so the day routes `inconclusive` and needs CoinAPI
cross-validation/fill), and a day with real `SUB` events. This is **enforced in code**:
`ingest/download_coinapi.py` refuses a backfill-scale pull (exit 4) — a single parity day, or a multi-day range with a small `--sample-mb` smoke (≤64 MB), is
allowed; a multi-day full pull (or an oversized `--sample-mb`) is blocked, `--allow-backfill` overrides.

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
Whether the underlying `book_delta_v2` *reconstruction* is also degraded on such days was **measured
2026-07-01** (§5a-QualityMap expanded validation): yes — on the 4 sampled crossed-source days the
reconstruction stays crossed 1.8–9.4 h (most reseed attempts blocked by invalid snapshots), so those
days are `inconclusive` pending CoinAPI cross-validation; degraded present-days get CoinAPI fill like gaps.

**Implementation (`recon/reseed.py`, synthetic-unit-validated + live-validated 2025-06-01:
67% → 0.015% crossed).** The policy is:
- **Seed:** parse the Lake `book` product into time-sorted candidates (`snapshots_from_lake_book_df`,
  thinned by a stride so the large product never fully materializes), validate each
  (`classify_snapshot`: two-sided, finite/positive, ≥N levels/side, uncrossed, optional sane spread),
  and seed the `OrderBook` from the first valid one. An invalid candidate (crossed/thin/one-sided) is
  skipped with a reason code; if none is valid the book cold-starts and `seed_accepted=False`.
- **Reseed:** snapshots are merged into the time-ordered delta stream as reseed events at their OWN
  timestamp; when the reconstructed book stays crossed continuously for ≥ `reseed_after_crossed_s`, the
  next valid snapshot REPLACES the whole state (dropping the stranded levels). Because a reseed event is
  applied at its own ts, a sample at grid `g` only ever reflects a reseed with `ts ≤ g` — **no
  look-ahead**; samples inside the crossed window (before the fixing snapshot) stay crossed and are
  reported/excluded, never silently back-patched.
- **Not a `seq` gap detector:** the trigger is the observable crossed book, NOT a `sequence_number`
  diff. Coinbase `book_delta_v2` duplicates `seq` across ~91% of rows (per-event, and the channel
  counts trades too), so a naive row-to-row `seq` diff is meaningless as a dropped-data signal;
  `OrderBook.apply()`'s monotonicity flag is informational only and is never consumed.
- **Reported:** `scripts/run_coinbase_parity.py` carries a `lake_reseed` block — seed accepted/rejected
  + reason, seed ts, reseed count/timestamps, snapshot reason codes, crossed-duration, and the
  **before(cold)/after(reseed) crossed rate A/B** (`--no-reseed` = seed-only arm, `--no-lake-seed` =
  pure cold-start). Residual crossed Lake samples are excluded from the parity comparison and counted.

A single day-open seed is **not** sufficient (the live failure is intraday level-stranding, not
cold-start); reseed-on-crossing is the fix. Prior-day seed carry-across and the vendor-switch-seam
reseed (Lake↔CoinAPI) remain follow-ups beyond this one-day pilot.

### 5a-QualityMap. Multi-day Coinbase quality map (quota-aware)
`scripts/run_coinbase_quality_map.py` generalizes the one-day parity gate's **Lake side** across many
days to answer the §10 open question — *how many PRESENT Coinbase days reconstruct to a usable
`book_delta_v2` after seed/reseed, and which present-but-degraded or missing days need CoinAPI fill?*
It runs the same `recon/reseed.py` seed/reseed reconstruction-quality path per day
(`reconstruct_lake_l2_at_samples_seeded` + `recon/parity.py::frame_quality`) — **Lake-only, no CoinAPI
replay, so no CoinAPI download** — and classifies each day with explicit thresholds (emitted in the
report JSON):

| class | meaning | rule (default thresholds) |
|---|---|---|
| `lake_usable` | present, seed accepted + trusted source, clean reconstruction | crossed ≤ **1%**, missing ≤ **2%**, thin ≤ **10%** (after reseed), seed-source crossed ≤ **5%** |
| `lake_present_degraded` | present, seed accepted + trusted source, a quality metric over the bar → CoinAPI fill | any usable crossed/missing/thin threshold exceeded |
| `missing_needs_coinapi` | no Lake `book_delta_v2` for the day (a gap) → CoinAPI fill | 0 delta rows (or lakeapi `NoFilesFound`) |
| `excluded` | out of the usable calendar for a non-Coinbase reason (e.g. a Binance gap) | day in `excluded_days_by_reason` (skipped before any Lake load) |
| `inconclusive` | cannot validate the reconstruction | no/all-rejected seed, **or** an accepted seed whose `book` source is itself crossed > **5%**, **or** the Lake load failed |

A confident `lake_usable`/`lake_present_degraded` verdict **requires both an accepted seed and a
trustworthy seed source**: on 2026-04-01 the `book` product is **31.75% crossed**, so although ~68% of
candidates are valid and a seed IS accepted, the source exceeds the **5%** crossed-candidate bar → the
day is `inconclusive`, not silently usable (a clean-looking reconstruction off a flaky seed source can't
be trusted). The classifier keys off missing `book_delta_v2`, no/rejected seed snapshots, the crossed
seed-source fraction, the crossed rate after reseed, the missing-book fraction, and thin top-K depth; the
per-day record additionally **records** (it does not classify on) whether CoinAPI parity — a local
parquet or a calendar-verified flat-files `book` — was available, for downstream fill planning.

**Quota-aware (docs §2.1/§6/§8).** It prints `lakeapi.used_data(sess)` before/after, estimates the
request from measured per-day sizes (`book_delta_v2` ~0.30 GB/day from §6; the `book` 20-level snapshot
product ~0.18 GB/day ≈ 275k rows → ~0.48 GB/day, a conservative upper bound), and **refuses a broad
pull** (> `--max-auto-gb`, default 5 GB) unless `--allow-broad` — and *always* refuses a pull that would
breach the 300 GB/month quota headroom, override or not (and refuses the whole run, fail-safe, if
`used_data` is unreadable). The default day set is small: **2025-06-01** (the validated clean day →
`lake_usable`) and **2026-04-01** (31.75%-crossed `book` seed source → `inconclusive`); add more with
`--days` / `--days-file` / `--include-gap-days N`. The full report is written to
`data/reports/coinbase_quality_map.json` (git-ignored).

**Live native smoke — default two-day set, 2026-07-01 (`--engine native --no-cold-ab`).** This validates
the native quality-map path and the quota guard, but it is **not** the broad production quality map.
Report summary: `lake_usable=1`, `inconclusive=1`, estimated Lake download 0.96 GB, usage before/after
0.26 GB / 31 days (quota 300 GB; provider usage may lag), native engine selected.

| day | class | key metrics / reason |
|---|---|---|
| 2025-06-01 | `lake_usable` | 16,517,806 delta rows; crossed_rate_after 0.000150 (13/86,400 samples); missing 0.000023; thin 0.000012; 3 reseeds |
| 2026-04-01 | `inconclusive` | 34,657,476 delta rows; accepted seed but seed source unreliable (`seed_source_crossed_frac=0.3751>0.05`); crossed_rate_after 0.391; 6 reseeds |

One per-day record shape:

```json
{"day": "2025-06-01", "classification": "lake_usable",
 "reasons": ["seed_accepted", "crossed_rate_after=0.0002", "missing_book_fraction=0.0000",
             "thin_depth_fraction=0.0000"],
 "lake_book_delta_v2_present": true, "lake_delta_rows": 16500000,
 "seed": {"seed_accepted": true, "reseed_count": 3, "snapshot_reason_codes": {"ok": 65466}},
 "quality": {"crossed_rate_after": 0.00015, "crossed_rate_cold": 0.6704,
             "missing_book_fraction": 0.0, "thin_depth_fraction": 0.0},
 "coinapi": {"parquet_local": false, "fillable": null},
 "calendar": {"in_usable_days": true, "is_coinbase_fill_day": false, "excluded_reason": null}}
```

**Expanded validation — 10-day map with gap/seam coverage, 2026-07-01 (`--engine native --no-cold-ab`).**
Three quota-gated runs, each estimated under the 5 GB auto cap (no `--allow-broad`):

```bash
# 1) default two-day set + first 3 documented Coinbase book-gap days (est ~2.40 GB)
.venv/bin/python scripts/run_coinbase_quality_map.py --engine native --no-cold-ab --include-gap-days 3
# 2) follow-up: 5 representative present days — the 2024-08-05 volatile/crash day, both seams of the
#    33-day hole, a mid-window and a late-window day (est ~2.40 GB)
.venv/bin/python scripts/run_coinbase_quality_map.py --engine native --no-cold-ab \
  --days 2024-08-05,2024-12-04,2025-01-07,2025-10-15,2026-06-15
# 3) consolidated 10-day artifact — re-reads 1)+2) from the local lakeapi cache (est 4.80 GB,
#    ~0 GB incremental download)
.venv/bin/python scripts/run_coinbase_quality_map.py --engine native --no-cold-ab --include-gap-days 3 \
  --days 2025-06-01,2026-04-01,2024-08-05,2024-12-04,2025-01-07,2025-10-15,2026-06-15
```

Generated 2026-07-01 (20:09:11Z / 20:11:06Z / 20:17:13Z), native engine (tick scale 100) on all three
runs; per-day metrics identical across runs (deterministic replay). Quota: `used_data` read 0.26 GB /
31 days before and after every run (the vendor counter may lag ~60 min). **Measured wire transfer** —
the S3 objects the runs fetched (metadata-only `ListObjectsV2` sizes, 7 present days, both products):
**1.48 GB `book_delta_v2` (173 M rows) + 0.36 GB `book` = 1.84 GB**, vs the 4.80 GB conservative
unique-day estimate; gap days cost ~0. Object sizes equal bytes transferred here because lakeapi
(0.22.3) `_download_one` GETs each partition object whole (the `columns=` projection is applied
after download), but they are **not yet vendor-confirmed spend**: cache-busted `used_data` re-reads
at 21:20Z and 21:39Z still returned 0.26 GB with vendor `update_time` 19:20:30Z (pre-run), so what
Crypto Lake ultimately counts against the quota for these runs is pending — treat 1.84 GB as the
wire-transfer measurement and re-check `used_data` before the next sizeable pull. The `book`
estimator constant (0.18 GB/day) is ~3.5× the measured ~0.05 GB/day — the auto-cap gate
over-estimates, as designed.

Counts (n=10): **lake_usable 2, lake_present_degraded 1, missing_needs_coinapi 3, excluded 0,
inconclusive 4.** Per-day (rates are fractions of the 86,400 1 s grid samples):

| day | class | key reason | delta rows | crossed after | missing | thin | reseeds (blocked) |
|---|---|---|---|---|---|---|---|
| 2025-06-01 | `lake_usable` | clean seeded recon | 16,517,806 | 0.000150 | 0.000023 | 0.000012 | 3 (0) |
| 2026-04-01 | `inconclusive` | `seed_source_crossed_frac=0.3751` | 34,657,476 | 0.3910 | 0.000023 | 0.000012 | 6 (21,449) |
| 2024-08-05 | `inconclusive` | `seed_source_crossed_frac=0.2878`; book starts 16:08:35Z | 19,492,977 | 0.0858 | 0.6726 | 0 | 0 (5,560) |
| 2024-12-04 | `inconclusive` | `seed_source_crossed_frac=0.0836` | 29,583,498 | 0.0743 | 0 | 0 | 7 (4,154) |
| 2025-01-07 | `lake_present_degraded` | `missing_book_fraction=0.6146>0.02`; book resumes 14:45:00Z | 16,810,189 | 0.000116 | 0.6146 | 0 | 2 (0) |
| 2025-10-15 | `inconclusive` | `seed_source_crossed_frac=0.2833` | 39,418,924 | 0.2715 | 0.000012 | 0.000012 | 7 (17,235) |
| 2026-06-15 | `lake_usable` | clean seeded recon | 16,794,631 | 0.000174 | 0 | 0 | 4 (0) |
| 2024-07-14 | `missing_needs_coinapi` | Lake book absent; CoinAPI fill calendar-verified | 0 | — | — | — | — |
| 2024-08-06 | `missing_needs_coinapi` | Lake book absent; CoinAPI fill calendar-verified | 0 | — | — | — | — |
| 2024-08-19 | `missing_needs_coinapi` | Lake book absent; CoinAPI fill calendar-verified | 0 | — | — | — | — |

Findings:

1. **The crossed-`book` seed-source problem is widespread, not a 2026-04-01 oddity.** 4 of the 7
   sampled present days exceed the 5% seed-source bar (8.4–37.5% of seed candidates crossed), spread
   across Aug '24, Dec '24, Oct '25 and Apr '26. On those days most reseed attempts are blocked by
   invalid snapshots (4,154–21,449 blocked) and the reconstructed book stays crossed for hours
   (crossed duration after reseed 1.8–9.4 h), so Lake data alone cannot certify them.
2. **Gap edges bleed into adjacent "present" days as leading partial days.** 2024-08-05
   `book_delta_v2` only starts 16:08:35Z (67.3% of the grid missing — the crash morning itself is
   absent, and the next day is a full gap); 2025-01-07 resumes 14:45:00Z (61.5% missing) as the
   33-day hole ends mid-day. 2025-01-07 is otherwise clean where present (crossed 0.000116, clean
   seed source) → the classic partial-day CoinAPI-fill shape; 2024-08-05 has BOTH the partial day and
   a crossed seed source. Fill planning must budget partial-day fills on seam days, not only the
   calendar's full-gap days.
3. **Gap days route correctly and are fillable.** All 3 documented book-gap days raise lakeapi
   `NoFilesFound` → `missing_needs_coinapi`, and each is calendar-verified fillable from CoinAPI flat
   files (`coinapi.fillable=true`).
4. **Tooling validated at multi-day scale.** Native engine selected on all runs, three runs produced
   identical per-day metrics, the quota estimator/auto-cap gated every run, and the consolidated
   cache-hit run re-classified 10 days with ~0 incremental download.

**Conclusion: backfill stays LOCKED.** The expanded map does not clear the gate — only 2 of 7 sampled
present days classify `lake_usable`; 4 are `inconclusive` (unreliable seed source, no verdict possible
from Lake alone) and 1 is a degraded seam day needing a partial-day CoinAPI fill. Still missing before
unlock: (a) CoinAPI cross-validation (or another trusted seed source) for the crossed-seed-source
days — the §10 multi-day reseed validation; (b) partial-day fill handling for seam days; (c) the broad
production map over all 652 present days — ~313 GB at the tool's conservative 0.48 GB/day gate
estimate (refused as a one-shot, by design), ~170 GB at the measured wire rate of ~0.26 GB/day
(vendor-counter confirmation pending) — either way a stage-across-quota-windows pull alongside the
archive downloads (§6).

**Backfill stays LOCKED.** This is a validation/quality-map tool: it does not download CoinAPI and does
not unlock the §5a backfill gate (still enforced in `ingest/download_coinapi.py` / `ingest/_common.py`).
Bulk backfill remains gated until the multi-day quality map (and the §10 multi-day reseed validation)
passes.

**Staging the broad map (batch planner).** `scripts/plan_coinbase_quality_map_batches.py` turns the
stage-across-quota-windows requirement above into deterministic day batches: it reads
`data/usable_calendar.json` and selects every day with a PRESENT Lake book — `lake_all_days` (652)
plus the 5 trade-only fill days (`book: false` = only trades gapped; their book quality still needs
validating before the backfill gate, and they stay listed separately in the manifest because each
still needs a CoinAPI trades fill) = **657 days**. It withholds the 47 book-gap fill days (Lake book
absent — they classify `missing_needs_coinapi` at ~0 GB; map them separately via the runner's
`--include-gap-days`) and the 26 non-Coinbase `excluded_days_by_reason`, then chunks the rest into
`--days-file` batches under a configurable GB budget. Default: **250 GB/batch** (current planning
target — one batch per monthly quota window with headroom under the 300 GB cap) at the
**conservative 0.48 GB/day** §6 estimate (matches the runner's quota estimator; measured wire rate is
lower, ~0.26 GB/day). The current calendar plans **2 batches** (520 + 137 days ≈ 249.6 + 65.8 GB,
~315.4 GB total). Batch files plus a manifest (day counts,
per-batch GB estimates, the exact runner command per batch) land under the git-ignored
`data/tmp/coinbase_quality_map_batches/`; batch files are byte-deterministic for a given
calendar + budget. **Planning only:** the planner performs no vendor I/O — it does not run Lake
downloads and does not unlock the §5a backfill gate. Each batch still passes through the runner's own
`used_data`/headroom quota gate when executed, so run at most one batch per quota window and re-check
`lakeapi.used_data` first (a ~249.6 GB batch estimate only fits under the 300 GB cap − 10 GB headroom
if less than ~40 GB is already used that month).

```bash
# 1) plan the batches (add --dry-run to print the plan without writing files)
.venv/bin/python scripts/plan_coinbase_quality_map_batches.py \
  --max-gb-per-batch 250 \
  --gb-per-day 0.48

# 2) run ONE batch this quota window (the runner re-checks used_data + quota headroom).
#    Per-batch --out-dir: the runner writes a fixed coinbase_quality_map.json under --out-dir, so
#    staged batches would otherwise overwrite each other's report; --usable-calendar pins the
#    exact calendar the plan was built from. Both are already baked into the manifest's commands.
.venv/bin/python scripts/run_coinbase_quality_map.py \
  --engine native \
  --no-cold-ab \
  --days-file data/tmp/coinbase_quality_map_batches/batch_001_days.txt \
  --usable-calendar data/usable_calendar.json \
  --out-dir data/reports/coinbase_quality_map_batches/batch_001 \
  --allow-broad
```

**Replay engine (Python reference vs native).** The pure-Python seed/reseed replay
(`recon/reseed.py`) is the **correctness reference/oracle**, but it is single-process and its per-event
`max(dict)/min(dict)` best-bid/ask scans make it O(N·L) — the 2026-07-01 Python-only quality-map smoke
run was still CPU-bound after hours on the 16.5M/34.7M-row default days and cannot support multi-day
(let alone Binance ~109M rows/day) work. The **native engine** (`recon_native`, Rust/PyO3 —
docs/native-recon.md) is the throughput implementation required for operational multi-day runs: an
integer-tick order book with O(log L) best-bid/ask and no per-row object boxing. Both
`scripts/run_coinbase_quality_map.py` and `scripts/run_coinbase_parity.py` take
`--engine {auto,python,native}` (default `auto` = native when the extension is built and the symbol has
a verified tick scale, else Python; explicit `--engine native` fails before any Lake load if
unavailable/unverified), and record the selected engine in the report JSON. The native path preserves
the Python seed/reseed semantics exactly (pinned by native-vs-Python conformance tests) and, in
quality-map mode, classifies from metrics-only meta without materializing the top-K frame. On a
synthetic 1M-row / 10 000-level / 20%-churn fixture the native engine is **~1293× faster than Python**
(244.35 s → 0.189 s) with byte-identical output (`scripts/bench_recon_engine.py`; 12th Gen i5-12400F,
Python 3.12, rustc 1.94). The default two-day live quality-map smoke and the expanded 10-day validation
(see above) have run successfully with the native engine (2026-07-01), but the full-window production
quality map is still quota-gated (§9); backfill stays locked until that broader map passes.

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

| Span | Crypto Lake (all feeds) | Individual-plan quota implication | + CoinAPI backfill (~52 fill days, L3) | **Total raw** |
|---|---|---|---|---|
| 12 mo | 427 GB | >1 monthly cap; stage across **2** quota windows or upgrade | ~120 GB | **~0.55 TB** |
| 18 mo | 643 GB | >2 monthly caps; stage across **3** quota windows or upgrade | ~120 GB | **~0.76 TB** |
| 24 mo | 858 GB | ~3 monthly caps exactly; use **3-4** windows with headroom or upgrade | ~120 GB | **~0.98 TB** |

(Backfilled L3 is transient — it aggregates to top-K L2 at ~10 GB after recon. Fill-day count scales
with the chosen span; ~52 is the full 2-yr usable-calendar set, §5b.)

**Crypto Lake quota constraint:** the measured all-feed Lake footprint is ~1.17 GB/day, so the
300 GB/month individual-plan cap covers at most ~256 all-feed days before any safety margin. Small
validation pulls (single-day parity, the *sampled* multi-day Coinbase quality map — the 2026-07-01
10-day run measured 1.84 GB of wire transfer — metadata coverage checks) are fine; the **full-window**
quality map over all 652 present days is *not* small (~313 GB conservative / ~170 GB measured
wire-rate, §5a-QualityMap) and must be staged like the archive. A full 12-24 mo Lake archive is **not** a
one-shot pull on this plan: stage by month/quota
window, project only needed columns, process/recon day-by-day, and keep resumable manifests so a
run can stop before the quota is tight.

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
| Crypto Lake subscription | ~$64/mo individual plan, **300 GB/month download limit** |
| CoinAPI Coinbase backfill (§5b, all verified), book 47 d / trades 52 d | **~$92** (book 84.6 GB≈$85 + trades 2.6 GB≈$8) at $1/$3 per GB; −$25 ≈ **$67 OOP** |
| ↳ if only the single 33-day hole is filled | ~$82 |
| Compute | $0 (local); optional one-off same-region (**eu-west-1**) spot for recon |
| Storage | $0 (existing 2 TB SSD) |

- **Crypto Lake quota:** pricing checked 2026-06-30; individual plan is 300 GB/month. The planned
  18-mo all-feed Lake pull is ~643 GB, so either stage it across ~3 monthly quota windows (with
  headroom) or use a higher plan. Do not schedule broad raw Lake downloads as if the plan were
  unlimited.
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
.venv/bin/python scripts/run_coinbase_quality_map.py    # multi-day Lake quality map (§5a-QualityMap; quota-aware, prints used_data, refuses broad pulls without --allow-broad) -> data/reports/
.venv/bin/python scripts/plan_coinbase_quality_map_batches.py   # stage the broad quality map into quota-window day batches (§5a-QualityMap; planning only, NO vendor I/O; --dry-run prints without writing) -> data/tmp/
# NOTE: multi-day BULK pulls are the backfill and are GATED — download_coinapi.py refuses a >1-day full
# pull (exit 4) until the §5a parity + reseed gates pass. A single day, or a multi-day range with a small
# --sample-mb smoke (≤64MB), is allowed; --allow-backfill overrides once the gate passes (Spend Mgmt on, §8).
```
Coverage windows are **anchored on the `END` env var** (default `2026-06-22`); without it they reproduce
the original snapshot. (The `coinapi_flatfiles.py`/`download_coinapi.py` day arguments are explicit
already.) Scripts are throttled, resumable, and exit cleanly on quota/billing gates. Before any broad
Crypto Lake `load_data` run, check current monthly usage with `lakeapi.used_data(sess)` and leave quota
headroom. Secrets in `.env` (git-ignored). **Coverage/size/timestamp figures in this doc are the
2026-06-22 snapshot** — re-run with a new `END` to refresh; Crypto Lake pricing/quota was checked
2026-06-30.

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
- [x] **Recon-level L3→L2 / L2 parity (2025-06-01)** — reconstruct Lake `book_delta_v2`→top-K and CoinAPI
      `limitbook_full`→top-K on the same overlap day; compare per-level price/size **and labels** at the
      bar/label horizons; characterize the ~$249 second-scale spike population (do **not** assume wash-out).
      *(Tooling: `recon/coinapi.py`, `recon/parity.py`, `recon/reseed.py`, `scripts/run_coinbase_parity.py`.*
      ***Live run 2025-06-01:** CoinAPI side RESOLVED (`MATCH`=`decrement`, 0% crossed) AND Lake side RESOLVED
      by seed/reseed (67% → 0.015% crossed); |Δmid| median $0.00, corr 0.99999778, spikes >$50 = 2/86k.)*
- [x] **`book_delta_v2` continuous reconstruction + reseed policy** (§5a-Recon) — seed/reseed IMPLEMENTED
      (`recon/reseed.py`, synthetic-unit-validated: valid-seed-usable, crossed-seed-rejected,
      stranded-recovers-on-reseed, no-look-ahead, tolerance-window, `seq`-duplicates-don't-trigger,
      cold-start-equivalence) and **LIVE-VALIDATED 2025-06-01 (2026-06-30): 67.04% → 0.015% crossed,
      `|Δmid|` median $0.00, corr 0.99999778, 3 reseeds** (see §5a Measured results / seed-reseed A/B).
- [ ] **Multi-day reseed validation before backfill unlock** — a day where the `book` product is itself
      crossed (e.g. 2026-04-01 — the 2026-07-01 quality map measured that a seed IS accepted on such
      days but the day routes `inconclusive` via the crossed-seed-source bar → needs CoinAPI
      cross-validation), a vendor-seam day, and a day with real `SUB` events. Prior-day seed carry +
      vendor-switch-seam reseed still deferred.
- [ ] **Crypto Lake Coinbase quality map** — how many *present* days have a degraded `book_delta_v2`
      *reconstruction* (not just the `book` snapshot product)? Degraded present-days get CoinAPI fill.
      *(Tooling: `scripts/run_coinbase_quality_map.py` — §5a-QualityMap, quota-aware, classifies
      lake_usable / lake_present_degraded / missing_needs_coinapi / excluded / inconclusive. Live
      native runs 2026-07-01: the default two-day smoke, then the expanded 10-day map
      (§5a-QualityMap "Expanded validation"): 2 `lake_usable`, 1 degraded seam day, 3 gaps,
      4 `inconclusive` — the crossed seed source recurs on 4 of 7 sampled present days
      (8.4–37.5% crossed candidates, Aug '24–Apr '26), and seam days lose the leading 61–67% of the
      day. Remaining gate: CoinAPI cross-validation for crossed-seed-source days, partial-day fill
      handling for seam days, plus the full-window map (~313 GB conservative / ~170 GB measured
      wire-rate → staged across quota windows); backfill stays locked until it passes.)*

Other open items:
- [ ] **Trade validation breadth** — extend §5b checks to multiple days/regimes per venue.
- [ ] **Within-timestamp ordering for CoinAPI** (no `sequence_number`; rely on `seq` + L3 `order_id`).
- [ ] **Binance downloader** — not yet built; same throttled/resumable/partitioned pattern as
      `download_coinapi.py`, streaming per day (109 M rows). Read direct via pyarrow S3 (`eu-west-1`) or lakeapi.
- [ ] **Liquidations sparsity** — confirm low coverage is genuine (no liquidations) vs missing files.

---

*Last verified 2026-06-22 for coverage/size/timestamp figures; Crypto Lake pricing/quota checked
2026-06-30. Re-run §9 to refresh vendor measurements.*
