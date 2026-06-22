# ingest — CoinAPI Coinbase verification

Fills the spec §4 Coinbase gap (Crypto Lake's Coinbase book is ~80%/gappy).
Two CoinAPI access paths, **separate credit pools**:

| Script | Product | Credit pool | What it gives |
|---|---|---|---|
| `coinapi_rest.py` | Market Data REST | the **$25 free credit** | symbol coverage dates; trades + top-N L2 schema. Cheap validation. |
| `coinapi_flatfiles.py` | Flat Files (S3) | separate (must fund) | **full-depth incremental `limitbook_full`** (L2/L3) — the production pull. |

## Setup
```bash
echo "COINAPI_KEY=your-key" > .env        # already present
.venv/bin/pip install boto3 pandas        # already installed
```

## Run
```bash
.venv/bin/python ingest/coinapi_rest.py          # validate via the $25 REST credit
.venv/bin/python ingest/coinapi_flatfiles.py 45  # day-level coverage + 8MB schema sample (last 45d)
```

Both exit `2` with guidance if the account has no usable credit (HTTP 403
`Insufficient Usage Credits`). The $25 is granted only after a payment method is
verified and shows as Usage Credits in the Customer Portal (Billing). REST and
Flat Files credit are **separate** — fund the pool you intend to use.

## Cost discipline
- `coinapi_rest.py`: a few credits total (100 data points = 1 credit; order-book
  date queries capped at 10 credits).
- `coinapi_flatfiles.py`: coverage via cheap LIST calls; schema via a bounded
  **8 MB HTTP Range GET** of one file (not the multi-GB full day). It prints
  per-day size and an 18-month projection before any bulk download.

## Downloader → partitioned Parquet (`download_coinapi.py`)

Pulls `limitbook_full` day-by-day into Hive-partitioned Parquet (spec §12.1):
```
data/raw/limitbook_full/exchange=COINBASE/symbol=BTC-USD/dt=YYYY-MM-DD/data.parquet
```
```bash
# smoke test (cheap — parses first 8 MB only, writes to data/raw/_sample/)
.venv/bin/python ingest/download_coinapi.py --start 2026-05-28 --end 2026-05-28 --sample-mb 8

# real pull (resumable — re-run after interruption and it skips finished days)
.venv/bin/python ingest/download_coinapi.py --start 2025-01-01 --end 2025-06-30
.venv/bin/python ingest/download_coinapi.py --start 2025-01-01 --end 2025-01-31 --keep-raw
```
Key properties:
- **Throttled**: every S3 call via the shared 8/min `RateLimiter`; one `get_object`
  stream per file (1 request — avoids boto3 multipart, which would trip the 10/min tier limit).
- **Resumable**: skips days whose `data.parquet` exists; atomic `os.replace` from `.tmp`;
  per-day audit log in `data/raw/_manifest.jsonl`. Stale temp files are swept on startup.
- **Memory-safe**: chunked CSV→Parquet (ZSTD) row groups — never loads a full day in RAM.
- **Faithful schema** (lossless, no date assumption — recon's job):
  `seq, time_exchange_ns, time_coinapi_ns, update_type, is_buy, entry_px, entry_sx, order_id`.
  `seq` (row order in the file) is the canonical event order (there is no `sequence_number`);
  `*_ns` are ns-since-midnight UTC — recon combines with the partition `dt`.
- Targets **consolidated daily** partitions; the recent ~3-week hourly tail is out of scope
  (that's live capture). `--sample-mb` output is isolated under `_sample/` so it never
  shadows a real partition.

> Heads-up: BTC-USD `limitbook_full` is ~1.9 GB/day compressed (L3) → ~1 TB / 18 mo.
> See `coinapi-coinbase-fit` memory for the size mitigation options.

## What this verifies (spec §4 checklist)
1. Coinbase BTC-USD coverage/gaps in a recent window vs Crypto Lake's ~80%.
2. `limitbook_full` schema: `time_exchange` + `time_coinapi` (double-stamped →
   §5.3 event-time recon), incremental `update_type`, full depth.
3. Per-day compressed size → cost projection for the 12–24mo SSL span.
