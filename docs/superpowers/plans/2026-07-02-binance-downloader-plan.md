# Binance Crypto Lake Downloader & Native Reconstruction — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a staged, resumable, quota-aware downloader for Binance futures/spot Crypto Lake feeds and a per-day/per-instrument native reconstruction pass that emits bar-ready top-K L2 plus normalized trades/funding/OI/liquidations tables — the primary-signal substrate the model consumes (spec §1, docs/data.md §5b/§6/§10 "Binance downloader — not yet built").

**Architecture:** Two decoupled stages that mirror the existing CoinAPI/quality-map patterns.
**Stage 1 — download/normalize** (`ingest/download_lake_binance.py`): pull each `(feed, exchange, symbol, day)` partition from Crypto Lake S3 (`eu-west-1`), stream it row-group-by-row-group into a lossless, Hive-partitioned, ZSTD Parquet **normalized raw store** — never decompressing to CSV, never holding a full day in RAM. Quota is the binding constraint (300 GB/month, docs §2.1), so this stage runs once per quota window, staged by a deterministic batch planner, and is fully resumable.
**Stage 2 — reconstruct** (`scripts/run_binance_recon.py`): read the *local* normalized store — including the `book` 20-level snapshot Stage 1 pulls as the seed source — (quota-free, re-runnable) and, per `(instrument, day)`, run the existing `recon` seed/reseed engine — native when a verified tick scale exists, Python oracle otherwise — to emit top-K L2 book frames; trades/funding/OI/liquidations are normalized passthrough tables. Every output carries row counts, checksums, and a schema version in a manifest.

**Tech Stack:** Python 3.12, `lakeapi` 0.22.3 + explicit `boto3.Session` (subscriber keys, `eu-west-1`), `pyarrow` streaming Parquet, the existing `recon/` package (`reseed.py`, `native.py`, `parity.py`) and the Rust/PyO3 `recon_native` engine (docs/native-recon.md), `pytest` with synthetic fixtures only (no live vendor in CI).

---

## Scope, altitude, and non-goals

**This is a docs/spec branch.** It writes exactly one file — this plan. No code is added on this branch, so no `pytest` run is required (AGENTS.md: docs-only PRs validate with `git diff --check`).

**No vendor/API calls are run on this branch:** no Crypto Lake `load_data`/`list_data`, no CoinAPI, no native benchmark, no quality-map batch. All commands below are for the *future* implementation branch.

**In scope (the products this plan covers):**

| Instrument | `exchange` | `symbol` | Feeds |
|---|---|---|---|
| Binance **BTC-USDT-PERP** (futures) | `BINANCE_FUTURES` | `BTC-USDT-PERP` | `book_delta_v2`, `trades`, `funding`, `open_interest`, `liquidations` |
| Binance **BTC-USDT** (spot) | `BINANCE` | `BTC-USDT` | `book_delta_v2`, `trades` |

(The **perp** pair is used in `ingest/verify_lake.py:21-22` / `ingest/verify_lake2.py`; the **spot** pair `BINANCE`/`BTC-USDT` in `ingest/verify_trades_and_calendar.py:98-99`; the `liquidations` feed id in `ingest/verify_lake.py:70-72`. Reuse them verbatim — do not re-invent.)

**Non-goals (do not do these in the implementation PR):**
- No CoinAPI backfill unlock — the §5a gate in `ingest/_common.py` stays as-is; this plan does not touch Coinbase.
- No broad/bulk Lake archive pull. Stage 1 supports it but the implementation PR runs at most a **one-day** cheap validation (Phase 2), never the historical archive.
- No new bar/feature/model code — Phase 4 only defines the *seam* the bar builder plugs into.
- No re-implementation of `recon/reseed.py` seed/reseed semantics or the native engine — reuse them.
- No new production dependency without calling it out in the PR (AGENTS.md).

---

## File structure (create/modify map)

Design each unit with one responsibility; keep vendor-schema knowledge at the ingestion boundary (AGENTS.md "Keep vendor-specific schema knowledge at ingestion boundaries").

| Path | New? | Responsibility |
|---|---|---|
| `ingest/lake_binance.py` | Create | **Stdlib/pandas-light** pure helpers: feed/instrument registry, partition-path builders, manifest read/append, per-day/feed state, quota estimation, and the broad-pull gate. No `boto3`/`lakeapi`/`pyarrow` import at module top so CI unit tests import it without vendor deps (mirrors the `ingest/_common.py` split). |
| `ingest/download_lake_binance.py` | Create | Stage-1 CLI. Streams vendor partitions → normalized Parquet. Imports `lake_binance` + `lakeapi`/`pyarrow` (vendor-touching, like `download_coinapi.py`). |
| `scripts/plan_lake_binance_batches.py` | Create | Deterministic quota-window batch planner (mirror of `scripts/plan_coinbase_quality_map_batches.py`): splits a day range into GB-budgeted `--days-file` batches under the 300 GB/month cap. Planning only, no vendor I/O. |
| `scripts/run_binance_recon.py` | Create | Stage-2 runner. Reads the local normalized store; per `(instrument, day)` emits top-K L2 (book_delta_v2) and normalized trades/funding/OI/liq. `--engine {auto,python,native}`, `--jobs N` day-parallel. |
| `recon/native.py` | Modify (`_TICK_SCALE`) | Add the **verified** Binance tick scales after Phase 1 confirms them (see Risks Q1). |
| `tests/test_lake_binance_paths.py` | Create | Partition-path + registry generation (pure). |
| `tests/test_lake_binance_manifest.py` | Create | Manifest append/read, resume/idempotency, per-day state (pure, `tmp_path`). |
| `tests/test_lake_binance_schema.py` | Create | Normalized-schema validation + vendor-drift fail-loud (synthetic frames). |
| `tests/test_lake_binance_engine_time.py` | Create | Engine-time selection + partial-`origin_time` → `received_time` fallback / drop-and-record (synthetic frames). |
| `tests/test_lake_binance_gate.py` | Create | Quota estimation + broad-pull gate exit codes (pure). |
| `tests/test_plan_lake_binance_batches.py` | Create | Batch planner determinism/budget (pure). |
| `tests/test_binance_recon_conformance.py` | Create | Native-vs-Python top-K conformance on a Binance-shaped synthetic fixture (`price_scale=10`), skipped when native absent. |

---

## Requirement 1 — Scope products

Encode the products as a frozen registry in `ingest/lake_binance.py` so every downstream path, manifest key, and quota estimate derives from one source (DRY). Sketch:

```python
from dataclasses import dataclass

FEEDS = ("book_delta_v2", "trades", "funding", "open_interest", "liquidations")

@dataclass(frozen=True)
class Instrument:
    key: str            # "binance-perp" | "binance-spot"
    exchange: str       # lakeapi exchange partition value
    symbol: str         # lakeapi symbol partition value
    feeds: tuple[str, ...]

INSTRUMENTS = {
    "binance-perp": Instrument("binance-perp", "BINANCE_FUTURES", "BTC-USDT-PERP",
                               ("book_delta_v2", "trades", "funding",
                                "open_interest", "liquidations")),
    "binance-spot": Instrument("binance-spot", "BINANCE", "BTC-USDT",
                               ("book_delta_v2", "trades")),
}

# Per-feed handling class — drives which stage-2 path runs and error tolerance.
FEED_KIND = {
    "book_delta_v2": "delta",    # → stage-2 seed/reseed → top-K L2 (label ≠ the `book` seed product)
    "trades":        "trades",   # → normalize (origin_time-sorted); Binance already sorted (§5b)
    "funding":       "scalar",   # → normalize; expected 8-hourly cadence — confirm in Phase 1 (Risk Q6)
    "open_interest": "scalar",   # → normalize; expected periodic snapshots — confirm in Phase 1 (Risk Q6)
    "liquidations":  "events",   # → normalize; SPARSE — missing/empty files are OK (Risk Q2)
}

# Seed-input product (NOT one of the scoped output feeds): Lake's `book` 20-level snapshot product
# is downloaded per instrument SOLELY to seed book_delta_v2 reconstruction (Requirement 5), because
# book_delta_v2 starts mid-stream with no per-day snapshot (docs §4.1). It is consumed by Stage 2,
# never emitted as a processed output. Mirrors the Coinbase reference `run_coinbase_quality_map.py`
# (LAKE_PRODUCTS = ("book_delta_v2", "book")).
SEED_PRODUCT = "book"            # → stage-1 downloads to data/raw/lake/book/...; stage-2 seed source
```

`--feeds` defaults to all feeds valid for the chosen instrument; an invalid `(instrument, feed)` pair (e.g. `funding` on spot) is rejected before any vendor call. **Whenever `book_delta_v2` is selected, Stage 1 also pulls the `book` `SEED_PRODUCT` for that instrument** (it is the seed source Stage 2 reads locally); its bytes count toward the quota estimate (Requirement 7).

---

## Requirement 2 — Storage layout

Partition keys everywhere: **`exchange`, `symbol`, `feed`, `date`** (Hive-style, matching the Lake bucket layout `market-data/cryptofeed/{table}/exchange={E}/symbol={S}/dt={date}/` and `download_coinapi.py`'s `limitbook_full/exchange=/symbol=/dt=/`).

```
# ── vendor cache (transient, git-ignored) ───────────────────────────────────
.lake_cache/                                    # lakeapi botocache (compressed vendor parquet)

# ── normalized raw store (persisted, quota-motivated, git-ignored data/raw/) ─
data/raw/lake/{feed}/exchange={E}/symbol={S}/dt={YYYY-MM-DD}/data.parquet
data/raw/lake/book/exchange={E}/symbol={S}/dt={YYYY-MM-DD}/data.parquet    # `book` 20-level snapshot — SEED INPUT ONLY (Requirement 5); never an emitted output
data/raw/lake/_manifest.jsonl                   # one JSON line per (feed|book,E,S,dt) written

# ── processed / reconstructed outputs (persisted, git-ignored data/processed/)─
data/processed/topk_l2/exchange={E}/symbol={S}/dt={YYYY-MM-DD}/data.parquet    # book_delta_v2 recon
data/processed/trades/exchange={E}/symbol={S}/dt={YYYY-MM-DD}/data.parquet
data/processed/funding/exchange={E}/symbol={S}/dt={YYYY-MM-DD}/data.parquet
data/processed/open_interest/exchange={E}/symbol={S}/dt={YYYY-MM-DD}/data.parquet
data/processed/liquidations/exchange={E}/symbol={S}/dt={YYYY-MM-DD}/data.parquet
data/processed/_manifest.jsonl                  # one JSON line per (output,E,S,dt) written

# ── run reports & staging plans (git-ignored) ───────────────────────────────
data/reports/binance_download/<run-id>.json     # per-run summary (counts, GB, used_data, timings)
data/tmp/binance_download_batches/              # batch_NNN_days.txt + manifest.json (staging)
```

- **raw/cache:** `.lake_cache/` is lakeapi's own cache — transient, never committed, cleared if keys change (docs §2.1). The downloader may instead read partitions directly via `pyarrow.fs.S3FileSystem` (see Requirement 3); either way nothing is decompressed to a `.csv`/`.csv.gz` intermediate.
- **normalized output:** `data/raw/lake/{feed}/...` — lossless re-partition of the vendor parquet in *our* schema (Requirement 6), ZSTD-compressed, one `data.parquet` per partition. We persist this because quota is a one-shot-per-window budget: download once, reconstruct locally as many times as needed. The `book` `SEED_PRODUCT` is stored the same way under `data/raw/lake/book/...` (seed input for Stage-2 reconstruction, never emitted downstream).
- **processed/reconstructed output:** `data/processed/...` — top-K L2 frames + normalized scalar/event tables, GB-scale.
- **manifest paths:** `_manifest.jsonl` at each store root (`data/raw/lake/`, `data/processed/`), append-only, one record per written partition (see Requirement 6 for record shape). Plus the per-run report and the batch-plan manifest.
- **ignored/generated files:** already covered by `.gitignore` — `data/raw/`, `data/processed/`, `data/reports/`, `data/tmp/`, `.lake_cache/`, `*.parquet`, `*.csv.gz`. **Do not** add any of these to git; **do not** commit fixtures larger than the tiny synthetic ones under `tests/fixtures/`.

---

## Requirement 3 — Staged download workflow

Stage 1 is the only quota-consuming stage. Behaviors:

- **Dry-run (`--dry-run`):** metadata-only. Calls `lakeapi.list_data(...)` per `(feed, E, S)` (no parquet download), prints the resolved day set, per-feed presence/gaps, and the **estimated GB** from measured per-day sizes (Requirement 7). Writes the plan to the run report but performs **zero** parquet transfer. Note: `list_data` is still a **live Lake metadata call** (subscriber credentials), so `--dry-run` is **approval-gated** like all vendor I/O — it just transfers no parquet and consumes ~no quota. The only fully offline planning entry point is `scripts/plan_lake_binance_batches.py`.
- **Manifest planning:** the day list × feeds × estimated GB is the plan. `--manifest PATH` overrides the default `_manifest.jsonl` location (per-day/feed state lives here). The separate `scripts/plan_lake_binance_batches.py` stages a long range into quota-window batches (deterministic, planning-only, mirrors the Coinbase batch planner).
- **Quota-budgeted batches:** each run estimates its request; refuses a broad pull (> `--max-gb`, default 5 GB) unless `--allow-broad`, and **always** refuses a pull that would breach the 300 GB/month headroom regardless of `--allow-broad` (mirrors `run_coinbase_quality_map.py`). Run at most one batch per monthly quota window.
- **Resumability / idempotency:** a `(feed, E, S, dt)` whose final `data.parquet` exists is skipped (unless `--overwrite`). Writes are atomic: stream to `data.parquet.tmp`, `os.replace()` on success. `--resume` (default on for a range) reruns only missing/errored units. Stale `*.tmp` from an interrupted run are cleaned at startup (mirrors `download_coinapi.py:cleanup_tmp`).
- **Per-day / per-instrument state:** every completed or failed unit appends a manifest record (`status ∈ {ok, skip, missing, error}`), so a rerun reads the manifest + on-disk parquet to know exactly what remains.
- **Retry / backoff:** transient S3/botocore errors (`SlowDown`, `RequestTimeout`, connection resets) retry with capped exponential backoff + jitter (e.g. 5 tries, base 1 s, cap 60 s). A **quota/credit** error is a hard stop (exit 2), never retried. Retries are per-unit so one bad partition never re-pulls a completed one.
- **Partial failure handling:** a unit that errors after retries is logged (`status: "error"`, exception summary), the run continues to the next unit, and the process exits **3** (partial) so orchestration can rerun with `--resume`. A `missing` partition (no vendor file) is not an error for `liquidations` (sparse; Risk Q2) but *is* recorded.
- **No decompressed raw intermediates:** read the vendor parquet as compressed row-group batches (`pyarrow.parquet.ParquetFile(...).iter_batches(...)` over an `S3FileSystem` handle, or over the lakeapi-cached object) and write our ZSTD Parquet via `pyarrow.parquet.ParquetWriter` a batch at a time — the CoinAPI streaming shape (`download_coinapi.py:write_parquet`). Never `to_csv`, never materialize a whole `book_delta_v2` day (109 M rows) in RAM.

---

## Requirement 4 — Crypto Lake access

- **Region `eu-west-1`.** Build the session with a **Lake-only** env loader — the `scripts/verify_book_delta_v2.py::lake_session()` pattern (reads `aws_access_key_id`/`aws_secret_access_key`/`region` straight from `.env`), **not** `ingest/verify_lake.py::lake_session()` / `ingest/_common.load_env`, which `SystemExit`s unless `COINAPI_KEY` is set (`ingest/_common.py:41-42`). This downloader is Lake-only and must **not** require CoinAPI creds, so `ingest/lake_binance.py` provides its own stdlib `lake_session()` (AWS keys from `.env`, no `COINAPI_KEY`). Construct an explicit `boto3.Session(aws_access_key_id=…, aws_secret_access_key=…, region_name=env.get("region","eu-west-1"))` — **not** the default `~/.aws` chain (a different account → `AccessDenied`, docs §2.1). Pass the session into every `lakeapi` call (`boto3_session=sess`); for direct `pyarrow.fs.S3FileSystem`, pass the same keys + `region="eu-west-1"`.
- **`origin_time` first, documented `received_time` fallback.** `recon.ingest.shared_engine_time_col(*dfs)` / `is_populated` (>99%) is only a **coarse selector**; the hard per-day gate before recon is **100% populated**, because `recon` then calls `_require_populated`, which raises on *any* ≤0/NaT row (`recon/ingest.py:83-87`). So a day where `origin_time` is 99.x%-but-not-100% populated must **not** be fed to recon on `origin_time` — it would pass the >99% selector and then crash. Policy: use `origin_time` only when it is **fully populated** for the day; otherwise either (a) fall back to `received_time` for the whole day when *it* is fully populated (Binance Tokyo capture → 4.4 ms median lag, a tight proxy), or (b) drop the few invalid-timestamp rows before reconstruction/passthrough and record the dropped count — either way **record `engine_time_col` (and any dropped-row count) in the manifest**, never silent. `origin_time` is measured 100%-populated for Binance `book_delta_v2` and `trades` (docs §5/§5b), so both normally run on exchange time; population for `funding`/`open_interest`/`liquidations` is **not yet measured** (Risk Q9) — the Phase-1 probe measures it, and the same fallback applies to those passthrough tables too. **The resolver is joint, not per-frame:** for seeded reconstruction the delta frame and the `book` seed frame must resolve to the **same** engine-time column (via `recon.ingest.shared_engine_time_col(*dfs)` across all frames handed to recon) — selecting per-frame could put deltas on `received_time` and the seed on `origin_time`, mixing exchange and capture clocks, which reorders seed/reseed events relative to deltas and can inject lookahead or delay reseeds. `recon.reseed.snapshots_from_lake_book_df` itself `_require_populated`s that shared column on the `book` frame, so both frames must be fully populated on it.
- **Metadata/schema probe before broad pulls.** Phase 1 uses `lakeapi.list_data(...)` (metadata, no download) for coverage/gaps and a **bounded** schema probe: fetch one small slice per feed (a ≤2-minute window like `scripts/verify_book_delta_v2.py`, or `list_data` + a single-row-group read) to record the exact columns/dtypes of `funding`/`open_interest`/`liquidations` (whose schemas are not yet measured — Risk Q6) into `tests/fixtures/` schema snapshots. The probe is never a broad scan.
- **Avoid broad CI scans.** Tests never call `lakeapi`/`boto3`/`pyarrow.fs` — all vendor entry points are behind `if __name__ == "__main__"` or injected clients, and unit tests inject a fake lister/reader (AGENTS.md "Avoid broad scans over raw data in CI"). `ingest/lake_binance.py` must import with no vendor dependency.
- **Usage telemetry, not a hard stop.** Print `lakeapi.used_data(sess)` before and after each run and record it in the report. The counter **lags** (docs §5a-QualityMap measured it stale for >60 min), so it is *not* the gate: the gate is our own estimate from measured per-day sizes vs the 300 GB cap − headroom. If `used_data` is unreadable, fail safe (exit 2) rather than proceed blind.

> **Wire-bytes caveat (load-bearing).** lakeapi 0.22.3 `_download_one` GETs each partition object **whole**; a `columns=` projection is applied *after* download (docs §5a-QualityMap). So column projection saves RAM, **not** quota. Quota estimates must use whole-object sizes; the direct `pyarrow.fs` path has the same property unless we push down row-group/column selection at read (a Phase-5 optimization, not assumed here).

---

## Requirement 5 — Reconstruction

Stage 2 reuses the existing engine unchanged; this plan only wires Binance instruments to it.

- **Native engine per day/instrument.** Resolve the engine with `recon.native.resolve_engine(requested, exchange=…, symbol=…)` **before any load**: `--engine native` requires the extension *and* a verified tick scale, else the caller aborts (never silent fallback); `--engine auto` uses native when available+verified, else the Python oracle; `--engine python` always uses `recon.reseed.reconstruct_lake_l2_at_samples_seeded`.
- **Top-K output contract.** Identical to the Coinbase path (`recon/native.py::_assemble`) — **same column order**: `mid, microprice`, then `bid_{i}_price, bid_{i}_size, ask_{i}_price, ask_{i}_size` for `i in 0..k-1`, then `sample_ts` **last** (default `k=10`). Preserve this exact order so DataFrame-equality against the Python/native oracle holds (`_assemble` appends `sample_ts` last, `recon/native.py:266-274`) and the processed schema matches the Coinbase top-K contract. Emitted level prices are the **source floats**, never `tick/scale` round-trips. `meta` carries `seed_accepted, seed_ts, seed_reason, reseed_count, reseed_ts, reseed_blocked_invalid_snapshot, snapshot_reason_codes, crossed_rate, crossed_samples, missing_book_fraction, thin_depth_fraction, crossed_duration_s, policy` — written into the processed manifest per day.
- **Per-day independent parallelism.** `(instrument, day)` jobs are independent once the day-boundary seed policy is explicit (docs/data.md §7: "process events sequentially … parallel per (day, instrument)"; the "day-level process parallelism is the main wall-clock lever" framing is in `docs/superpowers/plans/2026-07-01-native-recon-engine.md` §Follow-on architecture). `--jobs N` fans out by `(instrument, day)` only — never split a single book stream (it is stateful/sequential). Bound `N` by RAM (Requirement 7).
- **Exchange-time ordering.** Deltas replay in `(origin_time, sequence_number, original_row_index)` order (`np.lexsort((seq, ts))` + stable row-index tiebreak), exactly as `reconstruct_lake_l2_at_samples_seeded` and the native core already do. Samples are "as of" `sample_ts` (`event_ts ≤ sample_ts`); `size==0` removes a level; sizes are absolute. **Verify Binance `sequence_number` semantics** (Risk Q5) — Coinbase duplicates it per-row; if Binance is per-event monotonic the tiebreak still holds but gap-detection interpretation differs.
- **Seed / reseed policy.** `book_delta_v2` starts mid-stream (no per-day snapshot block, docs §4.1) for Binance too, so seeding is required. Seed from the **locally-downloaded** Lake `book` 20-level snapshot product (`data/raw/lake/book/...`, the `SEED_PRODUCT` Stage 1 pulls alongside `book_delta_v2`, so Stage 2 stays quota-free) via `recon.reseed.snapshots_from_lake_book_df(...)` + `classify_snapshot(...)` (two-sided, finite/positive, ≥N levels, uncrossed, sane spread); reseed when the reconstructed book stays crossed ≥ `reseed_after_crossed_s`; reseed events apply at their own ts (no look-ahead); residual crossed samples stay visible in metrics. **Validate the Binance `book` seed source is not itself crossed** on sampled days before trusting it (the Coinbase `book` product was crossed 8–37% on several days — docs §5a-QualityMap; Binance coverage/quality is higher but this is unmeasured → Risk Q3).
- **Missing snapshots / cold starts.** If no valid seed is accepted, the day cold-starts and is flagged (`seed_accepted=False`); the quality-map-style classification treats such a day as inconclusive rather than silently emitting a bad book. Never carry book state across a partition-day gap.
- **Validation against the Python oracle on small fixtures.** `tests/test_binance_recon_conformance.py` builds a tiny synthetic Binance-shaped `book_delta_v2` frame (`side_is_bid` bool, absolute sizes, `price_scale=10`) and asserts native `(frame, meta)` equals Python on the boundary cases already covered for Coinbase in `tests/test_native_recon.py` (valid seed, stranded→reseed, same-ts delta/snapshot order, no-valid-seed, `frame_out=False` metrics). No vendor access.

---

## Requirement 6 — Downstream outputs

Each output is a normalized internal contract (AGENTS.md "Downstream code should consume normalized internal contracts"), stamped with a schema version.

- **Bar-ready top-K L2** (`data/processed/topk_l2/...`): the top-K frame above, sampled at a caller-provided grid. For Phase-2/3 validation use a fixed 1 s grid (the quality-map grid); Phase 4 swaps in the bar-clock sample times from the **future** `bars/` module (not yet in the repo — Phase-4 work) — the book-at-bar snapshot. The bar builder is responsible for the strict-`<` book-at-trade rule (E0.1 replay-equivalence, spec §5.3) — the recon sampler exposes "as-of `sample_ts`"; the bar builder chooses pre-trade sample times.
- **Trades stream** (`data/processed/trades/...`): normalized `origin_time, received_time, price, quantity, side, trade_id`. Binance trades are already `origin_time`-sorted with unique monotonic `trade_id` (§5b), so this is a validated passthrough; still assert sortedness and re-sort defensively (the shared clock sorts by `origin_time`).
- **Funding / OI / liquidations normalized tables** (`data/processed/{funding,open_interest,liquidations}/...`): schema fixed after the Phase-1 probe (Risk Q6). **Expected (Binance domain default, to be confirmed by the probe)** — funding: `origin_time, received_time, funding_rate` (+ `next_funding_time` if present), ~8-hourly; open_interest: `origin_time, received_time, open_interest`, periodic; liquidations: `origin_time, received_time, price, quantity, side`, sparse/event-driven. Resolve columns with the `_pick`-style alias approach (`recon.ingest`) and **fail loudly on drift** — never silently mis-column.
- **Source manifests:** every written partition appends one record. Shape:

```json
{"feed": "book_delta_v2", "exchange": "BINANCE_FUTURES", "symbol": "BTC-USDT-PERP",
 "dt": "2026-04-01", "status": "ok", "engine_time_col": "origin_time",
 "src_bytes": 601621234, "rows": 109300000, "out_bytes": 573800000,
 "sha256": "…", "schema_version": "lake_binance/1", "secs": 812.4,
 "ts": "2026-07-02T12:00:00+00:00"}
```

- **Row counts / checksums / schema versions:** `rows` and `sha256` (of the final `data.parquet` bytes) go in the **manifest** (a sidecar record) — **not** in the file's own Parquet key-value metadata, since embedding the hash would change the very bytes being hashed; the Parquet KV metadata instead carries `schema_version` + `rows` (both independent of the byte hash). The schema version is a **per-store/per-output** constant — `RAW_SCHEMA_VERSION = "lake_binance/1"` for the normalized raw store and a `PROCESSED_SCHEMA_VERSION` map (`{"topk_l2": "topk_l2/1", "trades": "trades/1", …}`) for each processed output — bumped on any schema change. Reconciling manifest `rows`/`sha256` against re-read files is a cheap audit (AGENTS.md testing rules).

---

## Requirement 7 — Resource constraints

Per-day sizes (docs §6 rows are **measured**, BTC 2026-04-01; the `book` seed product and the Binance-only totals are **derived**):

| Feed / instrument | MB/day | rows/day |
|---|---|---|
| Binance perp `book_delta_v2` | 573.8 (§6) | **109.3 M** |
| Binance perp `trades`+`funding`+`OI`+`liq` | ~33 (§6, combined) | — |
| Binance spot `book_delta_v2` + `trades` | 261 (§6, combined) | — |
| `book` seed product (perp + spot, seed input only) | ~360 (derived: 2 × ~180; Coinbase ref ~0.18 GB/day, docs §5a-QualityMap) | — |
| **Binance-only, output feeds + `book` seed** | **~1.23 GB/day** (derived: all-feeds 1.17 §6 − Coinbase 0.30 + ~0.36 seed) | — |

- **109 M rows/day (perp `book_delta_v2`) is the binding row count.** Stage 1 streams row-groups (never materializes the day). Stage 2 recon needs columnar arrays (`ts:int64, seq:int64, price:f64, size:f64, side:bool` ≈ 3.6 GB for 109 M rows) — acceptable within 32 GB for **one** day at a time; a streaming Arrow row-group reader into the native core is the documented follow-on (`docs/superpowers/plans/2026-07-01-native-recon-engine.md` §Follow-on architecture: "streaming Arrow/parquet read is the next bottleneck after replay") and is the required path before high perp `--jobs`.
- **32 GB RAM is the binding constraint.** Bound `--jobs`: perp `book_delta_v2` days peak ~4–6 GB each → keep `N ≤ 4` for perp book recon; lighter feeds (trades/funding/OI/liq, spot) can use higher `N`. The downloader (stage 1) is I/O-bound and streams, so its `--jobs` is bounded by S3 throughput, not RAM.
- **2 TB SSD:** Binance-only raw ≈ 1.23 GB/day (output feeds + `book` seed) → ~671 GB for 18 mo (547 d); processed top-K collapses to GB-scale (docs §7: "≤60 GB processed"). Comfortable on 2 TB. Keep raw compressed; **never** decompress `book_delta_v2` to disk (docs §7).
- **Stream/chunk per day; avoid multi-day raw loads.** Both stages operate one `(instrument, day)` at a time. Never `load_data` a multi-day range into one DataFrame.
- **Parallelism by day/instrument only.** A single book stream is sequential; `(instrument, day)` jobs are the only parallel axis. Quota-aware: parallel *download* still shares the one monthly budget — the gate estimates the whole batch, not per-job.

**Quota / storage budget (derived from docs §6):** Binance-only (output feeds + `book` seed) 18 mo ≈ **~671 GB** (≈ 1.23 GB/day × 547 d) > two 300 GB/month windows → stage across **~3** quota windows (matching docs §6's all-feed staging), or use a higher plan. The batch planner enforces this.

---

## Requirement 8 — CLI / API shape (for the future implementation)

```
ingest/download_lake_binance.py                              # Stage 1
  --start YYYY-MM-DD          inclusive
  --end   YYYY-MM-DD          inclusive
  --exchange BINANCE_FUTURES  (or BINANCE)      # or use --instrument {binance-perp,binance-spot}
  --symbol   BTC-USDT-PERP    (or BTC-USDT)
  --feeds    book_delta_v2,trades,funding,open_interest,liquidations   # default: all valid; book_delta_v2 also pulls the `book` seed product
  --out      data/raw/lake    normalized store root
  --manifest PATH             override _manifest.jsonl location (per-day/feed state)
  --dry-run                   metadata + plan + GB estimate ONLY; zero parquet transfer
  --resume                    rerun only missing/errored units (default for a range)
  --overwrite                 re-download existing partitions
  --max-gb   5.0              refuse an estimated pull above this unless --allow-broad
  --allow-broad               permit a broad pull (still capped by the 300 GB/mo headroom gate)
  --engine   auto|native|python   reserved; download is engine-agnostic (recon uses it)
  --jobs     N                parallel by (instrument,day) only

scripts/run_binance_recon.py                                 # Stage 2 (local, quota-free)
  --start/--end/--instrument/--feeds/--k 10/--grid-s 1.0/--engine auto|native|python/--jobs N
  --raw  data/raw/lake        --out data/processed

scripts/plan_lake_binance_batches.py                         # staging (planning only, no vendor I/O)
  --start/--end  or  --calendar data/usable_calendar.json
  --max-gb-per-batch 250  --gb-per-day 1.23  --out-dir data/tmp/binance_download_batches  --dry-run
```

**Exit codes** (mirror the repo's small-int convention — `_common.BACKFILL_GATE_EXIT=4`, `plan_*_batches.CALENDAR_ERROR_EXIT=2`):

| Code | Meaning |
|---|---|
| 0 | all requested `(feed,day)` units ok or skipped |
| 2 | setup error (bad args, missing keys, unreadable calendar/`used_data`) **or a runtime vendor quota/credit hard-stop** — fail-safe, do not proceed |
| 3 | completed with ≥1 errored unit (partial) — rerun with `--resume` |
| 4 | broad-pull gate — estimated GB over `--max-gb` without `--allow-broad`; **or over the 300 GB/month quota headroom, which exits 4 regardless of `--allow-broad`** (matches Requirement 3 and the `check_broad_gate` test) |

**Logging / report artifacts:** human-readable per-unit lines to stdout (`{dt}  OK  {rows:,} rows  {src}MB→{out}MB  {secs}s`, like `download_coinapi.py`); a machine report `data/reports/binance_download/<run-id>.json` with `{args, days, feeds, est_gb, used_data_before/after, counts{ok,skip,missing,error}, total_rows, s3_requests, per_unit[]}`. `--dry-run` writes the same report with `transferred=0`.

---

## Requirement 9 — Tests (for the future implementation)

All synthetic; **no live vendor calls in tests** (inject fakes for any lister/reader). Follow TDD: write the failing test first, run it red, implement minimally, run it green, commit.

1. **Synthetic manifest planning** (`test_lake_binance_manifest.py`): given a fake `list_data` returning a known day set, `--dry-run` produces the expected day list, per-feed presence/gaps, and GB estimate; no writes to `data/raw`.
2. **Partition path generation** (`test_lake_binance_paths.py`): `raw_parquet_path(...)` → `.../raw/lake/{feed}/exchange={E}/symbol={S}/dt={date}/data.parquet` (and the `book` seed under `.../raw/lake/book/...`); `processed_parquet_path(...)` → `.../processed/{output}/exchange={E}/symbol={S}/dt={date}/data.parquet` (output ∈ {topk_l2,trades,funding,open_interest,liquidations}); invalid `(instrument,feed)` pairs raise.
3. **Resume / idempotency** (`test_lake_binance_manifest.py`): a `(feed,E,S,dt)` with an existing final parquet is skipped; an interrupted `.tmp` is cleaned; a second run with `--resume` re-does only missing/errored units; manifest records are append-only and de-dup on `(feed,E,S,dt)`.
4. **Schema validation** (`test_lake_binance_schema.py`): a synthetic frame with the documented columns normalizes cleanly and stamps `schema_version` — covering `book_delta_v2`, `trades`, the scalar/event feeds, **and the `book` 20-level snapshot seed product** (`bid_{i}_price`/`bid_{i}_size`/`ask_{i}_price`/`ask_{i}_size` — the exact columns `recon.reseed.snapshots_from_lake_book_df` reads, `recon/reseed.py:100,114-117`); a drifted frame (renamed/missing column, wrong dtype, unknown `side` value) raises a clear error listing seen columns (reuse the `recon.ingest._pick`/`_side_str` fail-loud pattern).
5. **Quota gate** (`test_lake_binance_gate.py`): estimate = Σ per-feed per-day GB × days; > `--max-gb` without `--allow-broad` → exit 4; > (300 − headroom) always → exit 4; unreadable `used_data` → exit 2; a one-day pull is allowed.
6. **Batch planner** (`test_plan_lake_binance_batches.py`): deterministic byte-identical batch files for a fixed range+budget; every batch ≤ budget and ≤ the runner-executable cap; `--dry-run` writes nothing.
7. **Native/Python conformance on tiny synthetic fixtures** (`test_binance_recon_conformance.py`): Binance-shaped `book_delta_v2` (`price_scale=10`) — native `(frame,meta)` equals Python on valid-seed / stranded→reseed / same-ts-order / no-seed / `frame_out=False`; **skipped** when `recon_native` is absent (`native_available()` is False), so `pytest -q` passes without Rust.
8. **Engine-time selection + `origin_time` fallback** (`test_lake_binance_engine_time.py`): exercises the Requirement 4 policy so an implementation cannot satisfy the plan while Stage 2 still crashes on a partial day — (a) 100% `origin_time` → selected, recon-ready; (b) **99.x%-but-not-100%** `origin_time` with fully-populated `received_time` → falls back to `received_time` for the whole day and stamps `engine_time_col`; (c) neither clock fully populated → the invalid-timestamp rows are dropped from the returned frame and the drop count is recorded; **(d) joint clock** — a delta frame and the `book` seed frame resolve to the **same** column (a partial-`origin_time` delta frame forces the seed frame to `received_time` too, never a mixed exchange/capture axis). In every case, assert every returned cleaned frame (the exact frames handed to recon) passes `recon.ingest._require_populated` on the **one** shared column — recon never receives a partially-populated or mixed-clock engine-time axis.

---

## Requirement 10 — Rollout phases

- **Phase 1 — metadata/schema probes + dry-run manifests.** The only offline/unattended step is `scripts/plan_lake_binance_batches.py` (pure planning, reads a local calendar — **no vendor I/O**). Everything else here is a **live Crypto Lake call and is approval-gated** (AGENTS.md — live vendor calls only when the user asks), even the cheap ones: `download_lake_binance.py --dry-run` still hits `lakeapi.list_data(...)` per feed with live subscriber credentials (metadata only — no parquet transfer, ~no quota, no `COINAPI_KEY`), and the bounded coverage/gap + `origin_time`/schema probe (a `download_lake_binance.py --probe` mode, or a small `ingest/lake_binance` probe) covers the perp/spot feeds **and** the `book` seed product plus `funding`/`open_interest`/`liquidations` schema + `origin_time` population (Risk Q9) and tick-scale confirmation (Risk Q1) — all through the **Lake-only** session path (Requirement 4). **Do not run `ingest/verify_lake.py`/`verify_lake2.py` as-is** — they require `COINAPI_KEY` via `ingest/_common.load_env` (`_common.py:41-42`) and would fail before any Lake call for a Lake-only user; reuse their coverage logic only if first switched to the Lake-only loader. **No archive transfer.** Deliverable: schema fixtures, confirmed identifiers/ticks/`origin_time`, dry-run report.
- **Phase 2 — one-day cheap validation.** Download **one** present day for perp + spot all feeds + the `book` seed (~1.23 GB, one quota-cheap pull), run stage-2 recon on it, and eyeball top-K sanity (uncrossed after seed, plausible depth) + trades/funding/OI/liq normalization. Record `used_data` before/after. Deliverable: one day end-to-end + report.
- **Phase 3 — staged historical pull.** Use `plan_lake_binance_batches.py` to stage the 12–24 mo span across quota windows; run **one batch per window**, re-checking `used_data` first; recon each batch locally. Deliverable: the normalized raw + processed stores, staged.
- **Phase 4 — bar/feature integration.** Feed the top-K L2 (sampled at bar-clock times, strict-`<` book-at-trade) + trades/funding/OI/liq into the **future** `bars/` module + feature-manifest pipeline (docs/feature-manifest.md; `bars/` is not yet in the repo — Phase-4 work). Deliverable: bar-ready feature matrix seam.
- **Phase 5 — monitoring & re-run support.** Re-run tail days as new data lands; alert on coverage drops, seed-rejection spikes, `used_data` nearing the cap, or schema drift. Deliverable: a small monitor over the manifests/reports.

---

## Requirement 11 — Risks & open questions

- **Q1 — Binance tick scales unverified (blocks native).** Native mode needs a verified `(exchange,symbol)→price_scale` where *every* price is an exact multiple of the tick (docs/native-recon.md). Expected: perp tick $0.10 → `price_scale=10`, spot tick $0.01 → `price_scale=100`. **Must be confirmed** against real data (min price increment across a sampled day) before adding to `recon/native.py::_TICK_SCALE`. Until then perp/spot recon runs Python (or `--engine auto` falls back). Coinbase's registry entry is the only verified one today.
- **Q2 — Missing/empty liquidations files.** Coverage is ~66%, sparse and event-driven (docs §5) — genuinely no liquidations on quiet days vs a missing file is unresolved (docs §10 open item). Treat `missing` liquidations as non-fatal (`sparse_ok`), record it, and confirm sparsity is real (empty file vs absent) in Phase 1 rather than masking a data hole.
- **Q3 — Binance `book` seed source may itself be crossed.** The Coinbase `book` product was crossed 8–37% on several days, routing them to fill (docs §5a-QualityMap). Binance quality is higher (96.4% coverage, 4.4 ms lag) but the seed-source crossed-fraction is **unmeasured** for Binance — measure it on sampled days before trusting `book`-seeded reconstruction. There is no CoinAPI backfill for Binance, so a crossed/rejected seed has **no fallback vendor**: the *implemented* behavior (`recon/reseed.py`) is to cold-start, set `seed_accepted=False`, and flag the day inconclusive (never emit a silently-bad book). A delta-cluster-derived seed (the "first dense multi-level cluster" option noted in docs §5a-Recon) is **not implemented** and would be new work if the crossed-`book` rate proves material for Binance.
- **Q4 — Native coverage/streaming dependency.** Multi-day and 109 M-row/day perp recon depend on the native engine and, ultimately, a streaming Arrow reader (`docs/superpowers/plans/2026-07-01-native-recon-engine.md` §Follow-on architecture). The Python oracle cannot support the full span (it was CPU-bound after hours on 34 M-row Coinbase days). If the streaming reader slips, perp recon `--jobs` must stay low to fit RAM, slowing Phase 3.
- **Q5 — Binance `sequence_number` semantics.** Binance native feeds use `U/u/pu` update-ID continuity (spec §5.3 / E0.1); Lake's `book_delta_v2` exposes `sequence_number`. Confirm whether it is per-event monotonic (unlike Coinbase's per-row duplicated seq) so the `(ts, seq, row_index)` tiebreak and any gap interpretation are correct. Affects reseed-gap detection, not the crossed-book reseed trigger (which is observable-state-driven).
- **Q6 — funding / OI / liquidations schema + cadence.** Their exact Lake schemas are not yet measured (only `book_delta_v2`/`trades` are, docs §4.1). Funding is 8-hourly, OI periodic, liquidations event-driven — **cadence differs from the per-event book**, so these are conditioner tables (spec §6 "conditioners, not primary"), not book streams; the Phase-1 probe fixes their columns and the schema test pins them.
- **Q7 — Quota/budget staging.** 18 mo Binance-only (output feeds + `book` seed) ≈ **~671 GB** (≈ 1.23 GB/day × 547 d) > two 300 GB windows; the archive pull must stage across **~3** windows with `used_data` re-checked each time (consistent with Requirement 7). The `used_data` counter lags, so the plan's own GB estimate is the gate, not the vendor counter.
- **Q8 — spot/perp schema drift.** Spot (`BINANCE`/`BTC-USDT`) and perp (`BINANCE_FUTURES`/`BTC-USDT-PERP`) `book_delta_v2`/`trades` are assumed schema-identical but this is unverified per-column; the alias-resolving normalizer fails loudly on any drift rather than mis-columning, and Phase 1 probes both.
- **Q9 — `origin_time` coverage for funding / OI / liquidations.** `origin_time` is measured 100%-populated only for `book_delta_v2` and `trades` (docs §5/§5b). Its population in the `funding`/`open_interest`/`liquidations` conditioner tables is **unmeasured** — the Phase-1 probe must measure it, and the passthrough normalizer applies the `origin_time`→`received_time` fallback (recording `engine_time_col` in the manifest) for these tables just as for book/trades, never silently.

---

## Implementation Tasks

Bite-sized, TDD, frequent commits. Task 1 and the test blocks below show concrete code; Tasks 2–7 give the failing-test intent plus the **exact existing pattern to mirror** (`file:lines`) rather than repeating large code bodies — the executor writes the test first, runs it red, implements against the cited pattern, runs it green, and commits. Reuse existing patterns — do not restructure unrelated files.

### Task 1: Feed/instrument registry + partition paths (pure)

**Files:** Create `ingest/lake_binance.py`; Test `tests/test_lake_binance_paths.py`.

- [ ] **Step 1 — Write the failing test.**

```python
# tests/test_lake_binance_paths.py
import pytest
from ingest import lake_binance as lb

def test_instruments_match_verified_identifiers():
    assert lb.INSTRUMENTS["binance-perp"].exchange == "BINANCE_FUTURES"
    assert lb.INSTRUMENTS["binance-perp"].symbol == "BTC-USDT-PERP"
    assert lb.INSTRUMENTS["binance-spot"].exchange == "BINANCE"
    assert lb.INSTRUMENTS["binance-spot"].symbol == "BTC-USDT"
    assert lb.INSTRUMENTS["binance-spot"].feeds == ("book_delta_v2", "trades")

def test_raw_parquet_path_is_hive_partitioned():
    p = lb.raw_parquet_path("data/raw/lake", "book_delta_v2",
                            "BINANCE_FUTURES", "BTC-USDT-PERP", "2026-04-01")
    assert p == ("data/raw/lake/book_delta_v2/exchange=BINANCE_FUTURES/"
                 "symbol=BTC-USDT-PERP/dt=2026-04-01/data.parquet")

def test_processed_path_uses_output_name_not_lake_feed():
    p = lb.processed_parquet_path("data/processed", "topk_l2",
                                  "BINANCE_FUTURES", "BTC-USDT-PERP", "2026-04-01")
    assert p == ("data/processed/topk_l2/exchange=BINANCE_FUTURES/"
                 "symbol=BTC-USDT-PERP/dt=2026-04-01/data.parquet")

def test_invalid_instrument_feed_pair_rejected():
    with pytest.raises(ValueError):
        lb.validate_feed("binance-spot", "funding")   # funding is perp-only
```

- [ ] **Step 2 — Run red.** `.venv/bin/python -m pytest tests/test_lake_binance_paths.py -q` → FAIL (`ModuleNotFoundError: ingest.lake_binance`).
- [ ] **Step 3 — Implement minimally:** the `Instrument` dataclass, `INSTRUMENTS`, `FEEDS`, `FEED_KIND` (as in Requirement 1), plus:

```python
import os

def raw_partition_dir(out_root, feed, exchange, symbol, day_iso):
    return os.path.join(out_root, feed, f"exchange={exchange}",
                        f"symbol={symbol}", f"dt={day_iso}")

def raw_parquet_path(out_root, feed, exchange, symbol, day_iso):
    return os.path.join(raw_partition_dir(out_root, feed, exchange, symbol, day_iso),
                        "data.parquet")

def processed_parquet_path(out_root, output, exchange, symbol, day_iso):
    # out_root e.g. "data/processed"; output ∈ {topk_l2,trades,funding,open_interest,liquidations}
    # NOTE: keyed by OUTPUT NAME (no "lake/" segment) — distinct from the raw store scheme.
    return os.path.join(out_root, output, f"exchange={exchange}",
                        f"symbol={symbol}", f"dt={day_iso}", "data.parquet")

def validate_feed(instrument_key: str, feed: str) -> None:
    inst = INSTRUMENTS[instrument_key]
    if feed not in inst.feeds:
        raise ValueError(f"feed {feed!r} not valid for {instrument_key} "
                         f"(valid: {inst.feeds})")
```

- [ ] **Step 4 — Run green.** Same command → PASS.
- [ ] **Step 5 — Commit.** `git add ingest/lake_binance.py tests/test_lake_binance_paths.py && git commit -m "feat: Binance Lake feed registry + partition paths"`

### Task 2: Manifest append/read + resume state (pure, tmp_path)

**Files:** Modify `ingest/lake_binance.py`; Test `tests/test_lake_binance_manifest.py`.

- [ ] **Step 1 — Failing test:** `manifest_append(root, rec)` writes one JSON line; `manifest_index(root)` returns `{(feed,exchange,symbol,dt): status}`; `is_done(root, ...)` is True iff a final `data.parquet` exists (use `tmp_path`, create an empty file). Assert append-only and that `is_done` is False for a `.tmp`-only partition.
- [ ] **Step 2 — Run red** (`Module/attribute` error).
- [ ] **Step 3 — Implement** `manifest_append`, `manifest_index`, `is_done`, `cleanup_tmp` mirroring `download_coinapi.py:137-155` (atomic append; `.tmp` cleanup walks the tree).
- [ ] **Step 4 — Run green.**
- [ ] **Step 5 — Commit.**

### Task 3: Schema normalization + engine-time selection + fail-loud drift (synthetic frames)

**Files:** Modify `ingest/lake_binance.py` (add `RAW_SCHEMA_VERSION`/`PROCESSED_SCHEMA_VERSION`, `normalize_book_delta_v2`, `normalize_trades`, `normalize_scalar`, and `resolve_engine_time(*dfs) -> (col, fallback_used, dfs_clean, dropped_rows)` — **joint** across every frame handed to recon (deltas + `book` seed): one shared engine-time column, returning the **cleaned** frames the caller feeds to recon, never the originals); Test `tests/test_lake_binance_schema.py`, `tests/test_lake_binance_engine_time.py`.

- [ ] **Step 1 — Failing test:** feed a synthetic pandas frame with the documented `book_delta_v2` columns → normalized frame has the canonical schema and stamps `RAW_SCHEMA_VERSION`; a frame with a renamed/missing column or an unknown `side` value raises `ValueError` naming the seen columns (reuse `recon.ingest._pick`/`_side_str` semantics).
- [ ] **Step 2 — Run red.**
- [ ] **Step 3 — Implement** the normalizers using the alias-resolution pattern from `recon/ingest.py` (do not duplicate — import the `_pick` idea; keep vendor knowledge here at the boundary).
- [ ] **Step 4 — Run green.**
- [ ] **Step 5 — Engine-time failing test** (`tests/test_lake_binance_engine_time.py`): the Requirement 9 item-8 cases — full `origin_time` selected; 99.x% `origin_time` + full `received_time` → whole-day fallback with `engine_time_col` stamped; neither clock full → drop ≤0/NaT rows + record the count; **plus a joint-clock case**: a delta frame with 99.x% `origin_time` + a `book` seed frame with full `origin_time` must resolve **both** to the same fallback column (`received_time`), never `received_time` for deltas and `origin_time` for the seed. In **every** case assert each returned cleaned frame (the exact frames handed to recon) passes `recon.ingest._require_populated` on the **one** chosen column. Run red.
- [ ] **Step 6 — Implement** `resolve_engine_time(*dfs) -> (col, fallback_used, dfs_clean, dropped_rows)` — **joint** across all frames (delta + `book` seed): reuse `recon.ingest.shared_engine_time_col(*dfs)`/`is_populated` as the coarse selector over ALL frames, then enforce the **100%-populated** gate (Requirement 4) on ONE shared column — pick `origin_time` if full in every frame, else `received_time` if full in every frame, else drop the ≤0/NaT rows on the chosen column from each frame; **return the cleaned frames** (`dfs_clean`, aligned to inputs) plus the single chosen column + fallback flag + per-frame drop counts. The caller feeds the cleaned frames (never the originals) to `reconstruct_lake_l2_at_samples_seeded` **and** `snapshots_from_lake_book_df` with the **same** `engine_time_col`, and records it + drop counts in the manifest. Run green.
- [ ] **Step 7 — Commit.**

### Task 4: Quota estimation + broad-pull gate (pure)

**Files:** Modify `ingest/lake_binance.py` (`LAKE_GB_PER_DAY`, `estimate_gb`, `check_broad_gate`); Test `tests/test_lake_binance_gate.py`.

- [ ] **Step 1 — Failing test:**

```python
from ingest import lake_binance as lb
import pytest

def test_estimate_includes_book_seed_when_book_delta_selected():
    # selecting book_delta_v2 also pulls its `book` seed product (0.574 + 0.18 GB/day)
    gb = lb.estimate_gb("binance-perp", ["book_delta_v2"], n_days=10)
    assert 7.3 < gb < 7.8            # 10 × (0.574 + 0.18) GB

def test_broad_gate_blocks_without_allow_broad():
    with pytest.raises(SystemExit) as e:
        lb.check_broad_gate(est_gb=50.0, max_gb=5.0, allow_broad=False,
                            used_gb=0.0, quota_gb=300.0, headroom_gb=10.0)
    assert e.value.code == 4

def test_broad_gate_blocks_over_headroom_even_with_allow_broad():
    with pytest.raises(SystemExit) as e:
        lb.check_broad_gate(est_gb=295.0, max_gb=1e9, allow_broad=True,
                            used_gb=20.0, quota_gb=300.0, headroom_gb=10.0)
    assert e.value.code == 4

def test_one_day_allowed():
    lb.check_broad_gate(est_gb=1.23, max_gb=5.0, allow_broad=False,     # one day, all feeds + book seed
                        used_gb=0.0, quota_gb=300.0, headroom_gb=10.0)   # no raise
```

- [ ] **Step 2 — Run red.**
- [ ] **Step 3 — Implement** `LAKE_GB_PER_DAY` (docs §6 measured + `book` seed ~0.18 GB/day/instrument, docs §5a-QualityMap), `estimate_gb` (adds the `book` seed cost whenever `book_delta_v2` is requested, mirroring `run_coinbase_quality_map.py`'s `LAKE_PRODUCTS=("book_delta_v2","book")`), and `check_broad_gate` raising `SystemExit(4)` on breach (mirror `_common.check_backfill_gate`). `used_gb` unreadable → caller exits 2.
- [ ] **Step 4 — Run green.**
- [ ] **Step 5 — Commit.**

### Task 5: Stage-1 downloader CLI (streaming, resumable)

**Files:** Create `ingest/download_lake_binance.py`; extend `tests/test_lake_binance_manifest.py` with an injected-fake-reader end-to-end test (no vendor).

- [ ] **Step 1 — Failing test:** drive `process_unit(...)` with a fake reader yielding two small pyarrow batches → writes `data.parquet` atomically, appends an `ok` manifest record with `rows`/`sha256`/`schema_version`, and a second call skips (idempotent). **Add a `feed="book"` (`SEED_PRODUCT`) case** with a 20-level snapshot fixture carrying the exact columns `snapshots_from_lake_book_df` parses (`bid_{i}_price`/`bid_{i}_size`/`ask_{i}_price`/`ask_{i}_size`, `recon/reseed.py:114-117`): assert it writes to `data/raw/lake/book/exchange=…/…/data.parquet` and appends its own manifest record — so a downloader that estimates the seed bytes but drops/mishandles the seed product **fails** this test (otherwise recon silently cold-starts and marks every day inconclusive). Assert no network import is required to run the test.
- [ ] **Step 2 — Run red.**
- [ ] **Step 3 — Implement** the CLI: arg parsing (Requirement 8), the **Lake-only** `lake_binance.lake_session()` (Requirement 4 — no `COINAPI_KEY` dependency), `--dry-run` via `lakeapi.list_data`, per-unit streaming normalize→ZSTD Parquet via `ParquetWriter` (batch loop like `download_coinapi.py:write_parquet`), retry/backoff, per-unit try/except → `status`, exit-code logic (0/2/3/4), report JSON. Keep `lakeapi`/`pyarrow` imports inside the vendor path so the pure helpers stay importable.
- [ ] **Step 4 — Run green** + `.venv/bin/python -m py_compile ingest/download_lake_binance.py`.
- [ ] **Step 5 — Commit.**

### Task 6: Batch planner (mirror the Coinbase planner)

**Files:** Create `scripts/plan_lake_binance_batches.py`; Test `tests/test_plan_lake_binance_batches.py`.

- [ ] **Step 1 — Failing test:** a fixed date range + `--max-gb-per-batch`/`--gb-per-day` yields deterministic, byte-identical `batch_NNN_days.txt` + `manifest.json`; every batch estimate ≤ budget; `--dry-run` writes nothing. Model it on `tests/test_plan_quality_map_batches.py`.
- [ ] **Step 2 — Run red.**
- [ ] **Step 3 — Implement** by adapting `scripts/plan_coinbase_quality_map_batches.py` (stdlib-only, `PlanError`, deterministic chunking, manifest with per-batch commands + est GB). Day source: a plain `--start/--end` range or `--calendar data/usable_calendar.json` (Binance-present intersection).
- [ ] **Step 4 — Run green** + `py_compile`.
- [ ] **Step 5 — Commit.**

### Task 7: Stage-2 recon runner + Binance tick scales

**Files:** Create `scripts/run_binance_recon.py`; Modify `recon/native.py` (`_TICK_SCALE`, **only after Q1 verified**); Test `tests/test_binance_recon_conformance.py`.

- [ ] **Step 1 — Failing test:** build a tiny synthetic Binance-shaped `book_delta_v2` frame (`price_scale=10`) + `book`-product snapshots; assert `resolve_engine("native", exchange="BINANCE_FUTURES", symbol="BTC-USDT-PERP")` returns native **iff** the scale is registered, and that native `(frame,meta)` equals Python (`reconstruct_lake_l2_at_samples_seeded`) on valid-seed / stranded→reseed / no-seed / `frame_out=False`. Skip native asserts when `native_available()` is False.
- [ ] **Step 2 — Run red.**
- [ ] **Step 3 — Implement** the runner (read local normalized parquet per `(instrument,day)`; `resolve_engine`; emit top-K L2 + normalized tables; `--jobs` by `(instrument,day)`; processed manifest with `meta`). Add the **verified** Binance ticks to `_TICK_SCALE` (Q1) — if unverified, leave them out and let `auto` fall back (the test asserts the fallback, not the scale).
- [ ] **Step 4 — Run green** + `py_compile`.
- [ ] **Step 5 — Commit.**

### Task 8: Docs cross-links

**Files:** Modify `docs/data.md` §10 (flip the "Binance downloader — not yet built" open item to "planned — see this doc"). Optionally add a one-line pointer to this plan from `docs/native-recon.md` (note: the streaming-reader / day-parallelism follow-on text lives in `docs/superpowers/plans/2026-07-01-native-recon-engine.md` §Follow-on architecture, which native-recon.md links to — cite that; do not assume a follow-on note already exists in native-recon.md). Keep edits minimal; do not restate the plan.

- [ ] **Step 1** — edit, `git diff --check`, commit.

---

## Validation

**This docs branch:**

```bash
git diff --check
git status -sb
```

(No `pytest` — no code added on this branch, per the writing-plans skill and AGENTS.md docs-only rule.)

**Future implementation branch** (per task, plus final):

```bash
.venv/bin/python -m pytest -q tests/test_lake_binance_paths.py tests/test_lake_binance_manifest.py \
    tests/test_lake_binance_schema.py tests/test_lake_binance_engine_time.py tests/test_lake_binance_gate.py \
    tests/test_plan_lake_binance_batches.py tests/test_binance_recon_conformance.py
.venv/bin/python -m py_compile ingest/download_lake_binance.py ingest/lake_binance.py \
    scripts/plan_lake_binance_batches.py scripts/run_binance_recon.py
.venv/bin/python -m pytest -q                 # full suite still green without Rust
git diff --check
```

Optional live validation (Phase 2, **explicit approval + one day only**, record `used_data` before/after):

```bash
.venv/bin/python ingest/download_lake_binance.py --instrument binance-perp --start 2026-04-01 --end 2026-04-01
.venv/bin/python scripts/run_binance_recon.py --instrument binance-perp --start 2026-04-01 --end 2026-04-01 --engine native
```

---

## PR Requirements (implementation branch)

- **Summary** — Binance downloader + recon scope.
- **Scope** — the two instruments/feeds above; explicitly *not* Coinbase/CoinAPI.
- **No vendor/API calls run** — state which cheap local checks ran (unit tests, `py_compile`) and that no live Lake/CoinAPI pull, no bulk download, and no broad quality map ran; CoinAPI backfill stays locked.
- **Validation** — test results with/without the native extension; any Phase-2 one-day run with exact day, `used_data` before/after, and report path.
- **Risks/follow-ups** — the Risks Q1–Q8 above; the streaming Arrow reader follow-on; staged archive still pending.

---

## Review Checklist

- Identifiers match the repo exactly — perp `BINANCE_FUTURES`/`BTC-USDT-PERP` (`verify_lake.py:21-22`), spot `BINANCE`/`BTC-USDT` (`verify_trades_and_calendar.py:98-99`), `liquidations` (`verify_lake.py:70-72`).
- `ingest/lake_binance.py` imports with **no** `boto3`/`lakeapi`/`pyarrow` at module top (CI-safe).
- No test touches a live vendor; native tests skip cleanly when the extension is absent.
- Quota gate runs **before** any Lake load; `used_data` unreadable → fail safe.
- `--engine native` never silently falls back after an explicit request.
- No raw/processed data, reports, caches, `.env`, or secrets committed; `.gitignore` already covers the output roots.
- `origin_time`→`received_time` fallback is recorded in the manifest, never silent.
- Binance tick scales added to `_TICK_SCALE` only after Q1 verification.

---

## Self-review (spec coverage)

Every task requirement maps to a section/task: **1** Scope → Requirement 1 + Task 1. **2** Storage → Requirement 2 + Tasks 1–2. **3** Staged workflow → Requirement 3 + Tasks 2,4,5. **4** Lake access → Requirement 4 + Task 5. **5** Reconstruction → Requirement 5 + Task 7. **6** Downstream outputs → Requirement 6 + Tasks 3,7. **7** Resources → Requirement 7 (informs `--jobs` in Tasks 5,7). **8** CLI/API → Requirement 8 + Tasks 5,6,7. **9** Tests → Requirement 9 + every task's Step 1. **10** Rollout → Requirement 10. **11** Risks → Requirement 11. No `TBD`/`TODO` placeholders; Tasks 2–7 cite the exact source pattern to mirror rather than restating code (intentional — see the Tasks intro). Types/names (`Instrument`, `SEED_PRODUCT`, `raw_parquet_path`, `processed_parquet_path`, `check_broad_gate`, `resolve_engine`, `_TICK_SCALE`, top-K columns) are consistent across tasks.

---

## Execution Handoff

Dedicated branch/worktree from latest `origin/master`:

```bash
cd /home/aaron/jepa-btc-forecasting
git checkout master && git pull --ff-only origin master
scripts/new_claude_worktree.sh binance-downloader
```

Implement task-by-task. If the branch grows large, split after Task 5: PR-1 = downloader (Tasks 1–6), PR-2 = recon runner + tick scales (Task 7). Do **not** run the historical archive pull or unlock any backfill gate in the implementation PR.
