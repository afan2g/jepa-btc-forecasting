# Multi-Day Trade-Feed Validation Breadth — Implementation Plan

> **For agentic workers:** this is a **spec + implementation plan** for the open `docs/data.md`
> §10 item "Trade validation breadth." **This branch ships the doc only** — no code, no vendor
> calls. The follow-up implementation branches (§Implementation Tasks) build
> `ingest/trade_checks.py` (pure, synthetic-tested) + `ingest/validate_trade_feeds.py` (the Lake
> CLI). Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans for
> those follow-ups. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the one-day, three-venue §5b trade-feed validation to **multiple days and market
regimes per venue** (Coinbase BTC-USD, Binance BTC-USDT spot, Binance BTC-USDT-PERP), with a
per-day/per-venue pass/warn/fail report and an explicit gate that the bar builder consumes — because
trades drive the bar clock (spec §5.1), so a mis-validated trade stream mis-times every bar, feature
vector, and label.

**Architecture:** A pure, **source-agnostic** checks module (`ingest/trade_checks.py`: thresholds,
reason-code constants, per-frame metric functions, `classify(...)`, `build_report`/`write_report`,
GB estimation, the Lake quota decision) that validates a *normalized* trade frame regardless of
vendor, plus thin loader/CLI wrappers: `ingest/validate_trade_feeds.py` loads Lake `trades`
partitions via `lakeapi`, and the gated Phase-3b path (§10) normalizes CoinAPI `TRADES` for fill days
— **both call the same checks module unchanged**. This mirrors the repo's established split (pure
`recon/stitch_policy.py` vs. the vendor `scripts/run_coinbase_quality_map.py` runner), so every check
is synthetic-testable with **no vendor I/O in tests**.

**Tech Stack:** Python 3.12, pandas/numpy, `lakeapi` + explicit boto3 session (Crypto Lake,
`eu-west-1`), pytest. No new dependencies. Reads Crypto Lake `trades` only; **CoinAPI is not touched
and the CoinAPI backfill gate stays LOCKED** (Coinbase CoinAPI-fill-day trade validation is a
gated follow-up, §8/§10-rollout).

**Interpreter:** commands use the repo-convention `.venv/bin/python`, run from the repo root. Agent
worktrees have **no `.venv`** — substitute the main checkout's interpreter
`/home/aaron/jepa-btc-forecasting/.venv/bin/python` from the worktree root; the commands are
otherwise unchanged.

**Scope (this branch):** this document + one minimal `docs/data.md` pointer edit (§10). **No new
Python, no tests run, no Crypto Lake / CoinAPI / native / bulk pulls, no full pytest.** No raw data,
reports, parquet/csv.gz, caches, `.env`, or secrets committed.

## Non-Goals

- No live Crypto Lake trade downloads and no CoinAPI pulls on this branch (Phase 2 does a bounded
  Lake run, **ask-first** per `AGENTS.md`).
- No changes to `ingest/verify_trades_and_calendar.py`'s existing one-day check or to the usable
  calendar's structure on this branch (Phase 3 integrates; it does not rewrite §5b's calendar).
- No CoinAPI-sourced fill-day trade validation *executed* here — it is specified as **Phase 3b /
  Task 5** (§10: download CoinAPI `TRADES`, normalize to the Lake schema, reuse `trade_checks.py`,
  clear `coinapi_fill_deferred`) but needs a bounded single-day CoinAPI pull and stays behind the
  still-locked backfill gate. It is the required mechanism to clear the deferral, not an optional
  extra (§8).
- No bar-builder changes on this branch (Phase 4 enforces the gate; the builder does not exist yet).
- No new market-regime taxonomy — regimes reuse the days already characterized in §5a-QualityMap.

## Background (what exists today)

`ingest/verify_trades_and_calendar.py::check_trades(sess, exch, sym, day)` validates **one day
(2025-06-01), one symbol per venue**, printing to stdout (no per-day JSON report, no gate). It
measures: row count + columns; `origin_time`/`received_time` "empty" fraction (sentinel
`< pd.Timestamp("2015-01-01")`); `received − origin` lag median/p95/negative; `side` value counts;
`trade_id` uniqueness + monotonic-in-file; and whether the file is ordered by `origin_time`. The
§5b table records the single-day result:

| Venue | rows (2025-06-01) | origin/recv empty | recv−origin lag med/p95 | `trade_id` | file order |
|---|---|---|---|---|---|
| Binance perp | 812,701 | 0% / 0% | 57 / 200 ms | int64, unique, monotonic | sorted by `origin_time` |
| Binance spot | 645,930 | 0% / 0% | 5 / 63 ms | int64, unique, monotonic | sorted by `origin_time` |
| Coinbase | 274,489 | 0% / 0% | 164 / 238 ms | int64, unique, **not** monotonic | **NOT** sorted by `origin_time` |

The load-bearing finding (§5b): **Coinbase trades are not stored in `origin_time` order and
`trade_id` is not monotonic — the clock must sort Coinbase trades by `origin_time`** (Binance feeds
are already ordered). "One day, one symbol each — extend to multi-day before relying on it." That
extension is this plan.

---

## 1. Validation goals

For every selected `(venue, day)` the validator answers, with machine-readable reason codes:

1. **Existence** — the `trades` partition exists and is non-empty for the venues/days needed by the
   modeling calendar (a missing/empty required partition is a hard fail; a missing partition on a
   known Coinbase CoinAPI-fill day is expected → routed, not failed).
2. **Timestamp fields & sorting semantics** — `origin_time` (exchange time) is present and populated;
   the frame yields a **non-decreasing engine clock after the chosen sort** (§5). Coinbase is the
   only venue where the sort changes the order; the check must confirm the sort *repairs* it.
3. **Row-count plausibility** — the day is not implausibly empty for the venue/regime (hard floor +
   a soft, regime-aware surface).
4. **Empty / missing / sparse partitions** — detect empty partitions, missing partitions, and
   missing/sparse *hours within* a present partition (quiet night vs. a real intraday gap,
   disambiguated against the calendar).
5. **Duplicate / non-monotonic exchange-time issues** — duplicate `origin_time` clusters, duplicate
   `trade_id`s, and any residual non-monotonicity of the engine clock after the sort.
6. **Price / size sanity** — prices positive & finite, single-trade jump distribution sane; sizes
   positive & finite (zero/negative sizes corrupt dollar bars), max-size plausible; notional volume
   plausible.
7. **Bar-clock suitability** — the trade stream can drive the notional bar clock: a monotonic engine
   clock, positive notional, and an inter-arrival distribution that supports the E0.3 gate ("median
   active-regime bar ≤ 2 s"). This is the summary judgement the bar builder gates on (§8).

Every result carries exactly one `status` from a **closed set of five**: **pass**, **warn** (usable
but surfaced), **fail** (blocking), and the two routed states **coinapi_fill** and **excluded**.
Structural problems — `empty_partition`, `missing_partition`, `load_error` — are **reason codes, not
statuses**: on a required non-fill day they resolve to `status: "fail"` (so they always land in
`summary.gate.blocking_failures`); a clean "no partition" on a calendar `trades`-fill day resolves to
`coinapi_fill`, and any day in the excluded calendar resolves to `excluded`. Keeping `fail` the single
blocking status means a consumer that keys off `status == "fail"` / `summary.counts.fail` can never
let a missing/empty/load-error partition escape the gate (§8).

## 2. Venues & products

Reuse the exact Crypto Lake identifiers already used in `ingest/verify_trades_and_calendar.py` and
`ingest/verify_lake.py` (do not invent new ones):

| Venue key | `exchange` | `symbol` | table | notes |
|---|---|---|---|---|
| `binance_perp` | `BINANCE_FUTURES` | `BTC-USDT-PERP` | `trades` | primary signal side; drives perp-notional bar clock (spec §5.1) |
| `binance_spot` | `BINANCE` | `BTC-USDT` | `trades` | primary signal side; combined-notional option |
| `coinbase` | `COINBASE` | `BTC-USD` | `trades` | label venue; Lake + CoinAPI-fill; the sort-order defect lives here |

Load (identical call shape to `check_trades`), one whole day `[00:00, next-00:00)`:

```python
df = lakeapi.load_data(table="trades", start=s, end=e, symbols=[sym], exchanges=[exch],
                       boto3_session=sess, drop_partition_cols=True)
```

`sess` is `verify_lake.lake_session()` (explicit boto3 `Session` with `.env` AWS keys, region
`eu-west-1`). Loaded/renamed trade columns (`lakeapi` renames `timestamp→origin_time`,
`receipt_timestamp→received_time`): **`origin_time`, `received_time`** (datetime64[ns]), **`price`**,
**`quantity`** (NOT `size` — `size` is a `book_delta_v2` field), **`side`** ∈ {`buy`, `sell`} (taker
side), **`trade_id`** (int64). Column-project by passing the RAW names if reducing bytes:
`columns=["timestamp", "receipt_timestamp", "price", "quantity", "side", "trade_id"]`.

## 3. Day selection strategy

Selection is **bounded by design** — no broad default run. Precedence (first match wins):

1. `--days D1,D2,…` — an explicit comma-separated `YYYY-MM-DD` list (exact days).
2. `--days-file PATH` — one `YYYY-MM-DD` per line (the format the batch planner emits and
   `run_coinbase_quality_map --days-file` accepts).
3. `--start/--end` — the **regime cohort** whose members fall in `[start, end]` (below), **plus** a
   deterministic, seeded stratified random sample of `--sample-n` additional *usable* days drawn
   across the train/val/test split spans (reproducible via `--seed`). Intersected with the usable
   calendar.
4. **no day args** — the **safe small default sample** (5 curated days, ≈1.3 GB, runs without
   `--allow-broad`): `2025-06-01`, `2024-08-05`, `2024-08-06`, `2025-01-07`, `2026-04-15`.
   `2024-08-06` is included on purpose because it is a **full Coinbase gap** (no Lake partition for
   either product), so the calendar's `coinbase_fill_days[2024-08-06].trades` is set and the one
   bounded live run exercises the real `coinapi_fill` routing §8 later makes a gate condition — the
   only default day guaranteed to hit it regardless of what the other days' trades partitions turn
   out to contain.

**Regime cohort** — days chosen for regime *diversity*. **Important (exact-product contract):** the
§5a-QualityMap facts cited below (book seam / resume times, crossed-seed fractions, L3 event counts)
are all `book_delta_v2` *reconstruction* results — a **different product** from `trades`. They pick
days worth testing; they are **not** ground truth for the `trades` feed and are **not** used to
pre-declare expected trade-hour gaps. Whether a day's Coinbase `trades` partition is complete, sparse,
or absent is what the validator **measures**, and fill routing is driven only by the calendar's
per-product `coinbase_fill_days[day].trades` flag (`data/usable_calendar.json`) — never inferred from
a book seam. So the cohort exercises the checks; it does not hardcode their verdicts.

| Cohort member | day(s) | regime / why the day is worth testing (book-product context ≠ trades verdict) |
|---|---|---|
| Clean known overlap | `2025-06-01` | the pilot day; all 3 venues Lake-present, clean — the pass baseline |
| High-volatility / crash | `2024-08-05` | crash morning; extreme Binance volume + wide jumps (high-vol stress). Its `book_delta_v2` seam is 16:08:35Z, but the `trades` partition's coverage is measured, not assumed |
| Crash-adjacent full gap | `2024-08-06` | full Coinbase gap (no Lake partition for either product) → the calendar marks it a trade-fill day → must route to `coinapi_fill`, not fail |
| Coinbase gap/fill seam | `2025-01-07` | 33-day-hole end; the `book_delta_v2` resume is 14:45:00Z — the validator measures whether `trades` is similarly sparse or fully present |
| High-vol vendor seam | `2024-12-04` | 63.2 M L3 `book` events (high-vol regime); trades coverage measured independently |
| Pilot-OOS integrity day | `2026-04-15` | inside the April-2026 pilot-OOS usable run (`2026-02-06→2026-05-05`); all 3 venues Lake-present; trade validation is outcome-blind integrity work, not model evaluation |
| Late-window clean | `2026-06-15` | `lake_usable` (book) late-window control |

**Random sample across splits** (requirement: "random sampled days across train/val/test"): given
`--start/--end`, draw `--sample-n` days deterministically (seeded `random.Random(seed)`) stratified
across the three validation spans (SSL-pretrain / head-finetune / April-2026 pilot-OOS integrity),
restricted to `usable_days` from `data/usable_calendar.json`. Deterministic so a re-run validates the
same days (the report records the resolved `days_selected`). The formal G1 holdout is selected later
outside the pilot and is not defined by this trade-integrity sample.

**Crash-context requirement** is satisfied by `2024-08-05` (+ its `2024-08-06` gap neighbour) in
both the cohort and the default sample.

**Bounded-live discipline:** the union of selected days × selected venues is passed through the GB
gate (§7) *before any load*. The curated cohort + default sample stay under the auto cap; a large
`--start/--end` span or a long `--days` list that exceeds the cap is refused unless `--allow-broad`
(exit 5), exactly like `run_coinbase_quality_map`.

## 4. Metrics

Computed per `(venue, day)` on the loaded frame **after** the §5 sort. Every metric lands in the
report's `days[].metrics`; the pass/warn/fail role is defined in §8. `<field>` names are the
loaded/renamed columns (§2). `null` timestamp = the existing sentinel `value < pd.Timestamp(
"2015-01-01")`.

**All time-based metrics use the post-fallback `engine_clock`, not raw `origin_time`** (§5): the
`engine_clock` is `origin_time` with `received_time` substituted per null/sentinel row, and the sort
key. Diffing raw `origin_time` on a sub-threshold fallback day would fabricate a ~55-year gap and
day-0 hour coverage from the 1970 sentinels; the clock the bar builder actually consumes is the one
that must be gated. Raw `origin_time` is retained only for the null-fraction metric (row 3) and
audit.

| # | metric key | how computed | role |
|---|---|---|---|
| 1 | `row_count` | `len(df)` | hard-fail below `min_rows_hard`; soft surface below regime floor |
| 2 | `first_ts` / `last_ts` | ISO of `engine_clock.min()` / `.max()` after sort | reported; used for hour coverage |
| 3 | `origin_time_null_frac` | `(origin_time < 2015-01-01).mean()` | fail if `> origin_time_null_max` (else warn `received_time_fallback_used` when nonzero); substitution itself is per-row unconditional (§5) |
| 4 | `received_time_available` / `received_time_null_frac` | column present; `(received_time < 2015-01-01).mean()` | fallback availability; a null-origin row with null `received_time` → fail (§5) |
| 5 | `monotonic_after_sort` | `engine_clock.is_monotonic_increasing` on the **post-fallback** clock after the §5 sort, treating any residual `NaT`/sentinel (`< 2015-01-01`) as invalid | **hard fail** if False |
| 6 | `was_presorted` | whether the *file* arrived `origin_time`-monotonic (pre-sort) | informational (Coinbase→False, Binance→True) |
| 7 | `dup_ts_cluster_count` / `dup_ts_max_cluster` | `origin_time` values with count>1; max multiplicity | warn if `dup_ts_max_cluster > dup_ts_cluster_warn` |
| 8 | `dup_trade_id_count` / `dup_trade_id_frac` | `len(df) - trade_id.nunique()` (if `trade_id` present) | warn if `> 0` |
| 9 | `price_min` / `price_max` / `price_median` / `price_p99_abs_ret` / `price_max_abs_ret` / `price_out_of_band_count` | min/max/median; p99 **and max** of `abs(price.pct_change())` after sort; count of prices outside `[median/price_range_factor, median*price_range_factor]` | fail on non-positive/NaN price; **`price_spike` fail** if `price_max_abs_ret > price_spike_warn` **or** `price_out_of_band_count > 0` (an isolated corrupt print the p99 misses — it directly corrupts the notional bar clock, §8); `price_jump_excess` **warn** if `price_p99_abs_ret > price_jump_warn` (broad regime churn, real) |
| 10 | `size_min` / `size_max` / `size_zero_frac` / `size_neg_frac` | on `quantity` | **hard fail** if `size_zero_frac>0` or `size_neg_frac>0`; **`size_out_of_band` fail** if `size_max > size_hard_max_btc` (bar-clock-corrupting); `size_out_of_range` **warn** if `size_max > size_max_btc` (unusually large but plausible) |
| 11 | `notional_sum` / `notional_max_trade` | `Σ price*quantity`; `max(price*quantity)` | fail if `notional_sum<=0`/NaN; the single-print corruption is caught upstream by rows 9–10 |
| 12 | `interarrival_median_s` / `_p95_s` / `_p99_s` / `_max_s` | `diff(engine_clock).dt.total_seconds()` after sort | warn if `interarrival_max_s > interarrival_gap_warn_s` (calendar-context-exempt) |
| 13 | `missing_hour_count` / `sparse_hour_count` | of the 24 UTC `engine_clock` hours: 0 rows / `< sparse_hour_min_rows` | `sparse_hour` warn; `missing_hour` warn up to `max_missing_hours`, else **`missing_hours_excess` fail** on a required non-fill day (§8) |
| 14 | `recv_origin_lag_median_ms` / `_p95_ms` / `_neg_frac` | `(received_time-origin_time)` ms | informational (Coinbase inherently higher); fail only if `_neg_frac > lag_neg_frac_max` |
| 15 | `side_values` | `side.value_counts()` dict | warn if any value ∉ {`buy`,`sell`} |
| 16 | `calendar_state` | crossed with `data/usable_calendar.json` (§8) | routes `coinapi_fill` / `excluded` |

All float metrics pass through `_json_safe` (non-finite → `null`) so the report is strict JSON.

## 5. Timestamp policy

- **Engine clock = `origin_time` (exchange/origin time).** It is the §5.3 engine-time axis
  (`shared_engine_time_col`) and is measured 100% populated (0% empty) on all three venues, so it is
  the primary sort key and bar-clock timestamp. The validator materializes a derived `engine_clock`
  column (= `origin_time`, with the row-level `received_time` fallback below applied); **every
  time-based metric in §4 — sort, `first_ts`/`last_ts`, inter-arrival, hour coverage — reads
  `engine_clock`, never raw `origin_time`**, which stays only for `origin_time_null_frac` and audit.
- **Fallback to `received_time` (per-row, unconditional).** The substitution is **not** gated by any
  threshold: **every** row whose `origin_time` is null/sentinel (`< 2015-01-01`) takes its
  `received_time` into the engine clock (the original `origin_time` is retained in the frame for
  audit). Gating the substitution behind `origin_time_null_max` would be unsound — a sub-threshold
  handful of sentinel `origin_time` rows are not `NaT`, so they sort to 1970 at the front and still
  read as `monotonic_after_sort=True`, silently corrupting the bar clock. The threshold governs
  **severity, not whether fallback happens**:
  - any row where **both** `origin_time` and `received_time` are null/sentinel → its engine clock is
    unresolvable → **hard fail** (`received_time_fallback_unavailable`), at any fraction;
  - if `origin_time_null_frac ≤ origin_time_null_max` (default 1%): substitute silently-recorded →
    **warn** (`received_time_fallback_used`);
  - if `origin_time_null_frac > origin_time_null_max`: too much of the day is off exchange time →
    **hard fail** (`origin_time_null_fraction_high`), even though every recoverable row was
    substituted.
  A day that needed *any* fallback records `used_received_time_fallback: true`. `monotonic_after_sort`
  is computed on the **post-substitution** engine clock and treats any remaining sentinel/null value
  as invalid (it fails, never reads as trivially monotonic) — falling off exchange time is never
  silently accepted.
- **Sorting rule (explicit):** sort ascending by the engine clock, **stable**, tie-breaking by
  original file/row order — i.e. `df.sort_values(clock_col, kind="mergesort")` on a
  `reset_index`-preserved frame. This matches the resolved CoinAPI within-timestamp policy (file/`seq`
  order is canonical; ties break by original row index; IDs are never an ordering key —
  `docs/superpowers/plans/2026-07-02-coinapi-within-timestamp-ordering.md`).
- **Per-venue implications:**
  - **Coinbase** — the sort is **load-bearing**: the file is not `origin_time`-ordered and
    `trade_id` is not monotonic, so the validator both applies the sort *and* asserts the sorted
    clock is monotonic (`monotonic_after_sort`). `was_presorted=False` is expected, not a fault.
  - **Binance spot & perp** — already `origin_time`-sorted with monotonic `trade_id`; the sort is a
    no-op but is still applied uniformly. `was_presorted=True`.
  - Positive-lag observation (`recv−origin ≥ 0`, measured 0% negative) is enforced softly
    (`lag_neg_frac_max`); Coinbase's larger lag (164/238 ms vs. Binance 5–57 ms) is informational.

## 6. Report / output schema

Written to `data/reports/trade_feed_validation.json` (the git-ignored `data/reports/` dir; CLI
`--out-dir`, default `data/reports`). Strict JSON via `write_report`:
`json.dump(_json_safe(report), f, indent=2, allow_nan=False)` + trailing newline, so `jq empty`
passes. Top-level shape mirrors the quality-map report `{"meta", "summary", "days"}`. **The values
below are illustrative (schema shape, not measured) — the live run fills them from the actual
`trades` partitions; nothing here is a hardcoded expected verdict:**

```json
{
  "meta": {
    "script": "ingest/validate_trade_feeds.py",
    "table": "trades",
    "anchor_end": "2026-06-22",
    "venues": ["binance_perp", "binance_spot", "coinbase"],
    "days_requested": ["2025-06-01", "2024-08-05", "2024-08-06", "2025-01-07", "2026-04-15"],
    "days_selected": ["2025-06-01", "2024-08-05", "2024-08-06", "2025-01-07", "2026-04-15"],
    "selection_mode": "default_sample",
    "seed": 0,
    "timestamp_policy": {"engine_clock": "origin_time", "fallback": "received_time",
                          "sort": "stable_by_engine_clock_then_file_order"},
    "thresholds": { "origin_time_null_max": 0.01, "min_rows_hard": 1000,
                    "interarrival_gap_warn_s": 120.0, "sparse_hour_min_rows": 60,
                    "max_missing_hours": 1, "price_jump_warn": 0.10, "price_spike_warn": 0.50,
                    "price_range_factor": 10.0, "size_max_btc": 500.0, "size_hard_max_btc": 5000.0,
                    "dup_ts_cluster_warn": 50, "lag_neg_frac_max": 0.001 },
    "trades_gb_per_day": {"binance_perp": 0.12, "binance_spot": 0.10, "coinbase": 0.05},
    "quota": { "ok": true, "reason": "ok", "est_gb": 1.35, "used_gb": 42.0,
               "quota_gb": 300.0, "max_auto_gb": 3.0, "allow_broad": false, "headroom_gb": 10.0,
               "used_gb_before": 42.0, "used_gb_after": 43.35 },
    "vendor_api": { "source": "crypto_lake", "region": "eu-west-1", "table": "trades",
                    "lakeapi_calls": 15, "dry_run": false, "coinapi_used": false },
    "source_artifacts": { "usable_calendar": "data/usable_calendar.json",
                          "usable_calendar_anchor_end": "2026-06-22" },
    "generated_utc": "2026-07-02T00:00:00+00:00",
    "note": "VALIDATION only (docs/data.md §5b / §10 trade-validation breadth). Reads Crypto Lake trades; the CoinAPI backfill gate stays LOCKED and this tool does not unlock it."
  },
  "summary": {
    "n_days": 5, "n_venues": 3,
    "counts": { "pass": 11, "warn": 3, "fail": 0, "coinapi_fill": 1, "excluded": 0 },
    "fail_day_venues": [],
    "warn_day_venues": [
      {"day": "2024-08-05", "venue": "coinbase", "reason_codes": ["sparse_hour"]},
      {"day": "2025-01-07", "venue": "coinbase", "reason_codes": ["sparse_hour"]},
      {"day": "2024-08-05", "venue": "binance_perp", "reason_codes": ["interarrival_gap_excess"]}],
    "by_venue": { "coinbase": {"pass": 2, "warn": 2, "fail": 0, "coinapi_fill": 1},
                  "binance_spot": {"pass": 5}, "binance_perp": {"pass": 4, "warn": 1} },
    "gate": { "lake_required_pass": true, "bars_ready": false, "blocking_failures": [],
              "coinapi_fill_deferred": [{"day": "2024-08-06", "venue": "coinbase"}] }
  },
  "days": [
    {
      "day": "2024-08-05", "venue": "coinbase", "exchange": "COINBASE", "symbol": "BTC-USD",
      "status": "warn", "reason_codes": ["sparse_hour", "sparse_hour:hour=05"],
      "vendor_source": "lake",
      "metrics": { "row_count": 921034, "first_ts": "2024-08-05T00:00:00.812Z",
                   "last_ts": "2024-08-05T23:59:59.771Z", "origin_time_null_frac": 0.0,
                   "received_time_available": true, "monotonic_after_sort": true,
                   "was_presorted": false, "used_received_time_fallback": false,
                   "dup_ts_cluster_count": 3, "dup_ts_max_cluster": 4,
                   "dup_trade_id_count": 0, "price_min": 49500.0, "price_max": 58000.0,
                   "price_median": 55200.0, "price_p99_abs_ret": 0.004,
                   "price_max_abs_ret": 0.031, "price_out_of_band_count": 0,
                   "size_min": 0.0001, "size_max": 42.5,
                   "size_zero_frac": 0.0, "size_neg_frac": 0.0, "notional_sum": 2.5e9,
                   "interarrival_median_s": 0.09, "interarrival_p99_s": 3.2,
                   "interarrival_max_s": 41.0, "missing_hour_count": 0, "sparse_hour_count": 1,
                   "recv_origin_lag_median_ms": 168.0, "recv_origin_lag_neg_frac": 0.0,
                   "side_values": {"buy": 456210, "sell": 464824} },
      "calendar_state": { "class": "lake_trades_present", "in_usable_days": true,
                          "fill": {"book": true, "trades": false},
                          "note": "book_delta_v2 seam 16:08:35Z is a separate product; the trades partition here is measured full-day present (warn only for one sparse overnight hour)" }
    }
  ]
}
```

**Reason codes** (stable module-level string constants, `SCREAMING_SNAKE` name → `snake_case`
value; a reason entry is either the bare code or a `code:detail`/`metric=value>threshold` string,
stable code first — the `run_coinbase_quality_map.classify_day` convention):

| code | meaning | default role |
|---|---|---|
| `ok` | all checks pass | pass |
| `empty_partition` | partition loaded but 0 rows | fail (unless fill/excluded) |
| `missing_partition` | no partition for the day | fail (unless fill/excluded) |
| `load_error` | load raised | fail |
| `origin_time_column_missing` | no `origin_time`/`timestamp` column | fail |
| `origin_time_null_fraction_high` | `origin_time_null_frac > origin_time_null_max` | **fail** on its own (too much off exchange time), even though every recoverable row was substituted (§5) |
| `received_time_fallback_used` | sub-threshold null `origin_time` rows fell back to `received_time` | warn |
| `received_time_fallback_unavailable` | needed fallback but `received_time` null/absent | fail |
| `nonmonotonic_after_sort` | sorted engine clock not non-decreasing / has `NaT` | fail |
| `price_out_of_range` | any price `<= 0` or NaN | fail |
| `size_nonpositive` | any `quantity <= 0` or NaN | fail |
| `notional_nonpositive` | `notional_sum <= 0`/NaN | fail |
| `row_count_implausibly_low` | `< min_rows_hard` | fail |
| `duplicate_timestamp_cluster` | large same-ns cluster | warn |
| `duplicate_trade_id` | repeated `trade_id` | warn |
| `price_jump_excess` | `price_p99_abs_ret > price_jump_warn` (broad regime churn) | warn |
| `price_spike` | `price_max_abs_ret > price_spike_warn` or `price_out_of_band_count > 0` (isolated corrupt print p99 misses; corrupts the notional clock) | **fail** (→ fix/quarantine/exclude, §8) |
| `size_out_of_range` | `size_max > size_max_btc` (unusually large but plausible) | warn |
| `size_out_of_band` | `size_max > size_hard_max_btc` (bar-clock-corrupting) | **fail** (→ fix/quarantine/exclude, §8) |
| `interarrival_gap_excess` | `interarrival_max_s > interarrival_gap_warn_s` | warn |
| `missing_hour` / `sparse_hour` | empty / sparse UTC `engine_clock` hour | warn |
| `missing_hours_excess` | required non-fill day, `missing_hour_count > max_missing_hours` | fail (→ fill/exclude) |
| `lag_negative` | `recv_origin_lag_neg_frac > lag_neg_frac_max` | fail |
| `side_value_unexpected` | `side ∉ {buy,sell}` | warn |
| `row_count_low` | below the regime soft floor | warn |
| `coinapi_fill_day` | Coinbase Lake trades missing on a known fill day | routed → `coinapi_fill` |
| `calendar_excluded_day` | day in `excluded_days_by_reason` | routed → `excluded` |

**Thresholds** are a frozen `TradeThresholds` dataclass with `as_dict()`, emitted into `meta.thresholds`
(the `Thresholds.as_dict()` pattern — every artifact records the knobs that produced it):

| knob | default | rationale |
|---|---|---|
| `origin_time_null_max` | 0.01 | matches `verify_lake`'s `< 0.01 → USABLE (exchange time)` bar |
| `min_rows_hard` | 1000 | a `trades` day under ~1k rows is a broken/near-empty partition, not a quiet day |
| `dup_ts_cluster_warn` | 50 | same-ns trades are normal in bursts; a >50-deep single-ns cluster is worth a look |
| `price_jump_warn` | 0.10 | a broad-day p99 >10% consecutive-trade churn — a genuinely volatile *regime*, so warn not fail |
| `price_spike_warn` | 0.50 | a >50% single-tick abs return is almost always one corrupt print (p99 can't see a lone outlier); **blocking** because it corrupts the notional clock |
| `price_range_factor` | 10.0 | any price outside `[median/10, median×10]` is grossly implausible intraday (BTC never moves 10× within a day), regime-agnostic vs. a hardcoded band; **blocking** |
| `size_max_btc` | 500.0 | a single BTC trade > 500 BTC is unusually large but can be a real block trade → warn |
| `size_hard_max_btc` | 5000.0 | a single BTC trade > 5000 BTC (~$300M) is bar-clock-corrupting and near-certainly a bad print → **blocking** |
| `interarrival_gap_warn_s` | 120.0 | a >2 min no-trade gap in a normally-active market; quiet-hour context exempts it |
| `sparse_hour_min_rows` | 60 | < 1 trade/min for a whole UTC hour is sparse |
| `max_missing_hours` | 1 | ≥2 fully-empty UTC hours on a continuously-traded BTC venue is a data gap, not quiet — a required non-fill day above this fails (§8) rather than passing as warn |
| `lag_neg_frac_max` | 0.001 | `received ≥ origin` should hold; a nonzero negative-lag fraction is a clock fault |

## 7. CLI shape (future implementation)

Script: `ingest/validate_trade_feeds.py` (thin Lake wrapper over `ingest/trade_checks.py`).

| arg | type / default | meaning |
|---|---|---|
| `--start` | `YYYY-MM-DD` | range start for cohort + stratified-sample selection (§3) |
| `--end` | `YYYY-MM-DD`, default `$END`/`2026-06-22` | range end / anchor (the `END=` convention) |
| `--days` | comma list | explicit `YYYY-MM-DD` days (highest precedence) |
| `--days-file` | path | one `YYYY-MM-DD` per line (batch-planner day-file format) |
| `--venues` | comma list, default all 3 | subset of `binance_perp,binance_spot,coinbase` |
| `--sample-n` | int, default 0 | extra deterministic stratified-random usable days for `--start/--end` |
| `--seed` | int, default 0 | seed for the stratified sample (reproducible) |
| `--usable-calendar` | path, default `data/usable_calendar.json` | calendar-crossing + fill/excluded routing (§8) |
| `--max-auto-gb` | float, default 3.0 | auto-allowed est-GB cap; larger needs `--allow-broad` |
| `--quota-gb` | float, default 300.0 | Crypto Lake monthly cap (mirrors `run_coinbase_quality_map.QUOTA_GB`) |
| `--headroom-gb` | float, default 10.0 | quota GB always left unused |
| `--allow-broad` | store_true | override the auto cap (still refused if it breaches quota headroom) |
| `--out-dir` | path, default `data/reports` | report dir (git-ignored); writes `trade_feed_validation.json` |
| `--strict` | store_true | exit `7` if any blocking fail (for CI gating); default exits 0 after writing the report |
| `--dry-run` | store_true | print the resolved plan (days × venues, est GB, quota decision) and write nothing — **no vendor calls** |

**No broad default run:** with no day args the validator runs the **safe small default sample** (5
curated days, ≈1.3 GB) which passes the auto cap. `--dry-run` runs full selection + GB estimation +
calendar crossing with **zero** Lake I/O (like the batch planner's dry-run: everything but the load
runs). The GB gate reuses the `run_coinbase_quality_map` pattern — `estimate_trades_gb(venues, days,
per_venue_gb)` + `quota_decision(...)` returning `ok`/`quota_headroom`/`exceeds_auto_cap`; a refused
plan exits `5` (`QUOTA_REFUSED_EXIT`); on failure to read `used_data` it fail-safe assumes
`used_gb = quota_gb`.

**Exit-code contract** (extends the repo's small-int convention `2` CoinAPI-quota / `3` parity / `4`
CoinAPI-backfill / `5` Lake-quota / `6` native): `0` ok; `5` Lake quota refused; `7`
(`VALIDATION_FAILED_EXIT`) when `--strict` and ≥1 blocking fail.

Examples:

```bash
# dry-run the default sample: prints days × venues, est GB, quota decision — NO vendor calls
.venv/bin/python ingest/validate_trade_feeds.py --dry-run

# bounded live run over an explicit regime set, all 3 venues (Phase 2)
.venv/bin/python ingest/validate_trade_feeds.py --days 2025-06-01,2024-08-05,2025-01-07,2026-04-15

# a broad sweep across a range (deliberate, budgeted) — refused without --allow-broad
.venv/bin/python ingest/validate_trade_feeds.py --start 2024-06-22 --end 2026-05-05 \
    --sample-n 20 --seed 7 --allow-broad
```

## 8. Gating decision

**What must pass before bars/features may use a venue's trade feed for a span.** For every
`(venue, day)` in the span that is a **required Lake day** (present in `usable_days`, not a Coinbase
CoinAPI-fill day for that product, not `excluded`), `status ∈ {pass, warn}`. Any `fail` blocks the
bar builder for that `(venue, span)` until fixed, filled, or the day is excluded from the calendar —
mirroring the "never silently dropped" discipline (a fail is surfaced in
`summary.gate.blocking_failures`, never quietly skipped).

**Fail (blocking)** — each is a reason code that sets `status: "fail"` (the single blocking status,
§1) on a required non-fill day, so it always appears in `summary.gate.blocking_failures`:

- `missing_partition` / `empty_partition` on a required (non-fill) day (a clean "no partition" on a
  calendar `trades`-fill day routes to `coinapi_fill` instead; on an excluded day → `excluded`)
- `load_error` (an unexpected load exception — always `fail`, distinct from an expected
  "no files" → `missing_partition`), `origin_time_column_missing`
- `origin_time_null_fraction_high` — `origin_time_null_frac > origin_time_null_max`, **on its own**
  (too much of the bar clock is on receive-time even though every recoverable row was substituted;
  per §5 the threshold is a hard-fail bound, not merely a warn)
- `received_time_fallback_unavailable` — a null-`origin_time` row whose `received_time` is also
  null (unresolvable engine clock), **on its own**, at any fraction
- `nonmonotonic_after_sort` (the sorted engine clock **is** the bar clock — non-monotonic is fatal)
- `missing_hours_excess` — a required non-fill day with `missing_hour_count > max_missing_hours`
  (a real intraday trade-clock hole; BTC on these venues trades every second, so ≥2 fully-empty UTC
  hours is a data gap, not a quiet market). The bar clock must not span an unfilled hole, so the day
  must be **filled (CoinAPI) or excluded**, never emitted across — this is the mechanism that turns a
  day-level "partition present" into an intraday-coverage gate. (Calendar fill days route to
  `coinapi_fill` before this check; sparse — non-empty — hours stay warn.)
- `price_out_of_range`, `size_nonpositive`, `notional_nonpositive`
- `price_spike` / `size_out_of_band` — an **isolated** grossly-out-of-band print (single-tick
  `price_max_abs_ret > price_spike_warn`, a price outside the robust `price_range_factor` band, or a
  size > `size_hard_max_btc`). The dollar bar clock sums `price × quantity`, so **one** such print
  prematurely trips a bar boundary and mis-times every downstream feature/label — a single outlier the
  robust p99/median checks are designed to see. Blocking, not warn: the day must be fixed, the print
  explicitly **quarantined** (a documented drop-mask the bar builder applies), or the day excluded —
  never silently consumed. The real-flash-vs-corrupt ambiguity is exactly why it needs *explicit*
  acceptance rather than passing the gate as a warning.
- `row_count_implausibly_low` (`< min_rows_hard`)
- `lag_negative` above `lag_neg_frac_max`

**Warn (usable, surfaced, non-blocking):** `duplicate_timestamp_cluster`, `duplicate_trade_id`,
`price_jump_excess` (broad p99 regime churn — real volatility, not a lone bad print),
`size_out_of_range` (unusually large but plausible block trade), `interarrival_gap_excess`,
`sparse_hour`, `missing_hour` (**up to `max_missing_hours`** — beyond that it escalates to the blocking
`missing_hours_excess` above), `side_value_unexpected`, `received_time_fallback_used`, `row_count_low`.
These are either legitimate market behaviour (flash moves, dead Sundays, same-ns bursts) or
informational; the bar builder may consume the day but the report retains the flags for stratified
diagnostics (experiment-plan "stratify all results by regime"). The split is deliberate: a *broad*
volatile-regime signal is warn, but a *discrete* corrupt-looking print that would poison the notional
accumulator blocks (above).

**How CoinAPI-fill days are handled.** A Coinbase day whose Lake `trades` are missing/partial and
which appears in `coinbase_fill_days` (with `trades: true`, i.e. Lake trades need CoinAPI fill) is
**not failed** for the missing Lake side. It is routed to `status = "coinapi_fill"` (reason
`coinapi_fill_day`) and listed under `summary.gate.coinapi_fill_deferred`. Its *trade-feed* validity
is then judged against the **CoinAPI** trade file by the **Phase 3b / Task 5** path — which
downloads the CoinAPI `TRADES` dataset, normalizes it to the Lake schema, and runs the **same**
`ingest/trade_checks.py::validate_trade_frame` on it (`vendor_source: "coinapi"`); a pass/warn
removes the day from `coinapi_fill_deferred`. This is the *only* mechanism that clears the deferral,
so it is required for `bars_ready` to hold on fill-day spans (all **52** of them, §5b) — without it
the gate is a dead-end. It is a **gated follow-up**: it needs a bounded single-day CoinAPI pull, and
**the CoinAPI backfill gate stays LOCKED** until the §5a multi-day parity/quality-map passes. Until
then a fill day is "pass-conditional-on-CoinAPI" (deferred, correctly not `bars_ready`), exactly
parallel to the quality-map `needs_fill` semantics. Because `trade_checks.py` is **source-agnostic**
(it validates a normalized frame and never imports a vendor client), the Lake and CoinAPI paths reuse
one validator — only the loader/normalizer differs. A day in `excluded_days_by_reason` is `status =
"excluded"` (out of scope; already dropped by the usable calendar) and never blocks.

**`summary.gate` fields (the bar builder gates on `bars_ready`, never `lake_required_pass`).** The two
booleans are deliberately distinct so a deferred, unvalidated fill can never read as "ready":

- `lake_required_pass` — every **required Lake** `(venue, day)` (in `usable_days`, not a fill day for
  that product, not excluded) is `pass`/`warn`. This is the Lake-side result only; it says nothing
  about fill days.
- `coinapi_fill_deferred` — the `(venue, day)` fill cases routed to `coinapi_fill` whose CoinAPI trade
  validation is still pending (gated behind the **locked** backfill). A span touching any of these is
  not yet buildable.
- `bars_ready` — `lake_required_pass` **and** `coinapi_fill_deferred == []` (every fill in scope
  validated). **Phase 4 gates on `bars_ready`**; because a fill day stays in `coinapi_fill_deferred`
  until its CoinAPI validation lands (which the locked backfill gate currently forbids), `bars_ready`
  is `false` for any span containing a fill day — so the builder can never span an unvalidated/locked
  Coinbase trade gap. In the example above `lake_required_pass` is `true` but `bars_ready` is `false`
  precisely because `2024-08-06` is deferred.
- `blocking_failures` — the required-day `status: "fail"` list; `lake_required_pass` is `false`
  whenever it is non-empty.

**Interaction with the modeling calendar (§5b) & CV (§8/E0.4):** the effective modeling calendar is
the contiguous usable runs (`2024-06-22→2026-02-04`, `2026-02-06→2026-05-05`); April 2026 is the
pilot OOS, while formal G1 OOS remains unselected and must be outside the pilot.
Purged+embargoed CPCV already drops label-span-overlapping bars at gap/seam boundaries, so a
warn-heavy seam day (e.g. `2025-01-07`) loses its boundary bars to embargo regardless; trade
validation gates the *interior* of each usable run.

## 9. Tests (future implementation)

All in `tests/test_trade_checks.py` against `ingest/trade_checks.py` — **pure, no vendor I/O**.
Synthetic frames come from a module-level helper (not a fixture), matching the repo pattern
(`_lake_df`, `make_matrix`, `simple_world`) and the existing synthetic-trades frame in
`tests/test_ingest.py`:

```python
import numpy as np, pandas as pd

# lakeapi returns origin_time/received_time as tz-NAIVE datetime64[ns] (§4), and the validator's
# null sentinel `< pd.Timestamp("2015-01-01")` is tz-naive too. Keep every synthetic timestamp naive
# (no "Z"/tz on `start`) — a tz-aware column vs. the naive cutoff raises "Cannot compare tz-naive and
# tz-aware", and assigning the naive SENTINEL into a tz-aware column coerces it to object.
SENTINEL = pd.Timestamp("1970-01-01")          # < 2015-01-01 → treated as null (§4/§5), tz-naive

def _trades_df(n=1000, start="2025-06-01T00:00:00", step_ms=80, *, presorted=True,
               full_day=False, null_origin=0.0, null_received=0.0, dup_ids=0,
               bad_price=False, bad_size=False, spike_price=False, empty_hours=()):
    """Deterministic synthetic Crypto Lake `trades` frame (loaded/renamed columns).

    full_day=True spreads the n rows evenly across [00:00, 24:00) (step = 86_400_000 // n ms) so the
    24-hour missing/sparse-hour metric is exercisable; otherwise rows are step_ms apart from `start`.
    null_origin / null_received are FRACTIONS of leading rows set to the 1970 sentinel; the same
    leading rows overlap, so null_origin>0 with null_received>0 makes those rows unrecoverable.
    dup_ids is the number of EXTRA duplicate trade_ids created (first dup_ids+1 rows share one id).
    spike_price sets one middle row to an 11× price (660000 among constant 60000, just past the
    median×10 band) — an isolated corrupt print p99 abs-return cannot see but price_max_abs_ret /
    the robust band catch.
    """
    base = pd.Timestamp(start)
    step = (86_400_000 // n) if full_day else step_ms
    origin = base + pd.to_timedelta(np.arange(n) * step, unit="ms")
    if not presorted:                          # Coinbase shape: shuffle the file order
        origin = origin[np.random.RandomState(0).permutation(n)]
    df = pd.DataFrame({
        "origin_time": origin,
        "received_time": origin + pd.to_timedelta(160, unit="ms"),
        "price": np.full(n, 60000.0),
        "quantity": np.full(n, 0.01),
        "side": np.where(np.arange(n) % 2, "buy", "sell"),
        "trade_id": np.arange(n, dtype="int64"),
    })
    if null_origin:   df.loc[df.index[: int(n*null_origin)], "origin_time"] = SENTINEL
    if null_received: df.loc[df.index[: int(n*null_received)], "received_time"] = SENTINEL
    if dup_ids:       df.loc[df.index[: dup_ids + 1], "trade_id"] = df["trade_id"].iloc[0]
    if bad_price:     df.loc[df.index[0], "price"] = 0.0
    if bad_size:      df.loc[df.index[0], "quantity"] = -1.0
    if spike_price:   df.loc[df.index[n // 2], "price"] = 660000.0   # one 11× print among constants
    for h in empty_hours: df = df[df["origin_time"].dt.hour != h]
    return df.reset_index(drop=True)
```

Required test cases (TDD — write each failing test first).

**Test at the right altitude (avoids the coverage-gate footgun).** Because `missing_hours_excess` is
now blocking (§8), a short (`full_day=False`, ~80 s) fixture has 23 empty hours and would fail *any*
whole-day `classify(...)` regardless of what the case targets. So each per-dimension case (1–5b, 7, 8)
asserts the **isolated pure check/metric function** it names (`dup_trade_ids(df)`,
`price_checks(df)`, `interarrival(df)`, …) on a short frame — never the whole-day `classify`. The
hour-coverage case (6) and the end-to-end `classify`/report cases (9, 10) use a **full-day frame with
≥ `sparse_hour_min_rows` rows per hour** (e.g. `_trades_df(n=2400, full_day=True)` → ~100/hour) so a
clean day classifies `pass`, not a coverage `fail`.

1. **Missing timestamp field** — a frame with no `origin_time`/`timestamp` column →
   `origin_time_column_missing`, status `fail`.
2. **`origin_time` null fallback (three branches, matching §5 severity).**
   - **Sub-threshold warn:** `null_origin=0.005` (0.5% ≤ `origin_time_null_max`) with `received_time`
     present → engine clock uses `received_time` for those rows, `monotonic_after_sort=True`,
     `received_time_fallback_used`, status `warn`.
   - **Super-threshold fail:** `null_origin=0.02` (2% > 1%) with `received_time` present → the
     substitution still happens per-row, but too much of the day is off exchange time →
     `origin_time_null_fraction_high`, status `fail` (guards against the test accepting a >1%
     off-exchange-time day and weakening the gate).
   - **Unrecoverable fail:** `null_origin=0.005, null_received=0.005` (the same leading rows null in
     both) → `received_time_fallback_unavailable`, status `fail`, at any fraction.
3. **Non-monotonic → repaired by sort** (asserts `monotonic_after_sort(df)` in isolation, so the
   short fixture's hour coverage is irrelevant) — a `presorted=False` frame yields
   `monotonic_after_sort == True` (proves the Coinbase sort works); a frame with a `NaT` engine clock
   the sort cannot repair → `monotonic_after_sort == False` → `nonmonotonic_after_sort`, `fail`.
4. **Duplicate trade IDs** — `dup_ids=5` (the helper makes the first 6 rows share one id → exactly 5
   extra duplicates) → `dup_trade_id_count == 5`, `duplicate_trade_id` warn.
5. **Invalid price / size** — `bad_price=True` → `price_out_of_range` fail; `bad_size=True` →
   `size_nonpositive` fail; a zero-size row → `size_nonpositive` fail. A single `quantity` above
   `size_hard_max_btc` (e.g. 6000) → `size_out_of_band` **fail**; one above `size_max_btc` but below
   the hard ceiling (e.g. 800) → `size_out_of_range` **warn**.
5b. **Isolated positive price spike blocks** — `spike_price=True` (one 11× print among 1000 constant
   prices) → `price_p99_abs_ret ≈ 0` (p99 misses the two spike diffs) **but** `price_max_abs_ret` large
   and `price_out_of_band_count == 1` → `price_spike` **fail** (blocking, §8): one such print poisons
   the notional bar clock, so it must block/quarantine, not pass as warn. A broad-churn day
   (`price_p99_abs_ret > price_jump_warn` with no out-of-band print) → `price_jump_excess` **warn**
   (real volatile regime) — pins the deliberate broad-vs-isolated split.
6. **Sparse / missing hour + coverage gate** — build a **full-day** frame
   (`_trades_df(n=2400, full_day=True)` → ~100 rows in each of the 24 UTC hours). (The default 80 ms
   frame spans only ~80 s of hour 0 and cannot exercise a 24-hour metric.)
   - `empty_hours=(4,)` (1 missing ≤ `max_missing_hours`) on a **required non-fill** day →
     `missing_hour_count == 1`, `missing_hour` **warn**.
   - `empty_hours=(3,4)` (2 missing > `max_missing_hours=1`) on a **required non-fill** day →
     `missing_hours_excess` **fail** (a real intraday trade-clock hole must fill/exclude, not pass —
     the P1 gate fix); the same frame on a `coinbase_fill_days[day].trades` day routes to
     `coinapi_fill`, and on an excluded day → `excluded` (neither blocks).
   - thinning one non-empty hour below `sparse_hour_min_rows` (drop all but 30 of its rows) →
     `sparse_hour` **warn** at any count (a non-empty hour is plausibly quiet).
   - hours are computed on the `engine_clock` (§5), so a sub-threshold fallback frame's substituted
     rows land in their real hour, not 1970 (the P2 clock fix).
7. **Duplicate-timestamp cluster** — many rows sharing one `origin_time` above `dup_ts_cluster_warn`
   → `duplicate_timestamp_cluster` warn; a small burst does not trip it.
8. **Inter-arrival gap** — inject a 200 s gap → `interarrival_max_s ≈ 200`, `interarrival_gap_excess`
   warn.
9. **Calendar crossing + gate booleans** — write a synthetic `usable_calendar.json` to `tmp_path`
   (the `_calendar_dict`/`_write_calendar` pattern); a fill day (`coinbase_fill_days` with
   `trades: true`) with a missing Lake frame → `status "coinapi_fill"`; an excluded day →
   `"excluded"`; both are excluded from `gate.blocking_failures`. Assert the gate booleans:
   a report over otherwise-clean required days **plus** the deferred fill day →
   `gate.lake_required_pass == True` but `gate.bars_ready == False` (a deferred fill blocks
   readiness); with no fill days → `bars_ready == True`; with a required-day `fail` →
   `lake_required_pass == False` and the day in `blocking_failures`.
10. **Report JSON stability** — `build_report([...])` → `write_report` to `tmp_path`; assert
    `"NaN" not in txt and "Infinity" not in txt`, `json.loads(txt)` round-trips, and **byte-for-byte
    determinism across two runs** (the `generated_utc` timestamp is the only exempted field — pass a
    fixed timestamp in tests); every reason code / venue present in `summary` even at zero count
    (stable schema, the `for c in CLASSES: assert c in ...` pattern).
11. **Malformed-input validation** — a malformed calendar entry (`coinbase_fill_days` value not a
    `{book,trades}` dict) → `pytest.raises(ValueError, match="coinbase_fill_days")` (the parametrized
    malformed-input pattern).
12. **GB gate** — `estimate_trades_gb` + `quota_decision`: a plan over the auto cap without
    `--allow-broad` → `reason == "exceeds_auto_cap"`; over headroom → `"quota_headroom"`; within →
    `"ok"` (no vendor call; pure).
13. **No live vendor calls** — an import-side-effect guard test (like `test_verify_script.py`):
    importing `ingest.validate_trade_feeds` does not touch Lake (`main()` guarded), and a
    `monkeypatch` that replaces the load seam with a raiser asserts the synthetic path never calls
    `lakeapi.load_data`.

## 10. Rollout

- **Phase 1 — dry-run / sample day selection (first implementation branch).** Build
  `ingest/trade_checks.py` (thresholds, reason codes, metric functions, `classify`,
  `estimate_trades_gb`, `quota_decision`, `build_report`/`write_report`/`_json_safe`) and
  `ingest/validate_trade_feeds.py` (CLI, day selection, calendar crossing, `--dry-run`). Ship the
  full synthetic test suite (§9). **No vendor calls** — `--dry-run` prints the plan; CI runs the pure
  tests only.
- **Phase 2 — bounded live validation (ask-first).** Run the **safe small default sample** (5 days ×
  3 venues, ≈1.3 GB, incl. the `2024-08-06` fill-routing case) once against Crypto Lake, write
  `trade_feed_validation.json`, and record measured
  per-venue `trades_gb_per_day` to replace the provisional estimates. `AGENTS.md`: live/vendor pulls
  only when the user explicitly asks; document GB used and that the report artifact is git-ignored.
  Optionally widen to the full regime cohort (still bounded, still `--allow-broad`-gated).
- **Phase 3 — integrate into usable-calendar / build manifest.** Feed each `(venue, day)` verdict
  into the calendar/build provenance: a required day failing trade validation for a venue becomes a
  new exclusion reason (`trade_validation_fail:<venue>`) or a fill route; the feature/build manifest
  records the trade-validation report path + `generated_utc` as a source artifact (the
  feature-manifest `sources` list already accepts extra keys).
- **Phase 3b — CoinAPI trade-fill validation path (gated; clears `coinapi_fill_deferred`).** Without
  this phase `coinapi_fill_deferred` can never empty, so `bars_ready` (and thus Phase 4) is
  permanently `false` for every span touching one of the **52 Coinbase trade-fill days** (§5b) — i.e.
  most of the 2-year calendar. It has three concrete pieces, all behind the **still-LOCKED §5a
  backfill gate** (bounded single-day pulls only once unlocked; **not run on any docs branch**):
  1. **Download** — extend the CoinAPI fetch to the `TRADES` dataset. `ingest/download_coinapi.py`
     today hard-codes `DATA_TYPE = "LIMITBOOK_FULL"` (book only); add a trades mode
     (`--data-type TRADES` → `coinbase_btc_file(..., "TRADES")`, partitioned under
     `data/raw/trades/exchange=COINBASE/symbol=BTC-USD/dt=…`). `verify_trades_and_calendar.py` already
     probes `TRADES` *presence* per fill day — this fetches the file.
  2. **Normalize** — map CoinAPI TRADES columns to the loaded Lake schema
     (`origin_time`, `received_time`, `price`, `quantity`, `side`∈{buy,sell}, `trade_id`): CoinAPI's
     `is_buy`/`taker_side` → `side`, its exchange/received timestamps → `origin_time`/`received_time`
     (§4.2/§4.3). The result is a **vendor-normalized frame** the *same* `ingest/trade_checks.py`
     validates unchanged (the pure module is source-agnostic — it never imports a vendor client).
  3. **Validate + clear** — run `validate_trade_frame` on the normalized CoinAPI frame for each fill
     day, emit its own pass/warn/fail verdict (same schema, `vendor_source: "coinapi"`), and on a
     pass/warn **remove that `(venue, day)` from `coinapi_fill_deferred`** so `bars_ready` can become
     true for its span. A CoinAPI-side fail keeps the day deferred (surfaced, not silently dropped).
- **Phase 4 — enforce in the bar builder.** The bar builder refuses to emit bars for a `(venue,
  span)` unless `summary.gate.bars_ready` holds for that span — i.e. `lake_required_pass` **and** no
  `coinapi_fill_deferred` entry touches it (every required day passed, and every fill day is a
  *validated* CoinAPI fill from Phase 3b). It gates on `bars_ready`, **not** `lake_required_pass`, so
  it can never span an unvalidated/locked Coinbase trade gap. Lake-only spans (no fill days) become
  buildable after Phase 2; fill-day spans wait on Phase 3b (hence the §5a backfill unlock).

## Implementation Tasks

Follow-up branches (this branch ships only the doc + the §11 `docs/data.md` edit). Each is TDD;
commit per task.

### Task 1 (Phase 1a): pure checks module + tests

- Create `ingest/trade_checks.py`: `TradeThresholds` (frozen dataclass + `as_dict()`), reason-code
  constants (§6), the per-metric functions (§4), `classify(metrics, thresholds, calendar_state) ->
  (status, reason_codes)`, `validate_trade_frame(df, venue, day, thresholds, calendar_state) ->
  dict`, `estimate_trades_gb`, `quota_decision` (mirroring `run_coinbase_quality_map`),
  `build_report`, `write_report`, `_json_safe`. Pandas/numpy only — **no lakeapi/boto3 imports**.
- Create `tests/test_trade_checks.py`: the §9 cases (1–12), `_trades_df` helper, `tmp_path` calendar
  fixtures, strict-JSON + byte-determinism. TDD (tests first).

### Task 2 (Phase 1b): Lake CLI wrapper + guard test

- Create `ingest/validate_trade_feeds.py`: argparse (§7), day selection (§3), `lake_session` reuse,
  the `load_data(table="trades", …)` call per `(venue, day)`, calendar crossing, the GB gate
  (`--dry-run`/`--allow-broad`/exit 5), `main(argv)` guarded under `if __name__ == "__main__"`, exit
  codes (§7).
- Add §9 case 13 (import-side-effect guard + load-seam monkeypatch) to `tests/test_trade_checks.py`
  (or a sibling `tests/test_validate_trade_feeds.py`).

### Task 3 (Phase 2): bounded live run + estimate refinement — **ask-first**

- Run the default sample once; commit no data/report; update `meta.trades_gb_per_day` defaults and
  §5b prose with measured multi-day results; annotate the §10 item (still open until the integration
  and enforcement phases 3, 3b, and 4 land).

### Task 4 (Phase 3): calendar / manifest integration

- Wire verdicts into the usable calendar (new `trade_validation_fail:<venue>` reason) and record the
  report as a build-manifest source artifact.

### Task 5 (Phase 3b): CoinAPI trade-fill validation path — **gated (backfill LOCKED)**

- Extend `ingest/download_coinapi.py` with a `TRADES` dataset mode (currently `LIMITBOOK_FULL` only);
  add a CoinAPI-TRADES → Lake-schema normalizer (`is_buy`/`taker_side` → `side`, timestamps →
  `origin_time`/`received_time`); run the **same** `ingest/trade_checks.py::validate_trade_frame` on
  the normalized frame (`vendor_source: "coinapi"`), and clear passing/warning fill days from
  `coinapi_fill_deferred`. Synthetic tests for the normalizer (CoinAPI-shaped frame → normalized
  schema → verdict) run with no vendor I/O; the live single-day CoinAPI pulls stay behind the §5a
  backfill gate and are **not run** until it unlocks.

### Task 6 (Phase 4): bar-builder enforcement (with the bar builder)

- Gate bar emission on `summary.gate.bars_ready` (not `lake_required_pass`). Lake-only spans build
  after Task 3; fill-day spans build only once Task 5 clears their `coinapi_fill_deferred` entries,
  which itself waits on the §5a backfill unlock.

## Validation Commands

This branch is **docs-only** — no code, so no pytest:

```bash
git diff --check          # no whitespace/conflict errors
git status -sb            # only the plan doc + the docs/data.md pointer edit
```

(Interpreter note for the follow-up branches: `.venv/bin/python -m pytest -q tests/test_trade_checks.py`;
agent worktrees have no `.venv` — use `/home/aaron/jepa-btc-forecasting/.venv/bin/python`.)

## PR Requirements

- Title: `docs: plan trade validation breadth`.
- Body: **Summary**; **Scope** (docs-only; the two-file split and rollout phases); **No vendor/API
  calls run**; **Validation** (`git diff --check`, `git status -sb`); **Risks and assumptions**
  (provisional `trades_gb_per_day` estimates until Phase 2 measures them; CoinAPI-fill-day trade
  validation deferred behind the still-locked backfill gate; thresholds are first-pass and
  Phase-2-tunable; the CoinAPI trade-fill path (Task 5) is required to clear `coinapi_fill_deferred`
  and stays behind the locked backfill gate); **Follow-ups** (Tasks 1–6).
- Commit only docs — no data, reports, parquet/csv.gz, caches, or secrets. **CoinAPI backfill status:
  still LOCKED.**

## Review Checklist

- [ ] Venue/product identifiers match `ingest/verify_trades_and_calendar.py` verbatim
      (`BINANCE_FUTURES`/`BTC-USDT-PERP`, `BINANCE`/`BTC-USDT`, `COINBASE`/`BTC-USD`, table `trades`).
- [ ] Engine clock = `origin_time` with an explicit `received_time` fallback and a stable
      sort-by-clock-then-file-order rule; Coinbase sort is asserted to repair monotonicity.
- [ ] Every JSON-facing string (statuses, reason codes) is a stable module constant; report is strict
      JSON (`allow_nan=False`, `jq empty` passes) and byte-deterministic apart from `generated_utc`.
- [ ] Fail vs. warn split is justified per code; CoinAPI-fill days route (not fail) and stay behind
      the locked backfill gate; excluded days are out of scope.
- [ ] The bar-builder gate is `bars_ready` (`lake_required_pass` **and** no deferred fills), never
      `lake_required_pass` alone — a span with an unvalidated/locked Coinbase fill is not buildable.
- [ ] No vendor I/O in the pure module or its tests; `--dry-run` makes no Lake calls; the GB gate
      reuses the `run_coinbase_quality_map` quota pattern (exit 5).
- [ ] Day selection has no broad default; the no-arg default sample stays under the auto cap.
- [ ] `docs/data.md` §10 item is annotated (pointer only), **not** marked done; no unrelated
      Coinbase quality-map sections rewritten.
