# Bar / Label / Modeling-Data Producer — Implementation Plan (E0.3 · E0.4 · E0.5)

> **Altitude.** This is a **producer architecture spec + task breakdown**, not a
> per-step TDD plan. Each Task (T1–T10) below is scoped to become its own
> `docs/superpowers/plans/…` TDD plan (or a small cluster) on a future Claude
> branch — the same "each phase gets its own detailed TDD implementation plan"
> convention the experiment plan sets ([`docs/experiment-plan.md:3`](../../experiment-plan.md)).
>
> **Companions.** [`jepa_btc_forecasting_spec.md`](../../../jepa_btc_forecasting_spec.md)
> §5–§8/§10 (design), [`docs/experiment-plan.md`](../../experiment-plan.md)
> E0.3/E0.4/E0.5 + Phase 1/G1 (gates), [`docs/feature-manifest.md`](../../feature-manifest.md)
> (the output contract), [`docs/data.md`](../../data.md) §5/§5a/§5b (coverage +
> backfill gate). Section refs like "§5.4" point to the spec.

**Goal.** Build the offline **producer** that turns reconstructed Binance +
Coinbase event streams into the exact
`data/processed/model_matrix.parquet` + `data/processed/feature_manifest.json`
that the already-built consumer `eval.runner.run_from_manifest`
([`eval/runner.py`](../../../eval/runner.py)) loads, validates, and gates (G1).
Pipeline: **notional bars (E0.3) → per-bar features (§6 / E1.2) → triple-barrier
labels + uniqueness (E0.4) → cost columns (E0.5) → v1 manifest**. The CPCV,
DSR/PBO, and no-trade-band cost math are **already implemented and tested**
(`data/cv.py`, `eval/`); this producer *feeds* them — it does not reinvent them.

**Tech stack.** Python ≥3.12, pandas/numpy (core), pyarrow (parquet). Optional
Rust `native/recon_native` for replay speed (pure-Python fallback is the
correctness oracle — [`recon/native.py`](../../../recon/native.py)). No new
production dependencies expected.

**Interpreter.** Commands use the repo-convention `.venv/bin/python` from the
repo root. Agent worktrees share the main checkout's venv — substitute
`/home/aaron/jepa-btc-forecasting/.venv/bin/python` (see memory
`worktree-venv-location`); commands are otherwise unchanged.

---

## What already exists vs. what this plan builds

The **consumption side is done** (PRs #14–#26). The **production side is
empty**. This plan is exactly the missing producer.

| Layer | Status | Where |
| --- | --- | --- |
| Event-time reconstruction (E0.1): merge trades+deltas, book-at-T, apply-before-read | ✅ built | `recon/reconstruct.py:sample_topk_as_of`, `recon/orderbook.py:OrderBook.snapshot(k)`, `recon.live.LiveReconstructor` (streaming); `recon/merge.py:merge_sorted` = **bounded-fixture oracle only, forbidden for full days** |
| Trade record (engine time, side, price, amount) | ✅ built | `recon/events.py:14` `Trade(ts_engine, seq, side, price, amount)` |
| Mid / microprice from a book snapshot | ✅ built | `recon/orderbook.py:60-69` |
| CPCV + per-interval purge + embargo | ✅ built | `data/cv.py:cpcv_splits(t_event, t0, t1, *, n_groups, k, embargo_ns)` |
| No-trade-band net PnL, uniqueness-weighted Sharpe | ✅ built | `eval/cost.py:net_pnl`, `eval/cost.py:weighted_sharpe` |
| DSR + PBO(CSCV) + G1 gate + per-regime + gross/net | ✅ built | `eval/study.py:run_study`, `eval/stats.py:deflated_sharpe`,`pbo` |
| v1 feature-manifest schema + frame validation + leak screen | ✅ built | `eval/manifest.py` (`validate_manifest`,`validate_frame`,`load_manifest`,`feature_list`,`target_list`,`leaky_feature_names`) |
| ModelMatrix reserved-column contract | ✅ built | `eval/matrix.py:RESERVED` + `validate_matrix` |
| Manifest-driven runner + CLI | ✅ built | `eval/runner.py:run_from_manifest`, `scripts/run_baseline.py` |
| τ decay-window helper (E1.1) | ✅ built (pure) | `eval/tau.py:predictivity_curve`,`estimate_tau` |
| Usable all-feed calendar (704/730 d, OOS≈Apr 2026) | ✅ built | `data/usable_calendar.json`, `ingest/verify_trades_and_calendar.py` |
| Vendor-seam fill policy + label/feature seam masks (E0.4 partial-fill contract) | ✅ built (helpers) | `recon/stitch_policy.py` (`SeamPolicy`, `DayStitchPlan`, `label_valid_mask`, `feature_valid_mask`, `window_crosses_seam`, `window_vendor_sources`) |
| **Notional-bar sampler (dollar clock + time cap)** | ❌ **build (E0.3)** | new `bars/` |
| **Wire the stitch plan + seam masks into the producer** | ❌ **build (E0.4)** | new `bars/produce.py` (§C/§F/T9) |
| **Per-bar features (OFI/CVD/depth/slope/microprice-dev/intra-bar path)** | ❌ **build (§6/E1.2)** | new `bars/` |
| **Cross-venue alignment + Binance→Coinbase feature lag** | ❌ **build (§5.3/§13)** | new `bars/` |
| **Triple-barrier labels + forward returns + uniqueness** | ❌ **build (E0.4)** | new `data/labels.py`, `data/uniqueness.py` |
| **Per-row cost columns (`cost_bps`,`half_spread_bps`)** | ❌ **build (E0.5 producer side)** | new `bars/` |
| **ModelMatrix assembly + manifest emission** | ❌ **build (E0.3/E0.4)** | new `bars/produce.py`, `eval.manifest` writer |

> The only existing label code is `recon/parity.py:_signed_labels` — a
> **parity-comparison** directional label, **not** a training label. No
> triple-barrier, no forward return, no bar sampler exists (confirmed:
> `bars/` is absent).

**Explicitly NOT in scope of the producer plan (Non-goals):**

- The consumer (`eval/`), CPCV (`data/cv.py`), and cost/DSR/PBO math — built; we
  conform to them, we do not touch their signatures (21+ test call sites depend
  on them — see [`docs/superpowers/plans/2026-07-02-lightgbm-manifest-integration.md`](2026-07-02-lightgbm-manifest-integration.md)).
- Ingestion / backfill / vendor download (`ingest/`, `recon/coinapi.py`,
  `download_*`) — untouched. **No live vendor calls.**
- JEPA (`model/`,`train/`), the live loop, and the maker/passive-fill economics
  arm (a documented follow-up, not built — §G).
- "All non-reserved columns" feature inference — **forbidden** by AGENTS.md;
  `eval.manifest.unsafe_infer_feature_cols` is exploration-only and the producer
  writes an **explicit** `feature_cols` list (§H).

---

## Architecture

```
recon/ (built)                     bars/ (NEW)                         data/ (extend)         eval/ (built, consumes)
─────────────────┐    ┌──────────────────────────────────────┐   ┌───────────────────┐   ┌──────────────────────┐
stream-merge ────┼───▶│ clock.py     dollar bars + time cap   │   │ labels.py         │   │ manifest.validate_   │
Trade/Delta      │    │ snapshot.py  dual-book @ bar close     │──▶│  triple-barrier    │──▶│  frame  (contract)   │
OrderBook.snap ──┼───▶│ features.py  OFI/CVD/microprice_dev…   │   │ uniqueness.py     │   │ runner.run_from_     │
sample_topk_as_of│    │ align.py     cross-venue lag + basis   │   │ cv.py (BUILT)     │──▶│  manifest → G1       │
                 │    │ cost.py      cost_bps/half_spread_bps  │   └───────────────────┘   │ study/stats/cost     │
                 │    │ produce.py   assemble → parquet+manifest│                          └──────────────────────┘
                 │    └──────────────────────────────────────┘
```

**Module layout.** New top-level `bars/` package (matches spec §3 "`bars/`" and
sits above the built `recon/`). Labels/uniqueness extend the existing `data/`
package alongside the built `data/cv.py`. Manifest *writing* is a thin helper in
`eval/manifest.py` (the module that already owns the schema), so the producer and
consumer share one contract definition.

**Data flow (one instrument-day, then consolidate):**
a **streaming, day-partitioned k-way merge** of Binance+Coinbase deltas+trades on the
engine-time axis (§C.1; `recon.merge_sorted` is the bounded-fixture oracle **only**, never
the full-day path) → `bars.clock` (trailing-threshold schedule) emits bar boundaries →
`bars.align` sets the received-time **decision `t_event`** and per-venue reads (§C.2) →
`bars.snapshot` snapshots **both** books via `recon.reconstruct.sample_topk_as_of`
(apply-before-read, strict `<`; every event gated by its own **`received_time ≤ t_event`** —
ordered by `origin_time`; the Coinbase target book resolves to `coinbase_read_ts`, the last
observed book origin, §C.2) → `bars.features` builds the per-bar stationarized
vector → `data.labels` + `data.uniqueness` attach `y_fwd_bps`/`label`/`t_barrier`/
`uniqueness` (**per horizon**, off Coinbase mid) → `bars.cost` attaches `cost_bps`/
`half_spread_bps` → **guard-aware `stitch_policy` masks + `window_vendor_sources` drop
cross-seam/guard/uncovered rows** (§C.3) → per-day parquet → consolidate the labeled window
→ **`validate_frame` + `validate_matrix`** (fail closed — the NaN/inf/finite screens live in
`validate_matrix`, §H/T8) → write `model_matrix.parquet` + `feature_manifest.json`.

---

## §A. Bar clock construction (E0.3)

**Clock.** Emit a bar when **cumulative traded notional** crosses a threshold **or**
`T` seconds elapse, whichever first (spec §5.1–5.2 hybrid time cap). Clock off the
**trade stream**, never the book-update stream (§5.1: trades carry realized
aggression; quote churn is the noise JEPA discards). **Dollar, not volume** (§5.1:
BTC ranges 2×+; dollar bars are homoscedastic).

- **Reference stream — pre-E2.5 default = Coinbase/target-venue-triggered notional**
  (`price × amount` over Coinbase `recon.events.Trade`). **Why not the spec's
  Binance-perp clock yet:** a Binance-triggered close (a Binance **trade**) is observable only
  at its own `received_time`; offline that is exact per-event, but the **live** watermark needs
  a Binance trade-lag **tail (p99)** that E2.5 pins — until then it is unquantified for the live
  loop. A Coinbase-triggered close is a **local** trade whose `received_time` is known today
  (data.md §5b), so `t_event =` its receipt is fully quantified now (§C.2). **Post-E2.5
  target (spec §5.1, the information-optimal clock):** Binance-perp notional (deepest venue's
  aggression), **enabled once the Binance trade/book lags are pinned** — then §C.2 uses them.
  *Ablation knob:* combined Binance+Coinbase notional (§5.1 open; E2.2 resolves on
  downstream PnL). The trigger venue is a manifest parameter, so the E2.5 switch is a
  config change, not a rebuild-forcing rewrite.
- **Adaptive threshold** (E0.3) — **trailing / as-of only.** `threshold_d =
  rolling_7–30d_avg_dollar_volume(days < d) / target_bars_per_day`, computed from
  **prior days only** (strictly `< d`), tuned for **~1 bar / 0.5–2 s** in active regimes.
  Using day `d`'s own completed volume would leak future volume into `d`'s bar boundaries
  — a subtle sampling look-ahead. **Warm-up:** the first `warmup_days` (no full trailing
  window) use a fixed seed threshold and are flagged/excluded from the labeled matrix.
  The build records the **full per-day threshold schedule + its content hash** in the
  manifest (`bar_clock.threshold_schedule_hash`), not a single scalar — so the sampling is
  reproducible and auditably causal.
- **Time cap** `T` (≈2–5 s): bounds worst-case heteroscedasticity (§5.2 — a dead Sunday
  must not become one 45-min bar). Emit a **`emitted_by_time_cap`** boolean per bar
  (diagnostic `extra_cols`, opted-in via the manifest — never a feature).
- **Ordering gotcha (data.md §5b, measured 2025-06-01):** Binance trade feeds are stored
  in `origin_time` order with monotonic `trade_id`; **Coinbase trades are NOT sorted by
  `origin_time` and `trade_id` is not monotonic** — the clock **must sort Coinbase trades
  by `origin_time`** before accumulating. Total order is defined by `recon.events.order_key`
  (honored by both the streaming production merge and the `merge_sorted` fixture oracle);
  the sampler must not assume file order.

**Gate (E0.3):** median active-regime bar ≤ 2 s (so the 2 s horizon ≈ a few bars) +
the **log-scale time-per-bar histogram** artifact. Threshold calibration and this gate
require the real backfilled trade volume → **post-backfill** (see split). The clock
*code* and its determinism are testable now on synthetic/fixture trades.

**Builds on:** `recon/events.py:Trade` + `order_key`; the streaming k-way merge
(`recon/merge.py:merge_sorted` = fixture oracle only, §C.1).

---

## §B. Coinbase target mid / microprice inputs

The label is defined on the **Coinbase BTC-USD** book (the venue we trade), off
**mid and microprice — never last-trade** (spec §5/§6, E0.4). Both come directly from
the existing snapshot:

- `recon/orderbook.py:60-62` → **mid** = (best_bid + best_ask)/2.
- `recon/orderbook.py:64-69` → **microprice** = (ask_size·best_bid + bid_size·best_ask)/
  (bid_size + ask_size) — size-weighted fair value, robust to bid-ask bounce.

**Default:** label off **Coinbase mid** (primary anchor), carry **microprice** as an
ablation arm. Rationale for defaulting to mid despite microprice being the better
short-horizon fair-value proxy: **the vendor/seam parity gate that unlocks backfill
validates labels on mid, not microprice** — `recon/parity.py:_signed_labels` computes the
directional label from `mid` and `label_agreement` is fed `L["mid"]`/`C["mid"]`
(`recon/parity.py:68,243`). Using microprice as the primary target would train on an anchor
whose cross-vendor agreement at the stitch seams has never been validated. **Microprice as
primary is therefore gated on first adding a microprice-parity check** to the seam gate
(a prerequisite, tracked as a follow-up / T5 open question). The anchor is a
manifest-recorded label parameter. **Promoting microprice is a label rebuild, not a manifest
one-liner (Codex P2):** the triple-barrier `label`/`y_fwd_bps`/`t_barrier` are computed off the
anchor's price path and the v1 runner consumes exactly one target pair, so a per-bar microprice
*value* column carries no future barrier hits — the microprice arm needs T5 to re-run labels off
the microprice path (or emit a separate precomputed microprice target set). Emitting both mid and
microprice *base-price series* per bar is cheap; the *labels* are not free to re-anchor.

The producer snapshots the Coinbase target book at **`coinbase_read_ts`** — the last Coinbase
book state with `received_time ≤ t_event` (§C.2), **never at `t_event`** — the target mid, microprice, **and**
`half_spread_bps` (§G) all read at the lagged timestamp so labels and cost never use future
Coinbase book state (Codex P1). It records both mid and microprice as the label anchor
series; the forward label (§D) uses this lagged mid/microprice as the base price `P0`, with
the triple barrier and the emitted span running over `[t_event, t_barrier]` (§C.2).

---

## §C. Binance & Coinbase feature alignment (event-time, decision-time, seams)

Two async venues, two async channels each. Alignment reuses the built E0.1 machinery
and adds **decision-time**, **cross-venue latency**, and **vendor-seam** discipline.

1. **Single engine-time axis — streaming in production (Codex P2).** Reconstruct each
   venue's book by merging its trades + L2 deltas on `ts_engine` (`origin_time` is **100%
   populated** for `book_delta_v2` on both venues per data.md §5 — exchange time, no
   `received_time` fallback needed). **Production uses a streaming, day-partitioned k-way
   merge** over already-sorted/chunked inputs: `recon/merge.py:merge_sorted` **materializes
   both streams into one list and its docstring explicitly forbids full-day use** (Binance
   perp `book_delta_v2` ≈ 109 M rows/day, ~4 GB — AGENTS.md streaming rule), so it stays the
   bounded-fixture oracle the replay-equivalence test pins. T2/T9 reuse the streaming
   watermark merge `recon.live.LiveReconstructor` already implements, or the deferred Rust
   k-way merge (spec §3). Snapshot = book inclusive of all events with `ts_engine ≤` the read
   time, **strict `<` apply-before-read at the trade boundary**
   (`recon/reconstruct.py:sample_topk_as_of`, `order_key` deltas-before-trades).
   **CoinAPI fill segments are the exception (Codex P2), for *both* book and trades:**
   - **Book:** Coinbase gap days filled from CoinAPI `limitbook_full` must replay in strict
     **`seq` (file) order via `recon.coinapi`**, *not* a `ts_engine` merge — the opening
     SNAPSHOT block carries a **prior-day** `time_exchange` (data.md:134-140;
     `recon/coinapi.py:13-29`) that a timestamp merge would sort to day-end, corrupting the
     target book.
   - **Trades (new producer prerequisite):** the pre-E2.5 clock triggers on **Coinbase
     trades**, but the **52 Coinbase fill days** (data.md §5b — 47 need book, all 52 need
     trades, 2.6 GB, verified present in CoinAPI flat files) have **no Lake trades**, and a
     **CoinAPI trades normalizer/replay does not exist yet** (`recon/coinapi.py` is book-only;
     `download_coinapi.py` emits `limitbook_full` only). It is a **T1/T9 prerequisite** —
     without it the producer cannot emit bar closes or CVD on fill days and would silently
     build the claimed 704/730 usable matrix from Lake-only trades, dropping/corrupting exactly
     the backfilled calendar the plan depends on. **CoinAPI timestamps are ns-since-midnight,
     not absolute (Codex P1):** convert first — `received_time = day_open_ns + time_coinapi_ns`,
     origin `= day_open_ns + time_exchange_ns` (snapshot block clamped to day open, per
     `recon.coinapi`) — **before** the `received_time ≤ t_event` gate, or a raw time-of-day
     compare passes for every after-midnight event. Any tail-lag bound is CoinAPI-specific
     (`time_coinapi_ns − time_exchange_ns`), not the Lake figure.
   - **Dispatch:** T9 routes per fill segment by `vendor_source` (Lake → `ts_engine` streaming
     merge over Lake book+trades; CoinAPI → `seq`-order book replay **plus** the CoinAPI trades
     normalizer) before sampling.

2. **Decision time and per-event observability — the sample-timing rule (§13 pitfall, load-bearing).**
   Reconstruction runs on **exchange (origin) time** (the canonical book order), but an event
   is only *observable* at the trading box at its own **`received_time`** — captured per event
   alongside `origin_time` on both Lake venues (data.md §5/§5b; `recon/ingest.py` already
   carries both) and as `time_coinapi_ns` for CoinAPI fill days (§4.3). So the read rule is
   **exact and per-event, not a lag constant:**
   - **Decision time `t_event`** is on the box's received-time axis; every input is included
     **iff its `received_time ≤ t_event`** (ordered by `origin_time`, *gated* by
     `received_time`) — no median/constant approximation, so no delayed event leaks in.
   - **Trigger / bar close:** a bar closes on a **trade**; `t_event =` the trigger trade's
     **`received_time`** (not `origin + a lag constant`), so the decision is never placed
     before the trade is actually received.
   - **Book snapshots** (both venues — incl. the Coinbase target mid/microprice **and**
     `half_spread_bps`): include book events with `received_time ≤ t_event`; the Coinbase
     target **`coinbase_read_ts`** is the **origin time of the last such book event**
     (`≤ t_event`), so labels and cost never use unobserved future book state (§B).
   - **Trade-flow features** (CVD, aggressor imbalance, largest print): include trades with
     `received_time ≤ t_event`.
   - **Label anchor (Codex P2):** base price `P0` = the last-observable Coinbase mid/microprice
     at `coinbase_read_ts`, but the **triple barrier + emitted span run over `[t_event, t_barrier]`**
     (`t0 = t_event`, `t_barrier ≥ t_event`) — aligned with the CPCV purge/embargo (§F) and seam
     masks (§C.3); the lagged read supplies only `P0` (charging realistic latency drift into
     `y_fwd_bps`) and leaves **no** barrier hit in the `[coinbase_read_ts, t_event]` gap unpurged.
   - `t_available == t_event` is **correct by construction** — every input has
     `received_time ≤ t_event`; `availability_lag_ns` stays 0 (§E).
   - **Lag constants are TAIL bounds, never medians (Codex P1):** where a scalar is unavoidable —
     the **live** loop's straggler watermark and the manifest's *declared* bound — use a **pinned
     p99/max** feed lag, not the median. data.md reports median/p95 pairs (e.g. Coinbase trades
     164/238 ms, Binance perp book 4.4/149 ms); the **median is not conservative** — ~50 % of
     events exceed it, so reading at `t_event − median` would leak delayed events. Offline builds
     use actual `received_time` (exact); E2.5 measures the Binance figures before the
     Binance-triggered clock is enabled (§A, Q2).

3. **Vendor-seam exclusion — the partial-fill contract (E0.4, hard input).** Coinbase is
   stitched from Crypto Lake + CoinAPI at vendor **seams** (the 33-day hole and smaller
   gaps, data.md §5a/§5b). The producer **must consume the final reviewed per-day stitch
   plan** (`recon/stitch_policy.py:DayStitchPlan` — `.seams`, fill `segments`, `SeamPolicy`
   with `seam_guard_s = 60 s` = the longest label horizon) and **drop any bar row whose
   feature window `[t_feature_start, t_event]` or label window `[t_event, t_barrier]`
   crosses a seam or touches its guard band, or whose windows are not backed by a single
   vendor** (`{LAKE}` or `{COINAPI}`). Use the built helpers directly. For the **seam +
   guard-band** test use the guard-aware **`feature_valid_mask(...)` / `label_valid_mask(...)`**
   (`recon/stitch_policy.py:391,406` — both take `guard_ns` and reject a window that crosses
   a seam *or* touches its ±`guard_ns` band), **not** `window_crosses_seam` alone, which has
   **no `guard_ns`** (`recon/stitch_policy.py:385`) and would let a window sitting inside the
   60 s guard band survive (Codex P2). For the **vendor-coverage** test use
   **`window_vendor_sources(start, end, segments)` (`recon/stitch_policy.py:430`)** — the row
   is kept only when *both* its feature window
   `[t_feature_start, t_event]` and label window `[t_event, t_barrier]` return a singleton
   `{lake}` or `{coinapi}`; any mixed-vendor, `excluded`, or `UNCOVERED` (day-edge overhang)
   window is masked. **Do not use `vendor_source_at(...)` for this** — it is per-*sample*
   and only sees the endpoint, so it would miss an excluded/uncovered span *inside* the
   window (`label_valid_mask`/`feature_valid_mask` cover the seam/guard geometry on a
   regular grid but, per their own docstrings, must be intersected with the whole-window
   vendor set). Never train across a seam
   (`SeamPolicy.exclude_labels_crossing_seam`/`exclude_features_crossing_seam`). The
   consumed stitch-plan id + `SeamPolicy.as_dict()` are recorded in the manifest `sources`
   / `bar_clock` for reproducibility. This is a **T9 assembly step and a T9/T10 acceptance
   criterion** (§Task breakdown). The masking *code* is testable now on synthetic seams;
   the *final* seam list is a product of the backfill/parity gate (§before/after split).

4. **Cross-venue features:** Binance−Coinbase basis (mid spread), lagged Binance
   OFI→Coinbase (E1.2 #1 cross-venue signal). Perp state (funding/OI/liquidations) as
   **conditioners, not primary** (E2.4: OFI ≫ funding/OI at these horizons).

---

## §D. Label horizons

- **Ladder (default):** `{2s, 10s, 60s}` — the spec's band. E1.1 (`eval/tau.py`) measures
  τ on the real data and **adds a ~20–30 s rung near the decay knee**; 60 s stays as a
  decay/control arm. Horizons are a manifest field (`horizons: {"10s": 10_000_000_000,
  …}`) and a per-row `horizon` tag — multi-horizon is native to the schema. **Adding the τ-rung
  is not just a manifest edit (Codex P2):** the runner groups actual rows by `horizon` and
  rejects a declared horizon missing from the matrix, so T10 must **rerun label/matrix
  production** to emit the new bar×horizon rows (likely no code change, but the artifacts are
  rebuilt).
- **Vertical barrier = physical horizon** (§5.4 decoupling: input clock is notional, the
  *target* is fixed physical time). One matrix row per (bar, horizon) tag; the built
  runner groups by `horizon` and gates each rung independently
  (`eval/runner.py`, `eval/study.py`).
- **`y_fwd_bps`** = normalized forward return (bps) over the span **`[t_event, t_barrier]`**
  (decision time, §C — *not* the bar close), base price `P0` = the last-observable Coinbase
  target **mid** (§B default; microprice arm gated on parity) at `coinbase_read_ts` (§C.2);
  **never raw price** (§8).

---

## §E. `t_event`, `t_available`, and the no-lookahead rules

The producer must satisfy every invariant `validate_matrix`
([`eval/matrix.py:58-79`](../../../eval/matrix.py)) and `validate_frame`
([`eval/manifest.py`](../../../eval/manifest.py)) already enforce — these are the
contract, restated as production rules:

| Column | Definition (producer) | Enforced invariant |
| --- | --- | --- |
| `t_event` | **decision** time on the **received-time** axis = the trigger trade's `received_time` (the close is a trade; §C.2); every input gated by `received_time ≤ t_event`; **not** the raw bar close | int64 ns, non-null |
| `t_feature_start` | origin time of the **oldest** look-back event observed by `t_event` (`received_time ≤ t_event`) | `t_feature_start ≤ t_event`; observed look-back ≤ `max_lookback_ns` |
| `t_available` | when features become usable = **`t_event`** (synchronous) | `t_available == t_event` (every input has `received_time ≤ t_event` per §C.2, `availability_lag_ns = 0`) |
| `t_barrier` | first-barrier-hit time (TP/SL/time), forward from `t_event` | `t_event ≤ t_barrier ≤ t_event + horizons[tag]` |

**Seam integrity (§C.3):** additionally, every emitted row's `[t_feature_start, t_event]`
and `[t_event, t_barrier]` windows must be seam-/guard-clean and single-vendor-backed
(`recon/stitch_policy.py`); rows failing the masks are dropped, never NaN-carried into the
matrix (**`validate_matrix` rejects NaN/inf features** — §H; `validate_frame` covers only
columns/timing/dtypes, Codex P2). This is the value-level complement to the timing invariants
below.

**Value-level no-lookahead is the producer's own gate.** The manifest validates
*declared* timing and screens *names*; it does **not** prove feature *values* were
computed causally — that is producer-side work
([`docs/feature-manifest.md:18-20`](../../feature-manifest.md); reaffirmed as an
open gap in [`2026-07-02-lightgbm-manifest-integration.md`](2026-07-02-lightgbm-manifest-integration.md):
"the equivalent guard for bar/feature computation must land with the E0.3 feature
producer, which does not exist yet"). This plan **closes that gap** with a
bar/feature replay-equivalence test (§J, T3) mirroring the built
`tests/test_reconstruct_no_lookahead.py` / `tests/test_replay_equivalence.py`: feeding
deliberately out-of-order (live-shaped) events must yield **byte-identical** per-bar
features to the offline build.

---

## §F. Embargo / CPCV split generation

**No new CV code — wire to the built `data/cv.py`.** The producer's job is only to emit
the columns CPCV consumes and to pin `embargo_ns` correctly:

- `cpcv_splits(t_event, t0, t1, *, n_groups, k, embargo_ns)` is called by the runner with
  `t0 = t_event`, `t1 = t_barrier` (the label **span**). Purge is **per test interval**
  (`data/cv.py:54-65`) — correct for non-contiguous CPCV combos — so no train label span
  can straddle a test span regardless of embargo. The producer just guarantees
  `t_barrier` is the true resolution time.
- **`max_lookback_ns` spans to the decision time, including the observation delay (P3).** The
  producer sets `max_lookback_ns = max(t_event − t_feature_start)` over all rows, where
  `t_feature_start` is the **origin time of the earliest look-back event observed by `t_event`**
  (`received_time ≤ t_event`, §C.2). Measuring to `t_event` therefore **absorbs each feed's
  observation delay** (and, under the Binance-triggered clock, the gap between the trigger
  trade's origin and its `received_time = t_event`); it
  is **not** the raw feature-window length. Undersizing it would let a post-test train row's
  feature window reach into the test label span. The consumer cross-checks exactly this
  quantity (`eval/study.py:28` uses `(t_event − t_feature_start).max()`) and fails closed,
  but the producer must size it right at build time.
- **Embargo (E0.4): `embargo_ns ≥ max(label horizon, longest feature look-back)`.** The
  span-overlap purge already covers the label-horizon side; `embargo_ns` guards the
  **feature look-back** side after a test block. The producer sets
  `embargo_ns = max(max_lookback_ns, max horizon_ns)` and the schema enforces
  `embargo_ns ≥ max_lookback_ns` (`eval/manifest.py`), re-checked at runtime against the
  observed per-row look-back (`eval/study.py:28-32`). This makes the analysis in the
  LightGBM-integration plan's "Embargo vs label horizon" note hold by construction.
- **Uniqueness** = 1/(# label spans covering `t_event`) — the concurrency weight
  (`eval/synthetic.py:_concurrency_uniqueness` is the reference to port to
  `data/uniqueness.py`); optional sequential-bootstrap weights are a follow-up. **Compute it
  per horizon (Codex P2):** the matrix is one row per bar×horizon and the runner evaluates
  each horizon in its own `groupby("horizon")` slice (`eval/runner.py:60`), so concurrency
  must count only **same-horizon** spans. Porting `_concurrency_uniqueness` over the whole
  multi-horizon matrix would count the duplicated 2s/10s/60s rows at the same `t_event`
  against each other and depress weights, effective-trade counts, Sharpe, and PBO for every
  rung — even though those horizons are never evaluated together. Feeds the
  uniqueness-weighted Sharpe and PBO block weights (`eval/study.py:57`).
- **Seam-excluded rows are absent, not imputed** (§C.3): the seam/guard/vendor masks run
  *before* CPCV, so `cpcv_splits` and the embargo operate only on clean, single-vendor
  rows. A label span that would have straddled a seam simply does not exist — there is no
  cross-vendor span for the purge to reason about.
- **E0.4 leakage-control gate:** a deliberately-leaky control (random k-fold, no purge)
  must show **inflated** CV vs the purged/embargoed pipeline — a synthetic PASS/FAIL test
  (§J, T6) proving the controls bite.

---

## §G. Cost / eval assumptions for the LightGBM signal gate (E0.5)

**The evaluator is built** (`eval/cost.py`, `eval/study.py`, `eval/stats.py`). The
producer's only E0.5 obligation is to **emit honest per-row cost inputs**; this section
documents the assumptions the gate runs under so they are pre-registered (experiment-plan
cross-cutting discipline).

- **No-trade band** (`eval/cost.py:net_pnl`): trade only when `|forecast| >
  cost_bps + 2·half_spread_bps + margin_bps`. Round-trip taker crosses the spread twice
  (`spread_crossings=2`). **Honest taker fills** — no passive-fill-at-mid assumption.
- **`cost_bps`** (per row) = `2 × taker_fee_bps + slippage_bps`. Coinbase Advanced taker
  ranges ~120 bps (base) → ~5 bps (top tier) (spec §10.2); the assumed fee tier is a
  **pre-registered producer parameter**, recorded in the manifest `sources`/`bar_clock`
  block, not hidden in code. At realistic solo volume the cost wall is large — this is
  the G1 stakes (data.md/§10).
- **`half_spread_bps`** (per row) = ½·(Coinbase best_ask − best_bid)/mid from the target book
  at **`coinbase_read_ts`** (the last-observed Coinbase book, `received_time ≤ t_event`,
  §B/§C.2 — not `t_event`, so cost never uses future book state). `validate_matrix` requires
  both cost columns ≥ 0.
- **Gate (G1):** `run_study` reports gross **and** net side-by-side, MCC, DSR (vs the trial
  dispersion, effective-N), and **PBO via CSCV** (needs ≥32 finite OOS samples else
  `g1_inconclusive`; `eval/study.py:62`). A real G1 pass needs enough traded samples
  (`min_trades=30`, `min_eff_trades=10`; `eval/runner.py:DEFAULT_GATE`) → **post-backfill**.
- **E0.5 sanity gate (already built, keep green):** the evaluator scores a known-zero-edge
  synthetic series as DSR≈0 / PnL≤0 (`tests/test_gate_synthetic.py`) — it does not
  manufacture edge.
- **Documented gap (follow-up, not built):** a maker/selective-taker fill arm. The spec
  (§10.2) notes patient-maker is the only economic option at solo volume; the current
  harness models taker only. Flagged here so it is a conscious pre-registration, not an
  omission.

---

## §H. Feature manifest production and validation

The producer **writes an explicit v1 manifest** (`eval/manifest.py:MANIFEST_VERSION == 1`)
— never "all non-reserved columns" (AGENTS.md; `unsafe_infer_feature_cols` is
exploration-only). A thin `eval.manifest.write_manifest(manifest, path)` +
`build_manifest(...)` helper (new, same module that owns validation) serializes JSON with
sorted keys.

**Emitted manifest (v1 required fields, `eval/manifest.py`):**

- `manifest_version: 1`, `dataset_id`, `build_id` = **content hash over the matrix bytes +
  all build params, with `generated_at` EXCLUDED** (a wall-clock timestamp must not change
  the identity of an otherwise-identical build — §I/P3), `time: {unit: "ns", timezone:
  "UTC"}`.
- `bar_clock: {kind: "dollar", reference_stream, target_bars_per_day, time_cap_ns,
  warmup_days, threshold_schedule_hash, feed_lag_tail_ns {binance_book, binance_trade, coinbase_book, coinbase_trade — p99/max live-watermark bound; offline reads use per-event received_time, §C.2}, seam_policy}` —
  the **per-day trailing threshold schedule is pinned by hash** (§A, not a scalar), and
  `seam_policy` is `recon/stitch_policy.py:SeamPolicy.as_dict()` (§C.3). `emitted_by_time_cap`
  is an opted-in diagnostic `extra_cols`, never a feature.
- `feature_cols`: **explicit ordered** list (§below). `target_cols: ["y_fwd_bps",
  "label"]` (exactly what the baseline consumes — `eval/runner.py:BASELINE_TARGETS`).
  `reserved_cols`: full `eval.matrix.RESERVED`.
- `venues`: `[{exchange:"BINANCE",symbol:"BTCUSDT",role:"signal"}, {…perp…},
  {exchange:"COINBASE",symbol:"BTC-USD",role:"target"}]`.
- `horizons`: `{tag: ns}` (§D). `sources`: `["crypto-lake/book_delta_v2",
  "coinapi/limitbook_full", <reviewed-stitch/backfill-manifest-id>, …]` + fee-tier
  assumption — the **consumed stitch plan is a recorded source** (§C.3). `generated_at`:
  ISO-8601 UTC, **injectable** (a build param, fixed in tests) and **excluded from
  `build_id`** so identical rebuilds are byte-identical (§I/P3).
- `max_lookback_ns`, `embargo_ns` (`embargo_ns ≥ max_lookback_ns`, §F);
  `availability_lag_ns: 0` (synchronous — §C/§E). Optional `extra_cols`
  (`emitted_by_time_cap`, funding/OI diagnostics), `dtypes`, `gate` (the pre-registered
  G1 block, `eval/runner.py:resolve_gate`).

**Feature registry (explicit `feature_cols`, §6 / E1.2, stationarized).** Core names align
with the existing synthetic stand-ins (`eval/synthetic.py:FEATURES`) so fixtures
interoperate: `ofi_integrated` (multi-level Cont-style OFI — E1.2 #1), `microprice_dev`,
`queue_imb`, `spread_tick` (also the regime tag source), `cvd`; plus `depth_imbalance`,
`book_slope`, `vwap_minus_mid`, `trade_count`, `signed_vol`, `aggressor_imb`,
`largest_print`, `rv_intrabar`, `mae_intrabar`, `basis_binance_coinbase`,
`ofi_binance_lagged`, `elapsed_ns`, `tod_sin`/`tod_cos`. Perp state
(`funding`,`oi_change`,`liq_intensity`) enters as low-priority features or `extra_cols`
conditioners (E2.4). Final list is pinned per build in the manifest.

**Validation before write (fail closed):** the producer runs **both** `validate_frame(matrix,
manifest)` (columns/timing/leakage/horizons/dtypes) **and** `eval.matrix.validate_matrix(matrix,
feature_list(manifest))` — the NaN/inf/finite/duplicate screens that otherwise only run later
inside `run_study` (`eval/matrix.py`) — before writing. `validate_frame` **alone** would let a
matrix with NaN/inf features or costs persist and fail only at eval (Codex P2); running both (or
the full `run_from_manifest`) means a bad build **never** reaches `data/processed/`. A round-trip
test (§J) then runs the artifact through `run_from_manifest` — the same path the CLI uses.

**`regime`** column: default `spread_tick`-bucketed `{tight, wide}` (matches the built
per-regime slicing `eval/study.py:78` and `eval/synthetic.py`); volatility-regime
stratification is an additive tag (experiment-plan cross-cutting discipline).

---

## §I. Deterministic output paths and schemas

**Paths (all under git-ignored `data/` — AGENTS.md forbids committing vendor/raw data):**

- Per-day intermediates: `data/interim/model_matrix/dt=YYYY-MM-DD.parquet` (day-partitioned
  — AGENTS.md performance rule: multi-GB/day, never load the full window at once).
- Consolidated artifact (the labeled window): **`data/processed/model_matrix.parquet`** +
  **`data/processed/feature_manifest.json`** — the exact paths the integration test and
  CLI expect (`tests/test_baseline_integration.py`, `scripts/run_baseline.py`).

**Determinism (repo convention — `plan_lake_binance_batches.py`).** Determinism is defined
as **identical `model_matrix.parquet` bytes + identical `build_id`**, where `build_id` is
the content hash over (rows + all build params) **excluding the manifest `generated_at`**.
Rules: iterate the **sorted** usable-day list from `data/usable_calendar.json`; apply the
seam masks (§C.3) deterministically from the pinned stitch plan; stable-sort matrix rows by
`t_event` (also required for reproducible PBO blocking — `eval/study.py:58-61`); pin the
per-day trailing threshold schedule by hash (§A); seed all RNG
(`np.random.default_rng(seed)`). **`generated_at` is an injectable build param** (fixed in
tests, real wall-clock in production) and is the *only* field allowed to differ between two
otherwise-identical builds — it never enters `build_id` or the matrix bytes. This
reconciles the byte-identical-rebuild test (§J) with a real generation timestamp.

**ModelMatrix schema (one row per bar×horizon):** `feature_cols` (float, no NaN/inf) +
`RESERVED` = `y_fwd_bps`(float bps), `label`(int ∈{-1,0,1}), `t_event`/`t_barrier`/
`t_feature_start`/`t_available`(int64 ns), `cost_bps`/`half_spread_bps`(float ≥0),
`uniqueness`(float ∈(0,1]), `regime`(str), `horizon`(str tag) + opted-in `extra_cols`.

---

## §J. Synthetic tests and small fixture tests

Two tiers, matching the repo (pytest, `tests/`, seeded fixtures, skipif-gated real-data
tests — `tests/conftest.py:FIXTURES`, `tests/test_fixture_integration.py`).

**Tier 1 — synthetic + tiny committed fixtures (run in CI, no vendor data):**

- **Bar clock:** seeded trade stream → deterministic bar boundaries; a burst triggers on
  notional, a lull triggers on the time cap (`emitted_by_time_cap`); Coinbase-order
  scramble still yields identical bars (mirrors `tests/test_sample_reconstruct.py` scramble
  test). **PASS/FAIL:** median-bar-time on a planted active regime ≤ 2 s.
- **Threshold causality (P2b):** injecting a large volume spike on day `d` must **not**
  change any bar boundary on day `d` or earlier (the trailing threshold sees only days
  `< d`); warm-up days use the seed threshold and are flagged/excluded.
- **Decision-time / sample-timing (P1):** the label runs forward from `t_event`; **every read
  event has `received_time ≤ t_event`** — planting a **delayed** event (`received_time >
  t_event` but `origin_time ≤ t_event`) must **not** enter the snapshot/features (regression
  guard: a median-lag read would wrongly include it). Asserts `t_event ==` the trigger trade's
  `received_time`, the Coinbase target book resolves to `coinbase_read_ts` (last observed, not
  `t_event`), and `t_available == t_event` holds *without* look-ahead.
- **Seam masking (P2a):** a synthetic day with a planted seam (two vendor segments +
  `SeamPolicy` guard) drops every bar whose feature/label window crosses the seam **or sits
  inside the guard band** (guard-aware `feature_valid_mask`/`label_valid_mask`, **not**
  `window_crosses_seam` alone), and every window whose `window_vendor_sources` is not a
  singleton `{lake}`/`{coinapi}` — including the endpoint-clean-but-**`excluded`/`UNCOVERED`
  span *inside* the window** case (the per-sample `vendor_source_at` would miss it). No
  surviving row spans a seam or guard band
  (`recon/stitch_policy.py:label_valid_mask`/`feature_valid_mask`/`window_vendor_sources`).
- **Features:** hand-built L2+trade micro-fixture with a known OFI/CVD/microprice-dev →
  exact expected values. **Value-level no-lookahead (T3):** out-of-order replay ⇒
  byte-identical features (mirrors `tests/test_reconstruct_no_lookahead.py`).
- **Labels:** planted up/down/flat paths → expected triple-barrier `label` and sign of
  `y_fwd_bps` off the Coinbase **mid** anchor (§B); barrier resolves within the horizon
  (`t_barrier ≤ t_event + horizon_ns`). **Span-anchor (P2):** with `coinbase_read_ts <
  t_event`, the emitted `t0 == t_event` (not the lagged read), base price `P0` = the lagged
  mid, and a planted barrier hit in the `[coinbase_read_ts, t_event]` gap produces **no** row
  whose span starts before `t_event`.
- **Leakage-control gate (E0.4):** random k-fold (no purge) shows inflated CV vs
  purged/embargoed on a synthetic overlapping-label series — the controls bite.
- **Manifest round-trip:** producer emits matrix+manifest on a tiny fixture →
  `validate_frame` passes → `run_from_manifest` runs and returns the per-horizon result
  **schema** without crashing. Assert on **structure, not gate outcome** (Codex P3): a
  tiny/weak fixture yields an ordinary G1 **fail**, not `g1_inconclusive` — that flag needs a
  LightGBM rung to pass the solo gate with PBO unavailable (`eval/study.py:70-72`). Assert a
  specific G1 outcome only with a deliberately-planted-signal fixture sized to pass solo.
- **Determinism (P3):** two builds of the same fixture with **different injected
  `generated_at`** ⇒ **identical `model_matrix.parquet` bytes and identical `build_id`**
  (the timestamp is excluded from the hash); the manifests differ only in `generated_at`.

**Tier 2 — real-data (skipif-gated, runs only after backfill unlock):**
`data/processed/*` present ⇒ the E0.3 median-bar histogram, τ ladder, and the first real
G1 run. Skips cleanly today (matches `tests/test_baseline_integration.py`).

---

## Before-backfill vs. after-backfill

Backfill is **GATED and currently LOCKED** (data.md §5a/§9: `download_coinapi.py` refuses
>1-day full pulls (exit 4) until the §5a parity + reseed **multi-day** validation passes;
memory `crypto-lake-access-state` / `coinbase-parity-gate-findings`: as of **2026-07-03**
seam-day parity is validated but the broad full-window map is still the gate → backfill
LOCKED). The split is therefore decisive:

**Buildable & fully testable NOW (pre-backfill) — synthetic + tiny fixtures:**
all producer *code* (T1–T9): clock, trailing-threshold schedule, dual-book snapshot,
features, received-time per-venue feed-lag reads, **guard-aware seam-masking logic (on
synthetic seams)**, triple-barrier labels, per-horizon uniqueness, cost columns, manifest emission, the
end-to-end orchestrator; every Tier-1 test including value-level no-lookahead, sample-timing,
seam masking, threshold causality, the leakage-control gate, and determinism. This
exercises the **entire plumbing** through the built consumer without a byte of vendor data.

**Requires the final backfilled dataset (post-backfill unlock) — T10:**
the **final reviewed stitch plan / seam list** (the product of the §5a parity + reseed
multi-day validation — the masking *code* is pre-backfill, the *seam list it consumes* is
not); **threshold calibration** to hit the **E0.3 median-bar ≤ 2 s gate** (needs real
Binance trade volume); **τ measurement** (E1.1) to set the real horizon ladder; the
**time-per-bar histogram** artifact; the **first real G1 run** over the usable calendar
(704/730 d, OOS≈**April 2026** — data.md §5b) with real regime stratification, real DSR
trial count, and PBO over ≥32 OOS samples. These are gated on backfill unlock and are
**not** part of the code-complete producer.

---

## Task breakdown (future Claude branches)

Each task is one branch → its own TDD plan. Dependency order; all Tier-1-testable
pre-backfill except T10. Suggested branch names in `feat/…`.

| Task | Scope | Builds on (file:line) | Deliverable | Pre/Post backfill |
| --- | --- | --- | --- | --- |
| **T1** `feat/bars-clock` | Dollar-notional clock + hybrid time cap + **trailing/as-of-only per-day threshold schedule + warm-up** (P2b) + `emitted_by_time_cap`; Coinbase-order sort; **CoinAPI Coinbase-trades normalizer is a fill-day prerequisite (does not exist — §C.1)** | `recon/events.py:Trade`; streaming k-way merge (`recon/merge.py:merge_sorted` = fixture oracle only, §C.1) | `bars/clock.py` + threshold-causality test | Pre (calibration Post) |
| **T2** `feat/bars-snapshot` | Dual-book snapshot over events with **`received_time ≤ t_event`** (Coinbase target → `coinbase_read_ts` = last observed, P1); mid + microprice (both emitted) | `recon/reconstruct.py:sample_topk_as_of`, `recon/orderbook.py:60-69` | `bars/snapshot.py` + received-time test | Pre |
| **T3** `feat/bars-features` | Per-bar §6/E1.2 vector (OFI/CVD/microprice_dev/queue_imb/spread_tick/depth/slope/VWAP/intra-bar path); stationarization; **value-level no-lookahead test** | T2, `recon/orderbook.py:snapshot` | `bars/features.py` + no-lookahead test | Pre |
| **T4** `feat/bars-xvenue` | **`t_event` = trigger trade's `received_time`; every input gated by per-event `received_time ≤ t_event` (exact); p99/max tail only for the live watermark, never medians** (P1); basis; perp-state conditioners; **sample-timing test (delayed-event guard)** | T3, data.md §5/§5b | `bars/align.py` + sample-timing test | Pre (Binance tail from E2.5 Post) |
| **T5** `feat/labels-triple-barrier` | Triple-barrier (vol-scaled EWMA barriers, vertical=horizon, **off Coinbase mid** — P2c; microprice arm gated on parity) → `y_fwd_bps`/`label`/`t_barrier` per horizon; **span `[t_event, t_barrier]`, base price `P0` = last-observable `coinbase_read_ts` mid (P2)** | T2 target anchor | `data/labels.py` + span-anchor test | Pre |
| **T6** `feat/labels-uniqueness-cv` | Concurrency uniqueness **per horizon** (port `_concurrency_uniqueness`, group by `horizon` — P2); embargo sizing; **leakage-control gate test** | `data/cv.py:cpcv_splits`, `eval/synthetic.py:_concurrency_uniqueness`, `eval/runner.py:60` | `data/uniqueness.py` + E0.4 gate + per-horizon test | Pre |
| **T7** `feat/bars-cost` | Per-row `cost_bps` (2×taker+slippage, fee-tier param) + `half_spread_bps` from the Coinbase book at `coinbase_read_ts` (lagged, P1) | `eval/cost.py:net_pnl` (consumer) | `bars/cost.py` + tests | Pre |
| **T8** `feat/manifest-writer` | `eval.manifest.build_manifest`/`write_manifest`; explicit `feature_cols`; **`validate_frame` + `validate_matrix` before write** (fail closed, P2) | `eval/manifest.py`, `eval/matrix.py:validate_matrix` | manifest writer + round-trip + bad-row-rejection test | Pre |
| **T9** `feat/producer-orchestrator` | End-to-end per-day → consolidate labeled window; wire `data/usable_calendar.json`; **per-`vendor_source` replay dispatch (Lake→`ts_engine` merge; CoinAPI→`seq`-order book replay + trades normalizer — P2)**; **consume the stitch plan + apply guard-aware seam masks + `window_vendor_sources`** (P2a); `generated_at` injectable + excluded from `build_id` (P3); **`validate_frame` + `validate_matrix` before any `data/processed/` write** (fail closed, P2); integration test through `run_from_manifest`. **Acceptance: no surviving row crosses a seam; NaN/inf row rejected pre-write; byte-identical rebuild** | T1–T8, `eval/runner.py:run_from_manifest`, `recon/stitch_policy.py` | `bars/produce.py` + integration + seam-mask + determinism tests | Pre (synthetic seams) |
| **T10** `feat/producer-calibration` (Post) | Consume the **final reviewed seam list**; threshold calibration to E0.3 gate; τ ladder (`eval/tau.py`); histogram; first real G1. **Acceptance: seam masking holds on the real stitch plan** | T9, backfilled data + reviewed stitch plan | E0.3/E0.5 artifacts + G1 result | **Post** (backfill unlock) |

---

## Decisions & defaults

Forks resolved from the docs. Each is a **manifest parameter**, so an ablation needs no
schema change.

1. **Clock trigger venue** = **Coinbase/target-venue-triggered pre-E2.5** — the close is a
   **local** Coinbase trade whose `received_time` is known today (`t_event =` its receipt),
   with no dependence on an unpinned Binance live-watermark tail; the spec §5.1 **Binance-perp
   notional clock is the post-E2.5 target**, enabled once E2.5 pins the Binance trade-lag tail
   for the live loop. Combined B+C is an ablation knob (E2.2 decides).
   Trigger venue is a manifest parameter, so the switch is config, not a rewrite.
2. **Label anchor** = Coinbase **mid** (primary — the anchor the seam-parity gate
   validates, `recon/parity.py:68,243`), **microprice** as an ablation arm **gated on
   first adding a microprice-parity check** (P2c; §B).
3. **Sample timing / observability (P1):** decision `t_event` on the **received-time** axis;
   every input gated by its own **`received_time ≤ t_event`** (exact, per-event — the target
   venue is **not** zero-lag), with **p99/max tail** constants (never medians) only for the
   live watermark, giving `t_available == t_event` by construction — never `availability_lag_ns`
   (the consumer requires 0 — §C.2/§E).
4. **Vendor seams (P2a):** the producer **consumes the reviewed stitch plan** and applies
   `recon/stitch_policy.py` seam/guard/vendor masks; no training row crosses a seam. Hard
   input + T9/T10 acceptance (§C.3).
5. **Bar-clock threshold (P2b):** **trailing/as-of-only** per-day schedule (prior days
   only) + warm-up, pinned by hash in the manifest — not a single scalar (§A).
6. **Output** = day-partitioned intermediates → one consolidated
   `data/processed/model_matrix.parquet` (+ manifest), matching the built consumer's
   expected paths and AGENTS.md streaming rule.
7. **Determinism (P3)** = matrix bytes + `build_id`, with `generated_at` **injectable and
   excluded from the hash** (§I).
8. **Horizon ladder** = `{2s,10s,60s}` default; τ-rung (~20–30 s) added post-backfill as a
   manifest edit.
9. **CV / cost / DSR / PBO** = **reuse built code** (`data/cv.py`, `eval/`); the producer
   only emits their input columns.
10. **This PR is docs-only** — no `bars/` code yet (the contract is already pinned by
    `eval/matrix.py:RESERVED` + `docs/feature-manifest.md`, so a stub adds no clarity). A
    `bars/schema.py` contract module is **T8's** concern, not this PR's.

## Open questions (for reviewer / to resolve in the owning task)

- **Q1 (T7):** which Coinbase Advanced **fee tier** to pre-register as the default
  `taker_fee_bps` (base ~120 bps vs. an assumed volume tier)? Sets the G1 cost wall.
- **Q2 (T4/E2.5) — resolved for pre-E2.5, open for the switch:** the pre-E2.5 default is
  **Coinbase-triggered**, whose close is a local trade with a known `received_time` (§A/#1).
  The remaining question is the **post-E2.5 switch**: once E2.5 pins the Binance live-watermark
  tail, is the Binance-triggered clock's information-content gain
  (spec §5.1) worth the added lag-modeling risk (too-low ⇒ sample-timing leakage; too-high ⇒
  biases against the Binance-increment premise)? Gate the switch on E2.5 confidence + an E2.2
  downstream-PnL check.
- **Q3 (T5):** vol-scaling estimator for the horizontal barriers — EWMA half-life of the
  micro-window returns (spec says "EWMA"; the half-life is unspecified).
- **Q4 (T1):** `target_bars_per_day` / time-cap `T` / `warmup_days` seed values before real
  calibration (drives the E0.3 gate; only the *seed* is pre-backfill, the calibrated value
  is T10).
- **Q5 (T5/§B, P2c):** add a **microprice-parity check** to the seam gate before promoting
  microprice to the primary label anchor — or accept mid as primary indefinitely?

---

## Validation of THIS PR (docs-only)

- `git diff --check` — whitespace/conflict-marker clean.
- No code touched → `py_compile` N/A (stated per AGENTS.md testing rules).
- Self-review against `docs/experiment-plan.md` (E0.3/E0.4/E0.5, G1) and
  `docs/feature-manifest.md`: (a) explicit `feature_cols`, no all-non-reserved inference;
  (b) `target_cols == {y_fwd_bps, label}`; (c) `availability_lag_ns == 0`;
  (d) `embargo_ns ≥ max_lookback_ns`; (e) reserved-column set matches
  `eval/matrix.py:RESERVED`; (f) every interface reference cites a real file:line verified
  in this repo. **No live vendor calls run.**
- Review round 1 (P1–P3) incorporated and cross-checked: decision-time rule (§C.2/§E),
  seam masks (§C.3/§F/T9), trailing threshold (§A/§H), mid anchor (§B), determinism vs
  `generated_at` (§I) — each traced to verified code (`recon/stitch_policy.py:391`,
  `recon/parity.py:68,243`, `recon/stitch_policy.py:SeamPolicy`).
- Review round 2 (Codex on `52c915b`) incorporated: per-venue feed-lag reads incl. **nonzero
  Coinbase lag** (P1, §C.2), **guard-aware** seam masks (P2, §C.3), **per-horizon**
  uniqueness (P2, §F), **streaming k-way merge** for production with `merge_sorted` as the
  fixture oracle (P2, §C.1) — traced to `recon/merge.py:11-16`,
  `recon/stitch_policy.py:385,391,406`, `eval/runner.py:60`, data.md §5/§5b.
- Review round 3 (Codex on `34b87e9`) incorporated: **book-feed vs trade-feed lag split**
  (P1 — trigger uses the trade lag, snapshot the book lag; §C.2), **Coinbase target snapshot
  + `half_spread_bps` at the lagged `coinbase_read_ts`** (P1, §B/§G), **CoinAPI fill segments
  replay in `seq` order** not a `ts_engine` merge (P2, §C.1), and the tiny-fixture test
  asserts **schema not `g1_inconclusive`** (P3, §J) — traced to data.md §5b/134-140,
  `recon/coinapi.py:13-29`, `eval/study.py:70-72`.
- Review round 4 (Codex on `991991d`) incorporated: **CoinAPI trades routing** for the 52
  Coinbase fill days — a trades normalizer/replay is a new **producer prerequisite** (P2,
  §C.1; does not exist — `recon/coinapi.py` is book-only), and the **triple-barrier span/purge
  is anchored at `[t_event, t_barrier]`** with the lagged read supplying only the base price
  `P0` (P2, §C.2/§D/§B/T5) — traced to data.md §5b/§4.3.
- Review round 5 (Codex on `6a4f931`) incorporated: **offline reads gate on per-event
  `received_time ≤ t_event`** (exact), not `origin − median_lag` — a median leaves ~50 % of
  events as look-ahead; scalar lags are demoted to **p99/max tail bounds** for the live
  watermark only (P1, §C.2/§E/§J/§H) — and **promoting microprice requires a label rebuild**,
  not a manifest flip (the triple-barrier labels are computed off the anchor path; P2, §B) —
  traced to data.md §5/§5b median/p95 pairs, `eval/runner.py` single-target-pair.
- Review round 6 (Codex on `dfcbda7`) incorporated: **CoinAPI timestamps are ns-since-midnight**
  — convert to absolute (`day_open_ns + time_coinapi_ns`) **before** the `received_time ≤
  t_event` gate, else every after-midnight event passes (P1, §C.1) — and the producer runs
  **`validate_matrix` as well as `validate_frame` before writing** (the NaN/inf/finite screens
  live in `validate_matrix`, not `validate_frame`), so bad rows never reach `data/processed/`
  (P2, §H/T8) — traced to data.md §4.3, `eval/matrix.py:validate_matrix`.
- Review round 7 (Codex on `756c817`) incorporated: the **top-level Architecture data-flow**
  and **T9** now show `validate_frame` **+** `validate_matrix` gating the write (the §H/T8 rule
  is now consistent in the summary a T9 implementer reads first; P2).
- Review round 8 (Codex on `3629306`) incorporated: the §E seam note now attributes NaN/inf
  rejection to **`validate_matrix`** (not `validate_frame`, which checks only columns/timing;
  P2), and §D notes the τ-rung **requires rerunning label/matrix production** to emit the new
  bar×horizon rows — the runner rejects a declared-but-missing horizon, so it is not a
  manifest-only edit (P2, T10) — traced to `eval/matrix.py:validate_matrix`, `eval/runner.py:60`.

## Risks & assumptions

- **Assumes** `origin_time` stays 100% populated for `book_delta_v2` (data.md §5,
  measured 2026-06-22) — if a future span is empty, the sampler falls back to
  `received_time` (Tokyo capture keeps it a tight proxy; `recon/ingest.py` already
  supports the fallback).
- **Assumes** the consumer contract (`eval/matrix.py:RESERVED`, `eval/manifest.py` v1) is
  stable; if it changes, T8/T9 must re-sync. Low risk — it is frozen and heavily tested.
- **Risk:** threshold/latency/fee defaults (Q1–Q5) are seeds until real data; the plan
  isolates all as manifest parameters so calibration (T10) never forces a code change.
- **Risk (sample timing, P1):** offline correctness relies on per-event `received_time` being
  present and trustworthy on every feed (Lake `received_time`, CoinAPI `time_coinapi_ns`);
  where a scalar is used (live watermark, manifest bound) it must be a **p99/max tail**, never
  a median (~50 % of events exceed the median — a median read re-opens the ~5–8 % 2 s-label
  look-ahead). The Binance tail is deferred to E2.5 and gates enabling the Binance-triggered
  clock (Q2). All default conservative-high.
- **Risk (CoinAPI order + trades, P2):** Coinbase fill segments must replay in `seq` (file)
  order, not a `ts_engine` merge (the opening snapshot carries a prior-day timestamp); T9 must
  dispatch by `vendor_source` or the target book/labels corrupt. **And a CoinAPI
  Coinbase-trades normalizer does not exist yet** — without it the 52 fill days have no trade
  stream for the clock/CVD, silently shrinking the usable calendar. Both are T1/T9
  prerequisites (§C.1).
- **Risk (seams, P2a):** the producer is only seam-safe if it consumes the **final
  reviewed** stitch plan; a stale/partial seam list would let a cross-vendor window train.
  T9 asserts on synthetic seams; T10 re-asserts on the real plan post-backfill.
- **Risk:** Coinbase trade-order gotcha (data.md §5b) is one-day/one-symbol measured;
  extend to multi-day before relying on it (T1 sorts defensively regardless).
- **Backfill dependency:** T10 (and any *quantitative* E0.3/E0.5 gate result) is blocked
  until the §5a parity + reseed multi-day validation unlocks backfill; T1–T9 are not.

## Follow-ups (deferred, tracked)

- Maker/selective-taker fill economics arm for the cost model (§G; spec §10.2).
- Sequential-bootstrap sample weights beyond concurrency uniqueness (§F; López de Prado).
- Volatility-regime stratification tag beyond `spread_tick` buckets (§H).
- Imbalance/run bars and richer cross-venue context (spec §12.8 extensions).
- Rust `native/` bar/feature path for wall-clock once the Python producer is the oracle.
