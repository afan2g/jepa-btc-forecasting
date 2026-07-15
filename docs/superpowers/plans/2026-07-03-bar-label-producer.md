# Bar / Label / Modeling-Data Producer — Implementation Plan (E0.3 · E0.4 · E0.5)

> **Altitude.** This is a **producer architecture spec + task breakdown**, not a
> per-step TDD plan. Each implementation Task (T1–T9) below is scoped to become its own
> `docs/superpowers/plans/…` TDD plan (or a small cluster) on a future Claude
> branch — the same "each phase gets its own detailed TDD implementation plan"
> convention the experiment plan sets
> ([`docs/experiment-plan.md:3`](../../experiment-plan.md)). The historical T10
> label now denotes the one operational #69 transaction, not another code branch.
>
> **Companions.** [`jepa_btc_forecasting_spec.md`](../../../jepa_btc_forecasting_spec.md)
> §5–§8/§10 (design), [`docs/experiment-plan.md`](../../experiment-plan.md)
> E0.3/E0.4/E0.5 + Phase 1/G1 (gates), [`docs/feature-manifest.md`](../../feature-manifest.md)
> (the output contract), [`docs/data.md`](../../data.md) §5/§5a/§5b (coverage +
> backfill gate), and
> [`2026-07-10-staged-signal-acquisition.md`](2026-07-10-staged-signal-acquisition.md)
> (single-venue gate and conditional expansion), and
> [`../specs/2026-07-13-g0bn-protocol.md`](../specs/2026-07-13-g0bn-protocol.md)
> (binding G0-BN config, freeze, access, metric, and verdict contract). Section
> refs like "§5.4" point to the spec.
>
> **2026-07-11 binding amendment (#66/#67).** The first executable producer mode is
> `binance_single_venue`: Binance BTC-USDT perpetual L2 + trades supply the bar
> clock, features, labels, and costs for G0-BN. Coinbase, Binance spot,
> derivatives-state, and cross-venue modes remain in this architecture but are
> deferred until G0-BN passes. Executable sections below state their mode binding
> explicitly; older Coinbase-specific entries in the review-history changelog are
> historical evidence, not implementation instructions for G0-BN.
>
> **2026-07-13 issue-#83 contract.** G0-BN uses distinct `g0bn-*` config,
> ledger, freeze, holdout-plan/universe, transaction, raw-access claim,
> matrix-access claim, consumption, materialization-attestation, verdict, and
> report identities. Existing G0-CB/G0-XV code and April consumption semantics
> remain a legacy regression contract; they are not parameterized or relabeled
> for Binance/January.

**Goal.** Build the offline **producer** that first turns reconstructed Binance
BTC-USDT-perpetual event streams, and later optional Coinbase or cross-venue
streams, into the exact
`data/processed/model_matrix.parquet` + `data/processed/feature_manifest.json`
that the already-built generic/legacy consumer `eval.runner.run_from_manifest`
([`eval/runner.py`](../../../eval/runner.py)) loads, validates, and gates for
rebuildable G1 studies. G0-BN holdout deliberately bypasses that generic entry
point and uses #67's distinct two-burn scorer after blind materialization.
Pipeline: **notional bars (E0.3) → per-bar features (§6 / E1.2) → triple-barrier
labels + uniqueness (E0.4) → cost columns (E0.5) → v1 manifest**. CPCV and
lower-level cost/DSR/PBO primitives are already implemented and tested
(`data/cv.py`, `eval/`); G0-BN still requires #67's separate identities,
candidate ledger, moving-block uncertainty, selection, access control, and
verdict/report path. This producer feeds both consumers.

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

The generic/formal-G1 **consumption side is done** (PRs #14–#26), while the
distinct G0-BN consumer belongs to #67. Producer T1, T2, T3, T5, T6, and T7 are
built; T8, T9, and G0-BN mode/transaction integration remain. This plan is the
implementation ledger and contract for that remaining work.

| Layer | Status | Where |
| --- | --- | --- |
| Event-time reconstruction (E0.1): merge trades+deltas, book-at-T, apply-before-read | ✅ built | `recon/reconstruct.py:sample_topk_as_of`, `recon/orderbook.py:OrderBook.snapshot(k)`, `recon.live.LiveReconstructor` (streaming); `recon/merge.py:merge_sorted` = **bounded-fixture oracle only, forbidden for full days** |
| Reconstruction trade record (engine time, side, price, amount) | ✅ built | `recon/events.py:14` `Trade(ts_engine, seq, side, price, amount)` |
| Clock trade record (origin + received time, source identity, side, price, amount) | ✅ built | `bars/events.py:ClockTrade` |
| Mid / microprice from a book snapshot | ✅ built | `recon/orderbook.py:60-69` |
| CPCV + per-interval purge + embargo | ✅ built | `data/cv.py:cpcv_splits(t_event, t0, t1, *, n_groups, k, embargo_ns)` |
| No-trade-band net PnL, uniqueness-weighted Sharpe | ✅ built | `eval/cost.py:net_pnl`, `eval/cost.py:weighted_sharpe` |
| Legacy/formal-G1 DSR + PBO(CSCV) + gate + per-regime + gross/net | ✅ built | `eval/study.py:run_study`, `eval/stats.py:deflated_sharpe`,`pbo` |
| Distinct G0-BN config/ledger/freeze, two-burn transaction, moving-block uncertainty, selection, report/verdict | ❌ **#67, outside producer** | binding G0-BN spec |
| v1 feature-manifest schema + frame validation + leak screen | ✅ built | `eval/manifest.py` (`validate_manifest`,`validate_frame`,`load_manifest`,`feature_list`,`target_list`,`leaky_feature_names`) |
| ModelMatrix reserved-column contract | ✅ built | `eval/matrix.py:RESERVED` + `validate_matrix` |
| Manifest-driven runner + CLI | ✅ built | `eval/runner.py:run_from_manifest`, `scripts/run_baseline.py` |
| τ decay-window helper (E1.1) | ✅ built (pure) | `eval/tau.py:predictivity_curve`,`estimate_tau` |
| Usable all-feed calendar (704/730 d, OOS≈Apr 2026) | ✅ built | `data/usable_calendar.json`, `ingest/verify_trades_and_calendar.py` |
| Vendor-seam fill policy + label/feature seam masks (E0.4 partial-fill contract) | ✅ built (helpers) | `recon/stitch_policy.py` (`SeamPolicy`, `DayStitchPlan`, `label_valid_mask`, `feature_valid_mask`, `window_crosses_seam`, `window_vendor_sources`) |
| **Notional-bar sampler (dollar clock + time cap)** | ✅ built (T1; G0-BN routing/calibration remains) | `bars/clock.py`, `bars/events.py` |
| **Dual target-book observable/label reads + staleness** | ✅ built (T2) | `bars/snapshot.py` |
| **Wire certified coverage masks into the producer; add Coinbase stitch masks only in deferred modes** | ❌ **build (E0.4)** | new `bars/produce.py` (§C/§F/T9) |
| **Per-bar features (OFI/CVD/depth/slope/microprice-dev/intra-bar path)** | ✅ built (T3) | `bars/features.py` |
| **Cross-venue alignment + Binance→Coinbase feature lag** | ❌ **build (§5.3/§13)** | new `bars/` |
| **Triple-barrier labels + forward returns** | ✅ built (T5) | `data/labels.py` |
| **Same-horizon uniqueness + embargo sizing/leakage-control gate** | ✅ built (T6) | `data/uniqueness.py` |
| **Per-row cost columns (`cost_bps`,`half_spread_bps`)** | ✅ built (T7) | `bars/cost.py` |
| **ModelMatrix assembly + manifest emission** | ❌ **build (E0.3/E0.4)** | new `bars/produce.py`, `eval.manifest` writer |

`recon/parity.py:_signed_labels` remains only a parity-comparison directional
label. Training labels now live in `data/labels.py`; do not substitute the parity
helper. The remaining producer gaps are manifest writing and source-isolated
orchestration rather than the bar, snapshot, feature, label, uniqueness, or
cost primitives.

**Explicitly NOT in scope of the producer plan (Non-goals):**

- The legacy/formal-G1 consumer signatures, CPCV (`data/cv.py`), and low-level
  cost/DSR/PBO primitives — built; the producer conforms to them and does not
  parameterize the legacy evaluator. #67 separately implements the G0-BN
  consumer (21+ legacy test call sites remain regression coverage; see
  [`docs/superpowers/plans/2026-07-02-lightgbm-manifest-integration.md`](2026-07-02-lightgbm-manifest-integration.md)).
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
Trade/Delta      │    │ snapshot.py  observable + true target │──▶│  triple-barrier    │──▶│  frame  (contract)   │
OrderBook.snap ──┼───▶│ features.py  OFI/CVD/microprice_dev…   │   │ uniqueness.py     │   │ runner.run_from_     │
sample_topk_as_of│    │ align.py     optional cross-venue lag  │   │ cv.py (BUILT)     │──▶│  manifest → gates    │
                 │    │ cost.py      cost_bps/half_spread_bps  │   └───────────────────┘   │ study/stats/cost     │
                 │    │ produce.py   assemble → parquet+manifest│                          └──────────────────────┘
                 │    └──────────────────────────────────────┘
```

**Module layout.** New top-level `bars/` package (matches spec §3 "`bars/`" and
sits above the built `recon/`). Labels/uniqueness extend the existing `data/`
package alongside the built `data/cv.py`. Manifest *writing* is a thin helper in
`eval/manifest.py` (the module that already owns the schema), so the producer and
consumer share one contract definition.

**Packaging (Codex #P3; built by T1):** `pyproject.toml`
`[tool.setuptools.packages.find] include` contains `bars*`, and
`tests/test_packaging.py::test_baseline_packages_are_shipped` pins it. Keep that
regression green so a non-editable install / CLI can import the producer.

**Data flow (one instrument-day, then consolidate):** in the first
`binance_single_venue` mode, use one certified Binance perpetual L2/trade stream:
the Binance trade stream drives the notional clock; one received-time-gated book
read supplies observable features and costs; one origin-time read supplies the
offline label anchor. The same causal, span-safe label and validation contracts
below apply with Binance as the target venue. In deferred cross-venue mode,
a **streaming, day-partitioned k-way merge** of Binance+Coinbase deltas+trades on the
engine-time axis (§C.1; `recon.merge_sorted` is the bounded-fixture oracle **only**, never
the full-day path) → `bars.clock` (trailing-threshold schedule) emits bar boundaries →
`bars.align` sets the received-time **decision `t_event`** (§C.2) →
`bars.snapshot` reconstructs **two target-book reads** (§C.2; Coinbase is the
target in this deferred mode): the **observable book** at `target_read_ts` (last origin among events with
`received_time ≤ t_event` — features +
`half_spread_bps`) and the **true label book** at `t_event` (plain origin cut, offline ground
truth — `P0`); feature reads pre-filter `received_time ≤ t_event` then fold in origin order →
`bars.features` builds the per-bar stationarized vector (trade-flow features over the bar's
**origin-order members**, §C.2) → `data.labels` + `data.uniqueness` attach
`y_fwd_bps`/`label`/`t_barrier`/`uniqueness` (**per horizon**, **`P0` = true Coinbase mid at
`t_event`**) → `bars.cost` attaches `cost_bps` (incl. `target_read_ts→t_event` latency
slippage)/`half_spread_bps` → **guard-aware `stitch_policy` masks + `window_vendor_sources` drop
cross-seam/guard/uncovered rows** (§C.3) → per-day parquet → consolidate the labeled window
→ **`validate_frame` + `validate_matrix`** (fail closed — the NaN/inf/finite screens live in
`validate_matrix`, §H/T8) → write `model_matrix.parquet` + `feature_manifest.json`.

That last pre-write validation order applies to ordinary development and later
rebuildable modes. G0-BN holdout is the deliberate one-shot exception: after
the raw-access burn, T9 streams the frozen recipe once, closes/fsyncs the
derived matrix/manifest, and attests their actual hashes/counts without
reopening them. Only after the separate matrix-access burn may the sole scorer
first reopen and run `validate_frame` + `validate_matrix`; any failure is
terminal INCONCLUSIVE, never an invitation to rebuild.

### Staged dataset modes (binding before T1)

This is one producer architecture with three source modes, introduced in gate
order rather than maintained as separate forks:

- **`binance_single_venue` (first; G0-BN):** no Coinbase input is required or
  opened. Binance BTC-USDT perpetual supplies the clock, observable feature
  book, true label book, trades, labels, and costs. The manifest has one Binance
  venue declaration and an explicit ordered feature list containing only
  own-venue L2/trade features. Missing Coinbase, spot, derivatives-state, or
  multi-asset columns are not created or zero-filled.
- **`coinbase_only` (deferred):** Coinbase supplies the same target-venue
  contracts. This mode is retained for transfer testing after G0-BN passes.
- **`cross_venue` (deferred):** requires certified overlap from both venues and
  uses the streaming merge above. It emits matched Coinbase-only,
  Binance-only, and combined views over an identical row universe. A row
  lacking certified coverage is excluded from every arm, never represented by
  sentinel values.

**G0-BN evaluator boundary.** Acquire only `2025-11-01` through `2026-01-31`
inclusive. Producer calibration, feature selection, CPCV/PBO, and all model or
threshold choices use the immutable development partition
`[2025-11-01, 2026-01-01)`. The untouched fixed holdout is
`[2026-01-01, 2026-02-01)`. #68 uses a custodian identity and effective
permissions distinct from the developer/experiment operator to own and seal
the exact January raw and certified normalized L2/trade objects plus
activity-obscuring, outcome-blind inventory metadata. Variable-length byte
sizes and record counts stay inside custody until after the raw-access burn.
Developer-owned files plus `chmod` are not a custody boundary. No January
source payload/footer, bar, matrix, manifest,
label, cost, feature, forecast, or metric is read by the selection/evaluation
plane before an outcome-blind `g0bn-holdout-plan-v1` and complete
`g0bn-freeze-v1` exist.

The stable `g0bn-holdout-universe-v1` ID depends only on `g0bn-v1`, the exact
instrument, and fixed January/February bounds; pilot/config/freeze/source/plan/
result changes cannot mint another transaction. The freeze pins
`holdout_plan_sha256`, which enters the future OOS build recipe, and contains no
January build/manifest/matrix/logical-row hash, row/drop count, realized
adaptive schedule/state, or result.

#69 first acquires and holds the stable transaction's nonblocking process-owner
lock across every outcome-capable child. A concurrent live start exits
`transaction_already_running` without reading claims/data or mutating the
journal. After data-free refit/preflight, #69 atomically creates/fsyncs the
distinct `g0bn-raw-access-claim-v1` **before** the first January raw/normalized
object/payload/footer read. T9 then performs the sole blind materialization,
closes the artifacts, and writes `g0bn-materialization-attestation-v1` with the
actual hashes/counts without reopening them. Only after that completes does #69
atomically create/fsync `g0bn-matrix-access-claim-v1`, before the sole scorer
first opens the derived matrix/parquet/footer to validate and score. Both burns
belong to the same `g0bn-one-shot-v1` transaction and `g0bn-consumption-v1`
journal. A pre-burn owner death is retryable; only a later lock owner may
classify a post-burn nonterminal state as crash-left INCONCLUSIVE. Any crash or
materialization, transition, validation, fit, score, or write failure after
either burn is terminal INCONCLUSIVE with no resumable intermediate path. The
existing `2026-04-01` Binance smoke day is an integrity fixture only and must
not enter calibration, selection, or outcome reporting.

Issue #67 implements the separately typed config/identity, candidate ledger,
freeze/plan, generic-runner guard, stable-universe/two-burn one-shot runner,
materialization attestation, metrics/report, and
synthetic regression slices in the binding G0-BN spec. It may reuse pure #52
helpers but must not add G0-BN to the legacy G0-XV protocols, ledger, freeze,
record, or CLI. Issue #69 alone owns final operator values and the
decision-bearing transaction.

**Deferred cross-venue evaluator boundary.** If G0-BN passes, the prior
Coinbase/cross-venue protocol remains valid: regenerate every arm on a matched
row universe, perform development-only selection in one candidate ledger, and
score a separately frozen holdout once. Comparing a broad single-venue build
to a narrower combined arm is forbidden.

**Span-safe partition boundary (deep-review P2):** assigning a row by `t_event`
date is not enough because its forward label can read the next partition. For
each horizon, T9/#69 must prefilter before label generation or adjacent-day
loading. For G0-BN, a development candidate survives only when
`t_event + horizon_ns + guard_ns < 2026-01-01T00:00:00Z`; a holdout candidate
survives only when the same upper bound is before
`2026-02-01T00:00:00Z`. After construction, fail closed unless actual feature,
cost, and guarded label support remain in the assigned partition. Persist
bounds, horizons, guard, and rule version in the hash-pinned G0-BN partition
plan. Development drop counts are realized before freeze; the holdout plan pins
only the January count schema and sufficiency rules. Actual January counts are
first produced by blind materialization after the raw-access burn, enter its
attestation and the terminal report, and are never copied into the freeze. This is
deliberately distinct from the issue-#52 partition artifact, whose required
pre-filled holdout counts would need forbidden January materialization.

---

## §A. Bar clock construction (E0.3)

**Clock.** Emit a bar when **cumulative traded notional** crosses a threshold **or**
`T` seconds elapse, whichever first (spec §5.1–5.2 hybrid time cap). Clock off the
**trade stream**, never the book-update stream (§5.1: trades carry realized
aggression; quote churn is the noise JEPA discards). **Dollar, not volume** (§5.1:
BTC ranges 2×+; dollar bars are homoscedastic).

- **Reference stream — G0-BN default = Binance BTC-USDT-perpetual notional**
  (`price × amount` over Binance trades). This is the target venue's own
  aggression, so G0-BN does not need a cross-venue lag model to define bar
  closes. The received-time decision watermark and origin-time membership rules
  in §C.2 still apply. The offline G0-BN gate uses each event's certified
  `received_time` directly and **does not wait for E2.5 or live lag-tail
  calibration**. #64 must reject a source that cannot satisfy the normalized
  causal timestamp contract.
- **Deferred Coinbase/cross-venue clock:** Coinbase/target-venue-triggered
  notional (`price × amount` over Coinbase `recon.events.Trade`) remains the
  default when Coinbase is the target. Its monotone watermark is
  `max(t_event(N−1), max(received_time) over members, cap_fire)` (§C.2/#13),
  not the receipt time of only the trigger trade. A later cross-venue experiment
  may use Binance-perpetual or combined Binance+Coinbase notional, but live
  deployment of that remote-trigger clock requires E2.5 to pin the Binance
  trade-lag p99/tail. That deferred live requirement does not constrain G0-BN.
  The trigger venue remains a manifest parameter, so the later ablation is a
  configuration change rather than an architecture fork.
- **Adaptive threshold** (E0.3) — **trailing / as-of only.** `threshold_d =
  trailing_avg_dollar_volume(days < d, lookback_days) /
  target_bars_per_day`, computed from **prior days only** (strictly `< d`). The
  final config must supply one exact `lookback_days` and target; a design range
  or library default cannot enter the freeze. Calibration targets the E0.3
  active-regime gate.
  Using day `d`'s own completed volume would leak future volume into `d`'s bar boundaries
  — a subtle sampling look-ahead. **Warm-up:** the first `warmup_days` (no full trailing
  window) use a fixed seed threshold and are flagged/excluded from the labeled matrix.
  The build records the **full per-day threshold schedule *and* its content hash** in the
  manifest — `bar_clock.threshold_schedule` (the per-day values, or a named artifact
  path/`sources` entry) **plus** `threshold_schedule_hash`, **not a single scalar and not the
  hash alone** (Codex #A — a hash cannot recover the per-day thresholds for a rebuild/audit after
  a completed-volume, coverage-normalization, or calendar change) — so the sampling is
  reproducible and auditably causal. **Coverage normalization (Codex #12):** the trailing
  average sums each prior day's **raw** completed notional, so a low-coverage (~93%/gappy) or
  CoinAPI-filled day skews `threshold_d` for every later day whose 7–30 d window includes it
  (the day's *trades* feed the aggregate at clock-construction time even if that day's *rows*
  are later seam-masked). Normalize each day's volume by its **covered fraction** (or exclude
  sub-coverage days from the trailing average) — distinct from the future-volume look-ahead
  already handled above.
  **G0-BN freeze boundary (#83):** the development build records its realized
  schedule. The outcome-blind holdout plan freezes the causal update algorithm,
  parameters, and exact development-end initial state, not unrealizable January
  values. After the raw-access burn, T9 derives the January schedule
  sequentially from prior observed days and records the full realized
  schedule/hash in the OOS manifest, materialization attestation, and report.
  January may execute the frozen rule but may not select or reset it.
- **Time cap** `T`: bounds worst-case heteroscedasticity (§5.2 — a dead Sunday
  must not become one 45-min bar). Emit a **`emitted_by_time_cap`** boolean per bar
  (diagnostic `extra_cols`, opted-in via the manifest — never a feature). Its
  exact value is an unresolved required operator field until development
  calibration; there is no G0-BN numeric default.
- **Ordering gotcha (data.md §5b, measured 2025-06-01):** Binance trade feeds are stored
  in `origin_time` order with monotonic `trade_id`; **Coinbase trades are NOT sorted by
  `origin_time` and `trade_id` is not monotonic** — the clock **must sort Coinbase trades
  by `origin_time`** before accumulating. Total order is defined by `recon.events.order_key`
  (honored by both the streaming production merge and the `merge_sorted` fixture oracle);
  the sampler must not assume file order.

**Gate (E0.3):** median active-regime bar ≤ 2 s (so the 2 s horizon ≈ a few bars) +
the **log-scale time-per-bar histogram** artifact. G0-BN threshold calibration and this gate
require #68's certified Binance trade volume; deferred Coinbase modes require their
approved target data. The clock
*code* and its determinism are testable now on synthetic/fixture trades.

**Builds on:** `recon/events.py:Trade` + `order_key`; the streaming k-way merge
(`recon/merge.py:merge_sorted` = fixture oracle only, §C.1).

---

## §B. Target-venue mid / microprice inputs

The label is defined on the selected **target venue's book**, off mid or
microprice — never last-trade (spec §5/§6, E0.4). G0-BN binds the target to
`BINANCE_FUTURES/BTC-USDT-PERP`; deferred transfer/cross-venue modes bind it to
Coinbase BTC-USD. Both values come from the existing snapshot:

- `recon/orderbook.py:60-62` → **mid** = (best_bid + best_ask)/2.
- `recon/orderbook.py:64-69` → **microprice** = (ask_size·best_bid + bid_size·best_ask)/
  (bid_size + ask_size) — size-weighted fair value, robust to bid-ask bounce.

**Default:** label off the selected target venue's **mid** and carry microprice as
a feature/candidate ablation. #64 must certify Binance price/size semantics for
G0-BN. In the deferred stitched-Coinbase mode, the existing seam parity validates
mid but not microprice (`recon/parity.py:68,243`), so a Coinbase microprice label
requires a new parity check. In every mode, promoting microprice is a **label
rebuild**, not a manifest-only switch: `label`, `y_fwd_bps`, and `t_barrier` must
be recomputed from the microprice path. The manifest records the anchor.

**Two book reads, three roles (§C.2, Codex #1).** The producer reconstructs the
selected target book at two cutoffs per bar and keeps label, feature, and cost
roles apart:

- **Label base price `P0`** = the **true reconstructed mid** (or microprice) at **`t_event`** —
  a plain **origin-time** cut (`sample_topk_as_of` at the `t_event` origin cutoff). The label is
  offline **ground truth**, *not* observability-gated: reading the realized book at the decision
  time is correct, and the barrier path already runs forward over `[t_event, t_barrier]`. **`P0`
  is never read at `target_read_ts`** — that would fold the already-realized, *past-and-feature-
  observable* `[target_read_ts, t_event]` drift into `y_fwd_bps` (a common-mode target leak;
  see Changelog / #1).
- **Feature / cost book** = the **observable** book at **`target_read_ts`** (last origin among
  events with `received_time ≤ t_event`) — feeds book-shape features and `half_spread_bps` (§G).
  The entry-latency drift `target_read_ts → t_event` is charged **forward as `cost_bps`
  slippage** (§G), not backward into the label.
- **Staleness cap (Codex #8):** because the clock triggers on **trades** while the book is a
  separate channel, bars can keep closing through a target **book-feed dropout**.
  Drop any row whose observable book is older than a source-certified cap
  (`t_event − target_read_ts > staleness_cap_ns`) — timestamp presence is not
  gap absence. Deferred Coinbase modes additionally apply the known stitch/seam
  policy; the generic staleness cap still catches intra-vendor gaps.

Both mid and microprice are emitted as base-price series; the forward label (§D) uses the
`t_event` **mid** as `P0`. A microprice target requires the source-specific evidence and
label rebuild above. The triple barrier and emitted span remain `[t_event, t_barrier]`.

---

## §C. Source alignment (event time, decision time, coverage)

G0-BN has one target venue with asynchronous book and trade channels. Deferred
cross-venue modes add a second venue, cross-venue latency, and Coinbase vendor
seams. Every mode reuses E0.1 and the same decision-time discipline; optional
sources may not leak into or become prerequisites of `binance_single_venue`.

1. **Single engine-time axis — streaming in production (Codex P2).** Reconstruct each
   selected venue's book by merging its trades + L2 deltas on `ts_engine`; #64/#68
   must certify the G0-BN source's origin/receipt timestamp contract. **Production
   uses a streaming, day-partitioned k-way
   merge** over already-sorted/chunked inputs: `recon/merge.py:merge_sorted` **materializes
   both streams into one list and its docstring explicitly forbids full-day use** (Binance
   perp `book_delta_v2` ≈ 109 M rows/day, ~4 GB — AGENTS.md streaming rule), so it stays the
   bounded-fixture oracle the replay-equivalence test pins. T2/T9 reuse the streaming
   watermark merge `recon.live.LiveReconstructor` already implements, or the deferred Rust
   k-way merge (spec §3). Snapshot = book inclusive of all events with `ts_engine ≤` the read
   time, **strict `<` apply-before-read at the trade boundary**
   (`recon/reconstruct.py:sample_topk_as_of`, `order_key` deltas-before-trades).
   **`sample_topk_as_of` cuts on the ORIGIN axis (Codex #2)** (`time_of = ts_engine`,
   `recon/reconstruct.py:67,138`) — it folds every event with `origin ≤ cutoff`, which is **not**
   the received-gated set when origin ≠ received order. So **feature/cost** reconstruction must
   **pre-filter events to `received_time ≤ t_event`, then fold in origin order** (a delayed
   `origin ≤ target_read_ts` but `received > t_event` straggler must be excluded); the **label**
   read uses the plain origin cut at `t_event` (offline ground truth — §C.2/§B). Both then reuse
   the same top-K folding.
   **Deferred Coinbase/CoinAPI fill segments are the exception (Codex P2), for
   both book and trades; they are not opened by G0-BN:**
   - **Book:** Coinbase gap days filled from CoinAPI `limitbook_full` must replay in strict
     **`seq` (file) order via `recon.coinapi`**, *not* a `ts_engine` merge — the opening
     SNAPSHOT block carries a **prior-day** `time_exchange` (data.md:134-140;
     `recon/coinapi.py:13-29`) that a timestamp merge would sort to day-end, corrupting the
     target book.
   - **Trades (deferred Coinbase-mode prerequisite):** that mode's clock triggers on
     **Coinbase trades**, but the **52 Coinbase fill days** (data.md §5b — 47 need book, all 52 need
     trades, 2.6 GB, verified present in CoinAPI flat files) have **no Lake trades**, and a
     **CoinAPI trades normalizer/replay does not exist yet** (`recon/coinapi.py` is book-only;
     `download_coinapi.py` emits `limitbook_full` only). It is a prerequisite
     for the deferred Coinbase T1/T9 path, **not for #67/G0-BN** —
     without it the producer cannot emit bar closes or CVD on fill days and would silently
     build the deferred 704/730 matrix from Lake-only trades, dropping/corrupting exactly
     the backfilled calendar that mode depends on. **CoinAPI timestamps are ns-since-midnight,
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
   is only *observable* at the trading box at its own **`received_time`**. The
   selected source must normalize and certify both timestamps (Lake exposes
   `origin_time` + `received_time`; deferred CoinAPI uses `time_exchange_ns` +
   `time_coinapi_ns`). **Prerequisite (Codex P1):** the normalized
   `recon.events.Trade`/`Delta` keep only one `ts_engine` (`trades_from_df`/`deltas_from_df` drop
   the other timestamp), so T1/T2 must build a **received-time-bearing event record/table** that
   carries *both* origin (for ordering) and received (for gating). Then the read rule is **exact
   and per-event, not a lag constant:**
   - **Decision time `t_event`** is on the box's received-time axis; every input is included
     **iff its `received_time ≤ t_event`** (ordered by `origin_time`, *gated* by
     `received_time`) — no median/constant approximation, so no delayed event leaks in.
   - **Trigger / bar close — a MONOTONE watermark (Codex P1/#13):** the bar accumulates notional
     in **`origin_time`** order and closes on the threshold-crossing trade, but since origin ≠
     received order an earlier-origin trade can arrive *later*. `t_event` is therefore a
     **cumulative, non-decreasing** watermark:
     **`t_event(N) = max(t_event(N−1), max(received_time) over bar N's members, cap_fire(N))`**.
     `max(received_time)` over members *alone* is **not** monotone — a delayed trade in bar N can
     arrive after bar N+1's members, letting `t_event(N+1) < t_event(N)`, i.e. a later bar decided
     on membership the box could not have known until N's boundary resolved. Clamping to
     `t_event(N−1)` orders the decision times — **required** by the CPCV time-groups, PBO blocking,
     uniqueness, and the stable `t_event` sort (§F/§I). Live: a bounded watermark waits for
     stragglers, then boundaries resolve in arrival order; offline this is exact.
   - **Time-cap closes (`emitted_by_time_cap`):** for a cap-closed bar the `cap_fire(N) = t_cap`
     (§A) term dominates when the bar holds few or **zero** trades — `t_event = max(t_event(N−1),
     t_cap, max(received_time) over events)` is **never earlier than the cap fire time** (nor the
     prior bar's `t_event`), or `t_available`/labels would start before the bar was decidable.
   - **One decision per `(t_event, horizon)` — coalesce backlog ties (Codex deep-review #2):**
     clamping to `t_event(N−1)` can give several bars the **same** `t_event` when a delayed backlog
     drains at one instant. The evaluator scores rows **independently** (per-row PnL / trade counts,
     `eval/baseline.py:88-105`), so duplicate-`t_event` rows would be counted as **multiple** trade
     opportunities at one decision instant. Emit **at most one row per `(t_event, horizon)`** —
     coalesce a backlog into the **last-closing bar** (the most-informed features/label at that
     instant); T1/T9 dedupe on `(t_event, horizon)` before write, and §E lists it as an invariant.
   - **Feature/cost book snapshot (observable):** include target-book events with `received_time ≤
     t_event` (pre-filter, then fold in origin order — §C.1/#2); the **observable** target read
     **`target_read_ts`** is the origin time of the last such book event (`≤ t_event`), feeding
     book-shape **features** and `half_spread_bps` (§G). A staleness cap drops rows whose
     observable book is too old (§B/#8).
   - **Trade-flow features over the bar's origin-order MEMBERS (Codex #3):** `cvd`,
     `aggressor_imb`, `largest_print`, `signed_vol`, `vwap_minus_mid`, `ofi_integrated` are
     computed over exactly the trades that constitute the bar (accumulated in **origin order up to
     the crossing trade**), **not** the received-gated superset `{received ≤ t_event}`. The
     superset also contains early-arriving **next-bar** trades (origin after the crossing trade);
     folding them here would make `cvd` non-additive across bars and double-count prints, with the
     value depending on receive-time jitter. Members are automatically observable because
     **`t_event ≥` every member's `received_time`** (the monotone watermark is `≥
     max(member received_time)`, §above), so no separate gate is needed on them — only the
     point-in-time **book snapshot** uses the received-gate.
   - **Label base price `P0` — the TRUE book at `t_event` (Codex #1):** `P0` = mid (or
     microprice) of the **true reconstructed target book at `t_event`** — a plain **origin** cut,
     the offline **ground truth**, *not* observability-gated. The triple barrier + emitted span run
     over `[t_event, t_barrier]` (`t0 = t_event`), aligned with the CPCV purge/embargo (§F) and
     coverage/seam masks (§C.3). `P0` is **never** read at `target_read_ts`: doing so folds the
     already-realized, past-and-feature-observable `[target_read_ts, t_event]` drift into
     `y_fwd_bps` — a common-mode target leak the E0.4 control cannot catch (Changelog). Entry
     latency is charged **forward** as `cost_bps` slippage (§G).
   - `t_available == t_event` is **correct by construction** — every input has
     `received_time ≤ t_event`; `availability_lag_ns` stays 0 (§E).
   - **Lag constants are TAIL bounds, never medians (Codex P1):** where a scalar is unavoidable —
     the **live** loop's straggler watermark and the manifest's *declared* bound — use a **pinned
     p99/max** feed lag, not the median. data.md reports median/p95 pairs (e.g. Coinbase trades
     164/238 ms, Binance perp book 4.4/149 ms); the **median is not conservative** — ~50 % of
     events exceed it, so reading at `t_event − median` would leak delayed events. Offline builds,
     including G0-BN, use actual `received_time` and enable the target-venue clock
     without E2.5. E2.5 measures remote-feed tails only for deferred live
     cross-venue watermarking (§A, Q2).

3. **Coverage and vendor-seam exclusion (E0.4, hard input).** Every mode
   consumes a source-specific certified coverage artifact and drops unsupported
   feature/cost/label windows. G0-BN consumes #68's single-source Binance
   coverage and provenance; it does **not** require or open a Coinbase/CoinAPI
   stitch plan. In the deferred Coinbase mode, Crypto Lake + CoinAPI seams (the
   33-day hole and smaller gaps, data.md §5a/§5b) require the final reviewed
   `recon/stitch_policy.py:DayStitchPlan` (`.seams`, fill `segments`, `SeamPolicy`
   with `seam_guard_s = 60 s` = the longest label horizon) and **drop any bar row whose
   feature window `[t_feature_start, t_event]` or label window `[t_event, t_barrier]`
   crosses a seam or touches its guard band, or whose windows are not backed by a single
   vendor** (`{LAKE}` or `{COINAPI}`). Use the built helpers directly. For the **seam +
   guard-band** test, run **`window_crosses_seam` over each window's *actual* per-row span
   extended by `guard_ns` on both sides** — feature `window_crosses_seam(t_feature_start −
   guard_ns, t_event + guard_ns, seams)`, label `window_crosses_seam(t_event − guard_ns,
   t_barrier + guard_ns, seams)` — which is guard-aware (the ±`guard_ns` extension covers the
   guard band that `window_crosses_seam` alone lacks, `recon/stitch_policy.py:385`) **and** uses
   the true span. **Do not use `label_valid_mask(…, horizon_ns=…)` for the label window (Codex
   #A):** it masks the *full* `[t_event, t_event + horizon]` (`recon/stitch_policy.py:391-403`),
   so an **early-resolving barrier** (`t_barrier < t_event + horizon` — a TP/SL hit) is dropped
   for a seam that falls *after* its actual `t_barrier`, needlessly zeroing/undersizing the 60 s
   rung (#5) and contradicting the `[t_event, t_barrier]` actual-span rule above.
   (`feature_valid_mask`/`label_valid_mask` take a **scalar** `horizon_ns`/`lookback_ns` and check
   `[t, t±window]` — they **cannot** accept a per-row `t_barrier` absolute span, Codex #A; the
   per-row actual-span path must stay on `window_crosses_seam`/`window_vendor_sources` as above, or
   a **new vectorized start/end guard helper** — do not pass `t_barrier` as a duration.) For the
   **vendor-coverage** test use
   **`window_vendor_sources(start, end, segments)` (`recon/stitch_policy.py:430`)** — the row
   is kept only when *both* its feature window
   `[t_feature_start, t_event]` and label window **`[t_event, t_barrier]`** return a singleton
   `{lake}` or `{coinapi}` (with `P0` at `t_event` under the spine, the label window needs **no**
   back-extension — the observable feature/cost read at `target_read_ts` already sits inside the
   feature window, so `feature_valid_mask` covers it; Changelog / #6, #11); any mixed-vendor,
   `excluded`, or `UNCOVERED` (day-edge overhang)
   window is masked. **Do not use `vendor_source_at(...)` for this** — it is per-*sample*
   and only sees the endpoint, so it would miss an excluded/uncovered span *inside* the
   window (`label_valid_mask`/`feature_valid_mask` cover the seam/guard geometry on a
   regular grid but, per their own docstrings, must be intersected with the whole-window
   vendor set). Never train across a seam
   (`SeamPolicy.exclude_labels_crossing_seam`/`exclude_features_crossing_seam`).

   **Day-edge `UNCOVERED` overhang is a PARTITION ARTIFACT, not a real seam exclusion (Codex #9).**
   `window_vendor_sources` tags any window overhanging the per-day segment partition as
   `UNCOVERED` (`recon/stitch_policy.py:443-444`; its docstring flags cross-midnight as an
   unhandled bar-builder follow-up). With day-partitioned builds (§I), **every** bar within
   `max_lookback_ns` of day-start or within the horizon of day-end overhangs the partition and is
   masked — **even when the adjacent day is the same vendor with continuous coverage and no seam**
   — systematically deleting ~the last 60 s + first `max_lookback` of all 704 days
   (disproportionately the 60 s rung). Do **not** count this under the "correct" exclusions above:
   either **stitch adjacent-day segments for edge windows** (T9 loads the neighbor day's plan) or,
   if deferred, **quantify and explicitly accept** the row loss in the manifest.

   Every manifest records the consumed coverage-policy artifact. Deferred
   stitched-Coinbase manifests additionally record the stitch-plan id and
   `SeamPolicy.as_dict()` in `sources`/`bar_clock`. This is a T9 assembly and
   T9/#69 acceptance criterion for modes that contain seams; G0-BN tests source
   coverage without fabricating a Coinbase seam dependency.

4. **Deferred cross-venue features:** Binance−Coinbase basis (mid spread), lagged Binance
   OFI→Coinbase (E1.2 #1 cross-venue signal). Perp state (funding/OI/liquidations) as
   **conditioners, not primary** (E2.4: OFI ≫ funding/OI at these horizons).

---

## §D. Label horizons

- **G0-BN ladder (fixed):** `{2s, 10s, 60s}` — 2 s and 10 s are required
  primary predictive outcomes; 60 s is control-only and cannot select or rescue
  the verdict. `persistence_zero`, `microprice_raw`, `ofi_ridge`, `lgbm_reg`,
  and `lgbm_clf` run at all three horizons, so the initial ledger contains
  exactly 15 identities. The 60 s candidates remain unselected controls. E1.1
  (`eval/tau.py`) may measure τ only for a later
  formal gate; no data-derived ~20–30 s rung may enter G0-BN. Horizons remain a
  manifest field (`horizons: {"10s": 10_000_000_000, …}`) and a per-row
  `horizon` tag — multi-horizon is native to the schema. A later protocol that
  adds a τ rung must rerun label/matrix production; it cannot edit only the
  manifest because the runner requires actual rows for every declared horizon.
- **Vertical barrier = physical horizon** (§5.4 decoupling: input clock is notional, the
  *target* is fixed physical time). One matrix row per (bar, horizon) tag; the built
  runner groups by `horizon` and gates each rung independently
  (`eval/runner.py`, `eval/study.py`).
- **Every declared horizon must survive masking (Codex #5 — HIGH).** The 60 s rung's label
  window is 30× the 2 s rung's and the guard band (`seam_guard_s = 60 s`, §C.3) is sized to it,
  so 60 s rows are seam/guard-masked far more heavily. If a (sub-)window masks **all** rows of a
  declared horizon, `run_from_manifest` **rejects the whole matrix** (`eval/runner.py:47-50`); if
  it leaves fewer than `n_groups`, `cpcv_splits` raises `n_groups > n_samples`
  (`data/cv.py:47-50`). The producer must guarantee **≥ `n_groups` surviving rows per declared
  horizon** (or emit only horizons that survive), and the §J fixtures must be sized so masking
  cannot zero a horizon.
- **`y_fwd_bps`** = normalized forward return (bps) over the span **`[t_event, t_barrier]`**
  (decision time, §C — *not* the bar close), base price `P0` = the **true reconstructed selected
  target mid at `t_event`** (§B default), **never** the lagged
  `target_read_ts` read (Changelog / #1); **never raw price** (§8).

---

## §E. `t_event`, `t_available`, and the no-lookahead rules

The producer must satisfy every invariant `validate_matrix`
([`eval/matrix.py:58-79`](../../../eval/matrix.py)) and `validate_frame`
([`eval/manifest.py`](../../../eval/manifest.py)) already enforce — these are the
contract, restated as production rules:

| Column | Definition (producer) | Enforced invariant |
| --- | --- | --- |
| `t_event` | **decision** time = **monotone watermark** `max(t_event(N−1), max(received_time) over the bar's trades, cap_fire)` — **non-decreasing across bars** (§C.2/#13); every input gated by `received_time ≤ t_event`; **not** the raw bar close | int64 ns, non-null |
| `t_feature_start` | origin time of the **oldest** look-back event observed by `t_event` (`received_time ≤ t_event`) | `t_feature_start ≤ t_event`; observed look-back ≤ `max_lookback_ns` |
| `t_available` | when features become usable = **`t_event`** (synchronous) | `t_available == t_event` (every input has `received_time ≤ t_event` per §C.2, `availability_lag_ns = 0`) |
| `t_barrier` | first-barrier-hit time (TP/SL/time), forward from `t_event` | `t_event ≤ t_barrier ≤ t_event + horizons[tag]` |

**Coverage integrity (§C.3):** every emitted row's `[t_feature_start, t_event]`
feature window and **`[t_event, t_barrier]`** label window must be covered by the
declared source artifact. Modes with vendor seams additionally require guard-clean,
single-vendor-backed windows (`recon/stitch_policy.py`). The observable feature/cost book read at
`target_read_ts ≤ t_event` sits **inside** the feature window, so the feature-window policy already
covers it — no back-extension (Changelog / #6, #11). Rows failing the masks are dropped, never
NaN-carried into the matrix (**`validate_matrix` rejects NaN/inf features** — §H; `validate_frame`
covers only columns/timing/dtypes). This is the value-level complement to the timing invariants
below.

**Pilot-partition integrity:** G0-BN development rows satisfy the pre-label
conservative cutoff and their actual guarded feature/cost/label support ends
strictly before `2026-01-01`; January holdout support ends before
`2026-02-01`. Deferred modes pin their own boundaries (April→May for the prior
G0-XV protocol). This check runs before write and is independent of source
coverage validity; adjacent-day stitching may bridge ordinary day partitions
but never a holdout boundary. The G0-BN partition-plan binding is part of both
logical builds. Development counts enter the pre-freeze evidence; January
counts enter only the post-raw-burn materialization attestation/report, never
the freeze.

**Unique decision per `(t_event, horizon)` (Codex deep-review #2):** the monotone watermark can tie
several backlog bars to one `t_event`; the producer emits **exactly one row per `(t_event, horizon)`**
(coalesced to the last-closing bar, §C.2), so the evaluator's per-row PnL/trade counting
(`eval/baseline.py:88-105`) cannot score one decision instant as multiple opportunities.

**Well-defined values for legit edge emissions (Codex #4 — HIGH).** `validate_matrix` raises on
the **first** NaN feature (`eval/matrix.py:53-57`) and aborts the whole day/window build, so every
*legitimately emitted* row must be finite — the "drop mask-failing rows" policy above never fires
for these:
- **Zero-/one-trade bars** (cap-closed quiet intervals, §C.2; pinned by T3):
  every trade-flow feature on an empty trade set is explicitly **0**—`cvd`,
  `aggressor_imb`, `vwap_minus_mid`, `largest_print`, `rv_intrabar`,
  `mae_intrabar`, `signed_vol`, and `trade_count`; book-shape features still
  come from the observable snapshot. One-trade path statistics use their finite
  T3 definitions. Do not drop a quiet cap-closed row merely because it has no
  member trade.
- **Unresolved barrier:** `y_fwd_bps` when no TP/SL fires before the vertical barrier = the realized
  return to `t_barrier = t_event + horizon` (never NaN); `label` = 0 (time-barrier).
- **One-sided book:** `half_spread_bps` when the observable book at `target_read_ts` has an empty
  side — **drop the row** (no valid spread) rather than emit NaN.
These are producer obligations, tested in §J.

**Value-level no-lookahead is the producer's own gate.** The manifest validates
*declared* timing and screens *names*; it does **not** prove feature *values* were
computed causally—that is producer-side work
([`docs/feature-manifest.md:18-20`](../../feature-manifest.md)). The historical
gap identified by
[`2026-07-02-lightgbm-manifest-integration.md`](2026-07-02-lightgbm-manifest-integration.md)
is now closed by T3's no-lookahead and invariance tests in
`tests/test_bars_features.py`. Keep those tests, plus
`tests/test_reconstruct_no_lookahead.py` / `tests/test_replay_equivalence.py`, as
the value-level gate; manifest timing checks alone are insufficient.

---

## §F. Embargo / CPCV split generation

**No new CV code — wire to the built `data/cv.py`.** The producer's job is only to emit
the columns CPCV consumes and to pin `embargo_ns` correctly:

- `cpcv_splits(t_event, t0, t1, *, n_groups, k, embargo_ns)` is called by the runner with
  `t0 = t_event`, `t1 = t_barrier` (the label **span**). Purge is **per test interval**
  (`data/cv.py:54-65`) — correct for non-contiguous CPCV combos — so no train label span
  can straddle a test span regardless of embargo. The producer just guarantees
  `t_barrier` is the true resolution time.
- **`max_lookback_ns` spans to the decision time, including the observation delay (P3).** After
  applying the frozen robust cap and dropping every over-cap row, the producer
  sets `max_lookback_ns = max_retained(t_event − t_feature_start)`, where
  `t_feature_start` is the **origin time of the earliest look-back event observed by `t_event`**
  (`received_time ≤ t_event`, §C.2). Measuring to `t_event` therefore **absorbs each feed's
  observation delay** (and, under the Binance-triggered clock, the gap between the trigger
  trade's origin and its `received_time = t_event`); it
  is **not** the raw feature-window length. Undersizing it would let a post-test train row's
  feature window reach into the test label span. The consumer cross-checks exactly this
  quantity (`eval/study.py:28` uses `(t_event − t_feature_start).max()`) and fails closed,
  but the producer must size it right at build time. **Outlier-robust look-back (Codex #13):** a
  raw `.max()` lets a single late-received **old-origin straggler** (admitted by the
  `received_time ≤ t_event` gate) inflate `max_lookback_ns → embargo_ns →` CPCV purging across all
  704 days — and the §E `t_feature_start` invariant is *circular* (the outlier both sets and
  satisfies the bound). Size the look-back with a **robust cap** (high percentile), not a raw max,
  and **DROP** any row whose observed look-back exceeds the cap from the labeled matrix (Codex #B).
  *Flagging* a beyond-cap row while keeping it does **not** work — `validate_frame`/`run_study`
  recompute `max(t_event − t_feature_start)` from the emitted rows and reject a manifest whose
  `max_lookback_ns` is below that observed max; and *clipping* `t_feature_start` understates the
  true feature window → under-embargo. So: drop, or declare the true max and accept the larger
  purge. (Pairs with #7 — both reduce embargo bloat.)
- **Embargo (E0.4): `embargo_ns = max_lookback_ns` (Codex #7).** `t1=t_barrier`
  already includes the actual TP/SL/vertical label span. `_cpcv_iter` starts the
  embargo after `hi`, the merged test interval's **upper** bound = max
  `t_barrier` over test rows (`data/cv.py:61-63`). So — exactly as the
  span-overlap purge above states — the only clearance the embargo must *add* is the **retained
  feature look-back**: `embargo_ns = max_retained(t_event-t_feature_start)`. The schema only requires `embargo_ns ≥
  max_lookback_ns` (`eval/manifest.py`), re-checked at runtime (`eval/study.py:28-32`). Adding
  `max horizon_ns` would **double-count** the horizon and purge up to an extra 60 s of clean train
  rows after every test block — needlessly pushing a rung toward `g1_inconclusive`/fail given the
  ≥32-OOS / `min_trades=30` scarcity limits.
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

- **G0-BN no-trade band** (dedicated #67 scorer): trade only when `|forecast| >
  2·taker_fee_bps + base_slippage_bps + 2·half_spread_bps + margin_bps`.
  Every mask input is frozen or observable at decision time. The realized
  `latency_drift_bps` described below is charged to net PnL after selection and
  MUST NOT enter the mask. The existing `eval/cost.py:net_pnl` helper retains
  its legacy same-cost-for-mask-and-charge behavior for existing callers; #67
  must not pass G0-BN's realized `cost_bps` to that helper as the decision cost.
  Round-trip taker crosses the spread twice (`spread_crossings=2`). **Honest
  taker fills** — no passive-fill-at-mid assumption.
- **`cost_bps`** (per row) = `2 × taker_fee_bps + slippage_bps` using the
  exact serialized T7 `CostAssumption`. For G0-BN it pins `venue=binance`,
  product/source/version identity, the real scalar account-tier
  `taker_fee_bps`, aggregate `base_slippage_bps`, and
  `drift_policy=abs_true_over_observable_mid_v1`; deferred Coinbase modes use a
  separate assumption. Fee applicability/evidence and the no-trade margin are
  required canonical-config fields with no code defaults. **Entry-latency
  slippage (Codex #1):** `slippage_bps = base_slippage_bps +
  abs(true_t_event_mid / observable_mid - 1) * 1e4`. The observed
  `target_read_ts → t_event` drift is charged forward as a cost (the label `P0`
  remains the true `t_event` mid), never shifted into the label or used to
  select a trade. T8/T9 persist `latency_drift_bps` as a required non-feature
  diagnostic so #67 can reconcile `decision_cost_bps = cost_bps -
  latency_drift_bps` and charge full realized cost. V1 has no second
  configurable latency or entry/exit model; changing that formula requires a
  reviewed cost/protocol version.
- **`half_spread_bps`** (per row) = ½·(target best_ask − best_bid)/mid from the **observable**
  target book at **`target_read_ts`** (`received_time ≤ t_event`, §B/§C.2 — the observable book,
  so cost is realistic and uses no future state; a **one-sided** book → drop the row, §E/#4).
  `validate_matrix` requires both cost columns ≥ 0.
- **G0-BN statistical consumer (binding spec §4/§8):** persistence lift is
  exactly `sum(u*(y^2-(y-f)^2))/sum(u*y^2)`, equivalently
  `1-weighted_SSE_model/weighted_SSE_zero`; a zero/non-finite denominator is
  INCONCLUSIVE. The initial effective count is exactly five named candidates ×
  three horizons = 15; unique aborted or changed variants append and increase
  it, while exact deterministic retries are idempotent and conflicting results
  fail closed. Development primary selection uses a paired UTC-day
  circular two-day moving-block bootstrap (10,000 replicates, NumPy PCG64 seed
  0, linear percentiles), one-sided Bonferroni `alpha=0.05/8`, and passing
  PBO/integrity: prefer positive-net-lower-bound trade eligibility, otherwise a
  positive-lift-lower-bound candidate under frozen tie-breaks. OOS uses
  `alpha=0.05/2`. Each evaluated horizon needs ≥20 UTC days and
  `sum(uniqueness)>=100`; there is no row-IID fallback. Both primaries
  predictive plus either primary tradeable is PASS; both predictive with
  neither tradeable is PREDICTIVE_NOT_TRADEABLE; a valid/sufficient transaction
  with either nonpredictive is FAIL; integrity/sufficiency failure is
  INCONCLUSIVE. With `n_groups=6,k=2`, every row's five repeated CPCV test
  forecasts collapse by the binding lexicographic, ordered-float64 arithmetic
  mean and the resulting original-row series alone feeds lift, net, bootstrap,
  DSR, PBO, and selection. DSR uses
  `T=max(2,int(numpy.rint(numpy.float64(effective_trades))))`, with nearest/even
  half ties. PBO orders the five base identities as frozen above, then other
  successful lowercase SHA-256 trial IDs ascending; exact IS ties choose the
  first maximum, and OOS ranking counts every column mean less than or equal to
  the chosen value before division by `n_columns + 1`. The report includes
  paired lift/gross/net uncertainty,
  `decision_trade_rate`, MCC intervals with explicit undefined/degenerate
  reasons, DSR/PBO ledger/split/code provenance, tight/wide spread slices, and
  development-frozen volatility slices. The 60 s controls cannot select,
  authorize, or rescue.
- **Legacy/formal G1 consumer:** `run_study` reports gross **and** net side-by-side, MCC, DSR (vs the trial
  dispersion, effective-N), and **PBO via CSCV** (needs ≥32 finite OOS samples else
  `g1_inconclusive` **only when the gate would otherwise pass** — `g1_inconclusive = passing and
  not pbo_available`, `eval/study.py:70-72`; a weak <32-sample build is an ordinary **FAIL**, §J).
  A real formal-G1 pass needs enough traded samples (`min_trades=30`,
  `min_eff_trades=10`; `eval/runner.py:DEFAULT_GATE`). G0-BN reuses lower-level
  cost/statistical primitives but has its own five-candidate ledger, paired-day
  lift/net uncertainty, required 2 s + 10 s outcomes, and four-way verdict; it
  does not rename this G1 result. Deferred Coinbase/cross-venue gates retain
  their own data prerequisites.
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

- `manifest_version: 1`, `dataset_id`, `build_id` = **content hash over the canonical *logical
  row values* + all build params** (NOT file bytes — pandas/pyarrow embed version-stamped
  `created_by`/`pandas_version` metadata, so byte-identity is environment-coupled; §I/#10), with
  `generated_at` EXCLUDED, `time: {unit: "ns", timezone: "UTC"}`.
- `bar_clock: {kind: "dollar", reference_stream, target_bars_per_day,
  time_cap_ns, warmup_days, threshold_schedule (per-day values or named
  artifact path) + threshold_schedule_hash, feed_lag_tail_ns,
  coverage_policy[, seam_policy]}`. `feed_lag_tail_ns` contains **only feeds
  declared by that mode** and is a p99/max live-watermark bound; offline reads
  use exact per-event `received_time` (§C.2). `seam_policy` is present only for
  a build that consumes a vendor stitch plan. The per-day schedule and coverage
  policy are hash-pinned. `emitted_by_time_cap` is an opted-in diagnostic
  `extra_cols`, never a feature.
- `feature_cols`: **explicit ordered** list (§below). `target_cols: ["y_fwd_bps",
  "label"]` (exactly what the baseline consumes — `eval/runner.py:BASELINE_TARGETS`).
  `reserved_cols`: full `eval.matrix.RESERVED`.
- `venues` and `sources` are **mode-specific, never a union template**:
  - `binance_single_venue`: `venues` is exactly
    `[{exchange:"BINANCE_FUTURES", symbol:"BTC-USDT-PERP"}]`; `sources`
    contains only the #64-certified, #68-sealed allowlisted raw/normalized
    Binance Futures L2 snapshot/delta and trade sources, their certification/
    custody/coverage artifacts, the G0-BN partition plan, and the evidenced
    Binance cost assumption. It also carries exactly one `g0bn_protocol`
    binding. A development manifest binds `partition=development`; the blind-
    materialized OOS manifest additionally carries one `g0bn_holdout_plan`
    binding with the stable universe and transaction IDs plus plan/freeze
    hashes. Coinbase, CoinAPI, stitch artifacts, spot, other instruments/assets,
    funding/OI/liquidations/basis, and extra state feeds are forbidden and fail
    before parquet access wherever the API controls loading.
  - deferred `coinbase_only`/`cross_venue`: declare only venues actually opened
    by that build. A stitched Coinbase build records the CoinAPI/Lake source
    manifests and reviewed stitch plan; a matched cross-venue build records the
    certified overlap and every declared venue.
- `horizons`: `{tag: ns}` (§D). `generated_at`: ISO-8601 UTC, **injectable**
  (a build param, fixed in tests) and **excluded from `build_id`** so identical
  rebuilds share a `build_id`/logical rows (§I/#10).
- `max_lookback_ns`, `embargo_ns` (G0-BN requires equality,
  `embargo_ns = max_lookback_ns`, because `t_barrier` already encodes label
  span; schema-level `≥` remains for other protocols, §F);
  `availability_lag_ns: 0` (synchronous — §C/§E). Optional `extra_cols`
  (`emitted_by_time_cap`; funding/OI diagnostics only in a mode that declares
  those sources), `dtypes`, `gate` (the pre-registered gate block,
  `eval/runner.py:resolve_gate`).

Generic runner preflight rejects any holdout partition,
`g0bn_holdout_plan` binding, or `binance_single_venue_g0bn_oos` dataset before
calling a parquet loader. T9's holdout materializer instead requires the
matching already-durable raw-access claim and never reopens its output; the
dedicated scorer requires the later matrix-access claim. No generic override
flag is permitted.

**Feature registry (explicit `feature_cols`, §6 / E1.2, stationarized).** The
G0-BN registry contains only own-venue book/trade features:
`ofi_integrated`, `microprice_dev`, `queue_imb`, `spread_tick`, `cvd`,
`depth_imbalance`, `book_slope`, `vwap_minus_mid`, `trade_count`,
`signed_vol`, `aggressor_imb`, `largest_print`, `event_intensity`,
`rv_intrabar`, `mae_intrabar`, `elapsed_ns`, and `tod_sin`/`tod_cos`.
Deferred manifests may add `basis_binance_coinbase`, `ofi_binance_lagged`,
spot features, or perp-state conditioners (`funding`, `oi_change`,
`liq_intensity`) only when their mode declares and opens those sources. No
absent-source column is zero-filled. Final lists are pinned per build.
Core names align where possible with the existing synthetic stand-ins
(`eval/synthetic.py:FEATURES`). **Causal normalizers (Codex
deep-review #3):** every stationarizer — rolling z-score/scaler, EWMA, PCA-integrated OFI — must be
fit **as-of `≤ t_event`** (trailing/shifted state, **never** full-window/full-day statistics), or it
leaks future regimes into every feature while still passing `validate_frame` (which checks *declared*
timing, not value causality — feature-manifest.md:17-20). The normalizer's look-back **counts toward
`t_feature_start`/`max_lookback_ns`**, and §J's value-level no-lookahead test (T3) asserts a
post-`t_event` mutation cannot change a past row's normalized feature.

**Validation before write (fail closed):** for development and other rebuildable
modes, the producer runs **both** `validate_frame(matrix,
manifest)` (columns/timing/leakage/horizons/dtypes) **and** `eval.matrix.validate_matrix(matrix,
feature_list(manifest))` — the NaN/inf/finite/duplicate screens that otherwise only run later
inside `run_study` (`eval/matrix.py`) — before writing. `validate_frame` **alone** would let a
matrix with NaN/inf features or costs persist and fail only at eval (Codex P2); running both (or
the full `run_from_manifest`) means a bad build **never** reaches `data/processed/`. A round-trip
test (§J) then runs the artifact through `run_from_manifest` — the same path the CLI uses.

**G0-BN holdout exception:** the raw-burn-authorized blind materializer performs only
the frozen streaming/source/support checks needed to produce and hash outputs;
it closes/fsyncs them and attests actual hashes/counts without reopening a
derived matrix/parquet/footer. After the separate matrix-access burn, the sole
scorer runs the formal manifest/matrix validations above. A bad value is terminal
INCONCLUSIVE, not a rebuild path.

**`regime`** column: G0-BN freezes on development the numeric spread boundary
and exact rules `tight: spread_tick <= boundary`, `wide: spread_tick > boundary`
(compatible with the built per-regime slicing `eval/study.py:78` and
`eval/synthetic.py`). It also freezes the volatility statistic and numeric bin
edges on development; January never refits them. Empty slices and MCC
degeneracies report explicit reasons rather than fabricated zeroes.

---

## §I. Deterministic output paths and schemas

**Paths (all under git-ignored `data/` — AGENTS.md forbids committing vendor/raw data):**

- Per-day intermediates: `data/interim/model_matrix/dt=YYYY-MM-DD.parquet` (day-partitioned
  — AGENTS.md performance rule: multi-GB/day, never load the full window at once).
- Consolidated artifact (the labeled window): **`data/processed/model_matrix.parquet`** +
  **`data/processed/feature_manifest.json`** — the exact paths the integration test and
  CLI expect (`tests/test_baseline_integration.py`, `scripts/run_baseline.py`).

**Determinism (Codex #10).** `plan_lake_binance_batches.py` writes stdlib JSON/text (genuinely
byte-reproducible); **parquet is not** — pandas/pyarrow embed a version-stamped `created_by` +
`pandas_version` blob and writer options (compression, row-group size, dict encoding, statistics)
are unspecified, so two builds on identical data but a different pyarrow/pandas patch produce
different bytes. Determinism is therefore defined as **identical canonical *logical rows* +
identical `build_id`** (`build_id` = hash of canonical logical row values + build params, §H) —
**not** file-byte identity. The §J acceptance test compares **logical-row equality**
(canonicalized values); if byte-identity is wanted instead, first **pin/normalize writer options**
(`version`, `compression`, `use_dictionary`, `write_statistics`, row-group size). Rules: iterate
the **sorted** day list from the mode's hash-pinned certified coverage artifact
(`#68` for G0-BN; `data/usable_calendar.json` only for the deferred legacy
cross-venue scope); apply any coverage/seam masks (§C.3) deterministically;
stable-sort rows by `t_event` (also required for reproducible PBO blocking —
`eval/study.py:58-61`); pin the per-day threshold schedule by hash (§A); seed all RNG
(`np.random.default_rng(seed)`). **`generated_at` is an injectable build param** (fixed in tests,
real wall-clock in production) — the *only* field allowed to differ between otherwise-identical
builds; it never enters `build_id`.

**ModelMatrix schema (one row per bar×horizon):** `feature_cols` (float, no NaN/inf) +
`RESERVED` = `y_fwd_bps`(float bps), `label`(int ∈{-1,0,1}), `t_event`/`t_barrier`/
`t_feature_start`/`t_available`(int64 ns), `cost_bps`/`half_spread_bps`(float ≥0),
`uniqueness`(float ∈(0,1]), `regime`(str), `horizon`(str tag) + opted-in
`extra_cols`. G0-BN requires `latency_drift_bps` as a finite, non-negative
diagnostic `extra_cols` entry; it is never a feature.

---

## §J. Synthetic tests and small fixture tests

Two tiers, matching the repo (pytest, `tests/`, seeded fixtures, skipif-gated real-data
tests — `tests/conftest.py:FIXTURES`, `tests/test_fixture_integration.py`).

**Tier 1 — synthetic + tiny committed fixtures (run in CI, no vendor data):**

- **Bar clock:** seeded trade stream → deterministic bar boundaries; a burst triggers on
  notional, a lull triggers on the time cap (`emitted_by_time_cap`); Coinbase-order
  scramble still yields identical bars (mirrors `tests/test_sample_reconstruct.py` scramble
  test). **PASS/FAIL:** median-bar-time on a planted active regime ≤ 2 s.
- **Threshold causality (P2b):** `threshold_d` is computed from **prior-day** completed volume
  only, so injecting volume into day `d`'s **schedule input** must not change `threshold_d` (nor
  any threshold `≤ d`) — assert on the *threshold value*, **not** the bar boundaries, since the
  clock legitimately re-bins day `d` when its raw trades change (Codex P2: keep raw trades fixed
  and mutate only the completed-volume schedule feed, or inject the spike into a future day).
  Warm-up days use the seed threshold and are flagged/excluded.
- **Decision-time / sample-timing (P1):** the label runs forward from `t_event`; **every
  *feature* read event has `received_time ≤ t_event`** — planting a **delayed** event
  (`received_time > t_event` but `origin_time ≤ t_event`) must **not** enter the *feature* snapshot
  (regression guard: a plain-origin cut would wrongly include it, §C.1/#2). Asserts `t_event` is
  the **monotone watermark** `max(t_event(N−1), max(received_time) over members, cap_fire)`: plant a
  delayed trade in bar N whose `received_time` exceeds bar N+1's members — `t_event` must be
  **non-decreasing** (`t_event(N+1) ≥ t_event(N)`, #13); for an `emitted_by_time_cap` bar (incl. a
  **zero-trade** quiet interval), `t_event ≥` the cap fire time and the prior `t_event`;
  the **observable feature/cost book** resolves to `target_read_ts` while the **label `P0`**
  reads the *true* book at `t_event` (§C.2/#1); `t_available == t_event` holds *without* look-ahead.
- **Trade-flow membership (Codex #3):** plant an early-arriving next-bar trade (origin after the
  crossing trade, `received ≤ t_event`); the bar's `cvd`/`aggressor_imb`/`largest_print` must be
  computed over the **origin-order members** only — asserting `cvd` is bar-additive and prints are
  not double-counted into both bars.
- **Legit edge emissions (Codex #4):** a **zero-trade** cap-closed bar builds with trade-flow
  features = 0 (or is dropped per the pinned policy) and **passes `validate_matrix`** (no NaN); an
  unresolved-barrier row gets the realized return to `t_barrier` + `label = 0`; a one-sided-book
  row is **dropped**, not NaN-emitted. A dead-Sunday window cannot wedge the build.
- **Seam masking (P2a):** a synthetic day with a planted seam (two vendor segments +
  `SeamPolicy` guard) drops every bar whose feature/label window crosses the seam **or sits
  inside the guard band** — masked over each window's **actual per-row span extended by
  `guard_ns`** (`window_crosses_seam(t_event − guard_ns, t_barrier + guard_ns, …)`, §C.3), and
  every window whose `window_vendor_sources` is not a singleton `{lake}`/`{coinapi}` —
  including the endpoint-clean-but-**`excluded`/`UNCOVERED` span *inside* the window** case (the
  per-sample `vendor_source_at` would miss it). **Actual-span vs full-horizon (Codex #A/#15):**
  plant a 60 s row whose barrier resolves early (`t_barrier < t_event + 60 s`) with a seam in
  **`(t_barrier + guard_ns, t_event + 60 s]`** — past the actual span **and its guard band** — it
  must **survive** (the full-horizon `label_valid_mask` would wrongly drop it); a complementary row
  with a seam in **`(t_barrier, t_barrier + guard_ns]`** (inside the actual span's guard band) must
  still be **dropped**, since §C.3 extends the span by `guard_ns` before `window_crosses_seam`. **Horizon survival (Codex #5):** the
  multi-horizon fixture is sized so masking leaves **≥ `n_groups` rows per declared horizon** —
  else the 60 s rung is masked out and `run_from_manifest`/`cpcv_splits` crash instead of
  returning the per-horizon schema.
- **Features:** hand-built L2+trade micro-fixture with a known OFI/CVD/microprice-dev →
  exact expected values. **Value-level no-lookahead (T3):** out-of-order replay ⇒
  byte-identical features (mirrors `tests/test_reconstruct_no_lookahead.py`). **Causal
  normalizers (deep-review #3):** mutating data **after** a row's `t_event` must **not** change
  that row's normalized (z-score/EWMA/PCA) feature values — proves the stationarizer is fit
  as-of, not full-window (§H).
- **As-of barrier volatility (deep-review #4):** the triple-barrier **width** at `t_event` is a
  function of returns `≤ t_event` only — mutating returns **strictly after `t_event`**
  (`(t_event, t_barrier]`, **excluding** the as-of return ending *at* `t_event`, which legitimately
  feeds the trailing EWMA) cannot change the **barrier width**. (The `label` **does** correctly
  depend on that future path — a TP/SL hit or the vertical-barrier return may flip; only the *width*
  is as-of-invariant, so the test asserts on width, **not** `label`.) (§D/T5).
- **Coalesce backlog (deep-review #2):** feeding a delayed backlog that clamps several bars to one
  `t_event` yields **exactly one row per `(t_event, horizon)`** (the last-closing bar); no
  duplicate `(t_event, horizon)` reaches the matrix (§C.2/§E).
- **Labels:** planted up/down/flat paths → expected triple-barrier `label` and sign of
  `y_fwd_bps` off the selected target venue's **mid** anchor (§B); barrier resolves within the horizon
  (`t_barrier ≤ t_event + horizon_ns`). **Span-anchor:** `t0 == t_event` and **`P0` = the true
  reconstructed mid at `t_event`** (an origin cut, not the lagged `target_read_ts` read); since
  the label reads the offline ground-truth book at `t_event`, **no `[target_read_ts, t_event]`
  gap case applies** (Changelog / #1).
- **Leakage-control gate (E0.4):** random k-fold (no purge) shows inflated CV vs
  purged/embargoed on a synthetic overlapping-label series — the controls bite.
- **Pilot partition boundary (P2):** for G0-BN, plant December rows on both sides of the
  per-horizon conservative cutoff. The unsafe row is dropped **without opening a January
  source**, the safe row survives, and no emitted development row has
  `t_barrier + guard_ns >= 2026-01-01T00:00:00Z`. Mirror the fixture at January's end to
  prove holdout labels do not load February. Development counts reconcile to
  the pre-freeze G0-BN partition plan; January counts are produced only in the
  synthetic two-burn path and reconcile to its frozen schema/sufficiency rules.
  Retain equivalent boundary fixtures for deferred modes.
- **Manifest round-trip:** producer emits matrix+manifest on a tiny fixture →
  `validate_frame` passes → `run_from_manifest` runs and returns the per-horizon result
  **schema** without crashing. Assert on **structure, not gate outcome** (Codex P3): a
  tiny/weak fixture yields an ordinary G1 **fail**, not `g1_inconclusive` — that flag needs a
  LightGBM rung to pass the solo gate with PBO unavailable (`eval/study.py:70-72`). Assert a
  specific G1 outcome only with a deliberately-planted-signal fixture sized to pass solo.
- **Mode isolation / manifest templates (P2):** a `binance_single_venue`
  fixture declares exactly one `BINANCE_FUTURES/BTC-USDT-PERP` venue and only
  certified/allowlisted/sealed Binance Futures L2 snapshot/delta and trade plus
  certification/custody/coverage/cost sources. Injecting a Coinbase, CoinAPI,
  stitch, spot, other asset/instrument, funding, OI, liquidation, basis, or
  extra state source/feature fails the G0-BN mode contract before parquet
  access. Deferred mode fixtures prove their own explicit venue/source
  templates without changing G0-BN's row schema.
- **Holdout identity/two-burn order:** changing pilot/config/freeze/source/plan
  fields leaves the stable universe/transaction unchanged. A read spy proves a
  transaction-derived nonblocking process-owner lock spans all outcome-capable
  work; an active duplicate exits `transaction_already_running` without reading
  claims/data or mutating the journal. Raw claim `O_EXCL` + file/directory fsync
  precede the first sealed source/footer read; blind materialization closes and
  attests actual build/manifest/matrix/count/schedule hashes without reopening;
  matrix claim `O_EXCL` + fsync then precedes the scorer's first derived matrix/
  parquet/footer read. A pre-burn owner death is retryable, while only a later
  lock owner classifies a crash-left post-burn nonterminal state INCONCLUSIVE.
  Crashes and materialization/transition/validation/fit/score/write failures
  after either burn all leave terminal INCONCLUSIVE, with no intermediate
  resumption path.
- **Freeze/statistics/report:** a read spy proves freeze construction opens no
  January payload/footer and rejects January build/manifest/matrix/logical-row
  hashes, counts, realized schedule/state, or results. Hand-computed fixtures
  cover the exact uniqueness-weighted persistence-lift formula, zero
  denominator, paired circular two-day PCG64 bootstrap, `0.05/8` development
  and `0.05/2` OOS tails, ≥20-day/`sum(u)>=100` sufficiency, trade-first then
  predictive-only selection, DSR nearest/even effective-trade rounding,
  canonical PBO columns with first-maximum IS and less-than-or-equal OOS ties,
  all four verdicts, MCC degeneracy reasons, and proof that 60 s cannot select/
  rescue.
- **Determinism (P3/#10):** two builds of the same fixture with **different injected
  `generated_at`** ⇒ **identical canonical logical rows and identical `build_id`** (the timestamp
  is excluded from the hash) — assert **logical-row equality**, not raw parquet bytes (which are
  pyarrow/pandas-version-coupled unless writer options are pinned, §I); the manifests differ only
  in `generated_at`.

**Tier 2 — real-data (not a routine test; #69 only after source certification):**
the bounded development data produces the E0.3 median-bar histogram, the fixed
five-candidate x three-horizon ledger, and the outcome-blind config/plan/freeze.
The single #69 command burns raw access, blindly materializes/attests January,
burns matrix access, then validates and scores. There is no τ-selected G0-BN
rung and no post-burn second materialization, validation-only, or scoring path.
Coinbase and matched cross-venue arms run only after a G0-BN PASS. Formal G1
uses a later full-data build and separate holdout. #67 implements distinct
`g0bn-*` paths with synthetic fixtures; #69 alone owns the real score and
verdict.

---

## Before-backfill vs. after-backfill

Coinbase backfill remains gated behind the reviewed-manifest and spend controls,
but it is no longer a prerequisite for the first signal gate. G0-BN instead requires
#64 to certify the selected Binance source and #68 to acquire and certify the bounded
92-day L2+trade window. Those operations remain explicit, bounded, and auditable.
The split is therefore decisive:

**Buildable & fully testable NOW (pre-backfill) — synthetic + tiny fixtures:**
all producer *code* (T1–T9): clock, trailing-threshold schedule,
source-mode snapshot orchestration (Binance single-venue first; Coinbase-only and
dual-book cross-venue modes deferred),
features, received-time per-venue feed-lag reads, **guard-aware seam-masking logic (on
synthetic seams)**, triple-barrier labels, per-horizon uniqueness, cost columns, manifest emission, the
end-to-end orchestrator; every Tier-1 test including value-level no-lookahead, sample-timing,
seam masking, threshold causality, the leakage-control gate, and determinism. This
exercises the **entire plumbing** through the built consumer without a byte of vendor data.

**Requires certified bounded Binance data — #69 (the former T10 label):**
calibrate only from November–December, emit the time-per-bar histogram and
development build, resolve every operator value, run the fixed candidate
ledger, and build the outcome-blind config/holdout plan/freeze. After data-free
refit/preflight, atomically burn raw access before the first sealed January
source/footer read; blindly produce and attest the exact holdout; atomically
burn matrix access before the first derived matrix/parquet/footer read; then
validate, score, and report once. April 2026 remains an integrity-only fixture.
Coinbase seam plans, CoinAPI backfill, and matched cross-venue calibration are
conditional follow-on work after a G0-BN PASS.

---

## Task breakdown and implementation ledger

Each implementation task uses one branch and its own TDD plan; completed rows
remain here as dependency contracts. All T1–T9 and #67 slices are
Tier-1-testable on synthetic data. "T10" is now
only shorthand for the operational #69 transaction, not another implementation
branch or holdout route. Suggested implementation branch names remain in
`feat/…`.

| Task | Scope | Builds on (file:line) | Deliverable | Pre/Post backfill |
| --- | --- | --- | --- | --- |
| **T1** `feat/bars-clock` | **Built by #62.** Target-venue dollar-notional clock (accumulate in `origin_time` order, **`t_event` = monotone watermark `max(t_event(N−1), max(received_time) of the bar's trades, cap_fire)`, non-decreasing across bars** — P1/#13) + hybrid time cap + **trailing/as-of-only per-day threshold schedule + warm-up** (P2b) + `emitted_by_time_cap`; source-specific ordering normalizers. #67/T9 still owns explicit G0-BN source-mode routing and #69 owns real calibration. | `bars/events.py:ClockTrade`; streaming k-way merge (`recon/merge.py:merge_sorted` = fixture oracle only, §C.1) | `bars/clock.py`, `bars/events.py`, schedule/clock tests | **Done**; calibration #69 |
| **T2** `feat/bars-snapshot` | **Built by #74/#77.** Two target-venue reads (#1): observable book at `target_read_ts` (received-gated — features + `half_spread_bps`) and true label book at `t_event` (origin cut — `P0`); received prefilter, staleness cap, top-K mid/microprice. G0-BN target binding remains integration work. | `recon/reconstruct.py:sample_topk_as_of`, `recon/orderbook.py:60-69` | `bars/snapshot.py`, `tests/test_bars_snapshot.py` | **Done** |
| **T3** `feat/bars-features` | **Built by #78/#81.** Per-bar §6/E1.2 vector, causal stationarization, and value-level no-lookahead/scale tests. | T2, `recon/orderbook.py:snapshot` | `bars/features.py`, `tests/test_bars_features.py` | **Done** |
| **T4** `feat/bars-xvenue` | Deferred cross-venue increment: **`t_event` = monotone watermark `max(t_event(N−1), max(received_time) over the bar's trades, cap_fire)` (non-decreasing across bars — #13); every input gated by per-event `received_time ≤ t_event` (exact); p99/max tail only for the live watermark, never medians** (P1); basis; spot/perp and later Coinbase alignment; **sample-timing test (delayed-event guard)**. G0-BN does not depend on this task. | T3, data.md §5/§5b | `bars/align.py` + sample-timing test | Deferred until G0-BN PASS |
| **T5** `feat/labels-triple-barrier` | **Built by #79/#80.** Triple-barrier with trailing EWMA barriers, physical horizons, selected target-venue mid, true-`t_event` `P0`, exact span, and unresolved-barrier policy. #69 still supplies evidenced barrier parameters. | T2 target anchor | `data/labels.py`, `tests/test_labels.py` | **Done**; parameters #69 |
| **T6** `feat/labels-uniqueness-cv` | **Built by #75/#76.** Concurrency uniqueness per horizon, retained-lookback cap/embargo sizing, and the E0.4 leakage-control gate. | `data/cv.py:cpcv_splits`, `eval/synthetic.py:_concurrency_uniqueness`, `eval/runner.py:60` | `data/uniqueness.py`, `tests/test_uniqueness.py` | **Done** |
| **T7** `feat/bars-cost` | **Built by #82/#84.** Per-row `cost_bps = 2*taker_fee_bps + base_slippage_bps + abs(true_t_event_mid/observable_mid-1)*1e4` under `abs_true_over_observable_mid_v1`, plus `half_spread_bps` from the observable target book and the separate `latency_drift_bps` diagnostic. The strict `CostAssumption` pins venue/product/source/version, scalar fee, scalar base slippage, and drift policy; G0-BN additionally freezes fee applicability/evidence and no-trade margin. One-sided book → drop; never reuse Coinbase costs. | `bars/cost.py`, G0-BN spec §3/§8 | `tests/test_bars_cost.py` | **Done**; operator values #69 |
| **T8** `feat/manifest-writer` | `eval.manifest.build_manifest`/`write_manifest`; explicit ordered `feature_cols`; staged identities; G0-BN requires `latency_drift_bps` as a typed non-feature diagnostic in `extra_cols`; emit exactly one source dict each with `name=partition_contract` and `name=g0bn_protocol`; holdout additionally requires exactly one `name=g0bn_holdout_plan` binding with stable-universe/transaction/plan/freeze pins. Development/rebuildable modes run `validate_frame` + `validate_matrix` before write; holdout formal validation is deferred until after the matrix-access burn. Generic manifest preflight recognizes bindings before load. | `eval/manifest.py`, `eval/matrix.py:validate_matrix`, #67-A/D | manifest writer + binding/isolation/round-trip/bad-row tests | Pre |
| **T9** `feat/producer-orchestrator` | End-to-end day-partitioned materialization with explicit source modes, certified allowlists, source-specific replay, masks, and pre-label support cutoffs. It persists T7's realized `cost_bps`, observable `half_spread_bps`, and non-feature `latency_drift_bps` separately. Development may run normally. G0-BN holdout accepts only the exact frozen #68-custodian raw/normalized L2+trade allowlist plus matching durable `g0bn-raw-access-claim-v1`; no ranges/globs/fallbacks or precomputed January counts. It streams once, closes/fsyncs outputs, computes actual logical-row/matrix/manifest/build/count/schedule hashes, writes/fsyncs `g0bn-materialization-attestation-v1`, and never reopens a derived artifact. **Acceptance:** strict Binance isolation, no boundary crossing, all horizons survive, deterministic logical production, and read spies prove no source opens before raw burn and no derived output reopens in materialization. | T1–T8, #67-A/C/E, source certification/custody artifacts | `bars/produce.py` + isolation/two-burn/partition/determinism tests | Pre (synthetic fixtures) |
| **T10 / #69** (operational; no worker branch) | Use November–December only for source/clock/label/calibration decisions; then seal the final v1 operator config **before** registering or executing any candidate. Run the exact 15-entry base ledger plus append-only execution history; log/count but never select any off-protocol identity; select only primary candidates; build outcome-blind `g0bn-holdout-plan-v1` then `g0bn-freeze-v1`; perform data-free refit/preflight; atomically burn raw access; invoke T9 once and attest actual hashes; atomically burn matrix access; only then validate and score with #67's observable decision-cost mask/full realized-cost charge, then emit PASS, PREDICTIVE_NOT_TRADEABLE, FAIL, or INCONCLUSIVE. Any failure after either burn is terminal. No τ rung and no legacy G0-XV identity. | T7–T9, #64, #68, all #67 slices | E0.3/E0.5 artifacts + terminal `g0bn-report-v1` or INCONCLUSIVE consumption record | **#69 only** |

---

## Decisions & defaults

Forks resolved from the docs. Each is a **manifest parameter**, so an ablation needs no
schema change.

1. **Clock trigger venue** = **the selected target venue.** G0-BN uses Binance
   BTC-USDT-perpetual trades. Deferred Coinbase transfer uses Coinbase trades; a
   later matched cross-venue experiment may preregister Binance, Coinbase, or
   combined notional as an ablation. Every choice uses the monotone received-time
   watermark in §C.2 and is persisted in the manifest.
2. **Label anchor** = selected target venue **mid** (primary; Binance perpetual for
   G0-BN, Coinbase for deferred transfer/cross-venue evaluation). For Coinbase, this is
   the anchor the seam-parity gate validates (`recon/parity.py:68,243`). **Microprice** is
   an ablation arm gated on a **source-specific semantic/parity check** (P2c;
   §B). **`P0` is read from the TRUE book at `t_event`** (an
   origin cut, offline ground truth), **not** the observable `target_read_ts` read — the
   entry-latency drift is a `cost_bps` slippage, not a label shift (Changelog / #1).
3. **Sample timing / observability (P1):** decision `t_event` on the **received-time** axis;
   every input gated by its own **`received_time ≤ t_event`** (exact, per-event — the target
   venue is **not** zero-lag), with **p99/max tail** constants (never medians) only for the
   live watermark, giving `t_available == t_event` by construction — never `availability_lag_ns`
   (the consumer requires 0 — §C.2/§E).
4. **Coverage/vendor seams (P2a):** every mode consumes its certified coverage
   artifact. Only a mode with vendor seams consumes a reviewed stitch plan and
   applies `recon/stitch_policy.py`; G0-BN must not declare Coinbase/CoinAPI
   sources or seam artifacts. Hard input + T9/#69 acceptance (§C.3).
5. **Bar-clock threshold (P2b):** **trailing/as-of-only** per-day schedule (prior days
   only) + warm-up. Development pins its realized schedule; freeze pins the
   adaptive rule and exact development-end state, while blind materialization
   attests the realized January schedule — not a single scalar and never a
   pre-freeze January schedule (§A).
6. **Output** = day-partitioned intermediates → one consolidated
   `data/processed/model_matrix.parquet` (+ manifest), matching the built consumer's
   expected paths and AGENTS.md streaming rule.
7. **Determinism (P3/#10)** = canonical **logical rows** + `build_id` (not fragile parquet
   bytes), with `generated_at` **injectable and excluded from the hash** (§I).
8. **G0-BN horizon ladder** = exactly `{2s,10s,60s}`; 2 s and 10 s are
   primary and 60 s is control-only. A ~20–30 s τ rung is a later formal-gate
   protocol and requires a new matrix/config; it cannot enter or change the
   G0-BN verdict (§D).
9. **CV / cost primitives** may reuse pure built code (`data/cv.py`, `eval/`),
   but G0-BN keeps distinct config/ledger/freeze/consumption/report identities
   and its own moving-block uncertainty/selection/verdict path; the producer
   emits the required input columns.
10. **This PR is docs-only** — prior worker PRs already built the T1/T2/T3/T5/
    T6/T7 primitives under `bars/` and `data/`; this PR changes their planning
    contract but adds no code. A `bars/schema.py` contract module remains
    **T8's** concern, not this PR's.

## Required operator values (unresolved until owning task)

These fields have no code defaults and a `g0bn-freeze-v1` builder rejects them
when missing/TBD. They are decisions to resolve from source evidence and
November–December only, never January.

- **Q1 (#69; T7 implementation is complete):** the real account's Binance
  Futures **fee tier**, scalar one-way taker fee, applicability/evidence hash,
  scalar aggregate `base_slippage_bps` with evidence, and no-trade margin. T7's
  `abs_true_over_observable_mid_v1` latency-drift formula is fixed rather than an
  operator knob. A retail/VIP guess is invalid. A later Coinbase mode resolves
  its own cost assumption; neither mode may inherit the other's cost block.
- **Q2 (T4/E2.5) — G0-BN resolved; deferred live cross-venue question:** G0-BN
  is Binance-triggered and uses exact per-event `received_time` offline without
  E2.5. If a later Coinbase-targeted live experiment switches from a local
  Coinbase clock to a remote Binance-triggered clock, E2.5 must first pin the
  Binance live-watermark tail and E2.2 must show the information-content gain is
  worth the lag-modeling risk. This question cannot block or reroute G0-BN.
- **Q3 (T5/#69):** vol-scaling estimator for the horizontal barriers — EWMA half-life of the
  micro-window returns (spec says "EWMA"; the half-life is unspecified). The EWMA must be
  **trailing/as-of `≤ t_event`** (never using returns in/after `[t_event, t_barrier]`, deep-review
  #4), its params persisted; §J: mutating post-`t_event` returns cannot change the barrier width.
  There is no numeric half-life or TP/SL default; freeze fails until all are
  explicitly supplied with evidence.
- **Q4 (T1/#69):** exact adaptive `lookback_days`, `target_bars_per_day`,
  time-cap `T`, `warmup_days`, seed threshold, coverage-normalization rule,
  development schedule, and development-end state. Freeze records the causal
  January update rule/state, not a realized January schedule.
- **Q5 (T5/§B, P2c) — resolved policy:** keep mid as the primary target. Any future microprice
  target requires source-specific semantic/parity evidence and a complete label
  rebuild; stitched Coinbase additionally requires a microprice seam check.
- **Q6 (#64/#68):** selected provider/native product IDs, exact raw/normalized
  L2+trade object allowlist, normalized timestamp/sequence/gap policy,
  source/coverage/custodian-seal hashes, separate custodian/operator identities,
  effective ACL/IAM/bucket-policy evidence, and every outcome-blind included/
  excluded January day with an evidence hash.
- **Q7 (T1–T6/#69):** source-certified book staleness/received-lag caps,
  causal normalizer/lookback cap, partition guard, and TP/SL multipliers. A
  missing value fails config/freeze construction rather than inheriting an
  implementation default.
- **Q8 (#69 reporting):** development-derived numeric tight/wide spread
  boundary and exact inequality rule, plus the volatility statistic and numeric
  bin edges. January cannot refit them.

---

## Validation of THIS PR (docs-only)

- `git diff --check` — whitespace/conflict-marker clean.
- No code touched → `py_compile` N/A (stated per AGENTS.md testing rules).
- Self-review against `docs/experiment-plan.md` (E0.3/E0.4/E0.5, G1) and
  `docs/feature-manifest.md`: (a) explicit `feature_cols`, no all-non-reserved inference;
  (b) `target_cols == {y_fwd_bps, label}`; (c) `availability_lag_ns == 0`;
  (d) G0-BN `embargo_ns == max_retained(t_event-t_feature_start)` after
  `t_barrier`, with schema-level `≥` retained for other protocols; (e)
  reserved-column set matches `eval/matrix.py:RESERVED`; (f) exact
  Binance-source isolation; (g) separate custody and ordered raw/matrix burns;
  and (h) every interface reference cites a real file:line verified in this
  repo. **No live vendor calls run.**
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
  (P1 — trigger uses the trade lag, snapshot the book lag; §C.2), **`half_spread_bps` at the
  observable `coinbase_read_ts`** (P1, §G) — *(the **target/label** snapshot part is
  **SUPERSEDED**: `P0` moved to the true `t_event` book, /code-review #1)* — **CoinAPI fill segments
  replay in `seq` order** not a `ts_engine` merge (P2, §C.1), and the tiny-fixture test
  asserts **schema not `g1_inconclusive`** (P3, §J) — traced to data.md §5b/134-140,
  `recon/coinapi.py:13-29`, `eval/study.py:70-72`.
- Review round 4 (Codex on `991991d`) incorporated: **CoinAPI trades routing** for the 52
  Coinbase fill days — a trades normalizer/replay is a new **producer prerequisite** (P2,
  §C.1; does not exist — `recon/coinapi.py` is book-only), and the **triple-barrier span/purge
  is anchored at `[t_event, t_barrier]`** — *(the "lagged read supplies `P0`" part is
  **SUPERSEDED**: `P0` is the true `t_event` book, /code-review #1)* (P2, §C.2/§D/§B/T5) —
  traced to data.md §5b/§4.3.
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
- Review round 9 (Codex on `f98fd4b`) incorporated: a **received-time-bearing event record is
  an explicit T1/T2 prerequisite** — the normalized `Trade`/`Delta` keep only one `ts_engine`, so
  the `received_time ≤ t_event` gate is unimplementable on the stated contract (P1, §C.2/T1/T4);
  and because origin≠received order within a bar, **`t_event = max(received_time)` over the bar's
  trades**, not the crossing trade's (P1, §C.2/§E/§J); and decision #8's stale "manifest edit" for
  the τ-rung now matches §D (P2) — traced to `recon/events.py`, `recon/ingest.py`.
- Review round 10 (Codex on `faa1e3b`) incorporated: **time-cap closes** — for
  `emitted_by_time_cap` bars (few or **zero** trades) `t_event = max(t_cap, max(received_time))`
  and is **never earlier than the cap fire time** (P1, §C.2/§E/§J/T1) — and decision #1 now
  restates `max(received_time)` over the bar's trades rather than the single trigger receipt (P1).
- Review round 11 (Codex on `aefcb03`) incorporated: the **label vendor/seam mask back-extension**
  to `[coinbase_read_ts, t_barrier]` — *(**SUPERSEDED / obviated** by /code-review #1: with `P0` at
  `t_event` the label window is cleanly `[t_event, t_barrier]` and the back-extension is removed)* —
  and the **threshold-causality test asserts on `threshold_d`, not bar boundaries** (a real trade
  spike legitimately re-bins day `d`; mutate only the schedule feed or a future day; P2, §J).
- **Claude `/code-review` pass (2026-07-03, docs-only) — 14 findings.** *(Codex hit its
  code-review usage limit at round 11; these are from a Claude review, no bot — replied on the
  threads directly.)*
  - **SPINE / #1 (P1) — `P0` label anchor moved from `coinbase_read_ts` back to the true
    reconstructed mid at `t_event`. This SUPERSEDES the earlier lagged-`P0` decision** (Codex
    P1/P2, **rounds 4/10/11**), which introduced a **common-mode target leak**: reading `P0` at
    `coinbase_read_ts < t_event` folded the already-realized `[coinbase_read_ts, t_event]` drift
    into `y_fwd_bps` — drift that is *past* at the decision and observable through the bar's own
    Coinbase trade features (`vwap_minus_mid`/`cvd`/`ofi_integrated`), so a model fits it as
    spurious edge; being identical across the purged and leaky-control pipelines, the E0.4
    leakage-control gate cannot catch it. **Entry-latency is now a cost, not a label shift** —
    charged forward as `cost_bps` slippage (§G). Three-read discipline: **label** = true book at
    `t_event` (origin cut); **features** = observability-gated (`received ≤ t_event`, origin
    order); **cost** = observable book at `coinbase_read_ts` + latency slippage.
    (§B/§C.1/§C.2/§D/§E/§G/§J/T2/T5/T7)
  - **#6 & #11 resolved by OBVIATION:** with `P0` at `t_event` the label window is cleanly
    `[t_event, t_barrier]`; the round-11 back-extension to `[coinbase_read_ts, t_barrier]` is
    **removed** (§C.3/§E) — the observable feature/cost read at `coinbase_read_ts` already sits
    inside the feature window, so `feature_valid_mask` covers it (no separate patch).
  - **#2 (P1):** `sample_topk_as_of` cuts on **origin**, so feature/cost reads **pre-filter
    `received ≤ t_event` then fold in origin order**; the label keeps the plain origin cut (§C.1/§C.2).
  - **#3 (P1):** trade-flow features over the bar's **origin-order members**, not the received-gated
    superset (else CVD non-additive / prints double-counted; §C.2/§J).
  - **#4 (HIGH):** defined finite values for legit edge emissions — zero-/one-trade bars,
    unresolved barrier, one-sided book — so `validate_matrix` can't abort on a legitimate NaN (§E/§J).
  - **#5 (HIGH):** guarantee **≥ `n_groups` rows per declared horizon** (heavy 60 s masking else
    makes the runner/`cpcv_splits` reject the matrix); size §J fixtures accordingly (§D/§J).
  - **#7 (MED):** `embargo_ns = max_lookback_ns` — the test-interval upper edge already includes the
    horizon, so `max horizon_ns` double-counted and over-purged (§F).
  - **#8:** staleness cap on `t_event − coinbase_read_ts` for intra-vendor book-feed dropout (§B).
  - **#9 (MED-HIGH):** day-edge `UNCOVERED` overhang is a **partition artifact** — stitch
    adjacent-day segments for edge windows or quantify/accept the loss, not silent seam-safety (§C.3).
  - **#10 (MED):** `build_id` = hash of **logical rows** + params (parquet bytes are
    pyarrow/pandas-version-coupled); determinism test relaxed to logical equality (§H/§I).
  - **#12:** normalize the trailing threshold by covered fraction (§A).
  - **#13:** robust/capped `max_lookback_ns`, not a raw `.max()` a straggler inflates (§F).
  - **#14 (LOW):** §G wording — `g1_inconclusive` only *when the gate would otherwise pass* (§G).
- Review round 12 (Codex resumed after its limit reset, on `34c3885`) incorporated — 2 findings:
  **#A (P2):** the label seam/guard mask must run over the **actual `[t_event, t_barrier]` span**
  (guard-extended, via `window_crosses_seam`), **not** `label_valid_mask(horizon_ns)` which masks
  the full `[t_event, t_event+horizon]` and over-drops early-resolving (TP/SL) 60 s rows for a
  post-`t_barrier` seam — worsening the #5 horizon-survival risk (§C.3/§J); **#B (P3, historical
  pre-G0-BN contract):** post-backfill Coinbase-mode threshold calibration used real *Coinbase*
  volume. #66 supersedes that old global default: G0-BN uses Binance-perpetual volume directly,
  while deferred Coinbase-targeted builds retain their local Coinbase clock (§A/Q2).
- Review round 13 (Codex on `75ab2ae`) incorporated — 1 finding: **#13 (P1)** — `t_event` is a
  **monotone, cumulative watermark** `max(t_event(N−1), max(received_time) over members, cap_fire)`,
  **non-decreasing across bars**. Per-bar `max(received_time)` *alone* is not monotone: a delayed
  trade in bar N can arrive after bar N+1's members, letting `t_event(N+1) < t_event(N)` — a later
  bar decided on membership unknowable at its claimed time. Clamping to `t_event(N−1)` orders the
  decision times (required by CPCV time-groups, PBO blocking, uniqueness, the stable `t_event`
  sort). §C.2/§E/§J/T1/T4.
- Review round 14 (Codex on `4b1314c`) incorporated — 2 findings: **#A (P2):** `label_valid_mask`/
  `feature_valid_mask` take a **scalar** `horizon_ns`/`lookback_ns` and cannot accept a per-row
  `t_barrier` actual span — the actual-span path stays on `window_crosses_seam`/
  `window_vendor_sources` (or a new vectorized start/end guard helper), never routes `t_barrier`
  through `label_valid_mask` (§C.3); **#B (P2):** beyond-cap look-back stragglers must be **dropped**
  from the labeled matrix, not flagged (the consumer recomputes `max(t_event − t_feature_start)`
  and rejects/inflates) nor clipped (understates the window → under-embargo) — §F/#13.
- Review round 15 (Codex on `f180df8`) incorporated — 1 finding (P2): the §J actual-span seam
  fixture must plant the "survives" seam **past the guard band** (`(t_barrier + guard_ns, t_event +
  60 s]`), not merely past `t_barrier` — since §C.3 extends the actual span by `guard_ns`, a seam in
  `(t_barrier, t_barrier + guard_ns]` must still be **dropped**; the test now asserts both (§J).
- Review round 16 (Codex on `99151d0`) incorporated — 2 findings: **#A (P2):** §A's first clock-rule
  statement now matches the **monotone watermark** (not the single trigger-trade receipt), so T1's
  first read agrees with §C.2/T1; **#B (P3):** the new `bars/` package must be added to
  `pyproject.toml` `[tool.setuptools.packages.find] include` (currently `recon*/eval*/data*`) and
  `tests/test_packaging.py` extended, or non-editable installs / CLI fail to import the producer
  (carried by T1; §Module layout).
- Review round 17 (Codex on `bd202bb`) incorporated — 2 findings: **#A (P2):** the manifest persists
  the **full per-day `threshold_schedule`** (values or a named artifact path) **plus** its hash, not
  the hash alone — a hash can't recover the thresholds for a rebuild/audit (§A/§H); **#B (P3):** the
  §C.2 member-observability parenthetical now says members are observable because **`t_event ≥` their
  max `received_time`** (the watermark is `≥`, not `=`), consistent with the monotone rule.
- **Codex DEEP review (Codex on `b2be897`, requested per AGENTS.md Deep Review Guidelines) — 4
  design-level P2 findings (historical legacy G0-XV wording; superseded for
  G0-BN):** **#1** pilot OOS (April 2026) is **held out first** — G0-CB does not score
  it; legacy T10 calibrates/measures τ/selects on **pre-OOS** data only, pre-registers labels/CV/metrics,
  and **excludes April from the calibration/selection matrix** before G0-XV's sole score
  (data.md:22-25, experiment-plan:27-29; §split); **#2** the monotone
  watermark can **tie backlog bars to one `t_event`** — emit **one row per `(t_event, horizon)`**
  (coalesce to the last-closing bar) so the per-row-scoring evaluator (`eval/baseline.py:88-105`)
  can't count one decision as several trades (§C.2/§E/§J); **#3** feature **normalizers must be causal
  as-of `≤ t_event`** (never full-window), look-back counted in `max_lookback_ns` (§H/§J); **#4** the
  **triple-barrier vol is trailing/as-of** (returns `≤ t_event`), params persisted, with a test that
  post-`t_event` mutations can't change the barrier width (§D/T5/§J).
- Review round 19 (Codex on `48f8a16`) incorporated — 1 finding (P2): the §J as-of-vol test asserts
  only the **barrier width** is invariant to post-`t_event` returns — the `label` **does** correctly
  depend on the future path (`[t_event, t_barrier]`), so a TP/SL hit or vertical-barrier return may
  flip; the test must **not** assert `label` invariance (§J).
- Review round 20 (Codex on `ca8adbb`) incorporated — 1 finding (P2): the as-of-vol mutation window
  is **strictly after `t_event`** (`(t_event, t_barrier]`), excluding the as-of return *ending at*
  `t_event` which legitimately feeds the trailing EWMA — else a correct as-of width could change (§J).

## Risks & assumptions

- **Risk (label-leak fix is load-bearing, #1):** `P0` at `t_event` (the true reconstructed book),
  **not** `target_read_ts`, is the corrected discipline. A future edit that re-lags `P0` to the
  observable read reintroduces the common-mode target leak (Changelog); the
  `[target_read_ts, t_event]` entry drift belongs in realized `cost_bps`
  slippage, not the label or the decision-time no-trade mask.
- **Assume only what the #64/#68 source contract certifies.** G0-BN may not silently
  substitute `received_time` for missing origin time or vice versa; any fallback
  must be explicit, source-specific, and hash-pinned. Timestamp presence still
  does not prove gap absence, so every mode needs the §B staleness cap.
- **Assumes** the consumer contract (`eval/matrix.py:RESERVED`, `eval/manifest.py` v1) is
  stable; if it changes, T8/T9 must re-sync. Low risk — it is frozen and heavily tested.
- **Risk:** source/threshold/latency/fee/calibration/reporting choices (Q1–Q8) remain
  unresolved until their owning evidence exists. The plan makes them required
  canonical-config fields; #69 cannot freeze while any is missing/TBD and no
  code default may fill them.
- **Risk (sample timing, P1):** offline correctness relies on per-event `received_time` being
  present and trustworthy on every declared feed (for example Lake `received_time` or deferred
  CoinAPI `time_coinapi_ns`);
  where a scalar is used (live watermark, manifest bound) it must be a **p99/max tail**, never
  a median (~50 % of events exceed the median — a median read re-opens the ~5–8 % 2 s-label
  look-ahead). G0-BN's offline Binance clock uses exact receipts and is already
  enabled; E2.5 gates only a deferred live remote-trigger clock (Q2).
- **Deferred risk (CoinAPI order + trades, P2):** Coinbase fill segments must replay in `seq` (file)
  order, not a `ts_engine` merge (the opening snapshot carries a prior-day timestamp); T9 must
  dispatch by `vendor_source` or the target book/labels corrupt. **And a CoinAPI
  Coinbase-trades normalizer does not exist yet** — without it the 52 fill days have no trade
  stream for the clock/CVD, silently shrinking the usable calendar. Both are
  prerequisites only for the deferred Coinbase mode (§C.1), not G0-BN.
- **Risk (coverage/seams, P2a):** every mode must consume its final certified
  coverage artifact. A stitched Coinbase build additionally requires the final
  reviewed seam plan; G0-BN must not fabricate that dependency.
- **Deferred risk:** the Coinbase trade-order observation (data.md §5b) is
  one-day/one-symbol evidence; the Coinbase mode sorts defensively and validates
  it before use. It does not constrain G0-BN.
- **Data dependency:** the operational #69 transaction is blocked on #64 source
  certification, #68 bounded data plus independent custodian seal, T7–T9, and every #67 implementation
  slice. The §5a Coinbase backfill/parity gate applies only to deferred
  Coinbase/cross-venue calibration; T1–T9/#67 remain fixture-testable.

## Follow-ups (deferred, tracked)

- Maker/selective-taker fill economics arm for the cost model (§G; spec §10.2).
- Sequential-bootstrap sample weights beyond concurrency uniqueness (§F; López de Prado).
- Alternative volatility statistics/binnings beyond the required
  development-frozen G0-BN volatility slices (§H).
- Imbalance/run bars and richer cross-venue context (spec §12.8 extensions).
- Rust `native/` bar/feature path for wall-clock once the Python producer is the oracle.
