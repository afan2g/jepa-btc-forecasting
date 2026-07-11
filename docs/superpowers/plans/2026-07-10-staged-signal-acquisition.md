# Staged Signal Acquisition and Gate Protocol

**Status:** adopted 2026-07-10 and **amended 2026-07-11 by #66**. The
Binance-first amendment supersedes the former Coinbase-first execution order.
Completed Coinbase quality, reconstruction, manifest, and executor work remains
valid evidence and fallback infrastructure.

**Tracks:** #46 (original policy), #66 (current amendment).

**References:**

- `jepa_btc_forecasting_spec.md`
- `docs/experiment-plan.md`
- `docs/data.md`
- `docs/feature-manifest.md`
- `docs/superpowers/plans/2026-07-03-bar-label-producer.md`
- `docs/superpowers/plans/2026-07-02-binance-downloader-plan.md`
- `docs/superpowers/plans/2026-06-22-lightgbm-baseline.md`

## 1. Decision

Stage evidence and data spend in this order:

1. **Certify one Binance source (#64).** Use bounded existing/sample evidence to
   select Crypto Lake, CryptoHFTData, or a documented fallback for Binance
   BTC-USDT perpetual.
2. **Implement single-venue modeling support (#67).** Add a source-neutral
   `binance_single_venue` producer/evaluator mode.
3. **Acquire only the bounded core dataset (#68).** Pull Binance BTC-USDT
   perpetual L2 snapshots/deltas and trades for 92 days; no spot, auxiliary
   derivatives, Coinbase, other assets, or broad archive.
4. **Run `G0-BN` (#69).** Determine whether own-venue book/trade features predict
   Binance future mid returns with stable OOS lift and positive net performance.
5. **Expand one source at a time after PASS.** Test Binance spot, then
   derivatives state, then Coinbase transfer/cross-venue data, then other
   assets. Every rung must prove incremental OOS net-of-cost lift on fixed target
   rows.
6. **Acquire long history and build JEPA only after the cheap supervised gates.**

The first gate is deliberately one exchange, one instrument, and one asset. More
data sources are not assumed to help: they add missingness, alignment error,
domain shift, and trial multiplicity. They must earn inclusion.

## 2. G0-BN Gate Semantics

### 2.1 Hypothesis

Binance BTC-USDT perpetual L2 book and trade flow predict that instrument's own
2 s / 10 s future mid returns with stable OOS lift over persistence and positive
net performance after realistic Binance execution costs. The 60 s horizon is a
fixed decay/control arm.

Sub-second source events may be reconstructed causally, but a 100 ms forecast
horizon is outside this project's non-HFT scope.

### 2.2 Inputs and baseline ladder

The first gate uses only:

- Binance BTC-USDT perpetual L2 snapshots/deltas;
- Binance BTC-USDT perpetual trades;
- stationarized own-venue OFI, imbalance, microprice displacement,
  spread/tick, trade-flow composition, event intensity, and short realized
  volatility; and
- persistence/no-change, microprice displacement, penalized linear OFI, and
  LightGBM models.

Funding, open interest, liquidations, Binance spot, Coinbase, and other assets
are excluded from the first gate even when files are readily available.

### 2.3 Evaluation contract

- Reconstruct and feature on received-time-observable, deterministic event
  streams. No future snapshot or source event may affect a decision.
- Purge complete label spans and embargo at least the maximum label horizon and
  feature lookback.
- Freeze features, horizons, costs, latency, model configurations, thresholds,
  exclusions, splits, and the complete candidate ledger before OOS loading.
- Charge a versioned Binance fee schedule, observed spread, explicit latency,
  slippage, turnover, and the no-trade band. Report gross and net side by side.
- Report persistence lift, DSR, PBO, bootstrap uncertainty, turnover, and
  spread/volatility regime slices. Accuracy alone cannot pass.
- Every candidate/configuration/horizon/threshold/post-hoc variant enters one
  immutable effective trial count and PBO study.

### 2.4 Outcomes

- **PASS / tradeable:** stable OOS persistence lift and positive preregistered
  net evidence authorize the next incremental-data gate.
- **PREDICTIVE_NOT_TRADEABLE:** stable predictive lift but no taker-cost edge.
  Do not expand automatically; record a human decision for a separately scoped
  fair-value or maker-execution experiment.
- **FAIL:** no stable predictive lift. Stop Coinbase, cross-venue, multi-asset,
  broad-archive, supervised-deep, and JEPA work unless a new pivot is reviewed.
- **INCONCLUSIVE:** data, leakage, cost, PBO, or reproducibility failure. Repair
  fail-closed without selecting a replacement OOS from observed outcomes.

G0-BN is a bounded acquisition/signal screen, not the later full-data formal G1.
Formal G1 may confirm and expand only a premise that G0-BN has already passed.

## 3. Frozen G0-BN Window

| Use | Inclusive dates | Treatment |
|---|---|---|
| Bounded acquisition | `2025-11-01` through `2026-01-31` | 92 days; Binance perpetual L2+trades only. |
| Development/CPCV | `2025-11-01` through `2025-12-31` | 61 days; tuning, CPCV, calibration, and complete trial capture. |
| Fixed OOS | `2026-01-01` through `2026-01-31` | 31 days; physically/logically sealed until complete freeze. |

These are support-span partitions, not filters on `t_event`. Before any adjacent
source or label path is opened, a development candidate is dropped unless its
complete guarded feature, cost, and label support ends before
`2026-01-01T00:00:00Z`. January applies the symmetric future-boundary rule at
`2026-02-01T00:00:00Z`, so labels cannot silently consume February.

The producer records bounds, horizon map, guard, prefilter rule, source hashes,
and per-horizon drop counts in a hash-pinned partition artifact. The artifact
enters every dataset/build ID and is validated before CPCV, fit, or score.

The existing `2026-04-01` Binance smoke is integrity evidence only and is not a
G0-BN modeling partition. April remains frozen for the deferred cross-venue
workflow. G0-BN must not open April features, labels, costs, forecasts, PnL,
trade distributions, or model results.

## 4. Conditional Increment Ladder

After G0-BN PASS, add at most one capability per reviewed gate:

1. **Binance spot increment:** same exchange and asset, separate instrument;
   test incremental lift over the fixed perpetual-only champion.
2. **Perpetual-state increment:** funding, OI, liquidations, and basis; test
   beyond core book/trade flow rather than bundling them into the base model.
3. **Coinbase transfer/cross-venue increment:** #65 selects an affordable,
   certified Coinbase target/control contract. #34 acquires only that approved
   scope; #47/#48 run revised matched controls.
4. **Multi-asset increment:** ETH/SOL/etc. only after same-asset increments pass.
5. **Full history/formal G1:** acquire/reconstruct the approved longer span and
   freeze a new coverage-selected holdout outside all pilot OOS periods.
6. **Supervised deep and JEPA:** proceed only after the supervised signal and
   design gates justify complexity.

Every increment compares base and augmented models on identical target rows,
labels, costs, horizons, and splits. A source cannot win by changing coverage or
dropping difficult periods. Missing source data is an explicit exclusion in the
matched experiment, never a sentinel or zero-filled feature.

The former Coinbase-first `G0-CB`/`G0-XV` protocol is deferred, not silently
executed. If #65 selects an L1-only target feed, the own-book control and matched
arm contract must be reviewed before #47/#48; old multi-level wording cannot be
claimed unchanged. April's one-time holdout transaction remains governed by
#48/#52 if that rung is eventually authorized.

## 5. Dataset and Manifest Contract

The producer emits explicit builds:

- `binance_single_venue_g0bn_dev`: November-December Binance-perpetual features,
  Binance labels/costs, and no January or optional-source access.
- `binance_single_venue_g0bn_oos`: sealed January build opened only by the
  frozen G0-BN selection artifact.
- conditional increment builds: matched-row base and augmented manifests whose
  feature lists differ but whose row IDs, reserved columns, labels, costs,
  horizons, and splits are identical.
- full-data builds: produced only after the preceding acquisition gate passes.

Manifest v1 remains sufficient. `binance_single_venue` uses one venue entry
without a role because the same instrument supplies features and targets.
`dataset_id`, `build_id`, `venues`, `sources`, and explicit `feature_cols` encode
capability; training never infers columns from a frame.

Every manifest pins raw/processed source manifests and hashes, window,
exclusions, clock schedule, feature order, horizons, cost/gate block, partition
artifact, candidate ledger where applicable, and build ID.

## 6. Acquisition and Resource Gates

#64 must select the Binance source before #68 plans the 92-day pull. #68 then:

- enumerates exact source/product/day-or-hour units;
- records conservative and measured bytes, disk, quota, and runtime;
- obtains explicit human approval before vendor I/O;
- downloads atomically and resumably without opportunistic extra feeds;
- preserves source identity and forbids silent cross-vendor fallback;
- reconstructs day-by-day and publishes only certified outputs; and
- seals January outcome-bearing products until G0-BN freeze.

The completed nine-unit `2026-04-01` smoke measured ~0.687 decimal GB. Scaling
that observation over 92 days gives ~63.2 GB, but a single day is **not** an
upper bound, quota guarantee, or approval estimate. The current Crypto Lake
planner instead budgets the required futures delta/trade/seed units at
`0.7788 GB/day`, or **~71.65 GB for 92 days**; that value is also provisional
because some per-feed constants are derived and #64 may select another source.
#68 must replace both projections with the selected source's exact minimal-feed
manifest before approval and retain explicit quota headroom.

Broad Coinbase L3, the former 181-day nine-unit Binance pilot, multi-asset data,
and 12–24-month history remain blocked. A GO decision selects a source or next
experiment; it is never bulk-transfer authorization by itself.

## 7. Issue and Merge Boundaries

- #64: bounded Binance source-quality decision; may continue under this amended
  premise without reopening completed Coinbase work.
- #66: premise/docs/issues/roadmap amendment.
- #67: `binance_single_venue` producer and evaluator support; subissue of #37.
- #68: bounded 92-day Binance-perpetual acquisition and certification.
- #69: decision-bearing G0-BN execution.
- #65/#34/#47: deferred Coinbase target-data and Coinbase-only work, blocked on
  G0-BN PASS.
- #35/#36/#48: deferred six-month cross-venue acquisition/reconstruction/gate,
  blocked on G0-BN PASS and the revised Coinbase contract.
- #49/#50/#38: remaining archive, reconstruction, and formal G1, blocked on all
  preceding gates.

Operational downloads and generated reports remain untracked. Code or durable
docs use one issue, one Conventional Branch worktree, review, and human merge.
Expensive tests, reconstruction, and data jobs serialize on
`/tmp/jepa-expensive-compute.lock`.

## 8. Stop Conditions

Do not start the next source or archive when any of these holds:

- #64 has not certified a Binance source;
- #67 has not proven causal source isolation and fixed-OOS behavior;
- #68 manifests are incomplete, stale, degraded, or unsealed incorrectly;
- G0-BN is FAIL, INCONCLUSIVE, or lacks a reviewed decision;
- G0-BN is PREDICTIVE_NOT_TRADEABLE without an approved fair-value/maker pivot;
- an increment changes target rows, costs, horizons, or splits instead of
  measuring incremental features;
- OOS outcomes influenced source, feature, threshold, cost, model, or exclusion
  selection;
- trial counts, DSR, or PBO do not reconcile;
- vendor, quota, disk, byte, or spend approval is missing; or
- a policy deviation has not been recorded on its owning issue.
