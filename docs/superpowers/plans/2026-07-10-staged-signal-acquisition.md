# Staged Signal Acquisition and Gate Protocol

**Status:** adopted 2026-07-10, **amended 2026-07-11 by #66**, and specified
for G0-BN protocol/freeze semantics by #83 on 2026-07-13. The Binance-first
amendment supersedes the former Coinbase-first execution order. Completed
Coinbase quality, reconstruction, manifest, and executor work remains valid
evidence and fallback infrastructure.

**Tracks:** #46 (original policy), #66 (Binance-first amendment), #83
(executable G0-BN contract).

**References:**

- `jepa_btc_forecasting_spec.md`
- `docs/experiment-plan.md`
- `docs/data.md`
- `docs/feature-manifest.md`
- `docs/superpowers/plans/2026-07-03-bar-label-producer.md`
- `docs/superpowers/specs/2026-07-13-g0bn-protocol.md` (binding G0-BN
  protocol, freeze, one-shot access, metrics, and verdict)
- `docs/superpowers/plans/2026-07-02-binance-downloader-plan.md`
- `docs/superpowers/plans/2026-06-22-lightgbm-baseline.md`

## 1. Decision

Stage evidence and data spend in this order:

1. **Certify one Binance source (#64).** Use bounded existing/sample evidence to
   select Crypto Lake, CryptoHFTData, or a documented fallback for Binance
   BTC-USDT perpetual.
2. **Implement single-venue modeling support (#67).** Add the source-isolated
   `binance_single_venue` producer plus the distinct `g0bn-*` config, ledger,
   freeze, holdout-plan, transaction, runner, and report contracts. Do not
   parameterize or relabel the existing G0-CB/G0-XV evaluator.
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

The complete executable contract is
[`2026-07-13-g0bn-protocol.md`](../specs/2026-07-13-g0bn-protocol.md).
Its definitions are binding for #67 and #69. In particular, `g0bn-trial-v1`,
`g0bn-ledger-v1`, `g0bn-partition-plan-v1`, `g0bn-freeze-v1`,
`g0bn-holdout-plan-v1`, `g0bn-holdout-universe-v1`, `g0bn-one-shot-v1`,
`g0bn-raw-access-claim-v1`, `g0bn-matrix-access-claim-v1`,
`g0bn-consumption-v1`, `g0bn-materialization-attestation-v1`, and
`g0bn-verdict-v1` are distinct from all `g0xv-*` identities; the legacy
Coinbase-targeted evaluator remains a regression contract.

### 2.1 Hypothesis

Binance `BINANCE_FUTURES/BTC-USDT-PERP` (`BTCUSDT` linear perpetual) L2 book
and trade flow predict that instrument's own 2 s and 10 s future mid returns
with stable OOS lift over persistence, and at least one of those primary
horizons has positive net performance after realistic Binance execution costs.
Both primary horizons must establish predictivity. The 60 s horizon is a fixed
control-only arm and cannot select or rescue the verdict.

Sub-second source events may be reconstructed causally, but a 100 ms forecast
horizon is outside this project's non-HFT scope.

### 2.2 Inputs and baseline ladder

The first gate uses only:

- Binance BTC-USDT perpetual L2 snapshots/deltas;
- Binance BTC-USDT perpetual trades;
- stationarized own-venue OFI, imbalance, microprice displacement,
  spread/tick, trade-flow composition, event intensity, and short realized
  volatility; and
- the fixed five-candidate ladder, using these IDs at every horizon:
  `persistence_zero` (no change), `microprice_raw` (raw uncalibrated
  microprice displacement), `ofi_ridge` (uniqueness-weighted OFI-only Ridge),
  `lgbm_reg` (full-feature LightGBM regression), and `lgbm_clf` (full-feature
  LightGBM classification). Ordered features, complete runtime-resolved
  parameters, seeds/threads, preprocessing, code, and software-version hashes
  are part of each trial identity.
  `lgbm_clf` converts its signed probability difference to bps using the
  unweighted NumPy float64 population standard deviation (`ddof=0`) of the
  applicable purged training `y_fwd_bps`, plus exactly `1e-9`; uniqueness
  weights affect fitting but not that scale.

Funding, open interest, liquidations, Binance spot, Coinbase, and other assets
are excluded from the first gate even when files are readily available.

### 2.3 Evaluation contract

- Reconstruct and feature on received-time-observable, deterministic event
  streams. No future snapshot or source event may affect a decision.
- Before parquet access where the API controls loading, require exactly one
  `BINANCE_FUTURES/BTC-USDT-PERP` venue and only #64-certified, #68-sealed
  allowlisted L2 snapshot/delta and trade sources. Reject Coinbase, CoinAPI,
  spot, other assets/perpetuals, funding/OI/liquidations/basis, or any extra
  state/source feature.
- Purge complete label spans with `t0=t_event` and `t1=t_barrier`. Because the
  merged test interval already ends at its maximum `t_barrier`, set CPCV
  `embargo_ns = max_lookback_ns`; adding the horizon would double-count label
  span. Partition/source guards remain separate.
- With `n_groups=6,k=2`, enumerate the 15 test-group pairs lexicographically,
  require exactly five finite test forecasts per row, and collapse them by the
  ordered float64 arithmetic mean. Score each original row once; that collapsed
  series alone feeds development lift, net, bootstrap, DSR, PBO, and selection.
- Convert finite non-negative effective trades for DSR as
  `T=max(2,int(numpy.rint(numpy.float64(effective_trades))))`, using nearest/
  even half ties. For PBO, order the five base identities as frozen below, then
  other successful lowercase SHA-256 trial IDs ascending; exact IS ties choose
  the first maximum, and the OOS rank count includes all column means less than
  or equal to the chosen value before division by `n_columns + 1`.
- Canonically freeze the certified source, producer/clock/label definitions,
  ordered features, horizon roles, real fee tier, cost/slippage/latency block,
  exclusions, partitions/CV, model definitions, thresholds, outcome-blind OOS
  plan, software, and the complete G0-BN ledger before any January outcome read.
  The freeze pins the adaptive threshold rule and exact development-end state,
  never a realized January schedule or January build/manifest/row/count/result.
- Charge T7's exact versioned Binance cost assumption: two sides of one frozen
  scalar taker fee, two observable half-spread crossings, aggregate base
  slippage, and absolute `target_read_ts`-to-`t_event` mid drift under
  `abs_true_over_observable_mid_v1`, plus the frozen no-trade margin. The
  no-trade mask uses only the frozen fee/base allowance and observable spread;
  label-side realized drift is charged to net after selection and cannot affect
  the mask. G0-BN manifests and Parquet store `cost_bps`,
  `half_spread_bps`, and `latency_drift_bps` as exact float64/binary64 columns;
  float32 is invalid under the frozen `1e-12` reconciliation. Report gross and
  net side by side plus
  `decision_trade_rate=n_trades/n_valid_rows`.
- Report paired persistence-lift uncertainty, gross/net uncertainty, MCC
  intervals with undefined/degenerate reasons, development DSR/PBO provenance,
  tight/wide spread slices, and development-frozen volatility slices. Accuracy
  alone cannot pass.
- Every candidate/configuration/horizon/threshold/post-hoc variant enters the
  separate immutable G0-BN effective trial count and PBO study. The base ledger
  has exactly five candidates at each of three horizons (15 identities).
  Unique aborted or changed variants append entries and increase the count;
  exact deterministic retries are idempotent, and a conflicting result for the
  same identity fails closed.

### 2.4 Outcomes

Persistence lift is exactly
`sum(u*(y^2-(y-f)^2))/sum(u*y^2)`, equivalently
`1-weighted_SSE_model/weighted_SSE_zero`; a zero/non-finite denominator is
INCONCLUSIVE. Uncertainty uses a deterministic 10,000-replicate paired UTC-day
**circular two-day moving-block** bootstrap with NumPy PCG64 seed 0 and linear
percentiles, never row-IID resampling. Every evaluated horizon requires at least
20 UTC days and `sum(uniqueness) >= 100`.

Development uses one-sided Bonferroni `alpha=0.05/8` for the four selectable
candidates times two primary horizons. It first chooses among predictive-
eligible candidates with positive lift lower bounds and passing PBO/integrity;
it prefers a trade-eligible candidate with positive net lower bound, then falls
back deterministically to the best lift lower bound. All tie-breaks are frozen
without January. No 60 s candidate is selected. OOS uses one-sided Bonferroni
`alpha=0.05/2`: stable predictivity requires a positive lift lower bound at
**both** 2 s and 10 s, while tradeability requires at least one primary horizon
also to have a positive mean-daily-net lower bound and meet the frozen trade,
DSR, and PBO gates.

| Valid/sufficient transaction | Both primary horizons predictive | Any primary horizon tradeable | Outcome |
| --- | --- | --- | --- |
| No | any | any | **INCONCLUSIVE** |
| Yes | No | any | **FAIL** |
| Yes | Yes | No | **PREDICTIVE_NOT_TRADEABLE** |
| Yes | Yes | Yes | **PASS / tradeable** |

PREDICTIVE_NOT_TRADEABLE permits only a separately reviewed fair-value/maker
experiment. FAIL stops Coinbase, cross-venue, multi-asset, broad-archive,
supervised-deep, and JEPA work unless a new pivot is reviewed. Any failure after
either irreversible access burn is terminal INCONCLUSIVE; it does not permit a
second materialization, validation-only read, score, or outcome-selected
replacement holdout.

G0-BN is a bounded acquisition/signal screen, not the later full-data formal G1.
Formal G1 may confirm and expand only a premise that G0-BN has already passed.

## 3. Frozen G0-BN Window

| Use | Inclusive dates | Treatment |
|---|---|---|
| Bounded acquisition | `2025-11-01` through `2026-01-31` | 92 days; Binance perpetual L2+trades only. |
| Development/CPCV | `2025-11-01` through `2025-12-31` | 61 days; tuning, CPCV, calibration, and complete trial capture. |
| Fixed OOS | `2026-01-01` through `2026-01-31` | 31 days; a separate custodian may seal exact raw/normalized inputs, but the operator gets no outcome access before plan + freeze + the ordered access burns. |

These are support-span partitions, not filters on `t_event`. Before any adjacent
source or label path is opened, a development candidate is dropped unless its
complete guarded feature, cost, and label support ends before
`2026-01-01T00:00:00Z`. January applies the symmetric future-boundary rule at
`2026-02-01T00:00:00Z`, so labels cannot silently consume February.

The G0-BN partition plan freezes bounds, horizon map, guard, prefilter rule,
source hashes, development drop counts, the holdout drop-count schema, and
sufficiency thresholds. It does **not** contain realized January counts: those
would require forbidden pre-freeze materialization. January counts and content
hashes are created and attested only after the raw-access burn.

The coherent sequence is:

1. #68, under a custodian identity and permissions distinct from the
   developer/experiment operator, seals the exact raw and certified normalized
   L2/trade objects plus outcome-blind inventory metadata. Operator-owned files
   plus `chmod` do not satisfy custody.
2. November-December alone determines calibration and development-end adaptive
   state. The final v1 config is then sealed before the first candidate identity
   is registered; all trial executions and the two primary selections follow
   under that immutable config. A stable
   `holdout_universe_id` depends only on `g0bn-v1`, the exact instrument, and
   fixed January/February bounds—not pilot/config/freeze/source/plan/results.
3. An outcome-blind `holdout_plan_sha256` is built before January
   materialization and enters the future build recipe; `g0bn-freeze-v1` pins
   that plan but contains no January build ID, manifest/logical-row/matrix-file
   hash, row/drop count, realized schedule/state, or result.
4. #69 acquires and holds the stable transaction's nonblocking process-owner
   lock across all outcome-capable work, then completes data-free preflight/
   refit and atomically creates/fsyncs the raw-access burn before the first
   January raw/normalized object/payload/footer read. A concurrent live start
   exits `transaction_already_running` without reading claims/data or mutating
   the journal. Its sole blind materializer writes once and attests the actual
   manifest/logical-row/matrix-file/build/count/schedule hashes without
   reopening the derived artifacts.
5. Only after that materialization completes does #69 atomically create/fsync
   the separate matrix-access burn, before the sole scorer first opens the
   derived matrix/parquet/footer to validate and score.

Both burns belong to the same stable transaction/universe. A pre-burn owner
death is retryable; after acquiring the now-free lock, only a later owner may
classify a post-burn nonterminal state as crash-left INCONCLUSIVE. Any crash or
materialization, transition, validation, fit, score, or write failure after
either burn is terminal INCONCLUSIVE; no intermediate state is resumable.

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
- `binance_single_venue_g0bn_oos`: does not exist before the freeze and
  raw-access burn. #69 materializes it once from the holdout plan's exact sealed
  raw/normalized allowlist after the raw-access burn, then attests its actual
  manifest, logical-row, matrix-file, build, count, and realized-state hashes.
  Logical-row content is the modeling identity; the physical matrix-file hash
  is audit-only and cannot enter a development trial ID or effective trial
  count. The scorer opens it only after the separate matrix-access burn.
- conditional increment builds: matched-row base and augmented manifests whose
  feature lists differ but whose row IDs, reserved columns, labels, costs,
  horizons, and splits are identical.
- full-data builds: produced only after the preceding acquisition gate passes.

Manifest v1 remains sufficient. `binance_single_venue` uses exactly one venue
entry because the same instrument supplies features and targets. `dataset_id`,
`build_id`, `venues`, `sources`, and explicit ordered `feature_cols` encode
capability; training never infers columns from a frame. G0-BN manifests also
carry `partition_contract` and `g0bn_protocol` source bindings; the holdout adds
`g0bn_holdout_plan` with the stable universe/transaction and plan/freeze hashes.

Generic runners reject a holdout partition, holdout-plan binding, or
`binance_single_venue_g0bn_oos` during manifest-only preflight before any
parquet loader is called. There is no `--allow-holdout` flag. Only the dedicated
scorer may call lower-level scoring primitives, after its matrix-access burn.

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
- reconstructs development days day-by-day and publishes only certified
  outputs;
- uses a separate custodian identity/permission boundary to own and seal the
  exact January raw and certified normalized L2/trade inputs; records custodian
  and operator identities plus ACL/IAM/bucket-policy evidence; and
- publishes only outcome-blind object IDs, opaque checksums, schema/product IDs,
  declared timestamp-coverage bounds, and continuity/gap metadata. Variable-
  length byte sizes and record counts stay inside custody until after the raw-
  access burn. It does not create or expose a January modeling matrix, label,
  cost, feature, forecast, or metric. Developer-performed filesystem permission
  changes are insufficient custody.

The completed nine-unit `2026-04-01` smoke measured ~0.687 decimal GB. Scaling
that observation over 92 days gives ~63.2 GB, but a single day is **not** an
upper bound, quota guarantee, or approval estimate. The current Crypto Lake
required-feed constants instead sum to `0.7788 GB/day`, or **~71.65 GB for 92
days**; that value is also provisional because some per-feed constants are
derived and #64 may select another source. It is **not** the current batch
planner's output: `scripts/plan_lake_binance_batches.py` still budgets `1.2278
GB/day`, selects both perpetual and spot, and omits `--feeds` (therefore all valid
feeds). #68 must not use that full-archive planner as-is. It must replace both
projections with the selected source's exact minimal-feed manifest (or a
separately reviewed scoped planner) before approval and retain explicit quota
headroom.

Broad Coinbase L3, the former 181-day nine-unit Binance pilot, multi-asset data,
and 12–24-month history remain blocked. A GO decision selects a source or next
experiment; it is never bulk-transfer authorization by itself.

## 7. Issue and Merge Boundaries

- #64: bounded Binance source-quality decision; may continue under this amended
  premise without reopening completed Coinbase work.
- #66: premise/docs/issues/roadmap amendment.
- #67: reviewable config/identity, candidate-ledger, freeze/plan,
  generic-runner guard, stable-universe/two-burn one-shot runner,
  materialization attestation, metrics/report, and synthetic integration slices
  defined in the binding spec. It depends at integration on producer T7 (cost),
  T8 (manifest/bindings), and T9 (raw-claim-gated blind materializer), but
  performs no real execution.
- #68: bounded 92-day Binance-perpetual acquisition/certification and separate-
  custodian sealing of the exact January raw/normalized inputs.
- #69: final operator values, development evidence/config/freeze/plan, atomic
  raw-access burn, one blind January materialization/attestation, atomic matrix-
  access burn, one validation/score, and the terminal verdict. It is the only
  real T10-equivalent operation.
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
- a January outcome artifact existed before the authorized raw-access burn, the
  stable G0-BN transaction already has either claim, custody was not separate,
  or a burned attempt failed;
- trial counts, DSR, or PBO do not reconcile;
- vendor, quota, disk, byte, or spend approval is missing; or
- a policy deviation has not been recorded on its owning issue.
