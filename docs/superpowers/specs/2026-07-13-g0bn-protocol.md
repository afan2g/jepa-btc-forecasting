# G0-BN Protocol, Freeze, and One-Shot Holdout Contract

**Status:** binding design for issue #83. Issue #67 implements this contract;
issue #69 performs the one real-data transaction. This document is the
source-specific protocol authority for G0-BN. The staged acquisition plan and
producer plan remain authoritative for their wider scopes, but their G0-BN
summaries defer to this contract if wording differs.

**No execution is authorized by this document.** It specifies code and artifact
contracts only. Vendor I/O remains under #68 and the sole outcome-bearing run
remains under #69.

## 1. Protocol identity and legacy boundary

G0-BN asks one question: do Binance BTC-USDT linear-perpetual own-book and
own-trade features predict that instrument's future mid return at both primary
horizons, and does the frozen predictor clear a preregistered taker-cost model at
at least one of them?

G0-BN is not an arm, mode, or renamed result of the existing Coinbase-targeted
G0 evaluator. It gets its own artifact namespace:

| Artifact | Required identity |
| --- | --- |
| Canonical protocol config | `g0bn-protocol-config-v1` |
| Candidate trial | `g0bn-trial-v1` |
| Append-only trial ledger | `g0bn-ledger-v1` |
| Partition plan | `g0bn-partition-plan-v1` |
| Outcome-blind holdout plan | `g0bn-holdout-plan-v1` |
| Selection freeze | `g0bn-freeze-v1` |
| Stable holdout universe | `g0bn-holdout-universe-v1` |
| One-shot transaction identity | `g0bn-one-shot-v1` |
| Raw/source-access claim | `g0bn-raw-access-claim-v1` |
| Matrix-access claim | `g0bn-matrix-access-claim-v1` |
| Consumption state/journal | `g0bn-consumption-v1` |
| Blind-materialization attestation | `g0bn-materialization-attestation-v1` |
| Four-way verdict payload | `g0bn-verdict-v1` |
| Terminal report | `g0bn-report-v1` |

`g0bn-holdout-universe-v1` names the stable outcome universe;
`g0bn-one-shot-v1` derives the sole transaction from that universe. The two
access claims and `g0bn-consumption-v1` journal carry both IDs but have distinct
schemas and deterministic paths. None is the version-1 record in
`eval.consumption`.

The existing `g0cb`, `g0xv`, `g0xv-verdict`, `g0xv-freeze`,
`g0xv-holdout`, ledger version 1, April scope, validation states, and CLI
commands are a legacy regression contract. #67 may extract pure hashing,
locking, or matrix-validation helpers, but it must not:

- add G0-BN trials to `eval.ledger.TRIAL_PROTOCOLS` or a G0-XV ledger;
- make a Binance target pass `require_cross_venue_manifest`;
- emit a `g0xv-*` protocol string for a G0-BN artifact;
- reuse the April transaction ID or consumption record; or
- parameterize `run_g0xv_development`/`score_fixed_holdout` and relabel the
  result.

Legacy tests must remain unchanged and green. New G0-BN code uses separate
types, files, entry points, and transaction records.

## 2. Fixed instrument, sources, windows, and horizon roles

### 2.1 Instrument and source isolation

The single venue/instrument declaration is exactly:

```json
{
  "exchange": "BINANCE_FUTURES",
  "native_symbol": "BTCUSDT",
  "symbol": "BTC-USDT-PERP",
  "contract_type": "linear_perpetual",
  "base_asset": "BTC",
  "quote_asset": "USDT",
  "settlement_asset": "USDT"
}
```

The same instrument supplies the trade clock, L2 features, label mid, spread,
and simulated execution costs. The exact-source validator admits one venue
declaration—exactly the object above—and only the #64-certified allowlisted
Binance Futures L2 snapshot/delta and trade products. The final config pins one
provider, each exact native product/object ID, raw and normalized schema
versions, timestamp/sequence/gap rules, source-certification hash, #68
custodian-seal hash, and development processed-source hashes. The holdout plan
pins the exact sealed January raw/normalized object allowlist. January modeling
matrix/content hashes do not exist until blind materialization and enter only
the materialization attestation and terminal journal/report. Silent provider
fallback, a non-allowlisted object, or day-level source mixing is forbidden.

Manifest/source validation runs before any source or matrix parquet access
where the API still controls loading. It rejects a second venue; Coinbase or
CoinAPI; Binance spot; funding, open interest, liquidations, basis, or any other
state feed; another perpetual; another asset; an uncertified product; and any
extra source-derived field in `venues`, `sources`, `feature_cols`, cost inputs,
or the bar clock. Missing optional sources are never synthesized or zero-filled.

### 2.2 Support-span partitions

| Partition | UTC half-open interval | Purpose |
| --- | --- | --- |
| Development | `[2025-11-01T00:00:00Z, 2026-01-01T00:00:00Z)` | Clock/label calibration, CPCV, all candidate attempts, deterministic selection |
| Holdout | `[2026-01-01T00:00:00Z, 2026-02-01T00:00:00Z)` | One outcome-bearing materialization and score |

These are support intervals, not `t_event` date filters. The partition contract
uses the existing conservative rule:

```text
t_event + horizons[horizon] + partition_guard_ns < partition_end_ns
```

and also verifies the actual guarded span ending at `t_barrier`. Development
must not read January to decide whether a row survives; January must not read
February. `partition_guard_ns` is a producer/support guard pinned in the config.
It is not the CPCV embargo.

The existing issue-#52 partition artifact requires realized drop counts for
both development and holdout. G0-BN must not fill that holdout field before
freeze, because doing so would require materializing January. #67 therefore
uses `g0bn-partition-plan-v1`: it freezes the rule, bounds, horizons, guard,
development counts, holdout count schema, and sufficiency thresholds. Actual
January drop counts are first produced and attested after the raw-access burn.
Reusing the span-validation algorithm is fine; requiring a pre-freeze January
count is not.

### 2.3 Horizon roles

| Tag | Nanoseconds | Role | Verdict use |
| --- | ---: | --- | --- |
| `2s` | `2000000000` | primary | Required predictive outcome; may establish tradeability |
| `10s` | `10000000000` | primary | Required predictive outcome; may establish tradeability |
| `60s` | `60000000000` | control-only | Report decay/control evidence; cannot select or rescue a verdict |

All three horizons run the complete fixed ladder and enter the effective trial
count. A strong 60 s result is a follow-up hypothesis, never a G0-BN PASS. No
data-derived tau rung may be added to this protocol; it belongs to a later gate.

## 3. Canonical protocol config

### 3.1 Encoding and hash

`g0bn_protocol_config.json` is typed, strict JSON: UTF-8, sorted object keys,
compact separators, no NaN/Infinity, no unknown fields, and no implicit
defaults. Its `sha256` is SHA-256 of the canonical JSON with only the top-level
`sha256` and `generated_at` fields excluded. Arrays are ordered and therefore
decision-bearing. This is the same logical encoding rule as
`eval.hashing.canonical_json`, not a hash of pretty-printed bytes.

Every other G0-BN JSON artifact uses that canonical encoding too. Its embedded
artifact hash excludes exactly itself and a top-level `generated_at` when the
schema permits that field; all other fields remain hash-bearing. Candidate and
transaction IDs hash the exact identity objects shown below rather than a
container carrying the resulting ID. Binary matrix/source hashes use their
declared byte-level algorithms and are never substituted with JSON hashes.

Every referenced artifact carries an explicit SHA-256. A required operator or
protocol decision may not be `null`, `TBD`, `UNRESOLVED`, or omitted when the
selection freeze is built. An explicitly pinned estimator parameter whose
meaning is `None` (for example Ridge `max_iter`) remains JSON `null` and is
required to be present. A config-construction failure before the first candidate
trial consumes no outcome access and may be corrected. Once any
`g0bn-trial-v1` identity is registered, the v1 protocol config and its eligible
candidate set are immutable; config drift then fails closed rather than
reidentifying prior trials. Config drift after freeze is likewise forbidden.

All timestamps/durations, counts, and seeds are JSON integers (booleans are not
integers); bps, probabilities, and thresholds are finite JSON numbers; switches
are booleans; enum/ID/version fields are non-empty strings; every `*_sha256`
field is 64 lowercase hexadecimal characters (Git object IDs use the
repository's declared object format); and day arrays are sorted unique
canonical `YYYY-MM-DD` strings. Validators enforce exact nested field sets and
reject numeric strings or coercion.

### 3.2 Required top-level object

The schema has exactly these decision-bearing sections:

```json
{
  "schema": "g0bn-protocol-config-v1",
  "protocol_id": "g0bn-v1",
  "pilot_id": "g0bn-2025-11_2026-01-v1",
  "instrument": {},
  "source_certification": {},
  "producer": {},
  "clock": {},
  "features": {},
  "labels": {},
  "costs": {},
  "exclusions": {},
  "partition": {},
  "cv": {},
  "horizons": [],
  "candidates": [],
  "selection": {},
  "verdict_thresholds": {},
  "reporting": {},
  "oos": {},
  "software": {},
  "generated_at": "...",
  "sha256": "..."
}
```

| Section | Required typed content |
| --- | --- |
| `instrument` | Exact object in §2.1; no additional venue |
| `source_certification` | Provider and exact L2/trade product strings; normalized schema/timestamp/sequence/gap policies; certification, custodian-seal, coverage, permission-policy, and development source-manifest SHA-256 strings; distinct custodian/operator identities; no January modeling hash |
| `producer` | Producer entry point, repository commit/tree hash, ordered transform versions, received-time observability rule, staleness/lookback caps, partition-rule version, logical-row/build-ID algorithm, and physical-schema hash algorithm |
| `clock` | `kind=dollar`, Binance-perpetual reference stream, exact development schedule/hash, target bars/day, time cap, warm-up, coverage normalization, monotone watermark, and the frozen adaptive-threshold/OOS causal-update rule plus its exact development-end initial-state hash; no realized January schedule |
| `features` | Ordered registry below; formula/version and development-end normalizer initial-state hash per feature; frozen causal OOS update rule; `max_lookback_ns`; no realized January state |
| `labels` | Mid anchor; bps return formula; trailing EWMA barrier estimator and all parameters; TP/SL multipliers; unresolved-barrier policy; per-horizon uniqueness policy |
| `costs` | Exact T7 `CostAssumption`: `venue=binance`, product/source/version identity, scalar one-way `taker_fee_bps`, scalar aggregate `base_slippage_bps`, and `drift_policy=abs_true_over_observable_mid_v1`; fee-tier applicability/evidence hash; two fee sides; two spread crossings; no-trade margin; observable decision-cost versus realized charged-cost rule; binary64 cost-column storage and reconciliation tolerance |
| `exclusions` | Outcome-blind rule version; exact included days; map of every excluded day to reason and evidence hash; staleness/gap/one-sided-book/lookback rules and drop policy |
| `partition` | Exact nanosecond bounds; horizon map; `partition_guard_ns`; prefilter string; `schema=g0bn-partition-plan-v1` and its SHA-256; realized development counts; holdout count schema and sufficiency rules, but no realized holdout counts |
| `cv` | Fixed §3.4 `n_groups`, `k`, split/grouping/forecast-collapse versions, expected per-row test multiplicity, `embargo_ns`, DSR effective-trade integerization, PBO column/tie/rank rules and blocks/minimum rows, and DSR/PBO definitions and thresholds |
| `horizons` | Ordered objects `{tag, ns, role}` exactly as §2.3 |
| `candidates` | Ordered, complete definitions from §4, including full resolved model parameters and hashes |
| `selection` | Deterministic development ranking/tie rule; eligible candidate IDs; attempt-accounting policy |
| `verdict_thresholds` | All day/uniqueness/trade floors, circular two-day moving-block bootstrap settings, development and OOS Bonferroni tail levels, lift/net lower-bound comparisons, DSR/PBO cutoffs, and truth-table version |
| `reporting` | Exact metric/report versions, spread and volatility regime tag formulas/bin edges, component-cost fields, and strict report/verdict schemas |
| `oos` | Exact January scope, stable universe/transaction algorithms, holdout dataset ID, process-owner lock path/algorithm/lifetime and concurrent-start refusal, separate raw/source- and matrix-access boundaries, materializer/validator/scorer/report versions, output allowlist, and terminal-failure policy |
| `software` | Python, NumPy, pandas, scikit-learn, LightGBM, pyarrow, and repository versions/hashes; deterministic-thread/seed settings |

The final config repeats values rather than relying on code defaults. Code must
compare runtime-resolved values to the config before the raw-access burn.

An adaptive clock or causal normalizer may update from earlier January events
only after the raw/source-access burn. The freeze pins its algorithm,
parameters, ordering, and exact development-end state—not the
outcome-dependent January state values or realized threshold schedule. The
blind materializer derives those values sequentially and its attestation plus
the terminal OOS manifest/report record their full schedules/state hashes.
Using January to choose the rule, thresholds, or reset state remains forbidden.

### 3.3 Ordered G0-BN feature registry

The manifest and full-feature candidates use this exact order:

```text
ofi_integrated
microprice_dev
queue_imb
spread_tick
cvd
depth_imbalance
book_slope
vwap_minus_mid
trade_count
signed_vol
aggressor_imb
largest_print
event_intensity
rv_intrabar
mae_intrabar
elapsed_ns
tod_sin
tod_cos
```

`microprice_dev` is the observable-book microprice minus observable-book mid in
bps. `ofi_integrated` is the pinned causal integrated-OFI scalar. Changing a
formula, order, normalizer, lookback, or feature subset changes the candidate
and protocol hashes.

### 3.4 Fixed CV and development-statistic constants

G0-BN uses `n_groups=6` and `k=2`. Within each horizon, rows are stable-sorted
by `t_event` before `data.cv.cpcv_splits`; the producer's unique
`(t_event,horizon)` invariant makes that order total. Grouping is contiguous and
deterministic, span purge uses `t0=t_event`/`t1=t_barrier`, and
`embargo_ns=max_lookback_ns` as specified in §9. There is no random CV seed.

`cpcv_splits` enumerates the 15 test-group pairs in lexicographic
`itertools.combinations(range(6), 2)` order. Each row must therefore receive
exactly `C(5,1)=5` test predictions. For each trial/horizon, initialize an
IEEE-754 float64 forecast accumulator and integer counter in the stable row
order. Cast each split forecast to float64, require it to be finite, and add it
at its original row position in that fixed split order. After all 15 splits,
fail unless every counter is exactly 5, then define the sole CPCV-OOS forecast
as `f_i = forecast_sum_i / 5.0`. This rule is versioned
`mean_repeated_test_forecasts_v1`; there is no weighting, median, path-level row
duplication, or choice of one split. The `cv` block repeats the version,
combination order, dtype, and expected multiplicity, so `cv_sha256` makes the
collapse identity-bearing.

All development lift, gross/net, trade mask/count, bootstrap, aggregate trade
Sharpe/DSR, and per-trial PBO columns use that one collapsed `f_i` and score
each original row once. Per-split forecasts and fold diagnostics may be retained
for audit, but they cannot feed selection or create extra statistical rows.

For each horizon, development trade Sharpe is the unannualized
uniqueness-weighted mean of traded CPCV-OOS net bps divided by its weighted
population standard deviation. Effective trades are the sum of uniqueness
over those traded rows. G0-BN's versioned DSR calculation pins the following
inputs: candidate trade Sharpe; population standard deviation of the finite
trade Sharpes for all successfully scored unique trial identities at the same
horizon plus `1e-9`; the complete cross-horizon unique-identity ledger count as
`n_trials`; and the traded-PnL sample skew and Pearson kurtosis. Given the
already-computed finite, non-negative binary64 `effective_trades`, its DSR sample
count is exactly:

```text
T_rounded = numpy.rint(numpy.float64(effective_trades))
T = max(2, int(T_rounded))
```

`numpy.rint` is round-to-nearest with exact half cases going to the nearest even
integer (`2.5 -> 2`, `3.5 -> 4`). The result is range-checked and represented as
a signed int64 before conversion to the DSR scalar; there is no floor, ceiling,
half-away, or language-default alternative. The config, metric provenance, and
report carry both the unrounded effective-trade value and resulting `T`, plus
the rule ID `nearest_ties_to_even_int64_v1`. The Bailey/Lopez de Prado
normal-approximation formula then uses that `T` and requires `DSR > 0.95`.
An aborted unique trial identity increases `n_trials` even though it cannot
supply a Sharpe. An exact retry event under the same identity does not.

PBO uses `s=8` contiguous `numpy.array_split` blocks over the common,
chronologically ordered CPCV-OOS net-PnL matrix for every successfully scored
unique trial identity at that horizon, including deterministic baselines.
Columns are canonical: the five eligible base identities come first in §4.1
order, followed by every other successfully scored identity in ascending
lowercase `trial_id` SHA-256 order. Block means are uniqueness-weighted; all
`C(8,4)` train-block combinations use the complement as test. For each
combination, the IS-best is the first column attaining the exact maximum IS
mean in that canonical order, matching `numpy.argmax`; there is no tolerance,
jitter, or average-rank tie break. If `j*` is that column, its OOS rank count is
exactly `sum_j(oos_mean_j <= oos_mean_j*)`, so equal OOS values are included.
Divide that integer by `n_columns + 1`, take its logit, and count it toward PBO
iff the unrounded logit is strictly below zero. The ordered trial-ID list,
`first_max_v1` IS tie rule, and `less_equal_count_v1` OOS rank rule enter the
PBO input hash and config. PBO is
available only with at least two columns and 32 rows finite in every column,
and the frozen requirement is `PBO < 0.50`. These constants and metric-code
hashes are repeated in the config rather than inherited from legacy defaults.

## 4. Fixed candidate ladder and trial identity

### 4.1 Base candidates

Each horizon evaluates these five candidates in this order:

| Order | Candidate ID | Ordered inputs | Definition |
| ---: | --- | --- | --- |
| 0 | `persistence_zero` | `[]` | No-change/non-fitted forecast `0.0` bps; comparison baseline, never selected |
| 1 | `microprice_raw` | `[microprice_dev]` | Raw, uncalibrated observable-book microprice displacement: forecast `1.0 * microprice_dev` bps |
| 2 | `ofi_ridge` | `[ofi_integrated]` | Uniqueness-weighted Ridge regression on `y_fwd_bps` |
| 3 | `lgbm_reg` | Full §3.3 list | Uniqueness-weighted LightGBM bps regression |
| 4 | `lgbm_clf` | Full §3.3 list | Uniqueness-weighted 3-class LightGBM; forecast is `(P(+1)-P(-1))*training_y_std_bps` |

The Ridge fixed parameters are `alpha=1.0`, `fit_intercept=true`,
`copy_X=true`, `max_iter=null`, `tol=0.0001`, `solver=svd`,
`positive=false`, and `random_state=null`; no extra scaling is performed after
the producer's pinned causal stationarization.

Both LightGBM candidates fix `boosting_type=gbdt`, `num_leaves=31`,
`max_depth=-1`, `learning_rate=0.05`, `n_estimators=200`,
`subsample_for_bin=200000`, `min_split_gain=0.0`,
`min_child_weight=0.001`, `min_child_samples=50`, `subsample=0.8`,
`subsample_freq=0`, `colsample_bytree=1.0`, `reg_alpha=0.0`,
`reg_lambda=0.0`, `random_state=0`, `n_jobs=1`,
`importance_type=split`, `verbosity=-1`, `deterministic=true`, and
`force_col_wise=true`. Regression fixes `objective=regression`;
classification fixes `objective=multiclass`, `num_class=3`, and class order
`[-1,0,1]`.

For classifier CPCV, let `y_train_bps` be the finite `y_fwd_bps` values in the
canonical row order of that fold's purged training rows. Its scale is exactly:

```text
training_y_std_bps = numpy.std(
    numpy.asarray(y_train_bps, dtype=numpy.float64),
    dtype=numpy.float64,
    ddof=0,
) + numpy.float64(1e-9)
```

This is the unweighted binary64 population standard deviation: uniqueness
weights fit the classifier but do not enter the scale, and there is no pandas
sample-`std`, `ddof=1`, weighted, or float32 alternative. The terminal refit
uses the same rule over all frozen development rows. The value is never
computed from a CPCV test fold or January. The scale-rule ID
`unweighted_population_float64_plus_1e-9_v1` enters the candidate definition
and trial identity; the ordered split-ID-to-realized-scale list and the
terminal-refit scale enter candidate result/provenance hashes.

The two non-fitted candidates also carry full parameter objects and hashes:
`persistence_zero` hashes `{"forecast_bps":0.0}` and `microprice_raw` hashes
`{"input":"microprice_dev","input_unit":"bps","multiplier":1.0}`.

Those values are the fixed overrides, not permission to omit library defaults.
Every fitted candidate definition also stores:

- package, package version, estimator class, target, loss, sample-weight
  semantics, seed/thread settings, and prediction transform;
- the exact preprocessing/stationarization contract and hash (including the
  explicit absence of candidate-local scaling for `ofi_ridge`), ordered feature
  names, feature formula/version hashes, and candidate implementation/code hash;
- the complete runtime `get_params(deep=false)` result after overrides,
  including every default exposed by the installed pinned version; and
- `model_params_sha256` over that full resolved object.

Every non-fitted definition likewise pins its implementation/code hash,
preprocessing declaration, ordered inputs, and software version. Instantiation
is re-resolved and compared to the stored object before fitting. A package
default, seed, preprocessing, feature order, or code/version change therefore
creates a different trial rather than silently changing one.

### 4.2 Complete attempt ledger

The initial effective trial count is exactly `5 candidates * 3 horizons = 15`.
Deterministic/non-fitted baselines are trials too. Effective `N` counts unique
canonical trial identities, not process executions. A completed or aborted
unique identity counts once. An exact deterministic rerun under the same
identity is an idempotent execution event and does not increase `N`; it must
reproduce the existing result hash. A retry after an infrastructure abort may
append a later completion event under the same identity only when every
identity-bearing input and hash is unchanged.

Any attempted feature subset, model parameter, seed, preprocessing, cost
margin, threshold, label, or horizon changes the canonical identity, creates
another immutable trial, and increases `N`, even if that distinct trial aborts.
No failed or weak identity or event may be overwritten or silently replaced.
Because §4.1 is the complete eligible v1 ladder, an additional identity is not
eligible for v1 selection; it still counts in effective `N` and DSR provenance.
Promoting an added identity requires a separately reviewed future protocol
before any January access, not an in-place edit to `g0bn-v1`.

The G0-BN trial identity is the canonical hash of exactly:

```json
{
  "schema": "g0bn-trial-v1",
  "protocol_config_sha256": "...",
  "source_certification_sha256": "...",
  "development_dataset_id": "...",
  "development_build_id": "...",
  "development_manifest_sha256": "...",
  "development_matrix_sha256": "...",
  "partition_plan_sha256": "...",
  "candidate_id": "...",
  "candidate_definition_sha256": "...",
  "candidate_code_sha256": "...",
  "preprocessing": {},
  "preprocessing_sha256": "...",
  "model_params": {},
  "model_params_sha256": "...",
  "feature_cols": [],
  "seed_and_thread_settings": {},
  "software_versions_sha256": "...",
  "horizon": "2s",
  "horizon_role": "primary",
  "cv_sha256": "...",
  "label_sha256": "...",
  "cost_sha256": "...",
  "thresholds_sha256": "...",
  "variant": "base",
  "variant_params": {}
}
```

The complete resolved parameter and preprocessing objects—not only their
convenience hashes—are identity-bearing. Execution ordinals and timestamps are
ledger-event fields, never trial-identity fields. The separate append-only
ledger stores each identity plus a hash-chained execution-event history; records
starts, aborts, and completions; rejects a conflicting completed result for an
existing identity; and hashes both the ordered event history and canonical
identity/result set. Effective `N` is the number of unique identities, including
those with only an aborted event. It never imports G0-CB/G0-XV entries.

### 4.3 Development selection

CPCV produces one OOS forecast per development row for every trial using the
mandatory §3.4 five-prediction arithmetic-mean collapse. Selection is defined
only for the two primary horizons. For each of the four
non-persistence candidates at each primary horizon, the §8.3 paired circular
two-day bootstrap uses the development CPCV-OOS rows and the one-sided
Bonferroni level `alpha_dev = 0.05 / 8 = 0.00625`. The eight comparisons are
the four selectable candidates times the two primary horizons; lift and net are
two predeclared endpoint families, each using that same per-comparison level.

A candidate/horizon is **predictive-eligible** only when all forecasts and
paired sufficient statistics are finite on the exact common row universe, the
manifest/split/ledger/integrity checks pass, §8.3 development sufficiency holds,
the frozen per-horizon PBO is available and `< 0.50`, and the one-sided
development persistence-lift lower bound is strictly positive. A
**trade-eligible** candidate is predictive-eligible and additionally has a
strictly positive one-sided development mean-daily-net lower bound,
`n_trades >= 30`, `sum_i u_i * trade_i >= 10`, and development `DSR > 0.95`.
The DSR uses the complete effective ledger count; PBO uses the common
chronological CPCV-OOS matrix and carries the ledger, split, metric-code, and
input-matrix hashes.

For each primary horizon, prefer the trade-eligible set. Choose its candidate
by the descending tuple `(net_lower_bound, lift_lower_bound, point_net,
point_lift)`, then by earlier §4.1 order. If that set is empty, selection may
authorize a predictive-but-not-development-tradeable candidate, choosing from
the predictive-eligible set by descending `(lift_lower_bound, point_lift)`,
then earlier §4.1 order. All comparisons use unrounded values; displayed values
never break ties. Here `point_net` is the original-row `mean_daily_net_bps` and
`point_lift` is the original-row `L`, not bootstrap means. If either primary horizon has no predictive-eligible
candidate, the freeze cannot be built and January remains unopened.

No January value enters eligibility or a tie-break. The 60 s horizon has no
selected/champion candidate: all five preregistered 60 s trials remain
descriptive decay controls and may be reported OOS, but none can authorize,
select, or rescue a primary candidate or verdict.

## 5. Outcome-blind holdout plan and selection freeze

### 5.1 Raw acquisition versus outcome access

#68 may acquire, normalize where its certified ingestion boundary requires it,
and seal the exact January raw and normalized L2/trade source objects before
selection. This operation is custodial: a distinct custodian OS/service
identity owns the objects and seal, the developer/experiment-operator identity
has no payload/footer read permission, and the seal pins both identities plus
the effective ACL/IAM/bucket-policy evidence. A developer copying files or
running `chmod` is not separation of custody and cannot satisfy #68.

The authorized custodian may stream transport/source bytes and publish a
signed/hash-pinned, outcome-blind inventory containing object IDs, opaque
cryptographic checksums, schema/product IDs, declared timestamp-coverage bounds,
and continuity/gap flags. Variable-length byte sizes and source record counts
are January activity proxies: they stay inside custody until after the
raw/source-access burn and do not enter the public inventory, config, holdout
plan, or freeze. The custodian must not publish price, trade size, side, return,
spread, label, feature, cost, or other outcome-bearing values. The public
metadata may enumerate scope and outcome-blind exclusions only.

Once #68 seals the inventory, the selection/evaluation plane may not open or
decode a January raw or normalized source payload or parquet footer before the
raw/source-access claim. It may not materialize a January bar/matrix/manifest,
inspect price, size, side, return, label, spread, cost, feature, forecast,
decision, PnL, metric, or outcome distribution, or run a generic evaluator on
January. Filesystem path existence and the already-produced custodian seal JSON
are the only January execution-preflight inputs. If a January matrix already
exists, custody was not independent, or unauthorized outcome access is known
to have occurred, January is compromised: the transaction is INCONCLUSIVE and
no replacement is chosen from observed outcomes.

### 5.2 `g0bn-holdout-plan-v1`

The plan is built and hashed without January outcomes, after the canonical
protocol config and before `g0bn-freeze-v1`. It contains:

- the `holdout_universe_id`, deterministic transaction ID from §6.1, exact
  instrument object, holdout bounds, ordered included day list, and explicit
  outcome-blind excluded-day map;
- protocol-config, source-certification, raw-seal, coverage, and partition
  hashes;
- an exact allowlist of January raw and normalized L2/trade object IDs from the
  custodian seal (not discovered by a glob during execution), plus the
  custodian/operator identities and permission-policy evidence hashes;
- producer entry point and repository/software hashes, all materialization
  parameters, expected dataset ID `binance_single_venue_g0bn_oos`, and logical
  build-ID algorithm;
- expected v1 manifest bindings, horizon map, ordered features, reserved/extra
  columns, exact Arrow schema and schema-hash algorithm, output paths, streaming
  hash algorithms, validation/scoring sequence, and drop-count schema;
- the binding rule that `holdout_plan_sha256` itself enters the OOS logical
  build parameters and manifest binding, breaking any config/freeze/build
  identity cycle without guessing a build hash; and
- two claim boundaries: raw/source access before the first read/open of any
  January raw or normalized object/payload/footer, and matrix access only after
  blind materialization/attestation completes but before the first read/open of
  the derived matrix/parquet or its footer for validation/scoring. Manifest-
  only generic preflight remains permitted solely to reject access.

The plan specifies required drop-count categories and sufficiency comparisons,
not their January values. The holdout `build_id`, manifest hash, logical-row
hash, matrix hash, row/drop counts, realized threshold/normalizer schedules,
and result values are deliberately absent: they do not exist until the sole
blind materialization. The plan pins how each will be derived and attested.

### 5.3 `g0bn-freeze-v1`

The selection freeze is reproducible from development and metadata only. It
contains and pins:

- complete protocol config and holdout-plan hashes;
- selected candidate identity for each primary horizon and the unselected
  five-candidate 60 s control set;
- complete G0-BN ledger/history hashes, effective trial count, development result hash,
  development matrix/manifest/build hashes, split hash, DSR, and PBO evidence;
- source certification, raw seal, coverage, partition contract, producer code,
  and software hashes; and
- exact verdict thresholds, bootstrap/tail rules, cost assumptions, exclusions,
  OOS scope, holdout-universe ID, and transaction ID.

The builder re-derives selection and ledger reconciliation. It rejects missing
operator values, unknown keys, config drift, an unavailable PBO, an unsealed or
non-custodial holdout scope, or any January outcome field. In particular, the
freeze must not contain a January `build_id`, manifest/matrix/logical-row hash,
modeling row/drop count, realized adaptive schedule/state, forecast, metric, or
result. It contains `holdout_plan_sha256`, whose recipe will cause the future
build to bind back to the plan. Thus no OOS outcome data is needed—or
permitted—to build the freeze.

## 6. Protocol-specific one-shot transaction

### 6.1 Stable transaction ID

First compute `holdout_universe_id` as SHA-256 of the canonical JSON for exactly:

```json
{
  "schema": "g0bn-holdout-universe-v1",
  "protocol_id": "g0bn-v1",
  "instrument": {
    "base_asset": "BTC",
    "contract_type": "linear_perpetual",
    "exchange": "BINANCE_FUTURES",
    "native_symbol": "BTCUSDT",
    "quote_asset": "USDT",
    "settlement_asset": "USDT",
    "symbol": "BTC-USDT-PERP"
  },
  "holdout_start_ns": 1767225600000000000,
  "holdout_end_ns": 1769904000000000000
}
```

Then compute the transaction ID as SHA-256 of exactly
`{"schema":"g0bn-one-shot-v1","holdout_universe_id":"..."}` under the same
canonical encoding. Neither identity includes `pilot_id`, candidate/config,
freeze, source, seal, plan, build, manifest, row, attempt, or result values.
Changing one of those values cannot mint a second transaction over the same
outcomes. The consumption directory and both claim paths derive only from the
transaction ID; every record repeats and validates both stable IDs.

### 6.2 State machine

Every invocation first derives the stable transaction ID and opens one
never-replaced owner-lock file at the transaction-derived path using
`O_CREAT|O_RDWR|O_NOFOLLOW`, mode `0600`, with file/directory fsync on first
creation. It then attempts Linux `fcntl.flock(fd, LOCK_EX|LOCK_NB)` and keeps
that file description locked until all outcome-capable work has stopped and a
terminal journal update is fsynced. The file is never unlinked or atomically
replaced; its existence is not a burn, and only the kernel lock state indicates
an active owner. The implementation fails before the raw claim unless the
configured local filesystem and OS provide those semantics. Any outcome-
accessing child must inherit a duplicate of the locked file description and may
not detach, so the lock remains held while any compliant materializer or scorer
can still read or write January artifacts.

Lock contention returns the operational refusal `transaction_already_running`
without reading a claim, journal, January source, or derived artifact and
without writing any state. It is not a fifth verdict and does not consume the
transaction. There is no PID timeout, heartbeat expiry, or lock stealing. Once
a process successfully acquires the lock, no compliant owner is live: only then
may it inspect durable transaction state and classify a raw-claim-bearing
nonterminal state left by a prior process as crash-left INCONCLUSIVE. A
pre-raw-claim crash remains retryable because it accessed no January outcome.
The owner-lock algorithm, path, filesystem requirement, child-inheritance rule,
and non-mutating contention result are pinned in the `oos` config block.

```text
exclusive process-owner lock held around every state below

ABSENT
  | atomic raw-access O_CREAT|O_EXCL + file/directory fsync
  v
RAW_ACCESS_BURNED
  | sole blind materialization closes outputs and fsyncs attestation
  v
MATERIALIZED_UNOPENED
  | atomic matrix-access O_CREAT|O_EXCL + file/directory fsync
  v
MATRIX_ACCESS_BURNED ------------------> SCORED (terminal)
  | any post-raw-burn materialization/validation/fit/score/write error,
  | failed transition, or process loss from any nonterminal state
  v
INCONCLUSIVE (terminal)
```

The raw transition atomically creates `g0bn-raw-access-claim-v1` before the
first January raw/normalized object, payload, or footer read. The matrix
transition is a different atomic create of `g0bn-matrix-access-claim-v1`; it is
forbidden until the blind materializer has closed its outputs and durably
written `g0bn-materialization-attestation-v1`, and it must complete before the
scorer's first derived matrix/parquet/footer read. Both claims carry
the same universe/transaction IDs and config/freeze/plan hashes. A matrix claim
without the matching raw claim and attestation, or before their fsyncs, fails
closed.

`g0bn-consumption-v1` is the separate hash-chained journal of these states. Its
monotone history records both claim hashes, attestation hash when available,
stage, and terminal report/error hash. The claim files are irreversible even if
the journal's best-effort failure append cannot be written. After acquiring the
owner lock, finding a raw claim without a reconciled terminal `SCORED` or
`INCONCLUSIVE` state maps the crash-left transaction to terminal INCONCLUSIVE
without opening any January source or derived artifact. A process that cannot
acquire the owner lock must not perform that classification. There is no
resumable intermediate state, validation-only mode, second scoring call, or
post-burn re-entry path.

All predictable config, development-manifest, refit, shape, permission, and
output-path checks run before the raw claim. If any materialization, validation,
fit, score, or write work nevertheless fails after either burn, the same stable
transaction/universe is terminal INCONCLUSIVE; a config change cannot reset it.

### 6.3 One command owns the transaction

The dedicated #69 runner performs, in order:

1. Derive the stable IDs, acquire and hold the §6.2 process-owner lock, classify
   any crash-left durable state, then load and self-verify config, freeze,
   ledger, holdout plan, partition, source seal metadata, development manifest/
   matrix, and software hashes. Lock contention exits without mutation or data
   access.
2. Enforce the exact certified Binance source template; prepare the two selected
   primary candidates and all five 60 s controls by refitting every fitted
   candidate on its canonical full development rows and instantiating each
   non-fitted candidate; run prediction-shape
   smoke checks; verify custodian identities/policies; preflight fresh output
   paths; and verify that no January derived artifact exists. No January source
   or derived path/footer is opened.
3. Atomically create and durably fsync the raw-access claim. Only after that
   irreversible burn may the custodian release the scoped read capability for
   the exact plan allowlist.
4. The sole blind materializer opens only those allowlisted January
   raw/normalized objects and streams T9 once. It writes the OOS matrix and
   manifest whose build parameters bind `holdout_plan_sha256`; derives the
   frozen adaptive schedule/state causally; and computes the actual logical-row,
   matrix-byte, manifest, physical-schema, build, count, and schedule hashes
   while producing the artifacts. It does not reopen the derived parquet/footer
   or score outcomes.
5. Close and fsync all derived outputs, then atomically write and fsync
   `g0bn-materialization-attestation-v1` containing those actual hashes/counts,
   the physical-schema hash, and the raw claim/plan bindings. A newly discovered
   bad source/day or any materialization error is terminal INCONCLUSIVE, not a
   new exclusion.
6. Only now atomically create and fsync the matrix-access claim, pinning the
   attestation hash. Only after this second irreversible burn may the sole
   scorer first open the derived matrix/parquet/footer.
7. Validate the attestation, source isolation, manifest, physical schema/dtypes,
   matrix values, exact days, partition spans, horizon survival, duplicate
   decisions, logical build ID, realized schedule/state hashes, and every frozen
   pin. Then score the selected primary candidate plus `persistence_zero` at 2 s
   and 10 s, and all
   five frozen 60 s candidates as non-selective controls; compute the report and
   verdict in memory.
8. Atomically replace and fsync the consumption journal with one `SCORED`
   record that embeds the canonical `g0bn-report-v1` payload and its result
   hash. This single commit publishes both the score and state; no second result
   file is required for correctness. Any exception after step 3 leaves the
   transaction consumed and produces an INCONCLUSIVE failure append when
   possible. Release the owner lock only after the terminal journal fsync and
   after every outcome-capable child has exited and released its duplicate lock
   description.

The materializer cannot accept a date range, glob, fallback source, alternate
config, or unclaimed invocation. It consumes the exact plan allowlist and the
matching raw claim. The scorer cannot accept an unattested matrix or run before
the matching matrix claim. Neither phase is exposed through a generic runner.

## 7. Fail-closed generic-runner boundary

Every G0-BN manifest has exactly one source dict with `name=partition_contract`
and one with `name=g0bn_protocol`. A holdout manifest additionally has exactly
one source dict with `name=g0bn_holdout_plan`, carrying the holdout-universe ID,
transaction ID, and plan/freeze hashes. These explicit `name` values satisfy
manifest v1's existing source-entry shape and are uniqueness-checked by the
G0-BN validator. The OOS dataset ID is exactly
`binance_single_venue_g0bn_oos`.

Generic APIs and CLIs are never transaction-authorized. #67 adds a manifest-only
preflight used before `pandas.read_parquet`/`pyarrow.parquet.ParquetFile` or any
caller-supplied loader. It rejects when any of these is true:

- the partition binding says `holdout`;
- a `g0bn_holdout_plan` binding exists;
- `dataset_id == binance_single_venue_g0bn_oos`; or
- a `g0bn_protocol` build has a missing/ambiguous partition binding.

`scripts/run_baseline.py` and any path-based generic runner load and preflight
the manifest before the matrix. The in-memory `run_from_manifest(matrix,
manifest)` also rejects such a manifest immediately, but passing an already
loaded holdout frame to it is itself outside the authorized API. The dedicated
transaction scorer calls lower-level scoring primitives only after the
matrix-access burn; it does not add an override flag to the generic runner.
Renaming/removing bindings breaks the frozen hashes and is not an escape hatch.

## 8. OOS metrics, uncertainty, and verdict

### 8.1 Persistence lift

For one horizon, let `y_i` be future mid return in bps, `f_i` the frozen
candidate forecast, and `u_i` the same-horizon uniqueness weight. Persistence
means unchanged mid and therefore forecast `f0_i = 0` bps. The binding lift is
exactly:

```text
L = sum_i u_i * (y_i^2 - (y_i - f_i)^2) / sum_i u_i * y_i^2
```

Equivalently—and with no sign or baseline ambiguity—if
`weighted_SSE_model = sum_i u_i*(y_i-f_i)^2` and
`weighted_SSE_zero = sum_i u_i*y_i^2`, then
`L = 1 - weighted_SSE_model/weighted_SSE_zero`. No clipping or annualization is
applied. A zero or non-finite `weighted_SSE_zero`, numerator, or resulting
ratio is INCONCLUSIVE. The report also includes uniqueness-weighted RMSE/MAE
and MCC, but they do not replace this definition.

### 8.2 Net performance

For each row, the no-trade decision uses only costs observable or frozen at
decision time. Realized latency drift is still charged after the decision:

```text
fee_bps_i                 = 2 * frozen_taker_fee_bps
spread_bps_i              = 2 * half_spread_bps_i
decision_cost_bps_i       = fee_bps_i + frozen_base_slippage_bps
decision_total_cost_bps_i = decision_cost_bps_i + spread_bps_i
realized_total_cost_bps_i = cost_bps_i + spread_bps_i
trade_i                   = abs(f_i) > decision_total_cost_bps_i + no_trade_margin_bps
gross_i                   = trade_i * sign(f_i) * y_i
net_i                     = trade_i * (sign(f_i) * y_i - realized_total_cost_bps_i)
```

The frozen fee, frozen base-slippage allowance, and observable half-spread are
available when `trade_i` is chosen. `latency_drift_bps_i` is an ex-post
diagnostic derived from T2's origin-cut label book. It MUST NOT feed the
forecast, feature matrix, no-trade mask, or any per-row selection threshold. It
enters the realized charged cost; the resulting realized net metrics may affect
only the already-preregistered development ranking and terminal verdict, never
retroactively alter a trade mask. Costs are never optimized on January.
Component reconciliation is exactly:

```text
fee_bps_i           = 2 * frozen_taker_fee_bps
latency_drift_bps_i = abs(true_t_event_mid_i / observable_mid_i - 1) * 1e4
slippage_bps_i      = frozen_base_slippage_bps + latency_drift_bps_i
cost_bps_i          = fee_bps_i + slippage_bps_i
spread_bps_i        = 2 * half_spread_bps_i
cost_bps_i          = decision_cost_bps_i + latency_drift_bps_i
realized_total_cost_bps_i = decision_total_cost_bps_i + latency_drift_bps_i
```

`observable_mid_i` is the received-time-safe target book at `target_read_ts`;
`true_t_event_mid_i` is T2's label-anchor read at `t_event`. The frozen drift
policy is `abs_true_over_observable_mid_v1`. V1 has no second configurable
latency function or separate entry/exit slippage model: changing that formula
requires a reviewed cost/protocol version rather than an extra runtime
coefficient. T7's `cost_bps_i` remains the realized non-spread cost for
compatibility, and T8/T9 must persist `latency_drift_bps_i` as a required,
non-feature diagnostic. Every G0-BN manifest's `dtypes` map pins `cost_bps`,
`half_spread_bps`, and `latency_drift_bps` to `float64`; T9 supplies an explicit
Parquet/Arrow binary64 (`double`) schema for all three, forbids downcasts, and
includes the physical-schema hash in the materialization attestation. The
scorer rejects any manifest or physical dtype mismatch before arithmetic and
performs no float32 round trip. The dedicated G0-BN scorer derives
`decision_cost_bps_i`, verifies both reconciliation identities above within the
`math.isclose` policy `rel_tol=1e-12, abs_tol=1e-12` bps, and fails closed
(terminal INCONCLUSIVE after either burn) on a missing, negative, non-finite,
or inconsistent component. It MUST NOT call the legacy
`eval.cost.net_pnl` path with realized `cost_bps_i`, because that legacy helper
uses its supplied cost in both the mask and charge. The manifest's serialized
`CostAssumption` must exactly match its venue, product, and normalized source
identity. Report sums of gross, fee, decision cost, spread, base slippage,
latency drift, total slippage, realized total cost, and net bps. Define
`n_trades=sum(trade_i)`,
`effective_trades=sum(u_i * trade_i)`, and
`decision_trade_rate=n_trades/n_valid_rows`,
`round_trip_turnover_units=2*n_trades`, and
`round_trip_turnover_rate=round_trip_turnover_units/n_valid_rows`. The latter two
are the explicit unit-notional turnover proxy: each accepted decision has one
entry and one exit. These metrics do not model capital netting across overlapping
horizons, leverage, capacity, or inventory. A PASS therefore means the frozen
signal clears this taker-cost screen, not that a deployable portfolio has been
validated. A UTC day's gross
or net is the sum of its corresponding row bps, and `mean_daily_gross_bps` and
`mean_daily_net_bps` are arithmetic means across every frozen included day,
including zero-trade days.

Trade/sample Sharpe are the unannualized uniqueness-weighted mean divided by
weighted population standard deviation, respectively over traded rows and all
valid rows including no-trade zeros; a variance-zero or fewer-than-two-row
series reports `0.0` plus its degeneracy reason. MCC is Matthews correlation of
`sign(f_i)` and `sign(y_i)` on traded rows. If there are no decisions, either
side has fewer than two observed classes, or the MCC denominator is zero/non-
finite, MCC is JSON `null`, not `0.0`, and the report supplies the explicit
reason. The config pins the exact regime formulas/bin edges and metric-code
hashes.

### 8.3 Paired uncertainty

The config fixes `bootstrap_kind=paired_utc_day_circular_moving_block`,
`block_length_days=2`, `n_boot=10000`, `seed=0`, and NumPy `PCG64`. The
development and OOS stages instantiate the algorithm separately on their own
sorted UTC-day lists, but use the same fixed algorithm and seed. Let a stage
contain `D` sorted days indexed `0..D-1`, and set `M=ceil(D/2)`. Generate block
starts exactly as:

```python
rng = numpy.random.Generator(numpy.random.PCG64(0))
starts = rng.integers(
    0, D, size=(10000, M), endpoint=False, dtype=numpy.int64
)
```

For replicate `b`, replace each start `s=starts[b,j]` by the circular two-day
block `[s, (s+1) % D]`, concatenate blocks in `j` order, and retain the first
`D` day indices. Thus an odd `D` truncates only the second member of the final
block. Blocks are sampled with replacement; days within a block remain
adjacent, including the last-to-first circular pair. One derived draw matrix is
used for every candidate, horizon, and metric at that stage.

The draw hash is SHA-256 of canonical JSON for
`{"block_length_days":2,"days":days,"dtype":"<i8","schema":"g0bn-circular-day-bootstrap-v1","seed":0,"shape":[10000,D]}`,
then one LF byte, then the derived draw matrix converted to little-endian `<i8`
and serialized in C order. The development evidence records its draw hash. The
outcome-blind holdout plan can precompute the OOS draw hash because the day list
is frozen without materializing January.

Pairing is exact. For day `d`, compute
`A_d=sum_{i in d} u_i*y_i^2` and
`B_d=sum_{i in d} u_i*(y_i-f_i)^2` on the same rows. A replicate aggregates
`A_d-B_d` and `A_d` over the drawn days with multiplicity and recomputes `L`;
model and `persistence_zero` are never resampled separately. Gross and net
replicates aggregate the corresponding row-level daily sums with multiplicity
and divide by exactly `D`. Decision-rate replicates aggregate decision and
valid-row counts. MCC replicates aggregate the traded-row confusion-cell counts
over the same drawn days and recompute MCC. Point estimates always use the
original stage rows, not the mean of bootstrap replicates.

For any finite replicate vector `x`, define
`Q(p)=numpy.quantile(x, p, method="linear")` with no rounding. The descriptive
two-sided 95% percentile interval is `[Q(0.025), Q(0.975)]`. Development
selection uses the one-sided lower percentile
`Q(alpha_dev)` where `alpha_dev=0.05/8=0.00625`. OOS decisions use
`Q(alpha_oos)` where `alpha_oos=0.05/2=0.025`, reflecting the two co-primary
horizons. Lift and net are separate predeclared endpoint families. Their
one-sided tail test rejects `H0: theta <= 0` if and only if the applicable
unrounded lower percentile is strictly greater than zero; equality fails. No
studentization, normal approximation, BCa adjustment, or separately computed
p-value changes that rule.

All 10,000 lift and gross/net replicates required by a gate must be finite; a
zero/non-finite resampled persistence denominator invalidates the uncertainty
and makes the gate INCONCLUSIVE. MCC is descriptive: if its point or a
replicate is undefined, the report uses JSON `null` as applicable and records
`mcc_defined_replicates`, `mcc_undefined_replicates`, and a reason histogram
(`no_decisions`, `single_true_class`, `single_predicted_class`,
`zero_denominator`, or `nonfinite_input`). Its interval uses only finite MCC
replicates and is `null` with a reason when none exist.

At each stage, every evaluated horizon must contain at least 20 distinct UTC
days with a valid row and must satisfy `sum_i u_i >= 100` on its evaluated row
universe. Development selection additionally requires a common row universe
within a horizon for its candidate comparisons. Any failure is an integrity/
sufficiency failure. There is no row-IID, one-day-block, analytic-SE, or shorter-
block fallback. January cannot choose the block unit, seed, replicate count,
tail level, interval method, or sufficiency threshold.

### 8.4 Decision predicates

For each primary horizon `h`:

```text
predictive_h = oos_one_sided_lower_bound_alpha_0.025(L_h) > 0

tradeable_h = predictive_h
              and oos_one_sided_lower_bound_alpha_0.025(mean_daily_net_bps_h) > 0
              and n_trades_h >= 30
              and effective_trades_h >= 10
              and frozen_development_dsr_h > 0.95
              and frozen_development_pbo_available_h
              and frozen_development_pbo_h < 0.50
```

`predictive = predictive_2s and predictive_10s`; both primary outcomes are
required. `tradeable = tradeable_2s or tradeable_10s`; either primary horizon
may establish economics. The 60 s result is absent from both predicates.

`transaction_valid` is true only in the atomically committed `SCORED` journal
whose embedded report/result hash reconciles; both distinct claim files, the
materialization attestation, and every source/config/freeze/plan/software/output
hash reconcile; all §6.3 validations pass; and no post-raw-burn exception
occurred. `sufficient` is true only when every evaluated horizon has at least 20
valid UTC days, `sum_i u_i >= 100`, all required bootstrap replicates and
decision metrics are finite, and the frozen scope/accounting reconciles.
Integrity, sufficiency, custody, or reproducibility failure has precedence over
metric values:

| Transaction valid and sufficient? | Both primary horizons predictive? | Any primary horizon tradeable? | Verdict |
| --- | --- | --- | --- |
| No | any | any | `INCONCLUSIVE` |
| Yes | No | any | `FAIL` |
| Yes | Yes | No | `PREDICTIVE_NOT_TRADEABLE` |
| Yes | Yes | Yes | `PASS` |

FAIL stops expansion. PREDICTIVE_NOT_TRADEABLE permits only a separately
reviewed fair-value/maker study. PASS authorizes consideration of the next
increment; it is not vendor-spend authorization by itself. INCONCLUSIVE retires
this January outcome after the raw/source-access burn; it does not authorize a
second materialization, validation-only read, another score, or an
outcome-selected replacement.

### 8.5 Exact terminal report

The strict `g0bn-report-v1` JSON embedded in the terminal `SCORED` journal
contains:

- protocol/config/freeze/plan/holdout-universe/transaction/ledger/source/
  partition/software hashes, both claim hashes, materialization-attestation
  hash, and the transaction history/state;
- exact instrument, included/excluded days, source objects, raw seal, producer
  version, actual OOS manifest/build/logical-row/matrix/physical-schema hashes,
  row/drop counts, realized adaptive schedule/state hashes, and validation
  checks;
- the complete append-only trial count/history hash, primary selected
  identities/definitions, the five unselected 60 s controls, full resolved
  model/preprocessing/code/version hashes, ordered features, horizon roles,
  and development DSR/PBO values with their ledger, split, matrix, metric-code,
  and effective-trial-count provenance;
- per evaluated candidate/horizon: zero-persistence and candidate weighted loss,
  exact lift `L`, paired percentile interval and applicable one-sided lower
  bound, RMSE, MAE, mean/sum gross, fee, decision cost, spread, base slippage,
  latency drift, total slippage, realized total cost, and net with paired
  uncertainty, both Sharpes, trades, effective trades,
  `decision_trade_rate`, `round_trip_turnover_units`,
  `round_trip_turnover_rate`, and tight/wide spread plus development-frozen
  volatility slices;
- MCC point estimate, paired moving-block interval, defined/undefined replicate
  counts, and explicit point/replicate degeneracy reasons;
- bootstrap algorithm/version/block length/seed/day list/draw-matrix hash,
  development/OOS Bonferroni provenance, percentile method, and every decision
  threshold; and
- every truth-table predicate in a strict `g0bn-verdict-v1` payload, final
  verdict, and result SHA-256.

The frozen reporting block supplies the numeric development-derived spread
boundary and exact rule `tight: spread_tick <= boundary`, `wide: spread_tick >
boundary`. It also supplies the exact volatility statistic and numeric bin
edges fitted on development only. January cannot recompute quantiles or move an
edge. Empty slice metrics are `null` with `empty_slice`; undefined MCC uses the
reasons in §8.3. These descriptive slices never select or rescue a candidate.

A post-burn failure instead ends the `g0bn-consumption-v1` journal with an
INCONCLUSIVE failure payload containing the terminal stage, both claim states,
available attestation/output hashes, and error hash, with no fabricated metric
or `SCORED` report.

`result_sha256` is the canonical report hash with only `result_sha256` and
top-level `generated_at` excluded, matching §3.1; the nested verdict hash is
computed independently under the same self-field rule.

No report field feeds another selection. Post-report analysis is descriptive
and cannot change the verdict.

## 9. CV embargo semantics

For CPCV, `t0=t_event` and `t1=t_barrier`; `t1` already contains each row's
actual realized label span, whether TP, SL, or vertical barrier. `data/cv.py`
first removes every train label span overlapping each merged test-label
interval, whose upper edge is already the maximum test `t_barrier`. It then
starts the embargo after that `t_barrier` upper edge. Consequently:

```text
embargo_ns = max_lookback_ns
```

where `max_lookback_ns = max_retained(t_event - t_feature_start)` after rows
beyond the configured robust lookback cap have been dropped, not clipped. That
difference already includes observation delay. Adding the nominal 2 s, 10 s,
or 60 s horizon again double-counts label span and contradicts `data/cv.py`.
`partition_guard_ns` and source seam guards are separate support-integrity
parameters and do not change this CV equation.

## 10. #67 implementation slices and dependencies

Create reviewable subissues in this order; names here are scopes, not permission
to mutate GitHub from this plan:

| Slice | Scope | Depends on |
| --- | --- | --- |
| **67-A — config and identities** | Strict config/plan schemas, canonical hashes, exact source/horizon/candidate validation, stable universe/transaction IDs | none; pure synthetic |
| **67-B — candidate engine and ledger** | Five-candidate G0-BN development engine, full parameter resolution, CPCV forecasts, DSR/PBO, deterministic selection, separate append-only ledger | 67-A; T8 manifest reader contract |
| **67-C — freeze and holdout plan** | Outcome-blind plan builder, reproducible freeze reconstruction, exact-scope and no-OOS-field validation | 67-A/B; T8 bindings |
| **67-D — generic-runner guard** | Manifest-only holdout rejection in path/in-memory generic APIs and CLI, proven before loader invocation | 67-A; T8 bindings |
| **67-E — one-shot runner** | Process-owner lock with non-mutating active-run refusal, separate durable raw- and matrix-access claims, crash-left non-resumable state machine, pre-burn dev fit, blind T9 materialization/attestation, post-matrix-burn validation/score, terminal journal | 67-C/D; T7 cost contract; T8 writer/bindings; T9 callable materializer |
| **67-F — metrics and report** | Dedicated split decision/realized-cost scorer, paired circular two-day moving-block bootstrap, Bonferroni tails, exact metrics, four predicates/verdicts, strict report and hashes | 67-B/E |
| **67-G — integration/regression** | End-to-end synthetic PASS/PNT/FAIL/INCONCLUSIVE cases and unchanged G0-CB/G0-XV suite | 67-A–F, T7–T9 |

T7 owns honest cost-column production and requires evidenced operator-supplied
Binance values. T8 owns manifest writing and the three G0-BN bindings. T9 owns
source-isolated deterministic blind materialization and attestation; holdout
mode requires the matching raw-access claim and never reopens its derived
output. #67 owns evaluation, freeze, two-burn access control, scoring, and
verdict code; it does not download or run January. #68's separate custodian
owns and seals the exact raw/normalized allowlist. #69 alone supplies final
operator values, builds the development evidence/config/plan/freeze, performs
both burns in order, invokes the sole blind materialization and sole scorer,
and records the real verdict. The old producer-plan T10 is therefore an
operational alias for #69, not another implementation branch or a second
holdout route.

## 11. Focused synthetic acceptance tests

At minimum, #67's subissues include cheap tests for:

1. exact one-venue/certified-L2+trade acceptance and pre-loader rejection of
   Coinbase, spot, other assets, extra state feeds, non-allowlisted objects, and
   extra source features;
2. canonical config/hash round trips, unknown/missing/TBD rejection, array-order
   sensitivity, and tamper detection;
3. exactly 15 base trial identities with IDs `persistence_zero`,
   `microprice_raw`, `ofi_ridge`, `lgbm_reg`, and `lgbm_clf` at each horizon;
   empty-feature persistence identity; ordered feature/preprocessing/parameter/
   seed/code/version sensitivity; and exact unweighted float64 population
   classifier scaling with `ddof=0`, `+1e-9`, and weights proven not to enter;
4. append-only G0-BN ledger behavior, exact-rerun idempotency, conflicting-result
   rejection, unique aborted/additional-variant counting with no replacement,
   and proof that
   legacy G0-CB/G0-XV ledgers/counts/results are unchanged;
5. lexicographic 15-split enumeration, exactly five finite test forecasts per
   row, ordered float64 arithmetic-mean collapse with missing/extra/non-finite
   coverage rejection, proof that the collapsed row series feeds lift/net/DSR/
   PBO, development trade-first then predictive-only selection, exact unrounded
   tie breaks, DSR `T` nearest/ties-to-even cases, canonical PBO column order,
   first-maximum IS and less-than-or-equal OOS tie cases, Bonferroni lower
   bounds, 60 s inability to select/pass/rescue, and DSR/PBO provenance from the
   pinned ledger;
6. freeze construction from development plus seal metadata with a read spy
   proving zero January loader calls; `holdout_plan_sha256` build binding; and
   rejection of a January build/manifest/matrix/row hash, count, schedule/state,
   or result field;
7. exact 31-day plan accounting, outcome-blind exclusions, symmetric partition
   bounds, stable universe/transaction IDs unchanged by config/freeze/source
   edits, and `embargo_ns=max_retained(t_event-t_feature_start)` after
   `t_barrier` without horizon double-counting;
8. generic CLI/API rejection of holdout bindings before a parquet loader spy is
   called, including missing/ambiguous binding cases;
9. independent process-owner `flock` and `O_EXCL`/fsync durability: a concurrent
   invocation gets `transaction_already_running` without reading claims/data or
   mutating the journal; the owner lock spans every outcome-capable child; raw
   access precedes the first sealed source/footer read and matrix access follows
   attestation but precedes the first derived matrix/parquet/footer read;
10. pre-burn owner death remaining retryable, while owner death after the raw
    burn plus materialization, transition, validation, fit, score, and output-
    write failures each leave the stable transaction terminal INCONCLUSIVE;
    only a new lock owner consumes a crash-left nonterminal state, without a
    January read;
11. a hand-computed paired circular two-day lift/gross/net/MCC bootstrap with
    PCG64 seed 0, 10,000 replicates, linear percentiles, development `0.05/8`
    and OOS `0.05/2` tails, `>=20` day and `sum(u)>=100` failures, and no row-IID
    fallback;
12. exact decision/realized-cost reconciliation plus a causality test that
    mutates only `true_t_event_mid`/`latency_drift_bps` while holding forecasts,
    observable books, and frozen costs fixed: the trade mask/count must remain
    byte-identical while realized net changes; float64 manifest/physical-schema
    round trips at the `1e-12` tolerance and float32 inputs fail closed; and
13. all four truth-table outcomes, full report schema/hash, and a check that no
    60 s metric can change the verdict.

All use synthetic/tiny fixtures. No test opens vendor data, runs #69, or creates
a second access path.

## 12. Values that remain freeze blockers

Some values are supplied by source evidence and some by the #69 operator. All
are explicit blockers to a final freeze, not defaults to hide in code:

- #64-selected provider, native product IDs, timestamp/sequence policy, and
  certification hash;
- #68 exact raw/normalized object allowlist, custodian-seal/coverage hashes,
  separate custodian/operator identities and effective permission-policy
  evidence, and outcome-blind included/excluded January day accounting;
- the real Binance Futures account/taker fee tier, scalar one-way fee in bps,
  applicability interval, and evidence hash (standard retail/VIP guesses are
  invalid);
- the scalar aggregate `base_slippage_bps` and its evidence hash; T7's observed
  `target_read_ts`-to-`t_event` absolute-mid drift and
  `abs_true_over_observable_mid_v1` policy are fixed, not operator-selectable;
- source-certified book staleness and received-lag caps;
- target bars/day, time cap, warm-up, exact adaptive-threshold/OOS causal update
  rule, development schedule and development-end initial state, and coverage
  normalization chosen on development only (never a realized January schedule);
- causal feature normalizer/lookback cap and triple-barrier EWMA half-life and
  TP/SL multipliers;
- the support-only `partition_guard_ns`, justified by the certified producer
  and source boundary rules;
- any outcome-blind source exclusion with an exact reason/evidence hash;
- development-derived numeric tight/wide spread boundary and frozen volatility
  statistic/bin edges used only for reporting slices.

The #64 source decision is evidence-bound rather than an operator preference.
The fixed horizon roles, candidate ladder, feature order, model overrides, CV
geometry, bootstrap, truth table, and threshold comparisons above are not
operator choices for #69. In particular, the plan supplies no guessed Binance
fee/slippage value and no guessed EWMA half-life or TP/SL multiplier; freeze
construction fails until each is explicitly supplied with evidence.
