# Feature Manifest Contract (v1)

The feature manifest is the versioned, self-describing record of a bar/feature
dataset build. It is the **single source of truth for which columns are model
inputs**: LightGBM (and later JEPA) training selects features explicitly from
`feature_cols`, never by inferring "all non-reserved columns" (AGENTS.md coding
standard). Code: `eval/manifest.py`. Tests: `tests/test_manifest.py`.

## Why

- **Leakage safety.** Targets (`y_fwd_bps`, `label`) and timing/cost columns are
  reserved and can never be selected as features; columns with label-derived
  names (`fwd`/`future`/`forward`/`barrier`/`label`/`target`/`outcome`
  substrings, or a bare `y`/`y_*` name) are rejected as features even if someone
  lists them. Declared timing metadata (`max_lookback_ns`,
  `availability_lag_ns`, `as_of_ns`, per-tag horizons) is checked against the
  actual frame's timing columns. Scope boundary: the manifest validates
  *declared* timing and screens names — verifying that feature **values** were
  really computed without look-ahead is the recon/bars job (replay-equivalence
  gate E0.1, `tests/test_reconstruct_no_lookahead.py`), not the manifest's.
- **Reproducibility.** A manifest pins the dataset/build IDs, bar clock, venues,
  horizons, and source artifacts that produced the matrix, plus a generation
  timestamp. Ordinary builds are reconstructable from the manifest and its
  referenced artifacts. A consumed G0-BN holdout is audit-reproducible but not
  re-executable: its manifest also reconciles the frozen plan, stable
  transaction, two claims, and materialization attestation.

## Schema (v1)

Required fields:

| Field | Meaning |
| --- | --- |
| `manifest_version` | Schema version; this code supports `1`. |
| `dataset_id` / `build_id` | Logical dataset and specific build (e.g. content hash or date-tag). |
| `bar_clock` | Dict with `kind` (e.g. `dollar`) plus clock params (threshold, time cap). |
| `time` | Exactly `{"unit": "ns", "timezone": "UTC"}` — int64 nanoseconds, UTC. |
| `feature_cols` | Explicit ordered model-input columns. |
| `target_cols` | Label columns (required; subset of `reserved_cols`). Core non-label reserved columns (timing/cost/weight/tag) are rejected as targets; of the core registry only `y_fwd_bps` and `label` qualify. |
| `reserved_cols` | Non-feature registry; must include `eval.matrix.RESERVED` in full. |
| `venues` | `{exchange, symbol[, role: signal|target]}` entries represented in the data. |
| `horizons` | Tag → physical duration in ns (e.g. `{"10s": 10000000000}`); matches the per-row `horizon` tag. |
| `sources` | Source artifacts/versions (e.g. `crypto-lake/book_delta_v2`). |
| `generated_at` | ISO-8601 timestamp with explicit timezone. |
| `max_lookback_ns` / `embargo_ns` | Longest retained feature look-back/observation delay and CV embargo. For G0-BN, `t1=t_barrier` already carries actual label span; embargo starts after `t_barrier` and `max_lookback_ns = max_retained(t_event-t_feature_start)` after over-cap rows are dropped. Thus `embargo_ns = max_lookback_ns`; never add the nominal horizon again. Schema validation remains `embargo_ns >= max_lookback_ns`. |

## Staged Dataset Modes

The 2026-07-11 Binance-first amendment does **not** change manifest version 1 or
add an inferred mode field. Dataset capability is explicit in `dataset_id`,
`venues`, `sources`, and especially `feature_cols`:

- **`binance_single_venue` (`G0-BN`):** one `venues` entry for
  `BINANCE_FUTURES/BTC-USDT-PERP` (role may be omitted because the same
  instrument supplies features and targets); only #64-certified, #68-sealed,
  allowlisted Binance Futures L2 snapshot/delta and trade sources;
  Binance-specific labels and costs; no Coinbase/CoinAPI, spot, other
  assets/perpetuals, funding/OI/liquidations/basis, extra state feeds, or
  source-derived feature columns. Validate this exact template before parquet
  access wherever the API controls loading.
- **Conditional increment manifests:** add spot, derivatives state, Coinbase,
  or another asset only through a new manifest/build. Never mutate the
  `binance_single_venue` feature list or zero-fill an unavailable source.
- **Matched ablations:** when an increment is tested, the base and augmented
  manifests must share target rows, reserved columns, labels, costs, horizons,
  splits, and source-independent row IDs. Only their explicit feature lists may
  differ.

`G0-BN` development and January OOS are physically/logically separate builds.
The development manifest's complete guarded support ends before
`2026-01-01T00:00:00Z`. A #68 custodian identity/permission boundary distinct
from the developer/experiment operator may already own and seal the exact
January raw and normalized source objects; operator-run `chmod` is insufficient.
The OOS matrix and manifest **do not exist** before the outcome-blind holdout
plan, complete freeze, data-free preflight/refit, and raw-access burn. #69's
sole blind materializer creates them once, closes them, and attests their actual
manifest/logical-row/matrix/build/count/schedule hashes. Only then does the
separate matrix-access burn occur, before the sole scorer first reopens the
derived matrix/parquet/footer to validate and score. Any failure after either
burn is terminal INCONCLUSIVE.

### G0-BN protocol bindings

Manifest v1 needs no new top-level key: its structured `sources` entries carry
the bindings, and the
[`G0-BN binding spec`](superpowers/specs/2026-07-13-g0bn-protocol.md)
validates their exact fields. Every G0-BN manifest contains exactly one source
dict whose `name` is each of:

- `partition_contract`: `schema=g0bn-partition-plan-v1`, its hash, and
  `partition` equal to `development` or `holdout`;
- `g0bn_protocol`: `protocol=g0bn-v1`, protocol-config hash, source
  certification hash, instrument identity, and horizon-role hash.

The blind-materialized holdout manifest additionally contains exactly one source
dict with `name=g0bn_holdout_plan`, `protocol=g0bn-one-shot-v1`,
`consumption_schema=g0bn-consumption-v1`, the stable
`g0bn-holdout-universe-v1` universe ID, transaction ID, holdout-plan hash, and
freeze hash. The universe ID depends only on `g0bn-v1`, the exact instrument,
and fixed January/February bounds; pilot/config/freeze/source/plan/result values
cannot mint another transaction. Its dataset ID is exactly
`binance_single_venue_g0bn_oos`. Development uses
`binance_single_venue_g0bn_dev` and must not carry a holdout-plan binding.

The G0-BN freeze pins the development manifest/content and outcome-blind
`holdout_plan_sha256`; that hash enters the future OOS build parameters and
breaks the build/freeze identity cycle. The freeze contains no fictional
January `build_id`, manifest/matrix/logical-row hash, row/drop count, realized
adaptive schedule/state, or result. Those values are first derived and attested
by blind materialization. Removing or renaming a binding changes the
manifest/config hashes and cannot turn a holdout build into a generic one.

Every G0-BN development and holdout manifest also lists
`latency_drift_bps` in `extra_cols`. Its `dtypes` map must contain
`cost_bps: float64`, `half_spread_bps: float64`, and
`latency_drift_bps: float64` for those cost fields, and the matrix stores each
as Parquet/Arrow binary64 (`double`) without a float32 downcast. The actual
frame dtypes must match before write and again after read. `latency_drift_bps`
is finite and non-negative and is a required non-feature diagnostic, not a
model input or target. T7's reserved `cost_bps` remains the realized non-spread
cost
`2*taker_fee_bps + base_slippage_bps + latency_drift_bps`. The dedicated G0-BN
scorer derives the decision cost as `cost_bps - latency_drift_bps`, reconciles
it to the frozen `2*taker_fee_bps + base_slippage_bps` under the binding
binary64 tolerance, and uses only that decision cost plus the observable spread
and frozen margin for the trade mask.
It charges the full realized `cost_bps` to net PnL. Missing, non-finite,
negative, or inconsistent drift diagnostics fail the one-shot transaction
INCONCLUSIVE; the legacy generic evaluator contract is unchanged.

Optional: `extra_cols` (explicitly allowed diagnostics columns), `dtypes`
(column → dtype expectations), `availability_lag_ns` (allowed `t_available -
t_event`, default 0 = synchronous), `as_of_ns` (snapshot bound on
`t_available`), `gate` (study gate block, validated by
`eval.runner.resolve_gate`). Unknown top-level keys are rejected (typo
protection), same as the gate block.

## How training consumes it

```python
from eval.manifest import load_manifest, validate_frame, feature_list

man = load_manifest("data/processed/feature_manifest.json")  # schema-validated
validate_frame(matrix, man)          # columns/timing/leakage vs the actual frame
X = matrix[feature_list(man)]        # exactly manifest features, in order
```

`eval.runner.run_from_manifest` applies `validate_frame` automatically whenever
the manifest carries `manifest_version`, selects features via `feature_list`
(manifest order), and adds three baseline-specific fail-closed checks: declared
`target_cols` must be exactly `{y_fwd_bps, label}` (what `evaluate_config`
trains on), `availability_lag_ns` must be 0 (the baseline is synchronous —
lag features upstream), and every declared horizon must be present in the
frame. The result dict echoes `{dataset_id, build_id, generated_at,
embargo_ns, max_lookback_ns, feature_cols}` under `"manifest"` so a run is
reproducible from its own output. The `scripts/run_baseline.py` CLI accepts
only v1 manifests (via `load_manifest`), and `run_from_manifest` refuses
non-versioned dicts with a migration error (including a full v1 manifest whose
`manifest_version` key was lost to a typo).
Note `gate` is optional at schema level but required by `run_from_manifest`
and the CLI (`eval.runner.resolve_gate`); JEPA pretraining manifests may omit
it and will consume the same manifest with
`validate_frame(df, man, require_targets=False)` (label columns may be absent
for unsupervised pretraining; everything else still applies).

**G0-BN holdout exception:** generic baseline APIs/CLIs are not transaction
entry points. #67 adds a manifest-only preflight before any parquet loader. It
rejects a `partition_contract.partition == holdout`, a
`g0bn_holdout_plan` binding, `dataset_id == binance_single_venue_g0bn_oos`, or
an ambiguous/missing partition binding on a G0-BN manifest. The in-memory
`run_from_manifest(matrix, manifest)` rejects the same manifests immediately;
callers are not authorized to preload one. There is no override flag. The
dedicated `g0bn-one-shot-v1` scorer alone may call lower-level scoring
primitives, and only after its distinct matrix-access burn. The blind
materializer is separately gated by the raw-access burn and does not call the
scorer or reopen its output.

## Validation behavior (fail closed)

- Missing required fields, unknown keys, unsupported versions fail.
- Duplicate columns fail — within each list, and duplicated column labels in
  the frame itself. Overlaps fail for feature/target, feature/reserved, and
  extra vs feature-or-reserved (targets are a subset of reserved by design).
- Feature/target and feature/reserved overlap fails; leaky-named features fail.
- Manifest features or (when required) targets missing from the frame fail.
  Timing/horizon columns are required even when targets are not.
- Frame columns not declared as feature/reserved/extra fail — diagnostics must
  be opted in via `extra_cols`, never silently carried.
- Timing columns must be non-null integer nanoseconds (nullable-NA and
  datetime64 frames fail closed); the frame must be non-empty; horizon tags
  must be strings. Then: `t_feature_start <= t_event`, observed look-back
  within `max_lookback_ns`, `0 <= t_available - t_event <=
  availability_lag_ns`, `t_available <= as_of_ns`, `t_event <= t_barrier <=
  t_event + horizon_ns` for the row's declared horizon tag; undeclared horizon
  tags fail.
- Declared `dtypes` mismatches fail.

`unsafe_infer_feature_cols` is the one **deliberately named** inference escape
hatch (exploration only): it returns non-reserved columns in frame order and
refuses any leaky-named candidate; columns you name in its `extra_cols`
argument are excluded from the result without leak-screening. Its output is for
writing into a manifest to pre-register — never for feeding a model directly.
