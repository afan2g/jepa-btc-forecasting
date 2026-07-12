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
  timestamp — a training run is reconstructable from the manifest alone.

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
| `max_lookback_ns` / `embargo_ns` | Longest feature look-back and CV embargo; `embargo_ns >= max_lookback_ns`. |

## Staged Dataset Modes

The 2026-07-11 Binance-first amendment does **not** change manifest version 1 or
add an inferred mode field. Dataset capability is explicit in `dataset_id`,
`venues`, `sources`, and especially `feature_cols`:

- **`binance_single_venue` (`G0-BN`):** one `venues` entry for
  `BINANCE_FUTURES/BTC-USDT-PERP` (role may be omitted because the same
  instrument supplies features and targets); only certified Binance L2/trade
  sources; Binance-specific labels and costs; no Coinbase, spot, auxiliary
  derivatives, or other-asset feature columns.
- **Conditional increment manifests:** add spot, derivatives state, Coinbase,
  or another asset only through a new manifest/build. Never mutate the
  `binance_single_venue` feature list or zero-fill an unavailable source.
- **Matched ablations:** when an increment is tested, the base and augmented
  manifests must share target rows, reserved columns, labels, costs, horizons,
  splits, and source-independent row IDs. Only their explicit feature lists may
  differ.

`G0-BN` development and January OOS are physically/logically separate builds.
The development manifest's complete guarded support ends before
`2026-01-01T00:00:00Z`; the sealed OOS build is not opened before candidate,
cost, threshold, and trial-ledger freeze.

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
