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
| `target_cols` | Label columns (required; subset of `reserved_cols`, e.g. `y_fwd_bps`, `label`). |
| `reserved_cols` | Non-feature registry; must include `eval.matrix.RESERVED` in full. |
| `venues` | `{exchange, symbol[, role: signal|target]}` entries represented in the data. |
| `horizons` | Tag → physical duration in ns (e.g. `{"10s": 10000000000}`); matches the per-row `horizon` tag. |
| `sources` | Source artifacts/versions (e.g. `crypto-lake/book_delta_v2`). |
| `generated_at` | ISO-8601 timestamp with explicit timezone. |
| `max_lookback_ns` / `embargo_ns` | Longest feature look-back and CV embargo; `embargo_ns >= max_lookback_ns`. |

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
the manifest carries `manifest_version`; legacy `{feature_cols, embargo_ns,
max_lookback_ns, gate}` dicts still work unchanged, and `validate_matrix` /
`run_study` keep enforcing the baseline invariants either way. A manifest that
carries v1-only fields but lost `manifest_version` (typo) is refused rather
than silently treated as legacy. Future JEPA
training should consume the same manifest with
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
