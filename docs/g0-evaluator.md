# G0 Evaluator (issue #52)

Executable evaluation path for the staged G0 screens
(`docs/superpowers/plans/2026-07-10-staged-signal-acquisition.md` §2–§3). It is the
consumer of issue #37's producer partitions; everything here runs on the producer's
hash-pinned contracts and synthetic fixtures — no vendor I/O anywhere in this layer.
The per-manifest formal G1 path (`eval/runner.py`, `scripts/run_baseline.py`) is
unchanged and remains the project-defining gate.

## Modules

| Module | Role |
| --- | --- |
| `eval/hashing.py` | One canonical hash encoding for every pinned object: JSON content hashes, matched-row content hashes (logical rows, not file bytes), CPCV split hashes. |
| `eval/partition.py` | Partition-contract schema + hash; manifest↔contract binding via a `sources` entry (`{"name": "partition_contract", "sha256", "partition", "boundary_drop_counts"}` — v1 manifests already allow extra source keys, so G1 needed no migration); fail-closed span validation. The conservative prefilter `t_event + horizons[horizon] + guard_ns < holdout_start_ns` is checked independently of `t_barrier`, so an early-resolving barrier cannot bypass it; the symmetric rule at `holdout_end_ns` keeps April labels out of May. |
| `eval/ledger.py` | Deterministic trial ledger. Trial identity = (protocol, arm, dataset/build, **ordered** feature list, model config, horizon, variant + params). Identical re-runs are idempotent; the same identity with a different result fails closed. Order-independent ledger hash. |
| `eval/g0.py` | `run_g0cb_study` (development-only; no holdout parameter exists; holdout-bound manifests rejected before data; every venue must declare `role: target` explicitly — omitted roles fail closed) and `run_g0xv_development` (ONE unified study over matched arms; reserved-row + split hashes must be identical across arms; DSR uses the complete effective trial count including imported G0-CB history; PBO runs over the common cross-arm candidate-PnL matrix — keeping exactly one naive benchmark column, since matched arms share bit-identical naive PnL — and fails closed when unavailable; the combined arm must beat the matched Coinbase-only control beyond a preregistered block-bootstrap noise band whose knobs are validated and whose degenerate small-sample case fails closed). Both paths run `validate_matrix` value-domain checks (non-negative costs, label domain, uniqueness range, finite inputs) before any fit. |
| `eval/freeze.py` | Hash-pinned selection artifact built strictly from development evidence: winner/config, resolved gate rules, frozen trade-validation thresholds, exact holdout scope (explicit day list — ranges/globs/months rejected — that together with explicitly **reasoned exclusions** must cover the FULL contract holdout window: the holdout cannot be silently shortened), the contract's holdout window, source pins (contract, arm manifests, matched row/split hashes, **per-arm full-content matrix hashes** covering feature values), trial history (counts derived from the pinned ledger). Only a passing G0-XV study authorizes a freeze, and the winner plus every horizon verdict (PBO/noise-band values included) must reconcile against ledger-pinned evidence — an edited dev-result JSON cannot promote a failed-closed study. |
| `eval/consumption.py` | One-time holdout transaction: `frozen → validated | validation_failed → scored`. The record file name is **derived from the holdout identity** (window bounds + dataset), so a regenerated freeze over the same holdout maps to the same transaction — a consumed or failed holdout can never be retried under a fresh record. Stale/regenerated artifacts, retries, partial-scope substitutions, non-boolean verdicts, and generic day selectors are rejected. |
| `eval/holdout.py` | `fit_frozen_config` (pre-holdout matched rows only, shared model definitions with `eval/baseline.py:fit_model`) + `score_fixed_holdout` (requires verified freeze + `validated` record; re-verifies contract/build pins, the reserved row-universe hash **and** the frozen arm's full feature-content hash — post-freeze feature substitution fails; exact-day scope is enforced on the whole build **and** the winner-horizon slice actually scored; duplicate holdout rows rejected; scores once, marks consumed, and records the consumed holdout's content hash for audit. `verify_only` reproduces an already-consumed score and returns **no metrics unless the recorded hash reproduces** — it is not an iterate-against-holdout oracle). |
| `scripts/run_g0.py` | Orchestrator CLI: `g0cb`, `g0xv-dev`, `freeze`, `holdout-open`, `holdout-validate`, `holdout-score`. G0-CB has no holdout arguments and fails on a holdout-bound manifest **before** the matrix file is opened; `g0xv-dev` requires the persisted G0-CB history (`--prior-ledger`, explicit `--no-prior-history` opt-out); trial ledgers persist attempted trials even when a run aborts; `holdout-validate` requires a strict JSON boolean verdict; holdout scoring refuses before any holdout read unless the transaction is in the right state and pre-flights the output path before the irreversible consumption step. |
| `eval/synthetic.py` (`make_g0_world`, …) | Synthetic staged-pilot fixtures implementing the producer contracts: span-safe dev/holdout partitions with real per-horizon boundary-drop counts, and matched arm builds (identical reserved rows, differing `feature_cols`). |

## Protocol flow

```
run_g0cb_study (dev only, every attempt → ledger)          # G0-CB, issue #47
   └─ G0-CB ledger ─┐
run_g0xv_development (matched arms, unified DSR/PBO) ──────# G0-XV dev, issue #48
   └─ build_freeze_artifact (pre-holdout, hash-pinned)
        └─ open_transaction (one-time)
             └─ record_trade_validation (#48, exact scope, once)
                  ├─ FAIL → blocking/inconclusive, scoring refused forever
                  └─ PASS → score_fixed_holdout (fit pre-April, score April once)
```

Outputs are tagged `g0cb-development` / `g0xv-development` / `g0xv-holdout` and carry
`development_only` / `g1_claim: false` markers; holdout metrics are terminal evidence
and cannot re-enter selection (the winner is pinned by the freeze hash the transaction
verifies).

## Tests

`tests/test_g0_partition.py`, `test_g0_ledger.py`, `test_g0cb.py`, `test_g0xv.py`,
`test_g0_freeze.py`, `test_g0_holdout.py`, `test_g0_cli.py` — synthetic PASS/FAIL and
leaky controls, support-boundary exactness, matched-arm fail-closed checks, stale-hash
and one-time-consumption rejection, DSR-ledger and PBO-common-matrix reconciliation,
and the arm-wise-false-pass-caught-by-unified-ledger control. April is never touched:
holdout tests run on synthetic fixtures only.

## Residual risks (documented, not silently accepted)

- **Within-day holdout row substitution.** The holdout build cannot be content-pinned at
  freeze time (it must not have been readable), so between trade validation and scoring
  it is bound by dataset/build identity, the span rules, exact day sets (whole build and
  winner slice), and the duplicate-row check — not by a pre-pinned content hash. The
  producer's `build_id` is a content hash over logical rows (bar/label plan §H/§I), which
  is what gives the string equality teeth; the scorer additionally records the consumed
  holdout's content hash in the transaction for after-the-fact audit. Full closure needs
  #48 to record the validated build's content identity at validation time.
- **Deterministic re-registration across environments.** Ledger idempotency assumes a
  re-run of the same trial identity reproduces the same result (LightGBM/sklearn pinned
  seeds on one machine). A library upgrade between runs fails closed with a
  different-result conflict rather than silently forking trial history.
