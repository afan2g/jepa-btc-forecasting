# LightGBM Feature-Manifest Integration — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the LightGBM baseline consume the v1 feature-manifest contract (PR #14, `eval/manifest.py` + `docs/feature-manifest.md`) end-to-end — features selected only from `feature_cols`, targets/horizons/availability cross-checked against what the baseline actually trains on, and the unvalidated legacy manifest path first screened, then removed.

**Architecture:** The call chain `scripts/run_baseline.py` → `eval.runner.run_from_manifest` → `eval.study.run_study` → `eval.baseline.evaluate_config` keeps its bare-`feature_cols` plumbing below the runner boundary (21 existing test call sites depend on those signatures). All manifest awareness concentrates in the runner and CLI: the CLI becomes v1-only via `load_manifest`, the runner's v1 branch consumes `feature_list`/`target_list` and adds baseline-specific fail-closed checks, and the legacy dict branch gets a leak screen in phase 1 and is deleted in phase 3. `validate_matrix` and `evaluate_config` gain cheap defense-in-depth (duplicate/numeric/NaN screens, X-width assert) that protects direct `run_study` callers too.

**Tech Stack:** Python 3.12, pandas/numpy, lightgbm + scikit-learn, pytest. No new dependencies.

**Scope:** Consumption side only. Producing the real `data/processed/feature_manifest.json` belongs to the bars (E0.3) / labels (E0.4) build jobs and is out of scope here (see Non-goals).

---

## Current state (verified against the repo at e71656b)

Where feature columns are selected today — the full chain:

| Site | What it does | Status |
|---|---|---|
| `eval/baseline.py:61` | `X = matrix[feature_cols].to_numpy(float)` — the **only** place a model input matrix is materialized. Ridge/LGBMRegressor/LGBMClassifier all consume this X. | Explicit list; no inference. |
| `eval/study.py:16,20,34` | `run_study(matrix, feature_cols, ...)` calls `validate_matrix(matrix, feature_cols)` then forwards the bare list to `evaluate_config` per config. | Explicit list; no inference. |
| `eval/runner.py:23-44` | `run_from_manifest(matrix, manifest)`: dual path. `manifest_version` present → `validate_frame(matrix, manifest)` up front. Legacy `{feature_cols, embargo_ns, max_lookback_ns, gate}` dicts → **no manifest or frame validation at the runner boundary** (only the `V1_ONLY_FIELDS` typo guard; `validate_matrix` still runs later inside `run_study`). Then `feats = manifest["feature_cols"]` by raw dict access (`eval/runner.py:36`), groupby horizon → `run_study`. | Explicit list; legacy path unvalidated. |
| `scripts/run_baseline.py:4-6,18` | CLI docstring documents **only the legacy shape**; loads with raw `json.load`, not `load_manifest`. | Steers users onto the unvalidated path. |
| `eval/manifest.py:298` | `unsafe_infer_feature_cols` — the **only** all-non-reserved-columns inference site in the repo. Exploration-only escape hatch; zero production callers. | Sanctioned by AGENTS.md only if a plan says so — see Non-goals. |

No production or script code path infers features from the frame. The AGENTS.md standard ("Use explicit manifests/contracts for modeling data. Do not infer feature columns from 'all non-reserved columns' unless a plan explicitly says so") is already respected structurally — what is missing is that the documented v1 consumption pattern (`docs/feature-manifest.md:51-59`: `load_manifest` → `validate_frame` → `matrix[feature_list(man)]`) has **zero production call sites**: `feature_list`, `target_list`, and `load_manifest` are used only by `tests/test_manifest.py`.

Gaps this plan closes (severity → phase):

1. **Leakage-risk** — the legacy branch of `run_from_manifest` skips `validate_frame`, and `validate_matrix` has no leaky-name screen. A legacy manifest listing `ob_imbalance_fwd_5s` or `y_ret_next` trains LightGBM on a label-derived column with no error. → Phase 1 (screen), Phase 3 (branch removed).
2. **Correctness** — `target_cols` is declared and validated but never consumed: `eval/baseline.py:62-63` hardcodes `y_fwd_bps`/`label`. A manifest declaring `target_cols=["y_fwd_bps"]` validates, yet `lgbm_clf` still trains on `label` — the manifest misdescribes the study. → Phase 1.
3. **Correctness** — duplicate-column hazard: `matrix[feature_cols]` returns *every* label match, so a duplicated frame label silently widens X; duplicated `feature_cols` entries double-weight a column. v1 catches both; the legacy path and direct `run_study` callers catch neither. → Phase 1.
4. **Correctness/UX** — `availability_lag_ns > 0` is schema-legal but collides with `validate_matrix`'s hard `t_available == t_event` requirement (`eval/matrix.py:28-30`), producing a confusing error deep inside `run_study` per horizon. → Phase 1 (reject up front for the baseline).
5. **Correctness (edge)** — manifest-declared horizons absent from the frame produce silently missing entries in the result dict (`eval/runner.py:41-43`); legacy `str()`-coerces non-string horizon tags that v1 rejects. → Phase 1 (coverage check), Phase 3 (coercion path removed).
6. **Correctness (crash asymmetry)** — feature columns have no numeric/NaN screening: `to_numpy(float)` dies opaquely on object dtypes, and NaN features crash Ridge mid-study while LightGBM would mask them. → Phase 1.
7. **Hygiene** — runner reads `manifest["feature_cols"]` raw instead of `feature_list()`; run output omits manifest identity despite the module docstring claiming runs are "reproducible from its own output"; CLI docstring/loader stale; LightGBM rungs have no pinned `random_state`. → Phases 1–2.

Not a gap (verified): feature **order** is preserved end-to-end — manifest order → pandas label selection → numpy column order → LightGBM (`feature_list` returns the list in order; `matrix[feature_cols]` selects by label in list order regardless of frame column order). Phase 1 pins this with a test. Embargo flow is also not a gap: `embargo_ns >= max_lookback_ns` is enforced at schema level (`eval/manifest.py:174-176`) and re-checked at runtime against the observed per-row look-back (`eval/study.py:28-32`).

---

## Target integration API (end state, after phase 3)

```python
# CLI (scripts/run_baseline.py) — the only real-data training entry point:
from eval.manifest import load_manifest
from eval.runner import resolve_gate, run_from_manifest

man = load_manifest(manifest_path)   # v1 REQUIRED; schema errors fail here,
resolve_gate(man)                    # ...and gate errors, before the parquet read
matrix = pd.read_parquet(matrix_path)
res = run_from_manifest(matrix, man)
```

Inside `run_from_manifest` (v1-only after phase 3):

- **Loading:** callers pass the dict; the CLI gets it from `load_manifest` (json + `validate_manifest`).
- **Frame check:** `validate_frame(matrix, manifest)` — columns, timing, leakage, horizons, dtypes.
- **Feature selection:** `feats = feature_list(manifest)` — a validated copy, **in manifest order**; downstream `matrix[feats]` preserves that order into numpy/LightGBM regardless of frame column order.
- **Target validation:** `set(target_list(manifest)) == {"y_fwd_bps", "label"}` — `evaluate_config` trains on exactly these (regression on `y_fwd_bps`, classification on `label`); anything else misdescribes the study → fail closed.
- **Horizon validation:** frame tags ⊆ declared (in `validate_frame`) **and** declared ⊆ frame tags (new runner check) — together, equality. Per-row `t_event <= t_barrier <= t_event + horizon_ns` per declared tag (in `validate_frame`).
- **Availability:** `availability_lag_ns != 0` rejected up front — the baseline is synchronous decide-and-act (`validate_matrix` enforces `t_available == t_event`); lag is modeled upstream by lagging features. `as_of_ns` enforced by `validate_frame`.
- **Reserved columns:** all of `eval.matrix.RESERVED` must be present in the frame (`validate_frame` + `validate_matrix`); features may never overlap them (both validators).
- **Unknown columns:** frame columns not declared feature/reserved/extra fail closed (`validate_frame`); diagnostics are opted in via `extra_cols` and are never selectable into X (selection is by explicit list only).
- **Reproducibility echo:** the result dict gains `res["manifest"]` = `{dataset_id, build_id, generated_at, embargo_ns, max_lookback_ns, feature_cols}` so a run really is reconstructable from its own output.

Signatures **unchanged**: `run_study(matrix, feature_cols, *, cost_default, n_groups, k, embargo_ns, max_lookback_ns, ...)` and `evaluate_config(matrix, feature_cols, model, *, n_groups, k, embargo_ns)` keep taking bare lists — `tests/test_study.py` (9 tests), `tests/test_gate_synthetic.py` (2), `tests/test_matrix.py` (8), `tests/test_baseline.py` (6) all pass feature lists directly and must keep working without manifests. Gate ownership also unchanged: `validate_manifest` only checks `gate` is a dict; `resolve_gate` owns keys/defaults (pinned by `test_gate_block_is_allowed_passthrough` and the three `resolve_gate` tests).

---

## Leakage protections (rule → enforcement site)

| Rule | v1 path | Legacy path (until phase 3) |
|---|---|---|
| Targets can never be features | `validate_manifest` feature/target + feature/reserved overlap (targets ⊆ reserved by design) | `validate_matrix` feature/RESERVED overlap (`eval/matrix.py:16-18`) |
| Label-derived names (`fwd`/`future`/`forward`/`barrier`/`label`/`target`/`outcome`, bare `y`/`y_*`) rejected as features | `validate_manifest` `_leaky_names` screen | **NEW (phase 1):** `leaky_feature_names()` screen in the runner's legacy branch |
| `t_feature_start <= t_event` | `validate_frame` | `validate_matrix` |
| Observed look-back ≤ declared `max_lookback_ns` | `validate_frame` + `run_study` runtime cross-check | `run_study` runtime cross-check (`eval/study.py:28-32`) |
| `embargo_ns >= max_lookback_ns` | `validate_manifest` + `run_study` | `run_study` |
| Synchronous availability (`t_available == t_event`) | `validate_matrix` + **NEW (phase 1):** `availability_lag_ns != 0` rejected up front in the runner | `validate_matrix` (`eval/matrix.py:28-30`) |
| `t_available <= as_of_ns` snapshot bound | `validate_frame` | not checked (legacy declares no `as_of_ns`; `V1_ONLY_FIELDS` guard refuses it) |
| `t_barrier` within declared horizon; undeclared frame tags fail | `validate_frame` | not checked (removed with the branch in phase 3) |
| Declared horizons must exist in the frame | **NEW (phase 1):** runner coverage check | n/a |
| Duplicate labels cannot widen X / smuggle a shadowed series past the screens | `validate_frame` + **NEW (phase 1):** `validate_matrix` dup checks + `evaluate_config` X-width assert | **NEW (phase 1):** same `validate_matrix`/`evaluate_config` checks (they run on both paths) |

**Nullable-dtype caveat (closed by Task 4):** `validate_matrix`'s plain timing comparisons and `run_study`'s observed-lookback `.max()` fail **open** under pandas nullable dtypes with `pd.NA` (`Series.all()` skips NA — the very hazard `validate_frame`'s comment documents), so the legacy-column cells for the three timing rows above are only unconditionally true once Task 4 adds the same integer/non-null timing guard `validate_frame` already has. Task 4 also screens `cost_bps`/`half_spread_bps`/`uniqueness` for NaN/`pd.NA` — otherwise validated on **neither** path in any phase (a NaN cost silently forces no-trade and biases turnover/PnL; an NA weight poisons the weighted Sharpe).

**Embargo vs label horizon (experiment-plan.md E0.4: "embargo >= max(label horizon, longest feature look-back)"):** deliberately NOT a new check. The label-horizon side is covered *structurally* by CPCV's per-test-span purge — `data/cv.py` merges the test rows' actual `[t0, t1] = [t_event, t_barrier]` spans and purges any train row overlapping them, so no train label span can straddle a test span regardless of embargo. `embargo_ns` guards the *feature look-back* side after the test block, and `embargo_ns >= max_lookback_ns` is enforced twice (schema + runtime). Adding `embargo_ns >= horizon_ns` would be redundant with the purge; if a future reviewer wants belt-and-braces, it belongs in `validate_manifest`, not the runner. (The reverse direction — test-row features reaching back toward earlier train labels — is deployment-realistic, not leakage: any train span overlapping a test span is purged, so surviving pre-test train labels are fully realized before the earliest test decision time in that interval.)

**Scope boundary (do not over-claim):** the manifest validates *declared* timing and screens *names*. Verifying that feature **values** were computed without look-ahead is producer-side work: the replay-equivalence gate (E0.1, `tests/test_reconstruct_no_lookahead.py`) covers order-book *reconstruction* today, and the equivalent guard for bar/feature *computation* must land with the E0.3 feature producer, which does not exist yet. Nothing in this plan changes or claims to close that — see Non-goals.

---

## Backward compatibility

- **Before real manifests exist** (`data/processed/` is absent in this worktree): everything below the runner boundary keeps bare-list signatures, so `test_study.py`, `test_gate_synthetic.py`, `test_matrix.py`, `test_baseline.py` run unchanged, no manifests needed.
- **Synthetic manifests ARE required** for runner-boundary tests. Phase 1 adds `eval.synthetic.make_manifest(feature_cols, max_lookback_ns, *, gate=None, **over)` — a schema-valid v1 manifest mirroring `make_matrix`'s defaults (`"10s"` horizon = 10^10 ns, embargo = look-back). `tests/test_manifest.py`'s hand-rolled `_manifest()` helper stays as-is (it tests the contract itself; hand-rolling is a feature there).
- **Exactly one green test breaks** if v1 became required at the runner today: `tests/test_runner.py::test_run_from_manifest_runs_and_echoes_resolved_gate` (the only full legacy-shape builder). Phase 1 migrates it to v1 and adds an explicit legacy-pin test; phase 3 deletes the pin.
- Two guard tests (`tests/test_manifest.py:311,321`) assert `ValueError` matching `"manifest_version"` for v1-fields-without-version dicts. They stay green through phase 3 **only if** the new rejection message keeps the literal `manifest_version` — phase 3's error message does.
- The skip-gated `tests/test_baseline_integration.py` currently uses raw `json.load` — a landmine the day a legacy-shaped real manifest appears. Phase 2 switches it to `load_manifest`, pinning that the real artifact must be v1.
- The three `resolve_gate` tests pass partial dicts (`{"feature_cols": []}`, gate-only) — `resolve_gate`'s contract (gate-block-only inspection) is untouched in every phase.

---

## File structure

**Phase 1** — Modify: `eval/manifest.py` (public leak-screen helper), `eval/synthetic.py` (`make_manifest`), `eval/runner.py` (v1 hardening + legacy screen), `eval/matrix.py` (dup + numeric/NaN screens), `eval/baseline.py` (X-width assert, `random_state`), `scripts/run_baseline.py` (v1-only CLI), `tests/test_manifest.py` (+1), `tests/test_matrix.py` (+5), `tests/test_baseline.py` (+2). Create: `tests/test_synthetic.py`. Rewrite: `tests/test_runner.py`.

**Phase 2** — Modify: `tests/test_baseline_integration.py`, `docs/feature-manifest.md`.

**Phase 3** — Modify: `eval/runner.py` (delete legacy branch), `tests/test_runner.py` (delete legacy pins, add required-version test), `tests/test_manifest.py` (comment updates), `docs/feature-manifest.md` (drop legacy mention), `eval/manifest.py` (delete `V1_ONLY_FIELDS` if unreferenced).

Nothing here touches `docs/data.md` (another branch owns it), `data/cv.py` (no manifest awareness by design), or `eval/study.py`/`eval/cost.py`/`eval/stats.py` (signatures pinned by tests).

---

# Phase 1 — manifest required in the new training path

## Task 1: Public leak-screen helper in `eval/manifest.py`

The runner's legacy branch (Task 3) needs the leaky-name screen without importing a private. Expose a one-line public wrapper.

**Files:**
- Modify: `eval/manifest.py` (after `_leaky_names`, ~line 56)
- Test: `tests/test_manifest.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_manifest.py` (and add `leaky_feature_names` to the existing `from eval.manifest import (...)` block at the top):

```python
def test_leaky_feature_names_public_helper():
    # Public wrapper for the runner's legacy-branch screen (and manifest-authoring tools).
    assert leaky_feature_names(["ofi_integrated", "mid_fwd_5s", "y", "spread_tick"]) == \
        ["mid_fwd_5s", "y"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_manifest.py -k public_helper -v`
Expected: **collection error** (not a FAILED test) with `ImportError: cannot import name 'leaky_feature_names'` — the top-of-module import takes the whole file down, so `-k` cannot deselect around it; nonzero exit is the signal.

- [ ] **Step 3: Implement**

Add to `eval/manifest.py`, directly after `_leaky_names`:

```python
def leaky_feature_names(cols) -> list[str]:
    """Public leak screen: the subset of cols with label-derived names (LEAKY_NAME_PATTERNS
    substrings, or a bare y/y_* name). Used by eval.runner's legacy branch and available to
    manifest-authoring tools."""
    return _leaky_names(cols)
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_manifest.py -k public_helper -v`
Expected: 1 passed

- [ ] **Step 5: Commit**

```bash
git add eval/manifest.py tests/test_manifest.py
git commit -m "feat: public leaky_feature_names leak-screen helper"
```

---

## Task 2: `eval.synthetic.make_manifest` — v1 manifests for tests

**Files:**
- Modify: `eval/synthetic.py`
- Create: `tests/test_synthetic.py`

- [ ] **Step 1: Write the failing test**

`tests/test_synthetic.py`:

```python
from eval.manifest import validate_frame, validate_manifest
from eval.synthetic import FEATURES, make_manifest, make_matrix


def test_make_manifest_is_schema_valid():
    validate_manifest(make_manifest(list(FEATURES), 10_000_000_000))


def test_make_manifest_matches_make_matrix_frame():
    df, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    validate_frame(df, make_manifest(feats, lb))  # helper mirrors the generator's contract
```

- [ ] **Step 2: Run to verify it fails**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_synthetic.py -v`
Expected: **collection error** with `ImportError: cannot import name 'make_manifest'`

- [ ] **Step 3: Implement**

Add to `eval/synthetic.py` — imports at the top (no cycle: `eval.manifest` imports only `eval.matrix`; nothing imports `eval.synthetic` from either):

```python
from eval.manifest import MANIFEST_VERSION
from eval.matrix import RESERVED
```

and the function after `make_matrix`:

```python
def make_manifest(feature_cols, max_lookback_ns, *, gate=None, **over):
    """A schema-valid v1 feature manifest matching make_matrix's defaults ("10s" horizon,
    embargo = look-back). Test/exploration helper — real builds write their own manifest.
    NOTE: horizons is coupled to make_matrix's default horizon_ns; if you pass make_matrix
    a different horizon_ns, override via **over (e.g. horizons={"2s": 2_000_000_000})."""
    man = {
        "manifest_version": MANIFEST_VERSION,
        "dataset_id": "synthetic",
        "build_id": "seeded",
        "bar_clock": {"kind": "synthetic"},
        "time": {"unit": "ns", "timezone": "UTC"},
        "feature_cols": list(feature_cols),
        "target_cols": ["y_fwd_bps", "label"],
        "reserved_cols": list(RESERVED),
        "venues": [{"exchange": "SYNTHETIC", "symbol": "BTC-TEST"}],
        "horizons": {"10s": 10_000_000_000},
        "sources": ["eval/synthetic.py"],
        "generated_at": "2026-07-02T00:00:00+00:00",
        "max_lookback_ns": int(max_lookback_ns),
        "embargo_ns": int(max_lookback_ns),
    }
    if gate is not None:
        man["gate"] = gate
    man.update(over)
    return man
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_synthetic.py -v`
Expected: 2 passed

- [ ] **Step 5: Commit**

```bash
git add eval/synthetic.py tests/test_synthetic.py
git commit -m "feat: make_manifest synthetic v1-manifest helper"
```

---

## Task 3: Runner v1 hardening + legacy leak screen

`run_from_manifest` consumes the contract via `feature_list`/`target_list`, adds three baseline-specific fail-closed checks, echoes manifest identity, and screens the legacy branch. `tests/test_runner.py` is rewritten in full (the legacy echo test migrates to v1; an explicit legacy pin remains until phase 3).

**Files:**
- Modify: `eval/runner.py` (full new content below)
- Rewrite: `tests/test_runner.py` (full new content below)

- [ ] **Step 1: Write the failing tests**

Replace `tests/test_runner.py` in full:

```python
import pytest
from eval.runner import resolve_gate, run_from_manifest, DEFAULT_GATE
from eval.synthetic import make_manifest, make_matrix

GATE = {"n_groups": 4, "k": 2, "min_trades": 1, "min_eff_trades": 1.0}


def _v1(feats, lb, **over):
    return make_manifest(feats, lb, gate=dict(GATE), **over)


# ---------- resolve_gate (contract unchanged) ----------

def test_resolve_gate_requires_block():
    with pytest.raises(ValueError, match="gate"):
        resolve_gate({"feature_cols": []})


def test_resolve_gate_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown gate"):
        resolve_gate({"gate": {"min_tradez": 5}})


def test_resolve_gate_fills_defaults():
    g = resolve_gate({"gate": {"k": 3}})
    assert g["k"] == 3 and g["min_trades"] == DEFAULT_GATE["min_trades"]


# ---------- v1 path ----------

def test_v1_manifest_runs_and_echoes_gate_and_identity():
    m, feats, lb = make_matrix(n=900, signal_strength=4.0, seed=8)
    res = run_from_manifest(m, _v1(feats, lb))
    assert res["gate"]["min_sample_sharpe"] == 0.0       # default filled into resolved config
    assert "10s" in res["horizons"] and "g1_pass" in res["horizons"]["10s"]
    assert res["manifest"]["dataset_id"] == "synthetic"  # reproducible from its own output
    assert res["manifest"]["feature_cols"] == feats


def test_v1_targets_must_match_baseline_consumption():
    # evaluate_config trains on exactly {y_fwd_bps, label}; a manifest declaring fewer
    # (or extra reserved targets) misdescribes what the study consumed.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="exactly"):
        run_from_manifest(m, _v1(feats, lb, target_cols=["y_fwd_bps"]))


def test_v1_availability_lag_rejected_for_synchronous_baseline():
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="synchronous"):
        run_from_manifest(m, _v1(feats, lb, availability_lag_ns=5_000_000))


def test_v1_declared_horizon_missing_from_matrix_rejected():
    # validate_frame checks frame tags are declared; the runner checks the converse so a
    # manifest declaring {10s, 60s} over a 10s-only build cannot silently return no 60s row.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    # NB: lb is already bound positionally to max_lookback_ns inside _v1 -> make_manifest;
    # passing max_lookback_ns=lb again via **over would TypeError at setup.
    man = _v1(feats, lb, horizons={"10s": 10_000_000_000, "60s": 60_000_000_000})
    with pytest.raises(ValueError, match="missing from the matrix"):
        run_from_manifest(m, man)


# ---------- legacy path (branch deleted in phase 3) ----------

def test_legacy_manifest_dict_still_runs():
    # Phase-3 removal target: delete this test with the legacy branch.
    m, feats, lb = make_matrix(signal_strength=4.0, seed=8)
    man = {"feature_cols": feats, "embargo_ns": lb, "max_lookback_ns": lb,
           "gate": {"n_groups": 6, "k": 2}}
    res = run_from_manifest(m, man)
    assert res["gate"]["min_sample_sharpe"] == 0.0
    assert "10s" in res["horizons"]


def test_legacy_manifest_rejects_leaky_feature_names():
    # The legacy branch skips validate_frame; the leak screen must never be skipped.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    m["mid_fwd_10s"] = 0.0
    man = {"feature_cols": feats + ["mid_fwd_10s"], "embargo_ns": lb,
           "max_lookback_ns": lb, "gate": {"n_groups": 4, "k": 2}}
    with pytest.raises(ValueError, match="leak"):
        run_from_manifest(m, man)


def test_legacy_manifest_missing_keys_fail_with_contract_error():
    # Raw KeyError is not a contract error; name the missing keys.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="legacy manifest missing"):
        run_from_manifest(m, {"feature_cols": feats, "gate": {"n_groups": 4, "k": 2}})
```

- [ ] **Step 2: Run to verify the new tests fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_runner.py -v`
Expected: the 3 `resolve_gate` tests and `test_legacy_manifest_dict_still_runs` PASS; the 4 v1 tests fail (`KeyError: 'manifest'` / no raise), `test_legacy_manifest_rejects_leaky_feature_names` and `test_legacy_manifest_missing_keys_fail_with_contract_error` fail (no raise / KeyError instead of ValueError).

- [ ] **Step 3: Implement**

Replace `eval/runner.py` in full:

```python
"""Manifest-driven G1 runner. The gate block is REQUIRED (pre-registration) and the
RESOLVED config is returned so every run is reproducible from its own output."""
from __future__ import annotations
import pandas as pd
from eval.manifest import (V1_ONLY_FIELDS, feature_list, leaky_feature_names,
                           target_list, validate_frame)
from eval.study import run_study

DEFAULT_GATE = {"n_groups": 6, "k": 2, "min_trades": 30, "min_eff_trades": 10.0,
                "min_sample_sharpe": 0.0, "dsr_thresh": 0.95, "pbo_thresh": 0.5}

# evaluate_config trains on exactly these (y_fwd_bps regression, label classification).
# A manifest declaring anything else would misdescribe what the study consumed.
BASELINE_TARGETS = frozenset(("y_fwd_bps", "label"))

_LEGACY_KEYS = ("feature_cols", "embargo_ns", "max_lookback_ns")


def resolve_gate(manifest: dict) -> dict:
    """Require the pre-registered 'gate' block; reject unknown (misspelled) keys; fill
    defaults; return the RESOLVED config."""
    if "gate" not in manifest:
        raise ValueError("manifest must include a pre-registered 'gate' block")
    unknown = set(manifest["gate"]) - set(DEFAULT_GATE)
    if unknown:
        raise ValueError(f"unknown gate keys (misspelled?): {sorted(unknown)}")
    return {**DEFAULT_GATE, **manifest["gate"]}


def run_from_manifest(matrix: pd.DataFrame, manifest: dict) -> dict:
    out = {}
    if "manifest_version" in manifest:
        # v1+ manifests are schema-validated and checked against the matrix up front.
        validate_frame(matrix, manifest)
        feats = feature_list(manifest)               # validated copy, manifest order
        targets = set(target_list(manifest))
        if targets != BASELINE_TARGETS:
            raise ValueError(f"the LightGBM baseline consumes exactly "
                             f"{sorted(BASELINE_TARGETS)} as targets; manifest declares "
                             f"{sorted(targets)}")
        if manifest.get("availability_lag_ns", 0) != 0:
            raise ValueError("the LightGBM baseline is synchronous (t_available == t_event); "
                             "availability_lag_ns > 0 is reserved for future consumers — "
                             "lag features upstream instead")
        # validate_frame checked frame tags are declared; check the converse so declared
        # horizons cannot silently vanish from the per-horizon results.
        missing_h = sorted(set(manifest["horizons"]) - set(matrix["horizon"].unique()))
        if missing_h:
            raise ValueError(f"manifest horizons missing from the matrix: {missing_h}; "
                             "the manifest must describe this exact build")
        out["manifest"] = {k: manifest[k] for k in
                           ("dataset_id", "build_id", "generated_at",
                            "embargo_ns", "max_lookback_ns")}
        out["manifest"]["feature_cols"] = feats
    else:
        markers = V1_ONLY_FIELDS & set(manifest)
        if markers:
            # a typo in 'manifest_version' itself must not silently select the
            # unvalidated legacy path
            raise ValueError(f"manifest carries v1 contract fields {sorted(markers)} but no "
                             "'manifest_version'; add manifest_version=1")
        missing = [k for k in _LEGACY_KEYS if k not in manifest]
        if missing:
            raise ValueError(f"legacy manifest missing required keys: {missing}")
        feats = list(manifest["feature_cols"])
        leaky = leaky_feature_names(feats)
        if leaky:
            # legacy dicts skip validate_frame; never skip the leak screen
            # (phase 3 removes this whole branch)
            raise ValueError(f"leaky feature names (label-derived): {leaky}")
    gate = resolve_gate(manifest)
    emb, lb = manifest["embargo_ns"], manifest["max_lookback_ns"]
    horizons = {}
    # observed=True: a categorical horizon column must not yield empty subframes for
    # unused categories (run_study crashes on an empty matrix under pandas 2.x defaults)
    for h, sub in matrix.groupby("horizon", observed=True):
        horizons[str(h)] = run_study(sub.reset_index(drop=True), feats, cost_default=None,
                                     embargo_ns=emb, max_lookback_ns=lb, **gate)
    out.update({"gate": gate, "horizons": horizons})
    return out
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_runner.py tests/test_manifest.py -v`
Expected: `tests/test_runner.py` 10 passed; every `tests/test_manifest.py` test still passes (in particular `test_run_from_manifest_accepts_valid_versioned_manifest` — its `_manifest` defaults declare exactly `["y_fwd_bps", "label"]` and one `"10s"` horizon — and `test_run_from_manifest_categorical_horizon_with_unused_categories`, since `.unique()` on a categorical returns observed values only, so the coverage check sees `{"10s"} - {"10s"} = ∅`).

- [ ] **Step 5: Commit**

```bash
git add eval/runner.py tests/test_runner.py
git commit -m "feat: runner consumes v1 manifest contract (targets/availability/horizon checks, identity echo, legacy leak screen)"
```

---

## Task 4: `validate_matrix` hardening — duplicates, numeric, NaN

Protects both manifest paths **and** direct `run_study` callers.

**Files:**
- Modify: `eval/matrix.py`
- Test: `tests/test_matrix.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_matrix.py` (add `import numpy as np` and `import pandas as pd` at the top if absent):

```python
def test_duplicate_frame_columns_rejected():
    # matrix[feature_cols] returns EVERY label match: a duplicated label silently widens X.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    dup = pd.concat([df, df[["cvd"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_matrix(dup, feats)


def test_duplicate_manifest_feature_entries_rejected():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_matrix(df, feats + [feats[0]])


def test_non_numeric_feature_rejected():
    # to_numpy(float) would die opaquely on object dtype; fail closed with the column name.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad["cvd"] = "high"
    with pytest.raises(ValueError, match="numeric"):
        validate_matrix(bad, feats)


def test_nan_feature_or_target_rejected():
    # NaN features crash Ridge mid-study while LightGBM would silently mask them;
    # NaN y_fwd_bps corrupts PnL; NA cost silently forces no-trade (band is NaN) and
    # NA uniqueness poisons the weighted Sharpe. Imputation belongs upstream.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad.loc[0, "cvd"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_matrix(bad, feats)
    bad2 = df.copy(); bad2.loc[0, "y_fwd_bps"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_matrix(bad2, feats)
    bad3 = df.copy()
    bad3["cost_bps"] = bad3["cost_bps"].astype("Float64"); bad3.loc[0, "cost_bps"] = pd.NA
    with pytest.raises(ValueError, match="NaN"):
        validate_matrix(bad3, feats)


def test_nullable_or_datetime_timing_fails_closed():
    # Plain comparisons fail open under pd.NA (Series.all() skips NA) and run_study's
    # observed-lookback .max() skips NA rows — mirror validate_frame's integer/non-null
    # timing guard so the legacy path and direct run_study callers are covered too.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy()
    bad["t_barrier"] = bad["t_barrier"].astype("Int64"); bad.loc[0, "t_barrier"] = pd.NA
    with pytest.raises(ValueError, match="null"):
        validate_matrix(bad, feats)
    bad2 = df.copy()
    bad2["t_event"] = pd.to_datetime(bad2["t_event"])  # datetime64, not int ns
    with pytest.raises(ValueError, match="integer"):
        validate_matrix(bad2, feats)
```

- [ ] **Step 2: Run to verify they fail**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_matrix.py -k "duplicate or numeric or nan or fails_closed" -v`
Expected: 5 FAILED (no raise, or a different error than the matched message). Note: plain `-k rejected` would over-match — `test_label_out_of_domain_rejected` and `test_negative_costs_rejected` already exist.

- [ ] **Step 3: Implement**

Replace `eval/matrix.py` in full (existing checks unchanged; new: duplicate screens at the top, timing dtype/null guard, numeric/NaN screen over features + study inputs):

```python
"""ModelMatrix contract: reserved-column registry + explicit feature manifest."""
from __future__ import annotations
import pandas as pd

RESERVED = (
    "y_fwd_bps", "label", "t_event", "t_barrier", "t_feature_start", "t_available",
    "cost_bps", "half_spread_bps", "uniqueness", "regime", "horizon",
)

_TIMING = ("t_event", "t_barrier", "t_feature_start", "t_available")
# Numeric study inputs beyond the features. NaN/pd.NA here fails open in the plain
# comparisons below (Series.all() skips NA) or corrupts PnL/weights downstream.
_NUMERIC = ("y_fwd_bps", "cost_bps", "half_spread_bps", "uniqueness")


def validate_matrix(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Validate the contract. Features come from the explicit manifest, never inferred."""
    dups = sorted(set(df.columns[df.columns.duplicated()]))
    if dups:
        # matrix[feature_cols] returns EVERY label match: a duplicated label silently
        # widens X and can smuggle a shadowed series past the name screens.
        raise ValueError(f"duplicate columns in ModelMatrix: {dups}")
    dup_feats = sorted({c for c in feature_cols if list(feature_cols).count(c) > 1})
    if dup_feats:
        raise ValueError(f"duplicate feature manifest entries: {dup_feats}")
    for c in RESERVED:
        if c not in df.columns:
            raise ValueError(f"ModelMatrix missing reserved column {c!r}")
    reserved_in_manifest = set(feature_cols) & set(RESERVED)
    if reserved_in_manifest:
        raise ValueError(f"feature manifest includes reserved columns: {reserved_in_manifest}")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"manifest features not in matrix: {missing}")
    for c in _TIMING:
        # pd.NA comparisons are skipped by Series.all() (fail-open) and datetime64 math
        # raises confusing TypeErrors — require non-null integer ns up front (mirrors
        # eval.manifest.validate_frame; needed here for the legacy path and direct
        # run_study callers).
        if not pd.api.types.is_integer_dtype(df[c]):
            raise ValueError(f"timing column {c!r} must be integer nanoseconds, "
                             f"got {df[c].dtype}")
        if df[c].isna().any():
            raise ValueError(f"timing column {c!r} contains nulls")
    for c in list(feature_cols) + list(_NUMERIC):
        # Ridge (always in the ladder) raises on NaN mid-study while LightGBM masks it,
        # to_numpy(float) dies opaquely on object dtypes, and NaN cost/uniqueness would
        # silently bias the no-trade band / Sharpe weights — fail closed with the name.
        if not pd.api.types.is_numeric_dtype(df[c]):
            raise ValueError(f"column {c!r} must be numeric, got {df[c].dtype}")
        if df[c].isna().any():
            raise ValueError(f"column {c!r} contains NaN; impute or drop upstream")
    if not (df["t_barrier"] >= df["t_event"]).all():
        raise ValueError("invalid span: require t_barrier >= t_event")
    if not (df["t_available"] >= df["t_event"]).all():
        raise ValueError("invalid timing: require t_available >= t_event")
    if not (df["t_feature_start"] <= df["t_event"]).all():
        raise ValueError("invalid timing: require t_feature_start <= t_event")
    if not (df["t_available"] == df["t_event"]).all():
        raise ValueError("baseline requires t_available == t_event (synchronous decide-and-act; "
                         "model cross-venue latency upstream by lagging features)")
    if not df["label"].isin((-1, 0, 1)).all():
        # lgbm_clf only maps classes +1/-1 to up/down; a stray class (e.g. {0,1,2} from a
        # mislabeled job) would be silently ignored and corrupt the forecast -> fail closed.
        raise ValueError("label must be in {-1, 0, +1}")
    if not ((df["uniqueness"] > 0) & (df["uniqueness"] <= 1)).all():
        raise ValueError("uniqueness must be in (0, 1]")
    # Fail closed on malformed cost rows: negative cost_bps or a crossed/negative
    # half_spread_bps would make total_cost negative, inverting the no-trade band (every row
    # trades) and turning the cost charge into credited PnL -> a bad book row could inflate G1.
    if not (df["cost_bps"] >= 0).all():
        raise ValueError("cost_bps must be non-negative (fees + slippage)")
    if not (df["half_spread_bps"] >= 0).all():
        raise ValueError("half_spread_bps must be non-negative (no crossed/negative spread)")
```

- [ ] **Step 4: Run to verify it passes**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_matrix.py tests/test_study.py tests/test_manifest.py -v`
Expected: all pass (`tests/test_matrix.py` = 8 existing + 5 new = 13 passed; adjust the total if the pre-existing count has drifted — the 5 new names must all pass). `test_study.py`/`test_manifest.py` confirm no synthetic fixture trips the new screens.

- [ ] **Step 5: Commit**

```bash
git add eval/matrix.py tests/test_matrix.py
git commit -m "feat: validate_matrix fails closed on duplicates, nullable timing, non-numeric/NaN inputs"
```

---

## Task 5: `evaluate_config` X-width assert + pinned `random_state`; order pin

**Files:**
- Modify: `eval/baseline.py`
- Test: `tests/test_baseline.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_baseline.py` (add `import pandas as pd` and `import pytest` at the top if absent):

```python
def test_evaluate_config_duplicate_guards():
    # Defense-in-depth for callers that bypass validate_matrix/validate_frame:
    # duplicated FRAME labels widen X; duplicated feature_cols ENTRIES double-weight
    # a column (and would sail past a width-only check, since df[["a","a"]] is width 2).
    df, feats, _ = make_matrix(n=200, signal_strength=1.0, seed=3)
    dup = pd.concat([df, df[[feats[0]]]], axis=1)
    with pytest.raises(ValueError, match="widened"):
        evaluate_config(dup, feats, "naive", n_groups=4, k=1, embargo_ns=0)
    with pytest.raises(ValueError, match="double-weight"):
        evaluate_config(df, feats + [feats[0]], "naive", n_groups=4, k=1, embargo_ns=0)


def test_feature_matrix_follows_manifest_order_not_frame_order():
    # Characterization pin (deliberately not failing-first): manifest order -> numpy
    # column order, regardless of frame column order (LightGBM reproducibility). Guards
    # against a future pandas behavior change or a rewrite of the selection idiom.
    df, feats, _ = make_matrix(n=100, signal_strength=1.0, seed=3)
    reordered = df[list(df.columns[::-1])]
    assert (reordered[feats].to_numpy(float) == df[feats].to_numpy(float)).all()
```

- [ ] **Step 2: Run to verify the first fails**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_baseline.py -k "duplicate or order" -v`
Expected: `test_evaluate_config_duplicate_guards` FAILED (no raise); the order pin passes (it documents existing behavior).

- [ ] **Step 3: Implement**

In `eval/baseline.py::evaluate_config`, replace the first line of the body:

```python
    X = matrix[feature_cols].to_numpy(float)
```

with:

```python
    feature_cols = list(feature_cols)
    if len(set(feature_cols)) != len(feature_cols):
        raise ValueError("duplicate feature_cols entries double-weight columns; "
                         "deduplicate the manifest")
    X = matrix[feature_cols].to_numpy(float)
    if X.shape[1] != len(feature_cols):
        raise ValueError("feature selection widened X (duplicate column labels in the "
                         "matrix); run validate_matrix/validate_frame first")
```

In `_fit_predict`, add `random_state=0` to both LightGBM constructors:

```python
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                              min_child_samples=50, subsample=0.8, verbose=-1,
                              random_state=0)
```

```python
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               min_child_samples=50, subsample=0.8, verbose=-1,
                               random_state=0)
```

(No-op today: `subsample` is inert with LightGBM's default `bagging_freq=0` and no feature sampling is configured, so runs are already deterministic — pinned so a future bagging/`feature_fraction` change cannot silently break run-to-run reproducibility. Deciding whether `subsample` should actually bag is a follow-up, since that changes model behavior and G1 results.)

- [ ] **Step 4: Run to verify it passes**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_baseline.py tests/test_gate_synthetic.py -v`
Expected: `tests/test_baseline.py` = 6 existing + 2 new = 8 passed; `tests/test_gate_synthetic.py` 2 passed (PASS on planted signal / FAIL on noise unchanged — confirms `random_state=0` did not alter results).

- [ ] **Step 5: Commit**

```bash
git add eval/baseline.py tests/test_baseline.py
git commit -m "feat: X-width duplicate guard + pinned LightGBM random_state"
```

---

## Task 6: v1-only CLI

**Files:**
- Rewrite: `scripts/run_baseline.py`

- [ ] **Step 1: Implement (no isolated unit test — the CLI is a thin shell over `run_from_manifest`, which Task 3 tests; syntax check + smoke below)**

Replace `scripts/run_baseline.py` in full:

```python
"""Run the G1 study on a real ModelMatrix parquet (bars E0.3 + labels E0.4 output).

Usage: .venv/bin/python scripts/run_baseline.py model_matrix.parquet feature_manifest.json
The manifest must be a v1 feature manifest (docs/feature-manifest.md) and must include
the pre-registered "gate" block. Legacy {feature_cols, embargo_ns, max_lookback_ns, gate}
dicts are NOT accepted here: write a v1 manifest and pre-register it.
"""
import sys, pathlib
# Run as a bare script (`python scripts/run_baseline.py ...`): Python puts this file's
# own dir (scripts/) on sys.path, not the repo root, so put the repo root first to make
# the `eval` package importable. Harmless when already importable.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import pandas as pd
from eval.manifest import load_manifest
from eval.runner import resolve_gate, run_from_manifest

def main(matrix_path, manifest_path):
    man = load_manifest(manifest_path)   # v1 schema-validated; fails before the parquet read
    resolve_gate(man)                    # gate errors also surface before the parquet read
    m = pd.read_parquet(matrix_path)
    res = run_from_manifest(m, man)
    ident = res["manifest"]
    print(f"manifest: {ident['dataset_id']} / {ident['build_id']} "
          f"({len(ident['feature_cols'])} features)")
    print(f"resolved gate: {res['gate']}")                # echo the EFFECTIVE (resolved) config
    for h, out in res["horizons"].items():
        status = "PASS" if out["g1_pass"] else ("INCONCLUSIVE" if out["g1_inconclusive"] else "FAIL")
        print(f"\n=== horizon {h} ===  G1: {status}  (winner={out['winner']}, pbo={out['pbo']:.3f})")
        for name, r in out["rungs"].items():
            print(f"  {name:9s} gross={r['gross_pnl']:.1f} net={r['net_pnl']:.1f} "
                  f"cost_wall={r['cost_wall']:.1f} trade_sr={r['trade_sharpe']:.3f} "
                  f"sample_sr={r['sample_sharpe']:.3f} dsr={r['dsr']:.3f} "
                  f"turnover={r['turnover']:.3f} mcc={r['mcc']:.3f} trades={r['n_trades']} "
                  f"pass={r['passes_solo']}")
        for reg, r in out["per_regime"].items():
            print(f"  regime {reg:6s}: net={r['net_pnl']:.1f} sample_sr={r['sample_sharpe']:.3f} n={r['n']}")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Syntax check + synthetic smoke**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m py_compile scripts/run_baseline.py`
Expected: exit 0, no output.

Run (end-to-end smoke on a temp synthetic build; **run from the worktree root** — the heredoc uses relative paths):

```bash
/home/aaron/jepa-btc-forecasting/.venv/bin/python - <<'EOF'
import json, subprocess, sys, tempfile, pathlib
sys.path.insert(0, ".")
from eval.synthetic import make_matrix, make_manifest
d = pathlib.Path(tempfile.mkdtemp())
df, feats, lb = make_matrix(n=900, signal_strength=4.0, seed=8)
df.to_parquet(d / "m.parquet")
man = make_manifest(feats, lb, gate={"n_groups": 4, "k": 2, "min_trades": 1, "min_eff_trades": 1.0})
(d / "man.json").write_text(json.dumps(man))
out = subprocess.run([sys.executable, "scripts/run_baseline.py", str(d / "m.parquet"), str(d / "man.json")],
                     capture_output=True, text=True)
print(out.stdout)
print(out.stderr, file=sys.stderr)   # surface the traceback if the CLI fails
sys.exit(out.returncode)
EOF
```

Expected: prints `manifest: synthetic / seeded (5 features)`, the resolved gate, and a `=== horizon 10s ===` block; exit 0.

- [ ] **Step 3: Phase-1 closeout — full suite**

Resource rule: first check `ps -o pid,etimes,pcpu,pmem,args -C python -C pytest`; skip (and note in the PR) if another agent is running a full suite or heavy data job.

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest -q`
Expected: all pass; integration tests (`test_baseline_integration.py`, `test_fixture_integration.py`, and other data-gated tests) skip without real data.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_baseline.py
git commit -m "feat: run_baseline CLI requires a v1 manifest via load_manifest"
```

---

# Phase 2 — baseline configs, tests, and docs consume v1

Mapping to the commissioned "phase 2: baseline configs updated": this repo has **no committed baseline config files** — the manifest JSON (with its pre-registered `gate` block) *is* the baseline's run configuration, and the `CONFIGS` ladder tuple in `eval/baseline.py:12` is code, not config. So "configs updated" here means: the real config artifact (`data/processed/feature_manifest.json`, once the E0.3/E0.4 producer emits it) is pinned to the v1 shape by the integration test (Task 7), and the contract doc that tells humans how to write one is synced (Task 8). Runner-level test fixtures already migrated with the runner change in Task 3.

## Task 7: Integration test pins the real artifact to v1

**Files:**
- Modify: `tests/test_baseline_integration.py`

- [ ] **Step 1: Implement (skip-gated test; it cannot run locally until real data exists — change is reviewed, not executed)**

Replace `tests/test_baseline_integration.py` in full:

```python
import pathlib, pytest
import pandas as pd
from eval.manifest import load_manifest
from eval.runner import run_from_manifest

MATRIX = pathlib.Path("data/processed/model_matrix.parquet")
MANIFEST = pathlib.Path("data/processed/feature_manifest.json")

pytestmark = pytest.mark.skipif(not (MATRIX.exists() and MANIFEST.exists()),
    reason="needs real ModelMatrix + manifest from bars (E0.3) + labels (E0.4)")

def test_real_matrix_runs_through_manifest():
    m = pd.read_parquet(MATRIX)
    man = load_manifest(MANIFEST)                        # the real artifact must be v1
    res = run_from_manifest(m, man)                      # same path the CLI uses
    assert res["gate"] and res["horizons"]
    assert res["manifest"]["dataset_id"]                 # identity echoed for reproducibility
    any_h = next(iter(res["horizons"].values()))
    assert "g1_pass" in any_h and any_h["per_regime"]
```

- [ ] **Step 2: Verify it still skips cleanly**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_baseline_integration.py -v`
Expected: 1 skipped (data/processed absent in agent worktrees).

- [ ] **Step 3: Commit**

```bash
git add tests/test_baseline_integration.py
git commit -m "test: real-data integration test loads the manifest via load_manifest (v1 required)"
```

---

## Task 8: Sync `docs/feature-manifest.md`

**Files:**
- Modify: `docs/feature-manifest.md` (the "How training consumes it" section only; do NOT touch `docs/data.md` — another branch owns it)

- [ ] **Step 1: Implement**

Replace the paragraph starting "`eval.runner.run_from_manifest` applies `validate_frame` automatically…" (currently `docs/feature-manifest.md:61-69`) with:

```markdown
`eval.runner.run_from_manifest` applies `validate_frame` automatically whenever
the manifest carries `manifest_version`, selects features via `feature_list`
(manifest order), and adds three baseline-specific fail-closed checks: declared
`target_cols` must be exactly `{y_fwd_bps, label}` (what `evaluate_config`
trains on), `availability_lag_ns` must be 0 (the baseline is synchronous —
lag features upstream), and every declared horizon must be present in the
frame. The result dict echoes `{dataset_id, build_id, generated_at,
embargo_ns, max_lookback_ns, feature_cols}` under `"manifest"` so a run is
reproducible from its own output. The `scripts/run_baseline.py` CLI accepts
only v1 manifests (via `load_manifest`); legacy `{feature_cols, embargo_ns,
max_lookback_ns, gate}` dicts still work for direct library callers of
`run_from_manifest` (with a leaky-name screen but no frame validation) until
the legacy branch is removed. A manifest that carries v1-only fields but lost
`manifest_version` (typo) is refused rather than silently treated as legacy.
Note `gate` is optional at schema level but required by `run_from_manifest`
and the CLI (`eval.runner.resolve_gate`); JEPA pretraining manifests may omit
it and will consume the same manifest with
`validate_frame(df, man, require_targets=False)` (label columns may be absent
for unsupervised pretraining; everything else still applies).
```

- [ ] **Step 2: Phase-2 closeout — check + full suite**

Run: `git diff --check`
Expected: no output.

Resource rule as in Task 6 Step 3, then run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest -q`
Expected: all pass; data-gated integration tests skip.

- [ ] **Step 3: Commit**

```bash
git add docs/feature-manifest.md
git commit -m "docs: feature-manifest consumption section matches the wired runner/CLI"
```

---

# Phase 3 — legacy path removed

**Preconditions (all must hold before starting):**
- Phases 1–2 merged.
- No other open branch constructs legacy manifest dicts (grep PRs/branches for `run_from_manifest` callers).
- Either the real `data/processed/feature_manifest.json` is already v1, or it still does not exist (nothing to migrate).

## Task 9: Require `manifest_version` in `run_from_manifest`

**Files:**
- Modify: `eval/runner.py`, `tests/test_runner.py`, `tests/test_manifest.py` (comments), `docs/feature-manifest.md`, possibly `eval/manifest.py`

- [ ] **Step 1: Update the tests first**

In `tests/test_runner.py`: delete `test_legacy_manifest_dict_still_runs`, `test_legacy_manifest_rejects_leaky_feature_names`, and `test_legacy_manifest_missing_keys_fail_with_contract_error`; add:

```python
def test_run_from_manifest_requires_versioned_manifest():
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    man = {"feature_cols": feats, "embargo_ns": lb, "max_lookback_ns": lb,
           "gate": {"n_groups": 4, "k": 2}}
    with pytest.raises(ValueError, match="manifest_version"):
        run_from_manifest(m, man)
```

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_runner.py -v`
Expected: `test_run_from_manifest_requires_versioned_manifest` FAILED (legacy dict still runs); other 7 pass.

- [ ] **Step 2: Implement**

In `eval/runner.py`, change the import line to:

```python
from eval.manifest import feature_list, target_list, validate_frame
```

delete the `_LEGACY_KEYS` constant, and replace `run_from_manifest` in full (the v1 body is Task 3's, dedented out of the now-redundant `if` wrapper; the legacy branch is gone):

```python
def run_from_manifest(matrix: pd.DataFrame, manifest: dict) -> dict:
    if "manifest_version" not in manifest:
        raise ValueError(
            "run_from_manifest requires a v1 feature manifest (add manifest_version=1; "
            "see docs/feature-manifest.md); legacy {feature_cols, embargo_ns, "
            "max_lookback_ns, gate} dicts are no longer accepted")
    validate_frame(matrix, manifest)
    feats = feature_list(manifest)               # validated copy, manifest order
    targets = set(target_list(manifest))
    if targets != BASELINE_TARGETS:
        raise ValueError(f"the LightGBM baseline consumes exactly "
                         f"{sorted(BASELINE_TARGETS)} as targets; manifest declares "
                         f"{sorted(targets)}")
    if manifest.get("availability_lag_ns", 0) != 0:
        raise ValueError("the LightGBM baseline is synchronous (t_available == t_event); "
                         "availability_lag_ns > 0 is reserved for future consumers — "
                         "lag features upstream instead")
    # validate_frame checked frame tags are declared; check the converse so declared
    # horizons cannot silently vanish from the per-horizon results.
    missing_h = sorted(set(manifest["horizons"]) - set(matrix["horizon"].unique()))
    if missing_h:
        raise ValueError(f"manifest horizons missing from the matrix: {missing_h}; "
                         "the manifest must describe this exact build")
    out = {"manifest": {k: manifest[k] for k in
                        ("dataset_id", "build_id", "generated_at",
                         "embargo_ns", "max_lookback_ns")}}
    out["manifest"]["feature_cols"] = feats
    gate = resolve_gate(manifest)
    emb, lb = manifest["embargo_ns"], manifest["max_lookback_ns"]
    horizons = {}
    # observed=True: a categorical horizon column must not yield empty subframes for
    # unused categories (run_study crashes on an empty matrix under pandas 2.x defaults)
    for h, sub in matrix.groupby("horizon", observed=True):
        horizons[str(h)] = run_study(sub.reset_index(drop=True), feats, cost_default=None,
                                     embargo_ns=emb, max_lookback_ns=lb, **gate)
    out.update({"gate": gate, "horizons": horizons})
    return out
```

The guard message keeps the literal `manifest_version`, so `tests/test_manifest.py::test_run_from_manifest_refuses_v1_fields_without_version` and `...refuses_optional_v1_fields_without_version` stay green. Do **not** rename them; only update their comments to say the guard is now the universal required-version error rather than a dual-path downgrade trap. Then grep `V1_ONLY_FIELDS`: its only references are `eval/runner.py` (just removed) and its definition — delete the constant and its explanatory comment (which names `eval.runner`) from `eval/manifest.py`. Keep `leaky_feature_names` (public API for manifest-authoring tools).

- [ ] **Step 3: Run to verify**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest tests/test_runner.py tests/test_manifest.py -v`
Expected: `tests/test_runner.py` 8 passed; `tests/test_manifest.py` all pass.

- [ ] **Step 4: Sync docs + commit**

In `docs/feature-manifest.md`, in the Task 8 text, replace the sentence

> The `scripts/run_baseline.py` CLI accepts only v1 manifests (via `load_manifest`); legacy `{feature_cols, embargo_ns, max_lookback_ns, gate}` dicts still work for direct library callers of `run_from_manifest` (with a leaky-name screen but no frame validation) until the legacy branch is removed.

with

> The `scripts/run_baseline.py` CLI accepts only v1 manifests (via `load_manifest`), and `run_from_manifest` refuses non-versioned dicts with a migration error.

```bash
git add eval/runner.py eval/manifest.py tests/test_runner.py tests/test_manifest.py docs/feature-manifest.md
git commit -m "feat!: run_from_manifest requires a v1 feature manifest (legacy dicts removed)"
```

- [ ] **Step 5: Full suite (resource rule: check `ps -o pid,etimes,pcpu,pmem,args -C python -C pytest` first; skip if another agent runs a heavy job)**

Run: `/home/aaron/jepa-btc-forecasting/.venv/bin/python -m pytest -q`
Expected: all pass, `tests/test_baseline_integration.py` skipped unless real data exists.

---

## Non-goals and deferred work

- **`unsafe_infer_feature_cols` stays** — restricted, not removed. It is the one deliberately named, exploration-only inference escape hatch sanctioned by AGENTS.md *because this plan explicitly says so*; it fails closed on leaky names, has zero production callers, and its output is for writing into a manifest to pre-register, never for feeding a model. No training entry point may consume it, in any phase.
- **`cost_default` removal** — `run_study` declares it and never reads it (costs are per-row `cost_bps`/`half_spread_bps`). Deleting it touches 15 test call sites plus the runner's own call; mechanical, separate PR.
- **`configs`/`extra_trials` stay out of the manifest surface** — the 2026-06-22 plan's "Still honestly deferred" block pins this deliberately (extending it requires a persistent cross-study trial ledger feeding DSR `n_trials`). This plan does not re-open it.
- **LightGBM `subsample` is inert** (`bagging_freq=0`) — making it actually bag changes model behavior and G1 results; decide separately. Task 5 only pins `random_state`.
- **Producer side** — emitting the real v1 `feature_manifest.json` from the bars/labels build jobs (E0.3/E0.4) is that pipeline's task; this plan pins the consumption contract it must satisfy (`load_manifest` in the CLI and integration test). The same producer must also bring the **value-level no-lookahead tests for feature computation** (the bar-level analogue of the E0.1 replay-equivalence gate, which today covers book reconstruction only) — until then, no test anywhere verifies feature values are lookahead-free, and manifest validation does not claim to.
- **JEPA path** — `validate_frame(df, man, require_targets=False)` consumption is future work; nothing here blocks it (the `BASELINE_TARGETS` check lives in the runner, not the contract).
- **`docs/data.md`** — untouched; another branch owns it.

## Risks and trade-offs

- **Strict horizon coverage** (declared ⊆ present) means a manifest cannot describe a superset build and run against a filtered slice that drops a whole tag. Accepted: the manifest pins `dataset_id`/`build_id` — it describes *this exact build* — and the alternative (silently missing result entries) is precisely gap 5. Regenerate the manifest when slicing away a horizon.
- **Strict target equality** blocks manifests that declare extra bespoke reserved targets the baseline ignores. Accepted for the same misdescription reason; the JEPA path will relax via `require_targets=False`, not by weakening the baseline check.
- **NaN/nullable fail-closed** (features, `y_fwd_bps`, cost/weight columns, nullable timing dtypes) could reject a real matrix whose features legitimately contain NaN — LightGBM alone would mask them. Accepted: imputation is an upstream, pre-registered modeling decision; silently letting LightGBM mask NaN while Ridge crashes is worse, and nullable `pd.NA` makes the existing plain-comparison checks fail *open*. If a real build needs NaN, that is a manifest/dtype design conversation, not a silent pass.
- **Test-count claims**: `tests/test_matrix.py` totals assume 8 pre-existing tests (verified 2026-07-02); if drifted, the named new tests are authoritative, not the totals.

## Self-review

- [ ] Every task's snippets are complete (no TBD/placeholder); commands include expected output.
- [ ] Names consistent across tasks: `leaky_feature_names` (Tasks 1, 3, 9), `make_manifest` (Tasks 2, 3, 6), `BASELINE_TARGETS`/`_LEGACY_KEYS` (Tasks 3, 9), `res["manifest"]` echo (Tasks 3, 6, 7, 8).
- [ ] Spec coverage: feature selection sites inventoried (Current state); target API defined (loading/order/targets/reserved/unknown); leakage checks mapped rule-by-rule incl. the E0.4 embargo-vs-horizon decision; backward compat incl. synthetic-manifest policy; three phases match the assignment (required-in-new-path / configs-updated / old-path-removed).
- [ ] No edits to `docs/data.md`; no vendor calls; no data jobs.
