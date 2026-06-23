# LightGBM Baseline + Signal-Existence Gate (G1) — Implementation Plan (rev. 5)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build experiment-plan **Phase 1** — τ measurement, a baseline ladder (naive → penalized-linear → LightGBM regressor → LightGBM classifier) evaluated under **per-test-span purged/embargoed CPCV** with **per-sample-cost** net PnL and **uniqueness-weighted** metrics, and a **study-level** Deflated-Sharpe + PBO that drives the **G1 gate** — all validated on synthetic known-signal-vs-noise data.

**Architecture:** A modeling/evaluation layer over an explicit `ModelMatrix` contract (features via a manifest + reserved columns for return, label, timing/availability, per-sample cost, and uniqueness weight). CPCV returns the **per-fold OOS distribution** (never collapsed to one vector); a `Study` collects per-config results into a trial ledger and computes DSR/PBO from the **real configs × OOS series**. The gate is proven on planted-signal (PASS) and pure-noise (FAIL) matrices, and reports **per-regime** breakdowns.

**Tech Stack:** Python 3.12, numpy/pandas, **lightgbm + scikit-learn**, `statistics.NormalDist` for DSR, pytest.

**Scope:** experiment-plan **E1.1** (τ) and **E1.3** (ladder + G1), plus the **purged/embargoed CPCV** it requires. Consumes a `ModelMatrix` from bars (E0.3) + labels (E0.4); a synthetic generator stands in until those exist (Task 9 is the only dependency-gated task).

---

## Revision note (what changed vs rev. 1) — addresses external review

| # | Finding | Fix in this rev |
|---|---|---|
| 1 | CPCV collapsed to one vector; PBO faked from time-bins of one strategy | CPCV returns the **per-fold Sharpe distribution + per-sample OOS PnL**; DSR/PBO computed at **study level** over real configs (Tasks 6–7) |
| 2 | Union-span purge empties train for non-contiguous CPCV combos | **Per-test-span purge** over merged test intervals; test asserts non-empty train for non-contiguous combos (Task 2) |
| 3 | Global cost too thin | **Per-sample `cost_bps`/`half_spread_bps`** in the contract; band uses them (Tasks 1, 3) |
| 4 | Contract lacks live-safe timing fields | Add `t_feature_start`, `t_available`, `max_lookback_ns`; **`embargo_ns` required** and must cover `max_lookback_ns` (Tasks 1, 2, 7) |
| 5 | No sample-uniqueness weighting | `uniqueness` column → `sample_weight` in fit + weighted PnL/Sharpe (Tasks 1, 3, 6) |
| 6 | Regime stratification carried, not enforced | Gate **reports per-regime** by slicing the best config's OOS PnL (Task 7) |
| 7 | Only regresses `y_fwd_bps` | Added **`lgbm_clf`** rung on `label`, proba→signed bps (Task 6) |
| 8 | Features inferred as "every non-required column" | **Explicit feature manifest + reserved-column registry** (Task 1) |

---

## rev. 3 fixes (second review round)

| # | Finding | Fix |
|---|---|---|
| 1 | `half_spread_bps` carried but not charged | `net_pnl` charges `cost_bps + 2×half_spread_bps` (mid-anchored round trip crosses twice) |
| 2 | `t_available` recorded but not used → credit for pre-actionable returns | `validate_matrix` **enforces `t_available == t_event`** (synchronous; latency handled upstream by lagging features) |
| 3 | G1 passes when PBO is `nan` | gate **fails closed**; `g1_inconclusive` flag when PBO uncomputable |
| 4 | DSR `T` = raw overlapping trade count | `T` = **effective sample size** = Σ uniqueness over trades |
| 5 | `lgbm_clf` crashes on single-class folds | single-class fold → zeros (no trades) |
| 6 | no gross/turnover/MCC | added to per-config result + study output |

### rev. 4 fixes (third review round)

| # | Finding | Fix |
|---|---|---|
| 1 | trade-only Sharpe feeds DSR/gate | report trade-level **and** sample/time-level Sharpe; gate adds a pre-registered `min_trades` turnover floor (lets DSR keep trade-level) |
| 2 | CLI omits gross/turnover/MCC/inconclusive | entrypoint prints gross, cost wall, turnover, MCC, and PASS/FAIL/INCONCLUSIVE |
| 3 | `best` chosen by net before gate → false fail | per-candidate gate; G1 passes if **any** non-naive config clears (DSR `n_trials` + study PBO handle multiple testing) |
| 4 | stale expected test counts | corrected (test_matrix 6, test_cost 5) |

### rev. 5 fixes (fourth review round)

| # | Finding | Fix |
|---|---|---|
| 1 | G1 gates on trade-level Sharpe only | gate adds an explicit `min_sample_sharpe` floor (capital experience), default 0.0, configurable |
| 2 | `min_trades` uses raw count under heavy overlap | gate also requires `t_eff >= min_eff_trades` (effective-trade floor) |
| 3 | per-regime slicing assumes a RangeIndex | use `groupby(...).indices` (positional), index-agnostic |
| 4 | gate params not in manifest (reproducibility) | entrypoint reads a pre-registered `gate{}` manifest block and echoes it |

---

## The `ModelMatrix` contract (revised)

A pandas DataFrame, one row per (bar, horizon) sample. **Columns are partitioned into three sets:**

**Reserved (required) — `eval.matrix.RESERVED`:**
- `y_fwd_bps` (float) — realized forward return at the horizon, bps, off mid/microprice.
- `label` (int ∈ {-1,0,+1}) — triple-barrier sign (0 = flat/time-out).
- `t_event` (int ns) — decision time (the bar's engine time).
- `t_barrier` (int ns) — end of the label span; span = `[t_event, t_barrier]`.
- `t_feature_start` (int ns) — earliest input timestamp any feature in the row depends on (for lookback/embargo verification). Must be ≤ `t_event`.
- `t_available` (int ns) — when the full feature vector was actually actionable. **For this baseline it must equal `t_event`** (synchronous decide-and-act); cross-venue/transport latency is handled *upstream* by lagging the feature so the value known at `t_event` already respects latency. Retained for future asynchronous-action modeling (where the label would instead anchor at `t_available`).
- `cost_bps` (float) — per-sample round-trip cost from **fees + slippage only** (`2×fee + slippage`), against a mid-anchored return. Spread is charged separately.
- `half_spread_bps` (float) — per-sample half-spread. Charged in PnL: a mid-anchored taker round trip crosses the spread twice, so total cost = `cost_bps + 2×half_spread_bps + margin`.
- `uniqueness` (float ∈ (0,1]) — average-uniqueness sample weight (overlapping labels overcount).
- `regime` (category/str) — spread/vol bucket for stratified reporting.
- `horizon` (str) — e.g. `2s`, `10s`, `60s`.

**Features:** an **explicit manifest** `feature_cols: list[str]` passed alongside the matrix — *not inferred*. Must be disjoint from `RESERVED`.

**Diagnostics:** any other columns (vendor flags, debug) — ignored by the model.

**Study config (not columns), pre-registered in the manifest `gate` block:** `max_lookback_ns` (longest feature look-back), `embargo_ns` (required; ≥ `max_lookback_ns`), and gate params `n_groups`, `k`, `min_trades`, `min_eff_trades`, `min_sample_sharpe`, `dsr_thresh`, `pbo_thresh`. The entrypoint reads and echoes these for reproducibility.

---

## File structure

- `data/__init__.py`, `data/cv.py` — per-test-span purged+embargoed CPCV (Task 2)
- `eval/__init__.py`, `eval/synthetic.py` — known-signal/noise matrix w/ full contract (Task 0)
- `eval/matrix.py` — reserved registry + manifest validation (Task 1)
- `eval/cost.py` — per-sample-cost band PnL + uniqueness-weighted Sharpe (Task 3)
- `eval/stats.py` — DSR + PBO (Task 4)
- `eval/tau.py` — τ estimator (Task 5)
- `eval/baseline.py` — config fit/predict + per-config CPCV evaluation (Task 6)
- `eval/study.py` — trial ledger, study-level DSR/PBO, per-regime G1 gate (Task 7)
- `scripts/run_baseline.py` — real-feature entrypoint (Task 9)
- `tests/test_*.py` — one per module + the synthetic gate (Tasks 1–8)

---

## Task 0: Scaffolding + synthetic generator (full contract)

**Files:** Create `data/__init__.py`, `eval/__init__.py`, `eval/synthetic.py`

- [ ] **Step 1: Install deps**

Run: `.venv/bin/python -m pip install lightgbm scikit-learn`
Expected: `Successfully installed lightgbm-... scikit-learn-...`

- [ ] **Step 2: Package markers**

`data/__init__.py`:
```python
"""Datasets, CV splits, labels (spec §3)."""
```
`eval/__init__.py`:
```python
"""Baseline, backtest harness, PnL/no-trade-band metrics (spec §3)."""
```

- [ ] **Step 3: Synthetic generator (emits every reserved column)**

`eval/synthetic.py`:
```python
"""Deterministic synthetic ModelMatrix with a KNOWN, tunable signal and the full
reserved-column contract (cost, uniqueness, timing/availability, regime)."""
from __future__ import annotations
import numpy as np
import pandas as pd

FEATURES = ["ofi_integrated", "microprice_dev", "queue_imb", "spread_tick", "cvd"]


def _concurrency_uniqueness(t0: np.ndarray, t1: np.ndarray) -> np.ndarray:
    """uniqueness_i = 1 / (# label spans covering t_event_i)."""
    t0s = np.sort(t0); t1s = np.sort(t1)
    started = np.searchsorted(t0s, t0, side="right")
    ended = np.searchsorted(t1s, t0, side="right")
    conc = np.maximum(started - ended, 1)
    return 1.0 / conc


def make_matrix(n: int = 8000, *, signal_strength: float, seed: int,
                horizon_ns: int = 10_000_000_000, noise_bps: float = 8.0,
                latency_ns: int = 50_000_000):
    """Returns (df, feature_cols, max_lookback_ns)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, len(FEATURES)))
    f = X[:, 0] * 1.0 + np.tanh(X[:, 1]) * 1.5 + (X[:, 2] > 0.5) * X[:, 3]
    y = signal_strength * f + rng.standard_normal(n) * noise_bps
    step = horizon_ns // 4                       # overlapping labels (concurrency ~4)
    t_event = (np.arange(n, dtype=np.int64) + 1) * step
    t_barrier = t_event + horizon_ns
    lookback = horizon_ns                        # feature window
    regime = np.where(X[:, 3] > 0, "tight", "wide")
    df = pd.DataFrame(X, columns=FEATURES)
    df["y_fwd_bps"] = y
    df["label"] = np.sign(y).astype(int)
    df["t_event"] = t_event
    df["t_barrier"] = t_barrier
    df["t_feature_start"] = t_event - lookback
    df["t_available"] = t_event  # synchronous baseline: latency handled upstream by lagging features
    df["cost_bps"] = np.where(regime == "wide", 4.0, 1.5)
    df["half_spread_bps"] = np.where(regime == "wide", 2.0, 0.6)
    df["uniqueness"] = _concurrency_uniqueness(t_event, t_barrier)
    df["regime"] = regime
    df["horizon"] = "10s"
    return df, list(FEATURES), int(lookback)
```

- [ ] **Step 4: Smoke-check**

Run: `.venv/bin/python -c "from eval.synthetic import make_matrix; d,f,lb=make_matrix(signal_strength=3,seed=0); print(d.shape, f, lb, d['uniqueness'].mean())"`
Expected: prints shape `(8000, 16)`, the 5 feature names, a lookback int, and a uniqueness mean < 1.

- [ ] **Step 5: Commit**
```bash
git init -q 2>/dev/null; git add data/__init__.py eval/__init__.py eval/synthetic.py
git commit -m "chore: scaffold baseline pkg + full-contract synthetic generator"
```

---

## Task 1: ModelMatrix contract — reserved registry + manifest

**Files:** Create `eval/matrix.py`, `tests/test_matrix.py`

- [ ] **Step 1: Failing test**

`tests/test_matrix.py`:
```python
import pytest
from eval.matrix import validate_matrix, RESERVED
from eval.synthetic import make_matrix


def test_valid_matrix_passes():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    validate_matrix(df, feats)  # no raise


def test_missing_reserved_column_raises():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="t_available"):
        validate_matrix(df.drop(columns=["t_available"]), feats)


def test_feature_manifest_must_be_disjoint_from_reserved():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="reserved"):
        validate_matrix(df, feats + ["cost_bps"])


def test_unknown_feature_in_manifest_raises():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="not in matrix"):
        validate_matrix(df, feats + ["nonexistent_feat"])


def test_timing_invariants_enforced():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad.loc[0, "t_available"] = bad.loc[0, "t_event"] - 1
    with pytest.raises(ValueError, match="t_available >= t_event"):
        validate_matrix(bad, feats)
    bad2 = df.copy(); bad2.loc[0, "t_feature_start"] = bad2.loc[0, "t_event"] + 1
    with pytest.raises(ValueError, match="t_feature_start <= t_event"):
        validate_matrix(bad2, feats)


def test_baseline_requires_synchronous_t_available():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)  # synthetic sets t_available == t_event
    bad = df.copy(); bad.loc[0, "t_available"] = bad.loc[0, "t_event"] + 1
    with pytest.raises(ValueError, match="t_available == t_event"):
        validate_matrix(bad, feats)
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_matrix.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.matrix'`

- [ ] **Step 3: Implement**

`eval/matrix.py`:
```python
"""ModelMatrix contract: reserved-column registry + explicit feature manifest."""
from __future__ import annotations
import pandas as pd

RESERVED = (
    "y_fwd_bps", "label", "t_event", "t_barrier", "t_feature_start", "t_available",
    "cost_bps", "half_spread_bps", "uniqueness", "regime", "horizon",
)


def validate_matrix(df: pd.DataFrame, feature_cols: list[str]) -> None:
    """Validate the contract. Features come from the explicit manifest, never inferred."""
    for c in RESERVED:
        if c not in df.columns:
            raise ValueError(f"ModelMatrix missing reserved column {c!r}")
    reserved_in_manifest = set(feature_cols) & set(RESERVED)
    if reserved_in_manifest:
        raise ValueError(f"feature manifest includes reserved columns: {reserved_in_manifest}")
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"manifest features not in matrix: {missing}")
    if not (df["t_barrier"] >= df["t_event"]).all():
        raise ValueError("invalid span: require t_barrier >= t_event")
    if not (df["t_available"] >= df["t_event"]).all():
        raise ValueError("invalid timing: require t_available >= t_event")
    if not (df["t_feature_start"] <= df["t_event"]).all():
        raise ValueError("invalid timing: require t_feature_start <= t_event")
    if not (df["t_available"] == df["t_event"]).all():
        raise ValueError("baseline requires t_available == t_event (synchronous decide-and-act; "
                         "model cross-venue latency upstream by lagging features)")
    if not ((df["uniqueness"] > 0) & (df["uniqueness"] <= 1)).all():
        raise ValueError("uniqueness must be in (0, 1]")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_matrix.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**
```bash
git add eval/matrix.py tests/test_matrix.py
git commit -m "feat: ModelMatrix reserved registry + explicit feature manifest"
```

---

## Task 2: Per-test-span purged + embargoed CPCV

**Files:** Create `data/cv.py`, `tests/test_cv.py`

- [ ] **Step 1: Failing test (incl. the non-contiguous non-empty-train guard)**

`tests/test_cv.py`:
```python
import numpy as np
from math import comb
from data.cv import make_time_groups, cpcv_splits


def _spans(n, span=10):
    t0 = (np.arange(n) * 5).astype(np.int64)
    return t0, (t0 + span).astype(np.int64)


def test_groups_balanced():
    t0, _ = _spans(120)
    g = make_time_groups(t0, n_groups=6)
    assert set(g.tolist()) == set(range(6))


def test_path_count_equals_n_choose_k():
    t0, t1 = _spans(120)
    assert len(list(cpcv_splits(t0, t0, t1, n_groups=6, k=2, embargo_ns=0))) == comb(6, 2)


def test_no_train_span_overlaps_any_test_span():
    t0, t1 = _spans(120)
    for tr, te in cpcv_splits(t0, t0, t1, n_groups=6, k=2, embargo_ns=0):
        for j in tr:
            assert not ((t0[j] <= t1[te]) & (t1[j] >= t0[te])).any()


def test_noncontiguous_combo_keeps_substantial_train():
    # THE rev-2 guard: union-span purge would empty this; per-span purge must not.
    t0, t1 = _spans(120)
    splits = list(cpcv_splits(t0, t0, t1, n_groups=6, k=2, embargo_ns=0))
    for tr, te in splits:
        assert len(tr) >= 40          # ~ (6-2)/6 of 120 minus purge halo, never empty


def test_embargo_drops_post_test_window():
    t0, t1 = _spans(120)
    emb = 50
    for tr, te in cpcv_splits(t0, t0, t1, n_groups=6, k=1, embargo_ns=emb):
        hi = t1[te].max()
        assert not ((t0[tr] > hi) & (t0[tr] <= hi + emb)).any()
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cv.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'data.cv'`

- [ ] **Step 3: Implement (merge test intervals, purge per interval)**

`data/cv.py`:
```python
"""Purged + embargoed Combinatorial Purged CV (López de Prado).

Purges per TEST INTERVAL: the test rows' spans are merged into disjoint intervals,
and a train row is purged only if it overlaps one of those intervals (or starts within
the embargo after one). This stays correct for NON-CONTIGUOUS CPCV combos, where a
union-span purge would wipe out nearly all training data.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations


def make_time_groups(t_event: np.ndarray, n_groups: int) -> np.ndarray:
    order = np.argsort(t_event, kind="stable")
    rank = np.empty(len(t_event), dtype=np.int64)
    rank[order] = np.arange(len(t_event))
    return (rank * n_groups // len(t_event)).astype(int)


def _merge_intervals(lo: np.ndarray, hi: np.ndarray):
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    merged = []
    for a, b in zip(lo, hi):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def cpcv_splits(t_event, t0, t1, *, n_groups: int, k: int, embargo_ns: int):
    """Yield (train_idx, test_idx) for every k-of-n_groups combination. embargo_ns is
    REQUIRED (set it ≥ the longest feature look-back to avoid feature-window leakage)."""
    t0 = np.asarray(t0); t1 = np.asarray(t1)
    groups = make_time_groups(t_event, n_groups)
    for combo in combinations(range(n_groups), k):
        test_mask = np.isin(groups, combo)
        test_idx = np.where(test_mask)[0]
        purge = np.zeros(len(t0), bool)
        for lo, hi in _merge_intervals(t0[test_idx], t1[test_idx]):
            purge |= (t0 <= hi) & (t1 >= lo)                 # span overlap
            purge |= (t0 > hi) & (t0 <= hi + embargo_ns)     # embargo after the block
        train_idx = np.where(~test_mask & ~purge)[0]
        yield train_idx, test_idx
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cv.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**
```bash
git add data/cv.py tests/test_cv.py
git commit -m "feat: per-test-span purged+embargoed CPCV (fixes non-contiguous empty-train)"
```

---

## Task 3: Per-sample-cost band PnL + uniqueness-weighted Sharpe

**Files:** Create `eval/cost.py`, `tests/test_cost.py`

- [ ] **Step 1: Failing test**

`tests/test_cost.py`:
```python
import numpy as np
from eval.cost import net_pnl, weighted_sharpe


def test_per_sample_cost_band():
    fc = np.array([3.0, 3.0, -5.0])
    rr = np.array([4.0, 4.0, -3.0])
    cost = np.array([5.0, 1.0, 1.0])     # sample0 band too high to trade; no spread here
    pnl, traded, gross = net_pnl(fc, rr, cost_bps=cost)
    assert traded.tolist() == [False, True, True]
    assert pnl[1] == 4.0 - 1.0           # +1*4 - 1
    assert pnl[2] == 3.0 - 1.0           # -1*-3 - 1
    assert gross[1] == 4.0


def test_spread_charged_via_two_crossings():
    fc = np.array([5.0]); rr = np.array([4.0])
    pnl, traded, gross = net_pnl(fc, rr, cost_bps=np.array([1.0]),
                                 half_spread_bps=np.array([0.5]))
    # total cost = 1 + 2*0.5 = 2 ; band 2 ; |5|>2 trade ; pnl = 4 - 2 ; gross = 4
    assert traded[0] and pnl[0] == 2.0 and gross[0] == 4.0


def test_zero_edge_loses_costs():
    rng = np.random.default_rng(0)
    rr = rng.standard_normal(5000) * 10
    fc = rng.standard_normal(5000) * 10
    pnl, traded, _ = net_pnl(fc, rr, cost_bps=np.full(5000, 2.0))
    assert pnl[traded].mean() < 0


def test_weighted_sharpe_respects_weights():
    pnl = np.array([1.0, 1.0, -10.0])
    w_low_on_outlier = np.array([1.0, 1.0, 0.01])
    w_equal = np.array([1.0, 1.0, 1.0])
    assert weighted_sharpe(pnl, w_low_on_outlier) > weighted_sharpe(pnl, w_equal)


def test_weighted_sharpe_zero_when_degenerate():
    assert weighted_sharpe(np.zeros(5), np.ones(5)) == 0.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_cost.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.cost'`

- [ ] **Step 3: Implement**

`eval/cost.py`:
```python
"""No-trade-band, fees-included PnL with PER-SAMPLE cost + spread; uniqueness-weighted Sharpe."""
from __future__ import annotations
import numpy as np


def net_pnl(forecast_bps, realized_bps, *, cost_bps, half_spread_bps=0.0,
            spread_crossings=2, margin_bps=0.0):
    """Trade when |forecast| > total per-sample band, where
    total_cost = cost_bps + spread_crossings*half_spread_bps. A mid-anchored taker round
    trip crosses the spread twice (buy at ask, sell at bid) -> spread_crossings=2.
    cost_bps/half_spread_bps may be scalar or per-sample arrays. Honest taker fills.
    Returns (pnl_per_sample, traded_mask, gross_pnl_per_sample)."""
    fc = np.asarray(forecast_bps, float); rr = np.asarray(realized_bps, float)
    total_cost = (np.asarray(cost_bps, float)
                  + spread_crossings * np.asarray(half_spread_bps, float)) * np.ones_like(fc)
    band = total_cost + margin_bps
    traded = np.abs(fc) > band
    gross = np.where(traded, np.sign(fc) * rr, 0.0)
    pnl = np.where(traded, gross - total_cost, 0.0)
    return pnl, traded, gross


def weighted_sharpe(pnl_per_sample, weights, *, trade_only: bool = True) -> float:
    """Uniqueness-weighted Sharpe. trade_only=True -> over traded samples (hit quality);
    trade_only=False -> over ALL decision samples incl. no-trade zeros (the strategy's
    sample/time-level Sharpe). Overlapping labels overcount, hence the weighting."""
    pnl = np.asarray(pnl_per_sample, float)
    w = np.asarray(weights, float)
    if trade_only:
        mask = pnl != 0.0
        p, ww = pnl[mask], w[mask]
    else:
        p, ww = pnl, w
    if len(p) < 2 or ww.sum() == 0:
        return 0.0
    mean = np.average(p, weights=ww)
    var = np.average((p - mean) ** 2, weights=ww)
    return float(mean / (np.sqrt(var) + 1e-12)) if var > 0 else 0.0
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_cost.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**
```bash
git add eval/cost.py tests/test_cost.py
git commit -m "feat: per-sample-cost band PnL + uniqueness-weighted Sharpe"
```

---

## Task 4: Deflated Sharpe + PBO

**Files:** Create `eval/stats.py`, `tests/test_stats.py`

- [ ] **Step 1: Failing test**

`tests/test_stats.py`:
```python
import numpy as np
from eval.stats import deflated_sharpe, pbo


def test_dsr_high_when_t_large_and_sr_clears_benchmark():
    d = deflated_sharpe(sr_hat=0.25, sr_trials_std=0.12, n_trials=4, T=3000, skew=0.0, kurt=3.0)
    assert d > 0.95


def test_dsr_low_when_sr_is_noise_max():
    d = deflated_sharpe(sr_hat=0.02, sr_trials_std=0.05, n_trials=1000, T=1500, skew=0.0, kurt=3.0)
    assert d < 0.5


def test_dsr_requires_two_trials():
    import pytest
    with pytest.raises(ValueError):
        deflated_sharpe(sr_hat=0.3, sr_trials_std=0.1, n_trials=1, T=100, skew=0.0, kurt=3.0)


def test_pbo_high_for_all_noise_trials():
    rng = np.random.default_rng(0)
    assert pbo(rng.standard_normal((400, 200)), s=8) > 0.35


def test_pbo_low_for_one_dominant_trial():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((400, 50)) * 0.1
    M[:, 0] += 1.0
    assert pbo(M, s=8) < 0.1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_stats.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.stats'`

- [ ] **Step 3: Implement**

`eval/stats.py`:
```python
"""Deflated Sharpe Ratio (Bailey & López de Prado 2014) + PBO via CSCV (NormalDist)."""
from __future__ import annotations
import numpy as np
from math import e
from itertools import combinations
from statistics import NormalDist

_N = NormalDist(0.0, 1.0)
_GAMMA = 0.5772156649015329


def deflated_sharpe(*, sr_hat, sr_trials_std, n_trials, T, skew, kurt) -> float:
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2 for the multiple-testing benchmark")
    sr0 = sr_trials_std * ((1 - _GAMMA) * _N.inv_cdf(1 - 1.0 / n_trials)
                           + _GAMMA * _N.inv_cdf(1 - 1.0 / (n_trials * e)))
    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4.0) * sr_hat ** 2))
    z = (sr_hat - sr0) * np.sqrt(max(T - 1, 1)) / denom
    return float(_N.cdf(z))


def pbo(pnl_matrix: np.ndarray, *, s: int = 8) -> float:
    """CSCV PBO over a (n_obs x n_trials) matrix; columns are distinct strategy configs."""
    M = np.asarray(pnl_matrix, float)
    if M.shape[1] < 2:
        raise ValueError("PBO needs >= 2 trial configs (columns)")
    blocks = np.array_split(np.arange(M.shape[0]), s)
    logits = []
    for tr in combinations(range(s), s // 2):
        te = [b for b in range(s) if b not in tr]
        is_perf = M[np.concatenate([blocks[b] for b in tr])].mean(0)
        oos_perf = M[np.concatenate([blocks[b] for b in te])].mean(0)
        best = int(np.argmax(is_perf))
        rank = min(max((oos_perf <= oos_perf[best]).mean(), 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    return float((np.array(logits) < 0).mean())
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_stats.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**
```bash
git add eval/stats.py tests/test_stats.py
git commit -m "feat: Deflated Sharpe + PBO (consumed at study level)"
```

---

## Task 5: τ estimator (E1.1)

**Files:** Create `eval/tau.py`, `tests/test_tau.py`

*(Unchanged from rev. 1.)*

- [ ] **Step 1: Failing test**

`tests/test_tau.py`:
```python
import numpy as np
from eval.tau import predictivity_curve, estimate_tau


def test_recovers_known_decay():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.standard_normal(n)
    horizons = [1, 2, 5, 10, 20, 40, 80]
    rets = {h: np.exp(-h / 10.0) * x + rng.standard_normal(n) for h in horizons}
    curve = predictivity_curve(x, rets)
    assert curve[1] > curve[80]
    assert 5 <= estimate_tau(curve, frac=1 / np.e) <= 20
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_tau.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.tau'`

- [ ] **Step 3: Implement**

`eval/tau.py`:
```python
"""Decay-window tau: predictive R^2 of a feature vs forward returns across horizons."""
from __future__ import annotations
import numpy as np


def predictivity_curve(feature, returns_by_h: dict) -> dict:
    f = np.asarray(feature, float)
    return {h: float(np.corrcoef(f, np.asarray(r, float))[0, 1] ** 2)
            for h, r in returns_by_h.items()}


def estimate_tau(curve: dict, *, frac: float = 0.3679) -> float:
    hs = sorted(curve); vals = [curve[h] for h in hs]
    thresh = frac * max(vals)
    for i in range(1, len(hs)):
        if vals[i] < thresh <= vals[i - 1]:
            x0, x1, y0, y1 = hs[i - 1], hs[i], vals[i - 1], vals[i]
            return float(x0 + (x1 - x0) * (y0 - thresh) / (y0 - y1 + 1e-12))
    return float(hs[-1])
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_tau.py -v`
Expected: PASS (1 passed)

- [ ] **Step 5: Commit**
```bash
git add eval/tau.py tests/test_tau.py
git commit -m "feat: tau decay-window estimator (E1.1)"
```

---

## Task 6: Per-config CPCV evaluation (returns the fold distribution)

**Files:** Create `eval/baseline.py`, `tests/test_baseline.py`

- [ ] **Step 1: Failing test**

`tests/test_baseline.py`:
```python
import numpy as np
from eval.baseline import evaluate_config, CONFIGS
from eval.synthetic import make_matrix


def test_returns_fold_distribution_not_collapsed():
    df, feats, _ = make_matrix(n=3000, signal_strength=3.0, seed=2)
    r = evaluate_config(df, feats, "lgbm_reg", n_groups=6, k=2, embargo_ns=0)
    from math import comb
    assert len(r.fold_sharpes) == comb(6, 2)          # one Sharpe per CPCV fold
    assert r.per_sample_pnl.shape[0] == len(df)        # per-sample OOS PnL kept
    assert np.isfinite(r.per_sample_pnl).any()


def test_classifier_config_runs_and_signs_forecast():
    df, feats, _ = make_matrix(n=3000, signal_strength=4.0, seed=5)
    r = evaluate_config(df, feats, "lgbm_clf", n_groups=5, k=1, embargo_ns=0)
    assert r.name == "lgbm_clf" and r.net_pnl is not None


def test_naive_makes_no_trades():
    df, feats, _ = make_matrix(n=2000, signal_strength=3.0, seed=6)
    r = evaluate_config(df, feats, "naive", n_groups=5, k=1, embargo_ns=0)
    assert r.n_trades == 0 and r.net_pnl == 0.0


def test_uniqueness_weight_is_passed_to_fit(monkeypatch):
    df, feats, _ = make_matrix(n=1500, signal_strength=3.0, seed=7)
    seen = {}
    import eval.baseline as B
    real = B._fit_predict
    def spy(model, Xtr, ytr, ltr, Xte, wtr, scale):
        seen["w_sum"] = float(np.asarray(wtr).sum()); return real(model, Xtr, ytr, ltr, Xte, wtr, scale)
    monkeypatch.setattr(B, "_fit_predict", spy)
    B.evaluate_config(df, feats, "ridge", n_groups=4, k=1, embargo_ns=0)
    assert seen["w_sum"] > 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_baseline.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.baseline'`

- [ ] **Step 3: Implement**

`eval/baseline.py`:
```python
"""Baseline ladder configs + per-config CPCV evaluation (keeps the fold distribution)."""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import matthews_corrcoef
import lightgbm as lgb
from data.cv import cpcv_splits
from eval.cost import net_pnl, weighted_sharpe

CONFIGS = ("naive", "ridge", "lgbm_reg", "lgbm_clf")


@dataclass
class ConfigResult:
    name: str
    fold_sharpes: np.ndarray        # one OOS Sharpe per CPCV fold (the distribution)
    per_sample_pnl: np.ndarray      # OOS net PnL per sample (nan if never tested)
    mean_fold_sharpe: float         # TRADE-level Sharpe (feeds DSR; gate also requires min_trades)
    sample_sharpe: float            # sample/time-level Sharpe incl. no-trade zeros (honest headline)
    net_pnl: float
    gross_pnl: float                # PnL before costs (gross - net = the cost wall)
    n_trades: int
    turnover: float                 # trades / OOS samples
    t_eff: float                    # effective sample size = Σ uniqueness over trades
    mcc: float                      # sign(forecast) vs sign(realized) over trades
    skew: float
    kurt: float


def _fit_predict(model, Xtr, ytr, ltr, Xte, wtr, scale):
    if model == "naive":
        return np.zeros(len(Xte))
    if model == "ridge":
        return Ridge(alpha=1.0).fit(Xtr, ytr, sample_weight=wtr).predict(Xte)
    if model == "lgbm_reg":
        m = lgb.LGBMRegressor(n_estimators=200, num_leaves=31, learning_rate=0.05,
                              min_child_samples=50, subsample=0.8, verbose=-1)
        m.fit(Xtr, ytr, sample_weight=wtr)
        return m.predict(Xte)
    if model == "lgbm_clf":
        if len(np.unique(ltr)) < 2:        # single-class fold (heavy-flat regime) -> no trades
            return np.zeros(len(Xte))
        m = lgb.LGBMClassifier(n_estimators=200, num_leaves=31, learning_rate=0.05,
                               min_child_samples=50, subsample=0.8, verbose=-1)
        m.fit(Xtr, ltr, sample_weight=wtr)
        proba = m.predict_proba(Xte); cls = list(m.classes_)
        p_up = proba[:, cls.index(1)] if 1 in cls else 0.0
        p_dn = proba[:, cls.index(-1)] if -1 in cls else 0.0
        return (p_up - p_dn) * scale          # signed score -> bps
    raise ValueError(f"unknown model {model!r}")


def evaluate_config(matrix: pd.DataFrame, feature_cols, model: str, *,
                    n_groups: int, k: int, embargo_ns: int) -> ConfigResult:
    X = matrix[feature_cols].to_numpy(float)
    y = matrix["y_fwd_bps"].to_numpy(float)
    lab = matrix["label"].to_numpy(int)
    cost = matrix["cost_bps"].to_numpy(float)
    half = matrix["half_spread_bps"].to_numpy(float)
    w = matrix["uniqueness"].to_numpy(float)
    te_t, t0, t1 = (matrix["t_event"].to_numpy(), matrix["t_event"].to_numpy(),
                    matrix["t_barrier"].to_numpy())
    acc_fc = np.zeros(len(matrix)); cnt = np.zeros(len(matrix)); fold_sharpes = []
    for tr, te in cpcv_splits(te_t, t0, t1, n_groups=n_groups, k=k, embargo_ns=embargo_ns):
        scale = float(y[tr].std() + 1e-9)
        fc = _fit_predict(model, X[tr], y[tr], lab[tr], X[te], w[tr], scale)
        fpnl, _, _ = net_pnl(fc, y[te], cost_bps=cost[te], half_spread_bps=half[te])
        fold_sharpes.append(weighted_sharpe(fpnl, w[te]))
        acc_fc[te] += fc; cnt[te] += 1
    seen = cnt > 0
    fc_ps = np.full(len(matrix), np.nan); fc_ps[seen] = acc_fc[seen] / cnt[seen]
    # Per-sample PnL from the averaged OOS forecast -> consistent pnl/gross/traded/mcc.
    pnl_s, traded_s, gross_s = net_pnl(fc_ps[seen], y[seen],
                                       cost_bps=cost[seen], half_spread_bps=half[seen])
    per_sample = np.full(len(matrix), np.nan); per_sample[seen] = pnl_s
    n_tr = int(traded_s.sum())
    t_eff = float(w[seen][traded_s].sum())
    pred_sign = np.sign(fc_ps[seen][traded_s]).astype(int)
    real_sign = np.sign(y[seen][traded_s]).astype(int)
    mcc = float(matthews_corrcoef(real_sign, pred_sign)) if n_tr > 1 and len(np.unique(pred_sign)) > 1 else 0.0
    pnl_traded = pnl_s[traded_s]
    sample_sharpe = weighted_sharpe(pnl_s, w[seen], trade_only=False)  # incl. no-trade zeros
    fs = np.array(fold_sharpes, float)
    return ConfigResult(
        name=model, fold_sharpes=fs, per_sample_pnl=per_sample,
        mean_fold_sharpe=float(fs.mean()), sample_sharpe=sample_sharpe,
        net_pnl=float(np.nansum(per_sample)), gross_pnl=float(gross_s.sum()), n_trades=n_tr,
        turnover=float(n_tr / max(int(seen.sum()), 1)), t_eff=t_eff, mcc=mcc,
        skew=float(pd.Series(pnl_traded).skew()) if n_tr > 2 else 0.0,
        kurt=float(pd.Series(pnl_traded).kurtosis() + 3.0) if n_tr > 3 else 3.0,
    )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_baseline.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**
```bash
git add eval/baseline.py tests/test_baseline.py
git commit -m "feat: per-config CPCV eval keeping fold-Sharpe distribution + classifier rung"
```

---

## Task 7: Study — trial ledger, study-level DSR/PBO, per-regime G1 gate

**Files:** Create `eval/study.py`, `tests/test_study.py`

- [ ] **Step 1: Failing test**

`tests/test_study.py`:
```python
import numpy as np
from eval.study import run_study
from eval.synthetic import make_matrix


def test_study_reports_regimes_and_dsr_pbo():
    df, feats, lb = make_matrix(n=4000, signal_strength=4.0, seed=8)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert "tight" in out["per_regime"] and "wide" in out["per_regime"]
    g = out["best"]
    assert {"dsr", "pbo", "net_pnl", "name"} <= g.keys()


def test_embargo_must_cover_lookback():
    import pytest
    df, feats, lb = make_matrix(n=1500, signal_strength=2.0, seed=9)
    with pytest.raises(ValueError, match="embargo"):
        run_study(df, feats, cost_default=None, n_groups=5, k=1,
                  embargo_ns=lb - 1, max_lookback_ns=lb)


def test_min_eff_trades_floor_blocks_pass():
    df, feats, lb = make_matrix(n=8000, signal_strength=6.0, seed=8)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb, min_eff_trades=10**9)
    assert out["g1_pass"] is False


def test_min_sample_sharpe_floor_blocks_pass():
    df, feats, lb = make_matrix(n=8000, signal_strength=6.0, seed=8)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb, min_sample_sharpe=10.0)
    assert out["g1_pass"] is False


def test_winner_is_a_passing_candidate():
    df, feats, lb = make_matrix(n=8000, signal_strength=6.0, seed=8)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    if out["g1_pass"]:
        assert out["winner"] is not None
        assert out["rungs"][out["winner"]]["passes_solo"] is True


def test_per_regime_handles_nonrange_index():
    df, feats, lb = make_matrix(n=4000, signal_strength=4.0, seed=8)
    df.index = df["t_event"].to_numpy()           # non-range (timestamp) index
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert set(out["per_regime"]) == {"tight", "wide"}
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_study.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'eval.study'`

- [ ] **Step 3: Implement**

`eval/study.py`:
```python
"""Study: run the ladder as TRIAL CONFIGS, compute DSR (best config vs trial dispersion,
with an EFFECTIVE sample size) and PBO (CSCV over configs x OOS-sample matrix), and the
G1 gate (FAIL-CLOSED when PBO is unavailable) with per-regime breakdown and gross/net."""
from __future__ import annotations
import numpy as np
import pandas as pd
from eval.matrix import validate_matrix
from eval.baseline import evaluate_config, CONFIGS
from eval.cost import weighted_sharpe
from eval.stats import deflated_sharpe, pbo


def run_study(matrix: pd.DataFrame, feature_cols, *, cost_default, n_groups: int, k: int,
              embargo_ns: int, max_lookback_ns: int, configs=CONFIGS, extra_trials: int = 0,
              dsr_thresh: float = 0.95, pbo_thresh: float = 0.5, min_trades: int = 30,
              min_eff_trades: float = 10.0, min_sample_sharpe: float = 0.0):
    validate_matrix(matrix, feature_cols)
    if embargo_ns < max_lookback_ns:
        raise ValueError(f"embargo_ns ({embargo_ns}) must cover max_lookback_ns ({max_lookback_ns})")

    results = {c: evaluate_config(matrix, feature_cols, c, n_groups=n_groups, k=k,
                                  embargo_ns=embargo_ns) for c in configs}
    naive = results["naive"]
    candidates = [r for c, r in results.items() if c != "naive"]

    # DSR per config: trade-level Sharpe vs across-trial dispersion; T = effective trade
    # count. The pre-registered min_trades floor guards against few-trade flukes.
    trial_sharpes = np.array([r.mean_fold_sharpe for r in results.values()])
    sr_std = float(trial_sharpes.std() + 1e-9)
    n_trials = max(2, len(results) + extra_trials)
    dsr_by = {r.name: deflated_sharpe(sr_hat=r.mean_fold_sharpe, sr_trials_std=sr_std,
                                      n_trials=n_trials, T=max(int(round(r.t_eff)), 2),
                                      skew=r.skew, kurt=r.kurt) for r in results.values()}

    # PBO over the configs x common-OOS-sample matrix (selection overfitting). Fail-closed.
    M = np.column_stack([r.per_sample_pnl for r in results.values()])
    rows = np.isfinite(M).all(axis=1)
    pbo_available = bool(rows.sum() >= 32)
    pbo_val = float(pbo(M[rows], s=8)) if pbo_available else float("nan")

    def _solo(r):  # per-candidate gate (multiple testing handled by DSR n_trials + PBO)
        return bool(r.net_pnl > 0 and dsr_by[r.name] > dsr_thresh
                    and r.n_trades >= min_trades and r.t_eff >= min_eff_trades
                    and r.sample_sharpe >= min_sample_sharpe
                    and r.net_pnl > naive.net_pnl)
    passing = [r for r in candidates if _solo(r)]
    g1 = bool(passing and pbo_available and pbo_val < pbo_thresh)
    g1_inconclusive = bool(passing and not pbo_available)        # would pass but PBO uncomputable
    winner = (max(passing, key=lambda r: r.net_pnl) if passing
              else max(candidates, key=lambda r: r.net_pnl))

    # Per-regime: slice the WINNER's OOS PnL (no refit); sample/time-level Sharpe.
    w = matrix["uniqueness"].to_numpy(float)
    per_regime = {}
    for reg, ii in matrix.groupby("regime").indices.items():   # .indices = positional rows (index-agnostic)
        p = winner.per_sample_pnl[ii]
        per_regime[str(reg)] = {"net_pnl": float(np.nansum(p)),
                                "sample_sharpe": weighted_sharpe(np.nan_to_num(p), w[ii], trade_only=False),
                                "n": int(np.isfinite(p).sum())}

    def _row(r):
        return {"net_pnl": r.net_pnl, "gross_pnl": r.gross_pnl,
                "cost_wall": r.gross_pnl - r.net_pnl, "trade_sharpe": r.mean_fold_sharpe,
                "sample_sharpe": r.sample_sharpe, "dsr": dsr_by[r.name], "n_trades": r.n_trades,
                "turnover": r.turnover, "mcc": r.mcc, "passes_solo": _solo(r)}
    return {
        "g1_pass": g1,
        "g1_inconclusive": g1_inconclusive,
        "winner": winner.name if passing else None,
        "pbo": pbo_val, "pbo_available": pbo_available,
        "best": {"name": winner.name, "net_pnl": winner.net_pnl, "gross_pnl": winner.gross_pnl,
                 "cost_wall": winner.gross_pnl - winner.net_pnl, "sharpe": winner.mean_fold_sharpe,
                 "trade_sharpe": winner.mean_fold_sharpe, "sample_sharpe": winner.sample_sharpe,
                 "dsr": dsr_by[winner.name], "pbo": pbo_val, "turnover": winner.turnover,
                 "mcc": winner.mcc, "n_trades": winner.n_trades},
        "rungs": {r.name: _row(r) for r in results.values()},
        "per_regime": per_regime,
    }
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_study.py -v`
Expected: PASS (6 passed)

- [ ] **Step 5: Commit**
```bash
git add eval/study.py tests/test_study.py
git commit -m "feat: Study trial-ledger DSR/PBO + per-regime G1 gate"
```

---

## Task 8: ⭐ The G1 gate on known-signal vs known-noise

**Files:** Create `tests/test_gate_synthetic.py`

- [ ] **Step 1: Write the test**

`tests/test_gate_synthetic.py`:
```python
from eval.synthetic import make_matrix
from eval.study import run_study


def test_gate_passes_on_planted_signal():
    df, feats, lb = make_matrix(n=8000, signal_strength=6.0, seed=21)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert out["g1_pass"] is True
    assert out["best"]["net_pnl"] > out["rungs"]["naive"]["net_pnl"]
    # edge must be visible in BOTH regimes (or at least the tight-spread one)
    assert out["per_regime"]["tight"]["net_pnl"] > 0


def test_gate_fails_on_pure_noise():
    df, feats, lb = make_matrix(n=8000, signal_strength=0.0, seed=22)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert out["g1_pass"] is False
    assert out["winner"] is None          # no candidate cleared the gate
```

- [ ] **Step 2: Run it**

Run: `.venv/bin/python -m pytest tests/test_gate_synthetic.py -v`
Expected: PASS (2 passed). If the planted-signal case is flaky, raise `signal_strength`; if the noise case passes the gate, tighten `dsr_thresh`/`pbo_thresh` in `run_study`. Record chosen thresholds.

- [ ] **Step 3: Full suite**

Run: `.venv/bin/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 4: Commit**
```bash
git add tests/test_gate_synthetic.py
git commit -m "test: G1 gate passes on signal, fails on noise, reports per-regime"
```

---

## Task 9: Real-feature integration entrypoint (dependency-gated)

**Files:** Create `scripts/run_baseline.py`, `tests/test_baseline_integration.py`

- [ ] **Step 1: Entrypoint**

`scripts/run_baseline.py`:
```python
"""Run the G1 study on a real ModelMatrix parquet (bars E0.3 + labels E0.4 output).

Usage: .venv/bin/python scripts/run_baseline.py model_matrix.parquet feature_manifest.json
The manifest JSON is {"feature_cols": [...], "max_lookback_ns": <int>, "embargo_ns": <int>,
 "gate": {"n_groups": 6, "k": 2, "min_trades": 30, "min_eff_trades": 10, "min_sample_sharpe": 0.0,
          "dsr_thresh": 0.95, "pbo_thresh": 0.5}}. The gate block is pre-registered and echoed in output.
"""
import sys, json
import pandas as pd
from eval.study import run_study

def main(matrix_path, manifest_path):
    m = pd.read_parquet(matrix_path)
    man = json.load(open(manifest_path))
    g = man.get("gate", {})  # pre-registered gate parameters
    for h, sub in m.groupby("horizon"):
        out = run_study(sub.reset_index(drop=True), man["feature_cols"], cost_default=None,
                        n_groups=g.get("n_groups", 6), k=g.get("k", 2),
                        embargo_ns=man["embargo_ns"], max_lookback_ns=man["max_lookback_ns"],
                        min_trades=g.get("min_trades", 30), min_eff_trades=g.get("min_eff_trades", 10.0),
                        min_sample_sharpe=g.get("min_sample_sharpe", 0.0),
                        dsr_thresh=g.get("dsr_thresh", 0.95), pbo_thresh=g.get("pbo_thresh", 0.5))
        status = "PASS" if out["g1_pass"] else ("INCONCLUSIVE" if out["g1_inconclusive"] else "FAIL")
        print(f"\n=== horizon {h} ===  G1: {status}  (winner={out['winner']}, pbo={out['pbo']:.3f}, gate={g})")
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

- [ ] **Step 2: SKIP-guarded integration test**

`tests/test_baseline_integration.py`:
```python
import pathlib, json, pytest
import pandas as pd
from eval.study import run_study

MATRIX = pathlib.Path("data/processed/model_matrix.parquet")
MANIFEST = pathlib.Path("data/processed/feature_manifest.json")

pytestmark = pytest.mark.skipif(not (MATRIX.exists() and MANIFEST.exists()),
    reason="needs real ModelMatrix + manifest from bars (E0.3) + labels (E0.4)")

def test_real_matrix_runs_through_study():
    m = pd.read_parquet(MATRIX); man = json.load(open(MANIFEST))
    out = run_study(m, man["feature_cols"], cost_default=None, n_groups=6, k=2,
                    embargo_ns=man["embargo_ns"], max_lookback_ns=man["max_lookback_ns"])
    assert "g1_pass" in out and out["per_regime"]
```

- [ ] **Step 3: Run (SKIP until real matrix exists)**

Run: `.venv/bin/python -m pytest tests/test_baseline_integration.py -v`
Expected: SKIP (or PASS once the matrix + manifest exist).

- [ ] **Step 4: Commit**
```bash
git add scripts/run_baseline.py tests/test_baseline_integration.py
git commit -m "feat: real-feature G1 study entrypoint (manifest-driven, per-regime)"
```

---

## Self-review (coverage of the review findings)

- **#1 (CPCV/DSR/PBO)** → `evaluate_config` returns `fold_sharpes` (distribution) + `per_sample_pnl`; `study.run_study` computes DSR from the **trial-config Sharpe dispersion** and PBO from the **configs × OOS-sample matrix** — no faked time-bin trials. ✓
- **#2 (purge)** → per-test-interval purge; `test_noncontiguous_combo_keeps_substantial_train` guards the empty-train regression. ✓
- **#3 (cost)** → per-sample `cost_bps`/`half_spread_bps` in the contract; `net_pnl` uses per-sample cost. ✓
- **#4 (timing)** → `t_feature_start`/`t_available`/`max_lookback_ns`; `embargo_ns` required and **must cover `max_lookback_ns`** (`run_study` raises otherwise). ✓
- **#5 (uniqueness)** → `uniqueness` column → `sample_weight` in every fit + `weighted_sharpe`. ✓
- **#6 (regime)** → `run_study` returns `per_regime`; the gate test asserts the tight-spread regime is positive. ✓
- **#7 (classifier)** → `lgbm_clf` rung (proba→signed bps). ✓
- **#8 (manifest)** → `RESERVED` registry + explicit `feature_cols`; `validate_matrix` rejects reserved-in-manifest and unknown features. ✓

**rev. 3 review (second round):**
- **#1 spread** → `net_pnl` charges `cost_bps + 2×half_spread_bps`; `test_spread_charged_via_two_crossings`. ✓
- **#2 actionable time** → `validate_matrix` enforces `t_available == t_event`; `test_baseline_requires_synchronous_t_available`. ✓
- **#3 PBO fail-open** → gate fails closed; `g1_inconclusive` when PBO uncomputable. ✓
- **#4 DSR T** → `T = round(Σ uniqueness over trades)` (effective sample size). ✓
- **#5 clf single-class** → `lgbm_clf` returns zeros on one-class folds. ✓
- **#6 gross/turnover/MCC** → added to `ConfigResult` + study `best`/`rungs` output. ✓

**rev. 4 review (third round):**
- **#1 trade-only Sharpe** → `weighted_sharpe(trade_only=...)`; `ConfigResult.sample_sharpe` reported alongside trade-level; gate adds pre-registered `min_trades` floor (default 30). ✓
- **#2 CLI** → entrypoint prints gross, cost_wall, turnover, MCC, trade/sample Sharpe, and PASS/FAIL/INCONCLUSIVE. ✓
- **#3 best-before-gate** → per-candidate `_solo` gate; G1 passes if any non-naive config clears; `winner` is `None` when none do. ✓
- **#4 stale counts** → test_matrix (6), test_cost (5) corrected. ✓

Note on **#1**: because DSR scales by √T, trade-level-Sharpe×√(n_trades) ≈ sample-level-Sharpe×√(n_all), so switching the metric barely moves the DSR z-score — the `min_trades` floor is the real guard against few-trade flukes, and the sample-level Sharpe is reported for honest sizing.

**rev. 5 review (fourth round):**
- **#1 sample-Sharpe floor** → `_solo` requires `sample_sharpe >= min_sample_sharpe` (default 0.0, configurable); G1 is now hit-quality **and** capital-experience. ✓
- **#2 effective-trade floor** → `_solo` also requires `t_eff >= min_eff_trades` (default 10), not just raw `n_trades`. ✓
- **#3 index-agnostic regimes** → per-regime uses `groupby(...).indices` (positional). New `test_per_regime_handles_nonrange_index`. ✓
- **#4 pre-registered gate** → manifest `gate{}` block read + echoed by the entrypoint. ✓
- Added tests: `test_min_eff_trades_floor_blocks_pass`, `test_min_sample_sharpe_floor_blocks_pass`, `test_winner_is_a_passing_candidate` (test_study now 6). ✓

**Still honestly deferred (called out, not silent):**
- **PBO power is weak with only 4 ladder configs** — it becomes meaningful when `extra_trials`/the config set grows during sweeps; wire a persistent study-wide trial ledger then. The hook (`extra_trials`, configs list) is in place.
- **DSR `sr_trials_std` from 4 configs is a noisy benchmark estimate** — same remedy (more configs in the ledger). The point estimate × √T term dominates for large-T real edges, so the gate is still informative.
- **Live-safety is *contract-checked*, not *enforced***: `t_available`/`t_feature_start` are validated for ordering, but verifying features actually respect them is the bars/recon job (E0.1/E0.3), asserted there.

---

## Execution handoff

Plan complete and saved to `docs/superpowers/plans/2026-06-22-lightgbm-baseline.md`. Two execution options:

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. Tasks 0–8 are self-contained (synthetic); only Task 9 depends on the bars/labels pipeline.
2. **Inline Execution** — execute here with checkpoints.

Which approach?
