"""G0-BN development statistics and deterministic selection (issue #88, slice 67-B).

Implements the development halves of spec sections 3.4, 4.3, 8.2 and 8.3
(docs/superpowers/specs/2026-07-13-g0bn-protocol.md):

- the section-8.2 decision/realized cost split: the no-trade mask uses only costs
  observable or frozen at decision time (frozen fee, frozen base slippage, observable
  half-spread, no-trade margin) while realized latency drift is charged after the
  decision; both component-reconciliation identities are verified per row under the
  binding math.isclose policy `rel_tol=1e-12, abs_tol=1e-12`. The legacy
  `eval.cost.net_pnl` is never called — it uses its supplied cost in both the mask
  and the charge, which the spec forbids for G0-BN;
- uniqueness-weighted trade Sharpe (weighted mean / weighted population standard
  deviation over traded rows) with explicit degeneracy reasons, and the exact DSR
  sample count `T = max(2, int(numpy.rint(effective_trades)))` under
  `nearest_ties_to_even_int64_v1`;
- the versioned G0-BN DSR input assembly (same-horizon scored-trial Sharpe
  dispersion + 1e-9, complete cross-horizon unique-identity ledger count, traded-PnL
  sample skew and Pearson kurtosis) feeding the Bailey/Lopez de Prado formula
  (`eval.stats.deflated_sharpe`, pure reuse);
- the exact G0-BN PBO: 8 contiguous `numpy.array_split` blocks over the common
  chronologically ordered CPCV-OOS net matrix, uniqueness-weighted block means,
  canonical base-ladder-then-ascending-trial-id columns, `first_max_v1` IS ties,
  `less_equal_count_v1` OOS ranks over `n_columns + 1`, strict `logit < 0`, and
  availability only with >= 2 columns and >= 32 rows;
- the section-8.3 paired circular two-day moving-block bootstrap (PCG64 seed 0,
  10,000 replicates, linear percentiles) instantiated on the frozen development
  included-day list, with the pinned draw-matrix hash, used at the development
  Bonferroni level `alpha_dev = 0.05/8` for the lift and mean-daily-net lower
  bounds; and
- deterministic development selection (spec section 4.3): trade-first then
  predictive-only, unrounded tuple ranking, earlier-ladder-order ties, 60s
  structurally control-only, and a freeze-blocking explicit failure when either
  primary horizon has no predictive-eligible candidate.

Selection consumes forecasts only after re-verifying them against the pinned
append-only ledger result hashes, so a tampered or substituted collapsed series
fails closed instead of leaking into eligibility.
"""
from __future__ import annotations

import hashlib
import math
from itertools import combinations

import numpy as np
import pandas as pd

from eval.g0bn_config import (
    BOOTSTRAP_DRAW_SCHEMA,
    BOOTSTRAP_KIND,
    CANDIDATE_IDS,
    DSR_ROUNDING_RULE,
    FORECAST_COLLAPSE_VERSION,
    PBO_IS_TIE_RULE,
    PBO_OOS_RANK_RULE,
    SELECTABLE_CANDIDATE_IDS,
    _fail,
    validate_protocol_config,
)
from eval.g0bn_engine import RESULT_SCHEMA, forecast_series_sha256
from eval.g0bn_identity import base_trial_identities, trial_id as _trial_id
from eval.hashing import canonical_json, hash_obj, split_hash
from eval.manifest import manifest_sha256
from eval.stats import deflated_sharpe
import eval.stats
from eval.writer import logical_row_sha256, ordered_manifest_columns

DEVELOPMENT_RESULT_SCHEMA = "g0bn-development-result-v1"
PBO_INPUT_SCHEMA = "g0bn-pbo-input-v1"

ALPHA_DEV = 0.05 / 8
N_BOOT = 10_000
BLOCK_LENGTH_DAYS = 2
BOOTSTRAP_SEED = 0
PBO_N_BLOCKS = 8
PBO_MIN_COLUMNS = 2
PBO_MIN_ROWS = 32

_DAY_NS = 86_400_000_000_000
_INT64_MAX = np.iinfo(np.int64).max

_N_GROUPS = 6
_K_TEST_GROUPS = 2


def _float(x) -> float:
    return float(np.float64(x))


def _source_sha256(*files) -> str:
    digest = hashlib.sha256()
    for path in files:
        with open(path, "rb") as f:
            digest.update(f.read())
    return digest.hexdigest()


def g0bn_dsr_code_sha256() -> str:
    """Runtime identity of the DSR implementation: this module (input assembly,
    trade Sharpe, T rounding) plus eval.stats (the Bailey/Lopez de Prado formula).
    The config's `cv.dsr.code_sha256` must pin exactly this value; a modified
    metric implementation fails closed instead of scoring under stale provenance."""
    return _source_sha256(__file__, eval.stats.__file__)


def g0bn_pbo_code_sha256() -> str:
    """Runtime identity of the G0-BN PBO implementation (this module)."""
    return _source_sha256(__file__)


# ------------------------------------------------------------------ cost split (8.2)

def decision_costs(rows: pd.DataFrame, config: dict) -> dict:
    """Split per-row costs into the decision-time band and the realized charge, and
    verify both section-8.2 reconciliation identities under the binding 1e-12
    binary64 policy. Fails closed on any missing, negative, non-finite, or
    inconsistent component."""
    assumption = config["costs"]["cost_assumption"]
    for name, value in (("taker_fee_bps", assumption["taker_fee_bps"]),
                        ("base_slippage_bps", assumption["base_slippage_bps"]),
                        ("no_trade_margin_bps",
                         config["costs"]["no_trade_margin_bps"])):
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not math.isfinite(value) or value < 0:
            _fail(name, f"must be a finite non-negative number; got {value!r}")
    fee_bps = 2.0 * float(assumption["taker_fee_bps"])
    decision_cost_bps = fee_bps + float(assumption["base_slippage_bps"])
    drift = rows["latency_drift_bps"].to_numpy(np.float64)
    half_spread = rows["half_spread_bps"].to_numpy(np.float64)
    cost = rows["cost_bps"].to_numpy(np.float64)
    if not np.isfinite(drift).all():
        _fail("latency_drift_bps", "must be finite (required non-feature diagnostic)")
    if (drift < 0).any():
        _fail("latency_drift_bps", "must be non-negative")
    if not np.isfinite(half_spread).all() or (half_spread < 0).any():
        _fail("half_spread_bps", "must be finite and non-negative")
    if not np.isfinite(cost).all() or (cost < 0).any():
        _fail("cost_bps", "must be finite and non-negative")

    def _isclose(a, b):
        # Vectorized math.isclose policy: rel_tol=1e-12, abs_tol=1e-12 (spec 8.2).
        return np.abs(a - b) <= np.maximum(
            1e-12 * np.maximum(np.abs(a), np.abs(b)), 1e-12)

    expected_cost = decision_cost_bps + drift
    if not _isclose(cost, expected_cost).all():
        _fail("cost_bps", "does not reconcile with decision_cost_bps + "
                          "latency_drift_bps (= 2*taker_fee_bps + base_slippage_bps "
                          "+ latency_drift_bps) under rel_tol=abs_tol=1e-12")
    spread_bps = 2.0 * half_spread
    decision_total = decision_cost_bps + spread_bps
    realized_total = cost + spread_bps
    # The second section-8.2 identity. With realized_total DERIVED from cost_bps
    # (development has no independent realized-total column), this reduces
    # algebraically to the first identity; it is kept as written so the 67-F OOS
    # scorer, which reads independent columns, inherits the exact two-check shape.
    if not _isclose(realized_total, decision_total + drift).all():
        _fail("realized_total_cost_bps", "does not reconcile with "
                                         "decision_total_cost_bps + latency_drift_bps")
    return {
        "fee_bps": fee_bps,
        "decision_cost_bps": decision_cost_bps,
        "spread_bps": spread_bps,
        "decision_total_cost_bps": decision_total,
        "realized_total_cost_bps": realized_total,
        "no_trade_margin_bps": float(config["costs"]["no_trade_margin_bps"]),
    }


def trade_economics(forecasts, rows: pd.DataFrame, config: dict) -> dict:
    """Section-8.2 trade rule on one collapsed CPCV-OOS forecast series: the mask
    compares |f| to the decision-time band only; realized drift is charged to net
    after the decision and can never alter the mask."""
    f = np.asarray(forecasts, dtype=np.float64)
    if f.shape != (len(rows),):
        _fail("forecasts", f"must supply exactly one forecast per row "
                           f"({len(rows)}); got shape {f.shape}")
    if not np.isfinite(f).all():
        _fail("forecasts", "must be finite")
    costs = decision_costs(rows, config)
    y = rows["y_fwd_bps"].to_numpy(np.float64)
    weights = rows["uniqueness"].to_numpy(np.float64)
    band = costs["decision_total_cost_bps"] + costs["no_trade_margin_bps"]
    traded = np.abs(f) > band
    signed = np.sign(f) * y
    gross = np.where(traded, signed, 0.0)
    net = np.where(traded, signed - costs["realized_total_cost_bps"], 0.0)
    return {
        "traded": traded,
        "gross": gross,
        "net": net,
        "n_trades": int(traded.sum()),
        "effective_trades": _float((weights * traded).sum()),
    }


# ------------------------------------------------------------- Sharpe / lift / DSR

def weighted_trade_sharpe(net, traded, weights):
    """Unannualized uniqueness-weighted trade Sharpe over traded rows: weighted mean
    divided by weighted POPULATION standard deviation. Degenerate series report
    (0.0, reason) exactly as spec section 8.2 requires."""
    net = np.asarray(net, dtype=np.float64)
    traded = np.asarray(traded, dtype=bool)
    w = np.asarray(weights, dtype=np.float64)
    p, ww = net[traded], w[traded]
    if p.size < 2:
        return 0.0, "fewer_than_two_traded_rows"
    total = ww.sum()
    if not (total > 0):
        return 0.0, "zero_weight_sum"
    mean = float(np.average(p, weights=ww))
    var = float(np.average((p - mean) ** 2, weights=ww))
    if not (var > 0):
        return 0.0, "zero_weighted_variance"
    return float(mean / math.sqrt(var)), None


def weighted_lift(y, f, u):
    """The binding section-8.1 persistence lift `L = 1 - weighted_SSE_model /
    weighted_SSE_zero` on the original rows. Returns (L, None) or (None, reason)
    when the persistence denominator is zero/non-finite (INCONCLUSIVE)."""
    y = np.asarray(y, dtype=np.float64)
    f = np.asarray(f, dtype=np.float64)
    u = np.asarray(u, dtype=np.float64)
    sse_zero = float(np.sum(u * y * y))
    sse_model = float(np.sum(u * (y - f) ** 2))
    if not math.isfinite(sse_zero) or sse_zero == 0.0:
        return None, "zero_persistence_denominator"
    lift = 1.0 - sse_model / sse_zero
    if not math.isfinite(lift):
        return None, "nonfinite_lift"
    return float(lift), None


def dsr_sample_count(effective_trades):
    """`T = max(2, int(numpy.rint(effective_trades)))` — round-to-nearest with exact
    halves to even, represented as int64 (rule `nearest_ties_to_even_int64_v1`).
    Returns (T, unrounded effective trades)."""
    et = np.float64(effective_trades)
    if not np.isfinite(et) or et < 0:
        raise ValueError(f"effective_trades must be finite and non-negative; "
                         f"got {effective_trades!r}")
    t_rounded = np.rint(et)
    if not (0 <= t_rounded <= _INT64_MAX):
        raise ValueError(f"rounded effective_trades {t_rounded!r} is outside the "
                         "signed int64 range")
    return max(2, int(np.int64(t_rounded))), float(et)


def _sample_skew_kurt(traded_net: np.ndarray):
    """Traded-PnL sample skew and Pearson (non-excess) kurtosis, with the legacy
    small-sample conventions (0.0 / 3.0) so degenerate series stay finite."""
    n = traded_net.size
    skew = float(pd.Series(traded_net).skew()) if n > 2 else 0.0
    kurt = float(pd.Series(traded_net).kurtosis() + 3.0) if n > 3 else 3.0
    if not math.isfinite(skew):
        skew = 0.0
    if not math.isfinite(kurt):
        kurt = 3.0
    return skew, kurt


# --------------------------------------------------------------------------- PBO

def pbo_input_sha256(matrix: np.ndarray, weights: np.ndarray, trial_ids) -> str:
    header = canonical_json({"schema": PBO_INPUT_SCHEMA,
                             "column_trial_ids": list(trial_ids),
                             "n_rows": int(matrix.shape[0]),
                             "n_columns": int(matrix.shape[1]),
                             "dtype": "<f8"})
    digest = hashlib.sha256()
    digest.update(header.encode())
    digest.update(b"\n")
    digest.update(np.ascontiguousarray(matrix, dtype="<f8").tobytes(order="C"))
    digest.update(np.ascontiguousarray(weights, dtype="<f8").tobytes(order="C"))
    return digest.hexdigest()


def g0bn_pbo(net_matrix, weights, column_trial_ids) -> dict:
    """Exact G0-BN CSCV PBO (spec section 3.4) over the common chronologically
    ordered CPCV-OOS net matrix. Columns must already be in canonical order (the
    five base identities in ladder order, then ascending trial ids); this function
    records the tie rules and the canonical input hash as provenance.

    A non-finite entry RAISES rather than returning available=False: only
    successfully scored identities enter PBO and their nets are finite by
    construction (finite collapsed forecasts, finite y, reconciled costs), so a
    non-finite value here is upstream tampering/integrity failure, not a
    small-sample availability condition."""
    M = np.asarray(net_matrix, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    ids = list(column_trial_ids)
    if M.ndim != 2 or M.shape[1] != len(ids) or len(set(ids)) != len(ids):
        _fail("pbo", "net matrix columns must match the unique canonical trial-id "
                     "list")
    if M.shape[0] != w.shape[0]:
        _fail("pbo", "weights must supply one uniqueness value per row")
    if not np.isfinite(M).all():
        _fail("pbo", "net matrix must be finite in every column (only successfully "
                     "scored identities enter PBO)")
    if not np.isfinite(w).all() or (w <= 0).any():
        _fail("pbo", "weights must be finite and positive")
    n_rows, n_cols = M.shape
    out = {
        "available": False,
        "value": None,
        "reason": None,
        "n_rows": int(n_rows),
        "n_columns": int(n_cols),
        "n_combinations": None,
        "n_blocks": PBO_N_BLOCKS,
        "is_tie_rule": PBO_IS_TIE_RULE,
        "oos_rank_rule": PBO_OOS_RANK_RULE,
        "column_order_rule": "base_ladder_then_ascending_trial_id_v1",
        "input_sha256": pbo_input_sha256(M, w, ids),
    }
    if n_cols < PBO_MIN_COLUMNS:
        out["reason"] = "fewer_than_2_columns"
        return out
    if n_rows < PBO_MIN_ROWS:
        out["reason"] = "fewer_than_32_rows"
        return out
    blocks = np.array_split(np.arange(n_rows), PBO_N_BLOCKS)
    below = 0
    total = 0
    for train_blocks in combinations(range(PBO_N_BLOCKS), PBO_N_BLOCKS // 2):
        test_blocks = [b for b in range(PBO_N_BLOCKS) if b not in train_blocks]
        train_rows = np.concatenate([blocks[b] for b in train_blocks])
        test_rows = np.concatenate([blocks[b] for b in test_blocks])
        is_mean = np.average(M[train_rows], axis=0, weights=w[train_rows])
        oos_mean = np.average(M[test_rows], axis=0, weights=w[test_rows])
        j_star = int(np.argmax(is_mean))                    # first_max_v1
        rank_count = int((oos_mean <= oos_mean[j_star]).sum())  # less_equal_count_v1
        rank = rank_count / (n_cols + 1)
        logit = math.log(rank / (1.0 - rank))
        below += int(logit < 0.0)                           # strictly below zero
        total += 1
    out["available"] = True
    out["value"] = below / total
    out["n_combinations"] = total
    return out


# ---------------------------------------------------------------- bootstrap (8.3)

def bootstrap_draws(days):
    """One derived day-index draw matrix for the whole development stage (spec
    section 8.3): PCG64 seed 0, 10,000 replicates of ceil(D/2) circular two-day
    blocks, concatenated in draw order and truncated to D. Returns (draw matrix,
    pinned draw hash)."""
    days = list(days)
    n_days = len(days)
    if n_days == 0:
        _fail("bootstrap", "requires a non-empty sorted day list")
    if sorted(set(days)) != days:
        _fail("bootstrap", "day list must be sorted unique canonical days")
    n_blocks = math.ceil(n_days / 2)
    rng = np.random.Generator(np.random.PCG64(BOOTSTRAP_SEED))
    starts = rng.integers(0, n_days, size=(N_BOOT, n_blocks), endpoint=False,
                          dtype=np.int64)
    expanded = np.empty((N_BOOT, 2 * n_blocks), dtype=np.int64)
    expanded[:, 0::2] = starts
    expanded[:, 1::2] = (starts + 1) % n_days
    draw = np.ascontiguousarray(expanded[:, :n_days])
    header = canonical_json({"block_length_days": BLOCK_LENGTH_DAYS, "days": days,
                             "dtype": "<i8", "schema": BOOTSTRAP_DRAW_SCHEMA,
                             "seed": BOOTSTRAP_SEED, "shape": [N_BOOT, n_days]})
    digest = hashlib.sha256()
    digest.update(header.encode())
    digest.update(b"\n")
    digest.update(np.ascontiguousarray(draw, dtype="<i8").tobytes(order="C"))
    return draw, digest.hexdigest()


def one_sided_lower_bound(replicates, alpha: float) -> float:
    """`Q(alpha)` with numpy's linear interpolation and no rounding (spec 8.3)."""
    return float(np.quantile(np.asarray(replicates, dtype=np.float64), alpha,
                             method="linear"))


def day_positions(t_event_ns, days) -> np.ndarray:
    """Map each row's UTC day to its position in the frozen included-day list;
    a row on a day outside the list fails closed."""
    index = {d: i for i, d in enumerate(days)}
    day_ints = np.asarray(t_event_ns, dtype=np.int64) // _DAY_NS
    dates = np.datetime_as_string(
        (np.datetime64("1970-01-01", "D") + day_ints.astype("timedelta64[D]")))
    positions = np.array([index.get(str(d), -1) for d in dates], dtype=np.int64)
    if (positions < 0).any():
        bad = sorted({str(d) for d, p in zip(dates, positions) if p < 0})
        _fail("bootstrap", f"rows on days outside the frozen included-day list: "
                           f"{bad[:5]}")
    return positions


def _day_sums(day_pos: np.ndarray, n_days: int, values: np.ndarray) -> np.ndarray:
    out = np.zeros(n_days, dtype=np.float64)
    np.add.at(out, day_pos, values)
    return out


def _lift_replicates(draw, day_pos, n_days, y, f, u):
    """Paired lift replicates: aggregate per-day A_d and (A_d - B_d) over the drawn
    days with multiplicity and recompute L (model and persistence are never
    resampled separately). Returns (replicates | None, reason | None)."""
    a_d = _day_sums(day_pos, n_days, u * y * y)
    amb_d = _day_sums(day_pos, n_days, u * (y * y - (y - f) ** 2))
    denom = a_d[draw].sum(axis=1)
    numer = amb_d[draw].sum(axis=1)
    if (denom == 0.0).any() or not np.isfinite(denom).all():
        return None, "zero_or_nonfinite_resampled_persistence_denominator"
    replicates = numer / denom
    if not np.isfinite(replicates).all():
        return None, "nonfinite_lift_replicates"
    return replicates, None


def _daily_mean_replicates(draw, day_pos, n_days, row_values):
    """Gross/net replicates: aggregate row-level daily sums with multiplicity and
    divide by exactly D (zero-trade and empty days included)."""
    per_day = _day_sums(day_pos, n_days, row_values)
    replicates = per_day[draw].sum(axis=1) / float(n_days)
    if not np.isfinite(replicates).all():
        return None, "nonfinite_net_replicates"
    return replicates, None


# ------------------------------------------------------------------- selection

def rank_selectable(entries, *, mode: str):
    """Deterministic descending ranking on unrounded values with earlier-ladder-
    order ties (spec 4.3). `trade` ranks by (net_lb, lift_lb, point_net,
    point_lift); `predictive` by (lift_lb, point_lift). Input order never matters."""
    if mode == "trade":
        keys = ("net_lower_bound", "lift_lower_bound", "point_net", "point_lift")
    elif mode == "predictive":
        keys = ("lift_lower_bound", "point_lift")
    else:
        raise ValueError(f"unknown selection mode {mode!r}")
    return sorted(entries, key=lambda e: tuple(-float(e[k]) for k in keys)
                  + (int(e["ladder_index"]),))


def _candidate_reasons(sufficiency, pbo_block, lift, lift_reason, lift_lower,
                       lift_valid_reason, base_complete):
    reasons = []
    if not base_complete:
        reasons.append("incomplete_base_ladder")
    if sufficiency["n_valid_days"] < sufficiency["min_valid_days"]:
        reasons.append("insufficient_valid_days")
    if sufficiency["uniqueness_sum"] < sufficiency["min_uniqueness_sum"]:
        reasons.append("insufficient_uniqueness_sum")
    if not pbo_block["available"]:
        reasons.append("pbo_unavailable")
    elif not (pbo_block["value"] < pbo_block["threshold"]):
        reasons.append("pbo_not_below_threshold")
    if lift is None:
        reasons.append(f"lift_point_invalid:{lift_reason}")
    if lift_valid_reason is not None:
        reasons.append(f"lift_bootstrap_invalid:{lift_valid_reason}")
    elif lift_lower is not None and not (lift_lower > 0.0):
        reasons.append("lift_lower_bound_not_positive")
    return reasons


def _trade_reasons(metrics, dsr, thresholds):
    reasons = []
    if metrics["net_lower_bound"] is None:
        reasons.append("net_bootstrap_invalid")
    elif not (metrics["net_lower_bound"] > 0.0):
        reasons.append("net_lower_bound_not_positive")
    if metrics["n_trades"] < thresholds["min_trades"]:
        reasons.append("insufficient_trades")
    if not (metrics["effective_trades"] >= thresholds["min_effective_trades"]):
        reasons.append("insufficient_effective_trades")
    if not (dsr > thresholds["dsr_threshold"]):
        reasons.append("dsr_not_above_threshold")
    return reasons


def development_selection(run, *, extra_forecasts: dict | None = None,
                          generated_at: str | None = None) -> dict:
    """Compute the complete deterministic development result from one engine run and
    its pinned ledger: per-horizon metrics with DSR/PBO provenance, and the
    trade-first / predictive-only selection for the two primary horizons.

    `extra_forecasts` supplies collapsed row series for additional (non-base)
    successfully scored identities in the ledger; they enter effective N, the DSR
    dispersion, and the PBO columns, but can never be selected (spec section 4.2:
    recordable, immutable, counted — not eligible)."""
    config = validate_protocol_config(run.config)
    # Fail fast on metric-implementation drift: the pinned DSR/PBO code hashes
    # must match the RUNNING implementations before any statistic is computed.
    running_dsr = g0bn_dsr_code_sha256()
    if config["cv"]["dsr"]["code_sha256"] != running_dsr:
        _fail("cv.dsr.code_sha256",
              f"config pins {config['cv']['dsr']['code_sha256']} but the running "
              f"DSR implementation hashes to {running_dsr}")
    running_pbo = g0bn_pbo_code_sha256()
    if config["cv"]["pbo"]["code_sha256"] != running_pbo:
        _fail("cv.pbo.code_sha256",
              f"config pins {config['cv']['pbo']['code_sha256']} but the running "
              f"PBO implementation hashes to {running_pbo}")
    ledger = run.ledger
    extra_forecasts = dict(extra_forecasts or {})
    data_identity = run.data_identity
    thresholds = config["verdict_thresholds"]
    roles = {h["tag"]: h["role"] for h in config["horizons"]}
    ladder_tags = [h["tag"] for h in config["horizons"]]
    primary_tags = [t for t in ladder_tags if roles[t] == "primary"]

    overlap = set(extra_forecasts) & set(run.forecasts)
    if overlap:
        _fail("extra_forecasts", f"may not override the run's own forecasts: "
                                 f"{sorted(overlap)[:3]}")
    unknown = [tid for tid in extra_forecasts if tid not in ledger.trial_ids()]
    if unknown:
        _fail("extra_forecasts", f"trials not present in the ledger: {unknown[:3]}")

    # Re-bind the rows this call actually scores to the pinned development data
    # identity. The ledger forecast hashes cover only (t_event, forecast) bytes,
    # so a carried DevelopmentRun with mutated reserved columns (y_fwd_bps,
    # costs, uniqueness) would otherwise pass every ledger check and corrupt
    # lift/net/DSR/PBO and the freeze choice.
    if manifest_sha256(run.manifest) != data_identity["development_manifest_sha256"]:
        _fail("manifest", "the run's manifest does not match the pinned "
                          "development manifest hash")
    scored_frame = pd.concat([run.horizon_rows[tag] for tag in ladder_tags],
                             ignore_index=True)
    scored_lrh = logical_row_sha256(scored_frame,
                                    ordered_manifest_columns(run.manifest))
    if scored_lrh != data_identity["development_logical_row_sha256"]:
        _fail("horizon_rows",
              f"the rows entering selection hash to logical rows {scored_lrh}, "
              f"not the pinned development_logical_row_sha256 "
              f"{data_identity['development_logical_row_sha256']} — refusing to "
              "score mutated or foreign rows")

    # ---- reconcile every successfully scored ladder-horizon trial with the ledger
    scored_by_tag: dict = {tag: [] for tag in ladder_tags}
    forecast_map: dict = {}
    for tid in ledger.scored_trial_ids():
        identity = ledger.identity_for(tid)
        for key in ("development_dataset_id", "development_build_id",
                    "development_manifest_sha256", "development_logical_row_sha256",
                    "partition_plan_sha256"):
            if identity[key] != data_identity[key]:
                _fail("ledger", f"scored trial {tid[:12]}... binds a foreign "
                                f"development data identity ({key})")
        if identity["protocol_config_sha256"] != config["sha256"]:
            _fail("ledger", f"scored trial {tid[:12]}... binds a foreign protocol "
                            "config")
        tag = identity["horizon"]
        if tag not in scored_by_tag:
            continue    # off-ladder horizon: counts in effective N only
        rows = run.horizon_rows[tag]
        forecasts = run.forecasts.get(tid)
        if forecasts is None:
            forecasts = extra_forecasts.get(tid)
        if forecasts is None:
            _fail("forecasts", f"successfully scored trial {tid[:12]}... at horizon "
                               f"{tag} has no collapsed forecast series; selection "
                               "requires the common row series for every scored "
                               "identity")
        forecasts = np.asarray(forecasts, dtype=np.float64)
        if forecasts.shape != (len(rows),):
            _fail("forecasts", f"trial {tid[:12]}... does not cover the common row "
                               f"universe at {tag} (expected {len(rows)} rows, got "
                               f"{forecasts.shape})")
        if not np.isfinite(forecasts).all():
            _fail("forecasts", f"trial {tid[:12]}... forecasts must be finite")
        result = ledger.result_for(tid)
        if result.get("schema") != RESULT_SCHEMA:
            _fail("ledger", f"scored trial {tid[:12]}... carries result schema "
                            f"{result.get('schema')!r}; only {RESULT_SCHEMA!r} "
                            "results may enter selection")
        if result.get("collapse_version") != FORECAST_COLLAPSE_VERSION:
            _fail("ledger", f"scored trial {tid[:12]}... was collapsed under "
                            f"{result.get('collapse_version')!r}; section 3.4 "
                            f"mandates {FORECAST_COLLAPSE_VERSION!r} for every "
                            "series entering lift/net/DSR/PBO")
        if result.get("n_rows") != len(rows):
            _fail("forecasts", f"trial {tid[:12]}... result n_rows does not match "
                               f"the common row universe at {tag}")
        expected = result.get("forecasts_sha256")
        actual = forecast_series_sha256(rows["t_event"].to_numpy(np.int64), tag,
                                        forecasts)
        if actual != expected:
            _fail("forecasts", f"trial {tid[:12]}... forecasts do not reproduce the "
                               "pinned ledger result hash (tampered or substituted "
                               "series)")
        forecast_map[tid] = forecasts
        scored_by_tag[tag].append(tid)

    # ---- base-ladder reconciliation against the ledger
    base_ids = base_trial_identities(config, data_identity)
    base_tid_by = {(i["horizon"], i["candidate_id"]): _trial_id(i)
                   for i in base_ids}
    missing_base = [tid for tid in base_tid_by.values()
                    if tid not in ledger.trial_ids()]
    if missing_base:
        _fail("ledger", f"the complete 15-trial base ladder must be registered "
                        f"(completed or aborted); missing {missing_base[:3]}")

    n_trials = ledger.n_effective_trials()
    days = list(config["exclusions"]["included_days"])
    draw, draw_sha256 = bootstrap_draws(days)

    horizons_out: dict = {}
    selection_out: dict = {}
    for tag in ladder_tags:
        rows = run.horizon_rows[tag]
        y = rows["y_fwd_bps"].to_numpy(np.float64)
        weights = rows["uniqueness"].to_numpy(np.float64)
        day_pos = day_positions(rows["t_event"].to_numpy(np.int64), days)
        n_valid_days = int(np.unique(day_pos).size)
        uniqueness_sum = _float(weights.sum())
        sufficiency = {
            "n_valid_days": n_valid_days,
            "uniqueness_sum": uniqueness_sum,
            "min_valid_days": thresholds["min_valid_days"],
            "min_uniqueness_sum": thresholds["min_uniqueness_sum"],
            "sufficient": bool(
                n_valid_days >= thresholds["min_valid_days"]
                and uniqueness_sum >= thresholds["min_uniqueness_sum"]),
        }
        horizon_split_sha256 = split_hash(rows, n_groups=_N_GROUPS,
                                          k=_K_TEST_GROUPS,
                                          embargo_ns=config["cv"]["embargo_ns"])

        # canonical scored order: base ladder first, then ascending trial id
        base_scored = [base_tid_by[(tag, cid)] for cid in CANDIDATE_IDS
                       if base_tid_by[(tag, cid)] in forecast_map]
        other_scored = sorted(tid for tid in scored_by_tag[tag]
                              if tid not in set(base_scored))
        ordered_scored = base_scored + other_scored
        base_complete = len(base_scored) == len(CANDIDATE_IDS)

        economics = {tid: trade_economics(forecast_map[tid], rows, config)
                     for tid in ordered_scored}
        sharpe_by = {}
        for tid in ordered_scored:
            econ = economics[tid]
            sharpe_by[tid] = weighted_trade_sharpe(econ["net"], econ["traded"],
                                                   weights)
        sharpes = np.asarray([sharpe_by[tid][0] for tid in ordered_scored],
                             dtype=np.float64)
        # Spec 3.4: the population std of the FINITE trade Sharpes at this horizon.
        # weighted_trade_sharpe reports degenerate series as a finite 0.0, so the
        # filter is a no-op today; it is applied literally so a future degenerate
        # representation cannot silently change the DSR benchmark.
        finite_sharpes = sharpes[np.isfinite(sharpes)]
        sr_trials_std = _float(np.std(finite_sharpes, dtype=np.float64) + 1e-9) \
            if ordered_scored else None

        pbo_block = dict(g0bn_pbo(
            np.column_stack([economics[tid]["net"] for tid in ordered_scored]),
            weights, ordered_scored)) if ordered_scored else {
            "available": False, "value": None, "reason": "no_scored_trials",
            "n_rows": 0, "n_columns": 0, "n_combinations": None,
            "n_blocks": PBO_N_BLOCKS, "is_tie_rule": PBO_IS_TIE_RULE,
            "oos_rank_rule": PBO_OOS_RANK_RULE,
            "column_order_rule": "base_ladder_then_ascending_trial_id_v1",
            "input_sha256": None}
        pbo_block.update({
            "column_trial_ids": ordered_scored,
            "threshold": thresholds["pbo_threshold"],
            "ledger_sha256": ledger.ledger_sha256(),
            "split_sha256": horizon_split_sha256,
            "code_sha256": config["cv"]["pbo"]["code_sha256"],
        })

        candidates_out: dict = {}
        selectable_entries = []
        for ladder_index, cid in enumerate(CANDIDATE_IDS):
            tid = base_tid_by[(tag, cid)]
            if tid not in forecast_map:
                candidates_out[cid] = {
                    "trial_id": tid, "candidate_id": cid,
                    "ladder_index": ladder_index, "scored": False,
                    "abort_error": None if ledger.result_for(tid) is not None
                    else "aborted_or_unscored",
                }
                continue
            forecasts = forecast_map[tid]
            econ = economics[tid]
            sharpe, sharpe_reason = sharpe_by[tid]
            lift, lift_reason = weighted_lift(y, forecasts, weights)
            lift_reps, lift_invalid = _lift_replicates(draw, day_pos, len(days), y,
                                                       forecasts, weights)
            lift_lower = (one_sided_lower_bound(lift_reps, ALPHA_DEV)
                          if lift_invalid is None else None)
            net_reps, net_invalid = _daily_mean_replicates(draw, day_pos, len(days),
                                                           econ["net"])
            net_lower = (one_sided_lower_bound(net_reps, ALPHA_DEV)
                         if net_invalid is None else None)
            gross_reps, _ = _daily_mean_replicates(draw, day_pos, len(days),
                                                   econ["gross"])
            mean_daily_net = _float(econ["net"].sum() / len(days))
            mean_daily_gross = _float(econ["gross"].sum() / len(days))
            skew, kurt = _sample_skew_kurt(econ["net"][econ["traded"]])
            T, effective_unrounded = dsr_sample_count(econ["effective_trades"])
            dsr = float(deflated_sharpe(sr_hat=sharpe,
                                        sr_trials_std=sr_trials_std,
                                        n_trials=n_trials, T=T, skew=skew,
                                        kurt=kurt))
            metrics = {
                "lift": lift,
                "lift_reason": lift_reason,
                "lift_lower_bound": lift_lower,
                "lift_bootstrap_reason": lift_invalid,
                "mean_daily_net_bps": mean_daily_net,
                "mean_daily_gross_bps": mean_daily_gross,
                "net_lower_bound": net_lower,
                "net_bootstrap_reason": net_invalid,
                "net_bps_sum": _float(econ["net"].sum()),
                "gross_bps_sum": _float(econ["gross"].sum()),
                "n_trades": econ["n_trades"],
                "effective_trades": econ["effective_trades"],
            }
            candidate = {
                "trial_id": tid,
                "candidate_id": cid,
                "ladder_index": ladder_index,
                "scored": True,
                "trade_sharpe": sharpe,
                "trade_sharpe_reason": sharpe_reason,
                "skew": skew,
                "kurt": kurt,
                "dsr": dsr,
                "dsr_provenance": {
                    "n_trials": n_trials,
                    "n_trials_source": "g0bn_ledger_unique_identity_count_v1",
                    "ledger_sha256": ledger.ledger_sha256(),
                    "sr_trials_std": sr_trials_std,
                    "same_horizon_scored_trial_ids": ordered_scored,
                    "effective_trades": effective_unrounded,
                    "T": T,
                    "rounding_rule": DSR_ROUNDING_RULE,
                    "epsilon": 1e-9,
                    "threshold": thresholds["dsr_threshold"],
                    "code_sha256": config["cv"]["dsr"]["code_sha256"],
                },
                "metrics": metrics,
            }
            if roles[tag] == "primary" and cid in SELECTABLE_CANDIDATE_IDS:
                reasons = _candidate_reasons(sufficiency, pbo_block, lift,
                                             lift_reason, lift_lower, lift_invalid,
                                             base_complete)
                predictive = not reasons
                trade_only_reasons = _trade_reasons(metrics, dsr, thresholds)
                trade = predictive and not trade_only_reasons
                candidate["predictive_eligible"] = predictive
                candidate["trade_eligible"] = trade
                candidate["reasons"] = reasons + (trade_only_reasons
                                                  if predictive else [])
                if predictive:
                    selectable_entries.append({
                        "candidate_id": cid,
                        "trial_id": tid,
                        "ladder_index": ladder_index,
                        "trade_eligible": trade,
                        "net_lower_bound": net_lower if net_lower is not None
                        else float("-inf"),
                        "lift_lower_bound": lift_lower,
                        "point_net": mean_daily_net,
                        "point_lift": lift,
                    })
            candidates_out[cid] = candidate

        horizons_out[tag] = {
            "role": roles[tag],
            "n_rows": int(len(rows)),
            "split_sha256": horizon_split_sha256,
            "sufficiency": sufficiency,
            "scored_trial_ids": ordered_scored,
            "other_scored_trial_ids": other_scored,
            "candidates": candidates_out,
            "pbo": pbo_block,
        }

        if roles[tag] != "primary":
            continue
        trade_entries = [e for e in selectable_entries if e["trade_eligible"]]
        if trade_entries:
            ranked = rank_selectable(trade_entries, mode="trade")
            selection_out[tag] = {
                "mode": "trade",
                "selected_candidate_id": ranked[0]["candidate_id"],
                "selected_trial_id": ranked[0]["trial_id"],
                "ranked_candidate_ids": [e["candidate_id"] for e in ranked],
            }
        elif selectable_entries:
            ranked = rank_selectable(selectable_entries, mode="predictive")
            selection_out[tag] = {
                "mode": "predictive",
                "selected_candidate_id": ranked[0]["candidate_id"],
                "selected_trial_id": ranked[0]["trial_id"],
                "ranked_candidate_ids": [e["candidate_id"] for e in ranked],
            }
        else:
            selection_out[tag] = {
                "mode": None,
                "selected_candidate_id": None,
                "selected_trial_id": None,
                "ranked_candidate_ids": [],
                "reason": "no_predictive_eligible_candidate",
            }

    freeze_blocked = any(selection_out[tag]["selected_candidate_id"] is None
                         for tag in primary_tags)
    result = {
        "schema": DEVELOPMENT_RESULT_SCHEMA,
        "protocol_id": config["protocol_id"],
        "pilot_id": config["pilot_id"],
        "protocol_config_sha256": config["sha256"],
        "data_identity": dict(data_identity),
        "ledger": {
            "n_effective_trials": n_trials,
            "ledger_sha256": ledger.ledger_sha256(),
            "history_sha256": ledger.history_sha256(),
            "identity_set_sha256": ledger.identity_set_sha256(),
        },
        "bootstrap": {
            "kind": BOOTSTRAP_KIND,
            "block_length_days": BLOCK_LENGTH_DAYS,
            "n_boot": N_BOOT,
            "seed": BOOTSTRAP_SEED,
            "bit_generator": "PCG64",
            "percentile_method": "linear",
            "alpha_dev": ALPHA_DEV,
            "days": days,
            "draw_sha256": draw_sha256,
        },
        "selection_rules": dict(config["selection"]),
        "horizons": horizons_out,
        "selection": selection_out,
        "freeze_blocked": freeze_blocked,
        "generated_at": generated_at,
    }
    result["result_sha256"] = hash_obj(result,
                                       exclude_keys=("result_sha256",
                                                     "generated_at"))
    return result
