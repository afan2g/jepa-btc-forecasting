"""G0-BN development statistics and deterministic selection tests (issue #88, 67-B).

Covers spec section 11 item 5 (selection half): the collapsed row series feeding
lift/net/DSR/PBO, trade-first then predictive-only selection, exact unrounded tie
breaks, DSR `T` nearest-ties-to-even cases, canonical PBO column order, first-maximum
IS and less-than-or-equal OOS tie cases, Bonferroni lower bounds, 60s inability to
select/pass/rescue, and DSR/PBO provenance from the pinned ledger. Plus the section
8.2 decision/realized cost split with the drift-causality check, and the section 8.3
development bootstrap with the pinned draw recipe. Synthetic bundles only.
"""
from __future__ import annotations

import copy
import hashlib
import math
from itertools import combinations

import numpy as np
import pandas as pd
import pytest

from eval.g0bn_engine import (
    DevelopmentRun,
    forecast_series_sha256,
    run_g0bn_development,
)
from eval.g0bn_ledger import G0BNLedger
from eval.g0bn_selection import (
    ALPHA_DEV,
    N_BOOT,
    bootstrap_draws,
    decision_costs,
    development_selection,
    dsr_sample_count,
    g0bn_pbo,
    one_sided_lower_bound,
    trade_economics,
    weighted_lift,
    weighted_trade_sharpe,
)
from eval.hashing import canonical_json, hash_obj
from eval.stats import deflated_sharpe
from tests.g0bn_dev_fixtures import (
    dev_bundle,
    dev_config,
    dev_data_identity,
    dev_manifest,
    dev_matrix,
)


# --------------------------------------------------------------------- DSR T rounding

def test_dsr_sample_count_nearest_ties_to_even():
    # numpy.rint: round-to-nearest, exact halves to the nearest EVEN integer.
    assert dsr_sample_count(2.5)[0] == 2
    assert dsr_sample_count(3.5)[0] == 4
    assert dsr_sample_count(8.5)[0] == 8
    assert dsr_sample_count(9.5)[0] == 10
    assert dsr_sample_count(10.49)[0] == 10
    assert dsr_sample_count(10.51)[0] == 11


def test_dsr_sample_count_floors_at_two_and_keeps_unrounded_value():
    T, unrounded = dsr_sample_count(0.0)
    assert (T, unrounded) == (2, 0.0)
    T, unrounded = dsr_sample_count(1.2)
    assert (T, unrounded) == (2, 1.2)
    T, unrounded = dsr_sample_count(147.9)
    assert (T, unrounded) == (148, 147.9)


def test_dsr_sample_count_rejects_invalid_inputs():
    for bad in (-1.0, float("nan"), float("inf"), 1e20):
        with pytest.raises(ValueError):
            dsr_sample_count(bad)


# ------------------------------------------------------------------ decision costs

def _cost_rows(n=4, fee=4.5, slip=0.5, drift=None, half=None):
    drift = np.full(n, 0.25) if drift is None else np.asarray(drift, float)
    half = np.full(n, 0.4) if half is None else np.asarray(half, float)
    return pd.DataFrame({
        "cost_bps": 2.0 * fee + slip + drift,
        "half_spread_bps": half,
        "latency_drift_bps": drift,
    })


def _cost_config(fee=4.5, slip=0.5, margin=0.25):
    return {"costs": {"cost_assumption": {"taker_fee_bps": fee,
                                          "base_slippage_bps": slip},
                      "no_trade_margin_bps": margin}}


def test_decision_costs_reconcile_and_split_decision_from_realized():
    rows = _cost_rows()
    costs = decision_costs(rows, _cost_config())
    assert costs["fee_bps"] == 9.0
    assert costs["decision_cost_bps"] == 9.5
    assert np.allclose(costs["spread_bps"], 0.8)
    assert np.allclose(costs["decision_total_cost_bps"], 10.3)
    assert np.allclose(costs["realized_total_cost_bps"], 9.5 + 0.25 + 0.8)


def test_decision_costs_fail_closed_on_reconciliation_break():
    rows = _cost_rows()
    rows.loc[1, "cost_bps"] += 1e-6
    with pytest.raises(ValueError, match="reconcil"):
        decision_costs(rows, _cost_config())


def test_decision_costs_fail_closed_on_bad_components():
    rows = _cost_rows()
    rows.loc[0, "latency_drift_bps"] = -0.1
    rows.loc[0, "cost_bps"] = 9.5 - 0.1
    with pytest.raises(ValueError, match="latency_drift"):
        decision_costs(rows, _cost_config())
    rows = _cost_rows()
    rows.loc[2, "latency_drift_bps"] = np.inf
    rows.loc[2, "cost_bps"] = np.inf
    with pytest.raises(ValueError, match="finite"):
        decision_costs(rows, _cost_config())


def test_trade_economics_hand_computed():
    rows = _cost_rows(n=3, drift=[0.0, 0.5, 0.0], half=[0.5, 0.5, 0.5])
    rows["y_fwd_bps"] = [20.0, -8.0, 30.0]
    rows["uniqueness"] = [1.0, 0.5, 0.8]
    # band = decision_total + margin = 9.5 + 1.0 + 0.25 = 10.75
    f = np.array([15.0, -12.0, 3.0])
    econ = trade_economics(f, rows, _cost_config())
    assert econ["traded"].tolist() == [True, True, False]
    # gross = sign(f)*y; net = gross - (cost_bps + spread)
    assert econ["gross"].tolist() == [20.0, 8.0, 0.0]
    assert econ["net"][0] == 20.0 - (9.5 + 0.0 + 1.0)
    assert econ["net"][1] == 8.0 - (9.5 + 0.5 + 1.0)
    assert econ["net"][2] == 0.0
    assert econ["n_trades"] == 2
    assert econ["effective_trades"] == 1.5


def test_latency_drift_charges_realized_cost_but_never_the_trade_mask():
    # Spec section 11 item 12 causality direction: mutate ONLY the realized drift
    # (and its dependent cost_bps identity); the mask/count must stay byte-identical
    # while realized net changes.
    base = _cost_rows(n=3, drift=[0.0, 0.0, 0.0], half=[0.5, 0.5, 0.5])
    base["y_fwd_bps"] = [20.0, -8.0, 30.0]
    base["uniqueness"] = [1.0, 0.5, 0.8]
    f = np.array([15.0, -12.0, 3.0])
    drifted = _cost_rows(n=3, drift=[2.0, 3.0, 4.0], half=[0.5, 0.5, 0.5])
    drifted["y_fwd_bps"] = base["y_fwd_bps"]
    drifted["uniqueness"] = base["uniqueness"]
    a = trade_economics(f, base, _cost_config())
    b = trade_economics(f, drifted, _cost_config())
    assert np.array_equal(a["traded"], b["traded"])
    assert a["n_trades"] == b["n_trades"]
    assert b["net"][0] == a["net"][0] - 2.0
    assert b["net"][1] == a["net"][1] - 3.0


# -------------------------------------------------------------------- trade sharpe

def test_weighted_trade_sharpe_hand_computed():
    net = np.array([2.0, -1.0, 3.0, 0.0])
    traded = np.array([True, True, True, False])
    w = np.array([1.0, 0.5, 0.5, 1.0])
    sharpe, reason = weighted_trade_sharpe(net, traded, w)
    p, ww = net[:3], w[:3]
    mean = np.average(p, weights=ww)
    var = np.average((p - mean) ** 2, weights=ww)
    assert reason is None
    assert sharpe == mean / math.sqrt(var)


def test_weighted_trade_sharpe_degeneracy_reasons():
    w = np.ones(3)
    sharpe, reason = weighted_trade_sharpe(np.zeros(3), np.array([True, False, False]), w)
    assert (sharpe, reason) == (0.0, "fewer_than_two_traded_rows")
    net = np.array([2.0, 2.0, 2.0])
    sharpe, reason = weighted_trade_sharpe(net, np.ones(3, bool), w)
    assert (sharpe, reason) == (0.0, "zero_weighted_variance")


# ------------------------------------------------------------------------- lift

def test_weighted_lift_two_forms_agree():
    y = np.array([3.0, -2.0, 1.0, 4.0])
    f = np.array([2.5, -1.0, 0.0, 5.0])
    u = np.array([1.0, 0.5, 0.8, 0.2])
    L, reason = weighted_lift(y, f, u)
    sse_model = float(np.sum(u * (y - f) ** 2))
    sse_zero = float(np.sum(u * y * y))
    assert reason is None
    assert L == pytest.approx(1.0 - sse_model / sse_zero, rel=0, abs=0)
    assert L == pytest.approx(
        float(np.sum(u * (y * y - (y - f) ** 2))) / sse_zero)


def test_weighted_lift_zero_denominator_is_inconclusive():
    L, reason = weighted_lift(np.zeros(3), np.ones(3), np.ones(3))
    assert L is None and reason == "zero_persistence_denominator"


# ---------------------------------------------------------------------- bootstrap

def test_bootstrap_draws_follow_the_pinned_recipe_exactly():
    days = [f"2025-11-{d:02d}" for d in range(1, 6)]      # D=5 (odd)
    draw, draw_sha = bootstrap_draws(days)
    D, M = 5, 3
    rng = np.random.Generator(np.random.PCG64(0))
    starts = rng.integers(0, D, size=(N_BOOT, M), endpoint=False, dtype=np.int64)
    expanded = np.empty((N_BOOT, 2 * M), dtype=np.int64)
    expanded[:, 0::2] = starts
    expanded[:, 1::2] = (starts + 1) % D
    assert draw.shape == (N_BOOT, D)
    assert np.array_equal(draw, expanded[:, :D])
    # Odd D truncates only the SECOND member of the final block.
    assert np.array_equal(draw[:, -1], starts[:, -1])
    header = canonical_json({"block_length_days": 2, "days": days, "dtype": "<i8",
                             "schema": "g0bn-circular-day-bootstrap-v1", "seed": 0,
                             "shape": [N_BOOT, D]})
    digest = hashlib.sha256()
    digest.update(header.encode())
    digest.update(b"\n")
    digest.update(np.ascontiguousarray(draw, dtype="<i8").tobytes(order="C"))
    assert draw_sha == digest.hexdigest()


def test_bootstrap_draws_keep_circular_adjacency():
    days = [f"2025-11-{d:02d}" for d in range(1, 7)]      # D=6 (even)
    draw, _ = bootstrap_draws(days)
    assert np.array_equal(draw[:, 1::2], (draw[:, 0::2] + 1) % 6)


def test_one_sided_lower_bound_is_linear_quantile_at_alpha():
    reps = np.linspace(-1.0, 1.0, 10001)
    assert one_sided_lower_bound(reps, ALPHA_DEV) == np.quantile(
        reps, ALPHA_DEV, method="linear")
    assert ALPHA_DEV == 0.05 / 8


# ---------------------------------------------------------------------------- PBO

def _reference_pbo(M, w, s=8):
    """Spec-literal CSCV reference (section 3.4), written independently: contiguous
    numpy.array_split blocks, uniqueness-weighted block means, first-maximum IS
    (numpy.argmax), less-or-equal OOS rank count over n_columns + 1, strict
    logit < 0."""
    rows = np.arange(M.shape[0])
    blocks = np.array_split(rows, s)
    below = 0
    total = 0
    for train in combinations(range(s), s // 2):
        test = [b for b in range(s) if b not in train]
        tr = np.concatenate([blocks[b] for b in train])
        te = np.concatenate([blocks[b] for b in test])
        is_mean = np.average(M[tr], axis=0, weights=w[tr])
        oos_mean = np.average(M[te], axis=0, weights=w[te])
        j_star = int(np.argmax(is_mean))
        rank_count = int((oos_mean <= oos_mean[j_star]).sum())
        rank = rank_count / (M.shape[1] + 1)
        logit = math.log(rank / (1.0 - rank))
        below += int(logit < 0.0)
        total += 1
    return below / total


def test_pbo_zero_when_is_best_stays_oos_best():
    M = np.column_stack([np.full(40, 1.0), np.full(40, 0.0), np.full(40, -1.0)])
    out = g0bn_pbo(M, np.ones(40), ["a", "b", "c"])
    assert out["available"] is True
    assert out["value"] == 0.0
    assert out["n_combinations"] == 70


def test_pbo_alternating_pattern_matches_spec_reference():
    n = 40
    col0 = np.where(np.arange(n) % 2 == 0, 10.0, -9.0)   # volatile but IS-dominant
    M = np.column_stack([col0, np.full(n, 0.4), np.full(n, 0.3)])
    w = np.ones(n)
    out = g0bn_pbo(M, w, ["a", "b", "c"])
    assert out["value"] == _reference_pbo(M, w)


def test_pbo_matches_spec_reference_on_seeded_matrices():
    rng = np.random.default_rng(17)
    for _ in range(3):
        M = rng.standard_normal((45, 4))
        w = rng.uniform(0.2, 1.0, 45)
        out = g0bn_pbo(M, w, ["a", "b", "c", "d"])
        assert out["value"] == _reference_pbo(M, w)


def test_pbo_rank_exactly_half_is_not_counted():
    # Construct a 3-column case where the IS-best's OOS rank count is exactly 2:
    # rank = 2/4 = 0.5 -> logit = 0.0, which is NOT strictly below zero.
    n = 40
    half = np.arange(n) < n // 2
    col0 = np.where(half, 4.0, -1.0)      # IS-best in combos dominated by first half
    col1 = np.where(half, -4.0, 3.0)
    col2 = np.full(n, -10.0)
    M = np.column_stack([col0, col1, col2])
    w = np.ones(n)
    out = g0bn_pbo(M, w, ["a", "b", "c"])
    assert out["value"] == _reference_pbo(M, w)


def test_pbo_first_max_tie_uses_the_earlier_canonical_column():
    # Columns 0 and 2 are IDENTICAL (exact IS-mean ties in every combination);
    # numpy.argmax must pick the earlier canonical column. Both tie OOS as well, so
    # the rank count includes the equal twin: rank = 3/4 -> logit > 0 -> never
    # counted, PBO = 0. An average-rank or random tie rule would differ.
    col = np.where(np.arange(40) % 3 == 0, 2.0, -0.5)
    M = np.column_stack([col, np.full(40, -1.0), col])
    out = g0bn_pbo(M, np.ones(40), ["a", "b", "c"])
    assert out["value"] == 0.0
    assert out["is_tie_rule"] == "first_max_v1"
    assert out["oos_rank_rule"] == "less_equal_count_v1"


def test_pbo_availability_rules():
    M = np.random.default_rng(0).standard_normal((31, 3))
    out = g0bn_pbo(M, np.ones(31), ["a", "b", "c"])
    assert out["available"] is False and out["value"] is None
    assert out["reason"] == "fewer_than_32_rows"
    out = g0bn_pbo(np.zeros((40, 1)), np.ones(40), ["a"])
    assert out["available"] is False and out["reason"] == "fewer_than_2_columns"


def test_pbo_rejects_non_finite_input():
    M = np.zeros((40, 2))
    M[3, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        g0bn_pbo(M, np.ones(40), ["a", "b"])


# ----------------------------------------------------------- full development runs

@pytest.fixture(scope="module")
def strong_run():
    frame, manifest, config, identity = dev_bundle()
    ledger = G0BNLedger()
    run = run_g0bn_development(frame, manifest, config, identity, ledger)
    return run, development_selection(run)


@pytest.fixture(scope="module")
def weak_run():
    frame, manifest, config, identity = dev_bundle(signal_bps=2.0, noise_bps=0.5,
                                                   seed=23)
    ledger = G0BNLedger()
    run = run_g0bn_development(frame, manifest, config, identity, ledger)
    return run, development_selection(run)


def test_development_result_structure_and_provenance(strong_run):
    run, result = strong_run
    assert result["schema"] == "g0bn-development-result-v1"
    assert result["protocol_config_sha256"] == run.config["sha256"]
    assert result["ledger"]["n_effective_trials"] == 15
    assert result["ledger"]["ledger_sha256"] == run.ledger.ledger_sha256()
    assert result["bootstrap"]["alpha_dev"] == 0.05 / 8
    assert result["bootstrap"]["n_boot"] == N_BOOT
    assert result["bootstrap"]["days"] == run.config["exclusions"]["included_days"]
    assert set(result["horizons"]) == {"2s", "10s", "60s"}
    for tag, block in result["horizons"].items():
        assert set(block["candidates"]) == {"persistence_zero", "microprice_raw",
                                            "ofi_ridge", "lgbm_reg", "lgbm_clf"}
        assert block["split_sha256"]
        assert block["pbo"]["available"] is True
        assert block["pbo"]["ledger_sha256"] == run.ledger.ledger_sha256()
        # canonical PBO columns: the five base identities in ladder order
        base_ids = [block["candidates"][c]["trial_id"] for c in
                    ("persistence_zero", "microprice_raw", "ofi_ridge",
                     "lgbm_reg", "lgbm_clf")]
        assert block["pbo"]["column_trial_ids"] == base_ids
    # verdict-bearing selection exists only for the two primary horizons
    assert set(result["selection"]) == {"2s", "10s"}
    recomputed = hash_obj(result, exclude_keys=("result_sha256", "generated_at"))
    assert result["result_sha256"] == recomputed


def test_dsr_provenance_reconstructs_the_reported_value(strong_run):
    run, result = strong_run
    for tag in ("2s", "10s", "60s"):
        block = result["horizons"][tag]
        sharpes = [block["candidates"][c]["trade_sharpe"]
                   for c in block["candidates"]]
        sr_std = float(np.std(np.asarray(sharpes, dtype=np.float64)) + 1e-9)
        for cand in block["candidates"].values():
            prov = cand["dsr_provenance"]
            assert prov["n_trials"] == 15
            assert prov["sr_trials_std"] == sr_std
            assert prov["rounding_rule"] == "nearest_ties_to_even_int64_v1"
            assert cand["dsr"] == deflated_sharpe(
                sr_hat=cand["trade_sharpe"], sr_trials_std=sr_std,
                n_trials=prov["n_trials"], T=prov["T"],
                skew=cand["skew"], kurt=cand["kurt"])


def test_strong_signal_selects_a_trade_eligible_candidate(strong_run):
    _, result = strong_run
    assert result["freeze_blocked"] is False
    for tag in ("2s", "10s"):
        sel = result["selection"][tag]
        assert sel["mode"] == "trade"
        assert sel["selected_candidate_id"] in ("microprice_raw", "ofi_ridge",
                                                "lgbm_reg", "lgbm_clf")
        chosen = result["horizons"][tag]["candidates"][sel["selected_candidate_id"]]
        assert chosen["trade_eligible"] is True
        assert chosen["metrics"]["n_trades"] >= 30
        assert chosen["metrics"]["net_lower_bound"] > 0
        assert chosen["dsr"] > 0.95


def test_selection_never_considers_persistence_or_60s(strong_run):
    _, result = strong_run
    assert "60s" not in result["selection"]
    for tag in ("2s", "10s"):
        assert result["selection"][tag]["selected_candidate_id"] != "persistence_zero"
        ranked = result["selection"][tag]["ranked_candidate_ids"]
        assert "persistence_zero" not in ranked
    controls = result["horizons"]["60s"]
    for cand in controls["candidates"].values():
        assert "predictive_eligible" not in cand
        assert "trade_eligible" not in cand


def test_weak_signal_falls_back_to_predictive_only(weak_run):
    _, result = weak_run
    assert result["freeze_blocked"] is False
    for tag in ("2s", "10s"):
        sel = result["selection"][tag]
        assert sel["mode"] == "predictive"
        for cand in result["horizons"][tag]["candidates"].values():
            assert cand["metrics"]["n_trades"] == 0
        chosen = result["horizons"][tag]["candidates"][sel["selected_candidate_id"]]
        assert chosen["predictive_eligible"] is True
        assert chosen["trade_eligible"] is False
        assert chosen["metrics"]["lift_lower_bound"] > 0


def test_no_signal_blocks_the_freeze():
    frame, manifest, config, identity = dev_bundle(seed=31)
    rng = np.random.default_rng(99)
    shuffled = frame.copy()
    for tag in ("2s", "10s", "60s"):
        idx = shuffled.index[shuffled["horizon"] == tag].to_numpy()
        y = shuffled.loc[idx, "y_fwd_bps"].to_numpy()
        y = y[rng.permutation(len(y))]
        shuffled.loc[idx, "y_fwd_bps"] = y
        label = np.zeros(len(y), dtype=np.int64)
        label[y > 9.0] = 1
        label[y < -9.0] = -1
        shuffled.loc[idx, "label"] = label
    manifest2 = dev_manifest(config, shuffled)
    identity2 = dev_data_identity(config, manifest2, shuffled)
    run = run_g0bn_development(shuffled, manifest2, config, identity2, G0BNLedger())
    result = development_selection(run)
    assert result["freeze_blocked"] is True
    for tag in ("2s", "10s"):
        sel = result["selection"][tag]
        assert sel["selected_candidate_id"] is None
        assert sel["mode"] is None


def test_insufficient_days_block_eligibility():
    config = dev_config()
    frame = dev_matrix(config, days=config["exclusions"]["included_days"][:10])
    manifest = dev_manifest(config, frame)
    identity = dev_data_identity(config, manifest, frame)
    run = run_g0bn_development(frame, manifest, config, identity, G0BNLedger())
    result = development_selection(run)
    assert result["freeze_blocked"] is True
    for tag in ("2s", "10s"):
        block = result["horizons"][tag]
        assert block["sufficiency"]["sufficient"] is False
        assert block["sufficiency"]["n_valid_days"] == 10
        for cid in ("microprice_raw", "ofi_ridge", "lgbm_reg", "lgbm_clf"):
            cand = block["candidates"][cid]
            assert cand["predictive_eligible"] is False
            assert "insufficient_valid_days" in cand["reasons"]


# ------------------------------------------------------- leakage / integrity guards

def test_selection_rejects_tampered_forecasts(strong_run):
    run, _ = strong_run
    tampered = DevelopmentRun(
        config=run.config, data_identity=run.data_identity,
        horizon_rows=run.horizon_rows, identities=run.identities,
        forecasts=dict(run.forecasts), split_scales=run.split_scales,
        aborted=run.aborted, ledger=run.ledger)
    tid = next(t for t, i in run.identities.items()
               if i["horizon"] == "2s" and i["candidate_id"] == "microprice_raw")
    tampered.forecasts[tid] = run.forecasts[tid] + 100.0   # free performance
    with pytest.raises(ValueError, match="forecast"):
        development_selection(tampered)


def test_selection_rejects_missing_scored_forecasts(strong_run):
    run, _ = strong_run
    crippled = DevelopmentRun(
        config=run.config, data_identity=run.data_identity,
        horizon_rows=run.horizon_rows, identities=run.identities,
        forecasts=dict(run.forecasts), split_scales=run.split_scales,
        aborted=run.aborted, ledger=run.ledger)
    tid = next(iter(crippled.forecasts))
    del crippled.forecasts[tid]
    with pytest.raises(ValueError, match="scored"):
        development_selection(crippled)


def test_selection_rejects_wrong_length_extra_forecasts(strong_run):
    run, _ = strong_run
    base = next(t for t, i in run.identities.items()
                if i["horizon"] == "2s" and i["candidate_id"] == "ofi_ridge")
    variant = dict(run.identities[base], variant="alpha_sweep",
                   variant_params={"alpha": 2.0})
    short = np.zeros(7)
    led = G0BNLedger()
    # replay the real events so the base ladder is present, then add the variant
    frame_rows = run.horizon_rows["2s"]
    for tid, ident in run.identities.items():
        led.record_start(ident)
        led.record_completion(ident, run.ledger.result_for(tid))
    led.record_completion(variant, {
        "schema": "g0bn-trial-result-v1", "n_rows": 7,
        "forecasts_sha256": forecast_series_sha256(
            frame_rows["t_event"].to_numpy(np.int64)[:7], "2s", short),
        "collapse_version": "mean_repeated_test_forecasts_v1",
        "split_scales": None})
    patched = DevelopmentRun(
        config=run.config, data_identity=run.data_identity,
        horizon_rows=run.horizon_rows, identities=run.identities,
        forecasts=run.forecasts, split_scales=run.split_scales,
        aborted=run.aborted, ledger=led)
    with pytest.raises(ValueError, match="common row"):
        development_selection(patched, extra_forecasts={hash_obj(variant): short})


# ---------------------------------------------- recordability without eligibility

def _run_with_variant(run, variant_forecasts):
    """Rebuild the run on a fresh ledger that ALSO contains one completed non-base
    variant trial at 2s with the supplied forecasts."""
    base = next(t for t, i in run.identities.items()
                if i["horizon"] == "2s" and i["candidate_id"] == "ofi_ridge")
    variant = dict(copy.deepcopy(run.identities[base]), variant="alpha_sweep",
                   variant_params={"alpha": 2.0})
    rows = run.horizon_rows["2s"]
    led = G0BNLedger()
    for tid, ident in run.identities.items():
        led.record_start(ident)
        led.record_completion(ident, run.ledger.result_for(tid))
    led.record_completion(variant, {
        "schema": "g0bn-trial-result-v1", "n_rows": int(len(rows)),
        "forecasts_sha256": forecast_series_sha256(
            rows["t_event"].to_numpy(np.int64), "2s", variant_forecasts),
        "collapse_version": "mean_repeated_test_forecasts_v1",
        "split_scales": None})
    patched = DevelopmentRun(
        config=run.config, data_identity=run.data_identity,
        horizon_rows=run.horizon_rows, identities=run.identities,
        forecasts=run.forecasts, split_scales=run.split_scales,
        aborted=run.aborted, ledger=led)
    vid = hash_obj(variant)
    return patched, vid, development_selection(
        patched, extra_forecasts={vid: variant_forecasts})


def test_completed_variant_counts_and_enters_provenance_but_cannot_be_selected(
        strong_run):
    run, baseline_result = strong_run
    rows = run.horizon_rows["2s"]
    perfect = rows["y_fwd_bps"].to_numpy(np.float64).copy()   # oracle forecasts
    patched, vid, result = _run_with_variant(run, perfect)
    assert result["ledger"]["n_effective_trials"] == 16
    block = result["horizons"]["2s"]
    # canonical order: five base ladder ids first, then the variant ascending
    assert block["pbo"]["column_trial_ids"][:5] == \
        baseline_result["horizons"]["2s"]["pbo"]["column_trial_ids"]
    assert block["pbo"]["column_trial_ids"][5:] == [vid]
    assert vid in block["other_scored_trial_ids"]
    # An oracle variant can never be selected: eligibility is base-ladder-only.
    sel = result["selection"]["2s"]
    assert sel["selected_trial_id"] != vid
    assert sel["selected_candidate_id"] in ("microprice_raw", "ofi_ridge",
                                            "lgbm_reg", "lgbm_clf")
    # DSR provenance sees BOTH the bigger ledger count and the variant's sharpe.
    cand = block["candidates"]["microprice_raw"]
    assert cand["dsr_provenance"]["n_trials"] == 16
    assert vid in block["scored_trial_ids"]


def test_aborted_variant_increases_n_trials_and_nothing_else(strong_run):
    run, baseline_result = strong_run
    base = next(t for t, i in run.identities.items()
                if i["horizon"] == "10s" and i["candidate_id"] == "lgbm_reg")
    variant = dict(copy.deepcopy(run.identities[base]), variant="seed_probe",
                   variant_params={"seed": 1})
    led = G0BNLedger()
    for tid, ident in run.identities.items():
        led.record_start(ident)
        led.record_completion(ident, run.ledger.result_for(tid))
    led.record_start(variant)
    led.record_abort(variant, error="killed")
    patched = DevelopmentRun(
        config=run.config, data_identity=run.data_identity,
        horizon_rows=run.horizon_rows, identities=run.identities,
        forecasts=run.forecasts, split_scales=run.split_scales,
        aborted=run.aborted, ledger=led)
    result = development_selection(patched)
    assert result["ledger"]["n_effective_trials"] == 16
    block = result["horizons"]["10s"]
    assert block["pbo"]["column_trial_ids"] == \
        baseline_result["horizons"]["10s"]["pbo"]["column_trial_ids"]
    for cand in block["candidates"].values():
        assert cand["dsr_provenance"]["n_trials"] == 16
    # Selection outcome (which candidate, which mode) is unchanged by the abort.
    assert {t: result["selection"][t]["selected_candidate_id"] for t in ("2s", "10s")} \
        == {t: baseline_result["selection"][t]["selected_candidate_id"]
            for t in ("2s", "10s")}


def test_60s_metrics_cannot_select_or_rescue(strong_run):
    run, baseline_result = strong_run
    led = G0BNLedger()
    forecasts = dict(run.forecasts)
    for tid, ident in run.identities.items():
        led.record_start(ident)
        if ident["horizon"] == "60s":
            rows = run.horizon_rows["60s"]
            oracle = rows["y_fwd_bps"].to_numpy(np.float64) * 100.0
            forecasts[tid] = oracle
            led.record_completion(ident, {
                "schema": "g0bn-trial-result-v1", "n_rows": int(len(rows)),
                "forecasts_sha256": forecast_series_sha256(
                    rows["t_event"].to_numpy(np.int64), "60s", oracle),
                "collapse_version": "mean_repeated_test_forecasts_v1",
                "split_scales": run.split_scales[tid]})
        else:
            led.record_completion(ident, run.ledger.result_for(tid))
    patched = DevelopmentRun(
        config=run.config, data_identity=run.data_identity,
        horizon_rows=run.horizon_rows, identities=run.identities,
        forecasts=forecasts, split_scales=run.split_scales,
        aborted=run.aborted, ledger=led)
    result = development_selection(patched)
    for tag in ("2s", "10s"):
        assert result["selection"][tag]["selected_candidate_id"] == \
            baseline_result["selection"][tag]["selected_candidate_id"]
        assert result["selection"][tag]["mode"] == \
            baseline_result["selection"][tag]["mode"]
        base_block = baseline_result["horizons"][tag]["candidates"]
        for cid, cand in result["horizons"][tag]["candidates"].items():
            assert cand["metrics"] == base_block[cid]["metrics"]
    assert "60s" not in result["selection"]


def test_selection_uses_unrounded_tuples_and_ladder_order_ties():
    from eval.g0bn_selection import rank_selectable
    entries = [
        {"candidate_id": "ofi_ridge", "ladder_index": 2, "trade_eligible": True,
         "net_lower_bound": 1.0 + 1e-13, "lift_lower_bound": 0.5,
         "point_net": 2.0, "point_lift": 0.6},
        {"candidate_id": "lgbm_reg", "ladder_index": 3, "trade_eligible": True,
         "net_lower_bound": 1.0, "lift_lower_bound": 0.5,
         "point_net": 2.0, "point_lift": 0.6},
    ]
    # an unrounded 1e-13 edge decides; displayed values would round to a tie
    assert rank_selectable(entries, mode="trade")[0]["candidate_id"] == "ofi_ridge"
    entries[0]["net_lower_bound"] = 1.0   # exact tie on every tuple element
    assert rank_selectable(entries, mode="trade")[0]["candidate_id"] == "ofi_ridge"
    # earlier ladder order breaks exact ties; swapping input order must not matter
    assert rank_selectable(list(reversed(entries)),
                           mode="trade")[0]["candidate_id"] == "ofi_ridge"
