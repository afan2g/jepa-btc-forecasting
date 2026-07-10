"""Unified matched G0-XV development study: matched-arm fail-closed validation, the
unified DSR/PBO ledger (G0-CB history included), the combined-vs-control noise band,
PBO fail-closed, and the arm-wise-false-pass-caught-by-unified-ledger control."""
import copy

import numpy as np
import pandas as pd
import pytest

import eval.g0 as g0
from eval.g0 import run_g0xv_development
from eval.hashing import hash_obj
from eval.ledger import TrialLedger, trial_identity
from eval.stats import deflated_sharpe
from eval.study import run_study
from eval.synthetic import make_g0_world

GATE = {"n_groups": 4, "k": 2, "min_trades": 5, "min_eff_trades": 3.0}
XV_GATE = {**GATE, "noise_band_n_boot": 500}


def _arms(world, mutate=None):
    arms = []
    for n in ("coinbase_only", "binance_only", "combined"):
        a = world["dev"]["arms"][n]
        arms.append({"name": n, "manifest": copy.deepcopy(a["manifest"]),
                     "matrix": a["matrix"].copy()})
    if mutate:
        mutate({a["name"]: a for a in arms})
    return arms


# ------------------------------------------------------------------ unified ledger/DSR
def test_unified_study_passes_with_imported_history(g0_pipeline):
    res = g0_pipeline["res_xv"]
    assert res["protocol"] == "g0xv-development"
    assert res["development_only"] is True and res["g1_claim"] is False
    assert res["g0xv_dev_pass"] is True
    assert res["winner"] and res["winner"]["arm"] != "coinbase_only"
    # 3 arms x 4 configs = 12 new trials + the 4 imported G0-CB trials
    assert res["ledger"]["n_imported_trials"] == 4
    assert res["ledger"]["n_effective_trials"] == 16
    assert res["ledger"]["ledger_sha256"] == g0_pipeline["led_xv"].ledger_hash()


def test_dsr_reconciles_to_unified_ledger_count(g0_pipeline):
    """Reported DSR must be reproducible from the EXPLICIT ledger: n_trials is the full
    effective count (G0-CB history included), dispersion is the horizon pool's."""
    res, led = g0_pipeline["res_xv"], g0_pipeline["led_xv"]
    pool = res["horizons"]["10s"]["candidates"]
    by_id = {e["identity_sha256"]: e["result"] for e in led.entries()}
    sr_std = float(np.array([r["trade_sharpe"] for r in pool.values()]).std() + 1e-9)
    n = res["ledger"]["n_effective_trials"]
    for cid, row in pool.items():
        lr = by_id[cid]
        expect = deflated_sharpe(sr_hat=row["trade_sharpe"], sr_trials_std=sr_std,
                                 n_trials=max(2, n), T=max(int(round(row["t_eff"])), 2),
                                 skew=lr["skew"], kurt=lr["kurt"])
        assert row["dsr"] == pytest.approx(expect), cid


def test_pbo_candidates_reconcile_to_common_matrix(g0_pipeline):
    """PBO must span EVERY registered G0-XV candidate of the horizon across all arms —
    the common development-OOS candidate-PnL matrix, not per-arm PBO. Matched arms share
    bit-identical naive PnL columns, so the matrix keeps exactly ONE naive benchmark
    (the control arm's) instead of a pass-friendly duplicate per arm."""
    res, led = g0_pipeline["res_xv"], g0_pipeline["led_xv"]
    h = res["horizons"]["10s"]
    xv = [e for e in led.entries()
          if e["identity"]["protocol"] == "g0xv" and e["identity"]["horizon"] == "10s"]
    xv_ids = {e["identity_sha256"] for e in xv}
    pbo_expected = {e["identity_sha256"] for e in xv
                    if e["identity"]["config"] != "naive"
                    or e["identity"]["arm"] == "coinbase_only"}
    assert set(h["pbo_candidates"]) == pbo_expected and len(h["pbo_candidates"]) == 10
    assert h["pbo_available"] is True and np.isfinite(h["pbo"])
    assert set(h["candidates"]) == xv_ids and len(xv_ids) == 12


# --------------------------------------------------------------- matched-arm fail-closed
def test_label_mismatch_across_arms_fails(g0_world):
    def mutate(arms):
        arms["binance_only"]["matrix"].loc[0, "y_fwd_bps"] += 1e-3
    with pytest.raises(ValueError, match="content differs"):
        run_g0xv_development(_arms(g0_world, mutate), g0_world["contract"],
                             gate=XV_GATE, ledger=TrialLedger())


def test_cost_and_regime_mismatch_across_arms_fails(g0_world):
    def mutate(arms):
        arms["combined"]["matrix"].loc[3, "cost_bps"] += 0.1
    with pytest.raises(ValueError, match="content differs"):
        run_g0xv_development(_arms(g0_world, mutate), g0_world["contract"],
                             gate=XV_GATE, ledger=TrialLedger())


def test_missing_rows_in_one_arm_fails(g0_world):
    def mutate(arms):
        arms["binance_only"]["matrix"] = arms["binance_only"]["matrix"].iloc[:-1]
    with pytest.raises(ValueError, match="matched row universe"):
        run_g0xv_development(_arms(g0_world, mutate), g0_world["contract"],
                             gate=XV_GATE, ledger=TrialLedger())


def test_embargo_mismatch_across_arms_fails(g0_world):
    def mutate(arms):
        arms["combined"]["manifest"]["embargo_ns"] *= 2
    with pytest.raises(ValueError, match="share one embargo"):
        run_g0xv_development(_arms(g0_world, mutate), g0_world["contract"],
                             gate=XV_GATE, ledger=TrialLedger())


def test_required_arms_and_unique_names(g0_world):
    arms = _arms(g0_world)
    with pytest.raises(ValueError, match="requires arm 'coinbase_only'"):
        run_g0xv_development(arms[1:], g0_world["contract"], gate=XV_GATE,
                             ledger=TrialLedger())
    with pytest.raises(ValueError, match="requires arm 'combined'"):
        run_g0xv_development(arms[:2], g0_world["contract"], gate=XV_GATE,
                             ledger=TrialLedger())
    # the preregistered binance_only ablation cannot be silently omitted either
    with pytest.raises(ValueError, match="requires arm 'binance_only'"):
        run_g0xv_development([arms[0], arms[2]], g0_world["contract"], gate=XV_GATE,
                             ledger=TrialLedger())
    dup = [arms[0], dict(arms[1], name="coinbase_only"), arms[2]]
    with pytest.raises(ValueError, match="duplicate arm names"):
        run_g0xv_development(dup, g0_world["contract"], gate=XV_GATE,
                             ledger=TrialLedger())


def test_arm_venue_roles_fail_closed(g0_world):
    """A cross-venue build labeled 'coinbase_only' cannot serve as the matched control,
    and a 'cross-venue' arm without a declared signal venue is rejected."""
    def swap_control(arms):
        combined = g0_world["dev"]["arms"]["combined"]
        arms["coinbase_only"]["manifest"] = copy.deepcopy(combined["manifest"])
        arms["coinbase_only"]["matrix"] = combined["matrix"].copy()
    with pytest.raises(ValueError, match="control arm is target-venue-only"):
        run_g0xv_development(_arms(g0_world, swap_control), g0_world["contract"],
                             gate=XV_GATE, ledger=TrialLedger())

    def strip_signal(arms):
        arms["binance_only"]["manifest"]["venues"] = [
            {"exchange": "COINBASE", "symbol": "BTC-USD", "role": "target"}]
    with pytest.raises(ValueError, match="declares no signal venue"):
        run_g0xv_development(_arms(g0_world, strip_signal), g0_world["contract"],
                             gate=XV_GATE, ledger=TrialLedger())


def test_holdout_bound_arm_rejected(g0_world):
    arms = _arms(g0_world)
    hold = g0_world["holdout"]["arms"]["combined"]
    arms[2] = {"name": "combined", "manifest": hold["manifest"], "matrix": hold["matrix"]}
    with pytest.raises(ValueError, match="accepts only 'development'"):
        run_g0xv_development(arms, g0_world["contract"], gate=XV_GATE,
                             ledger=TrialLedger())


def test_april_rows_in_an_arm_rejected(g0_world):
    def mutate(arms):
        apr = g0_world["holdout"]["arms"]["combined"]["matrix"]
        arms["combined"]["matrix"] = pd.concat([arms["combined"]["matrix"], apr.head(1)],
                                               ignore_index=True)
    with pytest.raises(ValueError, match="span-safe"):
        run_g0xv_development(_arms(g0_world, mutate), g0_world["contract"],
                             gate=XV_GATE, ledger=TrialLedger())


def test_missing_partition_contract_binding_fails(g0_world):
    def mutate(arms):
        arms["combined"]["manifest"]["sources"] = ["eval/synthetic.py"]
    with pytest.raises(ValueError, match="exactly one"):
        run_g0xv_development(_arms(g0_world, mutate), g0_world["contract"],
                             gate=XV_GATE, ledger=TrialLedger())


# ------------------------------------------------------------------- gate semantics
def test_pbo_unavailable_fails_closed_even_when_candidates_pass(g0_world, monkeypatch):
    """No PBO verdict => never a pass. Same strong world as the passing session study,
    with the PBO row floor forced out of reach: solo candidates still clear, the combined
    arm still beats control, but the study is blocking/inconclusive — not a pass."""
    monkeypatch.setattr(g0, "_PBO_MIN_ROWS", 10**9)
    res = run_g0xv_development(_arms(g0_world), g0_world["contract"], gate=XV_GATE,
                               ledger=TrialLedger())
    h = res["horizons"]["10s"]
    assert h["pbo_available"] is False and not np.isfinite(h["pbo"])
    assert h["solo_pass_cross_venue"] and h["noise_band"]["beats_control"]
    assert h["pass"] is False and h["inconclusive_blocking"] is True
    assert res["g0xv_dev_pass"] is False and res["inconclusive_blocking"] is True
    assert res["winner"] is None


def test_combined_must_beat_control_beyond_noise_band():
    """bn carries no signal: the combined arm cannot beat the matched Coinbase-only
    control beyond the preregistered bootstrap band, so G0-XV fails even though the
    Coinbase side has real edge."""
    w = make_g0_world(n_dev_bars=300, n_holdout_bars=10, cb_signal=4.0, bn_signal=0.0,
                      seed=1)
    arms = [{"name": n, **w["dev"]["arms"][n]}
            for n in ("coinbase_only", "binance_only", "combined")]
    res = run_g0xv_development(arms, w["contract"], gate=XV_GATE, ledger=TrialLedger())
    band = res["horizons"]["10s"]["noise_band"]
    assert band["beats_control"] is False
    assert band["band_low"] <= 0                     # delta indistinguishable from noise
    assert res["g0xv_dev_pass"] is False


def test_armwise_false_pass_is_caught_by_unified_ledger():
    """The acceptance-criterion control: a separate per-arm study over the Binance arm
    clears the G1-style gate on its own 4-config trial count, but the unified G0-XV
    ledger — carrying the full imported G0-CB search history — deflates the same
    candidate's DSR to failure. Per-arm significance cannot authorize the archive."""
    w = make_g0_world(n_dev_bars=300, n_holdout_bars=10, cb_signal=0.0, bn_signal=2.5,
                      seed=2)
    bn = w["dev"]["arms"]["binance_only"]
    solo = run_study(bn["matrix"], bn["manifest"]["feature_cols"], cost_default=None,
                     embargo_ns=bn["manifest"]["embargo_ns"],
                     max_lookback_ns=bn["manifest"]["max_lookback_ns"], **GATE)
    assert solo["g1_pass"] is True                   # the arm-wise study "passes"
    solo_dsr = solo["rungs"][solo["winner"]]["dsr"]
    assert solo_dsr > 0.95

    prior = TrialLedger()                            # the persisted G0-CB search history
    for i in range(400):
        prior.register(trial_identity(protocol="g0cb", arm="coinbase_only",
                                      dataset_id="d", build_id="hist",
                                      feature_cols=["f"], config="lgbm_reg",
                                      horizon="10s", variant=f"v{i}"), {"net_pnl": 0.0})
    arms = [{"name": n, **w["dev"]["arms"][n]}
            for n in ("coinbase_only", "binance_only", "combined")]
    res = run_g0xv_development(arms, w["contract"], gate=XV_GATE, ledger=TrialLedger(),
                               prior_ledgers=[prior])
    assert res["ledger"]["n_effective_trials"] == 412
    h = res["horizons"]["10s"]
    assert h["solo_pass_cross_venue"] == []          # DSR deflated below the gate
    assert res["g0xv_dev_pass"] is False and res["winner"] is None
    bn_dsrs = [c["dsr"] for c in h["candidates"].values()
               if c["arm"] == "binance_only" and c["config"] == solo["winner"]]
    assert bn_dsrs[0] < solo_dsr and bn_dsrs[0] < 0.95


def test_dev_result_and_freeze_are_invariant_to_holdout_content(g0_pipeline):
    """Selection is a pure function of development inputs: re-running the unified study
    and rebuilding the freeze artifact (the holdout is never an input) reproduces the
    identical result and freeze hash, so changing April labels cannot change the
    selected winner/configuration."""
    from eval.freeze import build_freeze_artifact
    w = g0_pipeline["world"]
    led = TrialLedger()
    cb_led = TrialLedger()
    cb = w["dev"]["arms"]["coinbase_only"]
    from eval.g0 import run_g0cb_study
    run_g0cb_study(cb["matrix"], cb["manifest"], w["contract"],
                   gate=g0_pipeline["gate"], ledger=cb_led)
    res = run_g0xv_development(_arms(w), w["contract"], gate=g0_pipeline["xv_gate"],
                               ledger=led, prior_ledgers=[cb_led])
    assert hash_obj(res) == hash_obj(g0_pipeline["res_xv"])
    rebuilt = build_freeze_artifact(res, contract=w["contract"], ledger=led,
                                    trade_validation_thresholds=g0_pipeline["thresholds"],
                                    holdout_scope=g0_pipeline["scope"],
                                    generated_at="1999-01-01T00:00:00+00:00")
    assert rebuilt["sha256"] == g0_pipeline["freeze"]["sha256"]
