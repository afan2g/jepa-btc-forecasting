"""One-time holdout transaction + fixed scorer: exact-scope validation first, scoring
only after PASS, stale/retry/substitution rejected (transactions are keyed by holdout
identity, not file path), one-time consumption, pre-April fit on content-pinned features,
May-support rejection, and reproducibility from the frozen artifact."""
import copy

import numpy as np
import pandas as pd
import pytest

from eval.consumption import (load_record, open_transaction, record_path_for,
                              record_trade_validation)
from eval.freeze import build_freeze_artifact
from eval.holdout import fit_frozen_config, score_fixed_holdout
from eval.synthetic import _iso_ns

H10 = 10_000_000_000
MAY = _iso_ns("2026-05-01T00:00:00+00:00")


def _open(tmp_path, pipe):
    open_transaction(tmp_path, pipe["freeze"])
    return tmp_path


def _load(records_dir, pipe):
    return load_record(record_path_for(records_dir, pipe["freeze"]))


def _validate(records_dir, pipe, *, passed=True, days=None, venues=None,
              thresholds=None):
    return record_trade_validation(
        records_dir, freeze_artifact=pipe["freeze"],
        scope_days=days if days is not None else pipe["scope"]["days"],
        scope_venues=venues if venues is not None else pipe["scope"]["venues"],
        thresholds=thresholds if thresholds is not None else pipe["thresholds"],
        passed=passed, report_sha256="a" * 64)


def _score(records_dir, pipe, **over):
    w = pipe["world"]
    arm = pipe["res_xv"]["winner"]["arm"]
    kw = dict(freeze_artifact=pipe["freeze"], records_dir=records_dir,
              contract=w["contract"],
              dev_matrix=w["dev"]["arms"][arm]["matrix"],
              dev_manifest=w["dev"]["arms"][arm]["manifest"],
              holdout_matrix=w["holdout"]["arms"][arm]["matrix"],
              holdout_manifest=w["holdout"]["arms"][arm]["manifest"])
    kw.update(over)
    return score_fixed_holdout(**kw)


def _other_freeze(pipe):
    """A regenerated selection artifact over the SAME holdout (different thresholds ->
    different sha256, same holdout identity)."""
    return build_freeze_artifact(
        pipe["res_xv"], contract=pipe["world"]["contract"], ledger=pipe["led_xv"],
        trade_validation_thresholds={**pipe["thresholds"], "extra_knob": 1},
        holdout_scope=pipe["scope"], generated_at="2026-07-10T12:00:00+00:00")


# ------------------------------------------------------------------------- transaction
def test_open_transaction_is_one_time_per_holdout(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    assert _load(tmp_path, g0_pipeline)["state"] == "frozen"
    with pytest.raises(ValueError, match="cannot be reused or replaced"):
        open_transaction(tmp_path, g0_pipeline["freeze"])
    # The record is keyed by HOLDOUT identity, not by the artifact: a regenerated freeze
    # over the same holdout maps to the same transaction and cannot open a fresh one.
    other = _other_freeze(g0_pipeline)
    assert other["sha256"] != g0_pipeline["freeze"]["sha256"]
    with pytest.raises(ValueError, match="cannot be reused or replaced"):
        open_transaction(tmp_path, other)


def test_validation_success_records_consumption(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    rec = _validate(tmp_path, g0_pipeline, passed=True)
    assert rec["state"] == "validated"
    events = [e["event"] for e in rec["history"]]
    assert events == ["opened", "trade_validation"]
    assert _load(tmp_path, g0_pipeline)["state"] == "validated"


def test_validation_rejects_stale_or_regenerated_artifact(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    other = _other_freeze(g0_pipeline)
    with pytest.raises(ValueError, match="stale or regenerated"):
        record_trade_validation(tmp_path, freeze_artifact=other,
                                scope_days=g0_pipeline["scope"]["days"],
                                scope_venues=g0_pipeline["scope"]["venues"],
                                thresholds=g0_pipeline["thresholds"],
                                passed=True, report_sha256="a" * 64)


def test_validation_scope_deviations_rejected(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    days = g0_pipeline["scope"]["days"]
    with pytest.raises(ValueError, match="do not exactly match"):
        _validate(tmp_path, g0_pipeline, days=days[:-1])            # partial scope
    with pytest.raises(ValueError, match="sorted, unique"):
        _validate(tmp_path, g0_pipeline, days=list(reversed(days)))  # reordered
    with pytest.raises(ValueError, match="explicit YYYY-MM-DD"):
        _validate(tmp_path, g0_pipeline, days=["2026-04-01..2026-04-30"])  # selector
    with pytest.raises(ValueError, match="venues"):
        _validate(tmp_path, g0_pipeline, venues=["coinbase", "binance_spot"])
    # a verdict produced under stale/looser thresholds cannot unlock scoring
    loose = {**g0_pipeline["thresholds"], "price_spike_warn": 99.0}
    with pytest.raises(ValueError, match="frozen\\s+trade-validation thresholds"):
        _validate(tmp_path, g0_pipeline, thresholds=loose)
    assert _load(tmp_path, g0_pipeline)["state"] == "frozen"  # nothing changed


def test_validation_is_single_shot(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    with pytest.raises(ValueError, match="exactly one validation attempt"):
        _validate(tmp_path, g0_pipeline, passed=True)

    fail_dir = tmp_path / "fail"
    fail_dir.mkdir()
    open_transaction(fail_dir, g0_pipeline["freeze"])
    _validate(fail_dir, g0_pipeline, passed=False)
    with pytest.raises(ValueError, match="exactly one validation attempt"):
        _validate(fail_dir, g0_pipeline, passed=True)   # no retry after a FAIL either


def test_validation_failure_blocks_scoring_permanently(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    rec = _validate(tmp_path, g0_pipeline, passed=False)
    assert rec["state"] == "validation_failed"
    with pytest.raises(ValueError, match="blocking/inconclusive"):
        _score(tmp_path, g0_pipeline)
    # ... and the failed transaction cannot be replaced, even by a regenerated freeze
    with pytest.raises(ValueError, match="cannot be reused or replaced"):
        open_transaction(tmp_path, g0_pipeline["freeze"])
    with pytest.raises(ValueError, match="cannot be reused or replaced"):
        open_transaction(tmp_path, _other_freeze(g0_pipeline))


def test_scoring_requires_validation_first(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    with pytest.raises(ValueError, match="'validated'"):
        _score(tmp_path, g0_pipeline)


# ------------------------------------------------------------------------------ scorer
def test_score_consumes_once_and_reproduces(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    with pytest.raises(ValueError, match="already-consumed"):
        _score(tmp_path, g0_pipeline, verify_only=True)   # nothing recorded yet

    res = _score(tmp_path, g0_pipeline)
    assert res["protocol"] == "g0xv-holdout" and res["consumed"] is True
    assert res["winner"] == g0_pipeline["freeze"]["winner"]
    assert "holdout_matrix_sha256" in res            # audit pin of what was consumed
    assert _load(tmp_path, g0_pipeline)["state"] == "scored"
    with pytest.raises(ValueError, match="consumed|already scored"):
        _score(tmp_path, g0_pipeline)                     # one-time consumption

    again = _score(tmp_path, g0_pipeline, verify_only=True)
    assert again["reproduces_recorded_score"] is True # reproducible from the artifact
    assert again["metrics"] == res["metrics"]
    assert _load(tmp_path, g0_pipeline)["state"] == "scored"  # verify mutated nothing

    # row order is canonicalized before the fit: the same frozen rows in a different
    # parquet order must reproduce the identical recorded score
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    shuffled = (w["dev"]["arms"][arm]["matrix"]
                .sample(frac=1.0, random_state=7).reset_index(drop=True))
    reordered = _score(tmp_path, g0_pipeline, dev_matrix=shuffled, verify_only=True)
    assert reordered["reproduces_recorded_score"] is True


def test_verify_only_is_not_a_holdout_oracle(tmp_path, g0_pipeline):
    """Non-reproducing inputs get NO metrics back: repeated verify calls with perturbed
    inputs must not become an iterate-against-holdout channel."""
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    _score(tmp_path, g0_pipeline)
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    mutated = w["holdout"]["arms"][arm]["matrix"].copy()
    mutated["y_fwd_bps"] = -mutated["y_fwd_bps"]
    mutated["label"] = -mutated["label"]
    res = _score(tmp_path, g0_pipeline, holdout_matrix=mutated, verify_only=True)
    assert res["reproduces_recorded_score"] is False
    assert "metrics" not in res and "winner" not in res


def test_score_rejects_wrong_dev_build_and_tampered_dev_rows(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    ctrl = w["dev"]["arms"]["coinbase_only"]
    if g0_pipeline["res_xv"]["winner"]["arm"] != "coinbase_only":
        with pytest.raises(ValueError, match="not the frozen"):
            _score(tmp_path, g0_pipeline, dev_matrix=ctrl["matrix"],
                   dev_manifest=ctrl["manifest"])
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    tampered = w["dev"]["arms"][arm]["matrix"].copy()
    tampered.loc[0, "y_fwd_bps"] += 1.0
    with pytest.raises(ValueError, match="exactly the rows it was selected on"):
        _score(tmp_path, g0_pipeline, dev_matrix=tampered)
    assert _load(tmp_path, g0_pipeline)["state"] == "validated"  # nothing consumed


def test_score_rejects_substituted_dev_feature_values(tmp_path, g0_pipeline):
    """The P1 red-team channel: reserved rows intact, winner FEATURE values replaced
    post-freeze (e.g. recomputed with holdout knowledge). The per-arm full content pin
    must reject the refit before any holdout access."""
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    winner = g0_pipeline["res_xv"]["winner"]
    poisoned = w["dev"]["arms"][winner["arm"]]["matrix"].copy()
    rng = np.random.default_rng(0)
    for c in winner["feature_cols"]:
        poisoned[c] = rng.standard_normal(len(poisoned))
    with pytest.raises(ValueError, match="FEATURE content"):
        _score(tmp_path, g0_pipeline, dev_matrix=poisoned)
    assert _load(tmp_path, g0_pipeline)["state"] == "validated"


def test_score_rejects_holdout_row_reaching_may(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    bad = w["holdout"]["arms"][arm]["matrix"].copy()
    te = MAY - H10                                    # April row whose support hits May
    last = bad.index[-1]
    bad.loc[last, "t_event"] = te
    bad.loc[last, "t_barrier"] = te + H10
    bad.loc[last, "t_available"] = te
    bad.loc[last, "t_feature_start"] = te - H10
    with pytest.raises(ValueError, match="span-safe"):
        _score(tmp_path, g0_pipeline, holdout_matrix=bad)


def test_score_rejects_partial_scope_or_duplicated_holdout(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    full = w["holdout"]["arms"][arm]["matrix"]
    day = pd.to_datetime(full["t_event"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")
    partial = full[day != g0_pipeline["scope"]["days"][0]].reset_index(drop=True)
    with pytest.raises(ValueError, match="do not exactly match the frozen scope"):
        _score(tmp_path, g0_pipeline, holdout_matrix=partial)
    duplicated = pd.concat([full, full.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate .* holdout rows"):
        _score(tmp_path, g0_pipeline, holdout_matrix=duplicated)


def test_score_rejects_wrong_holdout_build_or_missing_features(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    other_arm = "binance_only" if arm != "binance_only" else "combined"
    other = w["holdout"]["arms"][other_arm]
    with pytest.raises(ValueError, match="dataset/build does not match"):
        _score(tmp_path, g0_pipeline, holdout_matrix=other["matrix"],
               holdout_manifest=other["manifest"])
    crippled_man = copy.deepcopy(w["holdout"]["arms"][arm]["manifest"])
    dropped = crippled_man["feature_cols"].pop()
    crippled_mat = w["holdout"]["arms"][arm]["matrix"].drop(columns=[dropped])
    with pytest.raises(ValueError, match="lacks frozen winner features"):
        _score(tmp_path, g0_pipeline, holdout_matrix=crippled_mat,
               holdout_manifest=crippled_man)


def test_score_rejects_substituted_contract(tmp_path, g0_pipeline):
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    with pytest.raises(ValueError, match="frozen source pin"):
        _score(tmp_path, g0_pipeline,
               contract=dict(g0_pipeline["world"]["contract"], guard_ns=1))


# ----------------------------------------------------------------- fit discipline
def test_fit_consumes_dev_labels_but_cannot_see_holdout_labels(g0_pipeline):
    """Sensitivity anchor + independence: flipping DEVELOPMENT labels changes the fitted
    forecasts (the fit genuinely consumes labels), while holdout labels cannot reach the
    fit at all — forecasts on April features are bit-identical whatever April says."""
    w = g0_pipeline["world"]
    winner = g0_pipeline["res_xv"]["winner"]
    dev = w["dev"]["arms"][winner["arm"]]["matrix"]
    dev_slice = dev[dev["horizon"] == winner["horizon"]].reset_index(drop=True)
    fitted = fit_frozen_config(dev_slice, winner["feature_cols"], winner["config"])
    assert fitted["n_train"] == len(dev_slice)

    hold = w["holdout"]["arms"][winner["arm"]]["matrix"]
    hold_slice = hold[hold["horizon"] == winner["horizon"]].reset_index(drop=True)
    X = hold_slice[winner["feature_cols"]].to_numpy(float)
    fc = fitted["predict"](X)

    flipped_dev = dev_slice.copy()
    flipped_dev["y_fwd_bps"] = -flipped_dev["y_fwd_bps"]
    flipped_dev["label"] = -flipped_dev["label"]
    refit = fit_frozen_config(flipped_dev, winner["feature_cols"], winner["config"])
    assert not np.array_equal(refit["predict"](X), fc)   # labels DO drive the fit

    mutated_hold = hold_slice.copy()
    mutated_hold["y_fwd_bps"] = -mutated_hold["y_fwd_bps"]
    mutated_hold["label"] = -mutated_hold["label"]
    fc2 = fitted["predict"](mutated_hold[winner["feature_cols"]].to_numpy(float))
    assert np.array_equal(fc, fc2)                       # holdout labels cannot


@pytest.mark.parametrize("config", ["naive", "ridge", "lgbm_reg", "lgbm_clf"])
def test_fit_frozen_config_supports_all_rungs(g0_pipeline, config):
    w = g0_pipeline["world"]
    dev = w["dev"]["arms"]["combined"]["matrix"]
    feats = w["arm_features"]["combined"]
    fitted = fit_frozen_config(dev, feats, config)
    fc = fitted["predict"](dev[feats].head(8).to_numpy(float))
    assert fc.shape == (8,)
    if config == "naive":
        assert not fc.any()


def test_fit_frozen_config_rejects_unknown_config(g0_pipeline):
    w = g0_pipeline["world"]
    with pytest.raises(ValueError, match="unknown frozen config"):
        fit_frozen_config(w["dev"]["arms"]["combined"]["matrix"],
                          w["arm_features"]["combined"], "dlinear")


# ------------------------------------------------- winner-slice exact-scope (multi-horizon)
@pytest.fixture(scope="module")
def g0_multi_pipeline():
    """A two-horizon pipeline: the union-of-horizons day check alone cannot see a
    missing winner-horizon day, so the scorer must also pin the scored slice's days."""
    from eval.freeze import build_freeze_artifact
    from eval.g0 import run_g0xv_development
    from eval.ledger import TrialLedger
    from eval.synthetic import make_g0_world
    w = make_g0_world(n_dev_bars=300, n_holdout_bars=40, cb_signal=4.0, bn_signal=4.0,
                      seed=3, horizons={"10s": 10_000_000_000, "30s": 30_000_000_000})
    arms = [{"name": n, **w["dev"]["arms"][n]}
            for n in ("coinbase_only", "binance_only", "combined")]
    gate = {"n_groups": 4, "k": 2, "min_trades": 5, "min_eff_trades": 3.0,
            "noise_band_n_boot": 500}
    led = TrialLedger()
    res = run_g0xv_development(arms, w["contract"], gate=gate, ledger=led)
    assert res["g0xv_dev_pass"] and res["winner"], "multi-horizon fixture must pass"
    scope = {"days": list(w["holdout_days"]), "venues": ["coinbase"],
             "dataset_id": "synthetic-xv-pilot",
             "build_id": f"holdout-seeded-{res['winner']['arm']}",
             "excluded_days": {}}
    freeze = build_freeze_artifact(res, contract=w["contract"], ledger=led,
                                   trade_validation_thresholds={"min_rows": 10},
                                   holdout_scope=scope,
                                   generated_at="2026-07-10T12:00:00+00:00")
    return {"world": w, "res_xv": res, "freeze": freeze, "scope": scope, "ledger": led}


def test_score_rejects_partial_winner_horizon_coverage(tmp_path, g0_multi_pipeline):
    pipe = g0_multi_pipeline
    open_transaction(tmp_path, pipe["freeze"])
    record_trade_validation(tmp_path, freeze_artifact=pipe["freeze"],
                            scope_days=pipe["scope"]["days"],
                            scope_venues=pipe["scope"]["venues"],
                            thresholds={"min_rows": 10},
                            passed=True, report_sha256="a" * 64)
    w = pipe["world"]
    winner = pipe["res_xv"]["winner"]
    hold = w["holdout"]["arms"][winner["arm"]]["matrix"]
    day = pd.to_datetime(hold["t_event"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")
    mid_day = pipe["scope"]["days"][len(pipe["scope"]["days"]) // 2]
    # drop the WINNER horizon's rows on one frozen day; the other horizon still covers
    # that day, so the union-of-horizons check alone would pass
    partial = hold[~((day == mid_day)
                     & (hold["horizon"] == winner["horizon"]))].reset_index(drop=True)
    union_days = sorted(pd.to_datetime(partial["t_event"], unit="ns", utc=True)
                        .dt.strftime("%Y-%m-%d").unique())
    assert union_days == pipe["scope"]["days"]
    with pytest.raises(ValueError, match="partial winner-horizon coverage"):
        score_fixed_holdout(freeze_artifact=pipe["freeze"], records_dir=tmp_path,
                            contract=w["contract"],
                            dev_matrix=w["dev"]["arms"][winner["arm"]]["matrix"],
                            dev_manifest=w["dev"]["arms"][winner["arm"]]["manifest"],
                            holdout_matrix=partial,
                            holdout_manifest=w["holdout"]["arms"][winner["arm"]]["manifest"])


def test_freeze_rejects_truncated_horizon_set(g0_multi_pipeline):
    """Codex PR#60 round-17 P1: removing a horizon from the saved dev result (its
    verdict is still ledger-pinned) cannot slip past the deterministic-selection guard —
    the freeze enumerates the horizon set from the ledger."""
    from eval.freeze import build_freeze_artifact
    pipe = g0_multi_pipeline
    truncated = copy.deepcopy(pipe["res_xv"])
    victim = next(t for t in truncated["horizons"]
                  if t != truncated["winner"]["horizon"])
    del truncated["horizons"][victim]
    with pytest.raises(ValueError, match="ledger-pinned verdict horizons"):
        build_freeze_artifact(truncated, contract=pipe["world"]["contract"],
                              ledger=pipe["ledger"],
                              trade_validation_thresholds={"min_rows": 10},
                              holdout_scope=pipe["scope"],
                              generated_at="2026-07-10T12:00:00+00:00")


def test_score_requires_declared_targets(tmp_path, g0_pipeline):
    """Codex PR#60 round-17 P2: the holdout manifest must declare exactly the consumed
    targets (y_fwd_bps, label) — scoring under a manifest that hides its labels is
    refused before any matrix read."""
    _open(tmp_path, g0_pipeline)
    _validate(tmp_path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    hidden = copy.deepcopy(w["holdout"]["arms"][arm]["manifest"])
    hidden["target_cols"] = ["y_fwd_bps"]
    with pytest.raises(ValueError, match="declare exactly"):
        _score(tmp_path, g0_pipeline, holdout_manifest=hidden)
    # ... and the holdout manifest's single target venue must be Coinbase BTC-USD
    wrong_venue = copy.deepcopy(w["holdout"]["arms"][arm]["manifest"])
    for v in wrong_venue["venues"]:
        if v.get("role") == "target":
            v["symbol"] = "ETH-USD"
    with pytest.raises(ValueError, match="exactly one target venue COINBASE/BTC-USD"):
        _score(tmp_path, g0_pipeline, holdout_manifest=wrong_venue)
    assert _load(tmp_path, g0_pipeline)["state"] == "validated"
