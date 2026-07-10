"""One-time holdout transaction + fixed scorer: exact-scope validation first, scoring
only after PASS, stale/retry/substitution rejected, one-time consumption, pre-April fit,
May-support rejection, and reproducibility from the frozen artifact."""
import copy

import numpy as np
import pytest

from eval.consumption import (load_record, open_transaction, record_trade_validation)
from eval.freeze import build_freeze_artifact
from eval.holdout import fit_frozen_config, score_fixed_holdout
from eval.synthetic import _iso_ns

H10 = 10_000_000_000
MAY = _iso_ns("2026-05-01T00:00:00+00:00")


def _open(tmp_path, pipe, name="rec.json"):
    path = tmp_path / name
    open_transaction(path, pipe["freeze"])
    return path


def _validate(path, pipe, *, passed=True, days=None, venues=None):
    return record_trade_validation(
        path, freeze_artifact=pipe["freeze"],
        scope_days=days if days is not None else pipe["scope"]["days"],
        scope_venues=venues if venues is not None else pipe["scope"]["venues"],
        passed=passed, report_sha256="a" * 64)


def _score(path, pipe, **over):
    w = pipe["world"]
    arm = pipe["res_xv"]["winner"]["arm"]
    kw = dict(freeze_artifact=pipe["freeze"], record_path=path,
              contract=w["contract"],
              dev_matrix=w["dev"]["arms"][arm]["matrix"],
              dev_manifest=w["dev"]["arms"][arm]["manifest"],
              holdout_matrix=w["holdout"]["arms"][arm]["matrix"],
              holdout_manifest=w["holdout"]["arms"][arm]["manifest"])
    kw.update(over)
    return score_fixed_holdout(**kw)


# ------------------------------------------------------------------------- transaction
def test_open_transaction_is_one_time(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    assert load_record(path)["state"] == "frozen"
    with pytest.raises(ValueError, match="cannot be reused or replaced"):
        open_transaction(path, g0_pipeline["freeze"])


def test_validation_success_records_consumption(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    rec = _validate(path, g0_pipeline, passed=True)
    assert rec["state"] == "validated"
    events = [e["event"] for e in rec["history"]]
    assert events == ["opened", "trade_validation"]
    assert load_record(path)["state"] == "validated"


def test_validation_rejects_stale_or_regenerated_artifact(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    other = build_freeze_artifact(
        g0_pipeline["res_xv"], contract=g0_pipeline["world"]["contract"],
        ledger=g0_pipeline["led_xv"],
        trade_validation_thresholds={**g0_pipeline["thresholds"], "extra_knob": 1},
        holdout_scope=g0_pipeline["scope"], generated_at="2026-07-10T12:00:00+00:00")
    assert other["sha256"] != g0_pipeline["freeze"]["sha256"]
    with pytest.raises(ValueError, match="stale or regenerated"):
        record_trade_validation(path, freeze_artifact=other,
                                scope_days=g0_pipeline["scope"]["days"],
                                scope_venues=g0_pipeline["scope"]["venues"],
                                passed=True, report_sha256="a" * 64)


def test_validation_scope_deviations_rejected(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    days = g0_pipeline["scope"]["days"]
    with pytest.raises(ValueError, match="do not exactly match"):
        _validate(path, g0_pipeline, days=days[:-1])                 # partial scope
    with pytest.raises(ValueError, match="sorted, unique"):
        _validate(path, g0_pipeline, days=list(reversed(days)))     # reordered
    with pytest.raises(ValueError, match="explicit YYYY-MM-DD"):
        _validate(path, g0_pipeline, days=["2026-04-01..2026-04-30"])  # generic selector
    with pytest.raises(ValueError, match="venues"):
        _validate(path, g0_pipeline, venues=["coinbase", "binance_spot"])
    assert load_record(path)["state"] == "frozen"     # rejected attempts changed nothing


def test_validation_is_single_shot(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    _validate(path, g0_pipeline, passed=True)
    with pytest.raises(ValueError, match="exactly one validation attempt"):
        _validate(path, g0_pipeline, passed=True)

    path2 = _open(tmp_path, g0_pipeline, "rec2.json")
    _validate(path2, g0_pipeline, passed=False)
    with pytest.raises(ValueError, match="exactly one validation attempt"):
        _validate(path2, g0_pipeline, passed=True)    # no retry after a FAIL either


def test_validation_failure_blocks_scoring_permanently(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    rec = _validate(path, g0_pipeline, passed=False)
    assert rec["state"] == "validation_failed"
    with pytest.raises(ValueError, match="blocking/inconclusive"):
        _score(path, g0_pipeline)
    # ... and the failed transaction cannot be replaced
    with pytest.raises(ValueError, match="cannot be reused or replaced"):
        open_transaction(path, g0_pipeline["freeze"])


def test_scoring_requires_validation_first(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    with pytest.raises(ValueError, match="'validated'"):
        _score(path, g0_pipeline)


# ------------------------------------------------------------------------------ scorer
def test_score_consumes_once_and_reproduces(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    _validate(path, g0_pipeline, passed=True)
    with pytest.raises(ValueError, match="already-consumed"):
        _score(path, g0_pipeline, verify_only=True)   # nothing recorded yet

    res = _score(path, g0_pipeline)
    assert res["protocol"] == "g0xv-holdout" and res["consumed"] is True
    assert res["winner"] == g0_pipeline["freeze"]["winner"]
    assert load_record(path)["state"] == "scored"
    with pytest.raises(ValueError, match="consumed|already scored"):
        _score(path, g0_pipeline)                     # one-time consumption

    again = _score(path, g0_pipeline, verify_only=True)
    assert again["reproduces_recorded_score"] is True # reproducible from the artifact
    assert again["metrics"] == res["metrics"]
    assert load_record(path)["state"] == "scored"     # verify mutated nothing


def test_verify_only_detects_substituted_holdout(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    _validate(path, g0_pipeline, passed=True)
    _score(path, g0_pipeline)
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    mutated = w["holdout"]["arms"][arm]["matrix"].copy()
    mutated["y_fwd_bps"] = -mutated["y_fwd_bps"]
    mutated["label"] = -mutated["label"]
    res = _score(path, g0_pipeline, holdout_matrix=mutated, verify_only=True)
    assert res["reproduces_recorded_score"] is False


def test_score_rejects_wrong_dev_build_and_tampered_dev_rows(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    _validate(path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    ctrl = w["dev"]["arms"]["coinbase_only"]
    if g0_pipeline["res_xv"]["winner"]["arm"] != "coinbase_only":
        with pytest.raises(ValueError, match="not the frozen"):
            _score(path, g0_pipeline, dev_matrix=ctrl["matrix"],
                   dev_manifest=ctrl["manifest"])
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    tampered = w["dev"]["arms"][arm]["matrix"].copy()
    tampered.loc[0, "y_fwd_bps"] += 1.0
    with pytest.raises(ValueError, match="exactly the rows it was selected on"):
        _score(path, g0_pipeline, dev_matrix=tampered)
    assert load_record(path)["state"] == "validated"  # rejected attempts consume nothing


def test_score_rejects_holdout_row_reaching_may(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    _validate(path, g0_pipeline, passed=True)
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
        _score(path, g0_pipeline, holdout_matrix=bad)


def test_score_rejects_partial_scope_holdout(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    _validate(path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    full = w["holdout"]["arms"][arm]["matrix"]
    import pandas as pd
    day = pd.to_datetime(full["t_event"], unit="ns", utc=True).dt.strftime("%Y-%m-%d")
    partial = full[day != g0_pipeline["scope"]["days"][0]].reset_index(drop=True)
    with pytest.raises(ValueError, match="do not exactly match the frozen scope"):
        _score(path, g0_pipeline, holdout_matrix=partial)


def test_score_rejects_wrong_holdout_build_or_missing_features(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    _validate(path, g0_pipeline, passed=True)
    w = g0_pipeline["world"]
    arm = g0_pipeline["res_xv"]["winner"]["arm"]
    other_arm = "binance_only" if arm != "binance_only" else "combined"
    other = w["holdout"]["arms"][other_arm]
    with pytest.raises(ValueError, match="dataset/build does not match"):
        _score(path, g0_pipeline, holdout_matrix=other["matrix"],
               holdout_manifest=other["manifest"])
    crippled_man = copy.deepcopy(w["holdout"]["arms"][arm]["manifest"])
    dropped = crippled_man["feature_cols"].pop()
    crippled_mat = w["holdout"]["arms"][arm]["matrix"].drop(columns=[dropped])
    with pytest.raises(ValueError, match="lacks frozen winner features"):
        _score(path, g0_pipeline, holdout_matrix=crippled_mat,
               holdout_manifest=crippled_man)


def test_score_rejects_substituted_contract(tmp_path, g0_pipeline):
    path = _open(tmp_path, g0_pipeline)
    _validate(path, g0_pipeline, passed=True)
    with pytest.raises(ValueError, match="frozen source pin"):
        _score(path, g0_pipeline,
               contract=dict(g0_pipeline["world"]["contract"], guard_ns=1))


# ----------------------------------------------------------------- fit discipline
def test_fit_uses_only_development_rows_and_ignores_holdout_labels(g0_pipeline):
    """AC: no April-derived label enters the fit — forecasts on April features are
    bit-identical whatever April's labels say, and the fit consumes dev rows only."""
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
    mutated = hold_slice.copy()
    mutated["y_fwd_bps"] = -mutated["y_fwd_bps"]      # flip every April label
    mutated["label"] = -mutated["label"]
    fc2 = fitted["predict"](mutated[winner["feature_cols"]].to_numpy(float))
    assert np.array_equal(fc, fc2)


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
