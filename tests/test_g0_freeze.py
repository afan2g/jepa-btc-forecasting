"""Freeze artifact: hash pinning (volatile fields excluded), authorization discipline
(only a passing G0-XV development study freezes), exact-scope validation (generic day
selectors rejected), ledger-drift detection, and tamper detection on load."""
import copy
import json

import pytest

from eval.freeze import (build_freeze_artifact, freeze_hash, load_freeze,
                         validate_holdout_scope, verify_freeze, write_freeze)
from eval.ledger import TrialLedger, trial_identity


def _build(pipe, **over):
    kw = dict(contract=pipe["world"]["contract"], ledger=pipe["led_xv"],
              trade_validation_thresholds=pipe["thresholds"],
              holdout_scope=pipe["scope"], generated_at="2026-07-10T12:00:00+00:00")
    kw.update(over)
    return build_freeze_artifact(pipe["res_xv"], **kw)


def test_freeze_contents_and_hash_pin(g0_pipeline):
    art = g0_pipeline["freeze"]
    assert art["protocol"] == "g0xv-freeze"
    assert art["winner"] == g0_pipeline["res_xv"]["winner"]
    assert art["trial_history"]["n_effective_trials"] == 16
    assert art["trial_history"]["ledger_sha256"] == g0_pipeline["led_xv"].ledger_hash()
    assert art["sources"]["partition_contract_sha256"] \
        == g0_pipeline["res_xv"]["partition_contract_sha256"]
    assert set(art["sources"]["arm_manifests"]) == {"coinbase_only", "binance_only",
                                                    "combined"}
    # per-arm FULL content pins (reserved + feature values) back the holdout refit
    assert set(art["sources"]["arm_matrix_hashes"]) == set(art["sources"]["arm_manifests"])
    contract = g0_pipeline["world"]["contract"]
    assert art["holdout_window"] == {"holdout_start_ns": contract["holdout_start_ns"],
                                     "holdout_end_ns": contract["holdout_end_ns"]}
    assert verify_freeze(art)


def test_freeze_hash_excludes_generated_at_only(g0_pipeline):
    a = _build(g0_pipeline, generated_at="1999-01-01T00:00:00+00:00")
    assert a["sha256"] == g0_pipeline["freeze"]["sha256"]
    b = _build(g0_pipeline,
               trade_validation_thresholds={**g0_pipeline["thresholds"], "extra": 1})
    assert b["sha256"] != g0_pipeline["freeze"]["sha256"]


def test_freeze_requires_g0xv_development_result(g0_pipeline):
    with pytest.raises(ValueError, match="g0xv-development"):
        build_freeze_artifact(g0_pipeline["res_cb"], contract=g0_pipeline["world"]["contract"],
                              ledger=g0_pipeline["led_cb"],
                              trade_validation_thresholds=g0_pipeline["thresholds"],
                              holdout_scope=g0_pipeline["scope"],
                              generated_at="2026-07-10T12:00:00+00:00")


def test_failed_or_inconclusive_study_does_not_authorize(g0_pipeline):
    failed = copy.deepcopy(g0_pipeline["res_xv"])
    failed["g0xv_dev_pass"] = False
    with pytest.raises(ValueError, match="does not authorize"):
        build_freeze_artifact(
            failed, contract=g0_pipeline["world"]["contract"], ledger=g0_pipeline["led_xv"],
            trade_validation_thresholds=g0_pipeline["thresholds"],
            holdout_scope=g0_pipeline["scope"], generated_at="2026-07-10T12:00:00+00:00")


def test_freeze_rejects_substituted_contract(g0_pipeline):
    other = dict(g0_pipeline["world"]["contract"], guard_ns=1)
    with pytest.raises(ValueError, match="stale/substituted"):
        _build(g0_pipeline, contract=other)


def test_freeze_reconciles_winner_against_evidence_and_ledger(g0_pipeline):
    """Codex PR#60 P2: an edited dev result retaining the ledger hash must not freeze a
    winner that is not a passing cross-venue candidate — winner fields reconcile against
    the horizon pass evidence, the reported candidate row, AND the ledger identity."""
    base = g0_pipeline["res_xv"]

    edited = copy.deepcopy(base)
    edited["winner"]["config"] = "ridge"              # non-gate config, same identity
    with pytest.raises(ValueError, match="candidate row"):
        build_freeze_artifact(edited, contract=g0_pipeline["world"]["contract"],
                              ledger=g0_pipeline["led_xv"],
                              trade_validation_thresholds=g0_pipeline["thresholds"],
                              holdout_scope=g0_pipeline["scope"],
                              generated_at="2026-07-10T12:00:00+00:00")

    edited = copy.deepcopy(base)
    ctrl_naive = next(cid for cid, row in base["horizons"]["10s"]["candidates"].items()
                      if row["arm"] == "coinbase_only" and row["config"] == "naive")
    edited["winner"]["identity_sha256"] = ctrl_naive  # control-arm candidate
    with pytest.raises(ValueError, match="solo-passing cross-venue"):
        build_freeze_artifact(edited, contract=g0_pipeline["world"]["contract"],
                              ledger=g0_pipeline["led_xv"],
                              trade_validation_thresholds=g0_pipeline["thresholds"],
                              holdout_scope=g0_pipeline["scope"],
                              generated_at="2026-07-10T12:00:00+00:00")

    edited = copy.deepcopy(base)
    edited["winner"]["feature_cols"] = edited["winner"]["feature_cols"][:1]  # substituted
    with pytest.raises(ValueError, match="ledger trial identity"):
        build_freeze_artifact(edited, contract=g0_pipeline["world"]["contract"],
                              ledger=g0_pipeline["led_xv"],
                              trade_validation_thresholds=g0_pipeline["thresholds"],
                              holdout_scope=g0_pipeline["scope"],
                              generated_at="2026-07-10T12:00:00+00:00")

    edited = copy.deepcopy(base)
    edited["winner"]["horizon"] = "60s"               # horizon not in the result
    with pytest.raises(ValueError, match="not a passing horizon"):
        build_freeze_artifact(edited, contract=g0_pipeline["world"]["contract"],
                              ledger=g0_pipeline["led_xv"],
                              trade_validation_thresholds=g0_pipeline["thresholds"],
                              holdout_scope=g0_pipeline["scope"],
                              generated_at="2026-07-10T12:00:00+00:00")


def test_freeze_rejects_ledger_drift(g0_pipeline):
    drifted = TrialLedger()
    drifted.import_history(g0_pipeline["led_xv"])
    drifted.register(trial_identity(protocol="g0xv", arm="combined", dataset_id="d",
                                    build_id="post-hoc", feature_cols=["f"],
                                    config="lgbm_reg", horizon="10s"), {"net_pnl": 1.0})
    with pytest.raises(ValueError, match="ledger has changed"):
        _build(g0_pipeline, ledger=drifted)


@pytest.mark.parametrize("days,match", [
    (["2026-04-01..2026-04-30"], "explicit YYYY-MM-DD"),      # range selector
    (["2026-04"], "explicit YYYY-MM-DD"),                     # month selector
    ([{"start": "2026-04-01"}], "explicit ISO date strings"), # structured selector
    (["2026-04-02", "2026-04-01"], "sorted and unique"),
    (["2026-04-01", "2026-04-01"], "sorted and unique"),
    (["2026-03-31"], "outside the contract holdout window"),
    (["2026-05-01"], "outside the contract holdout window"),
    ([], "non-empty"),
])
def test_generic_or_out_of_window_day_selectors_rejected(g0_pipeline, days, match):
    scope = {**g0_pipeline["scope"], "days": days}
    with pytest.raises(ValueError, match=match):
        validate_holdout_scope(scope, g0_pipeline["world"]["contract"])


def test_scope_venue_and_field_validation(g0_pipeline):
    contract = g0_pipeline["world"]["contract"]
    with pytest.raises(ValueError, match="wildcards"):
        validate_holdout_scope({**g0_pipeline["scope"], "venues": ["*"]}, contract)
    with pytest.raises(ValueError, match="exactly the fields"):
        validate_holdout_scope({k: v for k, v in g0_pipeline["scope"].items()
                                if k != "build_id"}, contract)
    with pytest.raises(ValueError, match="non-empty string"):
        validate_holdout_scope({**g0_pipeline["scope"], "build_id": ""}, contract)


def test_threshold_validation(g0_pipeline):
    with pytest.raises(ValueError, match="non-empty dict"):
        _build(g0_pipeline, trade_validation_thresholds={})
    with pytest.raises(ValueError, match="scalar"):
        _build(g0_pipeline, trade_validation_thresholds={"bands": [1, 2]})


def test_write_load_roundtrip_and_tamper_detection(tmp_path, g0_pipeline):
    path = tmp_path / "freeze.json"
    write_freeze(g0_pipeline["freeze"], path)
    loaded = load_freeze(path)
    assert loaded["sha256"] == g0_pipeline["freeze"]["sha256"]
    assert freeze_hash(loaded) == loaded["sha256"]

    payload = json.loads(path.read_text())
    payload["winner"]["config"] = "ridge"             # tamper the frozen selection
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="embedded sha256"):
        load_freeze(path)
