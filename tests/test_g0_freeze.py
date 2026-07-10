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


def test_freeze_recomputes_solo_gate_from_pinned_ledger():
    """Codex PR#60 round-2 P1: editing g0xv_dev_pass / h['pass'] / the solo list in a
    saved dev result (ledger hash intact, winner assembled from a REAL ledger identity)
    cannot freeze a non-passing candidate — the solo gate is re-derived from the pinned
    ledger results (DSR with the full trial count, floors, naive benchmark)."""
    from eval.g0 import run_g0xv_development
    from eval.ledger import TrialLedger
    from eval.synthetic import make_g0_world

    w = make_g0_world(n_dev_bars=250, n_holdout_bars=10, cb_signal=4.0, bn_signal=0.0,
                      seed=1)
    arms = [{"name": n, **w["dev"]["arms"][n]}
            for n in ("coinbase_only", "binance_only", "combined")]
    led = TrialLedger()
    res = run_g0xv_development(arms, w["contract"],
                               gate={"n_groups": 4, "k": 2, "min_trades": 5,
                                     "min_eff_trades": 3.0, "noise_band_n_boot": 500},
                               ledger=led)
    assert res["g0xv_dev_pass"] is False              # genuinely failing study

    h = res["horizons"]["10s"]
    cid, row = max(((c, r) for c, r in h["candidates"].items()
                    if r["arm"] == "combined" and r["config"].startswith("lgbm")),
                   key=lambda cr: cr[1]["net_pnl"])
    entry = next(e for e in led.entries() if e["identity_sha256"] == cid)
    forged = copy.deepcopy(res)
    fh = forged["horizons"]["10s"]
    fh["pass"] = True                                 # edited verdicts only —
    fh["solo_pass_cross_venue"] = [cid]               # every number stays genuine
    fh["candidates"][cid]["passes_solo"] = True
    forged["g0xv_dev_pass"] = True
    forged["winner"] = {"arm": row["arm"], "config": row["config"], "horizon": "10s",
                        "variant": row["variant"], "identity_sha256": cid,
                        "feature_cols": entry["identity"]["feature_cols"],
                        "dataset_id": entry["identity"]["dataset_id"],
                        "build_id": entry["identity"]["build_id"],
                        "net_pnl": row["net_pnl"]}
    scope = {"days": list(w["holdout_days"]), "venues": ["coinbase"],
             "dataset_id": "synthetic-xv-pilot", "build_id": "holdout-seeded-combined"}
    # the pinned ledger verdict (registered at study time) refuses the forged pass
    with pytest.raises(ValueError, match="not a pass"):
        build_freeze_artifact(forged, contract=w["contract"], ledger=led,
                              trade_validation_thresholds={"min_rows": 10},
                              holdout_scope=scope,
                              generated_at="2026-07-10T12:00:00+00:00")


def test_freeze_rejects_verdict_flip_when_pbo_failed_closed(g0_world, monkeypatch):
    """Codex PR#60 round-3 P1: a study that fails ONLY because PBO is unavailable leaves
    genuinely solo-passing candidates, so the ledger solo-gate recompute alone cannot
    catch a verdict flip in the saved JSON. The horizon verdict (PBO/noise-band/pass) is
    ledger-pinned at study time and the freeze refuses when it is not a pass."""
    import copy as _copy

    import eval.g0 as g0
    from eval.g0 import run_g0xv_development
    from eval.ledger import TrialLedger

    monkeypatch.setattr(g0, "_PBO_MIN_ROWS", 10**9)   # force PBO unavailable
    arms = [{"name": n, "manifest": _copy.deepcopy(g0_world["dev"]["arms"][n]["manifest"]),
             "matrix": g0_world["dev"]["arms"][n]["matrix"].copy()}
            for n in ("coinbase_only", "binance_only", "combined")]
    led = TrialLedger()
    res = run_g0xv_development(arms, g0_world["contract"],
                               gate={"n_groups": 4, "k": 2, "min_trades": 5,
                                     "min_eff_trades": 3.0, "noise_band_n_boot": 500},
                               ledger=led)
    h = res["horizons"]["10s"]
    assert res["g0xv_dev_pass"] is False and res["inconclusive_blocking"] is True
    assert h["solo_pass_cross_venue"]                 # candidates genuinely pass solo

    cid = max(h["solo_pass_cross_venue"], key=lambda c: h["candidates"][c]["net_pnl"])
    row = h["candidates"][cid]
    entry = next(e for e in led.entries() if e["identity_sha256"] == cid)
    forged = copy.deepcopy(res)
    fh = forged["horizons"]["10s"]
    fh["pass"] = True                                 # flip the matrix-level verdicts
    fh["pbo_available"] = True
    fh["pbo"] = 0.0
    forged["g0xv_dev_pass"] = True
    forged["inconclusive_blocking"] = False
    forged["winner"] = {"arm": row["arm"], "config": row["config"], "horizon": "10s",
                        "variant": row["variant"], "identity_sha256": cid,
                        "feature_cols": entry["identity"]["feature_cols"],
                        "dataset_id": entry["identity"]["dataset_id"],
                        "build_id": entry["identity"]["build_id"],
                        "net_pnl": row["net_pnl"]}
    scope = {"days": list(g0_world["holdout_days"]), "venues": ["coinbase"],
             "dataset_id": "synthetic-xv-pilot",
             "build_id": f"holdout-seeded-{row['arm']}"}
    with pytest.raises(ValueError, match="pinned ledger verdict .* not a pass"):
        build_freeze_artifact(forged, contract=g0_world["contract"], ledger=led,
                              trade_validation_thresholds={"min_rows": 10},
                              holdout_scope=scope,
                              generated_at="2026-07-10T12:00:00+00:00")


def test_freeze_pins_arm_matrix_hashes_to_ledger_verdict(g0_pipeline):
    """Codex PR#60 round-5 P1: the per-arm full matrix hashes the freeze copies into
    sources (and the holdout refit verifies against) must be the LEDGER-pinned ones — an
    edited dev result cannot point the feature-substitution guard at a recomputed
    feature matrix."""
    edited = copy.deepcopy(g0_pipeline["res_xv"])
    arm = edited["winner"]["arm"]
    edited["arms"][arm]["matrix_content_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="arm_matrix_hashes"):
        build_freeze_artifact(edited, contract=g0_pipeline["world"]["contract"],
                              ledger=g0_pipeline["led_xv"],
                              trade_validation_thresholds=g0_pipeline["thresholds"],
                              holdout_scope=g0_pipeline["scope"],
                              generated_at="2026-07-10T12:00:00+00:00")


def test_freeze_with_append_only_ledger_across_studies(g0_world):
    """Codex PR#60 round-4 P2/P3: a ledger carrying a PRIOR study's g0xv trials and
    verdict entries (imported as search history) must neither poison the winner's DSR
    reconciliation (dispersion is over the verdict-pinned study pool, multiplicity over
    the full history) nor resolve the freeze to the stale study's verdict."""
    from eval.g0 import run_g0xv_development
    from eval.ledger import TrialLedger
    from eval.synthetic import make_g0_world

    # dsr_thresh 0.9: study B carries study A's 12 imported trials (n_eff 24), so its
    # candidates deflate below the default 0.95 — this test needs B to PASS to prove the
    # freeze reconciles under an appended history (the deflation itself is the point of
    # the unified count and is pinned by test_armwise_false_pass_is_caught_by_unified_ledger).
    xv_gate = {"n_groups": 4, "k": 2, "min_trades": 5, "min_eff_trades": 3.0,
               "noise_band_n_boot": 500, "dsr_thresh": 0.9}
    w_a = make_g0_world(n_dev_bars=300, n_holdout_bars=10, cb_signal=4.0, bn_signal=4.0,
                        seed=3, dataset_id="synthetic-xv-pilot-a")
    led_a = TrialLedger()
    run_g0xv_development([{"name": n, **w_a["dev"]["arms"][n]}
                          for n in ("coinbase_only", "binance_only", "combined")],
                         w_a["contract"], gate=xv_gate, ledger=led_a)

    arms_b = [{"name": n,
               "manifest": copy.deepcopy(g0_world["dev"]["arms"][n]["manifest"]),
               "matrix": g0_world["dev"]["arms"][n]["matrix"].copy()}
              for n in ("coinbase_only", "binance_only", "combined")]
    led_b = TrialLedger()
    res_b = run_g0xv_development(arms_b, g0_world["contract"], gate=xv_gate,
                                 ledger=led_b, prior_ledgers=[led_a])
    assert res_b["g0xv_dev_pass"] and res_b["winner"]
    verdicts_10s = [e for e in led_b.entries()
                    if e["identity"]["protocol"] == "g0xv-verdict"
                    and e["identity"]["horizon"] == "10s"]
    assert len(verdicts_10s) == 2                     # stale + current pinned verdicts
    scope = {"days": list(g0_world["holdout_days"]), "venues": ["coinbase"],
             "dataset_id": "synthetic-xv-pilot",
             "build_id": f"holdout-seeded-{res_b['winner']['arm']}"}
    art = build_freeze_artifact(res_b, contract=g0_world["contract"], ledger=led_b,
                                trade_validation_thresholds={"min_rows": 10},
                                holdout_scope=scope,
                                generated_at="2026-07-10T12:00:00+00:00")
    assert verify_freeze(art)
    assert art["trial_history"]["n_effective_trials"] == 24   # full multiplicity kept


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
