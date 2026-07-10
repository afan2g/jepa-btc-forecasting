import pathlib

import pytest

FIXTURES = pathlib.Path(__file__).parent / "fixtures"

# ---------------------------------------------------------------------- G0 (issue #52)
# Session-scoped because the LightGBM CPCV runs are the expensive part; tests that
# mutate matrices/manifests/ledgers MUST work on copies, never on these shared objects.

G0_GATE = {"n_groups": 4, "k": 2, "min_trades": 5, "min_eff_trades": 3.0}
G0_XV_GATE = {**G0_GATE, "noise_band_n_boot": 500}


@pytest.fixture(scope="session")
def g0_world():
    """Known-signal staged-pilot world: both venues carry edge, so G0-CB and G0-XV pass."""
    from eval.synthetic import make_g0_world
    return make_g0_world(n_dev_bars=300, n_holdout_bars=60, cb_signal=4.0, bn_signal=4.0,
                         seed=3)


@pytest.fixture(scope="session")
def g0_pipeline(g0_world):
    """The full development pipeline run ONCE: G0-CB on the control build, the unified
    G0-XV study with the imported G0-CB history, and a freeze artifact over the exact
    holdout scope."""
    from eval.freeze import build_freeze_artifact
    from eval.g0 import run_g0cb_study, run_g0xv_development
    from eval.ledger import TrialLedger

    w = g0_world
    cb = w["dev"]["arms"]["coinbase_only"]
    led_cb = TrialLedger()
    res_cb = run_g0cb_study(cb["matrix"], cb["manifest"], w["contract"], gate=G0_GATE,
                            ledger=led_cb)
    led_xv = TrialLedger()
    arms = [{"name": n, **w["dev"]["arms"][n]}
            for n in ("coinbase_only", "binance_only", "combined")]
    res_xv = run_g0xv_development(arms, w["contract"], gate=G0_XV_GATE, ledger=led_xv,
                                  prior_ledgers=[led_cb])
    assert res_xv["g0xv_dev_pass"] and res_xv["winner"], "session fixture must pass"
    thresholds = {"min_rows_hard": 1000, "price_spike_warn": 0.5}
    scope = {"days": list(w["holdout_days"]), "venues": ["coinbase"],
             "dataset_id": "synthetic-xv-pilot",
             "build_id": f"holdout-seeded-{res_xv['winner']['arm']}",
             "excluded_days": {}}
    freeze = build_freeze_artifact(res_xv, contract=w["contract"], ledger=led_xv,
                                   trade_validation_thresholds=thresholds,
                                   holdout_scope=scope,
                                   generated_at="2026-07-10T12:00:00+00:00")
    return {"world": w, "gate": G0_GATE, "xv_gate": G0_XV_GATE, "res_cb": res_cb,
            "led_cb": led_cb, "res_xv": res_xv, "led_xv": led_xv, "freeze": freeze,
            "scope": scope, "thresholds": thresholds}
