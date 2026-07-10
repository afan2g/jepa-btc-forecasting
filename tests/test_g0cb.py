"""G0-CB development-only screen: cannot accept/open holdout input, persists every
attempted candidate/variant, ledger-reconciled DSR, PASS/FAIL synthetic controls."""
import copy
import inspect

import numpy as np
import pandas as pd
import pytest

from eval.g0 import g0cb_manifest_prechecks, run_g0cb_study
from eval.ledger import TrialLedger
from eval.stats import deflated_sharpe
from eval.synthetic import G0_CB_FEATURES, make_g0_world

GATE = {"n_groups": 4, "k": 2, "min_trades": 5, "min_eff_trades": 3.0}


def _recompute_dsr(res: dict, ledger: TrialLedger, tag: str = "10s") -> None:
    """The acceptance-criterion reconciliation: every reported DSR must be reproducible
    from the explicit ledger (effective trial count + per-trial moments) and the
    horizon pool's trade-Sharpe dispersion."""
    pool = res["horizons"][tag]["candidates"]
    by_id = {e["identity_sha256"]: e["result"] for e in ledger.entries()}
    sr_std = float(np.array([r["trade_sharpe"] for r in pool.values()]).std() + 1e-9)
    n = res["ledger"]["n_effective_trials"]
    assert n == ledger.n_effective_trials()
    for cid, row in pool.items():
        lr = by_id[cid]
        expect = deflated_sharpe(sr_hat=row["trade_sharpe"], sr_trials_std=sr_std,
                                 n_trials=max(2, n), T=max(int(round(row["t_eff"])), 2),
                                 skew=lr["skew"], kurt=lr["kurt"])
        assert row["dsr"] == pytest.approx(expect), cid


def test_g0cb_passes_and_persists_every_attempt(g0_pipeline):
    res, led = g0_pipeline["res_cb"], g0_pipeline["led_cb"]
    assert res["protocol"] == "g0cb-development"
    assert res["development_only"] is True and res["g1_claim"] is False
    assert res["g0cb_pass"] is True
    cands = res["horizons"]["10s"]["candidates"]
    assert sorted(c["config"] for c in cands.values()) == ["lgbm_clf", "lgbm_reg",
                                                           "naive", "ridge"]
    assert led.n_effective_trials() == 4 == res["ledger"]["n_effective_trials"]
    assert res["ledger"]["ledger_sha256"] == led.ledger_hash()
    _recompute_dsr(res, led)


def test_g0cb_signature_has_no_holdout_input():
    params = inspect.signature(run_g0cb_study).parameters
    assert not any("holdout" in p.lower() for p in params), \
        "G0-CB must not be able to accept a holdout input"


def test_g0cb_rejects_holdout_bound_manifest_before_any_data(g0_world):
    man = g0_world["holdout"]["arms"]["coinbase_only"]["manifest"]
    with pytest.raises(ValueError, match="accepts only 'development'"):
        # matrix=None: the rejection must fire before the matrix is touched at all
        run_g0cb_study(None, man, g0_world["contract"], gate=GATE)
    with pytest.raises(ValueError, match="accepts only 'development'"):
        g0cb_manifest_prechecks(man, g0_world["contract"])


def test_g0cb_rejects_cross_venue_manifest_before_any_data(g0_world):
    man = g0_world["dev"]["arms"]["combined"]["manifest"]
    with pytest.raises(ValueError, match="target-venue-only"):
        run_g0cb_study(None, man, g0_world["contract"], gate=GATE)


def test_g0cb_venue_role_gate_fails_closed(g0_world):
    """`role` is optional in the manifest schema, so omitting it (or mislabeling the
    signal venue as 'target'-less) must NOT slip a cross-venue build past the
    target-venue-only screen: every venue must declare role 'target' explicitly."""
    man = copy.deepcopy(g0_world["dev"]["arms"]["combined"]["manifest"])
    for v in man["venues"]:
        v.pop("role", None)               # omitted roles: fail closed, not fail open
    with pytest.raises(ValueError, match="role 'target' explicitly"):
        g0cb_manifest_prechecks(man, g0_world["contract"])


def test_g0cb_rejects_april_rows_before_fit(g0_world):
    man = g0_world["dev"]["arms"]["coinbase_only"]["manifest"]
    apr = g0_world["holdout"]["arms"]["coinbase_only"]["matrix"]
    with pytest.raises(ValueError, match="span-safe"):
        run_g0cb_study(apr, man, g0_world["contract"], gate=GATE)


def test_g0cb_rejects_mixed_dev_and_april_rows(g0_world):
    man = g0_world["dev"]["arms"]["coinbase_only"]["manifest"]
    dev = g0_world["dev"]["arms"]["coinbase_only"]["matrix"]
    apr = g0_world["holdout"]["arms"]["coinbase_only"]["matrix"]
    mixed = pd.concat([dev, apr.head(1)], ignore_index=True)
    with pytest.raises(ValueError, match="span-safe"):
        run_g0cb_study(mixed, man, g0_world["contract"], gate=GATE)


def test_g0cb_variants_are_registered_trials_and_deflate_dsr(g0_world):
    w = g0_world
    cb = w["dev"]["arms"]["coinbase_only"]
    led = TrialLedger()
    variants = [{"name": "top3", "feature_cols": G0_CB_FEATURES[:3]}]
    res = run_g0cb_study(cb["matrix"], cb["manifest"], w["contract"], gate=GATE,
                         ledger=led, variants=variants)
    # 4 base configs + 3 variant configs (naive is feature-independent, base-only)
    assert led.n_effective_trials() == 7
    cands = res["horizons"]["10s"]["candidates"]
    assert sum(c["variant"] == "top3" for c in cands.values()) == 3
    _recompute_dsr(res, led)              # DSR counts the variant trials explicitly


def test_g0cb_variant_validation_fails_closed(g0_world):
    w = g0_world
    cb = w["dev"]["arms"]["coinbase_only"]
    for bad, match in ((({"name": "x", "feature_cols": ["nope"]}), "subset"),
                       (({"name": "x", "margin_bps": 2.0}), "unknown variant keys"),
                       (({"feature_cols": ["cb_ofi"]}), "non-empty 'name'")):
        with pytest.raises(ValueError, match=match):
            run_g0cb_study(cb["matrix"], cb["manifest"], w["contract"], gate=GATE,
                           variants=[bad])


def test_g0cb_rerun_is_idempotent_and_deterministic(g0_pipeline):
    from eval.hashing import hash_obj
    w, led = g0_pipeline["world"], g0_pipeline["led_cb"]
    cb = w["dev"]["arms"]["coinbase_only"]
    again = run_g0cb_study(cb["matrix"], cb["manifest"], w["contract"],
                           gate=g0_pipeline["gate"], ledger=led)
    assert led.n_effective_trials() == 4              # no double-count on rerun
    assert hash_obj(again) == hash_obj(g0_pipeline["res_cb"])


def test_g0cb_noise_control_fails():
    w = make_g0_world(n_dev_bars=250, n_holdout_bars=10, cb_signal=0.0, bn_signal=0.0,
                      seed=1)
    cb = w["dev"]["arms"]["coinbase_only"]
    res = run_g0cb_study(cb["matrix"], cb["manifest"], w["contract"], gate=GATE)
    assert res["g0cb_pass"] is False


def test_g0cb_leaky_feature_name_fails_closed(g0_world):
    w = g0_world
    cb = w["dev"]["arms"]["coinbase_only"]
    man = copy.deepcopy(cb["manifest"])
    man["feature_cols"] = man["feature_cols"] + ["cb_fwd_mid"]
    mat = cb["matrix"].copy()
    mat["cb_fwd_mid"] = mat["y_fwd_bps"]              # a genuine target leak
    with pytest.raises(ValueError, match="leaky"):
        run_g0cb_study(mat, man, w["contract"], gate=GATE)


def test_gate_changes_are_counted_as_new_trials(g0_world):
    """Codex PR#60 round-6: a post-hoc threshold change is ANOTHER TRIAL (staged
    protocol §2) — the resolved gate is part of the trial identity, so a rerun with a
    changed gate on the carried ledger adds multiplicity instead of silently reusing
    identities."""
    cb = g0_world["dev"]["arms"]["coinbase_only"]
    led = TrialLedger()
    run_g0cb_study(cb["matrix"], cb["manifest"], g0_world["contract"], gate=GATE,
                   ledger=led)
    assert led.n_effective_trials() == 4
    run_g0cb_study(cb["matrix"], cb["manifest"], g0_world["contract"],
                   gate={**GATE, "min_trades": 1}, ledger=led)
    assert led.n_effective_trials() == 8              # looser gate = 4 MORE trials


def test_g0_rejects_understated_declared_lookback(g0_world):
    """Codex PR#60 P1 regression pin: a build whose ACTUAL look-back exceeds the
    manifest's declared max_lookback_ns (and therefore the embargo sized to it) must
    fail closed BEFORE any candidate is evaluated — the under-embargoed CPCV would
    otherwise leak feature windows into test spans. Enforced by validate_frame inside
    _prepare_development_input on every G0 input path."""
    cb = g0_world["dev"]["arms"]["coinbase_only"]
    bad = cb["matrix"].copy()
    bad["t_feature_start"] = bad["t_event"] - 5 * (bad["t_event"] - bad["t_feature_start"])
    with pytest.raises(ValueError, match="observed look-back .* exceeds declared"):
        run_g0cb_study(bad, cb["manifest"], g0_world["contract"], gate=GATE)


def test_g0cb_rejects_unknown_gate_keys(g0_world):
    cb = g0_world["dev"]["arms"]["coinbase_only"]
    with pytest.raises(ValueError, match="unknown gate keys"):
        run_g0cb_study(cb["matrix"], cb["manifest"], g0_world["contract"],
                       gate={"min_tradez": 1})


def test_formal_g1_path_still_accepts_g0_builds(g0_world):
    """Backward compatibility: the unchanged per-manifest G1 runner consumes a G0 build
    (its partition-contract binding rides in the manifest `sources`, which v1 already
    allows) — the formal G1 path needed no migration."""
    from eval.runner import run_from_manifest
    cb = g0_world["dev"]["arms"]["coinbase_only"]
    man = copy.deepcopy(cb["manifest"])
    man["gate"] = GATE
    res = run_from_manifest(cb["matrix"], man)
    assert "10s" in res["horizons"] and "g1_pass" in res["horizons"]["10s"]
