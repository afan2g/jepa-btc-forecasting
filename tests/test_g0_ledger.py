"""Trial ledger: identity discipline, idempotent deterministic re-registration,
fail-closed conflicts/tampering, order-independent hashing, and G0-CB history import
(the DSR effective-trial-count substrate)."""
import json

import pytest

from eval.ledger import TrialLedger, identity_hash, trial_identity


def _ident(**over):
    base = dict(protocol="g0cb", arm="coinbase_only", dataset_id="d", build_id="b",
                feature_cols=["f1", "f2"], config="lgbm_reg", horizon="10s")
    base.update(over)
    return trial_identity(**base)


RESULT = {"net_pnl": 1.5, "trade_sharpe": 0.4, "n_trades": 10}


def test_identity_validation_fails_closed():
    with pytest.raises(ValueError, match="protocol"):
        _ident(protocol="g2")
    with pytest.raises(ValueError, match="arm"):
        _ident(arm="")
    with pytest.raises(ValueError, match="feature_cols"):
        _ident(feature_cols=[])
    with pytest.raises(ValueError, match="exactly the fields"):
        identity_hash({"protocol": "g0cb"})


def test_identity_hash_covers_every_dimension():
    base = identity_hash(_ident())
    assert identity_hash(_ident(arm="combined")) != base
    assert identity_hash(_ident(build_id="b2")) != base
    assert identity_hash(_ident(config="ridge")) != base
    assert identity_hash(_ident(horizon="60s")) != base
    assert identity_hash(_ident(variant="sub", variant_params={"feature_cols": ["f1"]})) != base
    assert identity_hash(_ident(feature_cols=["f2", "f1"])) != base   # ORDERED features


def test_register_dedups_identical_and_rejects_conflicts():
    led = TrialLedger()
    led.register(_ident(), RESULT)
    led.register(_ident(), dict(RESULT))              # identical rerun -> no-op
    assert led.n_effective_trials() == 1
    with pytest.raises(ValueError, match="DIFFERENT result"):
        led.register(_ident(), {**RESULT, "net_pnl": 9.9})


def test_nonfinite_results_are_sanitized():
    led = TrialLedger()
    e = led.register(_ident(), {**RESULT, "pbo": float("nan")})
    assert e["result"]["pbo"] is None                 # strict JSON, hashable


def test_ledger_hash_is_order_independent():
    a, b = TrialLedger(), TrialLedger()
    a.register(_ident(), RESULT)
    a.register(_ident(config="ridge"), RESULT)
    b.register(_ident(config="ridge"), RESULT)
    b.register(_ident(), RESULT)
    assert a.ledger_hash() == b.ledger_hash()
    b.register(_ident(config="naive"), RESULT)
    assert a.ledger_hash() != b.ledger_hash()


def test_import_history_counts_and_conflicts():
    cb, xv = TrialLedger(), TrialLedger()
    cb.register(_ident(), RESULT)
    cb.register(_ident(config="ridge"), RESULT)
    xv.register(_ident(protocol="g0xv"), RESULT)
    assert xv.import_history(cb) == 2
    assert xv.n_effective_trials() == 3
    assert xv.import_history(cb) == 0                 # idempotent
    bad = TrialLedger()
    bad.register(_ident(), {**RESULT, "net_pnl": -1.0})
    with pytest.raises(ValueError, match="DIFFERENT result"):
        xv.import_history(bad)


def test_save_load_roundtrip_and_tamper_detection(tmp_path):
    led = TrialLedger()
    led.register(_ident(), RESULT)
    led.register(_ident(config="ridge"), RESULT)
    path = tmp_path / "ledger.json"
    led.save(path)
    loaded = TrialLedger.load(path)
    assert loaded.ledger_hash() == led.ledger_hash()
    assert loaded.n_effective_trials() == 2

    payload = json.loads(path.read_text())
    payload["entries"][0]["result"]["net_pnl"] = 99.0            # tamper a result
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="tampered or corrupted"):
        TrialLedger.load(path)

    payload = json.loads(path.read_text())
    payload["entries"][0]["result"]["net_pnl"] = RESULT["net_pnl"]
    payload["ledger_sha256"] = "0" * 64                          # tamper the pin
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="tampered or corrupted"):
        TrialLedger.load(path)
