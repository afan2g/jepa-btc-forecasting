import pytest
from eval.runner import resolve_gate, run_from_manifest, DEFAULT_GATE
from eval.synthetic import make_manifest, make_matrix

GATE = {"n_groups": 4, "k": 2, "min_trades": 1, "min_eff_trades": 1.0}


def _v1(feats, lb, **over):
    return make_manifest(feats, lb, gate=dict(GATE), **over)


# ---------- resolve_gate (contract unchanged) ----------

def test_resolve_gate_requires_block():
    with pytest.raises(ValueError, match="gate"):
        resolve_gate({"feature_cols": []})


def test_resolve_gate_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown gate"):
        resolve_gate({"gate": {"min_tradez": 5}})


def test_resolve_gate_fills_defaults():
    g = resolve_gate({"gate": {"k": 3}})
    assert g["k"] == 3 and g["min_trades"] == DEFAULT_GATE["min_trades"]


# ---------- v1 path ----------

def test_v1_manifest_runs_and_echoes_gate_and_identity():
    m, feats, lb = make_matrix(n=900, signal_strength=4.0, seed=8)
    res = run_from_manifest(m, _v1(feats, lb))
    assert res["gate"]["min_sample_sharpe"] == 0.0       # default filled into resolved config
    assert "10s" in res["horizons"] and "g1_pass" in res["horizons"]["10s"]
    assert res["manifest"]["dataset_id"] == "synthetic"  # reproducible from its own output
    assert res["manifest"]["feature_cols"] == feats


def test_v1_targets_must_match_baseline_consumption():
    # evaluate_config trains on exactly {y_fwd_bps, label}; a manifest declaring fewer
    # (or extra reserved targets) misdescribes what the study consumed.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="exactly"):
        run_from_manifest(m, _v1(feats, lb, target_cols=["y_fwd_bps"]))


def test_v1_availability_lag_rejected_for_synchronous_baseline():
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="synchronous"):
        run_from_manifest(m, _v1(feats, lb, availability_lag_ns=5_000_000))


def test_v1_declared_horizon_missing_from_matrix_rejected():
    # validate_frame checks frame tags are declared; the runner checks the converse so a
    # manifest declaring {10s, 60s} over a 10s-only build cannot silently return no 60s row.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    # NB: lb is already bound positionally to max_lookback_ns inside _v1 -> make_manifest;
    # passing max_lookback_ns=lb again via **over would TypeError at setup.
    man = _v1(feats, lb, horizons={"10s": 10_000_000_000, "60s": 60_000_000_000})
    with pytest.raises(ValueError, match="missing from the matrix"):
        run_from_manifest(m, man)


# ---------- legacy path (branch deleted in phase 3) ----------

def test_legacy_manifest_dict_still_runs():
    # Phase-3 removal target: delete this test with the legacy branch.
    m, feats, lb = make_matrix(signal_strength=4.0, seed=8)
    man = {"feature_cols": feats, "embargo_ns": lb, "max_lookback_ns": lb,
           "gate": {"n_groups": 6, "k": 2}}
    res = run_from_manifest(m, man)
    assert res["gate"]["min_sample_sharpe"] == 0.0
    assert "10s" in res["horizons"]


def test_legacy_manifest_rejects_leaky_feature_names():
    # The legacy branch skips validate_frame; the leak screen must never be skipped.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    m["mid_fwd_10s"] = 0.0
    man = {"feature_cols": feats + ["mid_fwd_10s"], "embargo_ns": lb,
           "max_lookback_ns": lb, "gate": {"n_groups": 4, "k": 2}}
    with pytest.raises(ValueError, match="leak"):
        run_from_manifest(m, man)


def test_legacy_manifest_missing_keys_fail_with_contract_error():
    # Raw KeyError is not a contract error; name the missing keys.
    m, feats, lb = make_matrix(n=64, signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="legacy manifest missing"):
        run_from_manifest(m, {"feature_cols": feats, "gate": {"n_groups": 4, "k": 2}})
