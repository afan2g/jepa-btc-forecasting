import pytest
from eval.runner import resolve_gate, run_from_manifest, DEFAULT_GATE
from eval.synthetic import make_matrix


def test_resolve_gate_requires_block():
    with pytest.raises(ValueError, match="gate"):
        resolve_gate({"feature_cols": []})


def test_resolve_gate_rejects_unknown_keys():
    with pytest.raises(ValueError, match="unknown gate"):
        resolve_gate({"gate": {"min_tradez": 5}})


def test_resolve_gate_fills_defaults():
    g = resolve_gate({"gate": {"k": 3}})
    assert g["k"] == 3 and g["min_trades"] == DEFAULT_GATE["min_trades"]


def test_run_from_manifest_runs_and_echoes_resolved_gate():
    m, feats, lb = make_matrix(signal_strength=4.0, seed=8)
    man = {"feature_cols": feats, "embargo_ns": lb, "max_lookback_ns": lb,
           "gate": {"n_groups": 6, "k": 2}}
    res = run_from_manifest(m, man)
    assert res["gate"]["min_sample_sharpe"] == 0.0       # default filled into resolved config
    assert "10s" in res["horizons"] and "g1_pass" in res["horizons"]["10s"]
