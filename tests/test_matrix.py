import pytest
from eval.matrix import validate_matrix, RESERVED
from eval.synthetic import make_matrix


def test_valid_matrix_passes():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    validate_matrix(df, feats)  # no raise


def test_missing_reserved_column_raises():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="t_available"):
        validate_matrix(df.drop(columns=["t_available"]), feats)


def test_feature_manifest_must_be_disjoint_from_reserved():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="reserved"):
        validate_matrix(df, feats + ["cost_bps"])


def test_unknown_feature_in_manifest_raises():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="not in matrix"):
        validate_matrix(df, feats + ["nonexistent_feat"])


def test_timing_invariants_enforced():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad.loc[0, "t_available"] = bad.loc[0, "t_event"] - 1
    with pytest.raises(ValueError, match="t_available >= t_event"):
        validate_matrix(bad, feats)
    bad2 = df.copy(); bad2.loc[0, "t_feature_start"] = bad2.loc[0, "t_event"] + 1
    with pytest.raises(ValueError, match="t_feature_start <= t_event"):
        validate_matrix(bad2, feats)


def test_label_out_of_domain_rejected():
    # Contract restricts label to {-1,0,+1}; a stray class would corrupt lgbm_clf silently.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad.loc[0, "label"] = 2
    with pytest.raises(ValueError, match="label"):
        validate_matrix(bad, feats)


def test_negative_costs_rejected():
    # Fail closed: negative cost/spread would invert the no-trade band and credit costs as
    # PnL, manufacturing trades and inflating the gate.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad.loc[0, "cost_bps"] = -1.0
    with pytest.raises(ValueError, match="cost_bps"):
        validate_matrix(bad, feats)
    bad2 = df.copy(); bad2.loc[0, "half_spread_bps"] = -0.1
    with pytest.raises(ValueError, match="half_spread_bps"):
        validate_matrix(bad2, feats)


def test_baseline_requires_synchronous_t_available():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)  # synthetic sets t_available == t_event
    bad = df.copy(); bad.loc[0, "t_available"] = bad.loc[0, "t_event"] + 1
    with pytest.raises(ValueError, match="t_available == t_event"):
        validate_matrix(bad, feats)
