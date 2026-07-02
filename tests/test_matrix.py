import numpy as np
import pandas as pd
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


def test_duplicate_frame_columns_rejected():
    # matrix[feature_cols] returns EVERY label match: a duplicated label silently widens X.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    dup = pd.concat([df, df[["cvd"]]], axis=1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_matrix(dup, feats)


def test_duplicate_manifest_feature_entries_rejected():
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    with pytest.raises(ValueError, match="duplicate"):
        validate_matrix(df, feats + [feats[0]])


def test_non_numeric_feature_rejected():
    # to_numpy(float) would die opaquely on object dtype; fail closed with the column name.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad["cvd"] = "high"
    with pytest.raises(ValueError, match="numeric"):
        validate_matrix(bad, feats)


def test_nan_feature_or_target_rejected():
    # NaN features crash Ridge mid-study while LightGBM would silently mask them;
    # NaN y_fwd_bps corrupts PnL; NA cost silently forces no-trade (band is NaN) and
    # NA uniqueness poisons the weighted Sharpe. Imputation belongs upstream.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad.loc[0, "cvd"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_matrix(bad, feats)
    bad2 = df.copy(); bad2.loc[0, "y_fwd_bps"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        validate_matrix(bad2, feats)
    bad3 = df.copy()
    bad3["cost_bps"] = bad3["cost_bps"].astype("Float64"); bad3.loc[0, "cost_bps"] = pd.NA
    with pytest.raises(ValueError, match="NaN"):
        validate_matrix(bad3, feats)


def test_infinite_feature_or_cost_rejected():
    # np.inf passes both the dtype and isna screens: Ridge/sklearn abort on infinite X
    # mid-study, and an infinite cost silently forces no-trade — require finite values.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy(); bad.loc[0, "cvd"] = np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_matrix(bad, feats)
    bad2 = df.copy(); bad2.loc[0, "cost_bps"] = -np.inf
    with pytest.raises(ValueError, match="infinite"):
        validate_matrix(bad2, feats)


def test_nullable_or_datetime_timing_fails_closed():
    # Plain comparisons fail open under pd.NA (Series.all() skips NA) and run_study's
    # observed-lookback .max() skips NA rows — mirror validate_frame's integer/non-null
    # timing guard so the legacy path and direct run_study callers are covered too.
    df, feats, _ = make_matrix(signal_strength=1.0, seed=1)
    bad = df.copy()
    bad["t_barrier"] = bad["t_barrier"].astype("Int64"); bad.loc[0, "t_barrier"] = pd.NA
    with pytest.raises(ValueError, match="null"):
        validate_matrix(bad, feats)
    bad2 = df.copy()
    bad2["t_event"] = pd.to_datetime(bad2["t_event"])  # datetime64, not int ns
    with pytest.raises(ValueError, match="integer"):
        validate_matrix(bad2, feats)
