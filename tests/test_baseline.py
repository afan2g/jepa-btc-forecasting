import numpy as np
import pandas as pd
import pytest
from eval.baseline import evaluate_config, CONFIGS
from eval.synthetic import make_matrix


def test_returns_fold_distribution_not_collapsed():
    df, feats, _ = make_matrix(n=3000, signal_strength=3.0, seed=2)
    r = evaluate_config(df, feats, "lgbm_reg", n_groups=6, k=2, embargo_ns=0)
    from math import comb
    assert len(r.fold_sharpes) == comb(6, 2)          # one Sharpe per CPCV fold
    assert r.per_sample_pnl.shape[0] == len(df)        # per-sample OOS PnL kept
    assert np.isfinite(r.per_sample_pnl).any()


def test_classifier_config_runs_and_signs_forecast():
    df, feats, _ = make_matrix(n=3000, signal_strength=4.0, seed=5)
    r = evaluate_config(df, feats, "lgbm_clf", n_groups=5, k=1, embargo_ns=0)
    assert r.name == "lgbm_clf" and r.net_pnl is not None


def test_dsr_input_is_aggregate_trade_sharpe():
    # The DSR input must be the Sharpe of the AGGREGATE OOS traded series (consistent with
    # t_eff), not the equal-weight fold mean, so a tiny high-Sharpe fold can't lift the gate.
    from eval.cost import weighted_sharpe
    df, feats, _ = make_matrix(n=3000, signal_strength=3.0, seed=2)
    r = evaluate_config(df, feats, "lgbm_reg", n_groups=6, k=2, embargo_ns=0)
    w = df["uniqueness"].to_numpy(float)
    finite = np.isfinite(r.per_sample_pnl)
    p = r.per_sample_pnl[finite]
    expected = weighted_sharpe(p, w[finite], traded=(p != 0.0))   # aggregate traded series
    assert r.trade_sharpe == expected
    assert len(r.fold_sharpes) > 1                                 # fold distribution still kept


def test_naive_makes_no_trades():
    df, feats, _ = make_matrix(n=2000, signal_strength=3.0, seed=6)
    r = evaluate_config(df, feats, "naive", n_groups=5, k=1, embargo_ns=0)
    assert r.n_trades == 0 and r.net_pnl == 0.0


def test_degenerate_empty_train_fold_does_not_crash():
    # Clustered long-horizon spans: the per-test-span purge empties train on EVERY fold.
    # naive/lgbm_clf already survive; ridge/lgbm_reg must degrade to a no-trade zero fold,
    # not abort the whole study (review finding: empty-train crash).
    import pandas as pd
    from eval.synthetic import FEATURES
    n = 600
    rng = np.random.default_rng(0)
    H = 10 ** 12                                   # horizon dwarfs the step -> spans all overlap
    t_event = (np.arange(n, dtype=np.int64) + 1) * 1000
    df = pd.DataFrame(rng.standard_normal((n, len(FEATURES))), columns=list(FEATURES))
    df["y_fwd_bps"] = rng.standard_normal(n)
    df["label"] = np.sign(df["y_fwd_bps"]).astype(int)
    df["t_event"] = t_event
    df["t_barrier"] = t_event + H
    df["t_feature_start"] = t_event - H
    df["t_available"] = t_event
    df["cost_bps"] = 1.0
    df["half_spread_bps"] = 0.5
    df["uniqueness"] = 0.5
    df["regime"] = "tight"
    df["horizon"] = "10s"
    for model in CONFIGS:
        r = evaluate_config(df, list(FEATURES), model, n_groups=6, k=2, embargo_ns=0)
        assert r.name == model            # completes without raising
        assert np.isfinite(r.net_pnl)     # degenerate folds contribute zeros, not nan/crash


def test_uniqueness_weight_is_passed_to_fit(monkeypatch):
    df, feats, _ = make_matrix(n=1500, signal_strength=3.0, seed=7)
    seen = {}
    import eval.baseline as B
    real = B._fit_predict
    def spy(model, Xtr, ytr, ltr, Xte, wtr, scale):
        seen["w_sum"] = float(np.asarray(wtr).sum()); return real(model, Xtr, ytr, ltr, Xte, wtr, scale)
    monkeypatch.setattr(B, "_fit_predict", spy)
    B.evaluate_config(df, feats, "ridge", n_groups=4, k=1, embargo_ns=0)
    assert seen["w_sum"] > 0


def test_evaluate_config_duplicate_guards():
    # Defense-in-depth for callers that bypass validate_matrix/validate_frame:
    # duplicated FRAME labels widen X; duplicated feature_cols ENTRIES double-weight
    # a column (and would sail past a width-only check, since df[["a","a"]] is width 2).
    df, feats, _ = make_matrix(n=200, signal_strength=1.0, seed=3)
    dup = pd.concat([df, df[[feats[0]]]], axis=1)
    with pytest.raises(ValueError, match="widened"):
        evaluate_config(dup, feats, "naive", n_groups=4, k=1, embargo_ns=0)
    with pytest.raises(ValueError, match="double-weight"):
        evaluate_config(df, feats + [feats[0]], "naive", n_groups=4, k=1, embargo_ns=0)


def test_feature_matrix_follows_manifest_order_not_frame_order():
    # Characterization pin (deliberately not failing-first): manifest order -> numpy
    # column order, regardless of frame column order (LightGBM reproducibility). Guards
    # against a future pandas behavior change or a rewrite of the selection idiom.
    df, feats, _ = make_matrix(n=100, signal_strength=1.0, seed=3)
    reordered = df[list(df.columns[::-1])]
    assert (reordered[feats].to_numpy(float) == df[feats].to_numpy(float)).all()
