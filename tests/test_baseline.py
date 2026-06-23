import numpy as np
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


def test_naive_makes_no_trades():
    df, feats, _ = make_matrix(n=2000, signal_strength=3.0, seed=6)
    r = evaluate_config(df, feats, "naive", n_groups=5, k=1, embargo_ns=0)
    assert r.n_trades == 0 and r.net_pnl == 0.0


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
