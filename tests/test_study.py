import numpy as np
from eval.study import run_study
from eval.synthetic import make_matrix


def test_study_reports_regimes_and_dsr_pbo():
    df, feats, lb = make_matrix(n=4000, signal_strength=4.0, seed=8)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert "tight" in out["per_regime"] and "wide" in out["per_regime"]
    g = out["best"]
    assert {"dsr", "pbo", "net_pnl", "name"} <= g.keys()


def test_embargo_must_cover_lookback():
    import pytest
    df, feats, lb = make_matrix(n=1500, signal_strength=2.0, seed=9)
    with pytest.raises(ValueError, match="embargo"):
        run_study(df, feats, cost_default=None, n_groups=5, k=1,
                  embargo_ns=lb - 1, max_lookback_ns=lb)


def test_understated_max_lookback_is_rejected():
    import pytest
    df, feats, lb = make_matrix(n=1500, signal_strength=2.0, seed=9)
    # The matrix's true per-row look-back is lb = max(t_event - t_feature_start). Declare a
    # SMALLER max_lookback_ns with an embargo that still clears the embargo>=max_lookback
    # guard, so the ground-truth cross-check is the only thing that can catch the
    # understatement (which would otherwise leak the test span into post-block train rows).
    with pytest.raises(ValueError, match="look-back"):
        run_study(df, feats, cost_default=None, n_groups=5, k=1,
                  embargo_ns=lb // 2, max_lookback_ns=lb // 2)


def test_min_eff_trades_floor_blocks_pass():
    df, feats, lb = make_matrix(n=8000, signal_strength=6.0, seed=21)  # seed known to pass
    base = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                     embargo_ns=lb, max_lookback_ns=lb)
    assert base["g1_pass"] is True                                     # passes with default floor
    blocked = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                        embargo_ns=lb, max_lookback_ns=lb, min_eff_trades=10**9)
    assert blocked["g1_pass"] is False and blocked["winner"] is None   # the floor is the cause


def test_min_sample_sharpe_floor_blocks_pass():
    df, feats, lb = make_matrix(n=8000, signal_strength=6.0, seed=21)
    base = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                     embargo_ns=lb, max_lookback_ns=lb)
    assert base["g1_pass"] is True
    blocked = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                        embargo_ns=lb, max_lookback_ns=lb, min_sample_sharpe=10.0)
    assert blocked["g1_pass"] is False and blocked["winner"] is None


def test_winner_is_a_passing_candidate():
    df, feats, lb = make_matrix(n=8000, signal_strength=6.0, seed=8)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    if out["g1_pass"]:
        assert out["winner"] is not None
        assert out["rungs"][out["winner"]]["passes_solo"] is True


def test_study_pbo_is_invariant_to_row_order():
    # PBO's CSCV blocks must be chronological. A ModelMatrix not pre-sorted by t_event (e.g.
    # concatenated parquet partitions) must yield the SAME PBO/G1 as the sorted one. Use a
    # time-localized signal (first half predictive, second half noise) so PBO is genuinely
    # order-sensitive -> without the sort, the shuffled run diverges.
    import pandas as pd
    from eval.synthetic import FEATURES
    n = 4800
    rng = np.random.default_rng(0)
    X = rng.standard_normal((n, len(FEATURES)))
    f = X[:, 0] + np.tanh(X[:, 1]) * 1.5
    half = n // 2
    y = np.empty(n)
    y[:half] = 6.0 * f[:half] + rng.standard_normal(half) * 8
    y[half:] = rng.standard_normal(n - half) * 8
    H = 10_000_000_000
    te = (np.arange(n) + 1) * (H // 4)
    df = pd.DataFrame(X, columns=list(FEATURES))
    df["y_fwd_bps"] = y; df["label"] = np.sign(y).astype(int)
    df["t_event"] = te; df["t_barrier"] = te + H
    df["t_feature_start"] = te - H; df["t_available"] = te
    df["cost_bps"] = 1.5; df["half_spread_bps"] = 0.6; df["uniqueness"] = 0.25
    df["regime"] = np.where(X[:, 3] > 0, "tight", "wide"); df["horizon"] = "10s"
    feats = list(FEATURES)
    base = run_study(df, feats, cost_default=None, n_groups=6, k=2, embargo_ns=H, max_lookback_ns=H)
    shuffled = df.sample(frac=1.0, random_state=1).reset_index(drop=True)
    out = run_study(shuffled, feats, cost_default=None, n_groups=6, k=2, embargo_ns=H, max_lookback_ns=H)
    assert out["pbo"] == base["pbo"]          # PBO reproducible regardless of input row order
    assert out["g1_pass"] == base["g1_pass"]


def test_per_regime_handles_nonrange_index():
    df, feats, lb = make_matrix(n=4000, signal_strength=4.0, seed=8)
    df.index = df["t_event"].to_numpy()           # non-range (timestamp) index
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert set(out["per_regime"]) == {"tight", "wide"}
