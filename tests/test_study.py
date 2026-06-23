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


def test_per_regime_handles_nonrange_index():
    df, feats, lb = make_matrix(n=4000, signal_strength=4.0, seed=8)
    df.index = df["t_event"].to_numpy()           # non-range (timestamp) index
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert set(out["per_regime"]) == {"tight", "wide"}
