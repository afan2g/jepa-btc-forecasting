from eval.synthetic import make_matrix
from eval.study import run_study


def test_gate_passes_on_planted_signal():
    df, feats, lb = make_matrix(n=8000, signal_strength=6.0, seed=21)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert out["g1_pass"] is True
    assert out["best"]["net_pnl"] > out["rungs"]["naive"]["net_pnl"]
    # edge must be visible in BOTH regimes (or at least the tight-spread one)
    assert out["per_regime"]["tight"]["net_pnl"] > 0


def test_gate_fails_on_pure_noise():
    df, feats, lb = make_matrix(n=8000, signal_strength=0.0, seed=22)
    out = run_study(df, feats, cost_default=None, n_groups=6, k=2,
                    embargo_ns=lb, max_lookback_ns=lb)
    assert out["g1_pass"] is False
    assert out["winner"] is None          # no candidate cleared the gate
