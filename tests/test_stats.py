import numpy as np
from eval.stats import deflated_sharpe, pbo


def test_dsr_high_when_t_large_and_sr_clears_benchmark():
    d = deflated_sharpe(sr_hat=0.25, sr_trials_std=0.12, n_trials=4, T=3000, skew=0.0, kurt=3.0)
    assert d > 0.95


def test_dsr_low_when_sr_is_noise_max():
    d = deflated_sharpe(sr_hat=0.02, sr_trials_std=0.05, n_trials=1000, T=1500, skew=0.0, kurt=3.0)
    assert d < 0.5


def test_dsr_requires_two_trials():
    import pytest
    with pytest.raises(ValueError):
        deflated_sharpe(sr_hat=0.3, sr_trials_std=0.1, n_trials=1, T=100, skew=0.0, kurt=3.0)


def test_pbo_high_for_all_noise_trials():
    rng = np.random.default_rng(0)
    assert pbo(rng.standard_normal((400, 200)), s=8) > 0.35


def test_pbo_low_for_one_dominant_trial():
    rng = np.random.default_rng(0)
    M = rng.standard_normal((400, 50)) * 0.1
    M[:, 0] += 1.0
    assert pbo(M, s=8) < 0.1


def test_pbo_weights_are_threaded():
    # Uniform weights reproduce the unweighted result; a dense low-uniqueness cluster that
    # spuriously favors one config must change the gate input once it is down-weighted.
    rng = np.random.default_rng(1)
    M = rng.standard_normal((320, 6)) * 0.5
    M[:, 0] += 0.4                                   # config 0 has a mild genuine edge
    assert pbo(M, s=8, weights=np.ones(320)) == pbo(M, s=8)
    w = np.ones(320)
    M[:40, 1] += 6.0                                 # duplicated-exposure cluster favoring cfg 1
    w[:40] = 0.01                                    # ...but those rows are nearly non-unique
    assert pbo(M, s=8, weights=w) != pbo(M, s=8)


def test_pbo_counts_lower_half_boundary_ranks():
    # The IS-best config is OOS-worst on every imbalanced split -> selection overfitting.
    # The self-inclusive /N rank pegged those boundary cases at exactly 0.5 (logit 0), so
    # they were never counted and PBO collapsed to ~0. The (N+1) relative rank counts the
    # lower-half ranks, so an overfit ladder is no longer hidden.
    blocks = np.array([10.0, -10.0, 10.0, -10.0, 10.0, -10.0, 10.0, -10.0])
    M = np.column_stack([blocks, np.zeros(8)])
    assert pbo(M, s=8) > 0.3
