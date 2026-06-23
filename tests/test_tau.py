import numpy as np
from eval.tau import predictivity_curve, estimate_tau


def test_recovers_known_decay():
    rng = np.random.default_rng(0)
    n = 20000
    x = rng.standard_normal(n)
    horizons = [1, 2, 5, 10, 20, 40, 80]
    rets = {h: np.exp(-h / 10.0) * x + rng.standard_normal(n) for h in horizons}
    curve = predictivity_curve(x, rets)
    assert curve[1] > curve[80]
    assert 5 <= estimate_tau(curve, frac=1 / np.e) <= 20
