import numpy as np
from eval.cost import net_pnl, weighted_sharpe


def test_per_sample_cost_band():
    fc = np.array([3.0, 3.0, -5.0])
    rr = np.array([4.0, 4.0, -3.0])
    cost = np.array([5.0, 1.0, 1.0])     # sample0 band too high to trade; no spread here
    pnl, traded, gross = net_pnl(fc, rr, cost_bps=cost)
    assert traded.tolist() == [False, True, True]
    assert pnl[1] == 4.0 - 1.0           # +1*4 - 1
    assert pnl[2] == 3.0 - 1.0           # -1*-3 - 1
    assert gross[1] == 4.0


def test_spread_charged_via_two_crossings():
    fc = np.array([5.0]); rr = np.array([4.0])
    pnl, traded, gross = net_pnl(fc, rr, cost_bps=np.array([1.0]),
                                 half_spread_bps=np.array([0.5]))
    # total cost = 1 + 2*0.5 = 2 ; band 2 ; |5|>2 trade ; pnl = 4 - 2 ; gross = 4
    assert traded[0] and pnl[0] == 2.0 and gross[0] == 4.0


def test_zero_edge_loses_costs():
    rng = np.random.default_rng(0)
    rr = rng.standard_normal(5000) * 10
    fc = rng.standard_normal(5000) * 10
    pnl, traded, _ = net_pnl(fc, rr, cost_bps=np.full(5000, 2.0))
    assert pnl[traded].mean() < 0


def test_weighted_sharpe_respects_weights():
    pnl = np.array([1.0, 1.0, -10.0])
    w_low_on_outlier = np.array([1.0, 1.0, 0.01])
    w_equal = np.array([1.0, 1.0, 1.0])
    assert weighted_sharpe(pnl, w_low_on_outlier) > weighted_sharpe(pnl, w_equal)


def test_weighted_sharpe_zero_when_degenerate():
    assert weighted_sharpe(np.zeros(5), np.ones(5)) == 0.0
