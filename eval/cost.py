"""No-trade-band, fees-included PnL with PER-SAMPLE cost + spread; uniqueness-weighted Sharpe."""
from __future__ import annotations
import numpy as np


def net_pnl(forecast_bps, realized_bps, *, cost_bps, half_spread_bps=0.0,
            spread_crossings=2, margin_bps=0.0):
    """Trade when |forecast| > total per-sample band, where
    total_cost = cost_bps + spread_crossings*half_spread_bps. A mid-anchored taker round
    trip crosses the spread twice (buy at ask, sell at bid) -> spread_crossings=2.
    cost_bps/half_spread_bps may be scalar or per-sample arrays. Honest taker fills.
    Returns (pnl_per_sample, traded_mask, gross_pnl_per_sample)."""
    fc = np.asarray(forecast_bps, float); rr = np.asarray(realized_bps, float)
    total_cost = (np.asarray(cost_bps, float)
                  + spread_crossings * np.asarray(half_spread_bps, float)) * np.ones_like(fc)
    band = total_cost + margin_bps
    traded = np.abs(fc) > band
    gross = np.where(traded, np.sign(fc) * rr, 0.0)
    pnl = np.where(traded, gross - total_cost, 0.0)
    return pnl, traded, gross


def weighted_sharpe(pnl_per_sample, weights, *, trade_only: bool = True) -> float:
    """Uniqueness-weighted Sharpe. trade_only=True -> over traded samples (hit quality);
    trade_only=False -> over ALL decision samples incl. no-trade zeros (the strategy's
    sample/time-level Sharpe). Overlapping labels overcount, hence the weighting."""
    pnl = np.asarray(pnl_per_sample, float)
    w = np.asarray(weights, float)
    if trade_only:
        mask = pnl != 0.0
        p, ww = pnl[mask], w[mask]
    else:
        p, ww = pnl, w
    if len(p) < 2 or ww.sum() == 0:
        return 0.0
    mean = np.average(p, weights=ww)
    var = np.average((p - mean) ** 2, weights=ww)
    return float(mean / (np.sqrt(var) + 1e-12)) if var > 0 else 0.0
