"""Deflated Sharpe Ratio (Bailey & López de Prado 2014) + PBO via CSCV (NormalDist)."""
from __future__ import annotations
import numpy as np
from math import e
from itertools import combinations
from statistics import NormalDist

_N = NormalDist(0.0, 1.0)
_GAMMA = 0.5772156649015329


def deflated_sharpe(*, sr_hat, sr_trials_std, n_trials, T, skew, kurt) -> float:
    if n_trials < 2:
        raise ValueError("n_trials must be >= 2 for the multiple-testing benchmark")
    sr0 = sr_trials_std * ((1 - _GAMMA) * _N.inv_cdf(1 - 1.0 / n_trials)
                           + _GAMMA * _N.inv_cdf(1 - 1.0 / (n_trials * e)))
    denom = np.sqrt(max(1e-12, 1 - skew * sr_hat + ((kurt - 1) / 4.0) * sr_hat ** 2))
    z = (sr_hat - sr0) * np.sqrt(max(T - 1, 1)) / denom
    return float(_N.cdf(z))


def pbo(pnl_matrix: np.ndarray, *, s: int = 8, weights=None) -> float:
    """CSCV PBO over a (n_obs x n_trials) matrix; columns are distinct strategy configs.

    Block performance is a per-config mean over the rows in the block. Pass per-row
    ``weights`` (e.g. sample uniqueness) to weight that mean so heavily-overlapping-label
    clusters do not dominate IS/OOS rankings by duplicated exposure — consistent with the
    uniqueness weighting used everywhere else in the evaluator. weights=None -> equal
    weights (identical to the plain block mean)."""
    M = np.asarray(pnl_matrix, float)
    if M.shape[1] < 2:
        raise ValueError("PBO needs >= 2 trial configs (columns)")
    w = np.ones(M.shape[0]) if weights is None else np.asarray(weights, float)
    blocks = np.array_split(np.arange(M.shape[0]), s)
    logits = []
    for tr in combinations(range(s), s // 2):
        te = [b for b in range(s) if b not in tr]
        tr_rows = np.concatenate([blocks[b] for b in tr])
        te_rows = np.concatenate([blocks[b] for b in te])
        is_perf = np.average(M[tr_rows], axis=0, weights=w[tr_rows])
        oos_perf = np.average(M[te_rows], axis=0, weights=w[te_rows])
        best = int(np.argmax(is_perf))
        # Relative OOS rank of the IS-best config with an (N+1) denominator (CSCV). The
        # selected config is in the numerator, so a /N (mean) rank pegs the worst config at
        # 1/N and a lower-half boundary at exactly 0.5 (logit 0, never counted) -> PBO
        # underestimates overfitting. /(N+1) maps worst -> 1/(N+1) < 0.5, best -> N/(N+1) > 0.5.
        rank_count = int((oos_perf <= oos_perf[best]).sum())
        rank = min(max(rank_count / (M.shape[1] + 1), 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    return float((np.array(logits) < 0).mean())
