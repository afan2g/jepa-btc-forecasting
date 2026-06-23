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


def pbo(pnl_matrix: np.ndarray, *, s: int = 8) -> float:
    """CSCV PBO over a (n_obs x n_trials) matrix; columns are distinct strategy configs."""
    M = np.asarray(pnl_matrix, float)
    if M.shape[1] < 2:
        raise ValueError("PBO needs >= 2 trial configs (columns)")
    blocks = np.array_split(np.arange(M.shape[0]), s)
    logits = []
    for tr in combinations(range(s), s // 2):
        te = [b for b in range(s) if b not in tr]
        is_perf = M[np.concatenate([blocks[b] for b in tr])].mean(0)
        oos_perf = M[np.concatenate([blocks[b] for b in te])].mean(0)
        best = int(np.argmax(is_perf))
        rank = min(max((oos_perf <= oos_perf[best]).mean(), 1e-6), 1 - 1e-6)
        logits.append(np.log(rank / (1 - rank)))
    return float((np.array(logits) < 0).mean())
