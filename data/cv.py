"""Purged + embargoed Combinatorial Purged CV (López de Prado).

Purges per TEST INTERVAL: the test rows' spans are merged into disjoint intervals,
and a train row is purged only if it overlaps one of those intervals (or starts within
the embargo after one). This stays correct for NON-CONTIGUOUS CPCV combos, where a
union-span purge would wipe out nearly all training data.
"""
from __future__ import annotations
import numpy as np
from itertools import combinations


def make_time_groups(t_event: np.ndarray, n_groups: int) -> np.ndarray:
    order = np.argsort(t_event, kind="stable")
    rank = np.empty(len(t_event), dtype=np.int64)
    rank[order] = np.arange(len(t_event))
    return (rank * n_groups // len(t_event)).astype(int)


def _merge_intervals(lo: np.ndarray, hi: np.ndarray):
    order = np.argsort(lo, kind="stable")
    lo, hi = lo[order], hi[order]
    merged = []
    for a, b in zip(lo, hi):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return merged


def cpcv_splits(t_event, t0, t1, *, n_groups: int, k: int, embargo_ns: int):
    """Yield (train_idx, test_idx) for every k-of-n_groups combination. embargo_ns is
    REQUIRED (set it ≥ the longest feature look-back to avoid feature-window leakage)."""
    t0 = np.asarray(t0); t1 = np.asarray(t1)
    groups = make_time_groups(t_event, n_groups)
    for combo in combinations(range(n_groups), k):
        test_mask = np.isin(groups, combo)
        test_idx = np.where(test_mask)[0]
        purge = np.zeros(len(t0), bool)
        for lo, hi in _merge_intervals(t0[test_idx], t1[test_idx]):
            purge |= (t0 <= hi) & (t1 >= lo)                 # span overlap
            purge |= (t0 > hi) & (t0 <= hi + embargo_ns)     # embargo after the block
        train_idx = np.where(~test_mask & ~purge)[0]
        yield train_idx, test_idx
