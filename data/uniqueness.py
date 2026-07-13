"""Per-horizon concurrency uniqueness (E0.4, plan §F).

uniqueness_i = 1 / #{same-horizon label spans [t0_j, t1_j) covering t_event_i}. Spans are
HALF-OPEN: a span ending exactly at another row's t_event does not cover it — pinned to the
eval.synthetic._concurrency_uniqueness reference this module ports. Horizons must never be
mixed: the matrix is one row per bar x horizon and the runner evaluates each horizon in its
own groupby slice (eval/runner.py), so counting the duplicated 2s/10s/60s rows at the same
t_event against each other would depress weights, effective-trade counts, Sharpe, and PBO
for rungs that are never evaluated together.

All inputs are validated eagerly and fail closed: malformed spans or ambiguous duplicate
row keys raise instead of producing plausible-looking weights.
"""
from __future__ import annotations
import math
from typing import NamedTuple

import numpy as np


def _validated_times(name: str, x) -> np.ndarray:
    a = np.asarray(x)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-d; got shape {a.shape}")
    # Plain int/float nanoseconds only: timedelta64 satisfies issubdtype(..., np.number),
    # so gate on dtype.kind to keep NaT and datetime-like inputs out with an honest error.
    if a.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array of nanosecond times "
                         f"(int or float); got dtype {a.dtype}")
    if a.dtype.kind == "f" and not np.isfinite(a).all():
        raise ValueError(f"{name} contains non-finite values")
    return a


def _validated_spans(t_event, t_barrier) -> tuple[np.ndarray, np.ndarray]:
    t0 = _validated_times("t_event", t_event)
    t1 = _validated_times("t_barrier", t_barrier)
    if len(t0) != len(t1):
        raise ValueError(f"t_event and t_barrier lengths differ: {len(t0)} vs {len(t1)}")
    if not (t1 > t0).all():
        bad = int((t1 <= t0).sum())
        raise ValueError(f"{bad} label span(s) have t_barrier <= t_event (reversed or "
                         f"zero-length); every label span must resolve strictly after "
                         f"its t_event")
    return t0, t1


def concurrency_uniqueness(t_event, t_barrier) -> np.ndarray:
    """Uniqueness for ONE horizon's rows: 1 / (# spans [t_event_j, t_barrier_j) covering
    t_event_i). Output is aligned to the input row order; values are in (0, 1] because a
    row's own (validated, positive-length) span always covers its t_event. Duplicate
    t_event values are ambiguous row keys under the one-row-per-(t_event, horizon) matrix
    contract and fail closed."""
    t0, t1 = _validated_spans(t_event, t_barrier)
    if len(np.unique(t0)) != len(t0):
        raise ValueError("duplicate t_event within a horizon: the matrix contract is one "
                         "row per (t_event, horizon); duplicated keys make concurrency "
                         "weighting ambiguous")
    t0s = np.sort(t0)
    t1s = np.sort(t1)
    started = np.searchsorted(t0s, t0, side="right")   # spans with t0_j <= t_event_i
    ended = np.searchsorted(t1s, t0, side="right")     # spans with t1_j <= t_event_i
    conc = np.maximum(started - ended, 1)              # defensive; >=1 for validated spans
    return 1.0 / conc


def _validated_horizon(horizon, n: int) -> np.ndarray:
    tag = np.asarray(horizon, dtype=object)
    if tag.ndim != 1:
        raise ValueError(f"horizon must be 1-d; got shape {tag.shape}")
    if len(tag) != n:
        raise ValueError(f"horizon length ({len(tag)}) differs from t_event length ({n})")
    for v in tag:
        if not isinstance(v, str) or not v:
            raise ValueError(f"horizon tags must be non-empty strings (matrix contract: "
                             f"a str tag like '2s'/'10s'/'60s'); got {v!r}")
    return tag


def _validated_lookbacks(t_event, t_feature_start) -> np.ndarray:
    te = _validated_times("t_event", t_event)
    tfs = _validated_times("t_feature_start", t_feature_start)
    if len(te) != len(tfs):
        raise ValueError(f"t_event and t_feature_start lengths differ: "
                         f"{len(te)} vs {len(tfs)}")
    if not (tfs <= te).all():                # compare BEFORE subtracting: an unsigned
        bad = int((tfs > te).sum())          # difference would wrap, not go negative
        raise ValueError(f"{bad} row(s) have t_feature_start > t_event (negative "
                         f"look-back); the feature window must end at the decision time")
    lb = te - tfs
    if lb.dtype.kind == "i" and (lb < 0).any():
        # tfs <= te held, so a negative difference can only be signed int64 wraparound
        # (opposite-sign extremes); returning it would understate the embargo as garbage.
        raise ValueError("t_event - t_feature_start overflows int64; look-back times are "
                         "outside any representable nanosecond epoch range")
    return lb


def max_lookback_ns(t_event, t_feature_start) -> int:
    """max(t_event - t_feature_start) over the RETAINED rows given (apply any look-back
    cap mask before calling). Use the returned value as BOTH the manifest max_lookback_ns
    and embargo_ns: cpcv_splits applies the embargo from the merged test interval's upper
    bound (max t_barrier over test rows), which already includes the label horizon, so the
    only clearance the embargo must add is the feature look-back. Adding the horizon here
    would double-count it and needlessly purge clean train rows after every test block."""
    lb = _validated_lookbacks(t_event, t_feature_start)
    if len(lb) == 0:
        raise ValueError("max_lookback_ns is undefined for zero rows; an empty matrix "
                         "cannot size an embargo")
    return int(math.ceil(lb.max()))          # ceil: a fractional float look-back must
                                             # round UP — truncation would under-embargo


class LookbackCap(NamedTuple):
    keep: np.ndarray                 # bool mask aligned to the input rows
    n_dropped: int
    cap_ns: int
    retained_max_lookback_ns: int    # exact max over kept rows = manifest/embargo value


def apply_lookback_cap(t_event, t_feature_start, *, cap_ns) -> LookbackCap:
    """Outlier-robust look-back sizing: DROP every row whose observed look-back
    (t_event - t_feature_start) exceeds cap_ns, and report the exact retained maximum to
    declare as max_lookback_ns/embargo_ns. Dropping is the only sound policy — flagging a
    beyond-cap row while keeping it fails validate_frame/run_study (they recompute the max
    from emitted rows), and clipping t_feature_start understates the true feature window
    (under-embargo). cap_ns is an explicit caller-chosen parameter (e.g. from a high
    percentile of observed look-backs) so the policy is deterministic and auditable."""
    lb = _validated_lookbacks(t_event, t_feature_start)
    if not (isinstance(cap_ns, (int, np.integer))
            or (isinstance(cap_ns, float) and math.isfinite(cap_ns)
                and cap_ns.is_integer())) or cap_ns <= 0:
        raise ValueError(f"cap_ns must be a positive whole number of nanoseconds (a "
                         f"fractional cap cannot be reported exactly); got {cap_ns!r}")
    keep = lb <= cap_ns
    if not keep.any():
        raise ValueError(f"look-back cap {cap_ns} drops every row ({len(lb)} of "
                         f"{len(lb)}); the cap is misconfigured for this build")
    return LookbackCap(keep=keep, n_dropped=int((~keep).sum()), cap_ns=int(cap_ns),
                       retained_max_lookback_ns=int(math.ceil(lb[keep].max())))


def uniqueness_by_horizon(t_event, t_barrier, horizon) -> np.ndarray:
    """Uniqueness for a multi-horizon bar x horizon matrix: concurrency is counted
    INDEPENDENTLY within each horizon tag, so rows from one horizon never contribute to
    another's weights. Output is aligned positionally to the input rows, so any caller
    ordering (and any permutation of it) yields the same value per row."""
    t0, t1 = _validated_spans(t_event, t_barrier)
    tag = _validated_horizon(horizon, len(t0))
    out = np.empty(len(t0), dtype=float)
    for h in np.unique(tag):
        m = tag == h
        try:
            out[m] = concurrency_uniqueness(t0[m], t1[m])
        except ValueError as e:
            raise ValueError(f"horizon {h!r}: {e}") from e
    return out
