"""Deterministic content hashes for the G0 evaluator artifacts (issue #52).

Everything the staged protocol pins — partition contracts, trial ledgers, freeze
artifacts, matched-arm row universes, CPCV split assignments — is hashed here with ONE
canonical encoding, so "same hash" always means "same logical content" and never depends
on dict insertion order, file bytes, or pandas/pyarrow versions (the bar/label plan §I
determinism rule: logical rows, not file bytes)."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pandas as pd

from data.cv import make_time_groups


def canonical_json(obj, *, exclude_keys: tuple = ()) -> str:
    """Canonical JSON: sorted keys, compact separators, no NaN/Infinity (strict JSON),
    top-level `exclude_keys` removed (e.g. volatile generated_at timestamps)."""
    if isinstance(obj, dict) and exclude_keys:
        obj = {k: v for k, v in obj.items() if k not in exclude_keys}
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hash_obj(obj, *, exclude_keys: tuple = ()) -> str:
    return hashlib.sha256(canonical_json(obj, exclude_keys=exclude_keys).encode()).hexdigest()


def _column_bytes(s: pd.Series) -> bytes:
    """Deterministic bytes for one column. Numeric columns hash their float64/int64 buffer
    (exact value identity); everything else hashes the repr of each element (order-preserving,
    unambiguous separators)."""
    if pd.api.types.is_integer_dtype(s):
        return np.ascontiguousarray(s.to_numpy(np.int64)).tobytes()
    if pd.api.types.is_float_dtype(s) or pd.api.types.is_bool_dtype(s):
        return np.ascontiguousarray(s.to_numpy(np.float64)).tobytes()
    return "\x1f".join(repr(v) for v in s.tolist()).encode()


def canonical_row_order(df: pd.DataFrame) -> pd.DataFrame:
    """The canonical row order every hash and every cross-arm comparison uses: stable sort
    by (t_event, horizon). The producer guarantees one row per (t_event, horizon); the
    matched-arm validator rejects duplicates before relying on this order."""
    return df.sort_values(["t_event", "horizon"], kind="mergesort").reset_index(drop=True)


def matrix_content_hash(df: pd.DataFrame, cols) -> str:
    """Content hash over `cols` of `df` in canonical row order. Used to prove matched
    G0-XV arms share identical reserved rows (labels, costs, timing, uniqueness, regime,
    horizon) while their feature columns differ."""
    cols = list(cols)
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"matrix_content_hash: columns missing from frame: {missing}")
    ordered = canonical_row_order(df)
    h = hashlib.sha256()
    for c in cols:
        h.update(c.encode())
        h.update(b"\x00")
        h.update(_column_bytes(ordered[c]))
        h.update(b"\x00")
    return h.hexdigest()


def split_hash(df: pd.DataFrame, *, n_groups: int, k: int, embargo_ns: int) -> str:
    """Hash of the CPCV split assignment the study will consume: the time-group id of every
    row (canonical order) plus the split parameters. Identical (t_event, t_barrier) universes
    with identical parameters produce identical splits; anything else fails the matched-arm
    check downstream."""
    ordered = canonical_row_order(df)
    groups = make_time_groups(ordered["t_event"].to_numpy(), n_groups)
    h = hashlib.sha256()
    h.update(canonical_json({"n_groups": n_groups, "k": k, "embargo_ns": embargo_ns}).encode())
    h.update(np.ascontiguousarray(groups, dtype=np.int64).tobytes())
    h.update(np.ascontiguousarray(ordered["t_barrier"].to_numpy(np.int64)).tobytes())
    return h.hexdigest()
