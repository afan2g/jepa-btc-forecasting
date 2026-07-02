"""Offline tests: joint engine-time resolver (origin_time-first, documented received_time fallback).

Pure pandas, no vendor I/O. Exercises the Requirement-4 policy (plan Task 3 / Requirement 9 item 8):
the delta frame and the `book` seed frame handed to recon must resolve to ONE shared, FULLY-populated
engine-time column — never a mixed exchange/capture axis, never a partially-populated column that
recon's `_require_populated` would crash on. `resolve_engine_time` returns the CLEANED frames the
caller feeds to recon, plus the chosen column, a fallback flag, and per-frame dropped-row counts
(recorded in the manifest, never silent)."""
import numpy as np
import pandas as pd
import pytest

from ingest import lake_binance as lb
from recon.ingest import _require_populated

NS = 1_000_000_000  # 1 s in ns


def _ts(vals):
    """int64-ns engine-time column from second offsets (None -> 0 = unpopulated, like real NaT)."""
    return pd.Series([0 if v is None else int(v * NS) for v in vals], dtype="int64")


def _frame(origin, received, tag="a"):
    n = len(origin)
    return pd.DataFrame({"origin_time": _ts(origin), "received_time": _ts(received),
                         "price": np.arange(n, dtype="float64"), "tag": [tag] * n})


def _assert_recon_ready(dfs, col):
    for df in dfs:
        _require_populated(df, col)  # raises if ANY <=0 row remains on the chosen column


def test_full_origin_time_is_selected_no_fallback():
    df = _frame([1, 2, 3], [1, 2, 3])
    col, fallback, clean, dropped = lb.resolve_engine_time(df)
    assert col == "origin_time"
    assert fallback is False
    assert dropped == [0]
    assert len(clean[0]) == 3       # nothing dropped
    _assert_recon_ready(clean, col)


def test_partial_origin_full_received_falls_back_whole_day():
    # 99.x%-but-not-100% origin_time (one unpopulated row) + fully populated received_time:
    # prefer the WHOLE-DAY received_time fallback (no data loss) over dropping rows.
    origin = [1, 2, None] + list(range(4, 400))
    received = list(range(1, 400))
    df = _frame(origin, received)
    col, fallback, clean, dropped = lb.resolve_engine_time(df)
    assert col == "received_time"
    assert fallback is True
    assert dropped == [0]           # whole-day fallback, no rows dropped
    assert len(clean[0]) == len(df)
    _assert_recon_ready(clean, col)


def test_neither_clock_full_drops_bad_rows_and_records_count():
    # origin_time >99% populated but not 100%, received_time ALSO holed -> keep the preferred
    # origin_time column and drop only its <=0 rows, recording the count.
    origin = [1, 2, None] + list(range(4, 400))   # 1 bad origin row (>99% populated)
    received = [1, None, 3] + list(range(4, 400))  # received holed too -> no whole-day fallback
    df = _frame(origin, received)
    col, fallback, clean, dropped = lb.resolve_engine_time(df)
    assert col == "origin_time"     # preferred exchange clock retained
    assert fallback is False        # dropping rows is not a clock fallback
    assert dropped == [1]
    assert len(clean[0]) == len(df) - 1
    _assert_recon_ready(clean, col)


def test_joint_resolution_forces_one_shared_column_across_frames():
    # (d) a partial-origin delta frame + a FULL-origin `book` seed frame must resolve BOTH to
    # received_time — never deltas on received_time and the seed on origin_time (a mixed
    # exchange/capture axis reorders seed/reseed events relative to deltas).
    delta = _frame([1, 2, None] + list(range(4, 400)), list(range(1, 400)), tag="delta")
    seed = _frame(list(range(1, 51)), list(range(1, 51)), tag="seed")  # FULL origin_time
    col, fallback, clean, dropped = lb.resolve_engine_time(delta, seed)
    assert col == "received_time"   # ONE shared column for BOTH frames
    assert fallback is True
    assert dropped == [0, 0]
    _assert_recon_ready(clean, col)  # both cleaned frames populated on the SAME column


def test_no_frames_raises():
    with pytest.raises(ValueError):
        lb.resolve_engine_time()


def test_no_usable_clock_raises():
    # both clocks badly unpopulated -> no shared >99% column -> fail loud (recon cannot run)
    df = _frame([1, None, None, None], [None, None, None, 1])
    with pytest.raises(ValueError):
        lb.resolve_engine_time(df)
