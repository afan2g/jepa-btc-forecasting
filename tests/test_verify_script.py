"""The E0.2 capture script must be importable WITHOUT touching billable Lake APIs.

Importing it here would raise / hit the network if the live work ran at module top level;
that it imports cleanly (no .env, no credentials) proves the main() guard works.
"""
import pandas as pd
import pytest

# Importable as a namespace package via `python -m pytest` from the repo root.
verify = pytest.importorskip("scripts.verify_book_delta_v2")


def test_engine_col_prefers_populated_origin_time():
    base = 1668470400000000000
    df = pd.DataFrame({
        "origin_time": pd.to_datetime([base, base + 1]),
        "received_time": pd.to_datetime([base, base + 1]),
    })
    assert verify.engine_col(df) == "origin_time"


def test_engine_col_falls_back_when_origin_time_unpopulated():
    base = 1668470400000000000
    df = pd.DataFrame({
        "origin_time": pd.to_datetime([pd.NaT, pd.NaT]),   # present but empty (§4 fallback)
        "received_time": pd.to_datetime([base, base + 1]),
    })
    assert verify.engine_col(df) == "received_time"


def test_engine_col_raises_when_none_populated():
    df = pd.DataFrame({"origin_time": pd.to_datetime([pd.NaT])})
    with pytest.raises(SystemExit, match="no populated engine-time"):
        verify.engine_col(df)
