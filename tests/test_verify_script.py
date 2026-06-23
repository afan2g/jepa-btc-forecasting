"""The E0.2 capture script must be importable WITHOUT touching billable Lake APIs.

Importing it would raise / hit the network if the live work ran at module top level;
that it imports cleanly (no .env, no credentials) proves the main() guard works. The
engine-time-axis selection now lives in recon.ingest (see test_ingest.py).
"""
import pytest

# Importable as a namespace package via `python -m pytest` from the repo root.
verify = pytest.importorskip("scripts.verify_book_delta_v2")


def test_capture_script_imports_without_side_effects():
    # main() (live Lake work) must NOT have run on import; helpers must be callable.
    assert callable(verify.main)
    assert callable(verify.lake_session)
    # The script reuses the centralized §5.3 axis selector rather than its own copy.
    assert verify.shared_engine_time_col is not None
