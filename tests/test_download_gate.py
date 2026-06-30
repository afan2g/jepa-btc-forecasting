"""The CoinAPI backfill gate (docs/data.md §5a/§8): a backfill-scale pull is refused until the
recon-parity + reseed gates pass, so the documented 'do not backfill' decision is enforced in code,
not just prose. A single overlap day (the parity pilot) and a small multi-day --sample-mb smoke stay
allowed; an explicit --allow-backfill overrides.

The gate lives in ingest/_common.py (stdlib-only) precisely so this CI-safe unit test does NOT import
the downloader's pyarrow/boto3/coinapi_flatfiles deps (not in pyproject's default dependencies)."""
import datetime as dt
import importlib.util
import pathlib

import pytest

# ingest/ is not a package; load the lightweight _common module by path. It imports stdlib only.
_SPEC = importlib.util.spec_from_file_location(
    "ingest_common",
    pathlib.Path(__file__).resolve().parents[1] / "ingest" / "_common.py",
)
common = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(common)

D = dt.date
gate = common.check_backfill_gate
CAP = common.SMOKE_SAMPLE_CAP_MB


def test_single_day_pull_is_allowed():
    # the parity pilot: one overlap day, no override needed (size irrelevant for a single day)
    gate(D(2025, 6, 1), D(2025, 6, 1), sample_mb=0, allow_backfill=False)
    gate(D(2025, 6, 1), D(2025, 6, 1), sample_mb=10_000, allow_backfill=False)


def test_multi_day_full_pull_is_blocked(capsys):
    with pytest.raises(SystemExit) as ei:
        gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=0, allow_backfill=False)
    assert ei.value.code == common.BACKFILL_GATE_EXIT == 4
    assert "backfill" in capsys.readouterr().err.lower()


def test_multi_day_small_smoke_sample_is_allowed():
    # a small smoke across a range is bounded/cheap -> allowed without override
    gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=CAP, allow_backfill=False)
    gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=8, allow_backfill=False)


def test_multi_day_oversized_sample_is_blocked(capsys):
    # --sample-mb is used directly as the per-day S3 byte range, so a large "sample" across a range
    # is a near-full billable pull and must NOT bypass the gate (Codex P2).
    with pytest.raises(SystemExit) as ei:
        gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=CAP + 1, allow_backfill=False)
    assert ei.value.code == 4
    assert "smoke cap" in capsys.readouterr().err.lower()


def test_allow_backfill_overrides():
    gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=0, allow_backfill=True)
    gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=10_000, allow_backfill=True)
