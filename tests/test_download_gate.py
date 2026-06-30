"""The CoinAPI downloader's backfill gate (docs/data.md §5a/§8): a multi-day full pull is refused
until the recon-parity + reseed gates pass, so the documented 'do not backfill' decision is enforced
in code, not just prose. Single-day parity pulls and --sample-mb smoke tests stay allowed; an explicit
--allow-backfill overrides. Pure (no vendor I/O) so it runs in CI."""
import datetime as dt
import importlib.util
import pathlib

import pytest

# ingest/ is not a package; the module self-inserts its own dir on sys.path at import time, so the
# `from _common import ...` / `import coinapi_flatfiles` relative imports resolve when loaded by path.
_SPEC = importlib.util.spec_from_file_location(
    "download_coinapi",
    pathlib.Path(__file__).resolve().parents[1] / "ingest" / "download_coinapi.py",
)
dc = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(dc)

D = dt.date


def test_single_day_pull_is_allowed():
    # the parity pilot: one overlap day, no override needed
    dc.check_backfill_gate(D(2025, 6, 1), D(2025, 6, 1), sample_mb=0, allow_backfill=False)


def test_multi_day_full_pull_is_blocked(capsys):
    with pytest.raises(SystemExit) as ei:
        dc.check_backfill_gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=0, allow_backfill=False)
    assert ei.value.code == dc.BACKFILL_GATE_EXIT == 4
    assert "backfill" in capsys.readouterr().err.lower()


def test_multi_day_smoke_sample_is_allowed():
    # --sample-mb is a bounded dev smoke test, not a paid bulk pull
    dc.check_backfill_gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=8, allow_backfill=False)


def test_allow_backfill_overrides():
    dc.check_backfill_gate(D(2025, 1, 1), D(2025, 1, 31), sample_mb=0, allow_backfill=True)
