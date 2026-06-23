"""Skip-guarded REAL-DATA integration test for the Coinbase parity gate.

Runs ONLY when a developer has both local artifacts on disk for the overlap day:
  1. The CoinAPI parquet at data/raw/limitbook_full/exchange=COINBASE/symbol=BTC-USD/dt=<day>/
     (produced by `ingest/download_coinapi.py --start <day> --end <day>`), AND
  2. A Crypto Lake `book_delta_v2` parquet fixture for the day at
     tests/fixtures/coinbase_book_delta_v2_<day>.parquet (a developer drop from a live pull).

It exercises the real parity core (`run_parity_core`) on real local data and asserts the
report is well-formed and internally consistent. It NEVER touches a live vendor API, so it
stays skipped in normal CI (both artifacts absent). Set PARITY_DAY to override the day.
"""
import datetime as dt
import importlib.util
import math
import os
import pathlib

import pandas as pd
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
DAY = dt.date.fromisoformat(os.environ.get("PARITY_DAY", "2025-06-01"))
CAPI_PARQUET = (ROOT / "data" / "raw" / "limitbook_full" / "exchange=COINBASE"
                / "symbol=BTC-USD" / f"dt={DAY.isoformat()}" / "data.parquet")
LAKE_FIXTURE = ROOT / "tests" / "fixtures" / f"coinbase_book_delta_v2_{DAY.isoformat()}.parquet"

pytestmark = pytest.mark.skipif(
    not (CAPI_PARQUET.exists() and LAKE_FIXTURE.exists()),
    reason=("needs local one-day CoinAPI parquet + a Lake book_delta_v2 fixture "
            "(run ingest/download_coinapi.py for the day and drop the Lake fixture); "
            "never runs live vendor in CI"),
)

_SPEC = importlib.util.spec_from_file_location(
    "run_coinbase_parity", ROOT / "scripts" / "run_coinbase_parity.py")
rcp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rcp)


def test_real_one_day_parity_runs_and_is_well_formed(tmp_path):
    lake_df = pd.read_parquet(LAKE_FIXTURE)
    chunks = rcp.iter_coinapi_chunks(str(CAPI_PARQUET), chunk_rows=2_000_000)
    report, lake, capi = rcp.run_parity_core(lake_df, chunks, day=DAY, k=10, grid_ms=1000)

    assert report["parity"]["n_grid"] == 86400
    assert report["meta"]["coinapi_event_rows"] > 0
    for q in (report["lake_quality"], report["coinapi_quality"]):
        assert 0.0 <= q["crossed_rate"] <= 1.0
        assert 0.0 <= q["missing_book_fraction"] <= 1.0
    md = report["parity"]["mid_diff"]
    if md["median"] is not None:
        assert md["median"] >= 0.0 and md["max"] >= md["median"]
    # report must be strict-JSON-writable (jq empty contract)
    paths = rcp.write_report(report, lake, capi, str(tmp_path), DAY, 10, dump_grid=False)
    txt = pathlib.Path(paths["json"]).read_text()
    assert "NaN" not in txt and "Infinity" not in txt
    # spike characterization present (the known rare large-divergence concern)
    assert "spike_counts" in report["parity"] and isinstance(
        report["parity"]["top_spikes"], list)
    assert math.isfinite(report["parity"]["grid_s"])
