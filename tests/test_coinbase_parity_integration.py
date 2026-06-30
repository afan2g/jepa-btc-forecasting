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
# Optional Lake `book` (snapshot) fixture: if present, the test seeds/reseeds (§5a-Recon).
LAKE_BOOK_FIXTURE = ROOT / "tests" / "fixtures" / f"coinbase_book_{DAY.isoformat()}.parquet"

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

    # If a Lake `book` snapshot fixture is present, exercise the §5a-Recon seed/reseed path; the
    # report's lake_reseed block must show the before/after crossed rate and a valid seed.
    snaps = None
    if LAKE_BOOK_FIXTURE.exists():
        from recon.ingest import shared_engine_time_col
        from recon.reseed import snapshots_from_lake_book_df
        bdf = pd.read_parquet(LAKE_BOOK_FIXTURE)
        snaps = snapshots_from_lake_book_df(
            bdf, engine_time_col=shared_engine_time_col(bdf), max_levels=20,
            stride_ns=1_000_000_000)

    report, lake, capi = rcp.run_parity_core(
        lake_df, chunks, day=DAY, k=10, grid_ms=1000, lake_book_snapshots=snaps)

    lr = report["lake_reseed"]
    if snaps:
        assert lr["applied"] is True and lr["snapshot_candidates"] == len(snaps)
        assert 0.0 <= lr["crossed_rate_after"] <= 1.0 and 0.0 <= lr["crossed_rate_before"] <= 1.0
        # the reseed must not make things worse than cold-start
        assert lr["crossed_rate_after"] <= lr["crossed_rate_before"] + 1e-9
    else:
        assert lr["applied"] is False

    # Full grid spans the day; n_grid_full stays the TRUE grid even when warm-up + residual crossed
    # Lake samples are excluded from the compared subset (they are dropped via since_ts/exclude_ts,
    # not by undercounting the grid). The accounting identity ties the three together.
    assert report["parity"]["n_grid_full"] == 86400
    assert report["meta"]["grid_points"] == 86400
    assert 0 < report["parity"]["n_grid"] <= 86400
    if report["warmup"]["established"]:
        assert report["parity"]["since_ts"] == report["warmup"]["cutoff_ts"]
    assert (report["parity"]["n_grid"] == 86400 - report["warmup"]["excluded_samples"]
            - report["parity"]["n_excluded_crossed"])
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
