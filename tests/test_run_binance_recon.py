"""Offline e2e tests: the Stage-2 recon runner CLI (`scripts/run_binance_recon.py`, plan Task 7).

NO vendor I/O — a tiny synthetic Stage-1 raw store is written into tmp_path with pyarrow and the
CLI runs end-to-end against it: certified top-K publishing (exact column order, atomic write,
parquet KV schema_version/rows), fail-loud passthrough normalization, the resumable processed
manifest (status / rows / sha256 / schema version / engine-time choice / dropped rows /
classification / recon meta), resume + --overwrite semantics, the sparse-vs-required missing
policy, schema-drift -> error -> exit 3, and the explicit `--engine native` abort (exit 2, no
silent fallback — Binance has no verified tick scale yet, plan Risk Q1).

These tests need pyarrow (they build the raw store), so the module importorskips it; the
pyarrow-free core coverage lives in tests/test_lake_binance_seed_source.py.
"""
import datetime as dt
import importlib.util
import json
import os
import pathlib
import sys

import pandas as pd
import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from ingest import lake_binance as lb  # noqa: E402
from ingest.download_lake_binance import _sha256_file  # noqa: E402

# scripts/ is not a package — load the runner by path (same pattern as test_quality_map).
_SPEC = importlib.util.spec_from_file_location(
    "run_binance_recon_cli",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_binance_recon.py")
rbr = importlib.util.module_from_spec(_SPEC)
sys.modules["run_binance_recon_cli"] = rbr
_SPEC.loader.exec_module(rbr)

NS = 1_000_000_000
DAY = "2026-04-01"
DAY_OPEN = int(pd.Timestamp(DAY).value)
PERP = ("BINANCE_FUTURES", "BTC-USDT-PERP")
NLEV = 5   # book seed depth in fixtures (>= the CLI --seed-min-levels we pass)

TOPK_COLS_K2 = ["mid", "microprice",
                "bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size",
                "bid_1_price", "bid_1_size", "ask_1_price", "ask_1_size",
                "sample_ts"]


def _ts(*secs):
    return pd.to_datetime([DAY_OPEN + int(s * NS) for s in secs])


def _deltas_df():
    """Deltas that keep the seeded book two-sided/uncrossed all day ($0.10 perp price grid)."""
    rows = [(1, 1, True, 100.0, 2.0), (2, 2, False, 100.1, 2.0),
            (3, 3, True, 99.9, 1.0), (4, 4, False, 100.2, 1.0)]
    return pd.DataFrame({
        "origin_time": _ts(*[r[0] for r in rows]),
        "received_time": _ts(*[r[0] for r in rows]),
        "sequence_number": [r[1] for r in rows],
        "side_is_bid": [r[2] for r in rows],
        "price": [r[3] for r in rows],
        "size": [r[4] for r in rows]})


def _book_df(n=10, crossed=0):
    """`n` five-level `book` seed candidates, one per second from the day open; the first is
    always valid, the next `crossed` are crossed at the touch (bid 100.2 > ask 100.1)."""
    rows = []
    for i in range(n):
        bid0 = 100.2 if 1 <= i <= crossed else 100.0
        row = {"origin_time": DAY_OPEN + i * NS, "received_time": DAY_OPEN + i * NS}
        for lv in range(NLEV):
            row[f"bid_{lv}_price"] = bid0 - 0.1 * lv
            row[f"bid_{lv}_size"] = 1.0 + lv
            row[f"ask_{lv}_price"] = 100.1 + 0.1 * lv
            row[f"ask_{lv}_size"] = 1.0 + lv
        rows.append(row)
    df = pd.DataFrame(rows)
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    df["received_time"] = pd.to_datetime(df["received_time"])
    return df


def _trades_df():
    return pd.DataFrame({"origin_time": _ts(10, 20), "received_time": _ts(10, 20),
                         "price": [100.05, 100.15], "quantity": [0.5, 0.25],
                         "side": ["buy", "sell"], "trade_id": [1, 2]})


def _funding_df():
    return pd.DataFrame({"origin_time": _ts(0), "received_time": _ts(0),
                         "funding_rate": [0.0001]})


def _oi_df():
    return pd.DataFrame({"origin_time": _ts(0, 60), "received_time": _ts(0, 60),
                         "open_interest": [1000.0, 1001.5]})


def _liq_df():
    return pd.DataFrame({"origin_time": _ts(30), "received_time": _ts(30),
                         "price": [100.0], "quantity": [0.1], "side": ["sell"]})


ALL_RAW = {"book_delta_v2": _deltas_df, "book": _book_df, "trades": _trades_df,
           "funding": _funding_df, "open_interest": _oi_df, "liquidations": _liq_df}


def write_store(tmp_path, overrides=None):
    """Write the perp raw store for DAY. `overrides[feed]` = DataFrame replaces the default;
    None omits the partition entirely. Returns the raw root."""
    raw = str(tmp_path / "raw")
    overrides = overrides or {}
    for feed, default in ALL_RAW.items():
        df = overrides.get(feed, "default")
        df = default() if isinstance(df, str) else df
        if df is None:
            continue
        path = lb.raw_parquet_path(raw, feed, *PERP, DAY)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        pq.write_table(pa.Table.from_pandas(df, preserve_index=False), path)
    return raw


def run_cli(tmp_path, raw, *extra):
    out = str(tmp_path / "out")
    rep = str(tmp_path / "reports")
    rc = rbr.main(["--instrument", "binance-perp", "--start", DAY, "--end", DAY,
                   "--raw", raw, "--out", out, "--report-dir", rep,
                   "--grid-s", "3600", "--k", "2", "--seed-min-levels", "2", *extra])
    return rc, out, rep


def read_parquet(path):
    # ParquetFile, not read_table: the dataset reader would append hive partition columns
    # (exchange/symbol/dt) inferred from the path (pyarrow >= 24).
    with pq.ParquetFile(path) as pf:
        return pf.read().to_pandas()


def read_manifest(out):
    path = pathlib.Path(out) / lb.MANIFEST_NAME
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def last_by_output(out):
    return {r["output"]: r for r in read_manifest(out)}


def read_report(rep):
    files = sorted(pathlib.Path(rep).glob("*.json"))
    return json.loads(files[-1].read_text())


def outpath(out, output):
    return lb.processed_parquet_path(out, output, *PERP, DAY)


# --------------------------------------------------------------------------- happy path
def test_e2e_certifies_topk_and_normalizes_all_tables(tmp_path):
    rc, out, rep = run_cli(tmp_path, write_store(tmp_path))
    assert rc == 0
    # top-K: exact column order/dtypes, hourly grid, KV metadata carries schema_version + rows
    topk = outpath(out, "topk_l2")
    frame = read_parquet(topk)
    assert list(frame.columns) == TOPK_COLS_K2
    assert len(frame) == 24
    assert all(frame[c].dtype == "float64" for c in TOPK_COLS_K2[:-1])
    assert frame["sample_ts"].dtype == "int64"
    kv = pq.read_schema(topk).metadata
    assert kv[b"schema_version"] == b"topk_l2/1" and kv[b"rows"] == b"24"
    assert not os.path.exists(topk + ".tmp")
    # passthrough tables: canonical schemas, engine-time sorted
    trades = read_parquet(outpath(out, "trades"))
    assert list(trades.columns) == ["origin_time", "received_time", "price", "quantity",
                                    "side", "trade_id"]
    assert trades["origin_time"].is_monotonic_increasing
    for output in ("funding", "open_interest", "liquidations"):
        assert os.path.exists(outpath(out, output))
    # manifest: one record per output with the full audit trail
    recs = last_by_output(out)
    assert set(recs) == {"topk_l2", "trades", "funding", "open_interest", "liquidations"}
    t = recs["topk_l2"]
    assert t["status"] == "ok" and t["classification"] == rbr.CERTIFIED
    assert t["rows"] == 24 and t["schema_version"] == "topk_l2/1"
    assert t["sha256"] == _sha256_file(topk)              # digest of the published bytes
    assert t["engine"] == "python" and t["engine_time_col"] == "origin_time"
    assert t["engine_time_fallback"] is False
    assert t["dropped_rows"] == {"book_delta_v2": 0, "book": 0}
    assert t["seed_source_crossed_frac"] == 0.0
    assert t["seed"]["seed_accepted"] is True and t["seed"]["reseed_count"] == 0
    assert t["quality"]["crossed_rate"] == 0.0
    tr = recs["trades"]
    assert tr["status"] == "ok" and tr["rows"] == 2
    assert tr["schema_version"] == "trades/1" and tr["engine_time_col"] == "origin_time"
    assert tr["dropped_rows"] == {"trades": 0} and tr["resorted"] is False
    assert len(tr["sha256"]) == 64
    # report: auto engine resolved to python (no verified Binance tick scale — Risk Q1)
    report = read_report(rep)
    assert report["counts"]["ok"] == 5 and report["counts"]["error"] == 0
    assert report["engine_by_instrument"]["binance-perp"]["engine"] == "python"


def test_jobs_parallel_path_produces_same_outputs(tmp_path):
    rc, out, _ = run_cli(tmp_path, write_store(tmp_path), "--jobs", "3")
    assert rc == 0
    assert {r["status"] for r in last_by_output(out).values()} == {"ok"}
    assert os.path.exists(outpath(out, "topk_l2"))


# --------------------------------------------------------------------------- resume / overwrite
def test_second_run_skips_everything_without_rewriting(tmp_path):
    raw = write_store(tmp_path)
    rc1, out, rep = run_cli(tmp_path, raw)
    assert rc1 == 0
    before = {o: os.stat(outpath(out, o)).st_mtime_ns
              for o in ("topk_l2", "trades", "funding", "open_interest", "liquidations")}
    n_recs = len(read_manifest(out))
    rc2, _, _ = run_cli(tmp_path, raw)
    assert rc2 == 0
    report = read_report(rep)
    assert report["counts"] == {**report["counts"], "ok": 0, "skip": 5}
    after = {o: os.stat(outpath(out, o)).st_mtime_ns for o in before}
    assert after == before                                   # nothing rewritten
    assert len(read_manifest(out)) == n_recs                 # skips append no records


def test_overwrite_reruns_all_units(tmp_path):
    raw = write_store(tmp_path)
    rc1, out, rep = run_cli(tmp_path, raw)
    n_recs = len(read_manifest(out))
    rc2, _, _ = run_cli(tmp_path, raw, "--overwrite")
    assert rc1 == rc2 == 0
    assert read_report(rep)["counts"]["ok"] == 5
    assert len(read_manifest(out)) == 2 * n_recs


# --------------------------------------------------------------------------- seed-source gate e2e
def test_crossed_seed_source_publishes_no_topk_and_resumes_as_done(tmp_path):
    raw = write_store(tmp_path, {"book": _book_df(n=10, crossed=3)})     # 30% crossed
    rc, out, rep = run_cli(tmp_path, raw)
    assert rc == 0                                           # a verdict, not an error
    assert not os.path.exists(outpath(out, "topk_l2"))       # fail closed: nothing published
    t = last_by_output(out)["topk_l2"]
    assert t["status"] == rbr.INCONCLUSIVE
    assert t["classification"] == rbr.INCONCLUSIVE
    assert rbr.SEED_SOURCE_UNRELIABLE in t["reasons"]
    assert t["seed_source_crossed_frac"] == pytest.approx(0.3)
    assert t["gated_before_replay"] is True and t["rows"] == 0
    assert t["seed"]["seed_accepted"] is True                # an accepted seed does not rescue it
    assert read_report(rep)["counts"]["inconclusive"] == 1
    # deterministic verdict => resume treats it as done
    rc2, _, _ = run_cli(tmp_path, raw)
    assert rc2 == 0
    assert read_report(rep)["counts"]["skip"] == 5


def test_missing_seed_product_inconclusive_then_reruns_when_seed_appears(tmp_path):
    raw = write_store(tmp_path, {"book": None})
    rc, out, rep = run_cli(tmp_path, raw)
    assert rc == 0
    t = last_by_output(out)["topk_l2"]
    assert t["status"] == rbr.INCONCLUSIVE and "no_seed_snapshots" in t["reasons"]
    assert t["book_present"] is False
    assert not os.path.exists(outpath(out, "topk_l2"))
    # Stage-1 later pulls the seed product -> the unit re-runs and certifies
    path = lb.raw_parquet_path(raw, "book", *PERP, DAY)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pq.write_table(pa.Table.from_pandas(_book_df(), preserve_index=False), path)
    rc2, _, _ = run_cli(tmp_path, raw)
    assert rc2 == 0
    assert last_by_output(out)["topk_l2"]["status"] == "ok"
    assert os.path.exists(outpath(out, "topk_l2"))


# --------------------------------------------------------------------------- missing raw inputs
def test_missing_required_feed_exits_partial(tmp_path):
    rc, out, rep = run_cli(tmp_path, write_store(tmp_path, {"trades": None}))
    assert rc == rbr.PARTIAL_EXIT
    t = last_by_output(out)["trades"]
    assert t["status"] == "missing" and t["sparse_ok"] is False
    assert read_report(rep)["counts"]["missing_required"] == 1
    assert not os.path.exists(outpath(out, "trades"))


def test_missing_liquidations_is_sparse_ok(tmp_path):
    rc, out, rep = run_cli(tmp_path, write_store(tmp_path, {"liquidations": None}))
    assert rc == 0
    t = last_by_output(out)["liquidations"]
    assert t["status"] == "missing" and t["sparse_ok"] is True
    assert read_report(rep)["counts"]["missing_required"] == 0
    # still-absent raw partition => resume treats the miss as done
    rc2, _, _ = run_cli(tmp_path, write_store(tmp_path, {"liquidations": None}))
    assert read_report(rep)["counts"]["skip"] == 5


def test_empty_liquidations_partition_treated_as_sparse_missing(tmp_path):
    rc, out, _ = run_cli(tmp_path, write_store(tmp_path, {"liquidations": _liq_df().iloc[0:0]}))
    assert rc == 0
    t = last_by_output(out)["liquidations"]
    assert t["status"] == "missing" and t["sparse_ok"] is True and t["empty"] is True
    assert not os.path.exists(outpath(out, "liquidations"))


def test_missing_delta_partition_is_required_missing(tmp_path):
    rc, out, rep = run_cli(tmp_path, write_store(tmp_path, {"book_delta_v2": None}))
    assert rc == rbr.PARTIAL_EXIT
    t = last_by_output(out)["topk_l2"]
    assert t["status"] == "missing" and t["sparse_ok"] is False
    assert not os.path.exists(outpath(out, "topk_l2"))


# --------------------------------------------------------------------------- schema drift
def test_schema_drift_is_an_identifiable_error_not_output(tmp_path):
    bad = _trades_df().drop(columns=["price"])
    rc, out, rep = run_cli(tmp_path, write_store(tmp_path, {"trades": bad}))
    assert rc == rbr.PARTIAL_EXIT
    t = last_by_output(out)["trades"]
    assert t["status"] == "error" and "price" in t["error"]
    assert not os.path.exists(outpath(out, "trades"))        # no masquerading output
    assert read_report(rep)["counts"]["error"] == 1
    # errored units re-run on resume: fix the partition, rerun, unit succeeds
    path = lb.raw_parquet_path(pathlib.Path(out).parent / "raw", "trades", *PERP, DAY)
    pq.write_table(pa.Table.from_pandas(_trades_df(), preserve_index=False), str(path))
    rc2, _, _ = run_cli(tmp_path, str(pathlib.Path(out).parent / "raw"))
    assert rc2 == 0
    assert last_by_output(out)["trades"]["status"] == "ok"


def test_unknown_trade_side_value_fails_loud(tmp_path):
    rc, out, _ = run_cli(tmp_path, write_store(tmp_path,
                                               {"trades": _trades_df().assign(side=["buy",
                                                                                    "short"])}))
    assert rc == rbr.PARTIAL_EXIT
    t = last_by_output(out)["trades"]
    assert t["status"] == "error" and "short" in t["error"]


# --------------------------------------------------------------------------- engine selection
def test_explicit_native_engine_aborts_before_any_processing(tmp_path):
    # No Binance tick scale is registered (plan Risk Q1: unverified) => explicit native must fail
    # clearly BEFORE any unit runs — never a silent Python fallback (plan Review Checklist).
    raw = write_store(tmp_path)
    rc, out, rep = run_cli(tmp_path, raw, "--engine", "native")
    assert rc == rbr.SETUP_ERROR_EXIT
    assert read_manifest(out) == []                          # nothing processed
    assert not os.path.exists(outpath(out, "topk_l2"))
    assert not pathlib.Path(rep).exists() or not list(pathlib.Path(rep).glob("*.json"))


def test_engine_python_explicitly_selected(tmp_path):
    rc, out, rep = run_cli(tmp_path, write_store(tmp_path), "--engine", "python")
    assert rc == 0
    assert read_report(rep)["engine_by_instrument"]["binance-perp"]["engine"] == "python"


# --------------------------------------------------------------------------- raw vendor names
def test_raw_vendor_time_names_are_canonicalized(tmp_path):
    def raw_names(df):
        return df.rename(columns={"origin_time": "timestamp",
                                  "received_time": "receipt_timestamp"})
    raw = write_store(tmp_path, {feed: raw_names(fn()) for feed, fn in ALL_RAW.items()})
    rc, out, _ = run_cli(tmp_path, raw)
    assert rc == 0
    trades = read_parquet(outpath(out, "trades"))
    assert "origin_time" in trades.columns and "timestamp" not in trades.columns
    recs = last_by_output(out)
    assert recs["topk_l2"]["status"] == "ok"
    assert recs["topk_l2"]["engine_time_col"] == "origin_time"


# --------------------------------------------------------------------------- setup errors
def test_bad_args_exit_2(tmp_path):
    raw = write_store(tmp_path)
    out = str(tmp_path / "out")
    assert rbr.main(["--instrument", "nope", "--start", DAY, "--end", DAY,
                     "--raw", raw, "--out", out]) == rbr.SETUP_ERROR_EXIT
    assert rbr.main(["--instrument", "binance-perp", "--raw", raw, "--out", out]) \
        == rbr.SETUP_ERROR_EXIT                              # no day source
    assert rbr.main(["--instrument", "binance-perp", "--start", DAY, "--end", DAY,
                     "--raw", raw, "--out", out, "--grid-s", "7"]) \
        == rbr.SETUP_ERROR_EXIT                              # 7 s does not divide the day
    assert rbr.main(["--instrument", "binance-perp", "--start", DAY, "--end", DAY,
                     "--raw", raw, "--out", out, "--feeds", "funding",
                     "--instrument", "binance-spot"]) == rbr.SETUP_ERROR_EXIT  # invalid pair
