"""Offline tests: Stage-1 Binance Lake downloader (plan Task 5).

NO live vendor I/O — every Lake lister/reader/used_data probe is injected as a fake. The downloader
streams injected pyarrow batches (or pandas frames) into an atomic, Hive-partitioned ZSTD Parquet
raw store and appends a resume-ledger manifest record per unit. The `book` SEED_PRODUCT (Requirement
1) is exercised with a 20-level snapshot fixture so a downloader that estimates the seed bytes but
drops the seed product FAILS here (otherwise Stage-2 recon silently cold-starts every day).

These tests all need pyarrow (they write/read parquet), so the module importorskips it below. The
pyarrow-FREE downloader coverage (error taxonomy, exit-code contract, dry-run, import-safety, pure
helpers) lives in test_download_lake_binance_pure.py, which runs in the lightweight default-CI path.
"""
import json
import pathlib
import threading
import time

import pytest

# pyarrow is a downloader-only dep (pyproject `lake`/`baseline` extras), NOT a base dependency — skip
# this whole module (rather than error at collection) when the default suite runs without it. The pure
# helpers in ingest.lake_binance / download_lake_binance need only pandas, imported after this gate.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from ingest import download_lake_binance as dl  # noqa: E402
from ingest import lake_binance as lb  # noqa: E402

PERP = ("BINANCE_FUTURES", "BTC-USDT-PERP")
SPOT = ("BINANCE", "BTC-USDT")


# --------------------------------------------------------------------------- fixtures / fakes
def _batch(n, start=0):
    """A tiny 2-column book_delta_v2-shaped record batch."""
    return pa.record_batch({
        "origin_time": pa.array([start + i for i in range(n)], pa.int64()),
        "price": pa.array([100.0 + i for i in range(n)], pa.float64()),
    })


def _book_snapshot_frame(nlev=20, nrows=3):
    """A pandas 20-level `book` snapshot with the exact columns snapshots_from_lake_book_df reads."""
    import pandas as pd
    data = {"origin_time": [1_000 + i for i in range(nrows)],
            "received_time": [1_001 + i for i in range(nrows)]}
    for i in range(nlev):
        data[f"bid_{i}_price"] = [100.0 - i for _ in range(nrows)]
        data[f"bid_{i}_size"] = [1.0 + i for _ in range(nrows)]
        data[f"ask_{i}_price"] = [101.0 + i for _ in range(nrows)]
        data[f"ask_{i}_size"] = [1.5 + i for _ in range(nrows)]
    return pd.DataFrame(data)


class FakeReader:
    """Injected reader: reader(feed, exchange, symbol, day_iso) -> iterable of batches | None.

    `plan` maps a feed to one of: a list of batches/frames (stream), None (missing file), or an
    Exception instance/class (raised). `raise_n_then` schedules N transient raises before a feed
    finally streams. Records every call so tests can assert attempt counts / idempotency."""

    def __init__(self, plan=None, raise_n_then=None):
        self.plan = plan or {}
        self.raise_n_then = raise_n_then or {}   # feed -> (n_left, exc, then_batches)
        self.calls = []

    def __call__(self, feed, exchange, symbol, day_iso):
        self.calls.append((feed, exchange, symbol, day_iso))
        if feed in self.raise_n_then:
            n_left, exc, then = self.raise_n_then[feed]
            if n_left > 0:
                self.raise_n_then[feed] = (n_left - 1, exc, then)
                raise exc
            return iter(then)
        val = self.plan.get(feed, [_batch(2)])
        if val is None:
            return None
        if isinstance(val, BaseException):
            raise val
        if isinstance(val, type) and issubclass(val, BaseException):
            raise val()
        return iter(val)


def _count_calls(reader, feed):
    return sum(1 for c in reader.calls if c[0] == feed)


# --------------------------------------------------------------------------- process_unit: happy path
def test_process_unit_writes_parquet_and_ok_manifest(tmp_path):
    root = str(tmp_path)
    reader = FakeReader({"book_delta_v2": [_batch(2, 0), _batch(3, 2)]})
    res = dl.process_unit(reader, root, "book_delta_v2", *PERP, "2026-04-01",
                          sleep=lambda *_: None)
    assert res.status == "ok"
    assert res.rows == 5
    final = pathlib.Path(lb.raw_parquet_path(root, "book_delta_v2", *PERP, "2026-04-01"))
    assert final.exists()
    assert not final.with_suffix(".parquet.tmp").exists()   # no leftover temp
    assert pq.read_table(final).num_rows == 5               # streamed both batches losslessly
    # manifest: one ok record carrying rows / sha256 / schema_version / partition keys
    recs = [json.loads(x) for x in (tmp_path / lb.MANIFEST_NAME).read_text().splitlines()]
    assert len(recs) == 1
    r = recs[0]
    assert r["status"] == "ok" and r["rows"] == 5
    assert r["feed"] == "book_delta_v2" and r["exchange"] == PERP[0]
    assert r["symbol"] == PERP[1] and r["dt"] == "2026-04-01"
    assert r["schema_version"] == lb.RAW_SCHEMA_VERSION
    assert len(r["sha256"]) == 64
    assert r["sha256"] == dl._sha256_file(str(final))            # digest is of the PUBLISHED bytes


def test_process_unit_rejects_partition_missing_engine_time(tmp_path):
    # a drifted partition with no engine-time column must fail loud (status error), NOT be copied and
    # stamped ok — so vendor drift surfaces at download time, not later in Stage-2 after quota spent.
    bad = pa.record_batch({"px": pa.array([1.0, 2.0], pa.float64())})   # no origin/received/timestamp
    res = dl.process_unit(FakeReader({"trades": [bad]}), str(tmp_path), "trades", *PERP,
                          "2026-04-01", sleep=lambda *_: None)
    assert res.status == "error" and "error" in res.record
    assert not pathlib.Path(lb.raw_parquet_path(str(tmp_path), "trades", *PERP, "2026-04-01")).exists()


def test_process_unit_ok_records_schema_fingerprint_and_cols(tmp_path):
    res = dl.process_unit(FakeReader(), str(tmp_path), "trades", *PERP, "2026-04-01",
                          sleep=lambda *_: None)
    assert res.status == "ok"
    assert len(res.record["schema_fingerprint"]) == 16          # drift-detectable fingerprint recorded
    assert "origin_time" in res.record["schema_cols"]


def test_schema_fingerprint_changes_on_drift():
    a = pa.schema([("origin_time", pa.int64()), ("price", pa.float64())])
    b = pa.schema([("origin_time", pa.int64()), ("price", pa.int64())])     # dtype drift
    c = pa.schema([("origin_time", pa.int64()), ("px", pa.float64())])      # renamed column
    assert dl.schema_fingerprint(a) == dl.schema_fingerprint(a)
    assert dl.schema_fingerprint(a) != dl.schema_fingerprint(b)
    assert dl.schema_fingerprint(a) != dl.schema_fingerprint(c)


def test_parquet_kv_metadata_carries_schema_version_not_hash(tmp_path):
    # Requirement 6: schema_version lives in the parquet KV metadata; the sha256 does NOT (embedding
    # the hash would change the very bytes being hashed).
    root = str(tmp_path)
    dl.process_unit(FakeReader(), root, "trades", *PERP, "2026-04-01", sleep=lambda *_: None)
    md = pq.read_schema(lb.raw_parquet_path(root, "trades", *PERP, "2026-04-01")).metadata or {}
    assert md.get(b"schema_version") == lb.RAW_SCHEMA_VERSION.encode()
    assert not any(b"sha" in k.lower() for k in md)


# --------------------------------------------------------------------------- resume / idempotency
def test_process_unit_second_call_skips_without_rereading(tmp_path):
    root = str(tmp_path)
    dl.process_unit(FakeReader(), root, "trades", *PERP, "2026-04-01", sleep=lambda *_: None)
    reader2 = FakeReader()
    res = dl.process_unit(reader2, root, "trades", *PERP, "2026-04-01", sleep=lambda *_: None)
    assert res.status == "skip"
    assert reader2.calls == []                              # a done partition never re-reads vendor
    recs = [json.loads(x) for x in (tmp_path / lb.MANIFEST_NAME).read_text().splitlines()]
    assert [r["status"] for r in recs] == ["ok", "skip"]   # skip is recorded (append-only ledger)


def test_process_unit_overwrite_rereads(tmp_path):
    root = str(tmp_path)
    dl.process_unit(FakeReader(), root, "trades", *PERP, "2026-04-01", sleep=lambda *_: None)
    reader2 = FakeReader({"trades": [_batch(7)]})
    res = dl.process_unit(reader2, root, "trades", *PERP, "2026-04-01",
                          overwrite=True, sleep=lambda *_: None)
    assert res.status == "ok" and res.rows == 7
    assert reader2.calls != []


# --------------------------------------------------------------------------- missing / sparse
def test_process_unit_missing_file_records_missing(tmp_path):
    root = str(tmp_path)
    res = dl.process_unit(FakeReader({"liquidations": None}), root, "liquidations", *PERP,
                          "2026-04-01", sleep=lambda *_: None)
    assert res.status == "missing"
    assert not pathlib.Path(lb.raw_parquet_path(root, "liquidations", *PERP, "2026-04-01")).exists()
    r = json.loads((tmp_path / lb.MANIFEST_NAME).read_text().splitlines()[0])
    assert r["status"] == "missing" and r["feed"] == "liquidations"


def test_process_unit_overwrite_miss_removes_stale_final(tmp_path):
    # --overwrite that finds the partition now ABSENT must drop the previously-downloaded parquet, so
    # is_done doesn't keep serving obsolete raw data while the manifest says missing.
    root = str(tmp_path)
    final = pathlib.Path(lb.raw_parquet_path(root, "liquidations", *PERP, "2026-04-01"))
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"OLD DATA")                       # a previously-downloaded partition
    res = dl.process_unit(FakeReader({"liquidations": None}), root, "liquidations", *PERP,
                          "2026-04-01", overwrite=True, sparse_ok=True, sleep=lambda *_: None)
    assert res.status == "missing"
    assert not final.exists() and not lb.is_done(root, "liquidations", *PERP, "2026-04-01")


def test_process_unit_overwrite_empty_removes_stale_final(tmp_path):
    root = str(tmp_path)
    final = pathlib.Path(lb.raw_parquet_path(root, "liquidations", *PERP, "2026-04-01"))
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"OLD DATA")
    empty = pa.record_batch({"origin_time": pa.array([], pa.int64())})   # now empty
    res = dl.process_unit(FakeReader({"liquidations": [empty]}), root, "liquidations", *PERP,
                          "2026-04-01", overwrite=True, sparse_ok=True, sleep=lambda *_: None)
    assert res.status == "missing" and res.record.get("empty") is True
    assert not final.exists()


def test_process_unit_overwrite_error_keeps_stale_final(tmp_path):
    # on an --overwrite whose new read ERRORS (state undetermined), the old file is PRESERVED — a
    # transient failure must not destroy good data.
    root = str(tmp_path)
    final = pathlib.Path(lb.raw_parquet_path(root, "trades", *PERP, "2026-04-01"))
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_bytes(b"OLD DATA")
    reader = FakeReader({"trades": dl.TransientError("RequestTimeout")})
    res = dl.process_unit(reader, root, "trades", *PERP, "2026-04-01", overwrite=True,
                          retries=2, sleep=lambda *_: None)
    assert res.status == "error"
    assert final.read_bytes() == b"OLD DATA"             # preserved on failure


def test_process_unit_missing_stamps_sparse_ok(tmp_path):
    root = str(tmp_path)
    r1 = dl.process_unit(FakeReader({"liquidations": None}), root, "liquidations", *PERP,
                         "2026-04-01", sparse_ok=True, sleep=lambda *_: None)
    assert r1.status == "missing" and r1.record["sparse_ok"] is True     # expected quiet-day gap
    r2 = dl.process_unit(FakeReader({"trades": None}), root, "trades", *PERP, "2026-04-02",
                         sparse_ok=False, sleep=lambda *_: None)
    assert r2.status == "missing" and r2.record["sparse_ok"] is False    # required-feed hole


def test_process_unit_empty_partition_is_nonfatal_missing(tmp_path):
    # a present-but-empty vendor object (reader yields zero batches) is handled like a missing
    # partition under the sparse policy — NOT an error (a quiet-day liquidations file must not fail
    # the batch), and no parquet is published.
    root = str(tmp_path)
    res = dl.process_unit(FakeReader({"liquidations": []}), root, "liquidations", *PERP,
                          "2026-04-01", sparse_ok=True, sleep=lambda *_: None)
    assert res.status == "missing"
    assert res.record["sparse_ok"] is True and res.record.get("empty") is True
    assert not pathlib.Path(lb.raw_parquet_path(root, "liquidations", *PERP, "2026-04-01")).exists()


def test_process_unit_zero_row_batch_is_empty_not_ok(tmp_path):
    # a schema-bearing but ZERO-ROW batch must be treated as empty (missing), NOT published as an
    # `ok` 0-row parquet — else an empty required feed would exit 0 and be considered done.
    root = str(tmp_path)
    empty_batch = pa.record_batch({"origin_time": pa.array([], pa.int64())})
    res = dl.process_unit(FakeReader({"trades": [empty_batch]}), root, "trades", *PERP,
                          "2026-04-01", sparse_ok=False, sleep=lambda *_: None)
    assert res.status == "missing" and res.rows == 0
    assert res.record.get("empty") is True
    assert not pathlib.Path(lb.raw_parquet_path(root, "trades", *PERP, "2026-04-01")).exists()


def test_run_zero_row_required_feed_exits_3(tmp_path):
    # exact Codex scenario: a required feed exposed with a schema but zero rows must NOT exit 0.
    reader = _perp_reader()
    reader.plan["trades"] = [pa.record_batch({"origin_time": pa.array([], pa.int64())})]
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 3


def test_run_empty_liquidations_is_nonfatal(tmp_path):
    reader = _perp_reader()
    reader.plan["liquidations"] = []                      # present but zero-batch (quiet day)
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0                                      # empty sparse feed does not fail the batch


def test_run_empty_required_feed_exits_3(tmp_path):
    reader = _perp_reader()
    reader.plan["trades"] = []                            # a required feed present but empty → gap
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 3


# --------------------------------------------------------------------------- SEED_PRODUCT (book)
def test_process_unit_writes_book_seed_snapshot(tmp_path):
    # The 20-level `book` seed product must be downloaded (it seeds Stage-2 recon), written to its own
    # data/raw/lake/book/... partition, and manifested — mirrors run_coinbase_quality_map LAKE_PRODUCTS.
    root = str(tmp_path)
    snap = _book_snapshot_frame(nlev=20, nrows=3)
    res = dl.process_unit(FakeReader({"book": [snap]}), root, lb.SEED_PRODUCT, *PERP,
                          "2026-04-01", sleep=lambda *_: None)
    assert res.status == "ok" and res.rows == 3
    final = pathlib.Path(lb.raw_parquet_path(root, "book", *PERP, "2026-04-01"))
    assert final.exists()
    assert "raw/lake/book/exchange=BINANCE_FUTURES" in str(final).replace("\\", "/") \
        or final.parts[-5] == "book"
    cols = set(pq.read_schema(final).names)
    for c in ("bid_0_price", "bid_0_size", "ask_0_price", "ask_0_size", "origin_time"):
        assert c in cols                                   # seed columns preserved losslessly
    r = json.loads((tmp_path / lb.MANIFEST_NAME).read_text().splitlines()[0])
    assert r["feed"] == "book"


# --------------------------------------------------------------------------- retry / backoff
def test_process_unit_retries_transient_then_succeeds(tmp_path):
    root = str(tmp_path)
    reader = FakeReader(raise_n_then={"trades": (2, dl.TransientError("SlowDown"), [_batch(4)])})
    res = dl.process_unit(reader, root, "trades", *PERP, "2026-04-01",
                          retries=5, sleep=lambda *_: None)
    assert res.status == "ok" and res.rows == 4
    assert _count_calls(reader, "trades") == 3             # 2 transient failures + 1 success


def test_process_unit_transient_exhausted_becomes_error(tmp_path):
    root = str(tmp_path)
    reader = FakeReader({"trades": dl.TransientError("RequestTimeout")})
    res = dl.process_unit(reader, root, "trades", *PERP, "2026-04-01",
                          retries=3, sleep=lambda *_: None)
    assert res.status == "error"
    assert _count_calls(reader, "trades") == 3             # retried up to the cap, then gave up
    assert not pathlib.Path(lb.raw_parquet_path(root, "trades", *PERP, "2026-04-01")).exists()
    r = json.loads((tmp_path / lb.MANIFEST_NAME).read_text().splitlines()[0])
    assert r["status"] == "error" and "error" in r


def test_process_unit_backoff_sleeps_between_retries(tmp_path):
    # a recording sleep proves backoff actually fires (a no-op sleep would hide a deleted/broken
    # backoff): one sleep per retry, non-decreasing, capped. rng=1.0 removes jitter randomness.
    root = str(tmp_path)
    slept = []
    reader = FakeReader(raise_n_then={"trades": (3, dl.TransientError("SlowDown"), [_batch(1)])})
    res = dl.process_unit(reader, root, "trades", *PERP, "2026-04-01", retries=5,
                          sleep=lambda s: slept.append(s), rng=lambda: 1.0)
    assert res.status == "ok"
    assert len(slept) == 3                                  # attempts-1 sleeps (3 fail, 1 succeed)
    assert slept == [1.0, 2.0, 4.0]                         # capped exponential, no jitter at rng=1.0
    assert all(0 < s <= dl.DEFAULT_BACKOFF_CAP_S for s in slept)


def test_process_unit_quota_error_is_hard_stop(tmp_path):
    root = str(tmp_path)
    reader = FakeReader({"book_delta_v2": dl.QuotaError("download quota exceeded")})
    with pytest.raises(dl.QuotaError):
        dl.process_unit(reader, root, "book_delta_v2", *PERP, "2026-04-01",
                        retries=5, sleep=lambda *_: None)
    assert _count_calls(reader, "book_delta_v2") == 1      # quota is never retried


def test_process_unit_auth_error_is_hard_stop(tmp_path):
    root = str(tmp_path)
    reader = FakeReader({"book_delta_v2": dl.AuthError("InvalidAccessKeyId")})
    with pytest.raises(dl.AuthError):
        dl.process_unit(reader, root, "book_delta_v2", *PERP, "2026-04-01",
                        retries=5, sleep=lambda *_: None)
    assert _count_calls(reader, "book_delta_v2") == 1      # auth is never retried


def test_run_auth_error_exits_2_and_stops(tmp_path):
    # a wrong-keys/wrong-account AccessDenied must abort with setup exit 2, NOT record each partition
    # as an error (exit 3) and NOT keep re-issuing the same unauthorized request per unit.
    reader = _perp_reader()
    reader.plan["book_delta_v2"] = RuntimeError("AccessDenied: not authorized for this bucket")
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 2
    assert _count_calls(reader, "book_delta_v2") == 1      # not retried
    assert "funding" not in {c[0] for c in reader.calls}   # hard stop — later units not attempted


def test_process_unit_no_partial_parquet_on_midstream_failure(tmp_path):
    # a failure AFTER the first batch is written to .tmp must leave NO final parquet and NO temp.
    root = str(tmp_path)

    def _explode():
        yield _batch(2)
        raise dl.TransientError("connection reset")

    reader = FakeReader({"book_delta_v2": None})           # placeholder; override call below
    reader.plan["book_delta_v2"] = _explode()              # a generator that fails mid-stream
    res = dl.process_unit(reader, root, "book_delta_v2", *PERP, "2026-04-01",
                          retries=1, sleep=lambda *_: None)
    assert res.status == "error"
    part = pathlib.Path(lb.raw_partition_dir(root, "book_delta_v2", *PERP, "2026-04-01"))
    assert not (part / "data.parquet").exists()
    assert not (part / "data.parquet.tmp").exists()


def test_sigint_tmp_never_counts_complete_and_resume_restarts_safely(tmp_path):
    raw = tmp_path / "raw"

    def interrupted_reader(feed, exchange, symbol, day_iso):
        def stream():
            yield _batch(2)
            raise KeyboardInterrupt
        return stream()

    with pytest.raises(KeyboardInterrupt):
        dl.process_unit(interrupted_reader, str(raw), "trades", *SPOT, "2026-04-01",
                        sleep=lambda *_: None)

    final = pathlib.Path(lb.raw_parquet_path(str(raw), "trades", *SPOT, "2026-04-01"))
    tmp = pathlib.Path(str(final) + ".tmp")
    assert tmp.exists() and not final.exists()
    assert not lb.is_done(str(raw), "trades", *SPOT, "2026-04-01")
    assert not (raw / lb.MANIFEST_NAME).exists()

    reader = FakeReader({"trades": [_batch(3)]})
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--start", "2026-04-01", "--end", "2026-04-01", "--resume",
                    "--out", str(raw), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0
    assert final.exists() and not tmp.exists()
    assert pq.read_table(final).num_rows == 3
    records = [json.loads(line) for line in (raw / lb.MANIFEST_NAME).read_text().splitlines()]
    assert [record["status"] for record in records] == ["ok"]


# --------------------------------------------------------------------------- run() end to end
def _perp_reader():
    return FakeReader({"book_delta_v2": [_batch(5)], "book": [_book_snapshot_frame(nrows=2)],
                       "trades": [_batch(3)], "funding": [_batch(1)],
                       "open_interest": [_batch(1)], "liquidations": None})


def test_run_end_to_end_writes_all_units_incl_book_seed(tmp_path):
    reader = _perp_reader()
    report_dir = tmp_path / "reports"
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(report_dir)],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0
    # every scoped feed for perp + the book seed got written (liquidations missing is non-fatal)
    written = {c[0] for c in reader.calls}
    assert "book" in written and "book_delta_v2" in written and "funding" in written
    raw = tmp_path / "raw"
    assert pathlib.Path(lb.raw_parquet_path(str(raw), "book", *PERP, "2026-04-01")).exists()
    # a run report is written
    reports = list(report_dir.glob("*.json"))
    assert len(reports) == 1
    rep = json.loads(reports[0].read_text())
    assert rep["counts"]["ok"] >= 5 and rep["counts"]["missing"] == 1
    assert rep["counts"]["missing_required"] == 0        # the lone miss is sparse liquidations
    assert rep["dry_run"] is False


def test_serial_progress_reports_units_throughput_eta_and_terminal_state(tmp_path, capsys):
    report_dir = tmp_path / "rep"
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--start", "2026-04-01", "--end", "2026-04-02", "--jobs", "1",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(report_dir)],
                   reader=FakeReader({"trades": [_batch(2)]}),
                   used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0
    output = capsys.readouterr().out
    assert "[1/2]" in output and "[2/2]" in output
    assert "trades BINANCE BTC-USDT 2026-04-01" in output
    assert "status=ok rows=2" in output
    assert "bytes=" in output and "unit=" in output
    assert "aggregate elapsed=" in output and "rows/s=" in output and "output/s=" in output
    assert "observed-rate projection; not a bound" in output
    assert "does not guarantee 4x live scaling" in output

    report = json.loads(next(report_dir.glob("*.json")).read_text())
    assert report["progress"]["completed"] == report["progress"]["total"] == 2
    assert report["progress"]["terminal_state"] == "complete"
    assert report["progress"]["observed_eta_seconds"] == 0
    assert report["total_out_bytes"] > 0
    assert all(unit["status"] == "ok" and unit["secs"] >= 0
               and unit["out_bytes"] > 0 for unit in report["per_unit"])


def test_run_missing_required_feed_exits_3(tmp_path):
    # a missing REQUIRED feed (trades) is a real data gap → exit 3, never a silent success (exit 0).
    reader = _perp_reader()
    reader.plan["trades"] = None                          # no vendor file for a required feed
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 3
    rep = json.loads(next((tmp_path / "rep").glob("*.json")).read_text())
    assert rep["counts"]["missing_required"] == 1         # trades hole counted; liquidations is not


def test_run_missing_required_book_seed_exits_3(tmp_path):
    # the `book` seed is required (recon can't seed without it) → its miss is fatal too.
    reader = _perp_reader()
    reader.plan["book"] = None
    code = dl.main(["--instrument", "binance-perp", "--feeds", "book_delta_v2",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 3


def test_run_resume_gates_only_pending_units_not_full_batch(tmp_path):
    # A broad batch mostly downloaded: the quota/broad gate must estimate ONLY the not-done units,
    # else a --resume of one leftover partition spuriously exits 4 after the first run used quota.
    raw = tmp_path / "raw"
    days = [f"2026-04-{d:02d}" for d in range(1, 11)]     # 10 days
    for d in days[:-1]:                                   # pre-complete 9/10 → only the last pending
        p = pathlib.Path(lb.raw_parquet_path(str(raw), "trades", *SPOT, d))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"done")
    dfile = tmp_path / "days.txt"
    dfile.write_text("\n".join(days) + "\n")
    reader = FakeReader({"trades": [_batch(2)]})
    # --max-gb sits BETWEEN the full-batch estimate (10 × 0.021 GB) and one pending day (0.021 GB):
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades", "--days-file", str(dfile),
                    "--max-gb", "0.05", "--out", str(raw), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0                                      # gated on 1 pending day, not the full 10
    assert {c[3] for c in reader.calls} == {days[-1]}     # only the leftover day transferred


def test_run_partial_failure_exits_3(tmp_path, capsys):
    reader = _perp_reader()
    reader.plan["trades"] = dl.TransientError("RequestTimeout")     # one feed always fails
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep"),
                    "--retries", "2"],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 3
    output = capsys.readouterr().out
    assert "status=error" in output and "Finished partial." in output
    report = json.loads(next((tmp_path / "rep").glob("*.json")).read_text())
    assert report["counts"]["error"] == 1
    assert report["progress"]["terminal_state"] == "partial"
    assert report["progress"]["completed"] == report["progress"]["total"]
    assert any(unit["status"] == "error" and "RequestTimeout" in unit["error"]
               for unit in report["per_unit"])


def test_run_quota_error_exits_2(tmp_path):
    reader = _perp_reader()
    reader.plan["book_delta_v2"] = dl.QuotaError("Quota exceeded")
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 2


def test_run_resume_skips_already_done_units(tmp_path):
    raw = tmp_path / "raw"
    # pre-create the trades partition so resume skips it and only re-reads the rest
    done = pathlib.Path(lb.raw_parquet_path(str(raw), "trades", *PERP, "2026-04-01"))
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_bytes(b"existing")
    reader = _perp_reader()
    dl.main(["--instrument", "binance-perp", "--feeds", "book_delta_v2,trades",
             "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(raw),
             "--report-dir", str(tmp_path / "rep")],
            reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert "trades" not in {c[0] for c in reader.calls}   # already-done unit not re-read
    assert "book_delta_v2" in {c[0] for c in reader.calls}


def test_run_resume_skips_sparse_accepted_liquidations(tmp_path):
    # a quiet-day liquidations miss (no parquet) is recorded sparse-ok; a later --resume of the same
    # range must NOT re-hit Lake for it (idempotent resume, no gate re-charge).
    raw = tmp_path / "raw"
    argv = ["--instrument", "binance-perp", "--feeds", "liquidations",
            "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(raw),
            "--report-dir", str(tmp_path / "rep")]
    r1 = FakeReader({"liquidations": None})
    assert dl.main(argv, reader=r1, used_data_fn=lambda: 0.0, sleep=lambda *_: None) == 0
    assert _count_calls(r1, "liquidations") == 1                 # first run probed Lake
    r2 = FakeReader({"liquidations": None})
    assert dl.main(argv, reader=r2, used_data_fn=lambda: 0.0, sleep=lambda *_: None) == 0
    assert r2.calls == []                                        # resume does NOT re-probe the accepted day


def test_run_noop_resume_makes_no_vendor_call(tmp_path, monkeypatch):
    # a resume whose range is already complete must short-circuit BEFORE Lake setup: no session, no
    # used_data probe, exit 0 — even with credentials unavailable (idempotent no-op resume).
    raw = tmp_path / "raw"
    argv = ["--instrument", "binance-spot", "--feeds", "trades", "--start", "2026-04-01",
            "--end", "2026-04-01", "--out", str(raw), "--report-dir", str(tmp_path / "rep")]
    assert dl.main(argv, reader=FakeReader({"trades": [_batch(2)]}), used_data_fn=lambda: 0.0,
                   sleep=lambda *_: None) == 0                  # first run completes the range
    requested_tmp = pathlib.Path(
        lb.raw_parquet_path(str(raw), "trades", *SPOT, "2026-04-01") + ".tmp")
    unrelated_tmp = pathlib.Path(
        lb.raw_parquet_path(str(raw), "trades", *SPOT, "2026-03-01") + ".tmp")
    requested_tmp.write_bytes(b"stale requested temp")
    unrelated_tmp.parent.mkdir(parents=True, exist_ok=True)
    unrelated_tmp.write_bytes(b"unrelated temp")
    # resume via the LIVE path (no injected reader/used_data_fn); make setup explode if reached.
    monkeypatch.setattr(dl, "lake_session",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no creds")))
    def _boom_used():
        raise AssertionError("used_data must not be called on a no-op resume")
    assert dl.main(argv, used_data_fn=_boom_used, sleep=lambda *_: None) == 0
    assert not requested_tmp.exists()
    assert unrelated_tmp.exists()
    rep = json.loads(sorted((tmp_path / "rep").glob("*.json"))[-1].read_text())
    assert rep["n_pending"] == 0 and rep["used_data_before"] is None


def test_run_resume_retries_required_missing(tmp_path):
    # a REQUIRED feed that was missing (exit 3) must be RE-ATTEMPTED on resume (the gap may fill) —
    # unlike an accepted sparse miss, it is never treated as done.
    raw = tmp_path / "raw"
    argv = ["--instrument", "binance-spot", "--feeds", "trades",
            "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(raw),
            "--report-dir", str(tmp_path / "rep")]
    r1 = FakeReader({"trades": None})
    assert dl.main(argv, reader=r1, used_data_fn=lambda: 0.0, sleep=lambda *_: None) == 3
    r2 = FakeReader({"trades": [_batch(2)]})                     # data has since landed
    assert dl.main(argv, reader=r2, used_data_fn=lambda: 0.0, sleep=lambda *_: None) == 0
    assert _count_calls(r2, "trades") == 1                       # required miss retried, not skipped


def test_run_days_file_source(tmp_path):
    days_file = tmp_path / "days.txt"
    days_file.write_text("2026-04-01\n2026-04-02\n")
    reader = FakeReader({"trades": [_batch(2)]})
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--days-file", str(days_file), "--out", str(tmp_path / "raw"),
                    "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0
    days = {c[3] for c in reader.calls}
    assert days == {"2026-04-01", "2026-04-02"}
    assert "book" not in {c[0] for c in reader.calls}     # no `book` seed when book_delta_v2 unselected


def test_run_overwrite_flag_rereads_existing_partition(tmp_path):
    raw = tmp_path / "raw"
    done = pathlib.Path(lb.raw_parquet_path(str(raw), "trades", *SPOT, "2026-04-01"))
    done.parent.mkdir(parents=True, exist_ok=True)
    done.write_bytes(b"stale")
    reader = FakeReader({"trades": [_batch(2)]})
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades", "--overwrite",
                    "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(raw),
                    "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0
    assert "trades" in {c[0] for c in reader.calls}       # --overwrite re-reads the done partition


# --------------------------------------------------------------------------- parallel (--jobs) path
def test_run_jobs_parallel_writes_all_units_with_valid_manifest(tmp_path, capsys):
    raw = tmp_path / "raw"
    reader = _perp_reader()
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-03",
                    "--jobs", "4", "--out", str(raw), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0
    # every manifest line is intact JSON with the required keys (no interleaved/corrupt concurrent
    # appends) — the per-unit lock around manifest_append holds under real threads.
    lines = (raw / lb.MANIFEST_NAME).read_text().splitlines()
    recs = [json.loads(x) for x in lines]
    for r in recs:
        assert {"feed", "exchange", "symbol", "dt", "status"} <= set(r)
    ok = [r for r in recs if r["status"] == "ok"]
    assert any(r["feed"] == "book" for r in ok)           # book seed written under concurrency
    assert len({(r["feed"], r["dt"]) for r in ok}) == len(ok)   # no duplicate unit records
    output = capsys.readouterr().out
    assert "[18/18]" in output
    assert "aggregate elapsed=" in output and "observed-rate projection; not a bound" in output
    report = json.loads(next((tmp_path / "rep").glob("*.json")).read_text())
    assert report["progress"]["completed"] == report["progress"]["total"] == 18
    assert report["progress"]["terminal_state"] == "complete"
    assert report["counts"]["cancelled"] == report["counts"]["hard_stop"] == 0


def test_run_dedups_repeated_instrument_under_jobs(tmp_path):
    # a repeated instrument under --jobs>1 must process the unit ONCE (deduped) — never race two
    # writers on the same partition. Exactly one read, one ok record, one parquet.
    raw = tmp_path / "raw"
    reader = FakeReader({"trades": [_batch(2)]})
    code = dl.main(["--instrument", "binance-spot,binance-spot", "--feeds", "trades", "--jobs", "2",
                    "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(raw),
                    "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 0
    assert _count_calls(reader, "trades") == 1
    recs = [json.loads(x) for x in (raw / lb.MANIFEST_NAME).read_text().splitlines()]
    assert len([r for r in recs if r["status"] == "ok" and r["feed"] == "trades"]) == 1


def test_run_jobs_quota_hard_stop_cancels_pending_units(tmp_path, monkeypatch, capsys):
    # a quota/credit wall under --jobs must exit 2 AND cancel unstarted/in-flight units without
    # draining the 60-unit plan or writing synthetic cancellation records into the resume manifest.
    raw = tmp_path / "raw"
    cancel_event = threading.Event()
    partial_tmp_seen = threading.Event()
    monkeypatch.setattr(dl, "Event", lambda: cancel_event)

    class RaceReader:
        def __init__(self):
            self.calls = []

        def __call__(self, feed, exchange, symbol, day_iso):
            self.calls.append((feed, exchange, symbol, day_iso))
            if day_iso == "2026-04-01":
                def partial_stream():
                    yield _batch(2)
                    tmp = pathlib.Path(lb.raw_parquet_path(
                        str(raw), feed, exchange, symbol, day_iso) + ".tmp")
                    assert tmp.exists()
                    partial_tmp_seen.set()
                    assert cancel_event.wait(2.0)
                    yield _batch(2)
                return partial_stream()
            if day_iso == "2026-04-02":
                assert partial_tmp_seen.wait(2.0)
                raise dl.QuotaError("Quota exceeded")

            def wait_for_stop():
                assert cancel_event.wait(2.0)
                yield _batch(1)
            return wait_for_stop()

    reader = RaceReader()
    used_calls = []

    def used_data():
        used_calls.append(True)
        return 0.0

    code = dl.main(["--instrument", "binance-perp", "--feeds", "book_delta_v2",
                    "--start", "2026-04-01", "--end", "2026-04-30", "--jobs", "4",
                    "--allow-broad",
                    "--out", str(raw), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=used_data, sleep=lambda *_: None)
    assert code == 2
    assert partial_tmp_seen.is_set()                      # cancellation raced a real partial write
    assert len(reader.calls) >= 2
    assert len(reader.calls) <= 4                         # only the bounded initial wave may start
    assert len(used_calls) == 1                           # no post-hard-stop vendor telemetry call
    assert not list(raw.rglob("*.tmp"))

    manifest = raw / lb.MANIFEST_NAME
    records = ([json.loads(line) for line in manifest.read_text().splitlines()]
               if manifest.exists() else [])
    assert all(record["status"] in {"ok", "skip", "missing", "error"} for record in records)
    assert not any(record["status"] in {"hard_stop", "cancelled"} for record in records)

    report = json.loads(next((tmp_path / "rep").glob("*.json")).read_text())
    counts = report["counts"]
    assert counts["hard_stop"] >= 1 and counts["cancelled"] >= 56
    assert sum(counts[key] for key in ("ok", "skip", "missing", "error",
                                       "hard_stop", "cancelled")) == 60
    assert len(report["per_unit"]) == 60
    progress = report["progress"]
    assert progress["completed"] == progress["total"] == 60
    assert progress["observed_eta_seconds"] is None
    assert progress["terminal_state"] == "hard_stop"
    captured = capsys.readouterr()
    assert "status=hard_stop" in captured.out
    assert "status=cancelled" in captured.out
    assert "ETA unavailable after hard stop" in captured.out
    assert "post-stop quota telemetry was not called" in captured.err


# --------------------------------------------------------------------------- live-path helpers (offline)


def test_stream_parquet_batches_streams_row_groups(tmp_path):
    # the live reader must yield row-group batches, NEVER a whole (109 M-row) day as one object.
    import pyarrow as pa
    import pyarrow.fs as pafs
    p = tmp_path / "part.parquet"
    pq.write_table(pa.table({"origin_time": pa.array(range(2500), pa.int64()),
                             "price": pa.array([float(i) for i in range(2500)], pa.float64())}),
                   p, row_group_size=1000)
    fs = pafs.LocalFileSystem()
    batches = list(dl.stream_parquet_batches(fs, [str(p)], batch_size=1000))
    assert len(batches) >= 3                                  # streamed in batches, not one blob
    assert all(isinstance(b, pa.RecordBatch) for b in batches)
    assert sum(b.num_rows for b in batches) == 2500          # lossless
    # multiple objects in a partition stream concatenated, in sorted path order
    p2 = tmp_path / "part2.parquet"
    pq.write_table(pa.table({"origin_time": pa.array([9000, 9001], pa.int64()),
                             "price": pa.array([1.0, 2.0], pa.float64())}), p2)
    two = list(dl.stream_parquet_batches(fs, [str(p2), str(p)], batch_size=1000))
    assert sum(b.num_rows for b in two) == 2502


def test_process_unit_consumes_streaming_reader(tmp_path):
    # end-to-end proof the live-style streaming reader (row-group batches) flows through process_unit
    # unchanged and writes a lossless parquet — the exact shape _live_reader produces, offline.
    import pyarrow as pa
    import pyarrow.fs as pafs
    src = tmp_path / "src.parquet"

    pq.write_table(pa.table({"origin_time": pa.array(range(2500), pa.int64())}),
                   src, row_group_size=1000)
    fs = pafs.LocalFileSystem()

    def reader(feed, exchange, symbol, day_iso):
        return dl.stream_parquet_batches(fs, [str(src)], batch_size=1000)

    res = dl.process_unit(reader, str(tmp_path / "raw"), "book_delta_v2", *PERP, "2026-04-01",
                          sleep=lambda *_: None)
    assert res.status == "ok" and res.rows == 2500
    assert pq.read_table(lb.raw_parquet_path(str(tmp_path / "raw"), "book_delta_v2", *PERP,
                                             "2026-04-01")).num_rows == 2500


def test_row_group_prebuffer_coalesces_high_latency_reads_without_materializing_day(tmp_path):
    class CountingRaw:
        def __init__(self, path, delay_s):
            self._file = open(path, "rb")
            self.mode = "rb"
            self.delay_s = delay_s
            self.read_calls = 0
            self.bytes_read = 0

        @property
        def closed(self):
            return self._file.closed

        def readable(self):
            return True

        def seekable(self):
            return True

        def read(self, size=-1):
            self.read_calls += 1
            time.sleep(self.delay_s)
            data = self._file.read(size)
            self.bytes_read += len(data)
            return data

        def seek(self, offset, whence=0):
            return self._file.seek(offset, whence)

        def tell(self):
            return self._file.tell()

        def close(self):
            self._file.close()

    class CountingFilesystem:
        def __init__(self, delay_s):
            self.delay_s = delay_s
            self.sources = []

        def open_input_file(self, path):
            source = CountingRaw(path, self.delay_s)
            self.sources.append(source)
            return pa.PythonFile(source, mode="r")

    path = tmp_path / "latency.parquet"
    n_rows = 120_000
    pq.write_table(
        pa.table({f"c{i}": pa.array(range(n_rows), pa.int64()) for i in range(8)}),
        path, row_group_size=15_000, compression="snappy",
    )

    delay_s = 0.003
    baseline_fs = CountingFilesystem(delay_s=delay_s)
    baseline_started = time.monotonic()
    with baseline_fs.open_input_file(str(path)) as handle:
        parquet = pq.ParquetFile(handle, pre_buffer=False)
        baseline_rows = sum(batch.num_rows for batch in
                            parquet.iter_batches(batch_size=15_000))
    baseline_elapsed = time.monotonic() - baseline_started

    optimized_fs = CountingFilesystem(delay_s=delay_s)
    optimized_started = time.monotonic()
    optimized = list(dl.stream_parquet_batches(optimized_fs, [str(path)],
                                                batch_size=15_000))
    optimized_elapsed = time.monotonic() - optimized_started
    optimized_rows = sum(batch.num_rows for batch in optimized)
    baseline_calls = sum(source.read_calls for source in baseline_fs.sources)
    optimized_calls = sum(source.read_calls for source in optimized_fs.sources)
    baseline_bytes = sum(source.bytes_read for source in baseline_fs.sources)
    optimized_bytes = sum(source.bytes_read for source in optimized_fs.sources)
    print(f"synthetic 3ms-read benchmark: baseline={baseline_calls} reads/"
          f"{baseline_bytes} bytes/{baseline_elapsed:.3f}s; bounded-prebuffer="
          f"{optimized_calls} reads/{optimized_bytes} bytes/{optimized_elapsed:.3f}s")

    assert baseline_rows == optimized_rows == n_rows
    assert len(optimized) == 8                         # one bounded batch per row group
    assert baseline_calls >= optimized_calls * 4       # coalesced range reads, deterministic signal
    assert optimized_calls <= 10
    assert optimized_bytes == baseline_bytes           # no extra payload bytes
    assert (baseline_calls - optimized_calls) * 0.003 >= 0.1  # >100 ms at modeled 3 ms RTT
