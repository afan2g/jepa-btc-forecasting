"""Offline tests: Stage-1 Binance Lake downloader (plan Task 5).

NO live vendor I/O — every Lake lister/reader/used_data probe is injected as a fake. The downloader
streams injected pyarrow batches (or pandas frames) into an atomic, Hive-partitioned ZSTD Parquet
raw store and appends a resume-ledger manifest record per unit. The `book` SEED_PRODUCT (Requirement
1) is exercised with a 20-level snapshot fixture so a downloader that estimates the seed bytes but
drops the seed product FAILS here (otherwise Stage-2 recon silently cold-starts every day).

pyarrow is a real dependency here (it writes the parquet), but it must be imported LAZILY by the
downloader — `test_import_is_vendor_safe` asserts importing the module pulls in no boto3/lakeapi.
"""
import json
import pathlib
import subprocess
import sys

import pytest

# pyarrow is a downloader-only dep (pyproject `lake`/`baseline` extras), NOT a base dependency — skip
# this whole module (rather than error at collection) when the default suite runs without it. The pure
# helpers in ingest.lake_binance / download_lake_binance need only pandas, imported after this gate.
pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")

from ingest import download_lake_binance as dl  # noqa: E402
from ingest import lake_binance as lb  # noqa: E402

_ROOT = pathlib.Path(__file__).resolve().parents[1]
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


def test_feed_miss_is_fatal_policy():
    # only sparse/event feeds (liquidations) may go missing without failing the run; the `book` seed
    # and every other feed are required.
    assert dl.feed_miss_is_fatal("book_delta_v2") is True
    assert dl.feed_miss_is_fatal("trades") is True
    assert dl.feed_miss_is_fatal("funding") is True
    assert dl.feed_miss_is_fatal("open_interest") is True
    assert dl.feed_miss_is_fatal(lb.SEED_PRODUCT) is True                # `book` seed is required
    assert dl.feed_miss_is_fatal("liquidations") is False               # sparse, Risk Q2


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


def test_backoff_seconds_grows_and_is_capped():
    grow = [dl._backoff_seconds(a, 1.0, 60.0, lambda: 1.0) for a in range(1, 12)]
    assert grow[:3] == [1.0, 2.0, 4.0]                      # exponential
    assert all(s <= 60.0 for s in grow)                     # capped at 60 s
    assert dl._backoff_seconds(1, 1.0, 60.0, lambda: 0.0) == 0.5   # jitter halves at rng=0


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


# --------------------------------------------------------------------------- error classification
def test_classify_error():
    assert dl.classify_error(dl.QuotaError("x")) == "quota"
    assert dl.classify_error(dl.AuthError("x")) == "auth"
    assert dl.classify_error(dl.TransientError("x")) == "transient"
    assert dl.classify_error(RuntimeError("QuotaExceeded: over cap")) == "quota"
    assert dl.classify_error(RuntimeError("SlowDown, please retry")) == "transient"
    assert dl.classify_error(RuntimeError("RequestTimeout")) == "transient"
    assert dl.classify_error(RuntimeError("503 Service Unavailable")) == "transient"
    assert dl.classify_error(ValueError("schema drift: unknown column")) == "fatal"


def test_classify_error_auth_failures_are_hard_stops():
    # wrong subscriber keys / wrong AWS account → a run-fatal setup error, not a per-unit fatal.
    assert dl.classify_error(RuntimeError("AccessDenied: not authorized")) == "auth"
    assert dl.classify_error(RuntimeError("InvalidAccessKeyId")) == "auth"
    assert dl.classify_error(RuntimeError("SignatureDoesNotMatch")) == "auth"
    assert dl.classify_error(RuntimeError("The security token has expired")) == "auth"
    assert dl.classify_error(RuntimeError("403 malformed row 4037")) == "fatal"   # bare 403 not auth
    assert issubclass(dl.AuthError, dl.HardStop) and issubclass(dl.QuotaError, dl.HardStop)


def test_classify_error_markers_are_not_over_broad():
    # bare digit markers must NOT swallow ordinary messages that merely contain '500'/'502'/… —
    # those are genuine fatals, not transient (would otherwise burn retries re-downloading the day).
    assert dl.classify_error(ValueError("expected 500 columns, got 12")) == "fatal"
    assert dl.classify_error(ValueError("malformed row 5040")) == "fatal"
    # a throttle worded 'reduce your request rate' is transient, NOT a quota hard stop.
    assert dl.classify_error(RuntimeError("Please reduce your request rate")) == "transient"


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


def test_run_partial_failure_exits_3(tmp_path):
    reader = _perp_reader()
    reader.plan["trades"] = dl.TransientError("RequestTimeout")     # one feed always fails
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep"),
                    "--retries", "2"],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 3


def test_run_quota_error_exits_2(tmp_path):
    reader = _perp_reader()
    reader.plan["book_delta_v2"] = dl.QuotaError("Quota exceeded")
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 2


def test_run_broad_gate_blocks_before_any_read(tmp_path):
    reader = _perp_reader()
    with pytest.raises(SystemExit) as e:
        dl.main(["--instrument", "binance-perp", "--start", "2026-01-01", "--end", "2026-12-31",
                 "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert e.value.code == lb.BROAD_GATE_EXIT == 4
    assert reader.calls == []                              # gate runs BEFORE any vendor read


def test_run_allow_broad_still_blocked_over_quota_headroom(tmp_path):
    # even --allow-broad cannot breach the 300 GB/month headroom (used_data lags ~60 min)
    reader = _perp_reader()
    with pytest.raises(SystemExit) as e:
        dl.main(["--instrument", "binance-perp,binance-spot", "--start", "2026-01-01",
                 "--end", "2026-12-31", "--allow-broad", "--out", str(tmp_path / "raw"),
                 "--report-dir", str(tmp_path / "rep")],
                reader=reader, used_data_fn=lambda: 295.0, sleep=lambda *_: None)
    assert e.value.code == 4


def test_run_unreadable_used_data_fails_safe_exit_2(tmp_path):
    def _boom():
        raise RuntimeError("used_data unreadable")
    code = dl.main(["--instrument", "binance-perp", "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_perp_reader(), used_data_fn=_boom, sleep=lambda *_: None)
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


def test_sparse_accepted_reads_manifest(tmp_path):
    root = str(tmp_path)
    lb.manifest_append(root, {"feed": "liquidations", "exchange": PERP[0], "symbol": PERP[1],
                              "dt": "2026-04-01", "status": "missing", "sparse_ok": True})
    lb.manifest_append(root, {"feed": "trades", "exchange": PERP[0], "symbol": PERP[1],
                              "dt": "2026-04-01", "status": "missing", "sparse_ok": False})
    acc = dl.sparse_accepted(root)
    assert ("liquidations", *PERP, "2026-04-01") in acc          # sparse-ok miss is accepted (done)
    assert ("trades", *PERP, "2026-04-01") not in acc            # required miss stays pending
    # a later ok record (parquet written) supersedes the earlier sparse acceptance
    lb.manifest_append(root, {"feed": "liquidations", "exchange": PERP[0], "symbol": PERP[1],
                              "dt": "2026-04-01", "status": "ok", "rows": 5})
    assert ("liquidations", *PERP, "2026-04-01") not in dl.sparse_accepted(root)


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
    # resume via the LIVE path (no injected reader/used_data_fn); make setup explode if reached.
    monkeypatch.setattr(dl, "lake_session",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no creds")))
    def _boom_used():
        raise AssertionError("used_data must not be called on a no-op resume")
    assert dl.main(argv, used_data_fn=_boom_used, sleep=lambda *_: None) == 0
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


def test_run_invalid_instrument_feed_pair_exits_2(tmp_path):
    code = dl.main(["--instrument", "binance-spot", "--feeds", "funding",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_perp_reader(), used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 2                                       # funding is perp-only


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


def test_run_reversed_date_range_exits_2(tmp_path):
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--start", "2026-04-02", "--end", "2026-04-01", "--out", str(tmp_path / "raw"),
                    "--report-dir", str(tmp_path / "rep")],
                   reader=_perp_reader(), used_data_fn=lambda: 0.0)
    assert code == 2


def test_run_live_setup_failure_exits_2(tmp_path, monkeypatch):
    # with no injected reader/used_data_fn, main() builds the live session; a setup failure (missing
    # AWS keys, or boto3 absent) must return the documented exit 2, not raise / exit 1.
    def _boom(*a, **k):
        raise RuntimeError("Crypto Lake AWS key 'aws_access_key_id' not found")
    monkeypatch.setattr(dl, "lake_session", _boom)
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   sleep=lambda *_: None)               # no reader, no used_data_fn → forces session
    assert code == 2


def test_run_no_day_source_exits_2(tmp_path):
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_perp_reader(), used_data_fn=lambda: 0.0)
    assert code == 2


# --------------------------------------------------------------------------- parallel (--jobs) path
def test_run_jobs_parallel_writes_all_units_with_valid_manifest(tmp_path):
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


def test_run_jobs_quota_hard_stop_cancels_pending_units(tmp_path):
    # a quota/credit wall under --jobs must exit 2 AND cancel the queued units (not drain them):
    # only the ≤jobs already-in-flight units may run, so far fewer than all units are ever read.
    raw = tmp_path / "raw"
    reader = FakeReader({"book_delta_v2": dl.QuotaError("Quota exceeded"),
                         "book": [_book_snapshot_frame(nrows=1)]})
    code = dl.main(["--instrument", "binance-perp", "--feeds", "book_delta_v2",
                    "--start", "2026-04-01", "--end", "2026-04-30", "--jobs", "2", "--allow-broad",
                    "--out", str(raw), "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert code == 2
    # 30 days × (book_delta_v2 + book seed) = 60 units; a drained pool would read most of them.
    assert len(reader.calls) < 15                         # pending units were cancelled, not drained


# --------------------------------------------------------------------------- dry-run (fake lister)
def test_dry_run_uses_lister_and_writes_no_parquet(tmp_path):
    seen = []

    def fake_lister(feed, exchange, symbol):
        seen.append(feed)
        return ["2026-04-01"]                              # present days for this feed

    raw = tmp_path / "raw"
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades", "--dry-run",
                    "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(raw),
                    "--report-dir", str(tmp_path / "rep")],
                   lister=fake_lister, used_data_fn=lambda: 0.0)
    assert code == 0
    assert seen == ["trades"]
    assert not any(raw.rglob("*.parquet"))                # dry-run transfers zero parquet
    rep = json.loads(next((tmp_path / "rep").glob("*.json")).read_text())
    assert rep["dry_run"] is True and rep["transferred_gb"] == 0


def test_dry_run_auth_error_exits_2(tmp_path):
    # --dry-run's list_data is a LIVE Lake call; an auth wall must return the documented exit 2,
    # not escape as a traceback / exit 1.
    def bad_lister(feed, exchange, symbol):
        raise RuntimeError("AccessDenied: not authorized for this bucket")
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades", "--dry-run",
                    "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(tmp_path / "raw"),
                    "--report-dir", str(tmp_path / "rep")],
                   lister=bad_lister, used_data_fn=lambda: 0.0)
    assert code == 2


def test_pyproject_declares_lake_downloader_extra():
    # the downloader imports pyarrow/lakeapi/boto3 in its live path — they must be declared as an
    # installable extra so a base install running the CLI does not ModuleNotFoundError.
    import tomllib
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    lake = data["project"]["optional-dependencies"]["lake"]
    for dep in ("pyarrow", "lakeapi", "boto3"):
        assert any(dep in d for d in lake), f"{dep} missing from the `lake` extra"


# --------------------------------------------------------------------------- live-path helpers (offline)
def test_present_days_from_list_records_reads_dt_key():
    # lakeapi.list_data returns dicts keyed by `dt` — read that key, never stringify the whole dict
    # (which would make every requested day compare as missing on a live --dry-run).
    recs = [{"table": "trades", "exchange": "BINANCE", "symbol": "BTC-USDT",
             "dt": "2026-04-02", "filename": "b.parquet"},
            {"table": "trades", "exchange": "BINANCE", "symbol": "BTC-USDT",
             "dt": "2026-04-01", "filename": "a.parquet"}]
    assert dl.present_days_from_list_records(recs) == ["2026-04-01", "2026-04-02"]
    assert dl.present_days_from_list_records([{"no": "dt"}]) == []   # missing dt skipped, not bogus
    assert dl.present_days_from_list_records([]) == []
    assert dl.present_days_from_list_records(None) == []


def test_used_gb_from_response_reads_downloaded_gb():
    # lakeapi.used_data returns a dict ALREADY in GB — read downloaded_gb, no bytes conversion.
    assert dl.used_gb_from_response({"downloaded_gb": 151.35, "timeframe_days": 31}) \
        == pytest.approx(151.35)
    assert dl.used_gb_from_response({"downloaded_gb": 0}) == 0.0
    # a bare number (the old `float(used_data(...))` bug) or a missing key must FAIL, so main() exits
    # 2 fail-safe rather than gating against a wrong 0 usage.
    with pytest.raises((KeyError, TypeError)):
        dl.used_gb_from_response(151.35)
    with pytest.raises(KeyError):
        dl.used_gb_from_response({"timeframe_days": 31})


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


# --------------------------------------------------------------------------- import safety
def test_import_is_vendor_safe():
    # Requirement 1: the module must import with NO boto3/lakeapi AND no pyarrow at module top. We
    # cannot assert `'pyarrow' not in sys.modules` (pandas pulls it in transitively via recon.ingest),
    # so we BLOCK pyarrow (sys.modules['pyarrow']=None → any `import pyarrow` raises) and assert the
    # import still succeeds — proving every pyarrow touch is lazy, inside a live function. A hoisted
    # top-level `import pyarrow` would raise here and fail the test (the likeliest Req-1 regression).
    prog = (
        "import sys\n"
        "sys.modules['pyarrow'] = None\n"           # subsequent `import pyarrow` raises ImportError
        "import ingest.download_lake_binance\n"
        "assert 'boto3' not in sys.modules, 'boto3 imported at module load'\n"
        "assert 'lakeapi' not in sys.modules, 'lakeapi imported at module load'\n"
        "print('ok')\n")
    out = subprocess.run([sys.executable, "-c", prog], cwd=str(_ROOT),
                         capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "ok"
