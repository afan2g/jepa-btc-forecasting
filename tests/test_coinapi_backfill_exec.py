"""Execution tests for the reviewed-manifest CoinAPI backfill mode of
`ingest/download_coinapi.py` (issue #53). Everything runs against an INJECTED fake S3 client and
synthetic gz payloads under tmp_path — no live vendor call, no credentials, no real download.

Covers: dry-run default (plan written, zero vendor calls), execute downloading exactly the
planned sparse units (never intervening dates), the trades product handler + normalized parquet,
fail-closed header drift, spend-authorization refusals, the §5a multi-day gate, fingerprint-keyed
resume, conflict fail-closed, and the reconciled execution report."""
import gzip
import hashlib
import io
import json
import os
import pathlib as _pl
import sys

import pytest

# the downloader imports pyarrow/botocore at module import time; neither is in pyproject's base
# dependencies (they ride the [lake]/[baseline] extras), so skip — not error — in a light install.
# The stdlib-only planning tests live in test_coinapi_backfill_plan.py and deliberately do NOT skip.
pytest.importorskip("pyarrow")
pytest.importorskip("botocore")

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import coinapi_backfill_fixtures as fx  # noqa: E402

bf = fx.load_by_path("coinapi_backfill", "ingest/coinapi_backfill.py")
dl = fx.load_by_path("download_coinapi", "ingest/download_coinapi.py")
dl.ff.RL.interval = 0.0                       # no throttling sleeps in tests

BOOK_HEADER = "time_exchange;time_coinapi;update_type;is_buy;entry_px;entry_sx;order_id"
# documented flat-file TRADES header: the four trailing identifier columns follow taker_side
TRADES_HEADER = ("time_exchange;time_coinapi;guid;price;base_amount;taker_side;"
                 "id_exch_guid;id_exch_int_inc;order_id_maker;order_id_taker")


def book_gz(n=3):
    rows = [f"12:00:0{i}.0000000;12:00:0{i}.0100000;ADD;1;100.5;0.2{i};ord-{i}" for i in range(n)]
    return gzip.compress(("\n".join([BOOK_HEADER] + rows) + "\n").encode())


def trades_row(i, time_exchange, time_coinapi):
    return (f"{time_exchange};{time_coinapi};g-{i};100.5;0.2{i};BUY;"
            f"eg-{i};{i};mk-{i};tk-{i}")


def trades_gz(n=2, header=TRADES_HEADER, day="2025-01-10"):
    # CoinAPI flat-file TRADES time columns are FULL datetimes (docs show
    # 2025-02-14T13:30:03.5851480), unlike the book's time-of-day offsets
    rows = [trades_row(i, f"{day}T12:00:0{i}.0000000", f"{day}T12:00:0{i}.0100000")
            for i in range(n)]
    return gzip.compress(("\n".join([header] + rows) + "\n").encode())


def _key(data_type, day):
    compact = day.replace("-", "")
    return (f"T-{data_type}/D-{compact}/E-COINBASE/"
            f"IDDI-1234+SC-COINBASE_SPOT_BTC_USD+S-1.csv.gz")


class FakeS3:
    """Injected vendor client: list_objects_v2/get_object over an in-memory {key: bytes} store,
    with a call log so tests can prove exactly which days were touched."""
    def __init__(self, objects):
        self.objects = dict(objects)
        self.calls = []

    def list_objects_v2(self, Bucket, Prefix, **kw):
        self.calls.append(("list", Prefix))
        hits = [{"Key": k, "Size": len(v)} for k, v in sorted(self.objects.items())
                if k.startswith(Prefix)]
        return {"Contents": hits, "IsTruncated": False}

    def get_object(self, Bucket, Key, **kw):
        self.calls.append(("get", Key))
        return {"Body": io.BytesIO(self.objects[Key])}


def default_objects():
    return {
        _key("LIMITBOOK_FULL", "2025-01-02"): book_gz(3),
        _key("LIMITBOOK_FULL", "2025-01-03"): book_gz(4),
        _key("LIMITBOOK_FULL", "2025-01-10"): book_gz(5),
        _key("TRADES", "2025-01-10"): trades_gz(2, day="2025-01-10"),
        _key("TRADES", "2025-01-11"): trades_gz(3, day="2025-01-11"),
    }


@pytest.fixture
def ready(tmp_path):
    mpath, manifest = fx.ready_manifest(tmp_path)
    return {"manifest_path": mpath, "manifest": manifest, "sha": bf.sha256_file(mpath),
            "out": tmp_path / "raw", "plan_out": tmp_path / "plan.json",
            "report_out": tmp_path / "report.json"}


def run_cli(ready, extra, s3_factory=None):
    argv = ["--manifest", ready["manifest_path"], "--out", str(ready["out"]),
            "--plan-out", str(ready["plan_out"]), "--report-out", str(ready["report_out"]),
            "--generated-utc", "2026-07-10T03:00:00Z"] + extra
    return dl.main(argv, s3_factory=s3_factory)


def execute_args(ready, approve="10.0"):
    return ["--manifest-sha256", ready["sha"], "--execute", "--approve-usd", approve,
            "--spend-evidence", "issue #33 spend approval", "--allow-backfill"]


def _final(ready, product, day):
    return (ready["out"] / product / "exchange=COINBASE" / "symbol=BTC-USD"
            / f"dt={day}" / "data.parquet")


def _report(ready):
    with open(ready["report_out"]) as f:
        return json.load(f)


# =========================================================================== dry-run default
def test_dry_run_is_default_and_makes_no_vendor_calls(ready):
    def forbidden():
        raise AssertionError("dry-run must not build a vendor client")
    rc = run_cli(ready, [], s3_factory=forbidden)
    assert rc == 0
    with open(ready["plan_out"]) as f:
        plan = json.load(f)
    assert plan["meta"]["mode"] == "dry_run"
    assert [(u["product"], u["day"]) for u in plan["units"]] == [
        ("limitbook_full", "2025-01-02"), ("limitbook_full", "2025-01-03"),
        ("limitbook_full", "2025-01-10"), ("trades", "2025-01-10"), ("trades", "2025-01-11")]
    assert not ready["report_out"].exists()
    assert not (ready["out"] / "_backfill_state").exists()


def test_dry_run_refuses_blocking_manifest(tmp_path):
    mpath, _ = fx.blocking_manifest(tmp_path)
    rc = dl.main(["--manifest", mpath, "--plan-out", str(tmp_path / "p.json"),
                  "--generated-utc", "2026-07-10T03:00:00Z"])
    assert rc == 3
    assert not (tmp_path / "p.json").exists()


# =========================================================================== execute
def test_execute_downloads_exactly_the_planned_units(ready):
    fake = FakeS3(default_objects())
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 0
    for product, day in (("limitbook_full", "2025-01-02"), ("limitbook_full", "2025-01-03"),
                         ("limitbook_full", "2025-01-10"), ("trades", "2025-01-10"),
                         ("trades", "2025-01-11")):
        assert _final(ready, product, day).exists(), (product, day)
    # every vendor call targeted a planned unit's day — never an intervening date
    touched = {c[1].split("/D-")[1][:8] for c in fake.calls}
    assert touched == {"20250102", "20250103", "20250110", "20250111"}
    gets = [c for c in fake.calls if c[0] == "get"]
    assert len(gets) == 5

    report = _report(ready)
    rec = report["reconciliation"]
    assert rec["planned"] == 5 and rec["ok"] == 5 and rec["complete"] is True
    assert rec["bytes_downloaded"] == sum(len(v) for v in default_objects().values())
    assert report["spend"]["spend_evidence"] == "issue #33 spend approval"
    assert report["spend"]["approve_usd"] == 10.0
    assert report["meta"]["manifest"]["sha256"] == ready["sha"]
    by_key = {(u["product"], u["day"]): u for u in report["units"]}
    src = by_key[("trades", "2025-01-11")]
    assert src["src_sha256"] == hashlib.sha256(default_objects()[_key("TRADES", "2025-01-11")]).hexdigest()
    assert src["rows"] == 3 and src["out_sha256"] and src["out_bytes"] > 0

    # resume state keyed on source/product/day + manifest fingerprint
    st = bf.state_path(str(ready["out"]), ready["sha"], "trades", "2025-01-11")
    with open(st) as f:
        state = json.load(f)
    assert state["manifest_sha256"] == ready["sha"] and state["status"] == "ok"

    # legacy manifest rows preserved, now stamped with product + manifest fingerprint
    # (the run-authorization row has kind=backfill_run and no per-day fields — exclude it here)
    lines = [json.loads(x) for x in
             (ready["out"] / "_manifest.jsonl").read_text().splitlines()]
    lines = [r for r in lines if r.get("kind") != "backfill_run"]
    assert {(r["product"], r["dt"]) for r in lines} == {
        ("limitbook_full", "2025-01-02"), ("limitbook_full", "2025-01-03"),
        ("limitbook_full", "2025-01-10"), ("trades", "2025-01-10"), ("trades", "2025-01-11")}
    assert all(r["manifest_sha256"] == ready["sha"] for r in lines)


def test_trades_parquet_is_normalized(ready):
    import pyarrow.parquet as pq
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    # ParquetFile reads the file alone (read_table would infer hive partition cols from the path)
    t = pq.ParquetFile(_final(ready, "trades", "2025-01-10")).read()
    # the NORMALIZED trade contract (trade-validation plan Phase 3b / recon ingest:
    # origin_time, received_time, price, quantity, side∈{buy,sell}, trade_id) so trade_checks
    # validates CoinAPI fills unchanged — plus the vendor identifiers preserved losslessly
    assert t.column_names == ["seq", "origin_time", "received_time", "price", "quantity",
                              "side", "trade_id", "taker_side", "id_exch_guid",
                              "id_exch_int_inc", "order_id_maker", "order_id_taker"]
    assert t.column("seq").to_pylist() == [0, 1]
    assert t.column("trade_id").to_pylist() == ["g-0", "g-1"]
    assert t.column("side").to_pylist() == ["buy", "buy"]
    assert t.column("taker_side").to_pylist() == ["BUY", "BUY"]     # vendor value verbatim
    assert t.column("price").to_pylist() == [100.5, 100.5]
    assert t.column("quantity").to_pylist() == [0.20, 0.21]
    # the documented exchange/order identifiers ride along losslessly
    assert t.column("id_exch_guid").to_pylist() == ["eg-0", "eg-1"]
    assert t.column("id_exch_int_inc").to_pylist() == ["0", "1"]
    assert t.column("order_id_maker").to_pylist() == ["mk-0", "mk-1"]
    assert t.column("order_id_taker").to_pylist() == ["tk-0", "tk-1"]
    # "2025-01-10T12:00:00.0000000" -> absolute datetime64[ns] (Lake convention, naive UTC)
    import datetime
    assert t.column("origin_time").to_pylist()[0] == datetime.datetime(2025, 1, 10, 12, 0, 0)
    assert t.column("received_time").to_pylist()[1] == datetime.datetime(2025, 1, 10, 12, 0, 1,
                                                                         10000)


def test_trades_normalized_frame_passes_trade_checks(ready):
    # the whole point of the normalized contract: ingest/trade_checks.py consumes the parquet
    # unchanged (source-agnostic Phase-3b reuse, vendor_source="coinapi"), so a CoinAPI fill day
    # can clear coinapi_fill_deferred. A 2-row synthetic day rightly trips coverage METRICS —
    # what must hold is that no SCHEMA-level code fires (columns/side/time all understood).
    import pandas as pd
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    tc = fx.load_by_path("trade_checks", "ingest/trade_checks.py")
    df = pd.read_parquet(_final(ready, "trades", "2025-01-10"))
    rec = tc.validate_trade_frame(df, venue="coinbase", day="2025-01-10",
                                  vendor_source="coinapi")
    schema_codes = {tc.REQUIRED_COLUMN_MISSING, tc.ORIGIN_TIME_COLUMN_MISSING,
                    tc.SIDE_COLUMN_MISSING, tc.SIDE_VALUE_UNEXPECTED}
    assert not (set(rec["reason_codes"]) & schema_codes), rec["reason_codes"]
    m = rec["metrics"]
    assert m["dup_trade_id_count"] == 0 and m["trade_id_available"] is True


def test_trades_side_normalization(ready):
    import pyarrow.parquet as pq
    objs = default_objects()
    rows = ["2025-01-11T12:00:00.0000000;2025-01-11T12:00:00.0100000;g-0;100.5;0.20;"
            "SELL_ESTIMATED;eg-0;0;mk-0;tk-0",
            "2025-01-11T12:00:01.0000000;2025-01-11T12:00:01.0100000;g-1;100.5;0.21;"
            "BUY;eg-1;1;mk-1;tk-1"]
    objs[_key("TRADES", "2025-01-11")] = gzip.compress(
        ("\n".join([TRADES_HEADER] + rows) + "\n").encode())
    fake = FakeS3(objs)
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    t = pq.ParquetFile(_final(ready, "trades", "2025-01-11")).read()
    # *_ESTIMATED maps to its side; the raw vendor value stays available verbatim
    assert t.column("side").to_pylist() == ["sell", "buy"]
    assert t.column("taker_side").to_pylist() == ["SELL_ESTIMATED", "BUY"]


def test_trades_unmappable_side_fails_closed(ready):
    objs = default_objects()
    rows = ["2025-01-11T12:00:00.0000000;2025-01-11T12:00:00.0100000;g-0;100.5;0.20;"
            "UNKNOWN;eg-0;0;mk-0;tk-0"]
    objs[_key("TRADES", "2025-01-11")] = gzip.compress(
        ("\n".join([TRADES_HEADER] + rows) + "\n").encode())
    fake = FakeS3(objs)
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    assert not _final(ready, "trades", "2025-01-11").exists()
    unit = {(u["product"], u["day"]): u
            for u in _report(ready)["units"]}[("trades", "2025-01-11")]
    assert unit["status"] == "error" and "taker_side" in unit["error"]


def test_trades_without_identifier_columns_normalizes_with_nulls(ready):
    # only the six core columns are REQUIRED; a file lacking the documented identifier columns
    # still normalizes, with the identifier fields null (never fabricated)
    import pyarrow.parquet as pq
    objs = default_objects()
    core = "time_exchange;time_coinapi;guid;price;base_amount;taker_side"
    rows = [f"2025-01-11T12:00:0{i}.0000000;2025-01-11T12:00:0{i}.0100000;g-{i};100.5;0.2{i};BUY"
            for i in range(2)]
    objs[_key("TRADES", "2025-01-11")] = gzip.compress(("\n".join([core] + rows) + "\n").encode())
    fake = FakeS3(objs)
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    t = pq.ParquetFile(_final(ready, "trades", "2025-01-11")).read()
    assert t.column("trade_id").to_pylist() == ["g-0", "g-1"]
    assert t.column("id_exch_guid").to_pylist() == [None, None]
    assert t.column("order_id_taker").to_pylist() == [None, None]


def test_trades_time_of_day_offset_form_also_normalizes(ready):
    # tolerate the book-style offset form too: either vendor format lands as the same absolute
    # origin_time normalization (offset resolved against the partition day's midnight UTC)
    import datetime
    import pyarrow.parquet as pq
    objs = default_objects()
    rows = [trades_row(i, f"12:00:0{i}.0000000", f"12:00:0{i}.0100000") for i in range(3)]
    objs[_key("TRADES", "2025-01-11")] = gzip.compress(
        ("\n".join([TRADES_HEADER] + rows) + "\n").encode())
    fake = FakeS3(objs)
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    t = pq.ParquetFile(_final(ready, "trades", "2025-01-11")).read()
    assert t.column("origin_time").to_pylist() == [datetime.datetime(2025, 1, 11, 12, 0, i)
                                                   for i in range(3)]


def test_trades_datetime_outside_partition_day_is_preserved(ready):
    # faithful-lossless: a record stamped past the partition day's midnight keeps its true
    # absolute time rather than being clamped or wrapped — recon owns day-boundary logic
    import datetime
    import pyarrow.parquet as pq
    objs = default_objects()
    rows = [trades_row(0, "2025-01-11T23:59:59.0000000", "2025-01-11T23:59:59.0100000"),
            trades_row(1, "2025-01-12T00:00:00.0000000", "2025-01-12T00:00:00.0100000")]
    objs[_key("TRADES", "2025-01-11")] = gzip.compress(
        ("\n".join([TRADES_HEADER] + rows) + "\n").encode())
    fake = FakeS3(objs)
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    t = pq.ParquetFile(_final(ready, "trades", "2025-01-11")).read()
    assert t.column("origin_time").to_pylist() == [
        datetime.datetime(2025, 1, 11, 23, 59, 59), datetime.datetime(2025, 1, 12, 0, 0, 0)]


def test_execute_resume_skips_done_units_without_vendor_calls(ready):
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    fake2 = FakeS3(default_objects())
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake2, "coinapi"))
    assert rc == 0
    assert fake2.calls == []
    rep = _report(ready)
    rec = rep["reconciliation"]
    assert rec["done_prior"] == 5 and rec["ok"] == 0 and rec["complete"] is True
    # a resumed report stays auditable from the report alone: the done units carry the audit
    # fields (rows, src bytes/hash) recorded in state — but bill nothing this run
    done = {(u["product"], u["day"]): u for u in rep["units"]}[("trades", "2025-01-11")]
    assert done["rows"] == 3 and done["src_sha256"] and done["src_bytes"] > 0
    assert done["out_sha256"] and done.get("billed_bytes", 0) == 0
    assert rec["bytes_downloaded"] == 0


def test_foreign_output_is_a_conflict_and_never_overwritten(ready):
    final = _final(ready, "limitbook_full", "2025-01-02")
    final.parent.mkdir(parents=True)
    final.write_bytes(b"FOREIGN")                 # no state record vouches for this file
    fake = FakeS3(default_objects())
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    assert final.read_bytes() == b"FOREIGN"       # fail closed: not adopted, not overwritten
    assert not any("20250102" in c[1] for c in fake.calls if c[0] == "get")
    rec = _report(ready)["reconciliation"]
    assert rec["conflict"] == 1 and rec["ok"] == 4 and rec["complete"] is False
    assert not os.path.exists(
        bf.state_path(str(ready["out"]), ready["sha"], "limitbook_full", "2025-01-02"))


@pytest.mark.parametrize("drop", ["--manifest-sha256", "--approve-usd", "--spend-evidence"])
def test_execute_refused_without_authorization(ready, drop):
    args = execute_args(ready)
    i = args.index(drop)
    del args[i:i + 2]
    called = []
    rc = run_cli(ready, args, s3_factory=lambda: called.append(1))
    assert rc == 3
    assert called == []
    assert not ready["report_out"].exists()


def test_execute_refused_when_approval_below_planned_cost(ready):
    called = []
    rc = run_cli(ready, execute_args(ready, approve="1.00"),
                 s3_factory=lambda: called.append(1))
    assert rc == 3 and called == []


@pytest.mark.parametrize("approve", ["nan", "inf", "-1"])
def test_execute_refused_on_nonfinite_or_negative_approval(ready, approve):
    # float('nan') < high is False — a naive `<` check would let NaN through the spend gate
    called = []
    rc = run_cli(ready, execute_args(ready, approve=approve),
                 s3_factory=lambda: called.append(1))
    assert rc == 3 and called == []


def test_zero_unit_manifest_executes_to_empty_complete_report(tmp_path):
    # a ready manifest with nothing to fill: execute must not build a vendor client at all
    cal = fx.make_calendar(coinbase_fill_days={}, fill_status={}, excluded_days_by_reason={})
    days = [fx.day_rec(d, "lake_usable", fx.fill_block(False, "lake_usable"))
            for d in ("2025-01-01", "2025-01-02", "2025-01-03")]
    mpath, _ = fx.ready_manifest(tmp_path, cal=cal, report_days=days)
    sha = bf.sha256_file(mpath)
    def forbidden():
        raise AssertionError("zero-unit execute must not build a vendor client")
    rc = dl.main(["--manifest", mpath, "--manifest-sha256", sha, "--execute",
                  "--approve-usd", "1.0", "--spend-evidence", "n/a", "--out",
                  str(tmp_path / "raw"), "--plan-out", str(tmp_path / "plan.json"),
                  "--report-out", str(tmp_path / "report.json"),
                  "--generated-utc", "2026-07-10T03:00:00Z"], s3_factory=forbidden)
    assert rc == 0
    with open(tmp_path / "report.json") as f:
        rec = json.load(f)["reconciliation"]
    assert rec["planned"] == 0 and rec["complete"] is True


def test_execute_refused_on_pinned_hash_mismatch(ready):
    args = execute_args(ready)
    args[args.index("--manifest-sha256") + 1] = "0" * 64
    rc = run_cli(ready, args, s3_factory=lambda: (_ for _ in ()).throw(AssertionError))
    assert rc == 3


def test_multi_day_execute_requires_allow_backfill(ready):
    args = [a for a in execute_args(ready) if a != "--allow-backfill"]
    def forbidden():
        raise AssertionError("gate must fire before any client is built")
    with pytest.raises(SystemExit) as ei:
        run_cli(ready, args, s3_factory=forbidden)
    assert ei.value.code == 4


def test_single_day_execute_stays_allowed_without_override(ready):
    # bounded single-day pull (the parity-pilot allowance) — window spans one calendar day
    fake = FakeS3(default_objects())
    args = [a for a in execute_args(ready, approve="2.0") if a != "--allow-backfill"]
    rc = run_cli(ready, args + ["--pilot-start", "2025-01-10", "--pilot-end", "2025-01-10"],
                 s3_factory=lambda: (fake, "coinapi"))
    assert rc == 0
    assert {c[1].split("/D-")[1][:8] for c in fake.calls} == {"20250110"}


def test_pilot_window_execute_downloads_subset_only(ready):
    fake = FakeS3(default_objects())
    rc = run_cli(ready, execute_args(ready, approve="1.15")
                 + ["--pilot-start", "2025-01-10", "--pilot-end", "2025-01-11"],
                 s3_factory=lambda: (fake, "coinapi"))
    assert rc == 0
    assert {c[1].split("/D-")[1][:8] for c in fake.calls} == {"20250110", "20250111"}
    assert not _final(ready, "limitbook_full", "2025-01-02").exists()
    rec = _report(ready)["reconciliation"]
    assert rec["planned"] == 3 and rec["ok"] == 3 and rec["complete"] is True


def test_trades_header_drift_fails_closed(ready):
    objs = default_objects()
    bad = "time_exchange;time_coinapi;guid;price;size;taker_side"   # size != base_amount
    objs[_key("TRADES", "2025-01-11")] = trades_gz(3, header=bad)
    fake = FakeS3(objs)
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    assert not _final(ready, "trades", "2025-01-11").exists()
    rep = _report(ready)
    unit = {(u["product"], u["day"]): u for u in rep["units"]}[("trades", "2025-01-11")]
    assert unit["status"] == "error" and "base_amount" in unit["error"]
    assert rep["reconciliation"]["complete"] is False
    # post-GET failure keeps the source audit trail: the billed bytes are identified exactly
    assert unit["src_bytes"] > 0 and unit["src_sha256"]
    assert unit["billed_bytes"] == unit["src_bytes"]


def test_missing_vendor_file_is_recorded_not_fatal(ready):
    objs = default_objects()
    del objs[_key("LIMITBOOK_FULL", "2025-01-03")]
    fake = FakeS3(objs)
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    rec = _report(ready)["reconciliation"]
    assert rec["missing"] == 1 and rec["ok"] == 4 and rec["complete"] is False


def test_book_parquet_is_normalized(ready):
    import pyarrow.parquet as pq
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    t = pq.ParquetFile(_final(ready, "limitbook_full", "2025-01-02")).read()
    assert t.column_names == ["seq", "time_exchange_ns", "time_coinapi_ns", "update_type",
                              "is_buy", "entry_px", "entry_sx", "order_id"]
    assert t.column("seq").to_pylist() == [0, 1, 2]
    assert t.column("update_type").to_pylist() == ["ADD", "ADD", "ADD"]
    assert t.column("is_buy").to_pylist() == [True, True, True]
    assert t.column("order_id").to_pylist() == ["ord-0", "ord-1", "ord-2"]
    assert t.column("entry_sx").to_pylist() == [0.20, 0.21, 0.22]
    assert t.column("time_exchange_ns").to_pylist() == [(12 * 3600 + i) * 10**9 for i in range(3)]


def test_book_header_drift_fails_closed(ready):
    objs = default_objects()
    bad = BOOK_HEADER.replace(";order_id", "")     # order_id column gone
    rows = ["12:00:00.0000000;12:00:00.0100000;ADD;1;100.5;0.25"]
    objs[_key("LIMITBOOK_FULL", "2025-01-02")] = gzip.compress(
        ("\n".join([bad] + rows) + "\n").encode())
    fake = FakeS3(objs)
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    assert not _final(ready, "limitbook_full", "2025-01-02").exists()
    unit = {(u["product"], u["day"]): u
            for u in _report(ready)["units"]}[("limitbook_full", "2025-01-02")]
    assert unit["status"] == "error" and "order_id" in unit["error"]


def test_quota_abort_writes_report_and_exits_distinctly(ready):
    from botocore.exceptions import ClientError

    class QuotaS3(FakeS3):
        def get_object(self, Bucket, Key, **kw):
            if "20250103" in Key:
                self.calls.append(("get", Key))
                raise ClientError({"Error": {"Code": "403",
                                             "Message": "Insufficient Usage Credits"}},
                                  "GetObject")
            return super().get_object(Bucket, Key, **kw)

    fake = QuotaS3(default_objects())
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == dl.QUOTA_ABORT_EXIT == 5          # distinct from input-error exit 2
    rec = _report(ready)["reconciliation"]         # report still written on abort
    assert rec["planned"] == 5 and rec["accounted"] == 1 and rec["complete"] is False
    # nothing after the aborting unit was touched
    assert not any("20250110" in c[1] or "20250111" in c[1] for c in fake.calls)


def test_nonquota_clienterror_is_a_per_unit_error(ready):
    from botocore.exceptions import ClientError

    class FlakyS3(FakeS3):
        def list_objects_v2(self, Bucket, Prefix, **kw):
            if "20250103" in Prefix:
                raise ClientError({"Error": {"Code": "500", "Message": "InternalError"}},
                                  "ListObjectsV2")
            return super().list_objects_v2(Bucket, Prefix, **kw)

    fake = FlakyS3(default_objects())
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1                                 # run continues; report written
    rec = _report(ready)["reconciliation"]
    assert rec["error"] == 1 and rec["ok"] == 4 and rec["complete"] is False


def test_budget_guard_refuses_oversized_unit_before_any_get(ready):
    # the vendor LIST size is known before the billable GET: a unit whose projected cost would
    # push measured spend past --approve-usd is refused fail-closed, never downloaded
    class InflatedS3(FakeS3):
        def list_objects_v2(self, Bucket, Prefix, **kw):
            resp = super().list_objects_v2(Bucket, Prefix, **kw)
            if "20250102" in Prefix:
                for o in resp["Contents"]:
                    o["Size"] = 10_000_000_000     # 10 GB actual vs 2.27 GB estimated
            return resp

    fake = InflatedS3(default_objects())
    rc = run_cli(ready, execute_args(ready, approve="5.69"),
                 s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    assert not any(c[0] == "get" and "20250102" in c[1] for c in fake.calls)
    assert not _final(ready, "limitbook_full", "2025-01-02").exists()
    rec = _report(ready)["reconciliation"]
    assert rec["refused_budget"] == 1 and rec["complete"] is False


def test_failed_get_is_not_billed_but_delivered_bytes_are(ready):
    # billing commits when get_object returns a body (the billable moment): a GET that raises
    # before delivery must not consume the budget or report billed bytes, while a post-delivery
    # failure still bills — otherwise later units are refused against money never spent
    from botocore.exceptions import ClientError

    class FailingGetS3(FakeS3):
        def list_objects_v2(self, Bucket, Prefix, **kw):
            resp = super().list_objects_v2(Bucket, Prefix, **kw)
            if "20250102" in Prefix or "20250103" in Prefix:
                for o in resp["Contents"]:
                    o["Size"] = 4_000_000_000          # $4 projected per unit at $1/GB
            return resp

        def get_object(self, Bucket, Key, **kw):
            if "20250102" in Key:
                self.calls.append(("get", Key))
                raise ClientError({"Error": {"Code": "500", "Message": "InternalError"}},
                                  "GetObject")
            return super().get_object(Bucket, Key, **kw)

    fake = FailingGetS3(default_objects())
    rc = run_cli(ready, execute_args(ready, approve="5.69"),
                 s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    rep = _report(ready)
    by_key = {(u["product"], u["day"]): u for u in rep["units"]}
    # 01-02: GET raised before a body was delivered -> error, NOT billed
    assert by_key[("limitbook_full", "2025-01-02")]["status"] == "error"
    assert by_key[("limitbook_full", "2025-01-02")]["billed_bytes"] == 0
    assert by_key[("limitbook_full", "2025-01-02")]["src_sha256"] is None
    # 01-03: budget NOT consumed by the failed GET, so its own GET runs (4 <= 5.69) and its
    # delivered-then-size-mismatched bytes ARE billed
    assert any(c[0] == "get" and "20250103" in c[1] for c in fake.calls)
    assert by_key[("limitbook_full", "2025-01-03")]["billed_bytes"] == 4_000_000_000
    assert rep["reconciliation"]["refused_budget"] == 0


def test_billing_ledger_survives_run_boundaries(ready):
    # billed GETs are persisted to the append-only jsonl at the body-delivery moment: a later run
    # initializes its budget from the ledger, so repeated interrupted/failed retries can never
    # cumulatively re-bill past --approve-usd even though each single run stays under it
    class InflatedS3(FakeS3):
        def list_objects_v2(self, Bucket, Prefix, **kw):
            resp = super().list_objects_v2(Bucket, Prefix, **kw)
            if "20250102" in Prefix:
                for o in resp["Contents"]:
                    o["Size"] = 4_000_000_000          # $4 projected at $1/GB
            return resp

    # run 1: the 01-02 body IS delivered ($4 billed to the ledger), then fails post-GET
    # (size mismatch vs the lying LIST) — no state, no output
    fake = InflatedS3(default_objects())
    rc = run_cli(ready, execute_args(ready, approve="5.69"),
                 s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    ledger = [json.loads(x) for x in (ready["out"] / "_manifest.jsonl").read_text().splitlines()
              if '"billed_get"' in x]
    assert any(r["dt"] == "2025-01-02" and r["manifest_sha256"] == ready["sha"]
               and r["projected_usd"] >= 4.0 for r in ledger)
    # run 2, same cap: prior $4 is already committed, so retrying 01-02 (another $4) must be
    # refused BEFORE any GET — without the ledger this run would start from $0 and re-bill
    fake2 = InflatedS3(default_objects())
    rc = run_cli(ready, execute_args(ready, approve="5.69"),
                 s3_factory=lambda: (fake2, "coinapi"))
    assert rc == 1
    assert not any(c[0] == "get" and "20250102" in c[1] for c in fake2.calls)
    rep = _report(ready)
    by_key = {(u["product"], u["day"]): u for u in rep["units"]}
    assert by_key[("limitbook_full", "2025-01-02")]["status"] == "refused_budget"
    assert rep["spend"]["prior_billed_usd"] >= 4.0


def test_nonfinite_ledger_rows_are_ignored(ready):
    # json.loads accepts NaN/Infinity: a corrupt billed_get row must not poison the budget —
    # NaN would make every guard comparison false (fail-OPEN) and only crash at report write
    os.makedirs(ready["out"], exist_ok=True)
    with open(ready["out"] / "_manifest.jsonl", "a") as f:
        f.write('{"kind": "billed_get", "manifest_sha256": "%s", "projected_usd": NaN}\n'
                % ready["sha"])
        f.write('{"kind": "billed_get", "manifest_sha256": "%s", "projected_usd": Infinity}\n'
                % ready["sha"])
        f.write('{"kind": "billed_get", "manifest_sha256": "%s", "projected_usd": 0.5}\n'
                % ready["sha"])
    fake = FakeS3(default_objects())
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 0
    rep = _report(ready)
    assert rep["spend"]["prior_billed_usd"] == 0.5      # only the finite row counts
    assert rep["reconciliation"]["ok"] == 5


def test_overwrite_rerun_is_recorded_in_spend_evidence(ready):
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    assert _report(ready)["spend"]["overwrite"] is False
    fake2 = FakeS3(default_objects())
    rc = run_cli(ready, execute_args(ready) + ["--overwrite"],
                 s3_factory=lambda: (fake2, "coinapi"))
    assert rc == 0
    assert len([c for c in fake2.calls if c[0] == "get"]) == 5   # deliberate re-bill
    assert _report(ready)["spend"]["overwrite"] is True
    # the run authorization is durable in the append-only jsonl even if reports get replaced
    runs = [json.loads(x) for x in (ready["out"] / "_manifest.jsonl").read_text().splitlines()
            if '"backfill_run"' in x]
    assert len(runs) == 2
    assert runs[1]["overwrite"] is True and runs[1]["approve_usd"] == 10.0
    assert all(r["manifest_sha256"] == ready["sha"] for r in runs)


def test_execute_after_manifest_regenerated_conflicts(ready, tmp_path):
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    # regenerate the manifest (same content shape, new bytes -> new fingerprint)
    fx.mutate_manifest(ready["manifest_path"],
                       lambda m: m["meta"].update(generated_utc="2026-07-11T00:00:00Z"))
    new_sha = bf.sha256_file(ready["manifest_path"])
    assert new_sha != ready["sha"]
    fake2 = FakeS3(default_objects())
    args = execute_args(ready)
    args[args.index("--manifest-sha256") + 1] = new_sha
    rc = run_cli(ready, args, s3_factory=lambda: (fake2, "coinapi"))
    assert rc == 1
    assert fake2.calls == []                       # nothing adopted, nothing re-downloaded
    rec = _report(ready)["reconciliation"]
    assert rec["conflict"] == 5 and rec["complete"] is False


def test_corrupt_state_file_is_a_conflict(ready):
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    st = bf.state_path(str(ready["out"]), ready["sha"], "trades", "2025-01-11")
    with open(st, "w") as f:
        f.write("{corrupt")
    fake2 = FakeS3(default_objects())
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake2, "coinapi"))
    assert rc == 1
    rec = _report(ready)["reconciliation"]
    assert rec["conflict"] == 1 and rec["done_prior"] == 4


def test_keep_raw_never_archives_a_failed_unit(ready):
    objs = default_objects()
    objs[_key("TRADES", "2025-01-11")] = trades_gz(
        3, header="time_exchange;time_coinapi;guid;price;size;taker_side")
    fake = FakeS3(objs)
    rc = run_cli(ready, execute_args(ready) + ["--keep-raw"],
                 s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    raw = ready["out"] / "_raw_gz"
    assert (raw / "trades" / "exchange=COINBASE" / "symbol=BTC-USD" / "2025-01-10.csv.gz").exists()
    assert not (raw / "trades" / "exchange=COINBASE" / "symbol=BTC-USD"
                / "2025-01-11.csv.gz").exists()


def test_execution_report_units_carry_provenance(ready):
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    with open(ready["manifest_path"]) as f:
        day = {r["day"]: r for r in json.load(f)["days"]}
    unit = {(u["product"], u["day"]): u
            for u in _report(ready)["units"]}[("limitbook_full", "2025-01-03")]
    assert unit["provenance"]["fill"] == day["2025-01-03"]["book_fill"]   # stitch plan verbatim


def test_cli_process_exit_code_propagates_refusals(tmp_path):
    # the module footer must exit with main()'s return value: a refusal must reach the SHELL as
    # exit 3, not be swallowed into exit 0 (Codex P1)
    import subprocess
    mpath, _ = fx.blocking_manifest(tmp_path)
    proc = subprocess.run(
        [sys.executable, str(_pl.Path(dl.__file__)), "--manifest", mpath,
         "--plan-out", str(tmp_path / "p.json"), "--generated-utc", "2026-07-10T03:00:00Z"],
        capture_output=True, text=True, cwd=str(_pl.Path(dl.__file__).resolve().parents[1]))
    assert proc.returncode == 3, (proc.returncode, proc.stderr[-500:])
    assert "REFUSED" in proc.stderr


def test_post_get_failure_still_counts_against_the_spend_cap(ready):
    # once the GET ran, CoinAPI billed the bytes: a header/parse failure afterwards must still
    # accrue to the budget (and the report), or later units get budgeted as if it cost $0
    class LyingS3(FakeS3):
        def list_objects_v2(self, Bucket, Prefix, **kw):
            resp = super().list_objects_v2(Bucket, Prefix, **kw)
            if "20250102" in Prefix or "20250103" in Prefix:
                for o in resp["Contents"]:
                    o["Size"] = 4_000_000_000      # $4 projected per unit at $1/GB
            return resp

    fake = LyingS3(default_objects())
    # unit 01-02: passes the guard (4 <= 5.69), GET runs, then fails post-GET (size mismatch) ->
    # $4 must be committed. unit 01-03: 4 + 4 = 8 > 5.69 -> refused BEFORE its GET.
    rc = run_cli(ready, execute_args(ready, approve="5.69"),
                 s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    assert not any(c[0] == "get" and "20250103" in c[1] for c in fake.calls)
    rep = _report(ready)
    by_key = {(u["product"], u["day"]): u for u in rep["units"]}
    assert by_key[("limitbook_full", "2025-01-02")]["status"] == "error"
    assert by_key[("limitbook_full", "2025-01-03")]["status"] == "refused_budget"
    # the billed-but-failed unit's bytes are visible in the reconciled spend figures
    assert rep["reconciliation"]["bytes_downloaded"] >= 4_000_000_000
    assert rep["spend"]["measured_usd_at_model_rates"] >= 4.0


def test_manifest_mode_refuses_market_overrides(ready):
    # the reviewed manifest is pinned COINBASE/BTC-USD; a CLI override would download another
    # market's files and record them as Coinbase fills
    rc = run_cli(ready, ["--exchange", "BINANCE"])
    assert rc == 3
    rc = run_cli(ready, ["--symbol-out", "ETH-USD"])
    assert rc == 3


def test_process_day_range_path_still_works(tmp_path):
    # the pre-existing single-day parity path (issue #53: behavior preserved) driven by FakeS3
    import argparse
    import datetime
    fake = FakeS3(default_objects())
    ns = argparse.Namespace(out=str(tmp_path / "raw"), exchange="COINBASE",
                            symbol_tag="COINBASE_SPOT_BTC_USD", exchange_out="COINBASE",
                            symbol_out="BTC-USD", keep_raw=False, overwrite=False, sample_mb=0)
    os.makedirs(ns.out, exist_ok=True)
    status, rows = dl.process_day(fake, "coinapi", datetime.date(2025, 1, 2), ns)
    assert (status, rows) == ("ok", 3)
    final = (tmp_path / "raw" / "limitbook_full" / "exchange=COINBASE" / "symbol=BTC-USD"
             / "dt=2025-01-02" / "data.parquet")
    assert final.exists()
    assert dl.process_day(fake, "coinapi", datetime.date(2025, 1, 2), ns) == ("skip", 0)


# =========================================================================== CLI contract
def test_manifest_mode_rejects_range_and_sample_flags(ready):
    with pytest.raises(SystemExit) as ei:
        dl.main(["--manifest", ready["manifest_path"], "--start", "2025-01-01",
                 "--end", "2025-01-02"])
    assert ei.value.code == 2
    with pytest.raises(SystemExit) as ei:
        dl.main(["--manifest", ready["manifest_path"], "--sample-mb", "8"])
    assert ei.value.code == 2


def test_range_mode_still_requires_start_and_end():
    with pytest.raises(SystemExit) as ei:
        dl.main([])
    assert ei.value.code == 2
    with pytest.raises(SystemExit) as ei:
        dl.main(["--start", "2025-01-01"])
    assert ei.value.code == 2
