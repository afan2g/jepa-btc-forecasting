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

sys.path.insert(0, str(_pl.Path(__file__).resolve().parent))
import coinapi_backfill_fixtures as fx  # noqa: E402

bf = fx.load_by_path("coinapi_backfill", "ingest/coinapi_backfill.py")
dl = fx.load_by_path("download_coinapi", "ingest/download_coinapi.py")
dl.ff.RL.interval = 0.0                       # no throttling sleeps in tests

BOOK_HEADER = "time_exchange;time_coinapi;update_type;is_buy;entry_px;entry_sx;order_id"
TRADES_HEADER = "time_exchange;time_coinapi;guid;price;base_amount;taker_side"


def book_gz(n=3):
    rows = [f"12:00:0{i}.0000000;12:00:0{i}.0100000;ADD;1;100.5;0.2{i};ord-{i}" for i in range(n)]
    return gzip.compress(("\n".join([BOOK_HEADER] + rows) + "\n").encode())


def trades_gz(n=2, header=TRADES_HEADER):
    rows = [f"12:00:0{i}.0000000;12:00:0{i}.0100000;g-{i};100.5;0.2{i};BUY" for i in range(n)]
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
        _key("TRADES", "2025-01-10"): trades_gz(2),
        _key("TRADES", "2025-01-11"): trades_gz(3),
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
    lines = [json.loads(x) for x in
             (ready["out"] / "_manifest.jsonl").read_text().splitlines()]
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
    assert t.column_names == ["seq", "time_exchange_ns", "time_coinapi_ns", "guid", "price",
                              "base_amount", "taker_side"]
    assert t.column("seq").to_pylist() == [0, 1]
    assert t.column("guid").to_pylist() == ["g-0", "g-1"]
    assert t.column("taker_side").to_pylist() == ["BUY", "BUY"]
    assert t.column("price").to_pylist() == [100.5, 100.5]
    assert t.column("base_amount").to_pylist() == [0.20, 0.21]
    # "12:00:00.0000000" -> ns since midnight UTC
    assert t.column("time_exchange_ns").to_pylist()[0] == 12 * 3600 * 10**9


def test_execute_resume_skips_done_units_without_vendor_calls(ready):
    fake = FakeS3(default_objects())
    assert run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi")) == 0
    fake2 = FakeS3(default_objects())
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake2, "coinapi"))
    assert rc == 0
    assert fake2.calls == []
    rec = _report(ready)["reconciliation"]
    assert rec["done_prior"] == 5 and rec["ok"] == 0 and rec["complete"] is True


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


def test_missing_vendor_file_is_recorded_not_fatal(ready):
    objs = default_objects()
    del objs[_key("LIMITBOOK_FULL", "2025-01-03")]
    fake = FakeS3(objs)
    rc = run_cli(ready, execute_args(ready), s3_factory=lambda: (fake, "coinapi"))
    assert rc == 1
    rec = _report(ready)["reconciliation"]
    assert rec["missing"] == 1 and rec["ok"] == 4 and rec["complete"] is False


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
