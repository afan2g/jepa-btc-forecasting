"""Pyarrow-FREE downloader tests — the non-vendor CLI surface (error taxonomy, exit-code contract,
dry-run, import-safety, pure helpers). Deliberately NO `pytest.importorskip("pyarrow")`, so this
coverage runs in the lightweight default-CI path that lacks the `lake`/`baseline` extra (the
Parquet-writing / streaming tests live in test_download_lake_binance.py, which does importorskip).

Everything here calls pure `dl`/`lb` functions or exercises code paths that exit BEFORE any partition
is read/written, so no pyarrow is needed. Readers here must never actually be invoked."""
import json
import pathlib
import subprocess
import sys
import tomllib
import types

import pytest

from ingest import download_lake_binance as dl
from ingest import lake_binance as lb

_ROOT = pathlib.Path(__file__).resolve().parents[1]
PERP = ("BINANCE_FUTURES", "BTC-USDT-PERP")
SPOT = ("BINANCE", "BTC-USDT")


class _NullReader:
    """A reader that must NOT be invoked — used by tests whose code path exits (gate/setup) before any
    read. Records calls so a test can assert the gate/short-circuit ran before any vendor touch."""

    def __init__(self):
        self.calls = []

    def __call__(self, feed, exchange, symbol, day_iso):
        self.calls.append((feed, exchange, symbol, day_iso))
        raise AssertionError("reader must not be called in this test")


# --------------------------------------------------------------------------- error taxonomy (pure)
def test_classify_error():
    assert dl.classify_error(dl.QuotaError("x")) == "quota"
    assert dl.classify_error(dl.AuthError("x")) == "auth"
    assert dl.classify_error(dl.TransientError("x")) == "transient"
    assert dl.classify_error(RuntimeError("QuotaExceeded: over cap")) == "quota"
    assert dl.classify_error(RuntimeError("SlowDown, please retry")) == "transient"
    assert dl.classify_error(RuntimeError("RequestTimeout")) == "transient"
    assert dl.classify_error(RuntimeError("503 Service Unavailable")) == "transient"
    assert dl.classify_error(ValueError("schema drift: unknown column")) == "fatal"


def test_classify_error_s3_500_is_transient():
    # botocore surfaces a retryable HTTP 500 as `An error occurred (500) ...: Internal Server Error`
    msg = "An error occurred (500) when calling the GetObject operation: Internal Server Error"
    assert dl.classify_error(RuntimeError(msg)) == "transient"
    assert dl.classify_error(RuntimeError("An error occurred (503) when calling ...")) == "transient"


def test_classify_error_pyarrow_s3_forms():
    # PyArrow's S3FileSystem reports underscore codes / `HTTP status NNN`, unlike botocore.
    assert dl.classify_error(OSError("AWS Error ACCESS_DENIED during HeadObject: ...")) == "auth"
    assert dl.classify_error(OSError("AWS Error INVALID_ACCESS_KEY_ID ...")) == "auth"
    assert dl.classify_error(OSError("AWS Error SIGNATURE_DOES_NOT_MATCH ...")) == "auth"
    assert dl.classify_error(OSError("AWS Error UNKNOWN (HTTP status 503) during ListObjects")) \
        == "transient"
    assert dl.classify_error(OSError("AWS Error UNKNOWN (HTTP status 500) ...")) == "transient"
    assert dl.classify_error(OSError("AWS Error SLOW_DOWN during GetObject: ...")) == "transient"
    assert dl.classify_error(OSError("AWS Error SERVICE_UNAVAILABLE ...")) == "transient"


def test_classify_error_markers_are_not_over_broad():
    assert dl.classify_error(ValueError("expected 500 columns, got 12")) == "fatal"
    assert dl.classify_error(ValueError("malformed row 5040")) == "fatal"
    assert dl.classify_error(ValueError("byte offset 502341 invalid")) == "fatal"   # not (502)
    assert dl.classify_error(RuntimeError("Please reduce your request rate")) == "transient"


def test_classify_error_auth_failures_are_hard_stops():
    assert dl.classify_error(RuntimeError("AccessDenied: not authorized")) == "auth"
    assert dl.classify_error(RuntimeError("InvalidAccessKeyId")) == "auth"
    assert dl.classify_error(RuntimeError("SignatureDoesNotMatch")) == "auth"
    assert dl.classify_error(RuntimeError("The security token has expired")) == "auth"
    assert dl.classify_error(RuntimeError("403 malformed row 4037")) == "fatal"   # bare 403 not auth
    assert issubclass(dl.AuthError, dl.HardStop) and issubclass(dl.QuotaError, dl.HardStop)


# --------------------------------------------------------------------------- pure helpers
def test_present_days_from_list_records_reads_dt_key():
    recs = [{"table": "trades", "exchange": "BINANCE", "symbol": "BTC-USDT",
             "dt": "2026-04-02", "filename": "b.parquet"},
            {"table": "trades", "exchange": "BINANCE", "symbol": "BTC-USDT",
             "dt": "2026-04-01", "filename": "a.parquet"}]
    assert dl.present_days_from_list_records(recs) == ["2026-04-01", "2026-04-02"]
    assert dl.present_days_from_list_records([{"no": "dt"}]) == []
    assert dl.present_days_from_list_records([]) == []
    assert dl.present_days_from_list_records(None) == []


def test_used_gb_from_response_reads_downloaded_gb():
    assert dl.used_gb_from_response({"downloaded_gb": 151.35, "timeframe_days": 31}) \
        == pytest.approx(151.35)
    assert dl.used_gb_from_response({"downloaded_gb": 0}) == 0.0
    with pytest.raises((KeyError, TypeError)):
        dl.used_gb_from_response(151.35)
    with pytest.raises(KeyError):
        dl.used_gb_from_response({"timeframe_days": 31})


def test_feed_miss_is_fatal_policy():
    assert dl.feed_miss_is_fatal("book_delta_v2") is True
    assert dl.feed_miss_is_fatal("trades") is True
    assert dl.feed_miss_is_fatal("funding") is True
    assert dl.feed_miss_is_fatal("open_interest") is True
    assert dl.feed_miss_is_fatal(lb.SEED_PRODUCT) is True                # `book` seed is required
    assert dl.feed_miss_is_fatal("liquidations") is False               # sparse, Risk Q2


def test_sparse_accepted_reads_manifest(tmp_path):
    root = str(tmp_path)
    lb.manifest_append(root, {"feed": "liquidations", "exchange": PERP[0], "symbol": PERP[1],
                              "dt": "2026-04-01", "status": "missing", "sparse_ok": True})
    lb.manifest_append(root, {"feed": "trades", "exchange": PERP[0], "symbol": PERP[1],
                              "dt": "2026-04-01", "status": "missing", "sparse_ok": False})
    acc = dl.sparse_accepted(root)
    assert ("liquidations", *PERP, "2026-04-01") in acc
    assert ("trades", *PERP, "2026-04-01") not in acc
    lb.manifest_append(root, {"feed": "liquidations", "exchange": PERP[0], "symbol": PERP[1],
                              "dt": "2026-04-01", "status": "ok", "rows": 5})
    assert ("liquidations", *PERP, "2026-04-01") not in dl.sparse_accepted(root)


def test_backoff_seconds_grows_and_is_capped():
    grow = [dl._backoff_seconds(a, 1.0, 60.0, lambda: 1.0) for a in range(1, 12)]
    assert grow[:3] == [1.0, 2.0, 4.0]
    assert all(s <= 60.0 for s in grow)
    assert dl._backoff_seconds(1, 1.0, 60.0, lambda: 0.0) == 0.5


def test_validate_download_jobs_rejects_bool_and_out_of_bounds():
    for value in (True, False, None, 1.0, 0, -1, dl.MAX_DOWNLOAD_JOBS + 1):
        with pytest.raises(ValueError, match="jobs"):
            dl.validate_download_jobs(value)
    assert dl.validate_download_jobs(1) == 1
    assert dl.validate_download_jobs(dl.MAX_DOWNLOAD_JOBS) == 4


def test_runtime_projection_is_caveated_arithmetic_not_live_guarantee():
    days = dl.daterange("2025-11-01", "2025-12-31")
    units = dl.plan_units(["binance-perp"], "book_delta_v2,trades", days)
    projection = dl.runtime_projection(units, 4)
    assert projection["available"] is True
    assert projection["basis"]["exchange"] == "BINANCE_FUTURES"
    assert projection["basis"]["symbol"] == "BTC-USDT-PERP"
    assert projection["basis"]["seconds_by_feed"] == {
        "book_delta_v2": 1441.344,
        "book": 242.671,
        "trades": 24.936,
    }
    assert projection["basis"]["minutes_per_three_product_day"] == 28.483
    assert projection["serial_reference_seconds"] == pytest.approx(104246.011)
    assert projection["idealized_jobs_floor_seconds"] == pytest.approx(26061.503)
    assert "not a bound" in projection["caveat"]
    assert "does not guarantee 4x live scaling" in projection["caveat"]


def test_runtime_projection_refuses_unmeasured_spot_scope():
    units = dl.plan_units(["binance-spot"], "trades", ["2026-04-01"])
    projection = dl.runtime_projection(units, 1)
    assert projection["available"] is False
    assert projection["unmeasured_units"] == ["BINANCE/BTC-USDT/trades"]
    no_op = dl.runtime_projection([], 1, scope_units=units)
    assert no_op["available"] is False


@pytest.mark.parametrize("jobs", ["0", "-1", "5"])
def test_run_rejects_unsafe_jobs_before_any_vendor_or_quota_call(tmp_path, jobs):
    reader = _NullReader()

    def used_data():
        raise AssertionError("used_data must not be called for invalid --jobs")

    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--jobs", jobs, "--out", str(tmp_path / "raw"),
                    "--report-dir", str(tmp_path / "rep")],
                   reader=reader, used_data_fn=used_data)
    assert code == 2
    assert reader.calls == []


# --------------------------------------------------------------------------- CLI exit-code contract
def test_run_reversed_date_range_exits_2(tmp_path):
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--start", "2026-04-02", "--end", "2026-04-01", "--out", str(tmp_path / "raw"),
                    "--report-dir", str(tmp_path / "rep")],
                   reader=_NullReader(), used_data_fn=lambda: 0.0)
    assert code == 2


def test_run_no_day_source_exits_2(tmp_path):
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_NullReader(), used_data_fn=lambda: 0.0)
    assert code == 2


def test_run_invalid_instrument_feed_pair_exits_2(tmp_path):
    code = dl.main(["--instrument", "binance-spot", "--feeds", "funding",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_NullReader(), used_data_fn=lambda: 0.0)
    assert code == 2                                       # funding is perp-only


def test_run_unreadable_used_data_fails_safe_exit_2(tmp_path):
    def _boom():
        raise RuntimeError("used_data unreadable")
    code = dl.main(["--instrument", "binance-perp", "--feeds", "trades",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_NullReader(), used_data_fn=_boom, sleep=lambda *_: None)
    assert code == 2


def test_run_live_setup_failure_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "lake_session",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no keys")))
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   sleep=lambda *_: None)               # no reader/used_data → forces live session
    assert code == 2


def test_run_empty_feeds_exits_2(tmp_path):
    # `--feeds ,` (a wrapper expanding to only separators) must be rejected, not silently no-op to 0.
    code = dl.main(["--instrument", "binance-spot", "--feeds", ",",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_NullReader(), used_data_fn=lambda: 0.0)
    assert code == 2


def test_run_explicitly_empty_feeds_string_exits_2(tmp_path):
    # `--feeds ""` (an unset env var) is NOT the same as omitting --feeds: it must exit 2, not expand
    # to every feed and start unexpected Lake reads.
    code = dl.main(["--instrument", "binance-spot", "--feeds", "",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_NullReader(), used_data_fn=lambda: 0.0)
    assert code == 2
    # sanity: OMITTING --feeds still defaults to all valid feeds (None, not "")
    assert dl.resolve_feeds("binance-spot", None) == list(lb.INSTRUMENTS["binance-spot"].feeds)


def test_run_empty_instrument_exits_2(tmp_path):
    code = dl.main(["--instrument", " , ", "--feeds", "trades",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   reader=_NullReader(), used_data_fn=lambda: 0.0)
    assert code == 2


def test_plan_units_dedups_repeated_instruments_and_feeds():
    # repeated instruments/feeds must NOT emit identical (feed,E,S,dt) units — they would race on the
    # same data.parquet path under --jobs>1 and corrupt the partition.
    a = dl.plan_units(["binance-perp", "binance-perp"], "trades", ["2026-04-01"])
    assert len(a) == 1
    b = dl.plan_units(["binance-perp"], "trades,trades", ["2026-04-01"])
    assert len(b) == 1
    # book_delta_v2 (+book seed) repeated → 2 unique units (book_delta_v2, book), not 4
    c = dl.plan_units(["binance-perp"], "book_delta_v2,book_delta_v2", ["2026-04-01"])
    assert sorted(u.feed for u in c) == ["book", "book_delta_v2"]


def test_lake_bucket_missing_raises_normal_exception(monkeypatch):
    # if lakeapi's default_bucket lookup is missing/changed, _lake_bucket must raise a NORMAL
    # exception (not SystemExit, which is BaseException and would bypass main's `except Exception`
    # setup guard → exit 1). pytest.raises(RuntimeError) would not catch a SystemExit, so it proves it.
    fake = types.ModuleType("lakeapi")
    fake.load_data = types.SimpleNamespace()
    fake.load_data.__globals__ = {}                       # no 'default_bucket'
    monkeypatch.setitem(sys.modules, "lakeapi", fake)
    with pytest.raises(RuntimeError):
        dl._lake_bucket()


def test_run_live_reader_setup_failure_exits_2(tmp_path, monkeypatch):
    # building the live reader (imports pyarrow/lakeapi, resolves the bucket) can fail on an
    # incomplete `.[lake]` install — that must return the documented setup exit 2, not a traceback.
    monkeypatch.setattr(dl, "lake_session", lambda *a, **k: object())     # dummy session
    monkeypatch.setattr(dl, "_live_reader",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("no pyarrow")))
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   used_data_fn=lambda: 0.0, sleep=lambda *_: None)     # no reader → builds _live_reader
    assert code == 2


def test_run_dry_run_lister_setup_failure_exits_2(tmp_path, monkeypatch):
    monkeypatch.setattr(dl, "lake_session", lambda *a, **k: object())
    monkeypatch.setattr(dl, "_live_lister",
                        lambda *a, **k: (_ for _ in ()).throw(ImportError("no lakeapi")))
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades", "--dry-run",
                    "--start", "2026-04-01", "--end", "2026-04-01",
                    "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                   used_data_fn=lambda: 0.0)                            # no lister → builds _live_lister
    assert code == 2


def test_run_broad_gate_blocks_before_any_read(tmp_path):
    reader = _NullReader()
    with pytest.raises(SystemExit) as e:
        dl.main(["--instrument", "binance-perp", "--start", "2026-01-01", "--end", "2026-12-31",
                 "--out", str(tmp_path / "raw"), "--report-dir", str(tmp_path / "rep")],
                reader=reader, used_data_fn=lambda: 0.0, sleep=lambda *_: None)
    assert e.value.code == lb.BROAD_GATE_EXIT == 4
    assert reader.calls == []                              # gate runs BEFORE any vendor read


def test_run_allow_broad_still_blocked_over_quota_headroom(tmp_path):
    with pytest.raises(SystemExit) as e:
        dl.main(["--instrument", "binance-perp,binance-spot", "--start", "2026-01-01",
                 "--end", "2026-12-31", "--allow-broad", "--out", str(tmp_path / "raw"),
                 "--report-dir", str(tmp_path / "rep")],
                reader=_NullReader(), used_data_fn=lambda: 295.0, sleep=lambda *_: None)
    assert e.value.code == 4


# --------------------------------------------------------------------------- dry-run (fake lister)
def test_dry_run_uses_lister_and_writes_no_parquet(tmp_path):
    seen = []

    def fake_lister(feed, exchange, symbol, start, end):
        # list_data must be BOUNDED to the requested window (one day → one-day probe, not full history)
        seen.append((feed, start.date().isoformat(), end.date().isoformat()))
        return ["2026-04-01"]

    raw = tmp_path / "raw"
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades", "--dry-run",
                    "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(raw),
                    "--report-dir", str(tmp_path / "rep")],
                   lister=fake_lister, used_data_fn=lambda: 0.0)
    assert code == 0
    assert seen == [("trades", "2026-04-01", "2026-04-02")]   # bounded to [first, last+1) exclusive
    assert not any(raw.rglob("*.parquet"))                # dry-run transfers zero parquet
    rep = json.loads(next((tmp_path / "rep").glob("*.json")).read_text())
    assert rep["dry_run"] is True and rep["transferred_gb"] == 0
    assert rep["presence"]["binance-spot:trades"]["n_present"] == 1


def test_dry_run_auth_error_exits_2(tmp_path):
    def bad_lister(feed, exchange, symbol, start, end):
        raise RuntimeError("AccessDenied: not authorized for this bucket")
    code = dl.main(["--instrument", "binance-spot", "--feeds", "trades", "--dry-run",
                    "--start", "2026-04-01", "--end", "2026-04-01", "--out", str(tmp_path / "raw"),
                    "--report-dir", str(tmp_path / "rep")],
                   lister=bad_lister, used_data_fn=lambda: 0.0)
    assert code == 2


# --------------------------------------------------------------------------- packaging + import safety
def test_pyproject_declares_lake_downloader_extra():
    data = tomllib.loads((_ROOT / "pyproject.toml").read_text())
    lake = data["project"]["optional-dependencies"]["lake"]
    for dep in ("pyarrow", "lakeapi", "boto3"):
        assert any(dep in d for d in lake), f"{dep} missing from the `lake` extra"


def test_import_is_vendor_safe():
    # importing the module must NOT pull in boto3/lakeapi, AND must succeed with pyarrow BLOCKED
    # (every pyarrow touch is lazy) — the likeliest Requirement-1 regression.
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
