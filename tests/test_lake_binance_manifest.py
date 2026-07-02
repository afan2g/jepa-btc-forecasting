"""Offline tests: Binance raw-store manifest append/read + resume/idempotency state (plan Task 2).

Pure, tmp_path only — no vendor I/O. Mirrors download_coinapi.py's atomic append + tmp cleanup. The
manifest is the resume ledger: one JSON line per written (feed|book, exchange, symbol, dt), and a
partition is 'done' iff its FINAL data.parquet exists (a leftover .tmp never counts)."""
import json
import pathlib

from ingest import lake_binance as lb

_PART = ("BINANCE_FUTURES", "BTC-USDT-PERP")


def _rec(feed, day, status="ok", **extra):
    return {"feed": feed, "exchange": _PART[0], "symbol": _PART[1],
            "dt": day, "status": status, **extra}


def test_manifest_append_writes_one_json_line_per_record(tmp_path):
    root = str(tmp_path)
    lb.manifest_append(root, _rec("book_delta_v2", "2026-04-01"))
    lb.manifest_append(root, _rec("trades", "2026-04-01"))
    lines = (tmp_path / "_manifest.jsonl").read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0])["feed"] == "book_delta_v2"
    assert json.loads(lines[1])["feed"] == "trades"


def test_manifest_append_is_append_only(tmp_path):
    root = str(tmp_path)
    lb.manifest_append(root, _rec("book_delta_v2", "2026-04-01"))
    first = (tmp_path / "_manifest.jsonl").read_text()
    lb.manifest_append(root, _rec("book_delta_v2", "2026-04-02"))
    assert (tmp_path / "_manifest.jsonl").read_text().startswith(first)  # earlier line untouched


def test_manifest_index_keys_on_feed_exchange_symbol_dt(tmp_path):
    root = str(tmp_path)
    lb.manifest_append(root, _rec("book_delta_v2", "2026-04-01", status="ok"))
    lb.manifest_append(root, _rec("trades", "2026-04-01", status="missing"))
    idx = lb.manifest_index(root)
    assert idx[("book_delta_v2", *_PART, "2026-04-01")] == "ok"
    assert idx[("trades", *_PART, "2026-04-01")] == "missing"


def test_manifest_index_last_record_wins(tmp_path):
    # a --resume run that promotes an errored unit to ok must supersede the earlier record
    root = str(tmp_path)
    lb.manifest_append(root, _rec("book_delta_v2", "2026-04-01", status="error"))
    lb.manifest_append(root, _rec("book_delta_v2", "2026-04-01", status="ok"))
    assert lb.manifest_index(root)[("book_delta_v2", *_PART, "2026-04-01")] == "ok"


def test_manifest_index_empty_when_no_manifest(tmp_path):
    assert lb.manifest_index(str(tmp_path)) == {}


def test_manifest_index_skips_blank_and_malformed_lines(tmp_path):
    root = str(tmp_path)
    lb.manifest_append(root, _rec("book_delta_v2", "2026-04-01"))
    with open(tmp_path / "_manifest.jsonl", "a") as f:
        f.write("\n")                       # blank
        f.write("{not json}\n")             # malformed
        f.write(json.dumps({"no": "keys"}) + "\n")  # missing key fields
    idx = lb.manifest_index(root)
    assert idx == {("book_delta_v2", *_PART, "2026-04-01"): "ok"}


def test_is_done_true_only_when_final_parquet_exists(tmp_path):
    root = str(tmp_path)
    args = ("book_delta_v2", *_PART, "2026-04-01")
    assert lb.is_done(root, *args) is False
    final = pathlib.Path(lb.raw_parquet_path(root, *args))
    final.parent.mkdir(parents=True, exist_ok=True)
    (final.parent / "data.parquet.tmp").write_text("partial")  # a leftover .tmp is NOT done
    assert lb.is_done(root, *args) is False
    final.write_text("final")
    assert lb.is_done(root, *args) is True


def test_cleanup_tmp_removes_only_stale_tmp(tmp_path):
    root = str(tmp_path)
    args = ("book_delta_v2", *_PART, "2026-04-01")
    final = pathlib.Path(lb.raw_parquet_path(root, *args))
    final.parent.mkdir(parents=True, exist_ok=True)
    final.write_text("final")
    tmp = final.parent / "data.parquet.tmp"
    tmp.write_text("partial")
    removed = lb.cleanup_tmp(root)
    assert removed == 1
    assert not tmp.exists()
    assert final.exists()  # the published parquet survives
