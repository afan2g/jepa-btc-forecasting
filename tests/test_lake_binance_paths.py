"""Offline tests: Binance Lake feed/instrument registry + Hive partition paths + CI import safety.

Pure — no vendor I/O, no network. The registry is the single source of truth every downstream
path, manifest key, and quota estimate derives from (docs/superpowers/plans/2026-07-02-binance-
downloader-plan.md, Requirement 1/2). Also pins the CI-safety contract: ingest/lake_binance.py must
import with NO boto3/lakeapi/pyarrow at module top so unit tests run without the downloader's vendor
deps (mirrors the ingest/_common.py split; plan Review Checklist)."""
import ast
import pathlib
import subprocess
import sys

import pytest

from ingest import lake_binance as lb

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_SRC = REPO_ROOT / "ingest" / "lake_binance.py"


# --------------------------------------------------------------------------- registry
def test_instruments_match_verified_identifiers():
    assert lb.INSTRUMENTS["binance-perp"].exchange == "BINANCE_FUTURES"
    assert lb.INSTRUMENTS["binance-perp"].symbol == "BTC-USDT-PERP"
    assert lb.INSTRUMENTS["binance-spot"].exchange == "BINANCE"
    assert lb.INSTRUMENTS["binance-spot"].symbol == "BTC-USDT"
    assert lb.INSTRUMENTS["binance-spot"].feeds == ("book_delta_v2", "trades")


def test_perp_feeds_cover_all_scoped_products():
    assert lb.INSTRUMENTS["binance-perp"].feeds == (
        "book_delta_v2", "trades", "funding", "open_interest", "liquidations")


def test_seed_product_is_book_and_never_an_output_feed():
    # the `book` 20-level snapshot is the seed INPUT for book_delta_v2 recon, never an emitted feed
    assert lb.SEED_PRODUCT == "book"
    assert lb.SEED_PRODUCT not in lb.FEEDS


def test_feed_kind_classifies_every_feed():
    assert set(lb.FEED_KIND) == set(lb.FEEDS)
    assert lb.FEED_KIND["book_delta_v2"] == "delta"
    assert lb.FEED_KIND["liquidations"] == "events"


# --------------------------------------------------------------------------- paths
def test_raw_parquet_path_is_hive_partitioned():
    p = lb.raw_parquet_path("data/raw/lake", "book_delta_v2",
                            "BINANCE_FUTURES", "BTC-USDT-PERP", "2026-04-01")
    assert p == ("data/raw/lake/book_delta_v2/exchange=BINANCE_FUTURES/"
                 "symbol=BTC-USDT-PERP/dt=2026-04-01/data.parquet")


def test_seed_book_product_has_its_own_raw_partition():
    p = lb.raw_parquet_path("data/raw/lake", lb.SEED_PRODUCT,
                            "BINANCE_FUTURES", "BTC-USDT-PERP", "2026-04-01")
    assert p == ("data/raw/lake/book/exchange=BINANCE_FUTURES/"
                 "symbol=BTC-USDT-PERP/dt=2026-04-01/data.parquet")


def test_processed_path_uses_output_name_not_lake_feed():
    p = lb.processed_parquet_path("data/processed", "topk_l2",
                                  "BINANCE_FUTURES", "BTC-USDT-PERP", "2026-04-01")
    assert p == ("data/processed/topk_l2/exchange=BINANCE_FUTURES/"
                 "symbol=BTC-USDT-PERP/dt=2026-04-01/data.parquet")


def test_raw_partition_dir_has_no_filename():
    d = lb.raw_partition_dir("data/raw/lake", "trades", "BINANCE", "BTC-USDT", "2026-04-01")
    assert d == "data/raw/lake/trades/exchange=BINANCE/symbol=BTC-USDT/dt=2026-04-01"


def test_paths_are_deterministic():
    a = lb.raw_parquet_path("data/raw/lake", "trades", "BINANCE", "BTC-USDT", "2026-04-01")
    b = lb.raw_parquet_path("data/raw/lake", "trades", "BINANCE", "BTC-USDT", "2026-04-01")
    assert a == b


# --------------------------------------------------------------------------- feed validation
def test_invalid_instrument_feed_pair_rejected():
    with pytest.raises(ValueError):
        lb.validate_feed("binance-spot", "funding")   # funding is perp-only


def test_valid_instrument_feed_pairs_accepted():
    lb.validate_feed("binance-perp", "funding")       # no raise
    lb.validate_feed("binance-spot", "book_delta_v2")


def test_unknown_instrument_key_rejected():
    with pytest.raises(KeyError):
        lb.validate_feed("binance-margin", "trades")


# --------------------------------------------------------------------------- CI import safety
_VENDOR_ROOTS = {"boto3", "botocore", "lakeapi", "pyarrow", "s3fs"}


def _top_level_imported_roots(src_path: pathlib.Path) -> set[str]:
    """Root module names of MODULE-LEVEL imports only (not function/class-local)."""
    tree = ast.parse(src_path.read_text())
    roots: set[str] = set()
    for node in tree.body:  # only top-level statements — a lazy vendor import inside a fn is allowed
        if isinstance(node, ast.Import):
            roots.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            roots.add(node.module.split(".")[0])
    return roots


def test_no_vendor_import_at_module_top():
    roots = _top_level_imported_roots(MODULE_SRC)
    offenders = sorted(roots & _VENDOR_ROOTS)
    assert not offenders, f"vendor import at module top of lake_binance.py: {offenders}"


def test_import_pulls_no_network_client():
    # Fresh interpreter: importing the module must not load the S3/Lake network clients as a side
    # effect. pyarrow is intentionally NOT asserted here — pandas pulls it transitively; the AST
    # test above proves lake_binance itself never imports it at module top.
    code = ("import sys, ingest.lake_binance; "
            "bad=[m for m in ('boto3','lakeapi') if m in sys.modules]; "
            "assert not bad, 'network client imported: %r' % bad")
    r = subprocess.run([sys.executable, "-c", code], cwd=str(REPO_ROOT),
                       capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
