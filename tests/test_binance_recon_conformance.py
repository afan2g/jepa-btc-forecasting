"""Binance-shaped native-vs-Python conformance + engine-resolution gates (plan Task 7 Step 1;
issue #71).

Synthetic only, no vendor access. Two halves:

  * **Engine resolution** (always runs): the Binance tick scales ARE registered in
    `recon/native.py::_TICK_SCALE` — perp $0.10 tick -> scale 10, spot $0.01 tick -> scale 100 —
    measured by the #64 tick-scale step: ZERO off-tick prices across every price-bearing feed
    (`book_delta_v2`/`trades`/`book`) on Lake day 2026-04-01 (report
    `data/reports/binance_source_quality/tick_scale.json`, report_hash `d5025c58aa48…`, issue #71).
    So `--engine auto` selects native for exactly these pairs when the extension is importable and
    falls back to Python (with a note, never silently) when it is not; an explicit
    `--engine native` resolves to native when available and to an abort (the runner exits 2) when
    not. Unsupported pairs keep the documented fallback/abort contract. The extension-present and
    extension-absent branches are BOTH pinned deterministically (monkeypatching the module's `_rn`
    handle), so CI without the extension and a local build exercise the same assertions.

  * **Replay conformance** (skipped without `recon_native`): on Binance-shaped fixtures — prices
    on each instrument's measured tick grid, the registered scale passed explicitly — the native
    `(frame, meta)` must equal the Python oracle `reconstruct_lake_l2_at_samples_seeded` on the
    plan's boundary cases: valid seed, update/delete churn, stranded->reseed, same-ts
    delta/snapshot order, equal `(ts, seq)` source-row order, no valid seed, `frame_out=False`
    metrics-only, and one-tick equality/crossing boundaries. Every case runs for BOTH instruments
    so both registered scales are exercised. Also pins the Stage-2 runner's own dispatch
    (`recon_topk_day`) native == Python for both instruments.
"""
import importlib.util
import pathlib
import sys

import numpy as np
import pandas as pd
import pytest

from recon import native as rn
from recon.reseed import ReseedPolicy, book_snapshot, reconstruct_lake_l2_at_samples_seeded

native = pytest.mark.skipif(not rn.native_available(),
                            reason="recon_native extension not built (maturin develop)")

# scripts/ is not a package — load the Stage-2 runner by path (same pattern as test_quality_map).
_SPEC = importlib.util.spec_from_file_location(
    "run_binance_recon_conformance",
    pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_binance_recon.py")
rbr = importlib.util.module_from_spec(_SPEC)
sys.modules["run_binance_recon_conformance"] = rbr
_SPEC.loader.exec_module(rbr)

# The two measured Binance instruments (#64 tick-scale report; issue #71): perp $0.10 tick,
# spot $0.01 tick. SCALES pins the values the registry must carry; conformance fixtures build
# prices on the instrument's tick grid via TICKS.
PERP = ("BINANCE_FUTURES", "BTC-USDT-PERP")
SPOT = ("BINANCE", "BTC-USDT")
SCALES = {PERP: 10, SPOT: 100}
TICKS = {PERP: 0.1, SPOT: 0.01}
BOTH = pytest.mark.parametrize("inst", [PERP, SPOT], ids=["perp-scale10", "spot-scale100"])
NOW = ReseedPolicy(reseed_after_crossed_s=0.0, min_levels_per_side=1)


def _lake_df(rows):
    """Binance-shaped book_delta_v2 frame from (ts_ns, seq, is_bid, price, size) tuples."""
    df = pd.DataFrame(rows, columns=["origin_time", "sequence_number", "side_is_bid",
                                     "price", "size"])
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    return df


def _assert_conforms(df, grid, *, inst, k, snapshots=None, policy=NOW, frame_out=True):
    py_frame, py_meta = reconstruct_lake_l2_at_samples_seeded(
        df, grid, k=k, engine_time_col="origin_time", snapshots=snapshots, policy=policy,
        frame_out=frame_out)
    nat_frame, nat_meta = rn.reconstruct_lake_l2_at_samples_seeded_native(
        df, grid, k=k, engine_time_col="origin_time", snapshots=snapshots, policy=policy,
        frame_out=frame_out, price_scale=SCALES[inst])
    assert nat_meta == py_meta, f"meta mismatch\nnative={nat_meta}\npython={py_meta}"
    if frame_out:
        pd.testing.assert_frame_equal(nat_frame, py_frame, check_dtype=True)
    else:
        assert nat_frame is None and py_frame is None
    return nat_frame, nat_meta


# --------------------------------------------------------------------------- tick-scale registry
def test_binance_tick_scales_are_registered():
    # The #64-measured scales, exactly these two pairs (issue #71 acceptance criteria).
    assert rn.tick_scale_for(*PERP) == 10
    assert rn.tick_scale_for(*SPOT) == 100


def test_registry_contains_exactly_the_verified_pairs():
    # "Add ONLY those two pairs" (issue #71): pin the registry's exact contents AND scales. An
    # unintended registration would silently enable native mode for an unverified instrument
    # under --engine auto, defeating the fail-closed verified-tick contract — any future
    # addition must consciously update this test alongside its measurement evidence.
    assert rn._TICK_SCALE == {
        ("COINBASE", "BTC-USD"): 100,
        ("BINANCE_FUTURES", "BTC-USDT-PERP"): 10,
        ("BINANCE", "BTC-USDT"): 100,
    }


def test_tick_scale_lookup_is_case_insensitive():
    assert rn.tick_scale_for("binance_futures", "btc-usdt-perp") == 10
    assert rn.tick_scale_for("Binance_Futures", "Btc-Usdt-Perp") == 10
    assert rn.tick_scale_for("binance", "btc-usdt") == 100
    assert rn.tick_scale_for("Binance", "Btc-Usdt") == 100


def test_unsupported_binance_pairs_stay_unregistered():
    # Only the two MEASURED instruments are eligible — a near-miss symbol/exchange has no scale.
    assert rn.tick_scale_for("BINANCE", "ETH-USDT") is None
    assert rn.tick_scale_for("BINANCE_FUTURES", "ETH-USDT-PERP") is None
    assert rn.tick_scale_for("BINANCE", "BTC-USDT-PERP") is None      # perp symbol on spot venue
    assert rn.tick_scale_for("BINANCE_FUTURES", "BTC-USDT") is None   # spot symbol on perp venue


# --------------------------------------------------------------------------- engine resolution
@BOTH
def test_auto_selects_native_when_extension_present(monkeypatch, inst):
    # resolve_engine consults availability only through the module's `_rn` handle — pin it non-None
    # so this branch is deterministic even in CI without the built extension.
    monkeypatch.setattr(rn, "_rn", object())
    assert rn.resolve_engine("auto", exchange=inst[0], symbol=inst[1]) \
        == ("native", SCALES[inst], None)


@BOTH
def test_auto_falls_back_to_python_when_extension_absent(monkeypatch, inst):
    monkeypatch.setattr(rn, "_rn", None)
    eng, scale, note = rn.resolve_engine("auto", exchange=inst[0], symbol=inst[1])
    assert eng == "python" and scale is None
    assert note is not None                       # the fallback is announced, never silent


@BOTH
def test_explicit_native_resolves_when_extension_present(monkeypatch, inst):
    monkeypatch.setattr(rn, "_rn", object())
    assert rn.resolve_engine("native", exchange=inst[0], symbol=inst[1]) \
        == ("native", SCALES[inst], None)


@BOTH
def test_explicit_native_resolves_to_abort_when_extension_absent(monkeypatch, inst):
    # resolve_engine returns the python fallback + a reason; the CALLER must abort on an explicit
    # native request (the runner exits 2 — covered e2e in tests/test_run_binance_recon.py).
    monkeypatch.setattr(rn, "_rn", None)
    monkeypatch.setattr(rn, "_IMPORT_ERROR", ImportError("not built"))
    eng, scale, note = rn.resolve_engine("native", exchange=inst[0], symbol=inst[1])
    assert eng == "python" and scale is None and note is not None


def test_unsupported_pair_behavior_unchanged_even_with_extension(monkeypatch):
    # A pair without a verified scale must keep the pre-#71 contract regardless of availability:
    # auto -> announced Python fallback; explicit native -> abort path (python + reason).
    monkeypatch.setattr(rn, "_rn", object())
    eng, scale, note = rn.resolve_engine("auto", exchange="BINANCE", symbol="ETH-USDT")
    assert eng == "python" and scale is None and note is not None
    eng, scale, note = rn.resolve_engine("native", exchange="BINANCE", symbol="ETH-USDT")
    assert eng == "python" and scale is None and note is not None


def test_explicit_python_is_always_python():
    # The Python oracle stays reachable unconditionally (issue #71: correctness oracle).
    for inst in (PERP, SPOT):
        assert rn.resolve_engine("python", exchange=inst[0], symbol=inst[1]) \
            == ("python", None, None)


# --------------------------------------------------------------------------- valid seed
@native
@BOTH
def test_valid_seed_conformance_on_tick_grid(inst):
    t = TICKS[inst]
    df = _lake_df([(100, 1, True, 100.0, 1.0), (150, 2, False, 100.0 + t, 2.0),
                   (200, 3, True, 100.0, 3.0)])
    seed = [book_snapshot(0, bids=[(100.0, 2.0)], asks=[(100.0 + t, 3.0)])]
    _, m = _assert_conforms(df, [50, 120, 180, 250], inst=inst, k=2, snapshots=seed)
    assert m["seed_accepted"] is True and m["seed_reason"] == "ok"


# --------------------------------------------------------------------------- update/delete churn
@native
@BOTH
def test_update_delete_churn_conformance(inst):
    # Randomized on-tick-grid stream: absolute-size updates, size-0 deletes, duplicate ts/seq —
    # native and Python must agree on the full frame and every metric at this instrument's scale.
    scale, t = SCALES[inst], TICKS[inst]
    rng = np.random.default_rng(11)
    n = 400
    ts = np.sort(rng.integers(1, 300, size=n))            # populated: >0
    seq = rng.integers(0, 50, size=n)
    is_bid = rng.integers(0, 2, size=n).astype(bool)
    price = rng.integers(99 * scale, 101 * scale, size=n) / scale   # on the instrument tick grid
    size = rng.choice([0.0, 0.0, 1.0, 2.0, 5.0], size=n)  # heavy churn: ~40% deletes
    df = _lake_df(list(zip(ts.tolist(), seq.tolist(), is_bid.tolist(),
                           price.tolist(), size.tolist())))
    snaps = [book_snapshot(0, bids=[(100.0 - 50 * t, 1.0), (100.0 - 60 * t, 2.0)],
                           asks=[(100.0 + 50 * t, 1.0), (100.0 + 60 * t, 2.0)]),
             book_snapshot(150, bids=[(100.0 - 20 * t, 1.0)], asks=[(100.0 + 20 * t, 1.0)])]
    _assert_conforms(df, list(range(0, 300, 5)), inst=inst, k=10, snapshots=snaps)


# --------------------------------------------------------------------------- stranded -> reseed
def _stranded_df(t):
    return _lake_df([
        (10, 1, True, 100.0 + 2 * t, 1.0),    # bid > ask by one tick => crossed (stranded ask)
        (100, 2, False, 100.0 + t, 0.0),      # stranded ask removed (delayed clear)
        (100, 3, False, 100.0 + 3 * t, 1.0),  # fresh ask posts => uncrossed again
    ])


@native
@BOTH
def test_reseed_repair_conformance(inst):
    t = TICKS[inst]
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(100.0 + t, 1.0)]),
             book_snapshot(30, bids=[(100.0 + 2 * t, 1.0)], asks=[(100.0 + 3 * t, 1.0)])]
    _, m = _assert_conforms(_stranded_df(t), [5, 20, 50, 150], inst=inst, k=1, snapshots=snaps)
    assert m["reseed_count"] == 1 and m["reseed_ts"] == [30]
    assert m["crossed_samples"] == 1


# --------------------------------------------------------------------------- same-ts ordering
@native
@BOTH
def test_same_timestamp_delta_before_snapshot(inst):
    # Delta and snapshot share ts=10: the delta posts a crossing bid, then the same-ts snapshot is
    # authoritative and overwrites it — the sample at 10 must be uncrossed on BOTH engines.
    t = TICKS[inst]
    df = _lake_df([(10, 1, True, 100.0 + 5 * t, 1.0)])
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(100.0 + t, 1.0)]),
             book_snapshot(10, bids=[(100.0 + t, 1.0)], asks=[(100.0 + 2 * t, 1.0)])]
    frame, m = _assert_conforms(df, [10, 20], inst=inst, k=1, snapshots=snaps)
    assert m["crossed_samples"] == 0
    assert frame["bid_0_price"].tolist() == [100.0 + t, 100.0 + t]


# --------------------------------------------------------------------------- stable (ts,seq) order
@native
@BOTH
def test_equal_ts_seq_rows_keep_source_order(inst):
    # Two absolute-size updates to the SAME (ts, seq, side, price): the final size is
    # order-dependent, and np.lexsort is stable, so source order (5.0 then 9.0) must win => 9.0.
    # A native sort that reordered equal (ts, seq) rows would land on 5.0.
    t = TICKS[inst]
    df = _lake_df([
        (2, 1, True, 100.0, 1.0), (2, 1, False, 100.0 + t, 1.0),
        (5, 7, True, 100.0, 5.0), (5, 7, True, 100.0, 9.0),
    ])
    frame, _ = _assert_conforms(df, [10], inst=inst, k=1)
    assert frame.iloc[0]["bid_0_size"] == 9.0


@native
@BOTH
def test_equal_ts_seq_long_run_keeps_source_order(inst):
    # The 4-row case above cannot discriminate an unstable native sort (Rust's small-slice
    # insertion sort is de-facto stable below ~20 elements). Here a 60-update equal-(ts, seq,
    # side, price) run is INTERLEAVED with 60 distinct-key rows, so an unstable sort gathering
    # the equal keys across partitions has real reorder opportunity; the source-LAST absolute
    # size (60.0) must win on both engines.
    t = TICKS[inst]
    rows = []
    for i in range(60):
        rows.append((5, 7, True, 100.0, float(i + 1)))                        # equal-key run
        rows.append((3 + (i % 5), i, False, 100.0 + (1 + i % 3) * t, 1.0))    # distinct keys
    frame, _ = _assert_conforms(_lake_df(rows), [10], inst=inst, k=1)
    assert frame.iloc[0]["bid_0_size"] == 60.0


# --------------------------------------------------------------------------- no valid seed
@native
@BOTH
def test_no_valid_seed_conformance(inst):
    t = TICKS[inst]
    snaps = [book_snapshot(0, bids=[(100.0 + 3 * t, 1.0)],
                           asks=[(100.0 + 2 * t, 1.0)])]   # crossed => rejected
    _, m = _assert_conforms(_lake_df([(10, 1, True, 100.0, 1.0)]), [5, 20], inst=inst, k=1,
                            snapshots=snaps)
    assert m["seed_accepted"] is False and m["seed_reason"] == "crossed"


# --------------------------------------------------------------------------- metrics-only
@native
@BOTH
def test_frame_out_false_metrics_conformance(inst):
    t = TICKS[inst]
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(100.0 + t, 1.0)])]
    _assert_conforms(_stranded_df(t), [5, 20, 50, 150], inst=inst, k=1, snapshots=snaps,
                     frame_out=False)


# --------------------------------------------------------------------------- tick boundaries
@native
@BOTH
def test_one_tick_equality_and_crossing_boundaries(inst):
    # bid == ask (equal price, opposite sides) must count crossed on both engines; one tick apart
    # must not — the float-vs-integer-tick comparison boundary the verified-scale contract exists
    # for (docs/native-recon.md "Verified tick contract").
    t = TICKS[inst]
    df = _lake_df([(10, 1, True, 100.0 + t, 1.0),     # bid == ask at 100+t => crossed
                   (30, 2, True, 100.0 + t, 0.0),     # remove it
                   (30, 3, True, 100.0, 1.0)])        # bid one tick below ask => uncrossed
    seed = [book_snapshot(0, bids=[(100.0 - t, 1.0)], asks=[(100.0 + t, 1.0)])]
    _, m = _assert_conforms(df, [20, 40], inst=inst, k=1, snapshots=seed)
    assert m["crossed_samples"] == 1


# --------------------------------------------------------------------------- runner dispatch
@native
@BOTH
def test_runner_dispatch_native_equals_python(inst, tmp_path):
    # recon_topk_day (the Stage-2 per-day pipeline) must yield identical certified frames and
    # classifications under engine="native" and engine="python" on the same inputs, at each
    # instrument's registered scale.
    import datetime as dt
    t = TICKS[inst]
    day = dt.date(2026, 4, 1)
    day_ns = int(pd.Timestamp("2026-04-01").value)
    ns = 1_000_000_000
    deltas = _lake_df([(day_ns + 1 * ns, 1, True, 100.0, 2.0),
                       (day_ns + 2 * ns, 2, False, 100.0 + t, 2.0)])
    book = pd.DataFrame([{
        "origin_time": day_ns, "received_time": day_ns,
        "bid_0_price": 100.0, "bid_0_size": 1.0,
        "bid_1_price": 100.0 - t, "bid_1_size": 2.0,
        "ask_0_price": 100.0 + t, "ask_0_size": 1.0,
        "ask_1_price": 100.0 + 2 * t, "ask_1_size": 2.0}])
    kw = dict(day=day, k=2, grid_ms=3_600_000, policy=ReseedPolicy(min_levels_per_side=1),
              book_stride_ms=1000, thresholds=rbr.Thresholds())
    py_frame, py_info = rbr.recon_topk_day(deltas, book, engine="python", price_scale=None, **kw)
    nat_frame, nat_info = rbr.recon_topk_day(deltas, book, engine="native",
                                             price_scale=SCALES[inst], **kw)
    assert py_info["classification"] == nat_info["classification"] == rbr.CERTIFIED
    assert nat_info["seed"] == py_info["seed"]
    assert nat_info["quality"] == py_info["quality"]
    pd.testing.assert_frame_equal(nat_frame, py_frame, check_dtype=True)
