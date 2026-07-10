"""Binance-shaped native-vs-Python conformance + engine-resolution gates (plan Task 7 Step 1).

Synthetic only, no vendor access. Two halves:

  * **Engine resolution** (always runs): NO Binance tick scale is registered in
    `recon/native.py::_TICK_SCALE` — the expected perp $0.10 / spot $0.01 ticks are UNVERIFIED
    (plan Risk Q1; live verification is blocked by #35). So `--engine auto` must fall back to
    Python with a note and an explicit `--engine native` must resolve to an abort (the runner
    exits 2), never a silent fallback. These tests assert the FALLBACK, not the scale; when Q1
    verification lands, registering the scale flips them deliberately.

  * **Replay conformance** (skipped without `recon_native`): on Binance-shaped fixtures — prices
    on the EXPECTED perp $0.10 grid, `price_scale=10` passed explicitly — the native
    `(frame, meta)` must equal the Python oracle `reconstruct_lake_l2_at_samples_seeded` on the
    plan's boundary cases: valid seed, stranded->reseed, same-ts delta/snapshot order, no valid
    seed, `frame_out=False`, and one-tick equality/crossing boundaries. Also pins the Stage-2
    runner's own dispatch (`recon_topk_day`) native == Python.
"""
import importlib.util
import pathlib
import sys

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

# EXPECTED Binance perp price scale ($0.10 tick -> 10) — passed explicitly to the native entry
# point for synthetic conformance; deliberately NOT registered in _TICK_SCALE until verified (Q1).
SCALE = 10
PERP = ("BINANCE_FUTURES", "BTC-USDT-PERP")
SPOT = ("BINANCE", "BTC-USDT")
NOW = ReseedPolicy(reseed_after_crossed_s=0.0, min_levels_per_side=1)


def _lake_df(rows):
    """Binance-shaped book_delta_v2 frame from (ts_ns, seq, is_bid, price, size) tuples."""
    df = pd.DataFrame(rows, columns=["origin_time", "sequence_number", "side_is_bid",
                                     "price", "size"])
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    return df


def _assert_conforms(df, grid, *, k, snapshots=None, policy=NOW, frame_out=True):
    py_frame, py_meta = reconstruct_lake_l2_at_samples_seeded(
        df, grid, k=k, engine_time_col="origin_time", snapshots=snapshots, policy=policy,
        frame_out=frame_out)
    nat_frame, nat_meta = rn.reconstruct_lake_l2_at_samples_seeded_native(
        df, grid, k=k, engine_time_col="origin_time", snapshots=snapshots, policy=policy,
        frame_out=frame_out, price_scale=SCALE)
    assert nat_meta == py_meta, f"meta mismatch\nnative={nat_meta}\npython={py_meta}"
    if frame_out:
        pd.testing.assert_frame_equal(nat_frame, py_frame, check_dtype=True)
    else:
        assert nat_frame is None and py_frame is None
    return nat_frame, nat_meta


# --------------------------------------------------------------------------- engine resolution
def test_no_binance_tick_scale_is_registered():
    # Plan Risk Q1: the expected ticks are UNVERIFIED and live evidence is blocked by #35 —
    # the registry must not carry them until measured on real data.
    assert rn.tick_scale_for(*PERP) is None
    assert rn.tick_scale_for(*SPOT) is None


@pytest.mark.parametrize("exchange,symbol", [PERP, SPOT])
def test_auto_engine_falls_back_to_python_for_binance(exchange, symbol):
    eng, scale, note = rn.resolve_engine("auto", exchange=exchange, symbol=symbol)
    assert eng == "python" and scale is None
    assert note is not None                       # the fallback is announced, never silent


@pytest.mark.parametrize("exchange,symbol", [PERP, SPOT])
def test_explicit_native_resolves_to_abort_for_binance(exchange, symbol):
    # resolve_engine returns the python fallback + a reason; the CALLER must abort on an explicit
    # native request (the runner exits 2 — covered e2e in tests/test_run_binance_recon.py).
    eng, scale, note = rn.resolve_engine("native", exchange=exchange, symbol=symbol)
    assert eng == "python" and scale is None and note is not None


# --------------------------------------------------------------------------- valid seed
@native
def test_valid_seed_conformance_on_perp_tick_grid():
    df = _lake_df([(100, 1, True, 100.0, 1.0), (150, 2, False, 100.1, 2.0),
                   (200, 3, True, 100.0, 3.0)])
    seed = [book_snapshot(0, bids=[(100.0, 2.0)], asks=[(100.1, 3.0)])]
    _, m = _assert_conforms(df, [50, 120, 180, 250], k=2, snapshots=seed)
    assert m["seed_accepted"] is True and m["seed_reason"] == "ok"


# --------------------------------------------------------------------------- stranded -> reseed
def _stranded_df():
    return _lake_df([
        (10, 1, True, 100.2, 1.0),    # bid 100.2 > ask 100.1 => crossed (stranded ask)
        (100, 2, False, 100.1, 0.0),  # ask 100.1 removed (delayed clear)
        (100, 3, False, 100.3, 1.0),  # ask 100.3 posts => uncrossed again
    ])


@native
def test_reseed_repair_conformance():
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(100.1, 1.0)]),
             book_snapshot(30, bids=[(100.2, 1.0)], asks=[(100.3, 1.0)])]
    _, m = _assert_conforms(_stranded_df(), [5, 20, 50, 150], k=1, snapshots=snaps)
    assert m["reseed_count"] == 1 and m["reseed_ts"] == [30]
    assert m["crossed_samples"] == 1


# --------------------------------------------------------------------------- same-ts ordering
@native
def test_same_timestamp_delta_before_snapshot():
    # Delta and snapshot share ts=10: the delta posts a crossing bid, then the same-ts snapshot is
    # authoritative and overwrites it — the sample at 10 must be uncrossed on BOTH engines.
    df = _lake_df([(10, 1, True, 100.5, 1.0)])
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(100.1, 1.0)]),
             book_snapshot(10, bids=[(100.1, 1.0)], asks=[(100.2, 1.0)])]
    frame, m = _assert_conforms(df, [10, 20], k=1, snapshots=snaps)
    assert m["crossed_samples"] == 0
    assert frame["bid_0_price"].tolist() == [100.1, 100.1]


# --------------------------------------------------------------------------- no valid seed
@native
def test_no_valid_seed_conformance():
    snaps = [book_snapshot(0, bids=[(100.3, 1.0)], asks=[(100.2, 1.0)])]   # crossed => rejected
    _, m = _assert_conforms(_lake_df([(10, 1, True, 100.0, 1.0)]), [5, 20], k=1, snapshots=snaps)
    assert m["seed_accepted"] is False and m["seed_reason"] == "crossed"


# --------------------------------------------------------------------------- metrics-only
@native
def test_frame_out_false_metrics_conformance():
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(100.1, 1.0)])]
    _assert_conforms(_stranded_df(), [5, 20, 50, 150], k=1, snapshots=snaps, frame_out=False)


# --------------------------------------------------------------------------- tick boundaries
@native
def test_one_tick_equality_and_crossing_boundaries():
    # bid == ask (equal price, opposite sides) must count crossed on both engines; one tick apart
    # must not — the float-vs-integer-tick comparison boundary the verified-scale contract exists
    # for (docs/native-recon.md "Verified tick contract").
    df = _lake_df([(10, 1, True, 100.1, 1.0),     # bid 100.1 == ask 100.1 => crossed
                   (30, 2, True, 100.1, 0.0),     # remove it
                   (30, 3, True, 100.0, 1.0)])    # bid 100.0 < ask 100.1 => one tick uncrossed
    seed = [book_snapshot(0, bids=[(99.9, 1.0)], asks=[(100.1, 1.0)])]
    _, m = _assert_conforms(df, [20, 40], k=1, snapshots=seed)
    assert m["crossed_samples"] == 1


# --------------------------------------------------------------------------- runner dispatch
@native
def test_runner_dispatch_native_equals_python(tmp_path):
    # recon_topk_day (the Stage-2 per-day pipeline) must yield identical certified frames and
    # classifications under engine="native" and engine="python" on the same inputs.
    import datetime as dt
    day = dt.date(2026, 4, 1)
    day_ns = int(pd.Timestamp("2026-04-01").value)
    ns = 1_000_000_000
    deltas = _lake_df([(day_ns + 1 * ns, 1, True, 100.0, 2.0),
                       (day_ns + 2 * ns, 2, False, 100.1, 2.0)])
    book = pd.DataFrame([{
        "origin_time": day_ns, "received_time": day_ns,
        "bid_0_price": 100.0, "bid_0_size": 1.0, "bid_1_price": 99.9, "bid_1_size": 2.0,
        "ask_0_price": 100.1, "ask_0_size": 1.0, "ask_1_price": 100.2, "ask_1_size": 2.0}])
    kw = dict(day=day, k=2, grid_ms=3_600_000, policy=ReseedPolicy(min_levels_per_side=1),
              book_stride_ms=1000, thresholds=rbr.Thresholds())
    py_frame, py_info = rbr.recon_topk_day(deltas, book, engine="python", price_scale=None, **kw)
    nat_frame, nat_info = rbr.recon_topk_day(deltas, book, engine="native", price_scale=SCALE,
                                             **kw)
    assert py_info["classification"] == nat_info["classification"] == rbr.CERTIFIED
    assert nat_info["seed"] == py_info["seed"]
    assert nat_info["quality"] == py_info["quality"]
    pd.testing.assert_frame_equal(nat_frame, py_frame, check_dtype=True)
