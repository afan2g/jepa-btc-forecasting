"""Native seed/reseed replay conformance vs the Python oracle (docs/data.md §5a-Recon; plan
`docs/superpowers/plans/2026-07-01-native-recon-engine.md`).

The Python `recon.reseed.reconstruct_lake_l2_at_samples_seeded` is the correctness oracle. Every
native-vs-Python test builds the SAME synthetic Lake `book_delta_v2` frame + `book` snapshots and
asserts the native `(frame, meta)` is identical. All native tests are skipped cleanly when the
`recon_native` extension is not built, so `python -m pytest -q` still passes without Rust.

The Python-only reference tests at the bottom pin the metrics-only invariant (native quality-map mode
classifies off `meta` without the frame): `_replay` metrics must equal `frame_quality(frame)`.
"""
import math

import numpy as np
import pandas as pd
import pytest

from recon.parity import frame_quality
from recon.reseed import (
    ReseedPolicy,
    book_snapshot,
    reconstruct_lake_l2_at_samples_seeded,
    snapshots_from_lake_book_df,
)
from recon import native as rn

native = pytest.mark.skipif(not rn.native_available(),
                            reason="recon_native extension not built (maturin develop)")

# On-grid ($0.01) fixtures => scale 100; matches the COINBASE/BTC-USD tick contract.
SCALE = 100
NOW = ReseedPolicy(reseed_after_crossed_s=0.0, min_levels_per_side=1)


def _lake_df(rows):
    """Real-Lake-schema book_delta_v2 frame from (ts_ns, seq, is_bid, price, size) tuples."""
    df = pd.DataFrame(rows, columns=["origin_time", "sequence_number", "side_is_bid",
                                     "price", "size"])
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    return df


def _assert_conforms(df, grid, *, k, snapshots=None, policy=NOW, frame_out=True, scale=SCALE):
    """Run native + Python on the same inputs and assert identical frame and meta."""
    py_frame, py_meta = reconstruct_lake_l2_at_samples_seeded(
        df, grid, k=k, engine_time_col="origin_time", snapshots=snapshots, policy=policy,
        frame_out=frame_out)
    nat_frame, nat_meta = rn.reconstruct_lake_l2_at_samples_seeded_native(
        df, grid, k=k, engine_time_col="origin_time", snapshots=snapshots, policy=policy,
        frame_out=frame_out, price_scale=scale)
    assert nat_meta == py_meta, f"meta mismatch\nnative={nat_meta}\npython={py_meta}"
    if frame_out:
        pd.testing.assert_frame_equal(nat_frame, py_frame, check_dtype=True)
    else:
        assert nat_frame is None and py_frame is None
    return nat_frame, nat_meta


# --------------------------------------------------------------------------- capability / import
def test_native_capability_flag_is_boolean():
    assert isinstance(rn.native_available(), bool)


@native
def test_native_extension_reports_reason_enum_matching_python():
    import recon_native
    assert recon_native.N_REASONS == len(rn.REASON_CODES)
    assert recon_native.NO_SNAPSHOTS == 255


# --------------------------------------------------------------------------- engine resolution
def test_resolve_engine_python_is_always_python():
    assert rn.resolve_engine("python", exchange="COINBASE", symbol="BTC-USD") == ("python", None, None)


def test_resolve_engine_native_requires_verified_tick_scale():
    # An unknown symbol has no verified tick scale => explicit native cannot resolve to native (the
    # caller must abort). Independent of whether the extension is importable.
    eng, scale, note = rn.resolve_engine("native", exchange="COINBASE", symbol="NOPE-USD")
    assert eng == "python" and scale is None and note is not None


def test_resolve_engine_auto_never_errors_and_falls_back():
    eng, scale, note = rn.resolve_engine("auto", exchange="COINBASE", symbol="NOPE-USD")
    assert eng == "python" and scale is None  # unverified symbol => Python under auto


def test_resolve_engine_native_when_available_and_verified():
    eng, scale, note = rn.resolve_engine("native", exchange="COINBASE", symbol="BTC-USD")
    if rn.native_available():
        assert eng == "native" and scale == 100 and note is None
    else:
        assert eng == "python" and scale is None and note is not None  # abort path when not built


def test_tick_scale_for_known_and_unknown():
    assert rn.tick_scale_for("COINBASE", "BTC-USD") == 100
    assert rn.tick_scale_for("coinbase", "btc-usd") == 100   # case-insensitive
    assert rn.tick_scale_for("COINBASE", "ETH-USD") is None


# --------------------------------------------------------------------------- capability gate (stale build)
# A stale/incompatible recon_native must be rejected at import time so `--engine native` fails BEFORE
# any Lake load (not later at reconstruct, or — worse — with silently-misaligned reason codes).
def _fake_native(*, with_fn=True, n_reasons=None, no_snapshots=255, meta_abi=rn._META_ABI):
    import types
    n = len(rn.REASON_CODES) if n_reasons is None else n_reasons
    ns = types.SimpleNamespace(N_REASONS=n, NO_SNAPSHOTS=no_snapshots)
    if meta_abi is not None:
        ns.META_ABI = meta_abi
    if with_fn:
        ns.reconstruct_seeded = lambda *a, **k: None
    return ns


def test_validate_native_accepts_matching_abi():
    rn._validate_native(_fake_native())  # no raise


def test_validate_native_rejects_missing_entrypoint():
    with pytest.raises(ImportError):
        rn._validate_native(_fake_native(with_fn=False))


def test_validate_native_rejects_reason_enum_mismatch():
    with pytest.raises(ImportError):
        rn._validate_native(_fake_native(n_reasons=len(rn.REASON_CODES) + 1))


def test_validate_native_rejects_sentinel_mismatch():
    with pytest.raises(ImportError):
        rn._validate_native(_fake_native(no_snapshots=7))


def test_validate_native_rejects_meta_abi_mismatch():
    # A build whose result-dict ABI differs (e.g. pre-coverage v1) must be rejected at import, so a
    # stale extension can never silently drop the coverage block the quality map plans fills from.
    with pytest.raises(ImportError):
        rn._validate_native(_fake_native(meta_abi=rn._META_ABI - 1))


def test_validate_native_rejects_missing_meta_abi_attr():
    # Pre-coverage builds have no META_ABI at all — that is the stale-build signature.
    with pytest.raises(ImportError):
        rn._validate_native(_fake_native(meta_abi=None))


@native
def test_native_extension_reports_meta_abi_matching_wrapper():
    import recon_native
    assert recon_native.META_ABI == rn._META_ABI


@native
def test_real_native_module_passes_capability_gate():
    import recon_native
    rn._validate_native(recon_native)          # the built extension satisfies the ABI contract
    assert rn.native_available() is True


# --------------------------------------------------------------------------- valid seed
@native
def test_valid_seed_day_frame_and_meta_match_python():
    df = _lake_df([(100, 1, True, 100.0, 1.0), (150, 2, False, 101.0, 2.0),
                   (200, 3, True, 100.0, 3.0)])
    seed = [book_snapshot(0, bids=[(100.0, 2.0)], asks=[(101.0, 3.0)])]
    _, m = _assert_conforms(df, [50, 120, 180, 250], k=2, snapshots=seed)
    assert m["seed_accepted"] is True and m["seed_ts"] == 0 and m["seed_reason"] == "ok"


# --------------------------------------------------------------------------- reseed repair
def _stranded_df():
    return _lake_df([
        (10, 1, True, 102.0, 1.0),    # bid 102 > ask 101 => crossed (stranded ask)
        (100, 2, False, 101.0, 0.0),  # ask 101 removed (delayed clear)
        (100, 3, False, 103.0, 1.0),  # ask 103 posts => uncrossed again
    ])


@native
def test_reseed_repair_conformance():
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
             book_snapshot(30, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]
    _, m = _assert_conforms(_stranded_df(), [5, 20, 50, 150], k=1, snapshots=snaps)
    assert m["reseed_count"] == 1 and m["reseed_ts"] == [30]
    assert m["crossed_samples"] == 1


@native
def test_reseed_blocked_when_only_invalid_snapshots():
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
             book_snapshot(30, bids=[(105.0, 1.0)], asks=[(104.0, 1.0)])]  # crossed => unusable
    _, m = _assert_conforms(_stranded_df(), [20, 50], k=1, snapshots=snaps)
    assert m["reseed_count"] == 0 and m["reseed_blocked_invalid_snapshot"] >= 1


# --------------------------------------------------------------------------- same-ts delta/snapshot
@native
def test_same_timestamp_delta_before_snapshot():
    # A delta and a snapshot share ts=10: the delta would post bid 105 (cross), then the same-ts
    # snapshot OVERWRITES to a clean 100/101. As-of ts>=10 must read the SNAPSHOT (delta applied
    # first, snapshot second) — a divergence here would flip the sampled book.
    df = _lake_df([(5, 1, True, 90.0, 1.0), (10, 2, True, 105.0, 1.0)])
    snaps = [book_snapshot(5, bids=[(90.0, 1.0)], asks=[(91.0, 1.0)]),
             book_snapshot(10, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    frame, _ = _assert_conforms(df, [7, 20], k=1, snapshots=snaps)
    f = frame.set_index("sample_ts")
    assert f.loc[20, "bid_0_price"] == 100.0 and f.loc[20, "ask_0_price"] == 101.0


# --------------------------------------------------------------------------- stable (ts,seq) order
@native
def test_stable_equal_ts_seq_row_order_final_size_depends_on_order():
    # Two absolute-size updates to the SAME (ts, seq, side, price); the FINAL size is order-dependent.
    # np.lexsort is stable, so source order (5.0 then 9.0) must win => final 9.0. A native sort that
    # reordered equal (ts, seq) rows would land on 5.0.
    df = _lake_df([
        (2, 1, True, 100.0, 1.0), (2, 1, False, 101.0, 1.0),
        (5, 7, True, 100.0, 5.0), (5, 7, True, 100.0, 9.0),
    ])
    frame, _ = _assert_conforms(df, [10], k=1)
    assert frame.iloc[0]["bid_0_size"] == 9.0


@native
def test_sort_matches_lexsort_with_duplicate_ts_and_seq():
    # Duplicate timestamps + duplicate sequence numbers + interleaved sides. If native sorting drifts
    # from np.lexsort((seq, ts)), the reconstructed book (and frame) diverges from Python.
    rng = np.random.default_rng(7)
    ts = rng.integers(1, 6, size=60)                 # heavy ts duplication (populated: >0)
    seq = rng.integers(0, 3, size=60)                # heavy seq duplication
    is_bid = rng.integers(0, 2, size=60).astype(bool)
    px_cents = rng.integers(9990, 10010, size=60)    # on $0.01 grid
    price = px_cents / 100.0
    size = rng.choice([0.0, 1.0, 2.0, 3.0], size=60)
    df = _lake_df(list(zip(ts.tolist(), seq.tolist(), is_bid.tolist(),
                           price.tolist(), size.tolist())))
    seed = [book_snapshot(0, bids=[(99.90, 1.0)], asks=[(100.10, 1.0)])]
    _assert_conforms(df, list(range(0, 7)), k=5, snapshots=seed)


# --------------------------------------------------------------------------- no valid seed
@native
def test_no_valid_seed_meta_matches_python():
    df = _lake_df([(100, 1, True, 100.0, 1.0)])
    bad = [book_snapshot(0, bids=[(101.0, 1.0)], asks=[(100.0, 1.0)])]  # crossed => rejected
    _, m = _assert_conforms(df, [50, 150], k=1, snapshots=bad)
    assert m["seed_accepted"] is False and m["seed_reason"] == "crossed"
    assert m["snapshot_reason_codes"].get("crossed", 0) >= 1


@native
def test_no_snapshots_at_all_is_byte_identical_cold_start():
    df = _lake_df([(10, 1, True, 100.0, 2.0), (10, 2, False, 101.0, 3.0),
                   (30, 3, True, 100.0, 0.0), (30, 4, True, 99.0, 1.0)])
    _, m = _assert_conforms(df, [5, 15, 35], k=2, snapshots=None)
    assert m["seed_accepted"] is False and m["seed_reason"] == "no_snapshots"


# --------------------------------------------------------------------------- frame_out=False
@native
def test_frame_out_false_metrics_match_and_no_frame():
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
             book_snapshot(30, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]
    nat_frame, nat_meta = _assert_conforms(_stranded_df(), [5, 20, 50, 150], k=1,
                                           snapshots=snaps, frame_out=False)
    assert nat_frame is None


# --------------------------------------------------------------------------- metrics-only == frame_quality
@native
def test_native_metrics_only_matches_python_frame_quality_classification():
    # A fixture with crossed, missing (pre-seed), and thin samples. Native metrics-only (no top-K
    # frame) must agree with `frame_quality(python_frame)` on crossed+missing and with Python `meta`
    # on thin — the load-bearing invariant the quality-map native path relies on.
    df = _lake_df([
        (10, 1, True, 100.0, 1.0),    # one-sided (thin, bid only) until ask arrives
        (20, 2, False, 101.0, 1.0),
        (30, 3, True, 102.0, 1.0),    # strands ask101 => crossed
    ])
    seed = [book_snapshot(5, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    grid = [1, 15, 25, 40, 50]        # 1 => pre-seed missing; others exercise thin/crossed
    py_frame, py_meta = reconstruct_lake_l2_at_samples_seeded(
        df, grid, k=3, engine_time_col="origin_time", snapshots=seed,
        policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    fq = frame_quality(py_frame)
    _, nat_meta = rn.reconstruct_lake_l2_at_samples_seeded_native(
        df, grid, k=3, engine_time_col="origin_time", snapshots=seed,
        policy=ReseedPolicy(enabled=False, min_levels_per_side=1), frame_out=False, price_scale=SCALE)
    assert nat_meta["crossed_samples"] == fq["crossed_samples"]
    assert nat_meta["missing_book_samples"] == fq["missing_book_samples"]
    assert nat_meta["crossed_rate"] == fq["crossed_rate"]
    assert nat_meta["missing_book_fraction"] == fq["missing_book_fraction"]
    assert nat_meta["thin_depth_fraction"] == py_meta["thin_depth_fraction"]
    assert nat_meta["thin_depth_samples"] >= 1  # the one-sided/thin samples are actually exercised


# --------------------------------------------------------------------------- tick-conversion boundaries
@native
@pytest.mark.parametrize("bid_cents,expect_crossed", [
    (10001, False),  # bid 100.01 vs ask 100.02 => one tick uncrossed
    (10002, True),   # bid 100.02 == ask 100.02 => crossed (>=)
    (10003, True),   # bid 100.03 => one tick crossed
])
def test_tick_boundary_crossing_matches_python(bid_cents, expect_crossed):
    seed = [book_snapshot(0, bids=[(100.00, 1.0)], asks=[(100.02, 1.0)])]
    df = _lake_df([(10, 1, True, bid_cents / 100.0, 1.0)])   # move best bid to the boundary
    _, m = _assert_conforms(df, [20], k=1, snapshots=seed,
                            policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    assert (m["crossed_samples"] == 1) is expect_crossed


@native
def test_emitted_prices_are_source_floats_not_tick_roundtrip():
    # Feed a price whose f64 value != tick/scale round-trip (100.017 -> tick 10002 -> 100.02). The
    # emitted bid_0_price MUST be the SOURCE float 100.017, byte-identical to Python (which keys the
    # book by the exact float), never reconstructed as tick/scale.
    px = 100.017
    df = _lake_df([(10, 1, True, px, 1.0), (10, 2, False, 100.05, 1.0)])
    seed = [book_snapshot(0, bids=[(100.00, 1.0)], asks=[(100.05, 1.0)])]
    frame, _ = _assert_conforms(df, [20], k=1, snapshots=seed,
                                policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    got = frame.iloc[0]["bid_0_price"]
    assert got == px                      # exact source float
    assert got != round(px * SCALE) / SCALE  # and NOT the tick/scale round-trip


# --------------------------------------------------------------------------- NaN-padded snapshot parse
@native
def test_nan_padded_snapshot_preclassification_matches_python():
    # Snapshots parsed by snapshots_from_lake_book_df (NaN padding dropped) then classified — native
    # reuses the SAME parse+classify, so reason codes and seeding must match Python exactly.
    book_df = pd.DataFrame({
        "origin_time": pd.to_datetime([5, 30]),
        "bid_0_price": [100.0, 102.0], "bid_0_size": [1.0, 1.0],
        "bid_1_price": [99.0, np.nan], "bid_1_size": [1.0, np.nan],   # row1 NaN-padded
        "ask_0_price": [101.0, 103.0], "ask_0_size": [1.0, 1.0],
        "ask_1_price": [102.0, np.nan], "ask_1_size": [1.0, np.nan],
    })
    snaps = snapshots_from_lake_book_df(book_df, engine_time_col="origin_time")
    _, m = _assert_conforms(_stranded_df(), [5, 20, 50, 150], k=1, snapshots=snaps,
                            policy=ReseedPolicy(reseed_after_crossed_s=0.0, min_levels_per_side=1))
    assert m["reseed_count"] == 1


# --------------------------------------------------------------------------- trailing crossed duration
@native
def test_trailing_crossed_duration_close_out_matches_python():
    df = _lake_df([(10, 1, True, 102.0, 1.0)])   # crosses on the final event, stays crossed
    seed = [book_snapshot(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    _, m = _assert_conforms(df, [5, 10, 20, 30], k=1, snapshots=seed,
                            policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    assert m["crossed_samples"] == 3
    assert m["crossed_duration_ns"] == 20


# --------------------------------------------------------------------------- randomized fuzz
@native
@pytest.mark.parametrize("seed_val", [0, 1, 2, 3, 4])
def test_randomized_stream_conformance(seed_val):
    # Broad coverage: random on-grid deltas (with removals + duplicate ts/seq), a validated seed, and
    # a fixing snapshot — native and Python must agree on the full frame and every metric.
    rng = np.random.default_rng(seed_val)
    n = 400
    ts = np.sort(rng.integers(1, 300, size=n))    # populated: >0
    seq = rng.integers(0, 50, size=n)
    is_bid = rng.integers(0, 2, size=n).astype(bool)
    price = rng.integers(9900, 10100, size=n) / 100.0   # $0.01 grid around 100
    size = rng.choice([0.0, 0.0, 1.0, 2.0, 5.0], size=n)
    df = _lake_df(list(zip(ts.tolist(), seq.tolist(), is_bid.tolist(),
                           price.tolist(), size.tolist())))
    snaps = [book_snapshot(0, bids=[(99.50, 1.0), (99.40, 2.0)], asks=[(100.50, 1.0), (100.60, 2.0)]),
             book_snapshot(150, bids=[(99.80, 1.0)], asks=[(100.20, 1.0)])]
    grid = list(range(0, 300, 5))
    _assert_conforms(df, grid, k=10, snapshots=snaps,
                     policy=ReseedPolicy(reseed_after_crossed_s=0.0, min_levels_per_side=1))


# --------------------------------------------------------------------------- coverage metrics (Task 3)
# Both engines' meta carries a compact `coverage` block (plan-doc
# 2026-07-02-partial-day-fill-policy.md Task 3): maximal half-open [i0, i1) sample-index runs where
# the sample is NOT valid under the shared stitch-policy predicate (`valid_mask_from_frame` at
# min_levels_per_side=1: both top-of-book prices present, non-NaN, bid < ask), plus the first/last
# present-sample indices behind `lake_present_*`. The FRAME-derived mask stays the correctness
# oracle: the Python-only tests below pin replay coverage == frame-derived coverage, and
# `_assert_conforms` (full meta equality) pins native == Python on every fixture in this file.
from recon.stitch_policy import valid_mask_from_frame  # noqa: E402


def _mask_runs(mask):
    """Maximal half-open [i0, i1) index runs where `mask` is True — the test-local oracle."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        elif not m and start is not None:
            runs.append([start, i])
            start = None
    if start is not None:
        runs.append([start, len(mask)])
    return runs


def _frame_coverage(frame):
    """Frame-derived coverage oracle: invalid runs from `valid_mask_from_frame`, presence bounds
    from the notna top-of-book columns (the quality map's `_stitch_and_coverage` predicates)."""
    f = frame.sort_values("sample_ts")
    valid = valid_mask_from_frame(f)
    present = (f["bid_0_price"].notna() & f["ask_0_price"].notna()).to_numpy(dtype=bool)
    idx = np.flatnonzero(present)
    runs = _mask_runs(~valid)
    return {"present_first_idx": (int(idx[0]) if len(idx) else None),
            "present_last_idx": (int(idx[-1]) if len(idx) else None),
            "n_invalid_runs": len(runs), "invalid_runs_idx": runs}


def test_python_replay_coverage_matches_frame_derived_mask():
    # missing (pre-seed) + valid + crossed phases → a leading AND a trailing invalid run.
    df = _lake_df([(10, 1, True, 100.0, 1.0), (20, 2, False, 101.0, 1.0),
                   (30, 3, True, 102.0, 1.0)])
    seed = [book_snapshot(5, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    frame, meta = reconstruct_lake_l2_at_samples_seeded(
        df, [1, 15, 25, 40, 50], k=3, engine_time_col="origin_time", snapshots=seed,
        policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    assert meta["coverage"] == _frame_coverage(frame)
    assert meta["coverage"]["invalid_runs_idx"] == [[0, 1], [3, 5]]
    assert meta["coverage"]["n_invalid_runs"] == 2
    assert meta["coverage"]["present_first_idx"] == 1   # crossed samples are present, not valid
    assert meta["coverage"]["present_last_idx"] == 4


def test_python_replay_coverage_never_two_sided_book_is_one_run_no_presence():
    df = _lake_df([(10, 1, True, 100.0, 2.0)])          # bid-only book, never two-sided
    frame, meta = reconstruct_lake_l2_at_samples_seeded(
        df, [5, 15, 25], k=1, engine_time_col="origin_time", snapshots=None)
    assert meta["coverage"] == _frame_coverage(frame)
    assert meta["coverage"] == {"present_first_idx": None, "present_last_idx": None,
                                "n_invalid_runs": 1, "invalid_runs_idx": [[0, 3]]}


def test_python_replay_coverage_empty_grid_is_empty():
    df = _lake_df([(10, 1, True, 100.0, 1.0)])
    _, meta = reconstruct_lake_l2_at_samples_seeded(df, [], k=1, engine_time_col="origin_time")
    assert meta["coverage"] == {"present_first_idx": None, "present_last_idx": None,
                                "n_invalid_runs": 0, "invalid_runs_idx": []}


@native
def test_native_coverage_clean_day_has_no_invalid_runs():
    df = _lake_df([(10, 1, True, 100.0, 1.0)])
    seed = [book_snapshot(0, bids=[(100.0, 2.0)], asks=[(101.0, 3.0)])]
    _, m = _assert_conforms(df, [5, 15, 25], k=1, snapshots=seed)
    assert m["coverage"] == {"present_first_idx": 0, "present_last_idx": 2,
                             "n_invalid_runs": 0, "invalid_runs_idx": []}


@native
def test_native_coverage_leading_missing_prefix_is_one_leading_run():
    seed = [book_snapshot(30, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    df = _lake_df([(40, 1, True, 100.5, 1.0)])
    _, m = _assert_conforms(df, [0, 10, 20, 35, 45, 55], k=1, snapshots=seed)
    assert m["coverage"]["invalid_runs_idx"] == [[0, 3]]
    assert m["coverage"]["present_first_idx"] == 3
    assert m["coverage"]["present_last_idx"] == 5


@native
def test_native_coverage_trailing_crossed_suffix_is_one_trailing_run():
    # Crossed samples are invalid but PRESENT: the presence bound reaches the grid end while the
    # trailing invalid run records the unhealed crossing (closed at n, the half-open convention).
    seed = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    df = _lake_df([(10, 1, True, 102.0, 1.0)])
    _, m = _assert_conforms(df, [5, 15, 25], k=1, snapshots=seed,
                            policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    assert m["crossed_samples"] == 2
    assert m["coverage"]["invalid_runs_idx"] == [[1, 3]]
    assert m["coverage"]["present_first_idx"] == 0
    assert m["coverage"]["present_last_idx"] == 2


@native
def test_native_coverage_internal_crossed_gap_is_internal_run():
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
             book_snapshot(30, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]
    _, m = _assert_conforms(_stranded_df(), [5, 20, 50, 150], k=1, snapshots=snaps)
    assert m["reseed_count"] == 1                       # the reseed heals the gap mid-grid
    assert m["coverage"]["invalid_runs_idx"] == [[1, 2]]
    assert m["coverage"]["present_first_idx"] == 0
    assert m["coverage"]["present_last_idx"] == 3


@native
def test_native_coverage_missing_book_midday_counts_invalid():
    seed = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    df = _lake_df([(10, 1, False, 101.0, 0.0),          # removes the only ask → one-sided book
                   (30, 2, False, 101.0, 1.0)])         # restores it
    _, m = _assert_conforms(df, [5, 15, 25, 35], k=1, snapshots=seed,
                            policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    assert m["missing_book_samples"] == 2
    assert m["coverage"]["invalid_runs_idx"] == [[1, 3]]
    # Presence has a mid-day HOLE here; the block records BOUNDS only — exactly what
    # plan_day_stitch reads (first/last present sample) for lake_present_*.
    assert m["coverage"]["present_first_idx"] == 0
    assert m["coverage"]["present_last_idx"] == 3


@native
def test_native_coverage_multiple_runs_leading_and_trailing():
    seed = [book_snapshot(30, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    df = _lake_df([(60, 1, True, 102.0, 1.0)])          # crosses at 60, never heals
    _, m = _assert_conforms(df, [10, 40, 70], k=1, snapshots=seed,
                            policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    assert m["coverage"]["invalid_runs_idx"] == [[0, 1], [2, 3]]
    assert m["coverage"]["n_invalid_runs"] == 2


# --------------------------------------------------------------------------- Python-only reference
# (runs WITHOUT the extension — pins the invariant the native metrics-only mode relies on)
def test_python_replay_metrics_equal_frame_quality():
    df = _lake_df([
        (10, 1, True, 100.0, 1.0),
        (20, 2, False, 101.0, 1.0),
        (30, 3, True, 102.0, 1.0),    # strands ask101 => crossed
    ])
    seed = [book_snapshot(5, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    grid = [1, 15, 25, 40, 50]
    frame, meta = reconstruct_lake_l2_at_samples_seeded(
        df, grid, k=3, engine_time_col="origin_time", snapshots=seed,
        policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    fq = frame_quality(frame)
    assert meta["crossed_samples"] == fq["crossed_samples"]
    assert meta["missing_book_samples"] == fq["missing_book_samples"]
    assert meta["crossed_rate"] == fq["crossed_rate"]
    assert meta["missing_book_fraction"] == fq["missing_book_fraction"]
    # thin is present+uncrossed with < k levels on a side; recount from the frame to pin it too.
    f = frame.set_index("sample_ts")
    bid_cols = [c for c in f.columns if c.startswith("bid_") and c.endswith("_price")]
    ask_cols = [c for c in f.columns if c.startswith("ask_") and c.endswith("_price")]
    present = f["bid_0_price"].notna() & f["ask_0_price"].notna()
    uncrossed = present & (f["bid_0_price"] < f["ask_0_price"])
    thin = uncrossed & ((f[bid_cols].notna().sum(axis=1) < 3) | (f[ask_cols].notna().sum(axis=1) < 3))
    assert meta["thin_depth_samples"] == int(thin.sum())


def test_python_reference_thin_and_missing_are_distinct():
    # A sanity fixture ensuring the reference test above actually exercises missing AND thin AND
    # crossed (so the invariant is not vacuously true).
    df = _lake_df([(10, 1, True, 100.0, 1.0), (20, 2, False, 101.0, 1.0), (30, 3, True, 102.0, 1.0)])
    seed = [book_snapshot(5, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    _, meta = reconstruct_lake_l2_at_samples_seeded(
        df, [1, 15, 25, 40], k=3, engine_time_col="origin_time", snapshots=seed,
        policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    assert meta["missing_book_samples"] >= 1
    assert meta["thin_depth_samples"] >= 1
    assert meta["crossed_samples"] >= 1
    assert not math.isnan(meta["crossed_rate"])
