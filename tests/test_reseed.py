"""Unit tests for the Lake `book_delta_v2` seed/reseed policy (docs/data.md §5a-Recon).

These pin the fix for the live 2025-06-01 failure mode: a cold-started Lake reconstruction
strands levels (a `size=0` clearing update never lands) and the book freezes crossed ~67% of
the day. The policy seeds from the validated Lake `book` snapshot product and reseeds when the
reconstructed book stays crossed beyond a tolerance — WITHOUT a naive `sequence_number`-diff
gap detector (seq duplicates ~91% of rows and is per-event, not per-row).
"""
import math

import numpy as np
import pandas as pd

from recon.events import Delta
from recon.orderbook import OrderBook
from recon.reconstruct import reconstruct_lake_l2_at_samples
from recon.reseed import (
    BookSnapshot,
    ReseedPolicy,
    book_snapshot,
    classify_snapshot,
    reconstruct_lake_l2_at_samples_seeded,
    reconstruct_seeded,
    snapshots_from_lake_book_df,
)

# Reseed-immediately policy for hand-checkable small-ns streams (default tolerance is 2 s).
NOW = ReseedPolicy(reseed_after_crossed_s=0.0, min_levels_per_side=1)


def _lake_df(rows):
    """Real-Lake-schema book_delta_v2 frame from (ts_ns, seq, is_bid, price, size) tuples."""
    df = pd.DataFrame(rows, columns=["origin_time", "sequence_number", "side_is_bid",
                                     "price", "size"])
    df["origin_time"] = pd.to_datetime(df["origin_time"])
    return df


# --------------------------------------------------------------------------- classify_snapshot
def test_classify_accepts_a_valid_two_sided_uncrossed_snapshot():
    snap = book_snapshot(0, bids=[(100.0, 1.0), (99.0, 2.0)], asks=[(101.0, 1.0), (102.0, 2.0)])
    assert classify_snapshot(snap, min_levels_per_side=2) == "ok"


def test_classify_rejects_crossed_snapshot():
    snap = book_snapshot(0, bids=[(101.0, 1.0)], asks=[(100.0, 1.0)])  # bid >= ask
    assert classify_snapshot(snap, min_levels_per_side=1) == "crossed"


def test_classify_rejects_one_sided_snapshot():
    snap = book_snapshot(0, bids=[(100.0, 1.0)], asks=[])
    assert classify_snapshot(snap, min_levels_per_side=1) == "one_sided"


def test_classify_rejects_thin_depth():
    snap = book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
    assert classify_snapshot(snap, min_levels_per_side=3) == "thin_depth"


def test_classify_rejects_nonfinite_or_nonpositive_values():
    bad_size = book_snapshot(0, bids=[(100.0, 0.0)], asks=[(101.0, 1.0)])
    assert classify_snapshot(bad_size, min_levels_per_side=1) == "bad_values"
    bad_px = book_snapshot(0, bids=[(-1.0, 1.0)], asks=[(101.0, 1.0)])
    assert classify_snapshot(bad_px, min_levels_per_side=1) == "bad_values"
    inf_px = book_snapshot(0, bids=[(float("inf"), 1.0)], asks=[(101.0, 1.0)])
    assert classify_snapshot(inf_px, min_levels_per_side=1) == "bad_values"


def test_classify_optional_wide_spread_guard():
    snap = book_snapshot(0, bids=[(100.0, 1.0)], asks=[(150.0, 1.0)])  # 50% spread
    assert classify_snapshot(snap, min_levels_per_side=1) == "ok"      # off by default
    assert classify_snapshot(snap, min_levels_per_side=1, max_spread_frac=0.05) == "wide_spread"


# --------------------------------------------------------------------------- snapshot parsing
def test_snapshots_from_lake_book_df_parses_levels_and_drops_padding():
    df = pd.DataFrame({
        "origin_time": pd.to_datetime([10, 20]),
        "bid_0_price": [100.0, 100.5], "bid_0_size": [1.0, 2.0],
        "bid_1_price": [99.0, np.nan], "bid_1_size": [1.0, np.nan],   # row 1 has 1 bid level
        "ask_0_price": [101.0, 101.5], "ask_0_size": [1.0, 2.0],
        "ask_1_price": [102.0, 103.0], "ask_1_size": [1.0, 1.0],
    })
    snaps = snapshots_from_lake_book_df(df, engine_time_col="origin_time")
    assert [s.ts for s in snaps] == [10, 20]
    assert snaps[0].bids == ((100.0, 1.0), (99.0, 1.0))      # sorted desc
    assert snaps[0].asks == ((101.0, 1.0), (102.0, 1.0))     # sorted asc
    assert snaps[1].bids == ((100.5, 2.0),)                  # NaN padding dropped
    assert classify_snapshot(snaps[0], min_levels_per_side=2) == "ok"


def test_snapshots_from_lake_book_df_drops_finite_price_nan_size_padding():
    # A level with a finite price but NaN size is PADDING (dropped), not a malformed level — so it
    # must not poison the whole snapshot into "bad_values" and silently disable seeding.
    df = pd.DataFrame({
        "origin_time": pd.to_datetime([10]),
        "bid_0_price": [100.0], "bid_0_size": [1.0],
        "bid_1_price": [99.0], "bid_1_size": [np.nan],   # finite price, NaN size ⇒ padding
        "ask_0_price": [101.0], "ask_0_size": [1.0],
        "ask_1_price": [np.nan], "ask_1_size": [np.nan],
    })
    snaps = snapshots_from_lake_book_df(df, engine_time_col="origin_time")
    assert snaps[0].bids == ((100.0, 1.0),)                          # (99.0, NaN) dropped, not kept
    assert classify_snapshot(snaps[0], min_levels_per_side=1) == "ok"  # snapshot still valid


def test_snapshots_from_lake_book_df_thins_by_stride():
    df = pd.DataFrame({
        "origin_time": pd.to_datetime([10, 11, 12, 20, 21]),
        "bid_0_price": [100.0] * 5, "bid_0_size": [1.0] * 5,
        "ask_0_price": [101.0] * 5, "ask_0_size": [1.0] * 5,
    })
    snaps = snapshots_from_lake_book_df(df, engine_time_col="origin_time", stride_ns=10)
    assert [s.ts for s in snaps] == [10, 20]  # keep first, then first ts >= prev_kept + stride


# --------------------------------------------------------------------------- seeding
def test_valid_snapshot_seed_makes_an_empty_lake_book_usable():
    # A single delta at ts=100; without a seed, samples before it are an empty book.
    df = _lake_df([(100, 1, True, 100.0, 1.0)])
    seed = [book_snapshot(0, bids=[(100.0, 2.0)], asks=[(101.0, 3.0)])]
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [50, 150], k=1, engine_time_col="origin_time", snapshots=seed, policy=NOW)
    f = frame.set_index("sample_ts")
    assert f.loc[50, "bid_0_price"] == 100.0 and f.loc[50, "ask_0_price"] == 101.0
    assert f.loc[50, "mid"] == 100.5            # seed visible before the first delta
    assert m["seed_accepted"] is True and m["seed_ts"] == 0 and m["seed_reason"] == "ok"


def test_crossed_seed_is_rejected_and_book_cold_starts():
    df = _lake_df([(100, 1, True, 100.0, 1.0)])
    bad_seed = [book_snapshot(0, bids=[(101.0, 1.0)], asks=[(100.0, 1.0)])]  # crossed
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [50, 150], k=1, engine_time_col="origin_time", snapshots=bad_seed, policy=NOW)
    f = frame.set_index("sample_ts")
    assert math.isnan(f.loc[50, "bid_0_price"])  # no seed applied → empty pre-delta
    assert m["seed_accepted"] is False
    assert m["snapshot_reason_codes"].get("crossed", 0) >= 1


def test_first_valid_snapshot_seeds_even_if_earlier_ones_invalid():
    df = _lake_df([(300, 1, True, 100.0, 1.0)])
    snaps = [book_snapshot(0, bids=[(101.0, 1.0)], asks=[(100.0, 1.0)]),   # crossed → skip
             book_snapshot(100, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]  # valid → seed
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [50, 150], k=1, engine_time_col="origin_time", snapshots=snaps, policy=NOW)
    f = frame.set_index("sample_ts")
    assert math.isnan(f.loc[50, "bid_0_price"])   # before the valid seed at ts=100
    assert f.loc[150, "bid_0_price"] == 100.0     # seeded by ts=100, visible at 150
    assert m["seed_accepted"] is True and m["seed_ts"] == 100


# --------------------------------------------------------------------------- reseed / stranding
def _stranded_df():
    # Seeded book bid100/ask101. ts=10 a bid lands at 102 with NO ask removal (stranded ask101)
    # → crossed. ts=100 the delayed clear finally lands (ask101->0, ask103 posts) → uncrossed.
    return _lake_df([
        (10, 1, True, 102.0, 1.0),    # bid 102 > ask 101 ⇒ crossed (stranded ask)
        (100, 2, False, 101.0, 0.0),  # ask 101 removed (the delayed clear)
        (100, 3, False, 103.0, 1.0),  # ask 103 posts ⇒ uncrossed again
    ])


def test_stranded_stream_crosses_without_reseed():
    df = _stranded_df()
    seed = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [5, 20, 50, 150], k=1, engine_time_col="origin_time",
        snapshots=seed, policy=ReseedPolicy(reseed_after_crossed_s=0.0, enabled=False))
    f = frame.set_index("sample_ts")
    assert f.loc[20, "bid_0_price"] >= f.loc[20, "ask_0_price"]   # crossed, no reseed
    assert f.loc[50, "bid_0_price"] >= f.loc[50, "ask_0_price"]   # still crossed at 50
    assert m["reseed_count"] == 0
    assert m["crossed_samples"] == 2                             # samples at 20 and 50


def test_reseed_recovers_a_stranded_book():
    df = _stranded_df()
    # vendor `book` snapshot at ts=30 shows the true (uncrossed) book: bid102/ask103.
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
             book_snapshot(30, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [5, 20, 50, 150], k=1, engine_time_col="origin_time", snapshots=snaps, policy=NOW)
    f = frame.set_index("sample_ts")
    assert f.loc[20, "bid_0_price"] >= f.loc[20, "ask_0_price"]   # still crossed before reseed
    assert f.loc[50, "bid_0_price"] < f.loc[50, "ask_0_price"]    # reseed at 30 fixed it
    assert f.loc[50, "ask_0_price"] == 103.0
    assert m["reseed_count"] == 1 and m["reseed_ts"] == [30]
    assert m["crossed_samples"] == 1                             # only ts=20 remained crossed


def test_reseed_does_not_introduce_lookahead():
    # The reseed snapshot is at ts=30; a sample at ts=20 (< 30) must NOT see it (still crossed).
    df = _stranded_df()
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
             book_snapshot(30, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [20, 25, 30, 35], k=1, engine_time_col="origin_time", snapshots=snaps, policy=NOW)
    f = frame.set_index("sample_ts")
    assert f.loc[20, "bid_0_price"] == 102.0 and f.loc[20, "ask_0_price"] == 101.0  # crossed
    assert f.loc[25, "ask_0_price"] == 101.0                       # still pre-reseed
    assert f.loc[30, "ask_0_price"] == 103.0                       # reseed applied AT its ts
    assert 20 in m["crossed_sample_ts"] and 30 not in m["crossed_sample_ts"]


def test_terminal_crossed_episode_duration_reaches_grid_end():
    # If the book crosses on the FINAL event and stays crossed through the trailing samples, the
    # open crossed interval must close at the grid end (last sample), not the last EVENT time —
    # otherwise crossed_duration_s_after reports 0 while trailing samples are counted crossed.
    df = _lake_df([(10, 1, True, 102.0, 1.0)])   # crosses at ts=10 (the last event), stays crossed
    seed = [book_snapshot(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [5, 10, 20, 30], k=1, engine_time_col="origin_time", snapshots=seed,
        policy=ReseedPolicy(enabled=False, min_levels_per_side=1))
    assert m["crossed_samples"] == 3            # samples at 10, 20, 30 are crossed
    assert m["crossed_duration_ns"] == 20       # 30 (grid end) − 10 (onset), NOT 0


def test_reseed_blocked_when_only_invalid_snapshots_available():
    df = _stranded_df()
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),
             book_snapshot(30, bids=[(105.0, 1.0)], asks=[(104.0, 1.0)])]  # crossed → unusable
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [20, 50], k=1, engine_time_col="origin_time", snapshots=snaps, policy=NOW)
    assert m["reseed_count"] == 0
    assert m["reseed_blocked_invalid_snapshot"] >= 1


# --------------------------------------------------------------------------- tolerance window
def test_reseed_tolerance_window_skips_a_self_healing_transient_cross():
    # The default-ON tolerance: a cross that self-heals before `reseed_after_crossed_s` elapses must
    # NOT consume a reseed even when a valid snapshot is available inside the sub-tolerance window.
    S = 1_000_000_000
    df = _lake_df([
        (1 * S, 1, True, 102.0, 1.0),                       # cross at +1s (stranded ask101)
        (2 * S + 500_000_000, 2, False, 101.0, 0.0),        # ask101 cleared at +2.5s → self-heal
        (2 * S + 500_000_000, 3, False, 103.0, 1.0),
    ])
    snaps = [book_snapshot(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),            # seed
             book_snapshot(2 * S, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]       # valid, but
    policy = ReseedPolicy(reseed_after_crossed_s=2.0, min_levels_per_side=1)        # only 1s into cross
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [1 * S + 500_000_000, 3 * S], k=1, engine_time_col="origin_time",
        snapshots=snaps, policy=policy)
    assert m["reseed_count"] == 0                            # snapshot@+2s skipped (1s < 2s tolerance)
    f = frame.set_index("sample_ts")
    assert f.loc[3 * S, "bid_0_price"] < f.loc[3 * S, "ask_0_price"]   # self-healed by the +2.5s delta


def test_reseed_fires_once_cross_persists_beyond_tolerance_window():
    S = 1_000_000_000
    df = _lake_df([(1 * S, 1, True, 102.0, 1.0)])           # cross at +1s, never self-heals
    snaps = [book_snapshot(1, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)]),            # seed
             book_snapshot(2 * S, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)]),       # +1s → skip
             book_snapshot(4 * S, bids=[(102.0, 1.0)], asks=[(103.0, 1.0)])]       # +3s → reseed
    policy = ReseedPolicy(reseed_after_crossed_s=2.0, min_levels_per_side=1)
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [1 * S + 500_000_000, 3 * S, 5 * S], k=1, engine_time_col="origin_time",
        snapshots=snaps, policy=policy)
    assert m["reseed_count"] == 1 and m["reseed_ts"] == [4 * S]
    f = frame.set_index("sample_ts")
    assert f.loc[3 * S, "bid_0_price"] >= f.loc[3 * S, "ask_0_price"]   # still crossed before reseed
    assert f.loc[5 * S, "ask_0_price"] == 103.0                         # reseeded by +4s
    assert 3 * S in m["crossed_sample_ts"] and 5 * S not in m["crossed_sample_ts"]


# --------------------------------------------------------------------------- seq-diff guard
def test_sequence_number_duplicates_do_not_trigger_reseed_or_gap_handling():
    # Coinbase book_delta_v2 duplicates sequence_number across ~91% of rows (per-event, not
    # per-row). A duplicate/non-increasing seq must NOT be read as a dropped-data gap.
    df = _lake_df([
        (10, 5, True, 100.0, 2.0),
        (10, 5, False, 101.0, 3.0),   # same seq=5 as the prior row (a duplicate, expected)
        (20, 5, True, 100.0, 1.0),    # seq goes backwards/flat vs nothing dropped
    ])
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [15, 25], k=1, engine_time_col="origin_time", snapshots=snaps, policy=NOW)
    f = frame.set_index("sample_ts")
    assert f.loc[25, "bid_0_price"] == 100.0 and f.loc[25, "ask_0_price"] == 101.0  # uncrossed
    assert m["reseed_count"] == 0 and m["crossed_samples"] == 0


def test_orderbook_seq_signal_is_not_used_as_a_gap_detector_by_reseed():
    # The OrderBook.apply monotonicity signal is informational only; reseed never consumes it.
    ob = OrderBook()
    ob.apply(Delta(10, 5, "bid", 100.0, 1.0))
    assert ob.apply(Delta(10, 5, "ask", 101.0, 1.0)) is False   # non-increasing seq flagged
    # but the book is perfectly valid/uncrossed — no reseed implication.
    assert ob.best_bid() == 100.0 and ob.best_ask() == 101.0


# --------------------------------------------------------------------------- equivalence
def test_no_snapshot_path_is_byte_identical_to_cold_start():
    df = _lake_df([
        (10, 1, True, 100.0, 2.0), (10, 2, False, 101.0, 3.0),
        (30, 3, True, 100.0, 0.0), (30, 4, True, 99.0, 1.0),
        (50, 5, False, 101.0, 0.0), (50, 6, False, 102.0, 4.0),
    ])
    grid = [5, 15, 35, 55, 65]
    cold = reconstruct_lake_l2_at_samples(df, grid, k=2, engine_time_col="origin_time")
    seeded, m = reconstruct_lake_l2_at_samples_seeded(
        df, grid, k=2, engine_time_col="origin_time", snapshots=None)
    pd.testing.assert_frame_equal(seeded, cold, check_dtype=True)
    assert m["seed_accepted"] is False and m["reseed_count"] == 0


def test_reconstruct_seeded_delta_list_path_matches_array_path():
    deltas = [Delta(10, 1, "bid", 100.0, 2.0), Delta(10, 2, "ask", 101.0, 3.0),
              Delta(30, 3, "bid", 99.0, 1.0)]
    df = _lake_df([(d.ts_engine, d.seq, d.side == "bid", d.price, d.size) for d in deltas])
    snaps = [book_snapshot(0, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    via_list, ml = reconstruct_seeded(deltas, [5, 15, 35], k=2, snapshots=snaps, policy=NOW)
    via_arr, ma = reconstruct_lake_l2_at_samples_seeded(
        df, [5, 15, 35], k=2, engine_time_col="origin_time", snapshots=snaps, policy=NOW)
    pd.testing.assert_frame_equal(via_list, via_arr, check_dtype=True)
    assert ml["seed_ts"] == ma["seed_ts"] == 0


# --------------------------------------------------------------------------- day-boundary
def test_day_boundary_pre_seed_samples_are_empty_and_excluded():
    # No prior-day carry in the one-day pilot: samples before the first valid seed are an empty
    # book (warm-up), the seed establishes state from its ts forward.
    df = _lake_df([(200, 1, True, 100.0, 1.0)])
    seed = [book_snapshot(100, bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])]
    frame, m = reconstruct_lake_l2_at_samples_seeded(
        df, [50, 150], k=1, engine_time_col="origin_time", snapshots=seed, policy=NOW)
    f = frame.set_index("sample_ts")
    assert math.isnan(f.loc[50, "bid_0_price"])   # pre-seed (ts<100) empty
    assert f.loc[150, "bid_0_price"] == 100.0     # seeded from ts=100
    assert m["seed_ts"] == 100
