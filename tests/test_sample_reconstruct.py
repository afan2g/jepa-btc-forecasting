import math
import numpy as np
import pandas as pd
from recon.events import Delta
from recon.reconstruct import reconstruct_book_at_samples, reconstruct_lake_l2_at_samples
from recon.synthetic import simple_world


def _deltas():
    draw, _ = simple_world()
    return [Delta(**d) for d in draw]


def test_samples_before_first_delta_are_empty_book():
    out = reconstruct_book_at_samples(_deltas(), [5], k=2)
    row = out.iloc[0]
    assert row["sample_ts"] == 5
    assert math.isnan(row["mid"]) and math.isnan(row["bid_0_price"])


def test_samples_track_book_state_on_a_grid():
    # simple_world book: ts10 bid100@2/ask101@3 ; ts30 bid100->0,bid99@1 ; ts50 ask101->0,ask102@4
    out = reconstruct_book_at_samples(_deltas(), [15, 35, 55, 65], k=2).set_index("sample_ts")
    assert out.loc[15, "mid"] == 100.5 and out.loc[15, "bid_0_price"] == 100.0
    assert out.loc[35, "bid_0_price"] == 99.0 and out.loc[35, "mid"] == 100.0
    assert out.loc[55, "ask_0_price"] == 102.0 and out.loc[55, "mid"] == 100.5
    assert out.loc[65, "mid"] == 100.5  # trailing sample keeps final book


def test_as_of_is_inclusive_at_the_exact_sample_time_no_lookahead():
    # A sample AT a delta's ts must see that delta applied (apply-before-read), but never a
    # later one. g=30 sees the bid100->0 / bid99 moves; it must NOT see the ts=50 ask move.
    out = reconstruct_book_at_samples(_deltas(), [10, 30, 50], k=2).set_index("sample_ts")
    assert out.loc[10, "mid"] == 100.5
    assert out.loc[30, "bid_0_price"] == 99.0 and out.loc[30, "ask_0_price"] == 101.0
    assert out.loc[50, "ask_0_price"] == 102.0


def test_schema_is_stable_and_seed_is_not_mutated():
    from recon.orderbook import OrderBook
    seed = OrderBook()
    seed.apply(Delta(1, 1, "bid", 50.0, 9.0))
    out = reconstruct_book_at_samples(_deltas(), [5, 15], k=3, seed=seed)
    # seed had a bid@50; sample at g=5 (before any new delta) reflects ONLY the seed.
    assert out.set_index("sample_ts").loc[5, "bid_0_price"] == 50.0
    # seed object itself is untouched (copied, not mutated).
    assert seed.best_bid() == 50.0 and seed.best_ask() is None
    expected_cols = {"sample_ts", "mid", "microprice"} | {
        f"{s}_{i}_{f}" for s in ("bid", "ask") for i in range(3) for f in ("price", "size")
    }
    assert set(out.columns) == expected_cols
    assert isinstance(out, pd.DataFrame)


def test_array_lake_sampler_matches_delta_list_path():
    # The memory-safe array path (reconstruct_lake_l2_at_samples) must produce byte-identical
    # output to the Delta-list path on a real-Lake-schema DataFrame (origin_time datetime64,
    # sequence_number, side_is_bid bool). Includes an intentionally out-of-(ts,seq)-order row
    # to exercise the lexsort.
    draw, _ = simple_world()
    df = pd.DataFrame(draw).rename(columns={"ts_engine": "origin_time", "seq": "sequence_number"})
    df["origin_time"] = pd.to_datetime(df["origin_time"])          # ns -> datetime64[ns]
    df["side_is_bid"] = df["side"].map({"bid": True, "ask": False})
    df = df.drop(columns=["side"])
    shuffled = df.iloc[[5, 0, 3, 1, 4, 2]].reset_index(drop=True)   # scramble file order

    grid = [5, 10, 25, 35, 55, 65]
    via_array = reconstruct_lake_l2_at_samples(shuffled, grid, k=2, engine_time_col="origin_time")
    via_list = reconstruct_book_at_samples([Delta(**d) for d in draw], grid, k=2)
    pd.testing.assert_frame_equal(via_array, via_list, check_dtype=True)


def test_array_lake_sampler_handles_bool_and_seed():
    from recon.orderbook import OrderBook
    df = pd.DataFrame({
        "origin_time": pd.to_datetime([10, 10]),
        "sequence_number": np.array([1, 2], dtype="int64"),
        "side_is_bid": np.array([True, False]),
        "price": [100.0, 101.0], "size": [2.0, 3.0],
    })
    # bool fast-path of _decode_sides + as-of sampling
    out = reconstruct_lake_l2_at_samples(df, [15], k=1, engine_time_col="origin_time")
    assert out.iloc[0]["mid"] == 100.5

    # seed path (array sampler): a pre-seeded bid is reflected in a sample taken BEFORE any new
    # delta, the new delta supersedes the top, and the seed object itself is left unmutated.
    seed = OrderBook()
    seed.apply(Delta(1, 1, "bid", 50.0, 9.0))
    seeded = reconstruct_lake_l2_at_samples(df, [5, 15], k=1, engine_time_col="origin_time",
                                            seed=seed).set_index("sample_ts")
    assert seeded.loc[5, "bid_0_price"] == 50.0    # seed visible pre-delta
    assert seeded.loc[15, "bid_0_price"] == 100.0  # new delta tops the seed by +15s
    assert seed.best_bid() == 50.0 and seed.best_ask() is None  # seed not mutated
