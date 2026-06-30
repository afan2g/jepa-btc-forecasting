"""Unit tests for recon/parity.py and the cross-vendor same-book equivalence.

The marquee test (`test_lake_l2_and_coinapi_l3_same_book_match`) builds a Crypto Lake
`book_delta_v2` stream and a CoinAPI `limitbook_full` L3 stream that encode the SAME book,
runs both real reconstructors onto one grid, and asserts the top-K output is byte-identical —
the synthetic stand-in for docs/data.md §5a hard gate #1 (no vendor access needed)."""
import datetime as dt

import numpy as np
import pandas as pd

from recon.events import Delta
from recon.reconstruct import reconstruct_book_at_samples
from recon.coinapi import coinapi_frame_from_rows, reconstruct_coinapi_l2_at_samples
from recon.parity import compare_topk, label_agreement, lake_warmup_cutoff

DAY = dt.date(2025, 6, 1)
DAY_OPEN = pd.Timestamp("2025-06-01").value
S = 1_000_000_000


def frame(sample_ts, bid_p, bid_s, ask_p, ask_s):
    bid_p = np.asarray(bid_p, float)
    ask_p = np.asarray(ask_p, float)
    mid = (bid_p + ask_p) / 2.0
    return pd.DataFrame({
        "sample_ts": sample_ts, "mid": mid, "microprice": mid,
        "bid_0_price": bid_p, "bid_0_size": np.asarray(bid_s, float),
        "ask_0_price": ask_p, "ask_0_size": np.asarray(ask_s, float),
    })


# --------------------------------------------------------------------------- metric tests
def test_identical_frames_report_zero_divergence():
    f = frame([1, 2, 3], [100, 100, 100], [1, 1, 1], [101, 101, 101], [1, 1, 1])
    rep = compare_topk(f, f.copy(), k=1, grid_s=1, horizons_s=(2,))
    assert rep["mid_diff"]["median"] == 0.0 and rep["mid_diff"]["max"] == 0.0
    assert rep["crossed_rate"]["lake"] == 0.0 and rep["crossed_rate"]["capi"] == 0.0
    assert rep["missing_book"]["either_fraction"] == 0.0
    assert rep["label_agreement"]["2"]["agreement"] == 1.0


def test_constant_offset_shows_up_in_mid_and_spike_distribution():
    lake = frame([1, 2, 3], [105, 105, 105], [1, 1, 1], [106, 106, 106], [1, 1, 1])
    capi = frame([1, 2, 3], [100, 100, 100], [1, 1, 1], [101, 101, 101], [1, 1, 1])
    rep = compare_topk(lake, capi, k=1, grid_s=1, horizons_s=(2,))
    assert rep["mid_diff"]["median"] == 5.0 and rep["mid_diff"]["signed_mean"] == 5.0
    assert rep["spike_counts"][">1"] == 3   # all three rows differ by $5 > $1
    assert rep["spike_counts"][">5"] == 0   # but none strictly exceed $5
    assert rep["best_bid_diff"]["max"] == 5.0


def test_missing_book_fraction_is_measured():
    lake = frame([1, 2, 3], [100, np.nan, 100], [1, 1, 1], [101, 101, 101], [1, 1, 1])
    capi = frame([1, 2, 3], [100, 100, 100], [1, 1, 1], [101, 101, 101], [1, 1, 1])
    rep = compare_topk(lake, capi, k=1, grid_s=1, horizons_s=(2,))
    assert rep["missing_book"]["lake_fraction"] == 1 / 3
    assert rep["missing_book"]["both_present"] == 2


def test_label_disagreement_is_detected():
    ts = list(range(6))
    up = [100, 101, 102, 103, 104, 105]      # strictly rising mids
    dn = [105, 104, 103, 102, 101, 100]      # strictly falling mids
    lake = frame(ts, up, [1] * 6, [p + 1 for p in up], [1] * 6)
    capi = frame(ts, dn, [1] * 6, [p + 1 for p in dn], [1] * 6)
    la = label_agreement(lake.set_index("sample_ts")["mid"], capi.set_index("sample_ts")["mid"],
                         grid_s=1, horizons_s=(1,))
    assert la["1"]["agreement"] == 0.0 and la["1"]["disagree"] == la["1"]["n"]


def test_no_trade_band_collapses_small_moves_to_flat_agreement():
    ts = list(range(5))
    # tiny +1 moves each step; a wide band makes every move "flat" on both vendors → agree.
    mids = [100.0, 100.0001, 100.0002, 100.0003, 100.0004]
    f = frame(ts, mids, [1] * 5, [m + 1 for m in mids], [1] * 5)
    la = label_agreement(f.set_index("sample_ts")["mid"], f.set_index("sample_ts")["mid"],
                         grid_s=1, horizons_s=(1,), band_bps=5.0)
    assert la["1"]["both_flat"] == la["1"]["n"] and la["1"]["agreement"] == 1.0


# --------------------------------------------------------------------------- warm-up gate (P2.1)
def test_lake_warmup_cutoff_excludes_cold_start():
    # ts 0,1 = one-sided (no ask → cold-start warm-up); 2..5 = seeded & uncrossed.
    f = pd.DataFrame({
        "sample_ts": [0, 1, 2, 3, 4, 5],
        "mid": [np.nan, np.nan, 100.5, 100.5, 100.5, 100.5],
        "microprice": [np.nan, np.nan, 100.5, 100.5, 100.5, 100.5],
        "bid_0_price": [100, 100, 100, 100, 100, 100], "bid_0_size": [1] * 6,
        "ask_0_price": [np.nan, np.nan, 101, 101, 101, 101],
        "ask_0_size": [np.nan, np.nan, 1, 1, 1, 1],
    })
    assert lake_warmup_cutoff(f, min_consecutive=3, min_levels_per_side=1) == 4  # 3rd good sample
    assert lake_warmup_cutoff(f, min_consecutive=1) == 2                         # first good sample
    f2 = f.copy()
    f2[["ask_0_price", "ask_0_size", "mid"]] = np.nan                            # book never gets an ask
    assert lake_warmup_cutoff(f2) is None


def test_lake_warmup_cutoff_respects_min_depth():
    # both sides have only level 0; requiring 2 levels/side ⇒ never seeded.
    f = frame([0, 1, 2, 3], [100] * 4, [1] * 4, [101] * 4, [1] * 4)
    f["bid_1_price"] = np.nan
    f["ask_1_price"] = np.nan
    assert lake_warmup_cutoff(f, min_consecutive=1, min_levels_per_side=2) is None
    assert lake_warmup_cutoff(f, min_consecutive=1, min_levels_per_side=1) == 0


def test_exclude_ts_masks_labels_on_the_regular_grid_not_compacted():
    # Excluding an INTERIOR sample must not compact the time grid for label horizons (Codex P2):
    # _signed_labels uses positional shift(-step), so a removed interior row would make a 1-step
    # label jump the gap (t1→t3 = 2s, not 1s). The fix masks excluded points to NaN on the regular
    # grid → labels whose origin/target is excluded are UNDEFINED, never horizon-stretched.
    ts = [0, 1, 2, 3, 4]
    bid = [100, 101, 102, 103, 104]
    f = frame(ts, bid, [1] * 5, [p + 1 for p in bid], [1] * 5)
    rep = compare_topk(f, f.copy(), k=1, grid_s=1, horizons_s=(1,), exclude_ts={2})
    la = rep["label_agreement"]["1"]
    # valid 1-step label origins: t0 (target t1) and t3 (target t4). t1 (target t2 excluded) and t2
    # (excluded) are undefined ⇒ n == 2. A compacted grid would wrongly count t1→t3 ⇒ n == 3.
    assert la["n"] == 2
    assert rep["n_grid_full"] == 5 and rep["n_grid"] == 4 and rep["n_excluded_crossed"] == 1


def test_compare_topk_since_ts_restricts_to_warm_window():
    lake = frame([0, 1, 2, 3], [100, 100, 105, 105], [1] * 4, [101, 101, 106, 106], [1] * 4)
    capi = frame([0, 1, 2, 3], [100, 100, 100, 100], [1] * 4, [101, 101, 101, 101], [1] * 4)
    full = compare_topk(lake, capi, k=1, grid_s=1, horizons_s=(1,))
    warm = compare_topk(lake, capi, k=1, grid_s=1, horizons_s=(1,), since_ts=2)
    assert full["n_grid"] == 4 and warm["n_grid"] == 2 and warm["n_grid_full"] == 4
    assert warm["since_ts"] == 2
    # divergence ($5) only at ts>=2: excluding the matching early rows raises the measured median
    assert full["mid_diff"]["median"] == 2.5     # |Δ| over [0,0,5,5]
    assert warm["mid_diff"]["median"] == 5.0      # |Δ| over [5,5]


# --------------------------------------------------------------------------- per-level coverage (P2.2)
def test_per_level_coverage_marks_one_sided_depth():
    cols = {"sample_ts": [0, 1], "mid": [100.5, 100.5], "microprice": [100.5, 100.5],
            "bid_0_price": [100, 100], "bid_0_size": [1, 1], "bid_1_price": [99, 99],
            "bid_1_size": [1, 1], "ask_0_price": [101, 101], "ask_0_size": [1, 1],
            "ask_1_price": [102, 102], "ask_1_size": [1, 1]}
    lake = pd.DataFrame(cols)
    capi = pd.DataFrame(cols)
    capi[["bid_1_price", "bid_1_size"]] = np.nan  # CoinAPI lacks bid level 1 → an uncompared depth
    rep = compare_topk(lake, capi, k=2, grid_s=1, horizons_s=(1,))
    cov1 = rep["per_level"]["1"]["coverage"]["bid"]
    assert cov1["both_present"] == 0 and cov1["only_lake"] == 2 and cov1["only_capi"] == 0
    assert cov1["both_fraction"] == 0.0
    cov0 = rep["per_level"]["0"]["coverage"]["bid"]
    assert cov0["both_present"] == 2 and cov0["both_fraction"] == 1.0


# --------------------------------------------------------------------------- cross-vendor
def test_lake_l2_and_coinapi_l3_same_book_match():
    grid = [DAY_OPEN + i * S for i in range(1, 6)]  # samples at +1s..+5s

    # Crypto Lake book_delta_v2 (price-level, absolute size, 0=remove).
    deltas = [
        Delta(DAY_OPEN + 1 * S, 1, "bid", 100.0, 2.0),
        Delta(DAY_OPEN + 1 * S, 2, "ask", 101.0, 3.0),
        Delta(DAY_OPEN + 2 * S, 3, "bid", 100.0, 5.0),   # absolute resize 2 -> 5
        Delta(DAY_OPEN + 3 * S, 4, "bid", 99.0, 1.0),    # add a second bid level
    ]
    lake = reconstruct_book_at_samples(deltas, grid, k=2)

    # CoinAPI limitbook_full L3 — same book, order-by-order; SNAPSHOT stamped at prior-day close.
    rows = [
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="B", time_exchange_ns=86_399_999_000_000),
        dict(update_type="SNAPSHOT", is_buy=False, entry_px=101.0, entry_sx=3.0,
             order_id="A", time_exchange_ns=86_399_999_000_000),
        dict(update_type="SET", is_buy=True, entry_px=100.0, entry_sx=5.0,
             order_id="B", time_exchange_ns=2 * S),       # B 2 -> 5 (level 100 = 5)
        dict(update_type="ADD", is_buy=True, entry_px=99.0, entry_sx=1.0,
             order_id="B2", time_exchange_ns=3 * S),
    ]
    capi, q = reconstruct_coinapi_l2_at_samples(coinapi_frame_from_rows(rows), k=2,
                                                day=DAY, sample_ts=grid)

    cols = ["mid", "bid_0_price", "bid_0_size", "bid_1_price", "bid_1_size",
            "ask_0_price", "ask_0_size"]
    pd.testing.assert_frame_equal(
        lake.set_index("sample_ts")[cols], capi.set_index("sample_ts")[cols], check_dtype=True
    )
    rep = compare_topk(lake, capi, k=2, grid_s=1, horizons_s=(2,))
    assert rep["mid_diff"]["max"] == 0.0
    assert rep["per_level"]["0"]["bid_size"]["max"] == 0.0
    assert rep["per_level"]["1"]["bid_price"]["max"] == 0.0
    assert q["crossed_rate"] == 0.0 and q["missing_book_fraction"] == 0.0
