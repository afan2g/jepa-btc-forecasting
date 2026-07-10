"""Synthetic tests for the #54 CoinAPI snapshot-only seeding EXPERIMENT harness.

Everything here runs on synthetic fixtures — no vendor I/O, no credentials, CI-safe.
The harness under test lives in experiments/snapshot_seed.py and is deliberately
SEPARATE from production policy (recon/, scripts/run_coinbase_quality_map.py): a GO
result requires a separate reviewed implementation/policy PR, so nothing in
experiments/ may be imported by production code.
"""
from __future__ import annotations

import pandas as pd
import pytest

from recon.coinapi import coinapi_frame_from_rows

NS = 1_000_000_000
DAY = "2026-04-01"
DAY_OPEN = int(pd.Timestamp(DAY).value)


def s(sec: float) -> int:
    """ns-since-midnight for a time-of-day in seconds (the downloader schema unit)."""
    return int(sec * NS)


def synthetic_l3_day():
    """A tiny CoinAPI L3 day in downloader schema (seq order == row order).

    Opening SNAPSHOT block: 3 bid orders (two share the 100.0 level) + 2 ask orders.
    Aggregated L2 truth at day open:
        bids: 100.0 -> 3.0 (o1 1.0 + o2 2.0), 99.0 -> 5.0 (o3)
        asks: 101.0 -> 4.0 (o4), 102.0 -> 6.0 (o5)
    Then: t=10s ADD bid o6 100.5 x 1.5; t=20s MATCH (decrement) o1 -0.4 at 100.0;
          t=30s DELETE o4 (ask 101.0 emptied).
    """
    rows = [
        # SNAPSHOT rows carry a bogus prior-day time_exchange stamp on the real feed;
        # the extractor must clamp their label to the day open regardless of this value.
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=100.0, entry_sx=1.0,
             order_id="o1", time_exchange_ns=s(86399.9)),
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=100.0, entry_sx=2.0,
             order_id="o2", time_exchange_ns=s(86399.9)),
        dict(update_type="SNAPSHOT", is_buy=True, entry_px=99.0, entry_sx=5.0,
             order_id="o3", time_exchange_ns=s(86399.9)),
        dict(update_type="SNAPSHOT", is_buy=False, entry_px=101.0, entry_sx=4.0,
             order_id="o4", time_exchange_ns=s(86399.9)),
        dict(update_type="SNAPSHOT", is_buy=False, entry_px=102.0, entry_sx=6.0,
             order_id="o5", time_exchange_ns=s(86399.9)),
        dict(update_type="ADD", is_buy=True, entry_px=100.5, entry_sx=1.5,
             order_id="o6", time_exchange_ns=s(10)),
        dict(update_type="MATCH", is_buy=True, entry_px=100.0, entry_sx=0.4,
             order_id="o1", time_exchange_ns=s(20)),
        dict(update_type="DELETE", is_buy=False, entry_px=101.0, entry_sx=0.0,
             order_id="o4", time_exchange_ns=s(30)),
    ]
    return coinapi_frame_from_rows(rows)


# ------------------------------------------------------------------ snapshot extraction
class TestCoinapiSnapshotAt:
    def test_day_open_snapshot_is_the_snapshot_block_state(self):
        from experiments.snapshot_seed import coinapi_snapshot_at
        snap, prov = coinapi_snapshot_at(synthetic_l3_day(), day=DAY, at_ts=DAY_OPEN,
                                         size_policy="decrement")
        assert snap.ts == DAY_OPEN
        assert snap.bids == ((100.0, 3.0), (99.0, 5.0))
        assert snap.asks == ((101.0, 4.0), (102.0, 6.0))

    def test_as_of_excludes_later_events(self):
        from experiments.snapshot_seed import coinapi_snapshot_at
        snap, _ = coinapi_snapshot_at(synthetic_l3_day(), day=DAY,
                                      at_ts=DAY_OPEN + s(15), size_policy="decrement")
        # t=10s ADD included, t=20s MATCH excluded
        assert snap.bids == ((100.5, 1.5), (100.0, 3.0), (99.0, 5.0))
        assert snap.asks == ((101.0, 4.0), (102.0, 6.0))

    def test_as_of_is_inclusive_at_equal_timestamp(self):
        from experiments.snapshot_seed import coinapi_snapshot_at
        # Same convention as sample_topk_as_of: "as of g" reflects every event <= g.
        snap, _ = coinapi_snapshot_at(synthetic_l3_day(), day=DAY,
                                      at_ts=DAY_OPEN + s(20), size_policy="decrement")
        assert snap.bids == ((100.5, 1.5), (100.0, 2.6), (99.0, 5.0))

    def test_max_levels_truncates_to_best_prices(self):
        from experiments.snapshot_seed import coinapi_snapshot_at
        snap, prov = coinapi_snapshot_at(synthetic_l3_day(), day=DAY,
                                         at_ts=DAY_OPEN + s(15), max_levels=1,
                                         size_policy="decrement")
        assert snap.bids == ((100.5, 1.5),)
        assert snap.asks == ((101.0, 4.0),)
        assert prov["levels_available"] == {"bids": 3, "asks": 2}
        assert prov["levels_used"] == {"bids": 1, "asks": 1}

    def test_provenance_records_source_and_extraction(self):
        from experiments.snapshot_seed import coinapi_snapshot_at
        snap, prov = coinapi_snapshot_at(synthetic_l3_day(), day=DAY, at_ts=DAY_OPEN,
                                         size_policy="decrement")
        assert prov["vendor"] == "coinapi"
        assert prov["method"] == "l3_replay_as_of"
        assert prov["at_ts"] == DAY_OPEN
        assert prov["day"] == DAY
        assert prov["size_policy"] == "decrement"
        assert prov["events_applied"] == 5  # the SNAPSHOT block only
        assert prov["last_event_label_ts"] == DAY_OPEN


# ------------------------------------------------------------------ candidate acceptance
class TestClassifyCandidate:
    def _policy(self, **kw):
        from experiments.snapshot_seed import SnapshotAcceptance
        defaults = dict(min_levels_per_side=2, max_age_s=60.0, tick_scale=100)
        defaults.update(kw)
        return SnapshotAcceptance(**defaults)

    def _snap(self, ts=DAY_OPEN, bids=((100.0, 1.0), (99.0, 2.0)),
              asks=((101.0, 1.0), (102.0, 2.0))):
        from recon.reseed import book_snapshot
        return book_snapshot(ts, bids, asks)

    def test_good_candidate_is_ok(self):
        from experiments.snapshot_seed import classify_candidate
        assert classify_candidate(self._snap(), requested_ts=DAY_OPEN,
                                  policy=self._policy()) == "ok"

    def test_future_snapshot_rejected_for_causality(self):
        from experiments.snapshot_seed import classify_candidate
        snap = self._snap(ts=DAY_OPEN + s(1))
        assert classify_candidate(snap, requested_ts=DAY_OPEN,
                                  policy=self._policy()) == "future"

    def test_stale_snapshot_rejected(self):
        from experiments.snapshot_seed import classify_candidate
        snap = self._snap(ts=DAY_OPEN)
        assert classify_candidate(snap, requested_ts=DAY_OPEN + s(61),
                                  policy=self._policy(max_age_s=60.0)) == "stale"
        assert classify_candidate(snap, requested_ts=DAY_OPEN + s(60),
                                  policy=self._policy(max_age_s=60.0)) == "ok"

    def test_truncated_below_min_depth_rejected(self):
        from experiments.snapshot_seed import classify_candidate
        snap = self._snap(bids=((100.0, 1.0),))
        assert classify_candidate(snap, requested_ts=DAY_OPEN,
                                  policy=self._policy(min_levels_per_side=2)) == "thin_depth"

    def test_crossed_snapshot_rejected(self):
        from experiments.snapshot_seed import classify_candidate
        snap = self._snap(bids=((101.5, 1.0), (99.0, 2.0)))
        assert classify_candidate(snap, requested_ts=DAY_OPEN,
                                  policy=self._policy()) == "crossed"

    def test_malformed_values_rejected(self):
        from experiments.snapshot_seed import classify_candidate
        snap = self._snap(asks=((101.0, float("nan")), (102.0, 2.0)))
        assert classify_candidate(snap, requested_ts=DAY_OPEN,
                                  policy=self._policy()) == "bad_values"

    def test_off_tick_price_rejected(self):
        from experiments.snapshot_seed import classify_candidate
        # COINBASE BTC-USD tick is $0.01 (native tick scale 100); a price that is not an
        # exact tick multiple signals unit/venue drift in the snapshot source.
        snap = self._snap(bids=((100.001, 1.0), (99.0, 2.0)))
        assert classify_candidate(snap, requested_ts=DAY_OPEN,
                                  policy=self._policy(tick_scale=100)) == "off_tick"

    def test_tick_check_skipped_when_scale_unknown(self):
        from experiments.snapshot_seed import classify_candidate
        snap = self._snap(bids=((100.001, 1.0), (99.0, 2.0)))
        assert classify_candidate(snap, requested_ts=DAY_OPEN,
                                  policy=self._policy(tick_scale=None)) == "ok"

    def test_causality_precedes_structural_reasons(self):
        from experiments.snapshot_seed import classify_candidate
        # A future AND crossed snapshot must surface the causality violation first.
        snap = self._snap(ts=DAY_OPEN + s(1), bids=((101.5, 1.0), (99.0, 2.0)))
        assert classify_candidate(snap, requested_ts=DAY_OPEN,
                                  policy=self._policy()) == "future"


# ------------------------------------------------------------- snapshot stream emulation
def _topk_frame():
    """A 4-sample top-2 frame like the reconstructors emit (NaN = padding)."""
    nan = float("nan")
    rows = [
        # t0: full two levels both sides
        dict(sample_ts=DAY_OPEN + s(0), mid=100.5, microprice=100.5,
             bid_0_price=100.0, bid_0_size=1.0, bid_1_price=99.0, bid_1_size=2.0,
             ask_0_price=101.0, ask_0_size=1.5, ask_1_price=102.0, ask_1_size=2.5),
        # t1: identical to t0 (an unchanged second)
        dict(sample_ts=DAY_OPEN + s(1), mid=100.5, microprice=100.5,
             bid_0_price=100.0, bid_0_size=1.0, bid_1_price=99.0, bid_1_size=2.0,
             ask_0_price=101.0, ask_0_size=1.5, ask_1_price=102.0, ask_1_size=2.5),
        # t2: ask_0 size changes
        dict(sample_ts=DAY_OPEN + s(2), mid=100.5, microprice=100.4,
             bid_0_price=100.0, bid_0_size=1.0, bid_1_price=99.0, bid_1_size=2.0,
             ask_0_price=101.0, ask_0_size=0.5, ask_1_price=102.0, ask_1_size=2.5),
        # t3: bid side thins to one level (NaN pad on bid_1)
        dict(sample_ts=DAY_OPEN + s(3), mid=100.5, microprice=100.4,
             bid_0_price=100.0, bid_0_size=1.0, bid_1_price=nan, bid_1_size=nan,
             ask_0_price=101.0, ask_0_size=0.5, ask_1_price=102.0, ask_1_size=2.5),
    ]
    return pd.DataFrame(rows)


class TestSnapshotsFromTopkFrame:
    def test_emits_one_candidate_per_sample_with_nan_pads_dropped(self):
        from experiments.snapshot_seed import snapshots_from_topk_frame
        snaps, stats = snapshots_from_topk_frame(_topk_frame(), max_levels=2)
        assert [sn.ts for sn in snaps] == [DAY_OPEN + s(i) for i in range(4)]
        assert snaps[0].bids == ((100.0, 1.0), (99.0, 2.0))
        assert snaps[3].bids == ((100.0, 1.0),)  # NaN pad dropped, not poisoned
        assert snaps[2].asks == ((101.0, 0.5), (102.0, 2.5))

    def test_max_levels_truncates_below_frame_depth(self):
        from experiments.snapshot_seed import snapshots_from_topk_frame
        snaps, _ = snapshots_from_topk_frame(_topk_frame(), max_levels=1)
        assert snaps[0].bids == ((100.0, 1.0),)
        assert snaps[0].asks == ((101.0, 1.5),)

    def test_changed_second_stats_for_size_projection(self):
        from experiments.snapshot_seed import snapshots_from_topk_frame
        # The real limitbook_snapshot_X product records a row only when the top-X book
        # changed within the interval; unchanged seconds are what make it small. t1 is
        # unchanged at both depths; t2/t3 changed.
        _, stats = snapshots_from_topk_frame(_topk_frame(), max_levels=2)
        assert stats["n_samples"] == 4
        assert stats["n_changed"] == 3  # t0 (first), t2, t3
        _, stats1 = snapshots_from_topk_frame(_topk_frame(), max_levels=1)
        assert stats1["n_changed"] == 2  # at depth 1 only t0 and t2 differ

    def test_stride_thins_the_stream(self):
        from experiments.snapshot_seed import snapshots_from_topk_frame
        snaps, _ = snapshots_from_topk_frame(_topk_frame(), max_levels=2,
                                             stride_ns=2 * NS)
        assert [sn.ts for sn in snaps] == [DAY_OPEN + s(0), DAY_OPEN + s(2)]
