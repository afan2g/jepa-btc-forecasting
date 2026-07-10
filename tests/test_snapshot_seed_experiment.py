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


# ---------------------------------------------------------------- seeded Lake replay arm
def synthetic_lake_day():
    """A degraded synthetic Lake `book_delta_v2` day whose repair is hand-checkable.

    The clearing update for the 105.0 bid is MISSING from the stream (the §5a stranded-
    level failure mode): asks later move down through it, so from t=200s the
    reconstructed book is crossed (bid 105 >= ask 103) and STAYS crossed — only a
    full-state snapshot reseed can drop the stranded level.
    """
    rows = [
        # establish a 2-level book
        dict(origin_time=DAY_OPEN + s(10), sequence_number=1, side_is_bid=True,
             price=100.0, size=2.0),
        dict(origin_time=DAY_OPEN + s(10), sequence_number=2, side_is_bid=True,
             price=99.0, size=3.0),
        dict(origin_time=DAY_OPEN + s(10), sequence_number=3, side_is_bid=False,
             price=106.0, size=2.0),
        dict(origin_time=DAY_OPEN + s(10), sequence_number=4, side_is_bid=False,
             price=107.0, size=3.0),
        # bid appears at 105 ... its size=0 clear is LOST upstream
        dict(origin_time=DAY_OPEN + s(100), sequence_number=5, side_is_bid=True,
             price=105.0, size=5.0),
        # asks walk down through the stranded bid -> crossed from here on
        dict(origin_time=DAY_OPEN + s(200), sequence_number=6, side_is_bid=False,
             price=103.0, size=4.0),
        dict(origin_time=DAY_OPEN + s(210), sequence_number=7, side_is_bid=False,
             price=104.0, size=1.0),
        # later ordinary activity (keeps the day alive after the repair point)
        dict(origin_time=DAY_OPEN + s(400), sequence_number=8, side_is_bid=True,
             price=101.0, size=1.5),
    ]
    return pd.DataFrame(rows)


def _true_state_snapshot(ts):
    """The TRUE book at `ts`>=210s (what a trusted vendor snapshot would deliver):
    no stranded 105 bid, asks at 103/104 gone-and-replaced view kept simple."""
    from recon.reseed import book_snapshot
    return book_snapshot(ts, bids=((100.0, 2.0), (99.0, 3.0)),
                         asks=((103.0, 4.0), (104.0, 1.0)))


GRID = [DAY_OPEN + s(i * 30) for i in range(20)]  # 30 s grid over the first 10 min


class TestSeedLakeReplay:
    def _acceptance(self, **kw):
        from experiments.snapshot_seed import SnapshotAcceptance
        defaults = dict(min_levels_per_side=2, max_age_s=600.0, tick_scale=100)
        defaults.update(kw)
        return SnapshotAcceptance(**defaults)

    def _run(self, candidates, **kw):
        from experiments.snapshot_seed import seed_lake_replay
        return seed_lake_replay(synthetic_lake_day(), candidates, grid=GRID, k=2,
                                acceptance=self._acceptance(), **kw)

    def test_rejected_candidates_are_never_injected(self):
        from recon.reseed import book_snapshot
        crossed = book_snapshot(DAY_OPEN + s(300), bids=((105.0, 1.0), (99.0, 1.0)),
                                asks=((103.0, 1.0), (104.0, 1.0)))
        frame, meta = self._run([(crossed, {"origin": "test"})])
        cold, cold_meta = self._run([])
        assert meta["candidates"]["n_accepted"] == 0
        assert meta["candidates"]["rejected"][0]["reason"] == "crossed"
        assert meta["frame_hash"] == cold_meta["frame_hash"]  # byte-identical cold path

    def test_stale_candidate_rejected_by_requested_ts(self):
        snap = _true_state_snapshot(DAY_OPEN + s(300))
        prov = {"at_ts": DAY_OPEN + s(1200)}  # requested 15 min after the state's stamp
        frame, meta = self._run([(snap, prov)], )
        assert meta["candidates"]["n_accepted"] == 0
        assert meta["candidates"]["rejected"][0]["reason"] == "stale"

    def test_pre_seed_samples_identical_to_cold_start(self):
        snap = _true_state_snapshot(DAY_OPEN + s(300))
        seeded, _ = self._run([(snap, {})])
        cold, _ = self._run([])
        pre = [t for t in GRID if t < DAY_OPEN + s(300)]
        pd.testing.assert_frame_equal(
            seeded[seeded["sample_ts"].isin(pre)].reset_index(drop=True),
            cold[cold["sample_ts"].isin(pre)].reset_index(drop=True))

    def test_seed_repairs_the_stranded_crossing(self):
        snap = _true_state_snapshot(DAY_OPEN + s(300))
        seeded, meta = self._run([(snap, {})])
        cold, cold_meta = self._run([])
        # cold: crossed from 200s onward; seeded: uncrossed from 300s onward
        post = seeded[seeded["sample_ts"] >= DAY_OPEN + s(300)]
        assert (post["bid_0_price"] < post["ask_0_price"]).all()
        assert meta["crossed_samples"] < cold_meta["crossed_samples"]
        assert meta["seed_accepted"] is True

    def test_same_ts_delta_applies_before_snapshot(self):
        # A delta at EXACTLY the snapshot ts must be overwritten by the snapshot
        # (authoritative full state) — the production _merge_time_ordered rule.
        snap = _true_state_snapshot(DAY_OPEN + s(400))  # same ts as the seq=8 delta
        seeded, _ = self._run([(snap, {})])
        at = seeded[seeded["sample_ts"] == DAY_OPEN + s(420)].iloc[0]
        # snapshot state has bid_0 100.0; the same-ts delta (bid 101@1.5) was overwritten
        assert at["bid_0_price"] == 100.0

    def test_deterministic_replay_hash(self):
        snap = _true_state_snapshot(DAY_OPEN + s(300))
        _, m1 = self._run([(snap, {})])
        _, m2 = self._run([(snap, {})])
        assert m1["frame_hash"] == m2["frame_hash"]
        assert m1["report_hash"] == m2["report_hash"]

    def test_ledger_carries_provenance_and_acceptance(self):
        snap = _true_state_snapshot(DAY_OPEN + s(300))
        _, meta = self._run([(snap, {"vendor": "coinapi", "method": "l3_replay_as_of"})])
        led = meta["candidates"]
        assert led["n_total"] == 1 and led["n_accepted"] == 1
        acc = led["accepted"][0]
        assert acc["ts"] == DAY_OPEN + s(300)
        assert acc["provenance"]["vendor"] == "coinapi"
        assert meta["acceptance"] == self._acceptance().as_dict()


# ------------------------------------------------------------------- on-demand reseeding
def _true_state_provider(requested_ts):
    """Emulated vendor: returns the TRUE book state as of `requested_ts` (age 0)."""
    snap = _true_state_snapshot(requested_ts)
    return snap, {"vendor": "synthetic", "at_ts": requested_ts}


class TestOnDemandReseedArm:
    def _acceptance(self):
        from experiments.snapshot_seed import SnapshotAcceptance
        return SnapshotAcceptance(min_levels_per_side=2, max_age_s=600.0, tick_scale=100)

    def _run(self, provider, **kw):
        from experiments.snapshot_seed import on_demand_reseed_arm
        defaults = dict(grid=GRID, k=2, acceptance=self._acceptance(),
                        trigger_after_crossed_s=30.0, max_requests=8)
        defaults.update(kw)
        return on_demand_reseed_arm(synthetic_lake_day(), provider, **defaults)

    def test_single_trigger_repairs_the_day_with_one_request(self):
        frame, meta = self._run(_true_state_provider)
        log = meta["on_demand"]["request_log"]
        assert len(log) == 1
        # crossing is first observable at the 210 s sample; the request fires only after
        # the book has been crossed for the full trigger window (no future peeking).
        assert log[0]["requested_ts"] == DAY_OPEN + s(240)
        assert log[0]["injected"] is True
        post = frame[frame["sample_ts"] >= DAY_OPEN + s(240)]
        assert (post["bid_0_price"] < post["ask_0_price"]).all()

    def test_no_requests_when_never_crossed(self):
        clean = synthetic_lake_day().iloc[:4]  # only the clean 2-level establishment
        from experiments.snapshot_seed import on_demand_reseed_arm
        frame, meta = on_demand_reseed_arm(clean, _true_state_provider, grid=GRID, k=2,
                                           acceptance=self._acceptance(),
                                           trigger_after_crossed_s=30.0, max_requests=8)
        assert meta["on_demand"]["request_log"] == []
        assert meta["on_demand"]["terminated"] == "no_trigger"

    def test_max_requests_bounds_the_loop(self):
        frame, meta = self._run(_true_state_provider, max_requests=0)
        assert meta["on_demand"]["request_log"] == []
        assert meta["on_demand"]["terminated"] == "max_requests"

    def test_rejected_snapshot_stops_without_injection(self):
        from recon.reseed import book_snapshot

        def bad_provider(requested_ts):
            snap = book_snapshot(requested_ts, bids=((105.0, 1.0), (99.0, 1.0)),
                                 asks=((103.0, 1.0), (104.0, 1.0)))  # crossed
            return snap, {"vendor": "synthetic", "at_ts": requested_ts}

        frame, meta = self._run(bad_provider)
        log = meta["on_demand"]["request_log"]
        assert len(log) == 1
        assert log[0]["injected"] is False and log[0]["reason"] == "crossed"
        assert meta["on_demand"]["terminated"] == "no_progress"

    def test_deterministic_across_runs(self):
        _, m1 = self._run(_true_state_provider)
        _, m2 = self._run(_true_state_provider)
        assert m1["frame_hash"] == m2["frame_hash"]
        assert m1["on_demand"]["request_log"] == m2["on_demand"]["request_log"]
